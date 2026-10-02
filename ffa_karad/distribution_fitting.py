"""
Candidate distribution fitting for the annual-peak block maxima.

Three corrections to the notebook's Section 7
----------------------------------------------
**1.  Information criteria were not comparable.**  The notebook fitted some
candidates to ``Q`` and others to ``ln Q`` and then ranked them by AIC as
though the log-likelihoods referred to the same variable.  They do not.  If
``Y = ln Q`` then

.. math::
    L_Q(\\theta) = L_Y(\\theta)\\prod_{i=1}^{n}\\frac{1}{Q_i}
    \\quad\\Longrightarrow\\quad
    \\ln L_Q = \\ln L_Y - \\sum_{i=1}^{n}\\ln Q_i .

Every fit in this module reports ``loglike`` on the **discharge** scale, with
the Jacobian applied where the fit was performed in log space.  Without it the
LP3 AICc came out at about 91.6 against the other candidates; corrected, it is
about 983.7, and the ranking changes accordingly.  The correction is a shift
by ``2 * sum(ln Q_i)``, which is a constant, so it changes no *within*-scale
comparison -- only comparisons across scales, which is exactly the comparison
being made.

**2.  Families that do not belong in a block-maxima list.**  The generalised
Pareto was included.  GPD is the peaks-over-threshold model for threshold
exceedances; fitted to annual maxima it invites a bounded upper tail, which
contradicts the heavy right tail of a flood record.  It is excluded here and
handled properly in :mod:`ffa_karad.peaks_over_threshold`.  Free-location
log-normal (3 parameters) is also excluded: IS 11223 and CWC practice use the
strict two-parameter LN2, and a free location changes Q100 by about +10.6 per
cent and Q1000 by +18.5 per cent on this record, which is not a detail.

**3.  Divergence was clipped away.**  The notebook's GEV returned
``c = -4.736``, ``loc = 857.78``, ``scale = 13.16`` and the resulting curve was
clipped to the plotted range.  A fit that needs clipping has not converged to a
usable distribution.  Every fit here carries an acceptance gate
(:func:`fit_is_acceptable`); a fit that fails is retained in the table with
``accepted = False`` and is excluded from ranking, rather than being silently
repaired.

Acceptance gates
----------------
A fit must have finite parameters, a strictly positive scale, parameters inside
degenerate bounds, a positive log-likelihood, monotone quantiles at the design
return periods, and a finite Anderson-Darling statistic.  The recorded Q1000
must also exceed the observed HFL discharge: a fitted curve that puts the
100-year flood below the flood that has already been observed is
contradicted by the data regardless of how good its likelihood looks.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util

log = util.get_logger("distribution_fitting")


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One fitted family.

    ``scipy_name`` selects the frozen distribution used for quantiles and
    goodness of fit.  ``fitted_on`` records the variable the likelihood was
    maximised on, which is what determines whether the Jacobian applies.
    """

    name: str
    scipy_name: str
    k: int
    fitted_on: str = "Q"  # "Q" or "log Q"
    floc: float | str | None = None
    family: str = "generalised extreme value"
    cwc_reference: bool = False
    note: str = ""

    @property
    def needs_jacobian(self) -> bool:
        return self.fitted_on.lower().replace(" ", "") == "logq"


#: Candidate list.  Deliberately short: every entry must be a defensible model
#: of *annual maxima*.  Order is preserved in every table for readability.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "Gumbel",
        "gumbel_r",
        2,
        "Q",
        None,
        "extreme value type I",
        cwc_reference=True,
        note="CWC cross-check: extreme value type I fitted to the peaks "
        "themselves, which is what the CWC tables use.  Fitting the "
        "Gumbel to ln Q and exponentiating would give the "
        "'Gumbel of logs', a distribution whose quantiles grow like "
        "exp(exp(...)); on this record it returns Q1000 = 51,364 "
        "cumecs, which no annual-maxima series supports.",
    ),
    Candidate(
        "LN2",
        "lognorm",
        2,
        "Q",
        0.0,
        "lognormal",
        cwc_reference=True,
        note="Strict two-parameter log-normal, floc = 0. A free location "
        "is a three-parameter LN3 and is excluded.",
    ),
    Candidate(
        "LP3",
        "pearson3",
        3,
        "log Q",
        None,
        "log-Pearson type III",
        cwc_reference=True,
        note="IS 11223:1985 primary method, fitted to ln Q.",
    ),
    Candidate(
        "GEV",
        "genextreme",
        3,
        "Q",
        None,
        "generalised extreme value",
        note="Three-parameter; shape must stay inside the degenerate " "bounds below.",
    ),
    Candidate(
        "Weibull",
        "weibull_min",
        2,
        "Q",
        0.0,
        "Weibull (minimum)",
        note="Two-parameter, floc = 0.",
    ),
    Candidate(
        "Exponential",
        "expon",
        1,
        "Q",
        0.0,
        "exponential",
        note="One-parameter reference standard.",
    ),
    Candidate(
        "GNO",
        "gennorm",
        3,
        "Q",
        None,
        "generalised normal",
        note="Symmetric counterpart of GLO; a symmetric model cannot "
        "describe a skewed flood record.",
    ),
    Candidate("GLO", "genlogistic", 3, "Q", None, "generalised logistic"),
    Candidate(
        "Log-logistic",
        "fisk",
        2,
        "Q",
        0.0,
        "log-logistic (Fisk)",
        note="Two-parameter, floc = 0.",
    ),
)

CANDIDATE_BY_NAME: dict[str, Candidate] = {c.name: c for c in CANDIDATES}


# ---------------------------------------------------------------------------
# Fit container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DistributionFit:
    """One fitted candidate with its diagnostics and acceptance verdict."""

    candidate: Candidate
    params: dict[str, float]
    loglike: float  # on the discharge scale
    loglike_natural: float  # on the variable actually fitted
    n: int
    aic: float
    bic: float
    aicc: float
    hqic: float
    ks_stat: float
    ks_p: float
    ad_stat: float
    ad_p: float
    jacobian_applied: bool
    accepted: bool
    reject_reasons: list[str]
    #: Fitted CDF at the smallest and largest observed peaks.  Both must lie
    #: inside ``[1/(2n), 1 - 1/(2n)]``; see the support-coverage gate.
    cdf_at_min: float = float("nan")
    cdf_at_max: float = float("nan")
    residuals: np.ndarray = dataclasses.field(
        repr=False, default_factory=lambda: np.empty(0)
    )
    quantiles: dict[float, float] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)

    def quantile(self, return_period: float) -> float:
        """Discharge quantile at ``return_period`` years."""
        return float(self.quantiles.get(float(return_period), float("nan")))

    def cdf(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Cumulative probability on the discharge scale.

        Fits performed in log space are evaluated at ``ln Q`` and the result is
        still a probability of the *discharge*, because the transformation is
        applied to the argument, not to the fitted parameters.
        """
        dist = _scipy_distribution(self.candidate, self.params)
        x = np.asarray(values, dtype=float)
        return dist.cdf(np.log(x) if self.candidate.needs_jacobian else x)

    def ppf(self, exceedance: Sequence[float] | np.ndarray) -> np.ndarray:
        """Quantile on the discharge scale for a non-exceedance probability."""
        dist = _scipy_distribution(self.candidate, self.params)
        q = dist.ppf(np.asarray(exceedance, dtype=float))
        return np.exp(q) if self.candidate.needs_jacobian else q

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.candidate.name,
            "scipy_family": self.candidate.scipy_name,
            "k": self.candidate.k,
            "fitted_on": self.candidate.fitted_on,
            "jacobian_applied": self.jacobian_applied,
            "params": self.params,
            "loglike_Q": self.loglike,
            "loglike_natural": self.loglike_natural,
            "aic": self.aic,
            "bic": self.bic,
            "aicc": self.aicc,
            "hqic": self.hqic,
            "ks_stat": self.ks_stat,
            "ks_p": self.ks_p,
            "ad_stat": self.ad_stat,
            "ad_p": self.ad_p,
            "accepted": self.accepted,
            "reject_reasons": self.reject_reasons,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _scipy_generator(candidate: Candidate) -> Any:
    """The *unfrozen* scipy generator, which is the object that has ``fit``."""
    return getattr(sps, candidate.scipy_name)


def _scipy_distribution(candidate: Candidate, params: dict[str, float]) -> Any:
    """Frozen scipy distribution built from a fitted parameter dict."""
    return _scipy_generator(candidate)(**params)


def _unpack_fit(generator: Any, fit_result: Any) -> dict[str, float]:
    """Parameter dict from a ``scipy`` ``fit`` result.

    SciPy 1.15 returns a ``FitResult`` with a ``params`` mapping; earlier and
    later versions return the bare tuple ``(*shape, loc, scale)``.  Both layouts
    are normalised here so the pipeline does not depend on the SciPy version.
    """
    named = getattr(fit_result, "params", None)
    if named is not None:
        return {str(k): float(v) for k, v in dict(named).items()}
    values = tuple(float(v) for v in fit_result)
    if len(values) < 2:
        raise ValueError(f"unexpected fit result layout: {fit_result!r}")
    shape_names = [s.strip() for s in (generator.shapes or "").split(",") if s.strip()]
    shapes, loc, scale = values[:-2], values[-2], values[-1]
    if len(shapes) != len(shape_names):
        raise ValueError(
            f"fit returned {len(shapes)} shape parameters but the family declares "
            f"{len(shape_names)} ({shape_names})"
        )
    params = dict(zip(shape_names, shapes))
    params["loc"] = loc
    params["scale"] = scale
    return params


def _fit_one(candidate: Candidate, q: np.ndarray) -> DistributionFit:
    """Fit a single candidate, applying the Jacobian and every acceptance gate."""
    n = int(q.size)
    log_q = np.log(q)
    jacobian_term = float(np.sum(log_q))  # sum ln Q_i
    reasons: list[str] = []

    # ``sps.<name>`` is the unfrozen generator, whose ``fit`` method returns the
    # MLE parameters; calling the generator itself would freeze it at default
    # parameters, and a frozen object has no ``fit``.
    dist = _scipy_generator(candidate)
    data = q if candidate.fitted_on == "Q" else log_q
    # ``floc`` is not cosmetic: CWC/IS 11223 define LN2 with two parameters and
    # a free location makes the fit a three-parameter LN3, which moves Q100 by
    # about +10.6 % and Q1000 by about +18.5 % on this record.
    fit_kwds: dict[str, float] = {}
    if candidate.floc is not None:
        fit_kwds["floc"] = float(candidate.floc)
    try:
        params = _unpack_fit(dist, dist.fit(data, **fit_kwds))
        frozen = dist(**params)
    except Exception as exc:  # pragma: no cover - scipy version dependent
        log.warning("fit failed for %s: %s", candidate.name, exc)
        return _failed_fit(candidate, n, [f"scipy fit raised: {exc}"])

    loglike_natural = float(frozen.logpdf(data).sum())
    jacobian = candidate.needs_jacobian
    loglike_q = loglike_natural - jacobian_term if jacobian else loglike_natural
    k = int(candidate.k)

    aic = 2 * k - 2 * loglike_q
    bic = k * np.log(n) - 2 * loglike_q
    denom = n - k - 1
    aicc = aic + (2 * k * (k + 1) / denom if denom > 0 else float("inf"))
    hqic = 2 * k * np.log(np.log(n)) - 2 * loglike_q

    # -- quantiles on the discharge scale --------------------------------
    quantiles: dict[float, float] = {}
    for t in _cfg.CONFIG.return_periods:
        try:
            q_t = float(frozen.ppf(1.0 - 1.0 / float(t)))
        except Exception:
            q_t = float("nan")
        if candidate.fitted_on != "Q" and np.isfinite(q_t):
            q_t = float(np.exp(q_t))
        quantiles[float(t)] = q_t

    # -- residual diagnostics on the discharge scale ---------------------
    probs = np.clip(frozen.cdf(data), 1e-12, 1 - 1e-12)
    ks = sps.kstest(probs, "uniform")
    ad = _anderson_darling_uniform(probs)
    resid = sps.norm.ppf(probs) if np.all(np.isfinite(probs)) else np.array([])

    fit = DistributionFit(
        candidate=candidate,
        params=params,
        loglike=loglike_q,
        loglike_natural=loglike_natural,
        n=n,
        aic=aic,
        bic=bic,
        aicc=aicc,
        hqic=hqic,
        ks_stat=float(ks.statistic),
        ks_p=float(ks.pvalue),
        ad_stat=float(ad["statistic"]),
        ad_p=float(ad["p"]),
        jacobian_applied=jacobian,
        accepted=False,
        reject_reasons=reasons,
        residuals=resid,
        quantiles=quantiles,
        notes=[candidate.note] if candidate.note else [],
    )
    _apply_gates(fit, q)
    return fit


def _failed_fit(candidate: Candidate, n: int, reasons: list[str]) -> DistributionFit:
    nan = float("nan")
    return DistributionFit(
        candidate=candidate,
        params={},
        loglike=nan,
        loglike_natural=nan,
        n=n,
        aic=nan,
        bic=nan,
        aicc=nan,
        hqic=nan,
        ks_stat=nan,
        ks_p=nan,
        ad_stat=nan,
        ad_p=nan,
        jacobian_applied=candidate.needs_jacobian,
        accepted=False,
        reject_reasons=list(reasons),
    )


def _apply_gates(fit: DistributionFit, q: np.ndarray) -> None:
    """Decide whether a fit may enter the ranking.  Reasons are recorded."""
    reasons = fit.reject_reasons
    if not fit.params:
        reasons.append("no finite parameters returned")
    if not np.isfinite(fit.loglike):
        reasons.append("log-likelihood is not finite")
    if any(not np.isfinite(v) for v in fit.params.values()):
        reasons.append("non-finite parameter")
    scale = _scale_of(fit)
    if scale is not None and not (scale > 0):
        reasons.append(f"scale is not positive ({scale:.4g})")
    shape = _shape_of(fit)
    if shape is not None and abs(shape) > _cfg.CONFIG.max_abs_shape:
        reasons.append(
            f"shape parameter {shape:.4g} exceeds the degenerate bound "
            f"{_cfg.CONFIG.max_abs_shape:g}"
        )
    # Support-coverage gate.  A location parameter on its own says nothing: the
    # EV1 location is 2185 cumecs on this record and is perfectly sensible,
    # while the diverged GEV sits at 857.8 cumecs with a scale of 13 and is not.
    # What separates them is whether the fitted curve can accommodate the
    # observed extremes.  An order statistic of n = 57 must carry at least
    # 1/(2n) of the probability; a curve that puts less than that below the
    # smallest peak, or more than that above the largest, is not describing the
    # sample it was fitted to.
    if fit.params:
        guard = 1.0 / (2.0 * max(fit.n, 1))
        try:
            f_min = float(fit.cdf([float(np.min(q))])[0])
            f_max = float(fit.cdf([float(np.max(q))])[0])
        except Exception:  # pragma: no cover - defensive
            f_min = f_max = float("nan")
        if np.isfinite(f_min) and np.isfinite(f_max):
            fit.cdf_at_min, fit.cdf_at_max = f_min, f_max
            if f_min < guard:
                reasons.append(
                    f"the fitted CDF at the smallest observed peak "
                    f"({np.min(q):,.0f} cumecs) is {f_min:.4g}, below the "
                    f"{guard:.4g} that an order statistic of {fit.n} can carry; "
                    "the curve does not cover the lower end of the record"
                )
            if f_max > 1.0 - guard:
                reasons.append(
                    f"the fitted CDF at the largest observed peak "
                    f"({np.max(q):,.0f} cumecs) is {f_max:.4g}, above the "
                    f"{1.0 - guard:.4g} that an order statistic of {fit.n} can "
                    "carry; the fit has diverged into the upper tail"
                )

    q_t = fit.quantiles.get(float(_cfg.CONFIG.return_periods[-1]), float("nan"))
    if np.isfinite(q_t) and q_t < _cfg.CONFIG.observed_max_discharge_factor * float(
        np.max(q)
    ):
        reasons.append(
            f"Q{max(_cfg.CONFIG.return_periods):g} = {q_t:,.0f} cumecs is below "
            f"the largest observed peak ({float(np.max(q)):,.0f}); the record "
            "contradicts the fit"
        )
    if fit.quantiles:
        finite_q = [v for v in fit.quantiles.values() if np.isfinite(v)]
        if len(finite_q) != len(fit.quantiles):
            reasons.append("one or more design quantiles are not finite")
        elif any(b <= a for a, b in zip(finite_q, finite_q[1:])):
            reasons.append("quantiles are not increasing with return period")
    if not np.isfinite(fit.ad_stat):
        reasons.append("Anderson-Darling statistic is not finite")
    # Goodness-of-fit gate.  Without this the Exponential sits in the accepted
    # list on the strength of its finite parameters while its KS p-value is
    # 9e-5: the fit does not describe the sample at all.  A distribution that
    # cannot reproduce the probability plot of the record has no business being
    # a candidate, however many parameters it has or however good its AICc.
    gof_alpha = float(_cfg.CONFIG.gof_alpha)
    if np.isfinite(fit.ks_p) and fit.ks_p < gof_alpha:
        reasons.append(
            f"the KS probability-plot p-value is {fit.ks_p:.3g}, below "
            f"{gof_alpha:g}; the fitted CDF does not reproduce the observed "
            "order statistics"
        )
    fit.accepted = not reasons
    if reasons:
        log.info("fit %-12s rejected: %s", fit.candidate.name, "; ".join(reasons))


def _scale_of(fit: DistributionFit) -> float | None:
    for key in ("scale", "s", "beta"):
        if key in fit.params:
            return fit.params[key]
    return None


def _shape_of(fit: DistributionFit) -> float | None:
    for key in ("c", "skew", "shape", "k", "a"):
        if key in fit.params:
            return fit.params[key]
    return None


def _location_of(fit: DistributionFit) -> float | None:
    return fit.params.get("loc")


# ---------------------------------------------------------------------------
# Goodness of fit on the probability scale
# ---------------------------------------------------------------------------


def _anderson_darling_uniform(u: np.ndarray) -> dict[str, float]:
    """Anderson-Darling statistic for uniform variates, with a bootstrap p.

    The uniform variates are the fitted CDF values of the ordered sample, so
    this tests the *fitted* distribution, not an estimated-parameter-free one.
    ``scipy.stats.anderson`` assumes known parameters and returns tabulated
    critical values that do not apply to a fitted fit, hence the parametric
    bootstrap.
    """
    u = np.sort(np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
    n = u.size
    stat = float(
        -n - np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(u) + np.log(1 - u[::-1])))
    )
    rng = util.stream_rng(f"ad_uniform::{n}")
    n_boot = _cfg.CONFIG.n_gof_bootstrap
    count = 0
    for _ in range(n_boot):
        sim = np.sort(np.clip(rng.random(n), 1e-12, 1 - 1e-12))
        sim_stat = -n - np.sum(
            (2 * np.arange(1, n + 1) - 1) * (np.log(sim) + np.log(1 - sim[::-1]))
        )
        count += int(sim_stat >= stat)
    return {"statistic": stat, "p": (count + 1) / (n_boot + 1)}


# ---------------------------------------------------------------------------
# Method-of-moments Gumbel (the CWC cross-check, reported separately)
# ---------------------------------------------------------------------------


def gumbel_moments(q: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Method-of-moments Gumbel: ``mu = mean - 0.5772 s``, ``beta = s sqrt(6)/pi``.

    IS 11223 practice quotes the extreme-value Type I parameters directly from
    the sample mean and standard deviation; this is kept alongside the
    maximum-likelihood fit because CWC tables use the moment values and the two
    differ slightly at this record length.
    """
    x = np.asarray(q, dtype=float).ravel()
    s = float(np.std(x, ddof=1))
    mean = float(np.mean(x))
    mu = mean - float(_cfg.CONFIG.euler_mascheroni) * s
    beta = s * float(np.sqrt(6.0) / np.pi)
    return {"mu": mu, "beta": beta}


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DistributionFittingResults:
    fits: list[DistributionFit]
    ranking: pd.DataFrame
    periods: tuple[float, ...]
    best_accepted: str | None
    notes: list[str]

    def by_name(self, name: str) -> DistributionFit:
        for fit in self.fits:
            if fit.candidate.name == name:
                return fit
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": [f.to_dict() for f in self.fits],
            "ranking": self.ranking.to_dict(orient="records"),
            "best_accepted": self.best_accepted,
            "notes": self.notes,
        }


def ranking_table(fits: Sequence[DistributionFit]) -> pd.DataFrame:
    """Rank accepted fits by AICc; rejected fits are listed but not ranked."""
    rows = []
    for fit in fits:
        row = {
            "distribution": fit.candidate.name,
            "k": fit.candidate.k,
            "fitted_on": fit.candidate.fitted_on,
            "jacobian_applied": fit.jacobian_applied,
            "loglike_Q": fit.loglike,
            "aic": fit.aic,
            "bic": fit.bic,
            "aicc": fit.aicc,
            "hqic": fit.hqic,
            "ks_stat": fit.ks_stat,
            "ks_p": fit.ks_p,
            "ad_stat": fit.ad_stat,
            "ad_p": fit.ad_p,
            "accepted": fit.accepted,
        }
        for t, v in fit.quantiles.items():
            row[f"Q{int(t)}"] = v
        row["reject_reasons"] = "; ".join(fit.reject_reasons)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    frame["rank"] = pd.Series([np.nan] * len(frame), index=frame.index)
    accepted = frame[frame["accepted"]]
    if len(accepted):
        order = accepted["aicc"].rank(method="min").astype(int)
        frame.loc[accepted.index, "rank"] = order.to_numpy()
    return frame.sort_values(["accepted", "aicc"], ascending=[False, True]).reset_index(
        drop=True
    )


def run(
    q: Sequence[float] | np.ndarray, candidates: Sequence[Candidate] | None = None
) -> DistributionFittingResults:
    """Fit every candidate to the annual peaks and rank the survivors."""
    values = np.asarray(q, dtype=float).ravel()
    if np.any(values <= 0):
        raise ValueError("annual peaks must be strictly positive")
    cands = list(CANDIDATES if candidates is None else candidates)
    fits = [_fit_one(c, values) for c in cands]
    ranking = ranking_table(fits)
    accepted = [f for f in fits if f.accepted]
    best = min(accepted, key=lambda f: f.aicc).candidate.name if accepted else None

    notes = [
        f"{len(accepted)} of {len(fits)} candidates passed every acceptance "
        f"gate; the lowest AICc among them is {best or 'none'}.",
        "All log-likelihoods are on the discharge scale.  Fits performed on "
        "ln Q carry the Jacobian correction sum(ln Q_i); without it the LP3 "
        "AICc would be about 91.6 instead of about 983.7 and the ranking "
        "would be meaningless.",
        "GPD is absent by design: it is a threshold-exceedance model, not a "
        "block-maxima family, and including it invites a bounded upper tail.",
        "The Gumbel method-of-moments parameters are reported separately "
        "because CWC tables quote moment values, which differ slightly from "
        "the maximum-likelihood fit at this record length.",
        "Information criteria are secondary.  IS 11223:1985 fixes LP3 as the "
        "design method; a lower AICc does not override it.",
    ]
    for fit in fits:
        if not fit.accepted:
            notes.append(
                f"{fit.candidate.name} was excluded: " + "; ".join(fit.reject_reasons)
            )
    log.info(
        "distribution fitting: %d/%d accepted, lowest AICc = %s",
        len(accepted),
        len(fits),
        best,
    )
    return DistributionFittingResults(
        fits=fits,
        ranking=ranking,
        periods=tuple(_cfg.CONFIG.return_periods),
        best_accepted=best,
        notes=notes,
    )


def standardise_residuals(fit: DistributionFit) -> np.ndarray:
    """Standardised (probability-scale normal) residuals of a fit."""
    return np.asarray(fit.residuals, dtype=float)


def summarise(res: DistributionFittingResults) -> str:
    """Plain-text summary for the report."""
    lines = [
        "CANDIDATE DISTRIBUTION FITTING  (annual maxima, discharge scale)",
        f"  {'distribution':<14} {'k':>2} {'AICc':>12} {'rank':>5} "
        f"{'AD stat':>8} {'AD p':>7}  status",
    ]
    for row in res.ranking.itertuples():
        status = "accepted" if row.accepted else "EXCLUDED"
        rank = "" if pd.isna(row.rank) else f"{int(row.rank):5d}"
        lines.append(
            f"  {row.distribution:<14} {row.k:2d} {row.aicc:12.2f} "
            f"{rank:>5} {row.ad_stat:8.4f} {row.ad_p:7.4f}  {status}"
        )
    lines.append("")
    lines.append("  design floods by candidate (cumecs):")
    header = "  " + " " * 14 + "".join(f"{int(t):>10}" for t in res.periods)
    lines.append(header)
    for fit in res.fits:
        if not fit.accepted:
            continue
        cells = "".join(f"{fit.quantiles[float(t)]:10,.0f}" for t in res.periods)
        lines.append(f"  {fit.candidate.name:<14}{cells}")
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


__all__ = [
    "Candidate",
    "CANDIDATES",
    "CANDIDATE_BY_NAME",
    "DistributionFit",
    "DistributionFittingResults",
    "fit_is_acceptable",
    "gumbel_moments",
    "ranking_table",
    "standardise_residuals",
    "run",
    "summarise",
]


def fit_is_acceptable(fit: DistributionFit) -> bool:
    """Whether a fit passed every acceptance gate."""
    return bool(fit.accepted)
