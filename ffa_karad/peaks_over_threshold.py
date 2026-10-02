"""
Peaks-over-threshold analysis of the peak series.

What this module does and does not claim
----------------------------------------
POT is a **threshold-exceedance** method.  The block-maxima analysis in
:mod:`ffa_karad.distribution_fitting` takes one value per water year; POT takes
every peak above a threshold and models the exceedances with a generalised
Pareto distribution whose shape ``xi`` is the tail index.  The two answer
different questions and must not be merged -- which is exactly what the 2025
notebook did by fitting the GPD to annual maxima and then quoting its quantiles
as if they were block-maxima design floods.

The 2025 POT attempt also failed on its own terms: 14 exceedances, a profile
likelihood whose shape interval was dominated by ``xi < 0`` (a bounded tail,
which contradicts the Gumbel and LP3 curves), and no threshold-stability check.
This module adds:

* the **Jenkinson-Collison** threshold, which minimises the mean squared error of
  the exponential quantile function fitted to the ordered exceedances;
* **profile-penalty** likelihood for the threshold, with the penalty constant
  ``k`` exposed in :data:`ffa_karad.config.CONFIG.pot_penalty_k`;
* a **threshold-stability** scan across candidate thresholds;
* a **confidence interval on ``xi``** from the profile likelihood rather than a
  delta-method standard error;
* an **acceptance gate**: the number of exceedances must reach
  :data:`ffa_karad.config.CONFIG.pot_min_exceedances`, the shape interval must
  not be pinned at the optimiser's bound, and the return levels must be stable
  across candidate thresholds.  A failed gate suppresses the design values
  rather than publishing them.

Conventions
-----------
``zeta`` (the exceedance rate) is the mean number of exceedances **per year**,
``zeta = n_exceed / record_years``.  The design level at return period ``T``
years is

``z(T) = u + (beta/xi) * [(T*zeta)**xi - 1]``

which reduces to ``u - beta*ln(T*zeta)`` in the exponential limit ``xi -> 0``.
Both forms are asserted against each other in the test suite, because the 2025
notebook's ``1 - 1/T`` form of this expression is wrong by orders of magnitude
at long return periods.

Data requirement
----------------
POT needs several events per year.  The station sheet holds one annual maximum
per water year, so the exceedances here are annual exceedances of an
annual-maxima series -- a defensible but weak POT design.  The module therefore
reports the exceedance rate per year and labels the result exploratory.  If
``data/runoff data.xlsx`` (or any daily series) is supplied,
:func:`load_daily_peaks` uses it and the label is removed.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import optimize as sopt

from . import config as _cfg
from . import util

log = util.get_logger("peaks_over_threshold")

#: Fewest exceedances that can support a profile-likelihood GPD fit.  Below this
#: the shape estimate is a function of two or three order statistics.
MIN_FITTABLE_EXCEEDANCES = 10

#: Shape search bounds.  Beyond these the GPD is no longer a credible tail
#: model for an annual flood record and the fit is reported as at-the-bound.
XI_BOUNDS: tuple[float, float] = tuple(_cfg.CONFIG.pot_xi_bounds)


# ---------------------------------------------------------------------------
# Generalised Pareto internals
# ---------------------------------------------------------------------------


def _gpd_neg_loglike(xi: float, beta: float, exceed: np.ndarray) -> float:
    """Negative log-likelihood of the GPD with shape ``xi`` and scale ``beta``."""
    if not np.isfinite(beta) or beta <= 0:
        return math.inf
    y = np.asarray(exceed, dtype=float)
    xi = float(xi)
    if abs(xi) < 1e-8:
        # Exponential limit; the log form avoids cancellation as xi -> 0.
        return float(np.sum(np.log(beta) + y / beta))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        arg = 1.0 + xi * y / beta
    if not np.all(np.isfinite(arg)) or np.any(arg <= 0):
        return math.inf
    return float(np.sum(np.log(beta) + (1.0 + 1.0 / xi) * np.log(arg)))


def _profile_scale(
    xi: float, exceed: np.ndarray, iterations: int = 500, tol: float = 1e-12
) -> float:
    """Profile out the GPD scale for a fixed shape.

    With ``NLL = sum[ln(beta) + (1 + 1/xi) ln(1 + xi*y/beta)]``, the condition
    ``d(NLL)/d(beta) = 0`` rearranges to the implicit equation

    ``beta = (1 + xi) * mean( y / (1 + xi*y/beta) )``,

    solved here by fixed-point iteration starting from the sample mean.  The
    test suite checks this against the joint two-parameter maximum-likelihood
    fit; it is not the same as the probability-weighted-moment estimator
    ``mean(y*(1+xi*y/beta)**(-1/xi))``, which maximises a *weighted* objective
    and returns a different, biased shape.  For ``xi -> 0`` the solution is the
    mean exceedance, as it must be.
    """
    y = np.asarray(exceed, dtype=float)
    if y.size == 0:
        return float("nan")
    if abs(xi) < 1e-8:
        return float(y.mean())
    beta = float(y.mean())
    for _ in range(int(iterations)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            arg = 1.0 + xi * y / beta
            if not np.all(np.isfinite(arg)) or np.any(arg <= 0):
                return float(y.mean())
            updated = float((1.0 + xi) * np.mean(y / arg))
        if not np.isfinite(updated) or updated <= 0:
            return float(y.mean())
        converged = abs(updated - beta) <= tol * max(abs(beta), 1.0)
        beta = updated
        if converged:
            break
    return beta


def fit_gpd(
    exceed: np.ndarray, xi_bounds: tuple[float, float] = XI_BOUNDS
) -> dict[str, float]:
    """Profile-likelihood GPD fit of threshold exceedances.

    The shape comes from a one-dimensional minimisation of the profile
    log-likelihood (the scale has the closed form above), which is faster and
    more stable at these sample sizes than fitting both parameters at once.
    """
    y = np.asarray(exceed, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 5:
        raise ValueError("at least 5 exceedances are required for a GPD fit")
    if np.any(y < 0):
        raise ValueError("GPD exceedances must be non-negative")

    def _profile_nll(xi: float) -> float:
        return _gpd_neg_loglike(float(xi), _profile_scale(float(xi), y), y)

    result = sopt.minimize_scalar(
        _profile_nll, bounds=xi_bounds, method="bounded", options={"xatol": 1e-7}
    )
    xi_hat = float(result.x)
    span = abs(xi_bounds[1] - xi_bounds[0])
    at_bound = min(abs(xi_hat - xi_bounds[0]), abs(xi_hat - xi_bounds[1])) < 1e-3 * span
    return {
        "xi": xi_hat,
        "beta": _profile_scale(xi_hat, y),
        "threshold_scale_mean": float(y.mean()),
        "n_exceed": int(y.size),
        "neg_loglike": float(result.fun),
        "converged": bool(result.success),
        "at_shape_bound": bool(at_bound),
    }


def gpd_quantile(
    threshold: float,
    xi: float,
    beta: float,
    return_period: float,
    exceedance_rate: float,
) -> float:
    """POT design level at ``return_period`` years.

    ``z(T) = u + (beta/xi)[(T*zeta)**xi - 1]`` with the exponential limit
    ``z(T) = u - beta*ln(T*zeta)``.  Both branches are implemented because the
    limit is used whenever ``|xi|`` is small, and the two must agree there.
    """
    tz = float(return_period) * float(exceedance_rate)
    if tz <= 0:
        return float("nan")
    if abs(float(xi)) < 1e-8:
        return float(threshold - float(beta) * math.log(tz))
    return float(threshold + (float(beta) / float(xi)) * (tz ** float(xi) - 1.0))


def gpd_return_levels(
    threshold: float,
    xi: float,
    beta: float,
    periods: Sequence[float],
    exceedance_rate: float,
) -> pd.DataFrame:
    """Design level curve on the annual scale."""
    rows = []
    for t in periods:
        rows.append(
            {
                "return_period_yr": float(t),
                "return_level_cumecs": gpd_quantile(
                    threshold, xi, beta, float(t), exceedance_rate
                ),
                "exceedance_probability": 1.0 - 1.0 / float(t),
            }
        )
    return pd.DataFrame(rows)


def shape_profile_ci(
    exceed: np.ndarray,
    level: float = 0.95,
    xi_bounds: tuple[float, float] = XI_BOUNDS,
    n_grid: int = 401,
) -> pd.DataFrame:
    """Likelihood-ratio confidence interval for the GPD shape.

    The interval is the set of ``xi`` whose profile log-likelihood is within
    ``chi2(1, level)/2`` of the maximum.  A delta-method standard error is not
    used: below 100 exceedances the likelihood is visibly asymmetric in ``xi``,
    and it is asymmetric in exactly the direction that matters -- the negative
    side is much longer than the positive side.
    """
    y = np.asarray(exceed, dtype=float).ravel()
    grid = np.linspace(xi_bounds[0], xi_bounds[1], int(n_grid))
    values = np.array(
        [_gpd_neg_loglike(float(xi), _profile_scale(float(xi), y), y) for xi in grid]
    )
    best = float(np.nanmin(values))
    if not np.isfinite(best):
        return pd.DataFrame(
            {
                "xi": grid,
                "profile_nll": values,
                "delta_nll": values - best,
                "in_interval": values <= np.inf,
            }
        )
    threshold = best + 0.5 * float(_chi2_quantile(level))
    return pd.DataFrame(
        {
            "xi": grid,
            "profile_nll": values,
            "delta_nll": values - best,
            "in_interval": values <= threshold,
        }
    )


def _chi2_quantile(level: float) -> float:
    from scipy import stats as _sps

    return float(_sps.chi2.ppf(level, df=1))


def shape_interval(profile: pd.DataFrame) -> tuple[float, float]:
    """Lower and upper ``xi`` from a :func:`shape_profile_ci` table."""
    inside = profile[profile["in_interval"]] if len(profile) else profile
    if not len(inside):
        return float("nan"), float("nan")
    return float(inside["xi"].min()), float(inside["xi"].max())


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def jenkinson_collison_threshold(sorted_exceedances: np.ndarray) -> dict[str, Any]:
    """Jenkinson-Collison threshold.

    Coles (2001), *An Introduction to the Statistical Modeling of Extreme
    Values*, section 3.1.3: for each trial order statistic ``k`` an exponential
    is fitted to the exceedances above it and the sum of squared errors against
    the theoretical exponential quantiles is computed; the ``k`` with the
    smallest error is chosen.  The search starts above 5 per cent of the sample
    so that at least a handful of points remain.
    """
    z = np.sort(np.asarray(sorted_exceedances, dtype=float).ravel())
    m = z.size
    if m < 10:
        raise ValueError("Jenkinson-Collison search needs at least 10 points")
    best_k: int | None = None
    best_sse = math.inf
    rows = []
    for k in range(max(2, int(0.05 * m)), max(3, int(0.95 * m)) + 1):
        above = z[k - 1 :]
        if above.size < 3:
            continue
        rate = float(above.mean())
        if rate <= 0:
            continue
        theoretical = (
            -np.log1p(-np.arange(1, above.size + 1) / (above.size + 1.0)) * rate
        )
        sse = float(np.sum((above - theoretical) ** 2))
        rows.append(
            {
                "k": int(k),
                "threshold": float(z[k - 1]),
                "mse": sse,
                "n_above": int(above.size),
            }
        )
        if sse < best_sse:
            best_sse, best_k = sse, int(k)
    if best_k is None:
        raise ValueError("Jenkinson-Collison search found no usable threshold")
    return {
        "threshold": float(z[best_k - 1]),
        "k": best_k,
        "mse": best_sse,
        "n_exceed": int(m - best_k + 1),
        "search_table": pd.DataFrame(rows),
    }


def profile_penalty_threshold(
    base: np.ndarray,
    candidate_quantiles: Sequence[float] | None = None,
    penalty_k: float | None = None,
) -> pd.DataFrame:
    """Threshold chosen by minimising the profile-penalised likelihood.

    For a GPD threshold the criterion of Coles (2001, eq. 3.11) is
    ``mean(u) + k * se(mean(u))``.  The shape is re-estimated at every candidate
    threshold so that a threshold which merely produces a tight-looking fit at
    low sample size cannot win.
    """
    x = util.as_float_array(base)
    quantiles = tuple(
        _cfg.CONFIG.pot_threshold_quantiles
        if candidate_quantiles is None
        else candidate_quantiles
    )
    k = float(_cfg.CONFIG.pot_penalty_k if penalty_k is None else penalty_k)

    rows = []
    for prob in quantiles:
        threshold = float(np.quantile(x, float(prob)))
        exceed = x[x > threshold]
        if exceed.size < MIN_FITTABLE_EXCEEDANCES:
            rows.append(
                {
                    "threshold_quantile": float(prob),
                    "threshold": threshold,
                    "n_exceed": int(exceed.size),
                    "xi": float("nan"),
                    "beta": float("nan"),
                    "mean_exceedance": float("nan"),
                    "profile_penalty": float("inf"),
                    "neg_loglike": float("nan"),
                    "usable": False,
                }
            )
            continue
        fit = fit_gpd(exceed)
        mean_u = float(np.mean(exceed))
        rows.append(
            {
                "threshold_quantile": float(prob),
                "threshold": threshold,
                "n_exceed": int(exceed.size),
                "xi": fit["xi"],
                "beta": fit["beta"],
                "mean_exceedance": mean_u,
                "profile_penalty": mean_u + k * mean_u / math.sqrt(exceed.size),
                "neg_loglike": fit["neg_loglike"],
                "usable": True,
            }
        )
    table = pd.DataFrame(rows)
    if not table["usable"].any():
        raise ValueError("no candidate threshold produced a usable GPD fit")
    return table.sort_values("profile_penalty").reset_index(drop=True)


def threshold_stability(
    base: np.ndarray,
    periods: Sequence[float],
    record_years: float,
    candidate_quantiles: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return levels across candidate thresholds -- the stability check.

    A POT analysis is only credible if the estimated return levels barely move
    when the threshold is moved within its defensible range.
    """
    x = util.as_float_array(base)
    quantiles = tuple(
        _cfg.CONFIG.pot_threshold_quantiles
        if candidate_quantiles is None
        else candidate_quantiles
    )
    rows = []
    for prob in quantiles:
        threshold = float(np.quantile(x, float(prob)))
        exceed = x[x > threshold]
        if exceed.size < MIN_FITTABLE_EXCEEDANCES:
            continue
        try:
            fit = fit_gpd(exceed)
        except ValueError:
            continue
        zeta = float(exceed.size) / float(record_years)
        curve = gpd_return_levels(threshold, fit["xi"], fit["beta"], periods, zeta)
        for row in curve.itertuples():
            rows.append(
                {
                    "threshold_quantile": float(prob),
                    "threshold": threshold,
                    "n_exceed": int(exceed.size),
                    "xi": fit["xi"],
                    "exceedance_rate_per_year": zeta,
                    "return_period_yr": row.return_period_yr,
                    "return_level_cumecs": row.return_level_cumecs,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional sub-annual peak source
# ---------------------------------------------------------------------------


def load_daily_peaks(
    path: pathlib.Path | None = None, sheet: int | str = 0
) -> pd.DataFrame | None:
    """Load a sub-annual peak series from a workbook, if one is supplied.

    Returns ``None`` when the file is absent or has no recognisable discharge
    column.  The annual-maxima analysis does not need this function; it exists so
    that a proper POT run becomes possible the moment a daily record is
    available, instead of pretending 57 annual maxima constitute a POT sample.
    """
    path = pathlib.Path(path) if path else (_cfg.DATA_DIR / "runoff data.xlsx")
    if not path.exists():
        log.info("no sub-annual peak file at %s; POT will use the annual maxima", path)
        return None
    try:
        raw = pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:
        log.warning(
            "could not read %s (%s); continuing with annual maxima", path.name, exc
        )
        return None

    lowered = {str(c).strip().lower(): c for c in raw.columns}
    for key in (
        "discharge* (cumecs)",
        "discharge (cumecs)",
        "discharge",
        "cumecs",
        "flow",
        "q",
        "peak discharge",
    ):
        if key in lowered:
            column = lowered[key]
            series = pd.to_numeric(raw[column], errors="coerce").dropna()
            if series.empty:
                continue
            return pd.DataFrame(
                {
                    "peak_cumecs": series.to_numpy(dtype=float),
                    "source": f"{path.name}!{column}",
                }
            )
    log.warning(
        "no discharge-like column in %s; continuing with annual maxima", path.name
    )
    return None


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class POTResult:
    threshold: float
    threshold_source: str
    fit: dict[str, float]
    shape_ci: tuple[float, float]
    return_levels: pd.DataFrame
    threshold_scan: pd.DataFrame
    stability: pd.DataFrame
    exceedance_rate_per_year: float
    record_years: float
    accepted: bool
    reject_reasons: list[str]
    base_kind: str
    notes: list[str]

    def return_level(self, return_period: float) -> float:
        row = self.return_levels[
            np.isclose(self.return_levels["return_period_yr"], return_period)
        ]
        if not len(row):
            raise KeyError(return_period)
        return float(row["return_level_cumecs"].iloc[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "base_series": self.base_kind,
            "record_years": self.record_years,
            "exceedance_rate_per_year": self.exceedance_rate_per_year,
            "fit": self.fit,
            "shape_ci": {"lower": self.shape_ci[0], "upper": self.shape_ci[1]},
            "return_levels": self.return_levels.to_dict(orient="records"),
            "threshold_scan": self.threshold_scan.to_dict(orient="records"),
            "accepted": self.accepted,
            "reject_reasons": self.reject_reasons,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def _choose_threshold(base: np.ndarray, method: str) -> tuple[float, str, pd.DataFrame]:
    if method == "jenkinson_collison":
        jc = jenkinson_collison_threshold(base)
        return float(jc["threshold"]), "Jenkinson-Collison", jc["search_table"]
    if method == "profile_penalty":
        scan = profile_penalty_threshold(base)
        best = scan[scan["usable"]].iloc[0]
        return float(best["threshold"]), "profile-penalty likelihood", scan
    if method == "quantile":
        prob = float(np.median(_cfg.CONFIG.pot_threshold_quantiles))
        return (
            float(np.quantile(base, prob)),
            f"median candidate quantile ({prob:g})",
            pd.DataFrame(),
        )
    raise KeyError(method)


def run(
    base: Sequence[float] | np.ndarray,
    periods: Sequence[float] | None = None,
    method: str = "jenkinson_collison",
    record_years: float | None = None,
    sub_annual: pd.DataFrame | None = None,
) -> POTResult:
    """Fit a POT model and apply the acceptance gate.

    Parameters
    ----------
    base:
        Base series the threshold is applied to.  With the annual maxima as the
        base the result is explicitly labelled exploratory.
    method:
        ``"jenkinson_collison"`` (default), ``"profile_penalty"`` or
        ``"quantile"``.
    record_years:
        Number of years the base series spans; sets the exceedance rate
        ``zeta = n_exceed / record_years``.  Defaults to the base length, which
        is correct only when the base holds one value per year.
    sub_annual:
        Optional frame from :func:`load_daily_peaks`; when supplied it becomes
        the base series.
    """
    periods = tuple(
        float(t) for t in (_cfg.CONFIG.return_periods if periods is None else periods)
    )
    reasons: list[str] = []
    notes: list[str] = []

    if sub_annual is not None and len(sub_annual):
        base_series = util.as_float_array(sub_annual["peak_cumecs"])
        base_kind = f"sub-annual peaks ({sub_annual['source'].iloc[0]})"
        years = float(record_years) if record_years else 57.0
    else:
        base_series = util.as_float_array(base)
        base_kind = "annual maxima (annual-exceedance POT)"
        years = float(record_years) if record_years else float(base_series.size)
        notes.append(
            "The base series holds one value per water year, so the exceedances "
            f"are annual exceedances of an annual-maxima series: {base_series.size} "
            "points over the same number of years.  This is a weak POT design and "
            "the shape estimate is wide by construction."
        )

    threshold, source, scan = _choose_threshold(base_series, method)
    exceed = base_series[base_series > threshold]

    # Jenkinson-Collison minimises the exponential-plot error and, applied to a
    # heavy-tailed base series such as annual maxima, often returns a threshold
    # so high that nothing is left to fit.  When that happens the profile penalty
    # is used instead and the substitution is recorded, never hidden.
    if exceed.size < MIN_FITTABLE_EXCEEDANCES and method == "jenkinson_collison":
        original = threshold
        threshold, source, scan = _choose_threshold(base_series, "profile_penalty")
        exceed = base_series[base_series > threshold]
        notes.append(
            f"The Jenkinson-Collison threshold {original:,.0f} cumecs left only "
            f"{int((base_series > original).sum())} exceedances, which cannot "
            f"support a GPD fit; the profile-penalty threshold "
            f"{threshold:,.0f} cumecs was used instead."
        )
    if exceed.size < 5:
        raise ValueError(
            f"threshold {threshold:.4g} leaves only {exceed.size} exceedances; a "
            "GPD cannot be fitted. Widen the base series or lower the candidate "
            "threshold quantiles."
        )

    fit = fit_gpd(exceed)
    profile = shape_profile_ci(exceed, _cfg.CONFIG.ci_level)
    xi_lo, xi_hi = shape_interval(profile)
    fit["xi_ci_lower"] = xi_lo
    fit["xi_ci_upper"] = xi_hi
    zeta = float(exceed.size) / years

    levels = gpd_return_levels(threshold, fit["xi"], fit["beta"], periods, zeta)
    stability = threshold_stability(base_series, periods, years)

    # -- acceptance gate ---------------------------------------------------
    if exceed.size < _cfg.CONFIG.pot_min_exceedances:
        reasons.append(
            f"only {exceed.size} exceedances over {years:g} years (minimum "
            f"required {_cfg.CONFIG.pot_min_exceedances}); the shape estimate is "
            "not defensible"
        )
    if fit["at_shape_bound"]:
        reasons.append(
            f"the shape estimate xi = {fit['xi']:+.3f} sits on the optimiser "
            f"bound [{XI_BOUNDS[0]:g}, {XI_BOUNDS[1]:g}]; the likelihood is still "
            "increasing there, so no shape has been found"
        )
    if _cfg.CONFIG.pot_require_unbounded_tail and np.isfinite(xi_hi) and xi_hi < 0:
        reasons.append(
            f"the shape interval [{xi_lo:+.3f}, {xi_hi:+.3f}] lies entirely below "
            "zero, i.e. the data cannot exclude a bounded upper tail; this "
            "contradicts the Gumbel and LP3 block-maxima curves"
        )
    values = levels["return_level_cumecs"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        reasons.append("one or more return levels are not finite")
    elif any(b <= a for a, b in zip(values, values[1:])):
        reasons.append("return levels are not increasing with return period")
    if len(stability) and np.all(np.isfinite(values)):
        longest = float(max(periods))
        subset = stability[np.isclose(stability["return_period_yr"], longest)]
        pivot = float(values[-1])
        if len(subset) and pivot > 0:
            spread = float(
                subset["return_level_cumecs"].max()
                - subset["return_level_cumecs"].min()
            )
            if spread / pivot > 0.5:
                reasons.append(
                    f"the {longest:g}-year return level moves by "
                    f"{100.0 * spread / pivot:.0f} % across the candidate "
                    "thresholds; the estimate is threshold-dependent"
                )

    accepted = not reasons
    log.info(
        "POT: %d exceedances, xi = %+.3f [%+.3f, %+.3f], gate %s",
        exceed.size,
        fit["xi"],
        xi_lo,
        xi_hi,
        "PASS" if accepted else "FAIL",
    )

    notes += [
        f"Threshold {threshold:,.0f} cumecs selected by {source} on a "
        f"{base_kind} base series; {exceed.size} exceedances over {years:g} years "
        f"(zeta = {zeta:.4f} per year).",
        f"GPD shape xi = {fit['xi']:+.4f} with {_cfg.CONFIG.ci_level:.0%} profile "
        f"interval [{xi_lo:+.3f}, {xi_hi:+.3f}] and scale beta = "
        f"{fit['beta']:,.0f} cumecs.",
        "The shape interval comes from the profile likelihood, not a delta-method "
        "standard error: the likelihood is visibly asymmetric in xi at these "
        "sample sizes.",
        "Design levels use z(T) = u + (beta/xi)[(T*zeta)**xi - 1], the exact POT "
        "relation; the 1 - 1/T form used in the 2025 notebook is a different "
        "quantity and understates long-period levels by orders of magnitude.",
        "POT and block maxima are separate analyses.  The block-maxima design "
        "floods in the design table come from LP3 under IS 11223; these levels are "
        "an independent check and are reportable only when the gate passes.",
    ]
    if not accepted:
        notes.append(
            "ACCEPTANCE GATE FAILED -- the levels below must not be "
            "adopted as design values."
        )

    return POTResult(
        threshold=threshold,
        threshold_source=source,
        fit=fit,
        shape_ci=(xi_lo, xi_hi),
        return_levels=levels,
        threshold_scan=scan,
        stability=stability,
        exceedance_rate_per_year=zeta,
        record_years=years,
        accepted=accepted,
        reject_reasons=reasons,
        base_kind=base_kind,
        notes=notes,
    )


def summarise(res: POTResult) -> str:
    """Plain-text POT summary for the report."""
    lines = [
        "PEAKS OVER THRESHOLD  (GPD on threshold exceedances)",
        f"  base series            : {res.base_kind} over {res.record_years:g} years",
        f"  threshold              : {res.threshold:,.0f} cumecs ({res.threshold_source})",
        f"  exceedance rate zeta   : {res.exceedance_rate_per_year:.4f} per year "
        f"({int(res.fit['n_exceed'])} exceedances)",
        f"  GPD shape xi           : {res.fit['xi']:+.4f}  "
        f"{_cfg.CONFIG.ci_level:.0%} profile CI [{res.shape_ci[0]:+.3f}, "
        f"{res.shape_ci[1]:+.3f}]",
        f"  GPD scale beta         : {res.fit['beta']:,.0f} cumecs",
        f"  acceptance gate        : {'PASS' if res.accepted else 'FAIL'}",
        "",
        f"  {'T (yr)':>8} {'design level (cumecs)':>24}",
    ]
    for row in res.return_levels.itertuples():
        lines.append(f"  {row.return_period_yr:8.0f} {row.return_level_cumecs:24,.0f}")
    if res.reject_reasons:
        lines += ["", "  NOT ADOPTABLE BECAUSE:"]
        lines += [f"    - {r}" for r in res.reject_reasons]
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


__all__ = [
    "MIN_FITTABLE_EXCEEDANCES",
    "XI_BOUNDS",
    "fit_gpd",
    "gpd_quantile",
    "gpd_return_levels",
    "shape_profile_ci",
    "shape_interval",
    "jenkinson_collison_threshold",
    "profile_penalty_threshold",
    "threshold_stability",
    "load_daily_peaks",
    "POTResult",
    "run",
    "summarise",
]
