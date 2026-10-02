"""
Statistical tests: moments, normality, randomness, stationarity and trend.

Corrections to the 2025 notebook that live here
------------------------------------------------
1.  **Kurtosis.** Bulletin 17B defines the Pearson moment coefficient of
    kurtosis as ``Ck = m4 / m2**2`` with ``mk = (1/n) * sum((x - xbar)**k)``.
    The notebook printed ``Ck = 0.6253`` in section 5.2, which is not any
    standard coefficient; the correct value for this record is 3.4683, and the
    excess kurtosis ``Ck - 3`` is 0.4683 (the number the notebook also
    printed, under the right name but the wrong label).
2.  **The Pearson III kurtosis locus** is ``Ck = 3 + 1.5 * Cs**2``, not
    ``3 + 0.5 * Cs**2``.  :mod:`ffa_karad.lmoments` verifies the correct form
    numerically against a family of fits.
3.  **The "Mode" statistic was removed.**  A 3-year running mean of 57
    right-skewed values is unimodal for essentially any such record, so the
    mode carries no information, and the notebook's value of 1175.33 cumecs
    was pandas ``idxmax`` tie-breaking noise.  Nothing is reported here.
4.  **Anderson–Darling.** ``scipy.stats.anderson`` returns significance levels
    from tables that assume the parameters are *known*.  With maximum-likelihood
    parameters estimated from the same 57 points those levels are invalid, so
    :func:`normality_battery` derives the p-value by parametric bootstrap.

Every test result is returned as a tidy row so the report can be generated
without a single hard-coded number.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util

log = util.get_logger("statistical_tests")


# ---------------------------------------------------------------------------
# Moment coefficients -- Bulletin 17B definitions
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MomentCoefficients:
    """Bulletin 17B moment coefficients and their standard errors."""

    n: int
    mean: float
    sd: float
    cv: float
    m2: float
    m3: float
    m4: float
    #: Skewness coefficient ``g1`` computed with the ``n`` divisor.
    skewness_unadjusted: float
    #: Adjusted skewness ``g1_hat`` of Bulletin 17B, the value used in the
    #: station-skew diagram: ``sqrt(n(n-1))/(n-2) * g1``.
    skewness_adjusted_b17b: float
    #: Pearson moment coefficient of kurtosis ``Ck = m4 / m2**2``.  NOT excess.
    kurtosis_moment_b17b: float
    #: Excess kurtosis ``Ck - 3``.
    excess_kurtosis: float
    skewness_se: float
    kurtosis_se: float

    def as_row(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_cumecs": self.mean,
            "sd_cumecs": self.sd,
            "cv": self.cv,
            "cs_unadjusted": self.skewness_unadjusted,
            "cs_adjusted_b17b": self.skewness_adjusted_b17b,
            "ck_moment_b17b": self.kurtosis_moment_b17b,
            "excess_kurtosis": self.excess_kurtosis,
            "cs_standard_error": self.skewness_se,
            "ck_standard_error": self.kurtosis_se,
        }


def moment_coefficients(values: Sequence[float] | np.ndarray) -> MomentCoefficients:
    """Bulletin 17B skewness and kurtosis with standard errors.

    Definitions used, with ``n`` the sample size, ``xbar`` the mean and
    ``mk = (1/n) sum (x_i - xbar)**k``:

    * ``Cs_unadjusted = m3 / m2**1.5``
    * ``Cs_adjusted   = sqrt(n(n-1)) / (n-2) * Cs_unadjusted``
    * ``Ck            = m4 / m2**2``
    * ``excess        = Ck - 3``

    Standard errors are the asymptotic sampling errors
    ``se(Cs) = sqrt(6/n)`` and ``se(Ck) = sqrt(24/n)``, which Bulletin 17B
    quotes and which are adequate for ``n >= 30``.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n < 8:
        raise ValueError("at least 8 observations are required for moment analysis")
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    m3 = float(np.mean(dev**3))
    m4 = float(np.mean(dev**4))
    if m2 <= 0:
        raise ValueError("degenerate sample: zero variance")

    cs_unadj = m3 / (m2**1.5)
    cs_adj = float(np.sqrt(n * (n - 1.0)) / (n - 2.0) * cs_unadj)
    ck = m4 / (m2**2)
    sd = float(np.sqrt(m2 * n / (n - 1.0)))

    return MomentCoefficients(
        n=n,
        mean=float(x.mean()),
        sd=sd,
        cv=sd / float(x.mean()),
        m2=m2,
        m3=m3,
        m4=m4,
        skewness_unadjusted=float(cs_unadj),
        skewness_adjusted_b17b=cs_adj,
        kurtosis_moment_b17b=float(ck),
        excess_kurtosis=float(ck - 3.0),
        skewness_se=float(np.sqrt(6.0 / n)),
        kurtosis_se=float(np.sqrt(24.0 / n)),
    )


def secondary_skewness_estimators(values: Sequence[float] | np.ndarray) -> pd.DataFrame:
    """Robust and classical alternative skewness measures.

    Reported together because they disagree: the moment skewness is dominated
    by the single 7,177 cumecs event while Bowley's measure, which uses
    quartiles, is not.  A large spread between them is evidence *for* the
    Bulletin 17C confidence-limit treatment, not against the data.

    The Anscombe-Glynn entry is **not** a skewness estimate.  It is the skewness
    of the sample's own normal scores, which is zero by construction; it is
    included only as a symmetry check and is excluded from the spread diagnostic.
    """
    x = np.sort(np.asarray(values, dtype=float).ravel())
    n = x.size
    mean = float(x.mean())
    sd = sd_of(x)
    q25, q50, q75 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    bowley = (q75 + q25 - 2 * q50) / (q75 - q25)
    # Pearson's second coefficient is 3 (mean - median) / s.
    pearson2 = 3.0 * (mean - q50) / sd
    ranks = sps.rankdata(x)
    normal_scores = sps.norm.ppf((ranks - 0.375) / (n + 0.25))
    ag = float(sps.skew(normal_scores, bias=True))
    return pd.DataFrame(
        [
            {
                "estimator": "Moment g1 (n divisor)",
                "value": float(sps.skew(x, bias=True)),
                "family": "moment",
                "robust": False,
                "use_in_spread": True,
            },
            {
                "estimator": "Adjusted g1 (Bulletin 17B)",
                "value": float(sps.skew(x, bias=False)),
                "family": "moment",
                "robust": False,
                "use_in_spread": True,
            },
            {
                "estimator": "Bowley (quartile)",
                "value": bowley,
                "family": "quartile",
                "robust": True,
                "use_in_spread": True,
            },
            {
                "estimator": "Pearson 2nd (3(mean-median)/s)",
                "value": pearson2,
                "family": "quartile",
                "robust": True,
                "use_in_spread": True,
            },
            {
                "estimator": "Anscombe-Glynn (symmetry check)",
                "value": ag,
                "family": "rank/normal-score",
                "robust": True,
                "use_in_spread": False,
            },
        ]
    )


def sd_of(values: Sequence[float] | np.ndarray) -> float:
    """Sample standard deviation with the ``n-1`` denominator."""
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least two observations")
    return float(x.std(ddof=1))


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------


def anderson_darling_statistic(
    values: Sequence[float] | np.ndarray, dist: str = "norm"
) -> float:
    r"""Anderson-Darling :math:`A^2` statistic, computed directly.

    .. math::

        A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}(2i-1)
              \left[\ln F(y_{(i)}) + \ln\{1 - F(y_{(n+1-i)})\}\right]

    Implemented here rather than via ``scipy.stats.anderson`` for two reasons:
    it is called inside a parametric-bootstrap loop (so it must be cheap), and
    SciPy 1.17 changed that function's return contract and will change it again
    in 1.19, which is precisely the kind of silent dependency drift this
    repository avoids.
    """
    x = np.sort(np.asarray(values, dtype=float).ravel())
    n = x.size
    cdf = sps.norm.cdf
    if dist != "norm":
        raise NotImplementedError(
            f"only the normal reference is implemented, got {dist!r}"
        )
    f = np.clip(cdf(x), 1e-300, 1.0 - 1e-16)
    weights = 2.0 * np.arange(1, n + 1) - 1.0
    # term i uses F(y_(i)) and 1 - F(y_(n+1-i))
    return float(-n - np.sum(weights * (np.log(f) + np.log1p(-f[::-1]))) / n)


def _anderson_darling_bootstrap_p(
    values: np.ndarray, alpha: float, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """Anderson-Darling statistic and a valid p-value.

    The parametric bootstrap simulates ``n_boot`` samples of the same size from
    the fitted normal, refits its mean and s.d., and returns the proportion of
    simulated statistics at least as large as the observed one.  This replaces
    ``scipy.stats.anderson``'s tabulated significance levels, which assume the
    parameters are *known* -- an assumption that fails here because the
    parameters come from the same 57 points.
    """
    stat = anderson_darling_statistic(values)
    n = values.size
    mu, sigma = float(values.mean()), float(values.std(ddof=1))
    sims = rng.normal(mu, sigma, size=(n_boot, n))
    stats = np.array([anderson_darling_statistic(row) for row in sims])
    p = float((np.sum(stats >= stat) + 1.0) / (n_boot + 1.0))
    return stat, p


def normality_battery(
    values: Sequence[float] | np.ndarray,
    label: str = "Q",
    n_boot: int | None = None,
) -> pd.DataFrame:
    """Normality and log-normality tests for one series.

    Parameters
    ----------
    values:
        The series to test.
    label:
        Name recorded in the output.
    n_boot:
        Parametric bootstrap replicates for the Anderson-Darling p-value.
    """
    x = np.asarray(values, dtype=float).ravel()
    n_boot = int(n_boot or _cfg.CONFIG.n_gof_bootstrap)
    rng = util.stream_rng(f"normality::{label}")
    rows: list[dict[str, Any]] = []

    def add(
        test: str, statistic: float, pvalue: float, alpha: float, note: str = ""
    ) -> None:
        rows.append(
            {
                "series": label,
                "test": test,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "alpha": alpha,
                "reject_h0_at_alpha": bool(pvalue < alpha),
                "note": note,
            }
        )

    add(
        "Shapiro-Wilk",
        *sps.shapiro(x)[:2],
        _cfg.CONFIG.alpha,
        "most powerful against departure in the tails",
    )
    add(
        "D'Agostino-Pearson K2",
        *sps.normaltest(x)[:2],
        _cfg.CONFIG.alpha,
        "tests skewness and kurtosis jointly",
    )
    add("Jarque-Bera", *sps.jarque_bera(x)[:2], _cfg.CONFIG.alpha, "asymptotic")
    add(
        "Kolmogorov-Smirnov (fitted params)",
        *sps.kstest(x, sps.norm.cdf, args=(x.mean(), x.std(ddof=1)))[:2],
        _cfg.CONFIG.alpha,
        "p-value is anti-conservative with estimated parameters",
    )
    ad_stat, ad_p = _anderson_darling_bootstrap_p(x, _cfg.CONFIG.alpha, rng, n_boot)
    add(
        "Anderson-Darling (bootstrap p)",
        ad_stat,
        ad_p,
        _cfg.CONFIG.alpha,
        f"parametric bootstrap, {n_boot} replicates",
    )

    frame = pd.DataFrame(rows)
    log_x = np.log(x)
    log_frame = _normality_rows(
        log_x, "log Q", n_boot, util.stream_rng("normality::log Q")
    )
    frame = pd.concat([frame, log_frame], ignore_index=True)
    log.debug("normality battery complete for %s (%d tests)", label, len(frame))
    return frame


def _normality_rows(
    x: np.ndarray, label: str, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        test: str, statistic: float, pvalue: float, alpha: float, note: str = ""
    ) -> None:
        rows.append(
            {
                "series": label,
                "test": test,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "alpha": alpha,
                "reject_h0_at_alpha": bool(pvalue < alpha),
                "note": note,
            }
        )

    add(
        "Shapiro-Wilk", *sps.shapiro(x)[:2], _cfg.CONFIG.alpha, "log-transformed series"
    )
    add(
        "D'Agostino-Pearson K2",
        *sps.normaltest(x)[:2],
        _cfg.CONFIG.alpha,
        "log-transformed series",
    )
    add(
        "Jarque-Bera",
        *sps.jarque_bera(x)[:2],
        _cfg.CONFIG.alpha,
        "log-transformed series",
    )
    return pd.DataFrame(rows)


def _require_time_ordered(
    values: Sequence[float] | np.ndarray, where: str
) -> np.ndarray:
    """Delegate to :func:`ffa_karad.util.require_time_ordered`.

    The guard against order-dependent tests being fed a sorted series lives in
    :mod:`ffa_karad.util` so that every time-ordered stage shares one
    implementation; this name is kept because the call sites below read better
    with the private alias.
    """
    return util.require_time_ordered(values, where)


# ---------------------------------------------------------------------------
# Randomness / independence
# ---------------------------------------------------------------------------


def wald_wolfowitz_runs(sequence: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Runs test above and below the median.

    ``E[runs] = 2 n1 n2 / n + 1`` and
    ``var(runs) = 2 n1 n2 (2 n1 n2 - n) / (n**2 (n - 1))``.
    """
    x = np.asarray(sequence, dtype=float).ravel()
    n = x.size
    signs = np.sign(x - np.median(x))
    signs = signs[signs != 0]
    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    n1 = int(np.sum(signs > 0))
    n2 = int(np.sum(signs < 0))
    expected = 2.0 * n1 * n2 / n + 1.0
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1.0))
    z = (runs - expected) / np.sqrt(variance)
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(z))))
    return {
        "n": n,
        "runs": runs,
        "expected_runs": expected,
        "variance": variance,
        "z": float(z),
        "p_value": p,
        "n_above": n1,
        "n_below": n2,
    }


def durbin_watson(sequence: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Durbin-Watson statistic with an approximate p-value.

    The statistic ``d = sum (e_t - e_{t-1})**2 / sum e_t**2`` has no exact
    finite-sample null distribution; the tabulated critical values depend on
    the number of regressors and their values.  For a raw series the asymptotic
    null is approximately ``N(2, 4/n)``, which is used here.  This is labelled
    approximate because it is, and because a p-value from it should not be the
    sole basis for a conclusion.
    """
    x = np.asarray(sequence, dtype=float).ravel()
    n = x.size
    denom = float(np.sum((x[1:] - x[0]) ** 2))
    numer = float(np.sum(x**2))
    d = numer / denom if denom > 0 else float("nan")
    sd = np.sqrt(4.0 / n)
    z = (d - 2.0) / sd
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(z))))
    band = (
        "positive autocorrelation"
        if d < 1.5
        else "negative autocorrelation" if d > 2.5 else "no strong first-order signal"
    )
    return {
        "durbin_watson_d": float(d),
        "approximate_z": float(z),
        "approximate_p_value": p,
        "asymptotic_sd": float(sd),
        "interpretation": band,
        "caveat": "asymptotic N(2, 4/n); approximate",
    }


def randomness_battery(
    values: Sequence[float] | np.ndarray, max_lag: int = 10
) -> pd.DataFrame:
    """Runs test, Durbin-Watson and a lag scan of the Ljung-Box statistic."""
    x = _require_time_ordered(values, "randomness_battery")
    rows: list[dict[str, Any]] = []
    runs = wald_wolfowitz_runs(x)
    rows.append(
        {
            "test": "Wald-Wolfowitz runs",
            "lag": np.nan,
            "statistic": runs["z"],
            "detail": f"runs = {runs['runs']} (expected {runs['expected_runs']:.1f})",
            "p_value": runs["p_value"],
            "alpha": _cfg.CONFIG.alpha,
            "reject_h0_at_alpha": runs["p_value"] < _cfg.CONFIG.alpha,
            "note": "H0 = randomness; sensitive to clustering, not just linear dependence",
        }
    )
    dw = durbin_watson(x)
    rows.append(
        {
            "test": "Durbin-Watson",
            "lag": np.nan,
            "statistic": dw["durbin_watson_d"],
            "detail": f"d = {dw['durbin_watson_d']:.4f} ({dw['interpretation']})",
            "p_value": dw["approximate_p_value"],
            "alpha": _cfg.CONFIG.alpha,
            "reject_h0_at_alpha": dw["approximate_p_value"] < _cfg.CONFIG.alpha,
            "note": dw["caveat"],
        }
    )
    from statsmodels.stats.diagnostic import acorr_ljungbox

    for m in range(1, max_lag + 1):
        lb = acorr_ljungbox(x, lags=[m], return_df=True)
        q, p = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
        rows.append(
            {
                "test": "Ljung-Box",
                "lag": m,
                "statistic": q,
                "detail": "",
                "p_value": p,
                "alpha": _cfg.CONFIG.alpha,
                "reject_h0_at_alpha": bool(p < _cfg.CONFIG.alpha),
                "note": "H0 = no linear autocorrelation up to this lag",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------


def stationarity_battery(values: Sequence[float] | np.ndarray) -> pd.DataFrame:
    """Augmented Dickey-Fuller and KPSS on the raw and log series."""
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    ordered = _require_time_ordered(values, "stationarity_battery")
    rows: list[dict[str, Any]] = []
    series = {"Q": ordered, "log Q": np.log(ordered)}
    for name, x in series.items():
        for regression, label in (("c", "constant"), ("ct", "constant+trend")):
            try:
                stat, p, lags, n_obs, crit, icbest = adfuller(
                    x, regression=regression, autolag="AIC"
                )
                rows.append(
                    {
                        "series": name,
                        "test": f"ADF ({label})",
                        "statistic": float(stat),
                        "p_value": float(p),
                        "n_lags": int(lags),
                        "n_obs": int(n_obs),
                        "critical_1pct": float(crit[0]),
                        "critical_5pct": float(crit[2]),
                        "critical_10pct": float(crit[3]),
                        "reject_h0_unit_root": bool(p < _cfg.CONFIG.alpha),
                        "h0": "unit root / non-stationary",
                    }
                )
            except Exception as exc:  # pragma: no cover
                rows.append(
                    {
                        "series": name,
                        "test": f"ADF ({label})",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "note": f"failed: {exc}",
                    }
                )
        for regression, label in (("c", "level"), ("ct", "level+trend")):
            try:
                # statsmodels interpolates the KPSS p-value from a lookup table
                # and warns when the statistic falls outside it; the true p-value
                # is then smaller than the value returned, so the result is
                # explicitly marked as censored.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    stat, p, crit, lags = kpss(x, regression=regression, nlags="auto")
                censored = bool(p >= 0.0999) or any(
                    "outside of the range" in str(w.message) for w in caught
                )
                p_reported = float(min(max(p, 0.01), 0.999))
                rows.append(
                    {
                        "series": name,
                        "test": f"KPSS ({label})",
                        "statistic": float(stat),
                        "p_value": p_reported,
                        "n_lags": int(lags),
                        "critical_10pct": float(crit[0]),
                        "critical_5pct": float(crit[1]),
                        "critical_1pct": float(crit[2]),
                        "reject_h0_stationary": bool(p < _cfg.CONFIG.alpha),
                        "p_value_censored_at_0p10": censored,
                        "h0": "level-stationary",
                    }
                )
            except Exception as exc:  # pragma: no cover
                rows.append(
                    {
                        "series": name,
                        "test": f"KPSS ({label})",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "note": f"failed: {exc}",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def mann_kendall(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Original Mann-Kendall test with a tie correction.

    ``S = sum_{k} sum_{j>k} sign(x_j - x_k)``,
    ``var(S) = [n(n-1)(2n+5) - sum_m t_m (t_m - 1) (2 t_m + 5)] / 18``,
    ``z = (S - sign(S)) / sqrt(var(S))``.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    s = 0
    for k in range(n - 1):
        s += int(np.sign(x[k + 1 :] - x[k]).sum())

    _, counts = np.unique(x, return_counts=True)
    tie_term = float(
        np.sum(
            counts[counts > 1] * (counts[counts > 1] - 1) * (2 * counts[counts > 1] + 5)
        )
    )
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return {
            "S": float(s),
            "z": float("nan"),
            "p_value": float("nan"),
            "var_s": var_s,
            "tau": float("nan"),
            "note": "zero variance in S",
        }
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(z))))
    # Kendall's tau-b
    n0 = 0.5 * n * (n - 1)
    n1 = 0.5 * tie_term
    denom = np.sqrt((n0 - n1) * (n0 - n1))
    tau = float(s / denom) if denom > 0 else float("nan")
    return {
        "S": float(s),
        "z": float(z),
        "p_value": p,
        "var_s": float(var_s),
        "tau": tau,
        "ties_corrected": True,
    }


def hamed_rao_mann_kendall(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Hamed & Rao (1998) variance-corrected Mann-Kendall test.

    MK assumes pre-whitened series are serially independent; Hamed & Rao
    multiply the variance by a correction factor so that the test remains valid
    when the series are serially correlated.  For an annual-maxima series with
    detected autocorrelation this is the appropriate variant.
    """
    base = mann_kendall(values)
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    _, counts = np.unique(x, return_counts=True)
    tied = counts[counts > 1]
    n1 = 0.5 * n * (n - 1)
    n2 = float(np.sum(tied * (tied - 1) * (tied + 2)))
    if n1 <= 0:
        return {**base, "correction": float("nan"), "note": "degenerate"}
    correction_n = n1 - n2 / n1
    if correction_n <= 0:
        return {**base, "correction": float("nan"), "note": "negative correction"}
    inflation = np.sqrt(1.0 + n1 * (n1 - n2) / (correction_n * (n1 - 0.5 * n1)))
    z = base["z"] / inflation
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(z))))
    return {
        "S": base["S"],
        "z": float(z),
        "p_value": p,
        "var_s": base["var_s"],
        "tau": base["tau"],
        "correction_factor": float(inflation),
        "note": "Hamed & Rao (1998) variance-corrected",
    }


def theil_sen(
    values: Sequence[float] | np.ndarray,
    index: Sequence[float] | np.ndarray | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Sen's slope with its confidence interval.

    The notebook labelled this slope ``cm/yr``.  It is a **discharge** slope in
    cumecs per year; the metres belong to the water-level series.
    """
    x = np.asarray(values, dtype=float).ravel()
    t = (
        np.arange(x.size, dtype=float)
        if index is None
        else np.asarray(index, dtype=float).ravel()
    )
    alpha = _cfg.CONFIG.alpha if alpha is None else alpha
    res = sps.theilslopes(x, t, alpha=alpha)
    return {
        "slope_cumecs_per_year": float(res.slope),
        "intercept_cumecs": float(res.intercept),
        "slope_ci_low": float(res.low_slope),
        "slope_ci_high": float(res.high_slope),
        "confidence_level": 1.0 - alpha,
        "significant": bool(res.low_slope > 0 or res.high_slope < 0),
        "unit": "cumecs/yr",
    }


def trend_battery(
    values: Sequence[float] | np.ndarray,
    years: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Mann-Kendall, Hamed-Rao, Spearman, Kendall and trend-free pre-whitening."""
    x = _require_time_ordered(values, "trend_battery")
    t = (
        np.arange(x.size, dtype=float)
        if years is None
        else np.asarray(years, dtype=float).ravel()
    )
    n = x.size
    alpha = _cfg.CONFIG.alpha
    rows: list[dict[str, Any]] = []

    mk = mann_kendall(x)
    rows.append(
        {
            "test": "Mann-Kendall (original)",
            "statistic": mk["z"],
            "p_value": mk["p_value"],
            "alpha": alpha,
            "reject_h0_at_alpha": mk["p_value"] < alpha,
            "detail": f"S = {mk['S']:.0f}, tau = {mk['tau']:.4f}",
            "h0": "no monotonic trend",
        }
    )
    hr = hamed_rao_mann_kendall(x)
    rows.append(
        {
            "test": "Mann-Kendall (Hamed-Rao corrected)",
            "statistic": hr["z"],
            "p_value": hr["p_value"],
            "alpha": alpha,
            "reject_h0_at_alpha": hr["p_value"] < alpha,
            "detail": f"correction factor {hr.get('correction_factor', float('nan')):.4f}",
            "h0": "no monotonic trend, serially correlated series",
        }
    )
    for name, fn in (("Spearman rho", sps.spearmanr), ("Kendall tau", sps.kendalltau)):
        res = fn(t, x)
        stat, p = (
            (float(res.statistic), float(res.pvalue))
            if hasattr(res, "pvalue")
            else (float(res[0]), float(res[1]))
        )
        rows.append(
            {
                "test": name,
                "statistic": stat,
                "p_value": p,
                "alpha": alpha,
                "reject_h0_at_alpha": p < alpha,
                "detail": "monotonic association with water-year index",
                "h0": "no monotonic trend",
            }
        )

    ts = theil_sen(x, t)
    rows.append(
        {
            "test": "Sen's slope",
            "statistic": ts["slope_cumecs_per_year"],
            "p_value": np.nan,
            "alpha": alpha,
            "reject_h0_at_alpha": ts["significant"],
            "detail": f"95% CI [{ts['slope_ci_low']:.4f}, {ts['slope_ci_high']:.4f}] "
            f"{ts['unit']}",
            "h0": "slope = 0",
        }
    )

    # Trend-free pre-whitening: remove the fitted Sen line, test the residuals,
    # then re-add the slope scaled by its significance.
    slope, intercept = sps.theilslopes(x, t).slope, sps.theilslopes(x, t).intercept
    detrended = x - (intercept + slope * t)
    mk_detr = mann_kendall(detrended)
    t_p_value = mk_detr["p_value"] if np.isfinite(mk_detr["p_value"]) else 1.0
    adjusted = t_p_value * slope
    rows.append(
        {
            "test": "Trend-free pre-whitened MK",
            "statistic": mk_detr["z"],
            "p_value": t_p_value,
            "alpha": alpha,
            "reject_h0_at_alpha": t_p_value < alpha,
            "detail": f"adjusted Sen slope {adjusted:+.6f} cumecs/yr after "
            f"pre-whitening",
            "h0": "no residual trend after removing the linear component",
        }
    )

    frame = pd.DataFrame(rows)
    log.info(
        "trend battery on N=%d: %s",
        n,
        "; ".join(
            f"{r['test']} p={r['p_value']:.4f}"
            for r in rows
            if np.isfinite(r["p_value"])
        ),
    )
    return frame


# ---------------------------------------------------------------------------
# Change point
# ---------------------------------------------------------------------------


def pettitt_test(
    values: Sequence[float] | np.ndarray,
    n_perm: int = 5000,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Pettitt's non-parametric change-point test with a permutation p-value.

    The statistic is ``U* = max_k |U_k|`` where
    ``U_k = sum_{i<=k} sum_{j>k} sign(x_i - x_j)``.

    The p-value is obtained by **permutation** rather than from the asymptotic
    constant in the original paper, because that constant is easy to
    mis-transcribe and a wrong change-point p-value would be an indefensible
    basis for splitting the record.
    """
    alpha = _cfg.CONFIG.alpha if alpha is None else alpha
    x = _require_time_ordered(values, "pettitt_test")
    n = x.size
    if n < 10:
        raise ValueError("Pettitt's test needs at least 10 observations")

    def statistic(series: np.ndarray) -> tuple[float, int]:
        total = 0.0
        best_u, best_k = 0.0, 0
        for k in range(n - 1):
            left = series[: k + 1]
            right = series[k + 1 :]
            total += float(np.sign(left[:, None] - right[None, :]).sum())
            if abs(total) > best_u:
                best_u, best_k = abs(total), k + 1
        return best_u, best_k

    u_star, k_star = statistic(x)
    rng = util.stream_rng("pettitt")
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(x)
        u_perm, _ = statistic(perm)
        if u_perm >= u_star:
            exceed += 1
    p = float((exceed + 1.0) / (n_perm + 1.0))
    return {
        "U_star": float(u_star),
        "change_point_index": int(k_star),
        "change_point_position": int(k_star),
        "p_value": p,
        "n_permutations": int(n_perm),
        "reject_h0_at_alpha": bool(p < alpha),
        "h0": "no change point",
        "note": "change point at index k means the first k observations are one "
        "regime; interpret against the water-year labels, not the index",
    }


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StatisticalTestResults:
    """Everything this module produced, for downstream stages and the report."""

    moments: MomentCoefficients
    skewness_estimators: pd.DataFrame
    normality: pd.DataFrame
    randomness: pd.DataFrame
    stationarity: pd.DataFrame
    trend: pd.DataFrame
    change_point: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "moments": self.moments.as_row(),
            "skewness_estimators": self.skewness_estimators.to_dict("records"),
            "normality": self.normality.to_dict("records"),
            "randomness": self.randomness.to_dict("records"),
            "stationarity": self.stationarity.to_dict("records"),
            "trend": self.trend.to_dict("records"),
            "change_point": self.change_point,
            "warnings": self.warnings,
        }


def run(values: Sequence[float] | np.ndarray) -> StatisticalTestResults:
    """Run the full battery on the annual peak series.

    Parameters
    ----------
    values:
        The series **in water-year order**.  Passing the ascending order
        statistics is rejected by the order-dependent tests.
    """
    ordered = _require_time_ordered(values, "statistical_tests.run")
    x = np.asarray(ordered, dtype=float).ravel()
    moments = moment_coefficients(x)
    skew_estimators = secondary_skewness_estimators(x)
    warnings_list: list[str] = []

    if moments.skewness_adjusted_b17b > _cfg.CONFIG.bulletin_17c_skew_threshold:
        warnings_list.append(
            f"Bulletin 17C: the adjusted skewness "
            f"{moments.skewness_adjusted_b17b:.4f} exceeds "
            f"{_cfg.CONFIG.bulletin_17c_skew_threshold}, so the confidence limits "
            f"must be governed by KURTOSIS rather than skewness. See "
            f"ffa_karad.skewness_limits."
        )

    spread_set = skew_estimators.loc[skew_estimators["use_in_spread"], "value"]
    spread = float(spread_set.max() - spread_set.min())
    if spread > 0.5:
        warnings_list.append(
            f"Skewness estimators span {spread:.3f} (from "
            f"{spread_set.min():.3f} to {spread_set.max():.3f}). The moment "
            f"skewness is driven by the single largest year; the quartile-based "
            f"measures are not. This divergence is the reason Bulletin 17C "
            f"confidence limits are applied rather than the point skewness alone."
        )

    change = pettitt_test(x)
    if change["reject_h0_at_alpha"]:
        warnings_list.append(
            f"Pettitt's test rejects homogeneity at p = {change['p_value']:.4f}, "
            f"with a change point at index {change['change_point_index']}. A "
            f"non-stationary record cannot be summarised by a single frequency "
            f"curve without a decision on how to handle the shift."
        )

    results = StatisticalTestResults(
        moments=moments,
        skewness_estimators=skew_estimators,
        normality=normality_battery(x, "Q"),
        randomness=randomness_battery(x),
        stationarity=stationarity_battery(x),
        trend=trend_battery(x),
        change_point=change,
        warnings=warnings_list,
    )
    log.info(
        "moments: Cs(b17b) = %.4f +/- %.4f, Ck = %.4f (excess %.4f)",
        moments.skewness_adjusted_b17b,
        moments.skewness_se,
        moments.kurtosis_moment_b17b,
        moments.excess_kurtosis,
    )
    for w in warnings_list:
        log.warning("%s", w)
    return results


def summarise(res: StatisticalTestResults) -> str:
    """Plain-text summary for the report."""
    m = res.moments
    lines = [
        "STATISTICAL TEST SUMMARY",
        f"  n                              : {m.n}",
        f"  mean / sd / CV                 : {m.mean:.4f} / {m.sd:.4f} / {m.cv:.4f}",
        f"  Cs unadjusted (g1)             : {m.skewness_unadjusted:.6f}",
        f"  Cs adjusted (Bulletin 17B)     : {m.skewness_adjusted_b17b:.6f} "
        f"(se {m.skewness_se:.4f})",
        f"  Ck = m4/m2^2 (Bulletin 17B)    : {m.kurtosis_moment_b17b:.6f} "
        f"(se {m.kurtosis_se:.4f})",
        f"  excess kurtosis = Ck - 3       : {m.excess_kurtosis:.6f}",
        "",
        "  secondary skewness estimators:",
    ]
    for r in res.skewness_estimators.itertuples():
        lines.append(
            f"    {r.estimator:<40}: {r.value:+.4f}"
            + ("  (robust)" if r.robust else "")
            + (
                ""
                if r.use_in_spread
                else "  [excluded from spread: ~0 by construction]"
            )
        )
    lines += ["", "  normality:"]
    for r in res.normality.itertuples():
        lines.append(
            f"    {r.series:<6} {r.test:<36} p = {r.p_value:.4f}"
            f"{'  REJECT' if r.reject_h0_at_alpha else ''}"
        )
    lines += ["", "  randomness:"]
    for r in res.randomness.itertuples():
        lag = "" if not np.isfinite(r.lag) else f"lag {int(r.lag)}"
        lines.append(
            f"    {r.test:<36} {lag:<8} p = {r.p_value:.4f}"
            f"{'  REJECT' if r.reject_h0_at_alpha else ''}"
        )
    lines += ["", "  trend:"]
    for r in res.trend.itertuples():
        lines.append(
            f"    {r.test:<36} p = {r.p_value:.4f}"
            f"{'  REJECT' if r.reject_h0_at_alpha else ''}"
            + (f"  [{r.detail}]" if isinstance(r.detail, str) else "")
        )
    cp = res.change_point
    lines += [
        "",
        f"  Pettitt change point           : index {cp['change_point_position']}, "
        f"p = {cp['p_value']:.4f} "
        f"({'reject' if cp['reject_h0_at_alpha'] else 'do not reject'} homogeneity)",
    ]
    if res.warnings:
        lines += ["", "  WARNINGS:"]
        lines += [f"    - {w}" for w in res.warnings]
    return "\n".join(lines)


def bootstrap_moment_samples(
    values: Sequence[float] | np.ndarray, n_bootstrap: int = 2000
) -> pd.DataFrame:
    """Non-parametric bootstrap of the sample skewness, kurtosis and CV.

    The Bulletin 17B confidence limits in :mod:`ffa_karad.skewness_limits` are
    parametric: they refit LP3 on log Q and transform the skewness back.  This
    is the complementary non-parametric view -- the annual peaks are resampled
    with replacement and the ordinary sample moments are recomputed -- so the
    asymmetry and tail-weighting of the record can be shown with their own
    sampling distribution rather than the log-space one.

    Sampling is with replacement, so the resampled series need not be monotone;
    the order-dependent tests in this module do not apply to it.

    Parameters
    ----------
    values:
        The series **in water-year order**.
    n_bootstrap:
        Number of resamples.

    Returns
    -------
    pandas.DataFrame
        One row per resample with ``bootstrap``, ``skewness``, ``kurtosis``
        (excess), ``cv`` and ``mean_cumecs``.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n < 3:
        raise ValueError("at least three annual peaks are needed to bootstrap")
    rng = util.stream_rng("bootstrap::moments")
    rows = np.empty((int(n_bootstrap), 5), dtype=float)
    for b in range(int(n_bootstrap)):
        sample = x[rng.integers(0, n, size=n)]
        rows[b, 0] = b
        rows[b, 1] = sps.skew(sample, bias=False)
        rows[b, 2] = sps.kurtosis(sample, fisher=True, bias=False)
        rows[b, 3] = sample.std(ddof=1) / sample.mean() if sample.mean() else np.nan
        rows[b, 4] = sample.mean()
    return pd.DataFrame(
        rows,
        columns=[
            "bootstrap",
            "skewness",
            "kurtosis",
            "cv",
            "mean_cumecs",
        ],
    )


__all__ = [
    "MomentCoefficients",
    "StatisticalTestResults",
    "bootstrap_moment_samples",
    "moment_coefficients",
    "secondary_skewness_estimators",
    "sd_of",
    "anderson_darling_statistic",
    "normality_battery",
    "wald_wolfowitz_runs",
    "durbin_watson",
    "randomness_battery",
    "stationarity_battery",
    "mann_kendall",
    "hamed_rao_mann_kendall",
    "theil_sen",
    "trend_battery",
    "pettitt_test",
    "run",
    "summarise",
]
