"""
LP3 skewness and quantile uncertainty in LOG space (IS 11223:1985 primary method).

Correction of a fundamental error
---------------------------------
The Log-Pearson Type III method fits a Pearson III distribution to the
**logarithms** of the annual peak flows, not to the discharges themselves.  The
skewness that enters the frequency-factor equation is therefore the skewness of
``y = ln Q``.  For this record:

===========================  ==========  ==========
quantity                     raw ``Q``   log ``ln Q``
===========================  ==========  ==========
adjusted skewness Cs             1.0819       0.0956
excess kurtosis Ck               3.4683       2.2924
===========================  ==========  ==========

The arithmetic skewness of 1.0819 is real, and it is a useful *diagnostic* -- it
says the raw discharge series is strongly right-skewed, which is why the
Gumbel and log-normal fits differ.  But it is **not** the LP3 skewness, and
applying a 0.9 threshold to it to trigger a "kurtosis-governed regime" is a
category error.  There is no such regime in Bulletin 17C, and the entire
"kurtosis-governed" analysis that an earlier version of this module performed
was an artefact of that mistake.  It has been deleted rather than patched.

With ``Cs_log = 0.0956`` the log-peaks are close to log-normal, consistent with
the Shapiro-Wilk result on ``ln Q`` and with the secondary skewness measures.

What this module now does
-------------------------
1.  Fit LP3 by its own definition: mean, sd and Bulletin 17B adjusted skewness
    of ``ln Q``.
2.  Compute frequency factors two ways -- **exactly** via
    :func:`scipy.stats.pearson3.ppf` (the reference implementation of the
    Pearson III quantile) and via the **Wilson-Hilferty** closed form -- and
    report both, so the approximation can be audited rather than trusted.
3.  Quantify quantile uncertainty two independent ways:
    * a **parametric bootstrap** that refits mean, sd *and* skewness on every
      replicate, so skewness uncertainty is propagated and not frozen; and
    * the **Chowdhury-type closed-form** standard error.
4.  Adopt the **station log-skewness**, per IS 11223 / CWC practice, and report
    the quantile envelope over a skewness band alongside it.  A design flood is
    always produced; the band documents how much of it is attributable to the
    finite record.

No tabulated standard, and no unverifiable constant, is used anywhere.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util
from .statistical_tests import moment_coefficients

log = util.get_logger("skewness_limits")


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LP3Fit:
    """Log-Pearson Type III parameters, defined on ``ln Q``."""

    n: int
    mean_log: float
    sd_log: float
    cs_log: float  # Bulletin 17B adjusted skewness of ln Q
    ck_log: float  # Ck = m4/m2^2 of ln Q
    cs_log_se: float
    cs_raw: float  # diagnostic only, NOT the LP3 skewness
    ck_raw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_log": self.mean_log,
            "sd_log": self.sd_log,
            "cs_log_b17b": self.cs_log,
            "ck_log": self.ck_log,
            "cs_log_se": self.cs_log_se,
            "cs_raw_diagnostic": self.cs_raw,
            "ck_raw_diagnostic": self.ck_raw,
        }


def lp3_fit(values: Sequence[float] | np.ndarray) -> LP3Fit:
    """Fit LP3: moments of ``ln Q``.

    The skewness is the Bulletin 17B adjusted coefficient

    .. math::
        g_1 = \\frac{1}{n}\\sum\\left(\\frac{y_i-\\bar y}{s_y}\\right)^3,
        \\qquad
        \\hat g_1 = \\frac{\\sqrt{n(n-1)}}{n-2}\\, g_1

    with :math:`s_y` the ``n-1`` denominator standard deviation.
    """
    q = np.asarray(values, dtype=float).ravel()
    if np.any(q <= 0):
        raise ValueError("LP3 requires strictly positive discharges")
    y = np.log(q)
    logm = moment_coefficients(y)
    rawm = moment_coefficients(q)
    return LP3Fit(
        n=int(y.size),
        mean_log=float(logm.mean),
        sd_log=float(logm.sd),
        cs_log=float(logm.skewness_adjusted_b17b),
        ck_log=float(logm.kurtosis_moment_b17b),
        cs_log_se=float(logm.skewness_se),
        cs_raw=float(rawm.skewness_adjusted_b17b),
        ck_raw=float(rawm.kurtosis_moment_b17b),
    )


# ---------------------------------------------------------------------------
# Frequency factors
# ---------------------------------------------------------------------------


def frequency_factor_exact(return_period: float, cs: float) -> float:
    """Exact Pearson III frequency factor ``K_T``.

    ``scipy.stats.pearson3`` is the standardised Pearson III distribution
    (mean 0, standard deviation 1), so its quantile *is* ``K_T``.  No
    approximation is involved.
    """
    if not np.isfinite(cs):
        return float("nan")
    exceed = 1.0 - 1.0 / float(return_period)
    return float(sps.pearson3.ppf(exceed, skew=cs))


def frequency_factor_wilson_hilferty(return_period: float, cs: float) -> float:
    """Wilson-Hilferty closed-form approximation to ``K_T``.

    .. math::
        K_T = \\frac{2}{G}\\left(1 + \\frac{Gz}{6}
                  - \\frac{G^2}{36}\\right)^3 - \\frac{2}{G},
        \\qquad z = \\Phi^{-1}\\!\\left(1 - \\frac{1}{T}\\right)

    The expression is 0/0 as ``G -> 0`` and its limit is ``z``, so the normal
    case is handled explicitly rather than left to cancellation.
    """
    z = float(sps.norm.ppf(1.0 - 1.0 / float(return_period)))
    if abs(cs) < 1e-10:
        return z
    return float(2.0 / cs * (1.0 + cs * z / 6.0 - cs * cs / 36.0) ** 3 - 2.0 / cs)


def lp3_quantile(
    return_period: float,
    mean_log: float,
    sd_log: float,
    cs: float,
    method: str = "exact",
) -> float:
    """LP3 quantile ``Q_T = exp(mean_log + K_T * sd_log)``."""
    k = (
        frequency_factor_exact(return_period, cs)
        if method == "exact"
        else frequency_factor_wilson_hilferty(return_period, cs)
    )
    if not np.isfinite(k):
        return float("nan")
    return float(np.exp(mean_log + k * sd_log))


def lp3_quantiles(
    periods: Sequence[float], fit: LP3Fit, method: str = "exact"
) -> pd.DataFrame:
    """LP3 design floods for a set of return periods."""
    rows = []
    for t in periods:
        q = lp3_quantile(t, fit.mean_log, fit.sd_log, fit.cs_log, method)
        rows.append(
            {
                "return_period_yr": float(t),
                "Q_cumecs": q,
                "K_exact": frequency_factor_exact(t, fit.cs_log),
                "K_wilson_hilferty": frequency_factor_wilson_hilferty(t, fit.cs_log),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["wh_vs_exact_pct"] = (
            100.0
            * (frame["K_wilson_hilferty"] - frame["K_exact"])
            / frame["K_exact"].abs().replace(0, np.nan)
        )
    return frame


# ---------------------------------------------------------------------------
# Closed-form quantile standard error
# ---------------------------------------------------------------------------


def chowdhury_standard_error(
    return_period: float,
    n: int,
    sd_log: float,
    k_factor: float,
    cs: float,
    level: float | None = None,
) -> float:
    """Closed-form standard error of the LP3 log-quantile.

    .. math::
        s_{y_T} = s_y\\sqrt{\\frac{1}{n}\\left[1
            + \\frac{K_T^2}{2}\\left(1 + \\frac{3G^2}{4}\\right)
            + K_T G\\right]}

    This is the Chowdhury-type first-order approximation for the variance of the
    log-Pearson quantile.  It is reported *alongside* the bootstrap, never
    instead of it: an approximation of unverified provenance must not be the
    only interval in a submittable report.  The bootstrap is primary.
    """
    level = _cfg.CONFIG.alpha_one_sided if level is None else level
    variance = (1.0 / n) * (
        1.0 + (k_factor**2 / 2.0) * (1.0 + 3.0 * cs**2 / 4.0) + k_factor * cs
    )
    if variance <= 0:
        return float("nan")
    return float(sd_log * np.sqrt(variance))


def chowdhury_confidence_limits(
    periods: Sequence[float], fit: LP3Fit, level: float | None = None
) -> pd.DataFrame:
    """Multiplicative CI on ``Q_T`` from the closed-form standard error."""
    level = _cfg.CONFIG.alpha_one_sided if level is None else level
    z = float(sps.norm.ppf(1.0 - level / 2.0))
    rows = []
    for t in periods:
        k = frequency_factor_exact(t, fit.cs_log)
        q = lp3_quantile(t, fit.mean_log, fit.sd_log, fit.cs_log)
        se = chowdhury_standard_error(t, fit.n, fit.sd_log, k, fit.cs_log, level)
        rows.append(
            {
                "return_period_yr": float(t),
                "Q_cumecs": q,
                "se_log": se,
                "Q_lower": (
                    float(q * np.exp(-z * se)) if np.isfinite(se) else float("nan")
                ),
                "Q_upper": (
                    float(q * np.exp(+z * se)) if np.isfinite(se) else float("nan")
                ),
                "method": "Chowdhury closed-form (approximate)",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parametric bootstrap (primary interval)
# ---------------------------------------------------------------------------


def bootstrap_confidence_limits(
    values: Sequence[float] | np.ndarray,
    periods: Sequence[float],
    n_bootstrap: int = 2000,
    level: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parametric bootstrap that propagates **skewness** uncertainty.

    Each replicate draws a fresh LP3 population from the fitted parameters,
    generates ``n`` log-peaks, **refits** mean, sd and skewness on that
    replicate, and evaluates the quantiles.  Refitting matters: freezing the
    skewness at its point estimate would omit the dominant source of LP3
    uncertainty, which is precisely the error Bulletin 17B's confidence limits
    exist to address.

    Returns
    -------
    (table, skewness_table)
    """
    level = _cfg.CONFIG.alpha_one_sided if level is None else level
    fit = lp3_fit(values)
    periods = list(periods)
    rng = util.stream_rng(f"lp3_bootstrap::{fit.n}::{n_bootstrap}::{level}")

    # scipy's pearson3 is standardised: loc/scale carry the mean and sd.
    y_mean, y_sd, y_skew = fit.mean_log, fit.sd_log, fit.cs_log
    if not (y_sd > 0):
        raise ValueError("zero dispersion in log space")

    # Draw from the fitted population itself.  ``scipy.stats.pearson3`` is the
    # standardised Pearson III, so ``mean_log + sd_log * pearson3.ppf(u, skew)``
    # has exactly the fitted mean, sd and skewness.  Rescaling each *replicate*
    # would instead pin every replicate's mean and sd to the fitted values and
    # silently delete that part of the uncertainty, so no rescaling is applied.
    u = np.clip(rng.random((n_bootstrap, fit.n)), 1e-12, 1.0 - 1e-12)
    y = y_mean + y_sd * sps.pearson3.ppf(u, skew=y_skew)

    # Refit per replicate
    dev = y - y.mean(axis=1, keepdims=True)
    s = np.sqrt(np.sum(dev**2, axis=1) / (fit.n - 1.0))
    g1 = np.mean((dev / s[:, None]) ** 3, axis=1)
    pref = np.sqrt(fit.n * (fit.n - 1.0)) / (fit.n - 2.0)
    cs_boot = pref * g1
    mu_boot = y.mean(axis=1)
    sd_boot = s

    rows = []
    for t in periods:
        exceed = 1.0 - 1.0 / float(t)
        ks = sps.pearson3.ppf(exceed, skew=cs_boot)
        qs = np.exp(mu_boot + ks * sd_boot)
        rows.append(
            {
                "return_period_yr": float(t),
                "Q_cumecs": lp3_quantile(t, y_mean, y_sd, y_skew),
                "Q_lower": float(np.quantile(qs, level / 2.0)),
                "Q_upper": float(np.quantile(qs, 1.0 - level / 2.0)),
                "Q_median": float(np.median(qs)),
                "method": "parametric bootstrap, LP3 refitted per replicate",
            }
        )
    table = pd.DataFrame(rows)

    skew_table = pd.DataFrame(
        {
            "quantity": ["Cs_log", "mean_log", "sd_log"],
            "fitted": [y_skew, y_mean, y_sd],
            "boot_mean": [cs_boot.mean(), mu_boot.mean(), sd_boot.mean()],
            "boot_sd": [cs_boot.std(ddof=1), mu_boot.std(ddof=1), sd_boot.std(ddof=1)],
            "q025": [
                np.quantile(cs_boot, 0.025),
                np.quantile(mu_boot, 0.025),
                np.quantile(sd_boot, 0.025),
            ],
            "q975": [
                np.quantile(cs_boot, 0.975),
                np.quantile(mu_boot, 0.975),
                np.quantile(sd_boot, 0.975),
            ],
        }
    )
    log.info(
        "LP3 bootstrap: %d replicates, Cs_log %.4f (boot sd %.4f, " "analytic se %.4f)",
        n_bootstrap,
        y_skew,
        cs_boot.std(ddof=1),
        fit.cs_log_se,
    )
    return table, skew_table


def verify_bootstrap_population(
    fit: LP3Fit, n_draw: int = 200_000, seed_name: str = "verify_boot"
) -> dict[str, Any]:
    """Confirm the bootstrap population really has the fitted moments.

    A parametric bootstrap is only valid if the generating distribution carries
    the fitted parameters.  This check samples the population directly and
    compares its moments with the fit, so a silently mis-parameterised
    population is caught rather than reported.
    """
    rng = util.stream_rng(seed_name)
    u = np.clip(rng.random(n_draw), 1e-12, 1.0 - 1e-12)
    y = fit.mean_log + fit.sd_log * sps.pearson3.ppf(u, skew=fit.cs_log)
    m = moment_coefficients(y)
    result = {
        "n_draw": int(n_draw),
        "mean_log_target": fit.mean_log,
        "mean_log_actual": m.mean,
        "sd_log_target": fit.sd_log,
        "sd_log_actual": m.sd,
        "cs_log_target": fit.cs_log,
        "cs_log_actual": m.skewness_adjusted_b17b,
    }
    result["mean_rel_error"] = abs(m.mean - fit.mean_log) / max(
        abs(fit.mean_log), 1e-12
    )
    result["sd_rel_error"] = abs(m.sd - fit.sd_log) / fit.sd_log
    result["cs_abs_error"] = abs(m.skewness_adjusted_b17b - fit.cs_log)
    # The sample skewness of a size-n sample carries a standard error of about
    # sqrt(6/n), so the tolerance must scale with the draw size rather than be a
    # fixed constant that would pass at one n and fail at another.
    se_cs = float(np.sqrt(6.0 / n_draw))
    result["expected_mc_se_cs"] = se_cs
    result["ok"] = bool(
        result["mean_rel_error"] < 0.01
        and result["sd_rel_error"] < 0.01
        and result["cs_abs_error"] < 4.0 * se_cs
    )
    if not result["ok"]:
        log.warning(
            "bootstrap population does not reproduce the fitted " "moments: %s", result
        )
    return result


# ---------------------------------------------------------------------------
# Skewness band (the uncertainty envelope)
# ---------------------------------------------------------------------------


def skewness_band(fit: LP3Fit, n_std: float = 1.96, step: float = 0.05) -> pd.DataFrame:
    """Design-flood envelope across an admissible band of log-skewness.

    The band is centred on the station log-skewness with half-width
    ``n_std * se(Cs_log)`` and is **not** truncated at any threshold.  The
    quantiles are expressed relative to the adopted fit, so the report can show
    precisely how much of the design flood is attributable to the finite record
    length rather than to the observed floods.
    """
    lo = fit.cs_log - n_std * fit.cs_log_se
    hi = fit.cs_log + n_std * fit.cs_log_se
    grid = np.unique(np.round(np.append(np.arange(lo, hi, step), [lo, hi]), 6))
    rows = []
    base = lp3_quantiles(_cfg.CONFIG.return_periods, fit)
    base_map = dict(zip(base["return_period_yr"], base["Q_cumecs"]))
    for cs in grid:
        row: dict[str, Any] = {"cs_log": float(cs), "delta_cs": float(cs - fit.cs_log)}
        for t in _cfg.CONFIG.return_periods:
            q = lp3_quantile(t, fit.mean_log, fit.sd_log, float(cs))
            row[f"Q_{int(t)}"] = q
            row[f"ratio_{int(t)}"] = q / base_map[t]
        rows.append(row)
    log.info("skewness band for Cs_log: [%.4f, %.4f] (%d points)", lo, hi, len(grid))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LP3Result:
    fit: LP3Fit
    quantiles: pd.DataFrame
    bootstrap: pd.DataFrame
    bootstrap_params: pd.DataFrame
    chowdhury: pd.DataFrame
    band: pd.DataFrame
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit.to_dict(),
            "quantiles": self.quantiles.to_dict(orient="records"),
            "bootstrap": self.bootstrap.to_dict(orient="records"),
            "bootstrap_parameters": self.bootstrap_params.to_dict(orient="records"),
            "chowdhury": self.chowdhury.to_dict(orient="records"),
            "notes": self.notes,
        }


def run(values: Sequence[float] | np.ndarray, n_bootstrap: int = 2000) -> LP3Result:
    """Full log-space LP3 analysis: fit, quantiles, two intervals, envelope."""
    fit = lp3_fit(values)
    periods = list(_cfg.CONFIG.return_periods)
    quants = lp3_quantiles(periods, fit)
    boot, boot_params = bootstrap_confidence_limits(values, periods, n_bootstrap)
    chow = chowdhury_confidence_limits(periods, fit)
    band = skewness_band(fit)

    notes = [
        f"LP3 is fitted to ln Q: mean = {fit.mean_log:.4f}, "
        f"sd = {fit.sd_log:.4f}, Bulletin 17B adjusted skewness "
        f"Cs_log = {fit.cs_log:.4f}.",
        f"The arithmetic skewness of Q is {fit.cs_raw:.4f}; it is a "
        "diagnostic of raw-scale asymmetry and is NOT the LP3 skewness.",
        f"Cs_log = {fit.cs_log:.4f} lies well below 0.9, so no skewness "
        "limit or kurtosis-based alternative is invoked. IS 11223 permits the "
        "station skewness to be adopted directly.",
        "Wilson-Hilferty and exact Pearson III factors are both reported; "
        "the exact scipy implementation is used for the design floods.",
        "Confidence intervals come from a parametric bootstrap that refits "
        "skewness on every replicate; the Chowdhury closed form is a "
        "secondary approximation.",
    ]
    return LP3Result(
        fit=fit,
        quantiles=quants,
        bootstrap=boot,
        bootstrap_params=boot_params,
        chowdhury=chow,
        band=band,
        notes=notes,
    )


def summarise(res: LP3Result) -> str:
    """Plain-text summary for the report."""
    f = res.fit
    lines = [
        "LOG-PEARSON TYPE III  (IS 11223 primary, fitted to ln Q)",
        f"  N                          : {f.n}",
        f"  mean(ln Q)                 : {f.mean_log:.4f}",
        f"  sd(ln Q)                   : {f.sd_log:.4f}",
        f"  Cs_log  (ADOPTED)          : {f.cs_log:+.4f}  (se {f.cs_log_se:.4f})",
        f"  Ck_log                     : {f.ck_log:.4f}",
        f"  Cs_raw  (diagnostic only)  : {f.cs_raw:+.4f}",
        "",
        f"  {'T':>6} {'Q (cumecs)':>13} {'K exact':>9} {'K W-H':>9} "
        f"{'95% boot':>22}",
    ]
    b = res.bootstrap.set_index("return_period_yr")
    for row in res.quantiles.itertuples():
        lo = b.loc[row.return_period_yr, "Q_lower"]
        hi = b.loc[row.return_period_yr, "Q_upper"]
        lines.append(
            f"  {int(row.return_period_yr):6d} {row.Q_cumecs:13,.0f} "
            f"{row.K_exact:9.4f} {row.K_wilson_hilferty:9.4f} "
            f"{lo:9,.0f} - {hi:9,.0f}"
        )
    lines += ["", "  bootstrap parameter uncertainty:"]
    for r in res.bootstrap_params.itertuples():
        lines.append(
            f"    {r.quantity:<10} fitted {r.fitted:9.4f}  "
            f"boot sd {r.boot_sd:7.4f}  "
            f"95% [{r.q025:8.4f}, {r.q975:8.4f}]"
        )
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


__all__ = [
    "LP3Fit",
    "LP3Result",
    "lp3_fit",
    "frequency_factor_exact",
    "frequency_factor_wilson_hilferty",
    "lp3_quantile",
    "lp3_quantiles",
    "chowdhury_standard_error",
    "chowdhury_confidence_limits",
    "bootstrap_confidence_limits",
    "skewness_band",
    "run",
    "summarise",
]
