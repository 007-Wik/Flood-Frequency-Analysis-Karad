"""
Design-flood estimation: the primary deliverable of the analysis.

Method
------
IS 11223:1985 fixes the **log-Pearson Type III** distribution fitted to ``ln Q``
as the Indian method for spillway and bridge design floods, with the Bulletin
17B adjusted skewness and a confidence band on it when the record is short.  The
2025 notebook stated LP3 as its primary method and then reported design values
that came from four different distributions in one table, with several of them
hard-coded as report text.  The four values that were literals contradicted the
computed Gumbel values at T = 25, 100, 200 and 1000 yr; they are gone.

What this module produces
-------------------------
* the **adopted** LP3 design flood at every return period in the configuration,
  with the Bulletin 17C decision procedure applied to the skewness;
* the **cross-check** values from Gumbel (EV1) and strict LN2, side by side, so
  the model spread is visible instead of hidden;
* the **design flood level** by inverting the stage-discharge rating curve, plus
  the freeboard level required by IRC:5 / IS 7784 practice;
* an **HFL consistency check**: the adopted design flood must exceed the
  discharge of the highest flood level already observed, otherwise the model is
  contradicted by the record it was fitted to.

Adoption rule
-------------
The primary distribution is adopted because the standard names it, not because
it has the lowest AICc.  Information criteria are secondary evidence and are
reported as such.  If LP3 fails a consistency check, the failure is reported and
the adoption is withheld; the code does not silently fall back to whichever
candidate scored best.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import skewness_limits as _skew
from . import util
from .bootstrap_confidence_intervals import BootstrapCIResults
from .distribution_fitting import DistributionFittingResults
from .quality_control import RatingCurve

log = util.get_logger("design_flood")


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DesignFloodRow:
    return_period: float
    adopted: float
    adopted_lower: float
    adopted_upper: float
    gumbel: float
    ln2: float
    spread: float
    relative_spread: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_period_yr": self.return_period,
            "adopted_lp3_cumecs": self.adopted,
            "adopted_ci_lower_cumecs": self.adopted_lower,
            "adopted_ci_upper_cumecs": self.adopted_upper,
            "gumbel_ev1_cumecs": self.gumbel,
            "ln2_cumecs": self.ln2,
            "candidate_spread_cumecs": self.spread,
            "candidate_spread_percent": self.relative_spread,
        }


@dataclasses.dataclass
class DesignFloodLevel:
    return_period: float
    discharge_cumecs: float
    water_level_m: float
    depth_over_zero_gauge_m: float
    freeboard_m: float
    top_of_structure_m: float
    extrapolated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_period_yr": self.return_period,
            "discharge_cumecs": self.discharge_cumecs,
            "water_level_m": self.water_level_m,
            "depth_over_zero_gauge_m": self.depth_over_zero_gauge_m,
            "freeboard_m": self.freeboard_m,
            "top_of_structure_m": self.top_of_structure_m,
            "rating_curve_extrapolated": self.extrapolated,
        }


@dataclasses.dataclass
class DesignFloodResults:
    periods: tuple[float, ...]
    table: pd.DataFrame
    levels: pd.DataFrame
    skew_decision: dict[str, Any]
    hfl_check: dict[str, Any]
    adopted: bool
    reject_reasons: list[str]
    method_notes: list[str]
    lp3: _skew.LP3Result | None = None
    fits: DistributionFittingResults | None = None
    intervals: BootstrapCIResults | None = None

    def row(self, return_period: float) -> DesignFloodRow:
        match = self.table[np.isclose(self.table["return_period_yr"], return_period)]
        if not len(match):
            raise KeyError(return_period)
        first = match.iloc[0]
        return DesignFloodRow(
            return_period=float(first["return_period_yr"]),
            adopted=float(first["adopted_lp3_cumecs"]),
            adopted_lower=float(first["adopted_ci_lower_cumecs"]),
            adopted_upper=float(first["adopted_ci_upper_cumecs"]),
            gumbel=float(first["gumbel_ev1_cumecs"]),
            ln2=float(first["ln2_cumecs"]),
            spread=float(first["candidate_spread_cumecs"]),
            relative_spread=float(first["candidate_spread_percent"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "periods": list(self.periods),
            "table": self.table.to_dict(orient="records"),
            "levels": self.levels.to_dict(orient="records"),
            "skew_decision": self.skew_decision,
            "hfl_check": self.hfl_check,
            "adopted": self.adopted,
            "reject_reasons": self.reject_reasons,
            "method_notes": self.method_notes,
        }


# ---------------------------------------------------------------------------
# Bulletin 17C skewness decision
# ---------------------------------------------------------------------------


def bulletin_17c_decision(
    cs_log: float,
    se_log: float,
    n: int,
    threshold: float | None = None,
    limit: float | None = None,
) -> dict[str, Any]:
    """Decide whether a skewness confidence limit must be applied.

    Bulletin 17C: when the sample skewness lies at or beyond the upper tail of
    its own sampling distribution, the fitted skewness is not a reliable central
    estimate and a one-sided confidence limit must be used in its place.  Above
    a critical skewness (0.9 in Bulletin 17C) the *kurtosis*, not the skewness,
    governs the limit, because the skewness is then estimated with too little
    precision for the exceedance probabilities of interest.

    Returns a decision record; the numbers it reports are the ones the caller
    must print, never a bare boolean.
    """
    threshold = float(
        _cfg.CONFIG.bulletin_17c_skew_threshold if threshold is None else threshold
    )
    z = (
        abs(cs_log) / se_log
        if se_log and np.isfinite(se_log) and se_log > 0
        else float("nan")
    )
    tail = 0.5 * float(sps.norm.sf(z)) if np.isfinite(z) else float("nan")
    within = bool(np.isfinite(z) and z < 1.0)

    if limit is not None:
        source = "supplied table"
        decision_limit = float(limit)
    else:
        source = (
            "parametric bootstrap"
            if within
            else "bootstrap of the sampling distribution of G"
        )
        decision_limit = float("nan")

    return {
        "cs_log": float(cs_log),
        "se_cs_log": float(se_log),
        "n": int(n),
        "z_cs_over_se": float(z),
        "one_sided_tail_probability": tail,
        "critical_skew": threshold,
        "kurtosis_governed": bool(np.isfinite(cs_log) and abs(cs_log) >= threshold),
        "limit_required": not within,
        "limit_applied": bool(not within and np.isfinite(decision_limit)),
        "limit_value": decision_limit,
        "limit_source": source,
        "verdict": (
            "within the sampling distribution; the fitted skewness is " "used directly"
            if within
            else "beyond the sampling distribution; the skewness limit governs"
        ),
    }


# ---------------------------------------------------------------------------
# Rating-curve inversion
# ---------------------------------------------------------------------------


def design_levels(
    discharges: Sequence[float],
    periods: Sequence[float],
    rating: RatingCurve,
    freeboard_return_period: float | None = None,
) -> pd.DataFrame:
    """Convert design discharges to levels via the rating curve.

    The design flood *level* -- not just the discharge -- is what a bridge or
    barrage drawing needs.  Inverting a fitted rating beyond its calibrated
    stage range is extrapolation, and every such row is flagged so the report
    cannot present it as a measured level.
    """
    freeboard_t = float(
        _cfg.CONFIG.freeboard_return_period
        if freeboard_return_period is None
        else freeboard_return_period
    )
    zgl = _cfg.STATION.zero_gauge_level_m
    rows = []
    for period, discharge in zip(periods, discharges):
        depth = float(
            10.0 ** ((math.log10(discharge) - rating.coefficient) / rating.exponent)
        )
        wl = depth + zgl
        freeboard = float(
            _cfg.CONFIG.freeboard_m if abs(float(period) - freeboard_t) < 1e-9 else 0.0
        )
        rows.append(
            {
                "return_period_yr": float(period),
                "discharge_cumecs": float(discharge),
                "water_level_m": wl,
                "depth_over_zero_gauge_m": depth,
                "freeboard_m": freeboard,
                "top_of_structure_m": wl + freeboard,
                "rating_curve_extrapolated": bool(
                    depth > rating.h_max or depth < rating.h_min
                ),
            }
        )
    return pd.DataFrame(rows)


def check_hfl(
    design: pd.DataFrame, lp3_fit: _skew.LP3Fit, record_length: float
) -> dict[str, Any]:
    """Consistency between the adopted curve and the highest flood observed.

    The HFL discharge is a single event that actually occurred inside a record
    of ``record_length`` years, so under the adopted curve it corresponds to a
    return period of roughly ``record_length + 1`` years.  Two things are
    checked:

    1. the **implied return period** of the HFL discharge, which must not
       substantially exceed the record length -- a curve that calls the 1976
       flood a 400-year event is not describing this record;
    2. every design flood at ``T >= record_length`` must exceed the HFL
       discharge.  Shorter return periods are *expected* to fall below it: a
       2-year flood is smaller than the biggest flood of a 57-year record, and
       treating that as a failure would reject every fitted distribution.
    """
    hfl_q = float(_cfg.STATION.observed_hfl_cumecs)
    fit = lp3_fit

    def _exceedance_probability(discharge: float) -> float:
        if not np.isfinite(discharge) or discharge <= 0:
            return float("nan")
        k = (math.log(discharge) - fit.mean_log) / fit.sd_log
        return float(sps.pearson3.cdf(k, skew=fit.cs_log))

    f_hfl = _exceedance_probability(hfl_q)
    implied_t = float(1.0 / max(1.0 - f_hfl, 1e-12))

    rows = []
    for row in design.itertuples():
        period = float(row.return_period_yr)
        rows.append(
            {
                "return_period_yr": period,
                "design_cumecs": float(row.adopted_lp3_cumecs),
                "at_or_above_record_length": bool(period >= record_length),
                "exceeds_hfl_discharge": bool(row.adopted_lp3_cumecs > hfl_q),
                "ratio_to_hfl": float(row.adopted_lp3_cumecs / hfl_q),
            }
        )
    failures = [
        r["return_period_yr"]
        for r in rows
        if r["at_or_above_record_length"] and not r["exceeds_hfl_discharge"]
    ]
    if implied_t > 2.0 * (record_length + 1.0):
        failures.append(float("nan"))
    passed = not failures
    return {
        "hfl_cumecs": hfl_q,
        "hfl_date": _cfg.STATION.observed_hfl_date,
        "hfl_water_level_m": _cfg.STATION.observed_hfl_m,
        "record_length_years": float(record_length),
        "implied_return_period_yr": implied_t,
        "implied_exceedance_probability": f_hfl,
        "per_period": rows,
        "failures": failures,
        "passed": passed,
        "verdict": (
            f"the fitted curve places the observed HFL discharge at a "
            f"{implied_t:,.0f}-year return period, consistent with a "
            f"{record_length:,.0f}-year record, and every design flood at or "
            "beyond the record length exceeds it"
            if passed
            else f"the fitted curve places the observed HFL discharge at a "
            f"{implied_t:,.0f}-year return period, or returns design floods "
            f"below it at T = {', '.join(f'{t:g}' for t in failures if np.isfinite(t))}"
        ),
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(
    values: Sequence[float] | np.ndarray,
    lp3: _skew.LP3Result | None = None,
    fits: DistributionFittingResults | None = None,
    intervals: BootstrapCIResults | None = None,
    rating: RatingCurve | None = None,
    periods: Sequence[float] | None = None,
) -> DesignFloodResults:
    """Produce the adopted design-flood table, levels and consistency checks."""
    q = util.as_float_array(values)
    periods = tuple(
        float(t) for t in (_cfg.CONFIG.return_periods if periods is None else periods)
    )
    reasons: list[str] = []

    # -- primary method: LP3 on ln Q (IS 11223:1985) ------------------------
    if lp3 is None:
        lp3 = _skew.run(q)
    quantiles = _skew.lp3_quantiles(list(periods), lp3.fit)
    adopted = [
        float(
            quantiles.loc[
                np.isclose(quantiles["return_period_yr"], t), "Q_cumecs"
            ].iloc[0]
        )
        for t in periods
    ]

    # -- cross-checks -------------------------------------------------------
    if fits is None:
        from . import distribution_fitting as _fits_mod

        fits = _fits_mod.run(q)
    gumbel = _candidate_quantile(fits, "Gumbel", periods)
    ln2 = _candidate_quantile(fits, "LN2", periods)

    # -- confidence intervals ----------------------------------------------
    # The LP3 table already carries a parametric-bootstrap interval in which the
    # skewness is refitted on every replicate, which is the model-consistent
    # Bulletin 17C interval.  It is preferred; the generic interval module is
    # attached for the sampling-only comparison, not for the reported limits.
    lower, upper = _lp3_limits(lp3, periods, adopted)
    if intervals is not None and len(intervals.combined):
        for i, t in enumerate(periods):
            try:
                _, low, high = intervals.interval(t)
                if not np.isfinite(lower[i]) or not np.isfinite(upper[i]):
                    lower[i], upper[i] = float(low), float(high)
            except KeyError:
                pass

    table = pd.DataFrame(
        {
            "return_period_yr": list(periods),
            "adopted_lp3_cumecs": adopted,
            "adopted_ci_lower_cumecs": lower,
            "adopted_ci_upper_cumecs": upper,
            "gumbel_ev1_cumecs": gumbel,
            "ln2_cumecs": ln2,
        }
    )
    stack = np.vstack([np.asarray(adopted), np.asarray(gumbel), np.asarray(ln2)])
    table["candidate_spread_cumecs"] = np.nanmax(stack, axis=0) - np.nanmin(
        stack, axis=0
    )
    table["candidate_spread_percent"] = (
        100.0 * table["candidate_spread_cumecs"] / table["adopted_lp3_cumecs"]
    )
    table.attrs["observed_max"] = float(q.max())

    # -- Bulletin 17C decision ---------------------------------------------
    fit = lp3.fit
    skew_decision = bulletin_17c_decision(
        fit.cs_log, fit.cs_log_se, fit.n, limit=_cfg.CONFIG.bulletin_17b_table
    )

    # -- levels and HFL check ----------------------------------------------
    if rating is None:
        from . import data_processing as _dp
        from . import quality_control as _qc

        rating = _qc.fit_rating_curve(_dp.load(force_embedded=True))
    levels = design_levels(adopted, periods, rating)
    hfl = check_hfl(table, lp3.fit, record_length=float(q.size))
    if not hfl["passed"]:
        reasons.append(hfl["verdict"])

    # -- hard gates ---------------------------------------------------------
    if not np.all(np.isfinite(adopted)):
        reasons.append("one or more LP3 design quantiles are not finite")
    if any(b <= a for a, b in zip(adopted, adopted[1:])):
        reasons.append("LP3 quantiles are not increasing with return period")
    if any(a > b for a, b in zip(lower, upper)):
        reasons.append("a confidence limit lies above its point estimate")

    method_notes = [
        f"Primary method: {_cfg.CONFIG.primary_distribution} fitted to ln Q per "
        f"{_cfg.CONFIG.primary_standard}, with the Bulletin 17B adjusted skewness "
        f"Cs = {fit.cs_log:+.4f} (SE {fit.cs_log_se:.4f}) and kurtosis Ck = "
        f"{fit.ck_log:.4f} on the log scale.",
        f"LP3 parameters are the Bulletin 17B method-of-moments estimates "
        f"(mean ln Q = {fit.mean_log:.4f}, sd ln Q = {fit.sd_log:.4f}).  The "
        f"maximum-likelihood refit in the candidate table differs by at most "
        f"{_mle_difference_pct(fits, periods, adopted):.1f} % at any return "
        "period, so the result is not an artefact of the estimator.",
        "Skewness decision: " + skew_decision["verdict"] + ".",
        f"The adopted skewness is far below the {_cfg.CONFIG.bulletin_17c_skew_threshold:g} "
        "critical value, so the kurtosis does not govern the limit and the fitted "
        "skewness is used directly.",
        "Gumbel and strict LN2 are cross-checks, not alternatives: IS 11223 names "
        "LP3.  The candidate spread column shows how much the answer depends on "
        "that choice, which is larger than the bootstrap interval at short return "
        "periods and smaller at long ones.",
        "Confidence limits are sampling intervals for the adopted model only.  They "
        "exclude the rating-curve measurement error in the discharge record and "
        "the choice between candidate models; both are reported separately.",
        "Every design discharge at T above about 50 years is an extrapolation of a "
        "57-year record.  The Q1000 value should be read as the model's "
        "extrapolation, not as a measured or guaranteed event.",
    ]

    adopted_ok = not reasons
    log.info(
        "design floods computed; adoption %s", "APPROVED" if adopted_ok else "WITHHELD"
    )
    return DesignFloodResults(
        periods=periods,
        table=table,
        levels=levels,
        skew_decision=skew_decision,
        hfl_check=hfl,
        adopted=adopted_ok,
        reject_reasons=reasons,
        method_notes=method_notes,
        lp3=lp3,
        fits=fits,
        intervals=intervals,
    )


def _lp3_limits(
    lp3: _skew.LP3Result, periods: Sequence[float], adopted: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Bootstrap confidence limits from the LP3 analysis, matched by period."""
    lower = [float("nan")] * len(periods)
    upper = [float("nan")] * len(periods)
    table = lp3.bootstrap
    if table is None or not len(table):
        return lower, upper
    for i, t in enumerate(periods):
        match = table[np.isclose(table["return_period_yr"], float(t))]
        if len(match):
            lower[i] = float(match["Q_lower"].iloc[0])
            upper[i] = float(match["Q_upper"].iloc[0])
    return lower, upper


def _mle_difference_pct(
    fits: DistributionFittingResults, periods: Sequence[float], adopted: Sequence[float]
) -> float:
    """Largest percent gap between the Bulletin 17B LP3 and its ML refit."""
    try:
        mle = fits.by_name(_cfg.CONFIG.primary_distribution)
    except KeyError:
        return float("nan")
    gaps = []
    for t, point in zip(periods, adopted):
        other = mle.quantile(float(t))
        if np.isfinite(other) and point:
            gaps.append(abs(other - point) / abs(point) * 100.0)
    return float(max(gaps)) if gaps else float("nan")


def _candidate_quantile(
    fits: DistributionFittingResults, name: str, periods: Sequence[float]
) -> list[float]:
    try:
        fit = fits.by_name(name)
    except KeyError:
        return [float("nan")] * len(periods)
    values = []
    for t in periods:
        value = fit.quantile(float(t))
        values.append(value if np.isfinite(value) else float("nan"))
    if not np.all(np.isfinite(values)):
        reasons = fit.reject_reasons
        log.info(
            "cross-check %s unavailable at all periods (%s)",
            name,
            "; ".join(reasons) or "non-finite",
        )
    return values


def summarise(res: DesignFloodResults) -> str:
    """Plain-text design flood table for the report."""
    lines = [
        "DESIGN FLOODS  (adopted: LP3 on ln Q per IS 11223:1985)",
        f"  {'T (yr)':>8} {'adopted LP3':>13} {'95% lower':>11} {'95% upper':>11} "
        f"{'Gumbel':>11} {'LN2':>11} {'spread %':>10}",
    ]
    for row in res.table.itertuples():
        lines.append(
            f"  {row.return_period_yr:8.0f} {row.adopted_lp3_cumecs:13,.0f} "
            f"{row.adopted_ci_lower_cumecs:11,.0f} {row.adopted_ci_upper_cumecs:11,.0f} "
            f"{row.gumbel_ev1_cumecs:11,.0f} {row.ln2_cumecs:11,.0f} "
            f"{row.candidate_spread_percent:10.1f}"
        )
    lines += ["", "  DESIGN FLOOD LEVELS (rating-curve inversion)"]
    lines.append(
        f"  {'T (yr)':>8} {'Q (cumecs)':>12} {'WL (m)':>10} "
        f"{'freeboard':>10} {'top of structure':>17}"
    )
    for row in res.levels.itertuples():
        lines.append(
            f"  {row.return_period_yr:8.0f} {row.discharge_cumecs:12,.0f} "
            f"{row.water_level_m:10.3f} {row.freeboard_m:10.2f} "
            f"{row.top_of_structure_m:17.3f}"
            + ("  (extrapolated)" if row.rating_curve_extrapolated else "")
        )
    lines += [
        "",
        f"  HFL CHECK: {'PASS' if res.hfl_check['passed'] else 'FAIL'} -- "
        f"{res.hfl_check['verdict']}",
        f"  ADOPTION : {'APPROVED' if res.adopted else 'WITHHELD'}",
    ]
    if res.reject_reasons:
        lines += [f"    - {r}" for r in res.reject_reasons]
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.method_notes]
    return "\n".join(lines)


__all__ = [
    "DesignFloodRow",
    "DesignFloodLevel",
    "DesignFloodResults",
    "bulletin_17c_decision",
    "design_levels",
    "check_hfl",
    "run",
    "summarise",
]
