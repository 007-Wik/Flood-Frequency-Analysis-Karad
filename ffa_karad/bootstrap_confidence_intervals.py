"""
Confidence intervals for design-flood quantiles.

Why a dedicated module
----------------------
Three different uncertainties are conflated in most FFA reports, and only two
of them are quantifiable from this record:

1. **Sampling uncertainty** -- the record is 57 water years long, so even a
   perfectly known parent distribution gives a wide interval on Q100.  This is
   what a bootstrap quantifies, and it is the dominant term.
2. **Parameter / method uncertainty** -- LP3, Gumbel and LN2 disagree because
   they are different models, not because either is noisy.  A CI that ignored
   this would silently pick one model and hide the disagreement.  It is handled
   by :mod:`ffa_karad.estimation_design_flood`, which reports the spread across
   adopted candidates alongside each interval.
3. **Measurement uncertainty in the discharge itself** -- *not quantifiable
   here*.  The source column is a rating-curve derived discharge
   (:data:`ffa_karad.config.STATION.discharge_is_derived`), so the errors are
   correlated across years and heteroscedastic in stage.  No bootstrap here
   touches it; every interval is therefore an *understatement* of the total
   uncertainty and the report must say so.

Intervals produced
------------------
``nonparametric_ci``
    Case resampling of the observed record: distribution-free with respect to
    the fitted family, but at N = 57 the resample cannot invent information the
    record does not contain.
``parametric_ci``
    Records simulated from a fitted parent distribution held *fixed*, so the
    interval answers "if the model is right, how uncertain is its quantile at
    this record length".  This is the Bulletin 17C framing and, for annual
    maxima, the defensible one.
``bias_corrected_ci``
    The pivotal (basic) interval with the reverse-pivot correction of Efron &
    Tibshirani (1993, section 3.2), which removes the small upward bias of the
    percentile interval.
``achieved_coverage``
    Double bootstrap that measures the *realised* coverage of the interval
    instead of quoting the nominal 95 %.

Every replicate set is drawn from :func:`ffa_karad.util.stream_rng` keyed on
the estimator name and the sample size, so a re-run reproduces the intervals
bit-for-bit.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util

log = util.get_logger("bootstrap_ci")

#: ``rvs(draws, n) -> ndarray`` : random variates for the fitted parent.
RVS = Callable[[int, int], np.ndarray]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BootstrapCIResults:
    """All intervals for one sample and one estimator."""

    estimator_name: str
    periods: tuple[float, ...]
    level: float
    n: int
    nonparametric: pd.DataFrame
    parametric: pd.DataFrame
    bias_corrected: pd.DataFrame
    combined: pd.DataFrame
    notes: list[str]

    def interval(self, return_period: float) -> tuple[float, float, float]:
        """``(point, lower, upper)`` from the bias-corrected parametric interval."""
        row = self.combined[
            np.isclose(self.combined["return_period_yr"], return_period)
        ]
        if not len(row):
            raise KeyError(return_period)
        first = row.iloc[0]
        return (
            float(first["quantile_cumecs"]),
            float(first["ci_lower_cumecs"]),
            float(first["ci_upper_cumecs"]),
        )

    @property
    def method_bias(self) -> float:
        """Median relative bias of the simulated records against the point estimate."""
        table = self.parametric
        if not len(table):
            return float("nan")
        return float(
            np.median(
                (table["replicate_median"] - table["point"])
                / table["point"].abs().clip(lower=1e-9)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator_name,
            "level": self.level,
            "n": self.n,
            "method_relative_bias": self.method_bias,
            "nonparametric": self.nonparametric.to_dict(orient="records"),
            "parametric": self.parametric.to_dict(orient="records"),
            "bias_corrected": self.bias_corrected.to_dict(orient="records"),
            "combined": self.combined.to_dict(orient="records"),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Interval primitives
# ---------------------------------------------------------------------------


def percentile_interval(replicates: np.ndarray, level: float) -> tuple[float, float]:
    """Equal-tail percentile interval of the replicate quantiles."""
    alpha = 1.0 - level
    low, high = np.percentile(
        replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return float(low), float(high)


def bias_corrected_interval(
    replicates: np.ndarray, point: float, level: float
) -> tuple[float, float, float]:
    """Pivotal interval with the reverse-pivot bias correction.

    ``z0 = Phi^{-1}( #{q*_b < theta_hat} / B )`` shifts the pivotal interval
    ``[2*theta_hat - q_{1-a/2}, 2*theta_hat - q_{a/2}]`` onto the original scale.
    Returns ``(lower, upper, z0)``.
    """
    b = max(int(np.asarray(replicates).size), 2)
    prop = float(np.mean(replicates < point))
    prop = min(max(prop, 1.0 / b), 1.0 - 1.0 / b)
    z0 = float(sps.norm.ppf(prop))
    low_b, high_b = percentile_interval(replicates, level)
    return float(2.0 * point - (high_b - z0)), float(2.0 * point - (low_b - z0)), z0


def studentised_interval(
    point: float, replicates: np.ndarray, se_replicates: np.ndarray, level: float
) -> tuple[float, float]:
    """Percentile-t (studentised) bootstrap interval.

    ``se_replicates`` is the standard error of the quantile inside each
    replicate, so the interval allows the estimate's own precision to vary.
    Falls back to the percentile interval when the replicate standard errors are
    not usable.
    """
    se = np.asarray(se_replicates, dtype=float)
    se_point = (
        float(np.median(se[np.isfinite(se)]))
        if np.any(np.isfinite(se))
        else float("nan")
    )
    if not np.isfinite(se_point) or se_point <= 0:
        return percentile_interval(replicates, level)
    safe_se = np.where(np.isfinite(se) & (se > 0), se, se_point)
    t = (replicates - point) / safe_se
    alpha = 1.0 - level
    t_low, t_high = np.percentile(t, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(point - t_high * se_point), float(point - t_low * se_point)


def quantile_replicates(
    samples: np.ndarray, periods: Sequence[float]
) -> dict[float, np.ndarray]:
    """Empirical design quantiles of each column (row) of ``samples``."""
    rows = np.atleast_2d(np.asarray(samples, dtype=float))
    return {
        float(t): np.percentile(rows, 100.0 * (1.0 - 1.0 / float(t)), axis=1)
        for t in periods
    }


# ---------------------------------------------------------------------------
# Interval builders
# ---------------------------------------------------------------------------


def nonparametric_ci(
    values: Sequence[float] | np.ndarray,
    periods: Sequence[float],
    point_estimates: Sequence[float] | None = None,
    n_bootstrap: int | None = None,
    level: float | None = None,
    seed_name: str = "np_ci",
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Case-resampling (nonparametric) bootstrap interval on the design quantiles.

    Returns the summary table and the replicate quantiles, so that a bias
    correction can be applied to the *same* replicates.
    """
    x = util.as_float_array(values)
    n = int(x.size)
    level = _cfg.CONFIG.ci_level if level is None else float(level)
    draws = _cfg.CONFIG.n_bootstrap if n_bootstrap is None else int(n_bootstrap)
    points = (
        np.asarray(list(point_estimates), dtype=float)
        if point_estimates is not None
        else np.asarray(
            [float(sps.mstats.mquantiles(x, 1.0 - 1.0 / float(t))[0]) for t in periods],
            dtype=float,
        )
    )

    rng = util.stream_rng(f"{seed_name}::{n}::{draws}")
    samples = x[rng.integers(0, n, size=(draws, n))]
    replicates = quantile_replicates(samples, periods)

    rows = []
    for t, point in zip(periods, points):
        reps = replicates[float(t)]
        low, high = percentile_interval(reps, level)
        rows.append(
            {
                "return_period_yr": float(t),
                "point": float(point),
                "lower": low,
                "upper": high,
                "replicate_median": float(np.median(reps)),
                "replicate_mean": float(np.mean(reps)),
                "bias": float(np.median(reps) - float(point)),
            }
        )
    return pd.DataFrame(rows), replicates


def parametric_ci(
    rvs: RVS,
    n: int,
    periods: Sequence[float],
    point_estimates: Sequence[float] | None = None,
    n_bootstrap: int | None = None,
    level: float | None = None,
    seed_name: str = "param_ci",
    bounded: bool = True,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Interval from records simulated under a fitted parent held fixed.

    Parameters
    ----------
    rvs:
        Callable ``(draws, n) -> (draws, n)`` array of simulated records.
    bounded:
        Floor replicate quantiles at zero.  A three-parameter family can return
        a negative discharge at a long return period on some resamples, which is
        numerically valid and hydrologically meaningless.

    Returns the summary table and the replicate quantiles.
    """
    n = int(n)
    level = _cfg.CONFIG.ci_level if level is None else float(level)
    draws = _cfg.CONFIG.n_bootstrap if n_bootstrap is None else int(n_bootstrap)
    points = (
        np.asarray(list(point_estimates), dtype=float)
        if point_estimates is not None
        else None
    )

    samples = np.atleast_2d(np.asarray(rvs(draws, n), dtype=float))
    if samples.shape != (draws, n):
        raise ValueError(f"rvs returned shape {samples.shape}, expected {(draws, n)}")
    if bounded:
        samples = np.maximum(samples, 0.0)
    replicates = quantile_replicates(samples, periods)

    rows = []
    for i, t in enumerate(periods):
        reps = replicates[float(t)]
        point = float(points[i]) if points is not None else float(np.median(reps))
        low, high = percentile_interval(reps, level)
        rows.append(
            {
                "return_period_yr": float(t),
                "point": point,
                "lower": low,
                "upper": high,
                "replicate_median": float(np.median(reps)),
                "replicate_mean": float(np.mean(reps)),
                "bias": float(np.median(reps) - point),
            }
        )
    return pd.DataFrame(rows), replicates


def bias_corrected_table(
    replicates: Mapping[float, np.ndarray],
    points: Mapping[float, float] | None = None,
    level: float | None = None,
) -> pd.DataFrame:
    """Bias-corrected interval table built from pre-computed replicates."""
    level = _cfg.CONFIG.ci_level if level is None else float(level)
    rows = []
    for t, reps in replicates.items():
        point = (
            float(points[float(t)]) if points is not None else float(np.median(reps))
        )
        low, high, z0 = bias_corrected_interval(reps, point, level)
        rows.append(
            {
                "return_period_yr": float(t),
                "point": point,
                "lower": low,
                "upper": high,
                "z0": z0,
                "replicate_median": float(np.median(reps)),
                "bias": float(np.median(reps) - point),
            }
        )
    return pd.DataFrame(rows).sort_values("return_period_yr").reset_index(drop=True)


def achieved_coverage(
    rvs: RVS,
    n: int,
    periods: Sequence[float],
    n_inner: int = 500,
    n_outer: int = 200,
    level: float | None = None,
    seed_name: str = "coverage",
) -> pd.DataFrame:
    """Double-bootstrap estimate of the interval's *actual* coverage.

    Each of ``n_outer`` records is simulated from the fitted parent; an inner
    bootstrap of that record produces the interval; the replicate is counted as
    covered when the parent's true quantile lies inside.  A nominal 95 %
    percentile interval for a 57-year annual-maxima record does **not** achieve
    95 % at every return period, and this is where that is measured rather than
    asserted.
    """
    n = int(n)
    level = _cfg.CONFIG.ci_level if level is None else float(level)
    periods = tuple(float(t) for t in periods)
    outer = np.atleast_2d(np.asarray(rvs(int(n_outer), n), dtype=float))
    truth = {
        t: float(np.quantile(np.atleast_2d(outer), 1.0 - 1.0 / t)) for t in periods
    }

    rng = util.stream_rng(f"{seed_name}::{n}::{n_outer}::{n_inner}")
    counts = {t: 0 for t in periods}
    for record in outer:
        inner = record[rng.integers(0, n, size=(n_inner, n))]
        reps = quantile_replicates(inner, periods)
        for t in periods:
            low, high = percentile_interval(reps[t], level)
            counts[t] += int(low <= truth[t] <= high)
    return pd.DataFrame(
        [
            {
                "return_period_yr": t,
                "nominal_level": level,
                "empirical_coverage": counts[t] / float(n_outer),
                "parent_quantile": truth[t],
                "n_outer": int(n_outer),
                "n_inner": int(n_inner),
            }
            for t in periods
        ]
    )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def lognormal_rvs(loc: float, scale: float, seed_name: str) -> RVS:
    """``rvs`` callable drawing log-normal records with a deterministic seed."""

    def _rvs(draws: int, n: int) -> np.ndarray:
        rng = util.stream_rng(f"{seed_name}::{loc:.6f}::{scale:.6f}::{draws}x{n}")
        return rng.lognormal(loc, scale, size=(draws, n))

    return _rvs


def run(
    values: Sequence[float] | np.ndarray,
    periods: Sequence[float] | None = None,
    rvs: RVS | None = None,
    estimator_name: str | None = None,
    level: float | None = None,
) -> BootstrapCIResults:
    """Full interval report for the adopted distribution.

    Parameters
    ----------
    values:
        Observed annual peaks.
    periods:
        Return periods; defaults to :data:`ffa_karad.config.CONFIG.return_periods`.
    rvs:
        Sampling callable for the adopted parent distribution.  With ``None`` a
        two-parameter log-normal fitted to ``ln Q`` is used, which is enough to
        show the scale of the sampling uncertainty but is *not* the adopted
        model -- the label in the output says so.
    """
    x = util.as_float_array(values)
    n = int(x.size)
    periods = tuple(
        float(t) for t in (_cfg.CONFIG.return_periods if periods is None else periods)
    )
    level = _cfg.CONFIG.ci_level if level is None else float(level)

    np_table, np_reps = nonparametric_ci(x, periods, seed_name="np_ci")

    if rvs is None:
        log_q = np.log(x)
        loc, scale = float(log_q.mean()), float(log_q.std(ddof=1))
        rvs = lognormal_rvs(loc, scale, "param_ln2")
        estimator_name = estimator_name or (
            f"LN2 on ln Q (mean {loc:.4f}, sd {scale:.4f}) -- sampling "
            "uncertainty only, not the adopted LP3 model"
        )
        points = [
            float(np.exp(loc + scale * sps.norm.ppf(1.0 - 1.0 / t))) for t in periods
        ]
    else:
        # Without an analytic quantile the point estimate is the median of the
        # simulated records, which is the parent's quantile up to MC error.
        estimator_name = estimator_name or "fitted parent distribution"
        points = None

    par_table, par_reps = parametric_ci(
        rvs, n, periods, point_estimates=points, seed_name="param_ci"
    )
    point_map = {
        float(t): float(v)
        for t, v in zip(par_table["return_period_yr"], par_table["point"])
    }
    bc_table = bias_corrected_table(par_reps, point_map, level=level)

    combined = pd.DataFrame(
        {
            "return_period_yr": par_table["return_period_yr"],
            "quantile_cumecs": par_table["point"],
            "ci_lower_cumecs": bc_table["lower"].to_numpy(),
            "ci_upper_cumecs": bc_table["upper"].to_numpy(),
            "percentile_lower_cumecs": par_table["lower"].to_numpy(),
            "percentile_upper_cumecs": par_table["upper"].to_numpy(),
            "np_ci_lower_cumecs": np_table["lower"].to_numpy(),
            "np_ci_upper_cumecs": np_table["upper"].to_numpy(),
        }
    )
    combined["relative_width"] = (
        combined["ci_upper_cumecs"] - combined["ci_lower_cumecs"]
    ) / combined["quantile_cumecs"].abs().clip(lower=1e-9)

    notes = [
        f"Intervals are {level:.0%} equal-tail from "
        f"{_cfg.CONFIG.n_bootstrap:,} replicates; point estimates come from the "
        "adopted distribution, not from the sample.",
        "The nonparametric interval resamples the observed record; the "
        "parametric interval holds the fitted parent fixed and varies only the "
        "record.  Agreement between them indicates the record, not the model, "
        "is the limiting factor.",
        "Measurement uncertainty in the rating-curve derived discharge is NOT "
        "included.  Those errors are correlated across years and heteroscedastic "
        "in stage; no resampling scheme can represent them.  Every interval here "
        "is an understatement of the total uncertainty and the submission must "
        "state that explicitly.",
        "Parameter uncertainty across candidate distributions is reported "
        "separately in the design-flood table, which shows the LP3, Gumbel and "
        "LN2 values side by side.",
        "Q1000 extrapolates beyond a 57-year record by more than a factor of "
        "six in exceedance probability.  Its interval is reported because it is "
        "requested, not because the record supports it.",
        "The nonparametric upper limit saturates at the largest observed peak "
        "once the return period approaches the record length: a case resample "
        "cannot produce a value the record does not contain.  That saturation is "
        "a property of the data, not a defect, and it is why the parametric "
        "interval is the one quoted for design.",
    ]
    return BootstrapCIResults(
        estimator_name=estimator_name,
        periods=periods,
        level=level,
        n=n,
        nonparametric=np_table,
        parametric=par_table,
        bias_corrected=bc_table,
        combined=combined,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarise(res: BootstrapCIResults) -> str:
    """Plain-text interval table for the report."""
    lines = [
        f"DESIGN FLOOD CONFIDENCE INTERVALS  (n = {res.n}, {res.estimator_name})",
        f"  {'T (yr)':>8} {'Q (cumecs)':>12} {'lower':>11} {'upper':>11} "
        f"{'width %':>8} {'NP lower':>11} {'NP upper':>11}",
    ]
    for row in res.combined.itertuples():
        lines.append(
            f"  {row.return_period_yr:8.0f} {row.quantile_cumecs:12,.0f} "
            f"{row.ci_lower_cumecs:11,.0f} {row.ci_upper_cumecs:11,.0f} "
            f"{100.0 * row.relative_width:8.1f} {row.np_ci_lower_cumecs:11,.0f} "
            f"{row.np_ci_upper_cumecs:11,.0f}"
        )
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


__all__ = [
    "RVS",
    "BootstrapCIResults",
    "percentile_interval",
    "bias_corrected_interval",
    "studentised_interval",
    "quantile_replicates",
    "nonparametric_ci",
    "parametric_ci",
    "bias_corrected_table",
    "achieved_coverage",
    "lognormal_rvs",
    "run",
    "summarise",
]
