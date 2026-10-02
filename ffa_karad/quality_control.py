"""
Quality control of the annual peak series.

A flood-frequency curve is only as good as the record behind it.  This module
runs every check that can be run *before* a distribution is fitted, and returns
a structured report that downstream stages read.  A ``FAIL`` prevents adoption
of the analysis; a ``WARN`` must be carried into the limitations section of the
report.

The most important check here did not exist in the 2025 notebook: the
station **rating curve** consistency test.  Because the discharge column is
rating-curve derived (see :data:`ffa_karad.config.STATION.discharge_is_derived`),
every peak discharge must be consistent with its own water level under a single
smooth stage-discharge relation.  If the relation is loose, the discharges are
not credible regardless of how good the statistics are.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util
from .data_processing import DataBundle

log = util.get_logger("quality_control")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


# ---------------------------------------------------------------------------
# Report containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QCCheck:
    """One quality-control check and its verdict."""

    name: str
    status: str
    detail: str
    value: Any = None

    def as_row(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "status": self.status,
            "value": self.value,
            "detail": self.detail,
        }


@dataclasses.dataclass
class QCReport:
    """Aggregate quality-control outcome."""

    checks: list[QCCheck]
    rating_curve: "RatingCurve"
    summary_stats: dict[str, Any]
    notes: list[str]

    @property
    def n_pass(self) -> int:
        return sum(c.status == PASS for c in self.checks)

    @property
    def n_warn(self) -> int:
        return sum(c.status == WARN for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(c.status == FAIL for c in self.checks)

    @property
    def blocking(self) -> bool:
        """True when at least one check failed; adoption must be withheld."""
        return self.n_fail > 0

    @property
    def admissible(self) -> bool:
        """True when the record may be used to derive a design flood."""
        return self.n_fail == 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self.checks])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pass": self.n_pass,
            "n_warn": self.n_warn,
            "n_fail": self.n_fail,
            "blocking": self.blocking,
            "admissible": self.admissible,
            "checks": [c.as_row() for c in self.checks],
            "rating_curve": self.rating_curve.as_dict(),
            "summary_stats": self.summary_stats,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Rating curve
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RatingCurve:
    """Stage-discharge relation ``Q = 10**a * h**b`` with ``h = WL - ZGL``.

    A power-law rating curve is the simplest defensible form for annual peaks at
    a single station.  The reciprocal form ``h = 10**((log10 Q - a)/b)`` is used
    later to convert a design discharge into a design flood level, and is
    extrapolated only with an explicit warning flag.
    """

    coefficient: float  # a  (log10 units)
    exponent: float  # b
    r_squared: float
    rmse_log10: float
    slope_stderr: float
    intercept_stderr: float
    n: int
    h_min: float
    h_max: float
    #: Water years whose |studentised log-residual| exceeds 2.
    outliers: tuple[str, ...] = ()

    def discharge(self, wl_m: float | np.ndarray) -> np.ndarray:
        """Stage (m) to discharge (cumecs)."""
        h = np.asarray(wl_m, dtype=float) - _cfg.STATION.zero_gauge_level_m
        if np.any(h <= 0):
            raise ValueError(
                "water level at or below the zero gauge is not on the rating"
            )
        return 10.0 ** (self.coefficient + self.exponent * np.log10(h))

    def level(
        self, discharge_cumecs: float | np.ndarray, extrapolate: bool = False
    ) -> np.ndarray:
        """Discharge (cumecs) to stage (m) above MSL.

        Parameters
        ----------
        extrapolate:
            When ``False`` (default) a discharge beyond the calibrated range
            raises, because the power law is not reliable off the observed
            envelope.  Inverting the 1000-year discharge from a 57-year record
            *is* extrapolation, so the caller must pass ``True`` and the
            resulting value must be flagged in the report.
        """
        q = np.asarray(discharge_cumecs, dtype=float)
        h = 10.0 ** ((np.log10(q) - self.coefficient) / self.exponent)
        wl = h + _cfg.STATION.zero_gauge_level_m
        if not extrapolate:
            out_of_range = (h < self.h_min) | (h > self.h_max)
            if np.any(out_of_range):
                raise ValueError(
                    "inverting the rating curve outside the calibrated range "
                    f"[{self.h_min:.2f}, {self.h_max:.2f}] m requires extrapolate=True"
                )
        return wl

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": "Q = 10**a * h**b , h = WL - ZGL",
            "coefficient_a": self.coefficient,
            "exponent_b": self.exponent,
            "r_squared_log_log": self.r_squared,
            "rmse_log10": self.rmse_log10,
            "slope_stderr": self.slope_stderr,
            "intercept_stderr": self.intercept_stderr,
            "n": self.n,
            "h_min_m": self.h_min,
            "h_max_m": self.h_max,
            "outlier_water_years": list(self.outliers),
        }


def fit_rating_curve(bundle: DataBundle) -> RatingCurve:
    """Fit ``log10 Q = a + b log10 h`` by ordinary least squares."""
    frame = bundle.frame
    h = frame["depth_over_zero_gauge_m"].to_numpy(dtype=float)
    q = frame["q_cumecs"].to_numpy(dtype=float)
    if np.any(h <= 0):
        raise ValueError("one or more water levels lie at or below the zero gauge")
    if np.any(q <= 0):
        raise ValueError("one or more discharges are non-positive")

    x, y = np.log10(h), np.log10(q)
    n = x.size
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0:
        raise ValueError("all water levels identical; cannot fit a rating curve")
    sxy = float(((x - xbar) * (y - ybar)).sum())
    b = sxy / sxx
    a = float(ybar - b * xbar)
    resid = y - (a + b * x)
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    dof = n - 2
    sigma2 = ss_res / dof
    slope_se = float(np.sqrt(sigma2 / sxx))
    intercept_se = float(np.sqrt(sigma2 * (1.0 / n + xbar**2 / sxx)))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Studentised residuals to flag individual years that do not obey the curve.
    leverage = 1.0 / n + (x - xbar) ** 2 / sxx
    studentised = resid / (np.sqrt(sigma2) * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))
    flagged = tuple(frame.loc[np.abs(studentised) > 2.0, "wy_label"].tolist())

    return RatingCurve(
        coefficient=a,
        exponent=float(b),
        r_squared=float(r_squared),
        rmse_log10=float(np.sqrt(ss_res / n)),
        slope_stderr=slope_se,
        intercept_stderr=intercept_se,
        n=int(n),
        h_min=float(h.min()),
        h_max=float(h.max()),
        outliers=flagged,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check(
    name: str, ok: bool, detail: str, value: Any = None, warn_only: bool = False
) -> QCCheck:
    if ok:
        return QCCheck(name, PASS, detail, value)
    return QCCheck(name, WARN if warn_only else FAIL, detail, value)


def _monotonic_record(frame: pd.DataFrame) -> tuple[bool, list[int]]:
    years = frame["wy_start_year"].to_numpy()
    gaps = sorted({int(g) for g in np.unique(np.diff(years)) if g != 1})
    return (not gaps), gaps


def run_checks(bundle: DataBundle, expected_n: int = 57) -> QCReport:
    """Run the full quality-control battery on a loaded bundle."""
    frame = bundle.frame
    notes: list[str] = []
    checks: list[QCCheck] = []
    q = frame["q_cumecs"].to_numpy(dtype=float)
    wl = frame["wl_m"].to_numpy(dtype=float)

    # -- completeness ------------------------------------------------------
    checks.append(
        _check(
            "record_length",
            len(frame) == expected_n,
            f"{len(frame)} annual peaks available; the study record requires {expected_n}",
            len(frame),
        )
    )
    checks.append(
        _check(
            "discharge_numeric",
            bool(np.all(np.isfinite(q))),
            "all annual peak discharges parsed as finite numbers",
            int(np.isfinite(q).sum()),
        )
    )
    unparsed_dates = int(frame["peak_date"].isna().sum())
    checks.append(
        _check(
            "dates_parsed",
            unparsed_dates == 0,
            f"{unparsed_dates} peak dates could not be parsed"
            + ("" if unparsed_dates == 0 else " (dd-mm-yyyy expected)"),
            unparsed_dates,
            warn_only=True,
        )
    )
    contiguous, gaps = _monotonic_record(frame)
    checks.append(
        _check(
            "record_contiguous",
            contiguous,
            (
                "water years form a contiguous run"
                if contiguous
                else f"gaps in the water-year sequence of size(s) {gaps} yr"
            ),
            gaps,
        )
    )
    checks.append(
        _check(
            "water_year_labels_unique",
            not frame["wy_label"].duplicated().any(),
            "every water-year label is unique and taken verbatim from the source",
            int(frame["wy_label"].nunique()),
        )
    )

    # -- physical plausibility --------------------------------------------
    checks.append(
        _check(
            "discharge_positive",
            bool(np.all(q > 0)),
            "all discharges strictly positive (required for the log-normal family)",
            float(q.min()),
        )
    )
    above_datum = bool(np.all(wl > _cfg.STATION.zero_gauge_level_m))
    checks.append(
        _check(
            "water_level_above_datum",
            above_datum,
            f"all water levels above the zero gauge ({_cfg.STATION.zero_gauge_level_m} m)",
            float((wl - _cfg.STATION.zero_gauge_level_m).min()),
        )
    )
    cv = float(q.std(ddof=1) / q.mean())
    checks.append(
        _check(
            "dispersion_plausible",
            0.4 <= cv <= 2.5,
            f"coefficient of variation {cv:.3f} lies in the 0.4-2.5 band expected for "
            "annual maxima of a large Indian river",
            cv,
            warn_only=True,
        )
    )

    # -- stage-discharge consistency --------------------------------------
    rho = float(sps.spearmanr(q, wl).statistic)
    checks.append(
        _check(
            "stage_discharge_rank_monotone",
            rho > 0.9,
            f"Spearman rank correlation between discharge and water level is {rho:.4f}",
            rho,
        )
    )
    rating = fit_rating_curve(bundle)
    checks.append(
        _check(
            "rating_curve_fit",
            rating.r_squared > 0.95,
            f"log-log rating curve Q = 10**{rating.coefficient:.4f} * h**{rating.exponent:.4f} "
            f"explains {100 * rating.r_squared:.3f}% of the log-variance "
            f"(RMSE {rating.rmse_log10:.4f} log10 units)",
            rating.r_squared,
        )
    )
    checks.append(
        _check(
            "rating_curve_shape",
            1.5 <= rating.exponent <= 3.5,
            f"rating exponent b = {rating.exponent:.3f}; natural-channel values are "
            "usually 1.8-2.8, values outside 1.5-3.5 indicate an unstable datum or "
            "mixed control",
            rating.exponent,
            warn_only=True,
        )
    )
    checks.append(
        _check(
            "rating_curve_outliers",
            not rating.outliers,
            (
                "every year follows the single stage-discharge relation within 2 sigma"
                if not rating.outliers
                else f"{len(rating.outliers)} water year(s) deviate from the rating curve: "
                f"{', '.join(rating.outliers)}"
            ),
            list(rating.outliers),
            warn_only=True,
        )
    )

    # -- internal consistency of the extremes -----------------------------
    i_q = int(np.argmax(q))
    i_wl = int(np.argmax(wl))
    checks.append(
        _check(
            "maximum_consistent",
            i_q == i_wl,
            (
                f"the maximum-discharge year ({frame.loc[i_q, 'wy_label']}) is also the "
                f"maximum-water-level year ({frame.loc[i_wl, 'wy_label']})"
                if i_q == i_wl
                else f"maximum discharge occurs in {frame.loc[i_q, 'wy_label']} but maximum "
                f"water level in {frame.loc[i_wl, 'wy_label']}; the two records disagree"
            ),
            (frame.loc[i_q, "wy_label"], frame.loc[i_wl, "wy_label"]),
        )
    )
    hfl_match = (
        abs(float(q[i_wl]) - _cfg.STATION.observed_hfl_cumecs) <= 1.0
        and abs(float(wl[i_wl]) - _cfg.STATION.observed_hfl_m) <= 1e-3
    )
    checks.append(
        _check(
            "hfl_matches_station_abstract",
            hfl_match,
            (
                f"observed HFL {_cfg.STATION.observed_hfl_m} m on "
                f"{_cfg.STATION.observed_hfl_date} ({_cfg.STATION.observed_hfl_cumecs} cumecs) "
                "is reproduced by the data series"
                if hfl_match
                else "the HFL quoted in the station abstract is not reproduced by the "
                "supplied series; the two must be reconciled before design adoption"
            ),
            (_cfg.STATION.observed_hfl_cumecs, float(q[i_wl])),
        )
    )

    # -- provenance --------------------------------------------------------
    if _cfg.STATION.discharge_is_derived:
        checks.append(
            QCCheck(
                "discharge_provenance",
                WARN,
                _cfg.STATION.provenance_note,
                "rating-curve derived",
            )
        )
        notes.append(
            "Discharge values are rating-curve derived, so their errors are "
            "correlated across years and heteroscedastic. None of the bootstrap "
            "intervals in this package represent measurement error; a systematic "
            "rating revision shifts the whole frequency curve. This should be stated "
            "as a limitation and the rating curve can be re-derived from recent "
            "gaugings."
        )

    # -- seasonality -------------------------------------------------------
    non_monsoon = frame.loc[~frame["is_monsoon"], ["wy_label", "peak_date", "season"]]
    share = float((~frame["is_monsoon"]).mean())
    checks.append(
        _check(
            "seasonality_reasonable",
            share <= 0.20,
            f"{100 * share:.1f}% of annual peaks occur outside the June-September "
            "monsoon window"
            + (
                ""
                if non_monsoon.empty
                else f"; review {', '.join(non_monsoon['wy_label'])}"
            ),
            list(non_monsoon["wy_label"]),
            warn_only=True,
        )
    )
    if not non_monsoon.empty:
        notes.append(
            "Peaks outside the southwest-monsoon window ("
            + ", ".join(f"{r.wy_label} ({r.season})" for r in non_monsoon.itertuples())
            + ") indicate either genuine non-monsoon flood mechanisms or a date "
            "transcription error. Check before design adoption; a spurious peak inflates "
            "the upper tail of every fitted distribution."
        )

    # -- year-on-year plausibility ----------------------------------------
    ratio = np.abs(np.diff(q, prepend=q[0])) / q
    spikes = frame.loc[ratio > 0.75, "wy_label"].tolist()
    checks.append(
        _check(
            "no_extreme_year_on_year_jumps",
            not spikes,
            (
                "no year-on-year change in the peak exceeds 75% of the value"
                if not spikes
                else f"large year-on-year changes at {', '.join(spikes)}; a change-point or "
                "outlier test is required (see ffa_karad.outliers)"
            ),
            spikes,
            warn_only=True,
        )
    )

    stats = bundle.summary()
    report = QCReport(
        checks=checks, rating_curve=rating, summary_stats=stats, notes=notes
    )
    log.info(
        "QC complete: %d pass, %d warn, %d fail (blocking=%s)",
        report.n_pass,
        report.n_warn,
        report.n_fail,
        report.blocking,
    )
    for check in checks:
        if check.status != PASS:
            log.warning("%-34s %s: %s", check.name, check.status, check.detail)
    return report


def run(bundle: DataBundle, expected_n: int = 57) -> QCReport:
    """Quality-control entry point used by the pipeline orchestrator."""
    return run_checks(bundle, expected_n=expected_n)


def summarise(report: QCReport) -> str:
    """Plain-text QC summary for the front of the technical report."""
    lines = [
        "QUALITY CONTROL SUMMARY",
        f"  station            : {_cfg.STATION.station_id} {_cfg.STATION.station_name} "
        f"on {_cfg.STATION.river}",
        f"  record             : {report.summary_stats['wy_start']}-"
        f"{report.summary_stats['wy_end']} "
        f"(N = {report.summary_stats['n']})",
        f"  peaks              : {report.summary_stats['min_cumecs']:.0f} - "
        f"{report.summary_stats['max_cumecs']:.0f} cumecs, "
        f"mean {report.summary_stats['mean_cumecs']:.0f}, "
        f"CV {report.summary_stats['cv_percent']:.1f}%",
        f"  rating curve       : Q = 10**{report.rating_curve.coefficient:.4f} "
        f"* h**{report.rating_curve.exponent:.4f}  "
        f"(R2 = {report.rating_curve.r_squared:.4f})",
        f"  checks             : {report.n_pass} pass / {report.n_warn} warn / "
        f"{report.n_fail} fail",
        f"  admissible for FFA : {'YES' if report.admissible else 'NO'}",
    ]
    for note in report.notes:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "QCCheck",
    "QCReport",
    "RatingCurve",
    "fit_rating_curve",
    "run_checks",
    "run",
    "summarise",
]
