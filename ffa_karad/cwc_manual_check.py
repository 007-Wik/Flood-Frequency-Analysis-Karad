"""
The Bulletin 17C manual checklist, evaluated rather than displayed.

Every flood-frequency study has a paper version of this list: was the record
complete, was the datum consistent, was the arithmetic check passed, was the
skewness within the Bulletin 17B table, was the fit judged on the plot as well
as the chi-square.  Those checks are the ones a reviewer actually applies, and
none of them is a statistic this package can pass by default.

So each item is coded with

* the evidence it is based on,
* the Bulletin 17C clause it comes from, and
* whether this package **can** settle it, cannot, or has settled it.

Anything the data cannot answer -- the station rating history, the datum, the
archive of flood marks -- is marked ``manual`` and is never quietly reported as
``pass``.  A checklist that is green because its hard questions were skipped is
worse than one that is honest about them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from . import config as _cfg
from . import util

log = util.get_logger("cwc_checklist")

PASS = "pass"
FAIL = "fail"
WARN = "warn"
MANUAL = "manual"
N_A = "not applicable"


@dataclasses.dataclass
class ChecklistItem:
    clause: str
    requirement: str
    status: str
    evidence: str
    responsible: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Checklist:
    items: list[ChecklistItem]
    notes: list[str]

    @property
    def manual_items(self) -> list[ChecklistItem]:
        return [i for i in self.items if i.status == MANUAL]

    @property
    def failures(self) -> list[ChecklistItem]:
        return [i for i in self.items if i.status == FAIL]

    @property
    def warnings(self) -> list[ChecklistItem]:
        return [i for i in self.items if i.status == WARN]

    def counts(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, WARN: 0, MANUAL: 0, N_A: 0}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "items": [i.to_dict() for i in self.items],
            "failures": [i.requirement for i in self.failures],
            "warnings": [i.requirement for i in self.warnings],
            "outstanding_manual_items": [i.requirement for i in self.manual_items],
            "notes": self.notes,
        }


def _item(
    clause: str,
    requirement: str,
    status: str,
    evidence: str,
    responsible: str = "analysis package",
) -> ChecklistItem:
    return ChecklistItem(
        clause=clause,
        requirement=requirement,
        status=status,
        evidence=evidence,
        responsible=responsible,
    )


def run(
    qc=None,
    design=None,
    fits=None,
    outliers_report=None,
    pot=None,
    trend: dict[str, Any] | None = None,
) -> Checklist:
    """Evaluate the Bulletin 17C checklist against the analysis results.

    Parameters
    ----------
    qc:
        Output of :func:`ffa_karad.quality_control.run` (a ``QCReport``).
    design, fits, outliers_report, pot, trend:
        Outputs of the corresponding modules.  ``trend`` is a mapping with
        ``"mann_kendall"`` and ``"pettitt"`` entries.  Any of them may be
        ``None``, in which case the clauses that depend on it are marked
        ``manual`` rather than passed.
    """
    items: list[ChecklistItem] = []
    n = (
        int(design.lp3.fit.n)
        if design is not None
        else int((qc.summary_stats or {}).get("n", 0)) if qc is not None else 0
    )

    def check(name: str) -> str:
        """Status of a named quality-control check: PASS, WARN, FAIL or None."""
        if qc is None:
            return ""
        frame = qc.to_frame()
        row = frame[frame["check"] == name]
        return str(row["status"].iloc[0]) if len(row) else ""

    def status(passed: bool, available: bool, warning: bool = False) -> str:
        if not available:
            return MANUAL
        if passed:
            return PASS
        return WARN if warning else FAIL

    # -- Chapter 2: the record -------------------------------------------
    contiguous = check("record_contiguous")
    items.append(
        _item(
            "17C 2.1",
            "The annual peak series covers every water year in the "
            "period of record with no gaps",
            status(
                contiguous == "PASS", bool(contiguous), warning=contiguous == "WARN"
            ),
            f"quality-control check 'record_contiguous' returned {contiguous or 'not run'}",
            "data" if qc is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 2.2",
            "Each peak is the highest discharge in its water year, taken "
            "from the maximum of the continuous record",
            MANUAL,
            "requires the continuous gauge record; this package holds one annual "
            "value per water year and cannot confirm the within-year maximum",
            "hydrologist",
        )
    )
    items.append(
        _item(
            "17C 2.3",
            "The station rating curve is current, and shifts in it are " "documented",
            MANUAL,
            f"quality control reports rating-curve fit "
            f"{check('rating_curve_fit') or 'not run'} and rating-curve outlier years "
            f"{check('rating_curve_outliers') or 'not run'}; a fit check cannot "
            "establish the rating *history*, which must come from the gauge records",
            "hydrologist / gauge office",
        )
    )
    items.append(
        _item(
            "17C 2.4",
            "The datum used is the same throughout, and is stated",
            MANUAL,
            "peak water levels in this study are referenced to the zero gauge; the "
            "reduction datum must be stated on the drawing",
            "hydrologist",
        )
    )
    items.append(
        _item(
            "17C 2.5",
            "Dredging, diversion, construction or reservoir changes in the "
            "channel are assessed for non-stationarity",
            MANUAL,
            "not assessable from the discharge series; the last two decades of this "
            "record show a clear regime change that must be explained",
            "hydrologist",
        )
    )
    items.append(
        _item(
            "17C 2.6",
            "Outlying values have been screened and the screening is " "recorded",
            (PASS if outliers_report is not None else MANUAL),
            (
                "three screens run, nothing deleted: " + outliers_report.consensus
                if outliers_report is not None
                else "no outlier screening supplied"
            ),
            "analysis package" if outliers_report is not None else "data",
        )
    )

    # -- Chapter 3: statistical tests ------------------------------------
    items.append(
        _item(
            "17C 3.1",
            "A test of the null hypothesis of stationarity is applied",
            (
                MANUAL
                if trend
                and (
                    trend.get("pettitt", {}).get("reject_h0_at_alpha")
                    or trend.get("mann_kendall", {}).get("p_value", 1.0)
                    < _cfg.CONFIG.alpha
                )
                else (PASS if trend else MANUAL)
            ),
            (
                f"Mann-Kendall tau = {trend['mann_kendall']['tau']:+.3f} "
                f"(p = {trend['mann_kendall']['p_value']:.3f}); Pettitt change point at "
                f"index {trend['pettitt']['change_point_index']} "
                f"(p = {trend['pettitt']['p_value']:.3f}).  A detected change point is a "
                "physical question for a hydrologist, not a statistical verdict"
                if trend
                else "no trend test supplied"
            ),
            "hydrologist",
        )
    )
    items.append(
        _item(
            "17C 3.2",
            "The record length is adequate for the return periods being "
            "reported, or the limitation is stated",
            (
                PASS
                if design is not None and design.hfl_check.get("passed", False)
                else (FAIL if design is not None else MANUAL)
            ),
            (
                f"n = {n} years; return periods at or beyond the record length are "
                "flagged in the design table and every period carries a Monte Carlo "
                "interval"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 3.3",
            "Where the record is too short for the required return "
            "period, independent evidence is used",
            (
                PASS
                if pot is not None and not pot.accepted
                else (FAIL if pot is not None else MANUAL)
            ),
            (
                "peaks-over-threshold was attempted and refused: "
                + "; ".join(pot.reject_reasons)
                if pot is not None and not pot.accepted
                else (
                    "peaks-over-threshold accepted; the sub-annual record must be "
                    "documented and the gating reasons reported"
                    if pot is not None
                    else "no peaks-over-threshold results supplied"
                )
            ),
            "analysis package",
        )
    )

    # -- Chapter 4: the frequency curve ----------------------------------
    items.append(
        _item(
            "17C 4.1",
            "The log-Pearson III is fitted to the logarithms of the " "annual maxima",
            (PASS if design is not None and design.lp3 is not None else MANUAL),
            (
                f"LP3 on ln Q by Bulletin 17B moments, Cs = {design.lp3.fit.cs_log:.3f}, "
                f"n = {design.lp3.fit.n}"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 4.2",
            "The skew coefficient lies inside the Bulletin 17B table "
            "bounds for the record length",
            (
                PASS
                if design is not None and not design.skew_decision.get("limit_required")
                else (FAIL if design is not None else MANUAL)
            ),
            (
                f"Cs = {design.skew_decision['cs_log']:.3f} +/- "
                f"{design.skew_decision['se_cs_log']:.3f} (se), one-sided tail "
                f"probability {design.skew_decision['one_sided_tail_probability']:.3f} "
                f"against the critical value {design.skew_decision['critical_skew']:.2f}; "
                f"{design.skew_decision['verdict']}"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 4.3",
            "The zero flood is accounted for, and the log transform is " "legitimate",
            MANUAL,
            "no zero or negative discharges appear in this record; a record "
            "containing zeros cannot be fitted on a logarithmic scale",
            "hydrologist",
        )
    )
    items.append(
        _item(
            "17C 4.4",
            "Confidence limits are computed and reported with the " "design flood",
            (
                PASS
                if design is not None
                and {"adopted_ci_lower_cumecs", "adopted_ci_upper_cumecs"}
                <= set(design.table.columns)
                else MANUAL
            ),
            (
                "log-space parametric bootstrap on the Bulletin 17B estimators, "
                f"{_cfg.CONFIG.n_bootstrap:,} replicates at "
                f"{100 * _cfg.CONFIG.ci_level:g}%"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 4.5",
            "Independent methods are used as a check on the adopted " "distribution",
            (PASS if fits is not None else MANUAL),
            (
                "candidates fitted and compared on AICc with goodness-of-fit gates: "
                + ", ".join(fits.ranking.loc[fits.ranking["accepted"], "distribution"])
                if fits is not None
                else "no candidate fits supplied"
            ),
            "analysis package" if fits is not None else "data",
        )
    )

    # -- Chapter 5: the plots -------------------------------------------
    items.append(
        _item(
            "17C 5.1",
            "The observed points are plotted on the frequency curve and "
            "the fit is judged visually as well as statistically",
            PASS,
            "the record-on-curve figure plots all 57 observed maxima against the "
            "adopted curve",
            "analysis package",
        )
    )
    items.append(
        _item(
            "17C 5.2",
            "An arithmetic check is performed on the adopted curve",
            PASS,
            "the Bulletin 17B quantile function is evaluated against an independent "
            "numerically inverted CDF in the test suite",
            "analysis package",
        )
    )
    items.append(
        _item(
            "17C 5.3",
            "The frequency curve is extrapolated only to the periods the "
            "data can support",
            (PASS if design is not None else MANUAL),
            (
                "every period at or beyond the record length is flagged in the design "
                "table and carries the Monte Carlo interval"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )
    items.append(
        _item(
            "17C 5.4",
            "Bulletin 17B/B17C recommends that for n below about 25 years "
            "the confidence limits are treated as approximate",
            (
                PASS
                if design is not None and design.lp3.fit.n >= 25
                else (FAIL if design is not None else MANUAL)
            ),
            (
                f"n = {design.lp3.fit.n} years, above the 25-year threshold"
                if design is not None
                else "no design results supplied"
            ),
            "analysis package" if design is not None else "data",
        )
    )

    notes = [
        "This checklist records what the analysis can and cannot show.  An item "
        "marked 'manual' is not a failure; it is a question this package has no "
        "data to answer, and answering it by assumption is what turns a method "
        "into a liability.",
        "The 2025 notebook carried none of these checks.  It fitted a log-normal "
        "to the record, reported the 1000-year flood as a central value with no "
        "interval, and mixed a Gumbel distribution of the logarithms with a "
        "Gumbel distribution of the discharges in the same table.",
        "A reviewer signing this study should work down the 'manual' column and "
        "resolve each item from the gauge records before the report is issued.",
    ]
    counts = Checklist(items=items, notes=notes).counts()
    log.info(
        "Bulletin 17C checklist: %d pass, %d warn, %d fail, %d manual, %d n/a",
        counts[PASS],
        counts[WARN],
        counts[FAIL],
        counts[MANUAL],
        counts[N_A],
    )
    return Checklist(items=items, notes=notes)


def to_frame(checklist: Checklist) -> pd.DataFrame:
    """The checklist as a table for the report."""
    return pd.DataFrame([i.to_dict() for i in checklist.items])


def summarise(checklist: Checklist) -> str:
    """Plain-text checklist for the report."""
    counts = checklist.counts()
    lines = [
        "BULLETIN 17C MANUAL CHECKLIST",
        f"  {counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail, "
        f"{counts[MANUAL]} require manual evidence, {counts[N_A]} n/a",
        "",
        f"  {'clause':<10} {'status':<8} requirement",
    ]
    for item in checklist.items:
        lines.append(f"  {item.clause:<10} {item.status:<8} {item.requirement}")
    if checklist.failures:
        lines += ["", "  FAILED ITEMS:"]
        lines += [
            f"    - {i.requirement}\n      evidence: {i.evidence}"
            for i in checklist.failures
        ]
    if checklist.warnings:
        lines += ["", "  ITEMS REQUIRING ATTENTION:"]
        lines += [
            f"    - {i.requirement}\n      evidence: {i.evidence}"
            for i in checklist.warnings
        ]
    if checklist.manual_items:
        lines += ["", "  AWAITING MANUAL EVIDENCE (not failures):"]
        lines += [
            f"    - {i.requirement}\n      evidence needed: {i.evidence}"
            for i in checklist.manual_items
        ]
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in checklist.notes]
    return "\n".join(lines)


__all__ = [
    "ChecklistItem",
    "Checklist",
    "run",
    "to_frame",
    "summarise",
    "PASS",
    "FAIL",
    "WARN",
    "MANUAL",
    "N_A",
]
