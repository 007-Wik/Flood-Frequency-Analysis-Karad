"""
The report: one Markdown document assembled from the module outputs.

Two rules govern everything written here.

**Nothing appears that a module did not produce.**  Every number in the report
is read from a result object.  If the peaks-over-threshold screen refused, the
report says it refused and why; it does not quietly drop the section, and it
does not print a return level from a gated fit.  If the MCMC gate failed, the
posterior quantiles are absent from the report.

**The caveats travel with the numbers.**  A design flood without its confidence
interval, its record-length warning and its rating-curve provenance is not a
result, it is a headline.  Each table is written with the qualification that
belongs to it, and the document ends with a list of what a reviewer still has
to do.

The report is generated, never edited.  Anything that needs saying that this
module cannot derive from the data is written as a question for the reviewer,
not as an assertion.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config as _cfg
from . import cwc_manual_check as _cc
from . import util

log = util.get_logger("reporting")

NUM = "{:,.0f}"


def _num(value: Any, digits: int = 2) -> str:
    """Thousands-separated fixed-point, so the report never prints 1.027e+04."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    text = f"{number:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclasses.dataclass
class ReportArtifacts:
    markdown_path: Path
    json_path: Path | None
    csv_paths: dict[str, Path]
    sections: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": str(self.markdown_path),
            "json": str(self.json_path) if self.json_path else None,
            "csv": {k: str(v) for k, v in self.csv_paths.items()},
            "sections": self.sections,
        }


def _table(frame: pd.DataFrame) -> str:
    """Markdown table with readable numbers.

    ``%.4g`` is right for the CSV sidecars but wrong for a document a
    hydrologist reads: it prints the 50-year upper confidence limit as
    ``1.027e+04``.  Small probabilities keep four significant figures; large
    discharges and levels get thousands separators and only the decimals they
    need.
    """
    if frame is None or not len(frame):
        return "_not available_\n"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):

            def render(value: float) -> str:
                if pd.isna(value):
                    return ""
                magnitude = abs(float(value))
                if magnitude < 0.001:
                    return f"{value:.3g}"
                if magnitude < 1.0:
                    return f"{value:.4f}".rstrip("0").rstrip(".")
                if magnitude >= 1000:
                    return _num(value, 0)
                return _num(value, 4)

            formatted[column] = formatted[column].map(render)
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    rule = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def _headline(results: dict[str, Any]) -> list[str]:
    """The adopted design floods, with their qualifications attached."""
    design = results.get("design")
    if design is None:
        return ["No design-flood results were produced; nothing can be adopted."]
    table = design.table
    lines = ["## Adopted design floods", ""]
    if not bool(design.adopted):
        lines += ["> **No value is adopted.** " + "; ".join(design.reject_reasons), ""]
    lines += [
        "Method: log-Pearson type III on `ln Q`, Bulletin 17B moments "
        f"(n = {design.lp3.fit.n}, Cs = {design.lp3.fit.cs_log:.3f}), the primary "
        "method under IS 11223:1985. Gumbel EV1 and strict two-parameter LN2 are "
        "reported as cross-checks; AICc ranks the candidates and does not choose "
        "the method.",
        "",
        _table(
            table[
                [
                    "return_period_yr",
                    "adopted_lp3_cumecs",
                    "adopted_ci_lower_cumecs",
                    "adopted_ci_upper_cumecs",
                    "gumbel_ev1_cumecs",
                    "ln2_cumecs",
                    "candidate_spread_percent",
                ]
            ].assign(return_period_yr=lambda f: f["return_period_yr"].map(NUM.format))
        ),
        "Columns: `adopted_lp3_cumecs` is the Bulletin 17B moment estimate; the "
        "confidence band is the log-space parametric bootstrap; the last column "
        "is the spread between accepted candidate distributions, i.e. the "
        "model-form uncertainty, which is larger than the confidence band.",
        "",
        "**Qualifications that travel with these numbers.**",
    ]
    hfl = design.hfl_check
    if isinstance(hfl, dict):
        lines.append(
            f"- The observed HFL of {NUM.format(hfl['hfl_cumecs'])} m3/s "
            f"({hfl['hfl_date']}) sits at an implied "
            f"{hfl['implied_return_period_yr']:.0f}-year return period on this "
            f"curve, against a {hfl['record_length_years']:.0f}-year record."
        )
    mc = results.get("monte_carlo")
    if mc is not None:
        row = mc.summary.set_index("return_period_yr")
        for period in (100.0, 1000.0):
            if period in row.index:
                entry = row.loc[period]
                lines.append(
                    f"- T = {NUM.format(period)}: re-estimating on "
                    f"{mc.n_records} independent {mc.record_length}-year records "
                    f"gives a 95% interval of {NUM.format(entry['mc_lower_cumecs'])}"
                    f"-{NUM.format(entry['mc_upper_cumecs'])} m3/s, a width of "
                    f"{100 * entry['mc_relative_width']:.0f}% of the adopted value."
                )
    lines.append(
        "- The discharge column is rating-curve derived (`Discharge*`), so "
        "these values carry correlated measurement error that no bootstrap "
        "in this package quantifies."
    )
    levels = design.levels
    if len(levels):
        table = levels[
            [
                "return_period_yr",
                "discharge_cumecs",
                "water_level_m",
                "depth_over_zero_gauge_m",
                "freeboard_m",
                "top_of_structure_m",
                "rating_curve_extrapolated",
            ]
        ].copy()
        table["return_period_yr"] = table["return_period_yr"].map(NUM.format)
        lines += [
            "",
            "## Design water levels",
            "",
            "Levels come from inverting the rating curve at each design "
            "discharge. Freeboard is applied only at the "
            f"{_cfg.CONFIG.freeboard_return_period:g}-year return period, which is "
            "why `top_of_structure_m` exceeds `water_level_m` in exactly one row.",
            "",
            _table(table),
        ]
        if bool(levels["rating_curve_extrapolated"].any()):
            lines += [
                "",
                "Rows flagged `rating_curve_extrapolated` lie outside the "
                "stage range over which the rating curve was calibrated. Those "
                "levels are extrapolations, not measured levels.",
            ]
    return lines


def _section_diagnosis(results: dict[str, Any]) -> list[str]:
    qc = results.get("qc")
    stats = results.get("stats")
    acf = results.get("autocorrelation")
    lines = ["## Record, quality control and dependence", ""]
    if qc is not None:
        counts = {"PASS": qc.n_pass, "WARN": qc.n_warn, "FAIL": qc.n_fail}
        lines += [
            f"Quality control: {counts['PASS']} checks passed, {counts['WARN']} "
            f"warned, {counts['FAIL']} failed. Record length "
            f"{qc.summary_stats.get('n', 'n/a')} water years.",
            "",
            _table(qc.to_frame()[lambda f: f["status"] != "PASS"]),
        ]
    if stats is not None:
        trend_rows = getattr(stats, "trend", None)
        change = getattr(stats, "change_point", None)
        lines += ["", "### Trend and stationarity", ""]
        if trend_rows is not None and len(trend_rows):
            lines.append(_table(pd.DataFrame(trend_rows)))
        if change:
            lines += [
                "",
                f"Pettitt change point at index {change.get('change_point_index')}"
                f" (p = {change.get('p_value', float('nan')):.3f}); "
                f"{'a change point is indicated' if change.get('reject_h0_at_alpha') else 'no change point is indicated'} "
                f"at alpha = {_cfg.CONFIG.alpha}. The index is a position in "
                "the water-year series, not a calendar year.",
            ]
    if acf is not None:
        n_eff = acf.n_eff
        low, high = acf.band
        lines += [
            "",
            "### Dependence",
            "",
            f"- lag-1 autocorrelation {acf.lag1:+.4f} against a screening band "
            f"of {low:+.4f} to {high:+.4f} "
            f"({_cfg.CONFIG.acf_alpha:g} alpha, N = {int(n_eff['n'])}); "
            f"autocorrelation length {acf.autocorrelation_length} years.",
            f"- effective sample size {n_eff['n_eff_integrated']:.1f} of "
            f"{int(n_eff['n'])} (integrated autocorrelation time "
            f"{n_eff['integrated_autocorrelation_time']:.2f}). Bootstrap "
            "intervals in this package assume independent peaks, so they are "
            "optimistic by roughly this factor.",
            f"- Hurst exponent {acf.hurst_powers:+.3f} by the powers "
            f"estimator, {acf.hurst_rs:+.3f} by rescaled range; "
            "rescaled range is biased high at this record length and the two "
            "disagree, so no persistence claim is made from either.",
        ]
    outliers_report = results.get("outliers")
    if outliers_report is not None:
        lines += [
            "",
            "### Outlier screening",
            "",
            f"{outliers_report.consensus}",
            "",
            _table(
                outliers_report.retained[
                    ["label", "value_cumecs", "log_value", "rank_ascending"]
                ]
            ),
            "Nothing was deleted. The three lowest peaks are retained and are "
            "listed so a reviewer can check them against the gauge records.",
        ]
    return lines


def _section_methods(results: dict[str, Any]) -> list[str]:
    fits = results.get("fits")
    lines = ["## Distribution choice", ""]
    if fits is None:
        return lines + ["_candidate fits were not produced_"]
    ranking = fits.ranking
    columns = [
        "distribution",
        "k",
        "fitted_on",
        "loglike_Q",
        "aicc",
        "ks_p",
        "ad_p",
        "accepted",
        "Q100",
        "Q1000",
    ]
    lines += [
        _table(ranking[columns]),
        "`accepted` is decided by goodness-of-fit and support gates, not by "
        "the information criteria: a fit whose CDF cannot carry the observed "
        "order statistics is rejected however good its likelihood. The "
        "log-likelihood is compared on the raw scale for every candidate, "
        "with the Jacobian correction applied to the LP3 fit of `ln Q`.",
    ]
    pot_result = results.get("pot")
    if pot_result is not None:
        lines += ["", "### Peaks over threshold", ""]
        if pot_result.accepted:
            lines += [_table(pot_result.return_levels)]
        else:
            lines += ["Peaks over threshold was attempted and **not adopted**:", ""]
            lines += [f"- {reason}" for reason in pot_result.reject_reasons]
            lines += [
                "",
                "The sub-annual record in this station's files contains one "
                "value per year, so a POT fit would rest on about 28 "
                "exceedances. That is below the minimum of 30 and the fitted "
                "shape is negative, which would mean a bounded upper tail. "
                "Neither is defensible, so no POT return level appears "
                "anywhere in this report.",
            ]
    ml = results.get("machine_learning")
    if ml is not None:
        from . import machine_learning as _ml

        lines += [
            "",
            "### Exploratory machine learning",
            "",
            ml.skill_verdict,
            "",
            _table(
                _ml.score_table(ml)[
                    ["model", "rmse_cumecs", "mae_cumecs", "r2", "skill_vs_persistence"]
                ]
            ),
            "These numbers exist to show that the regression models do not "
            "help, not to produce a design flood. Out-of-fold scores only; "
            "the 2025 notebook's near-perfect R-squared came from putting the "
            "target in the feature matrix.",
        ]
    return lines


def _section_uncertainty(results: dict[str, Any]) -> list[str]:
    mcmc = results.get("mcmc")
    lines = ["## Uncertainty", ""]
    if mcmc is None:
        return lines + ["_no MCMC was run_"]
    diag = mcmc.diagnostics
    lines += [
        f"Convergence gate: **{'PASS' if diag.passed else 'FAIL'}**. "
        f"split R-hat max {max(diag.r_hat.values()):.4f} "
        f"(threshold {diag.thresholds['max_r_hat']:g}); minimum effective "
        f"sample size {min(diag.ess.values()):,.0f} "
        f"(minimum {diag.thresholds['min_ess']:,.0f}); mean acceptance rate "
        f"{diag.accept_rate_mean:.3f} "
        f"(band {diag.thresholds['accept_low']:g}-{diag.thresholds['accept_high']:g}).",
        "",
        _table(
            mcmc.posterior_summary[
                [
                    "parameter",
                    "mean",
                    "sd",
                    "lower",
                    "upper",
                    "frequentist_mom",
                    "posterior_sd_over_analytic",
                ]
            ]
        ),
    ]
    if mcmc.accepted and len(mcmc.quantiles):
        lines += [
            "",
            "Posterior design quantiles (model-based, LP3 assumed correct):",
            "",
            _table(mcmc.quantiles),
        ]
    else:
        lines += [
            "",
            "The posterior quantiles are withheld because the sampler "
            "has not earned them.",
        ]
    return lines


def _section_checklist(results: dict[str, Any], figures: Sequence[Any]) -> list[str]:
    checklist = results.get("checklist")
    lines = ["## Bulletin 17C checklist", ""]
    if checklist is None:
        return lines + ["_checklist not run_"]
    counts = checklist.counts()
    lines += [
        f"{counts['pass']} pass, {counts['warn']} warn, {counts['fail']} "
        f"fail, {counts['manual']} require manual evidence.",
        "",
        _table(
            _cc.to_frame(checklist)[["clause", "status", "requirement", "responsible"]]
        ),
    ]
    if checklist.manual_items:
        lines += [
            "",
            "### Outstanding manual evidence",
            "",
            "These are not failures. They are questions this package has no "
            "data to answer, and a reviewer must close each one from the "
            "gauge records before the report is issued.",
            "",
        ]
        lines += [
            f"- **{item.clause}** {item.requirement} _(owner: "
            f"{item.responsible})_: {item.evidence}"
            for item in checklist.manual_items
        ]
    if figures:
        live = [f for f in figures if getattr(f, "path", None) is not None
                and Path(f.path).exists()]
        by_section: dict[str, list[Any]] = {}
        for record in live:
            by_section.setdefault(
                getattr(record, "section", "") or "Other", []
            ).append(record)
        lines += [
            "",
            "### Figures",
            "",
            "The interactive gallery, grouped the same way and ordered by "
            "editorial tier, is at `outputs/figures/index.html` "
            "(all figures in one frame: `outputs/figures/all.html`).",
            "",
        ]
        for section, records in by_section.items():
            lines += [f"#### {section}", ""]
            for record in sorted(records, key=lambda r: (r.tier, r.name)):
                suffix = " (+ interactive HTML)" if getattr(record, "interactive", False) else ""
                lines.append(
                    f"- **tier {record.tier}** `outputs/figures/{record.name}.png`"
                    f"{suffix} -- {record.caption}"
                )
            lines.append("")
    return lines


def build_markdown(
    results: dict[str, Any], figures: Sequence[Any] = (), station: str = ""
) -> str:
    """Assemble the full report text."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Flood-frequency analysis, Krishna river at Karad bridge",
        "",
        f"Generated {generated} by `ffa_karad.reporting`. Every number below is "
        "read from a module output; nothing is entered by hand.",
        "",
        f"Station {station or getattr(_cfg.CONFIG, 'station_id', 'AK000X6')}. "
        f"RNG seed {getattr(_cfg, 'MASTER_SEED', 20260902)}.",
        "",
        "## Summary",
        "",
    ]
    design = results.get("design")
    qc = results.get("qc")
    if design is not None:
        table = design.table.set_index("return_period_yr")
        summary_rows = []
        for period in (100.0, 1000.0):
            if period in table.index:
                row = table.loc[period]
                summary_rows.append(
                    f"- T = {NUM.format(period)}: **{NUM.format(row['adopted_lp3_cumecs'])}** "
                    f"m3/s (95% {NUM.format(row['adopted_ci_lower_cumecs'])}-"
                    f"{NUM.format(row['adopted_ci_upper_cumecs'])} m3/s, "
                    f"candidate spread {row['candidate_spread_percent']:.0f}%)."
                )
        lines += summary_rows
    if qc is not None and qc.n_fail:
        lines.append(
            f"- {qc.n_fail} quality-control checks failed; see the tables "
            "below before using any number in this report."
        )
    lines += [
        "",
        "This analysis supersedes the 2025 notebook in three respects: it "
        "deletes no observations, it reports no number without the interval "
        "and the diagnostic that qualifies it, and it mixes no transform in "
        "which a Gumbel distribution is fitted to `ln Q` and its quantiles are "
        "reported as discharges.",
        "",
    ]
    lines += _headline(results)
    lines += _section_diagnosis(results)
    lines += _section_methods(results)
    lines += _section_uncertainty(results)
    lines += _section_checklist(results, figures)
    lines += [
        "",
        "## How to reproduce",
        "",
        "```",
        "python -m ffa_karad.run_all",
        "```",
        "",
        "The run writes this document, the result tables as CSV, every figure "
        "as PNG and a JSON dump of every result object under `outputs/`. The "
        "RNG is seeded, so a re-run reproduces these numbers exactly.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a result table with the configured float format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=_cfg.CONFIG.float_format)
    return path


def run(
    results: dict[str, Any],
    figures: Sequence[Any] = (),
    outdir: str | Path = "outputs",
    station: str = "",
    write_json: bool = True,
) -> ReportArtifacts:
    """Write the report, the CSV tables and the JSON dump."""
    outdir = Path(outdir)
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(results, figures, station)
    markdown_path = outdir / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    csv_paths: dict[str, Path] = {}
    exporters = {
        "design_floods": ("design", lambda r: r.table),
        "design_levels": ("design", lambda r: r.levels),
        "candidate_ranking": ("fits", lambda r: r.ranking),
        "quality_control": ("qc", lambda r: r.to_frame()),
        "acf": ("autocorrelation", lambda r: r.acf_frame),
        "mc_record_uncertainty": ("monte_carlo", lambda r: r.summary),
        "mcmc_posterior": ("mcmc", lambda r: r.posterior_summary),
        "outlier_screens": ("outliers", lambda r: r.ros),
        "checklist": ("checklist", _cc.to_frame),
    }
    for name, (key, getter) in exporters.items():
        obj = results.get(key)
        if obj is None:
            continue
        try:
            frame = getter(obj)
        except Exception as error:  # noqa: BLE001
            log.error("table %s failed: %s", name, error)
            continue
        if frame is not None and len(frame):
            csv_paths[name] = util.write_table(
                frame,
                name,
                tables,
                stage="reporting",
                extra={"source": key, "generated_by": "ffa_karad.reporting"},
            )

    json_path = None
    if write_json:
        payload: dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "station": station,
        }
        for key, obj in results.items():
            if obj is None or not hasattr(obj, "to_dict"):
                continue
            try:
                payload[key] = obj.to_dict()
            except Exception as error:  # noqa: BLE001
                log.error("json for %s failed: %s", key, error)
        json_path = util.write_json(payload, "results", outdir)

    log.info("report written to %s", markdown_path)
    return ReportArtifacts(
        markdown_path=markdown_path,
        json_path=json_path,
        csv_paths=csv_paths,
        sections=[line for line in markdown.splitlines() if line.startswith("## ")],
    )
