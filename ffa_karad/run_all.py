"""
Run the whole analysis: ``python -m ffa_karad.run_all``.

The stages run in dependency order and every stage's output is kept, so a
failure is visible rather than absorbed:

    data_processing -> quality_control -> statistical_tests -> lmoments
    -> autocorrelation -> outliers -> distribution_fitting -> skewness_limits
    -> bootstrap_confidence_intervals -> peaks_over_threshold
    -> estimation_design_flood -> uncertainty (Monte Carlo and MCMC)
    -> machine_learning -> cwc_manual_check -> visualization -> reporting

A stage that raises is recorded in ``stages`` with its traceback and the run
continues, because a missing peaks-over-threshold section is a smaller loss
than a missing design flood.  Stages whose results are *gated* -- POT, the MCMC
convergence test -- are not failures; their gates are part of the answer, and
the report says so.

Everything is seeded, so two runs of this module produce byte-identical tables
and figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import autocorrelation as _ac
from . import bootstrap_confidence_intervals as _bci
from . import config as _cfg
from . import cwc_manual_check as _cc
from . import data_processing as _dp
from . import distribution_fitting as _df
from . import documentation as _doc
from . import estimation_design_flood as _edf
from . import figure_suite as _fs
from . import lmoments as _lm
from . import machine_learning as _ml
from . import outliers as _out
from . import peaks_over_threshold as _pot
from . import quality_control as _qc
from . import reporting as _rep
from . import skewness_limits as _sk
from . import statistical_tests as _st
from . import uncertainty_monte_carlo_bayesian_mcmc as _unc
from . import util
from . import visualization as _viz

log = util.get_logger("run_all")


def _pick_mk(trend_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull the Hamed-Rao Mann-Kendall row out of the trend test table.

    ``statistical_tests.run`` returns a list of rows keyed by test name; the
    checklist wants a mapping with ``tau`` and ``p_value``.  The Hamed-Rao
    variant is preferred because the raw record is serially correlated.
    """
    for row in trend_rows:
        if "Hamed-Rao" in str(row.get("test", "")):
            detail = str(row.get("detail", ""))
            tau = None
            if "tau =" in detail:
                try:
                    tau = float(detail.split("tau =", 1)[1].split(",")[0])
                except ValueError:
                    tau = None
            return {
                "tau": tau if tau is not None else row.get("statistic"),
                "p_value": row.get("p_value"),
                "variant": "Hamed-Rao",
            }
    return {}


@dataclasses.dataclass
class StageRecord:
    name: str
    ok: bool
    seconds: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PipelineResult:
    results: dict[str, Any]
    stages: list[StageRecord]
    artifacts: _rep.ReportArtifacts | None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
        }


def run(
    outdir: str | Path = "outputs",
    n_mc_records: int | None = None,
    mcmc_draws: int | None = None,
    write_figures: bool | None = None,
) -> PipelineResult:
    """Execute every stage and write the report."""
    started = time.perf_counter()
    outdir = Path(outdir)
    results: dict[str, Any] = {}
    stages: list[StageRecord] = []

    def stage(name: str, function: Callable[[], Any], store: str | None = None) -> Any:
        t0 = time.perf_counter()
        try:
            value = function()
        except Exception as error:  # noqa: BLE001 - report and continue
            elapsed = time.perf_counter() - t0
            detail = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
            log.error("stage %s failed after %.2fs: %s", name, elapsed, detail)
            stages.append(
                StageRecord(name=name, ok=False, seconds=elapsed, detail=detail)
            )
            return None
        elapsed = time.perf_counter() - t0
        log.info("stage %-24s ok in %6.2fs", name, elapsed)
        stages.append(StageRecord(name=name, ok=True, seconds=elapsed))
        if store:
            results[store] = value
        return value

    # -- data and diagnostics -------------------------------------------
    bundle = stage("data_processing", lambda: _dp.load(), store="bundle")
    if bundle is None:
        log.error("cannot continue without the annual peak record")
        return PipelineResult(results=results, stages=stages, artifacts=None)

    stage("quality_control", lambda: _qc.run(bundle), store="qc")
    stage("statistical_tests", lambda: _st.run(bundle.q_ordered), store="stats")
    stage("lmoments", lambda: _lm.run(bundle.q), store="lmoments")
    stage("autocorrelation", lambda: _ac.run(bundle), store="autocorrelation")
    sub_annual = stage("sub_annual_peaks", _pot.load_daily_peaks, store="sub_annual")
    if sub_annual is None:
        log.info(
            "no sub-annual workbook; the POT screen will report why it "
            "cannot use the annual maxima"
        )
    outliers_report = stage(
        "outliers",
        lambda: _out.run(
            bundle.q_ordered,
            years=list(bundle.frame["wy_start_year"]),
            labels=list(bundle.frame["wy_label"]),
        ),
        store="outliers",
    )

    # -- the frequency curve ---------------------------------------------
    fits = stage("distribution_fitting", lambda: _df.run(bundle.q), store="fits")
    lp3 = stage("skewness_limits", lambda: _sk.run(bundle.q), store="lp3")
    intervals = stage(
        "bootstrap_confidence_intervals", lambda: _bci.run(bundle.q), store="intervals"
    )
    pot_result = stage(
        "peaks_over_threshold",
        lambda: _pot.run(bundle.q_ordered, sub_annual=sub_annual),
        store="pot",
    )
    design = stage(
        "estimation_design_flood",
        lambda: _edf.run(bundle.q, lp3=lp3, fits=fits, intervals=intervals),
        store="design",
    )

    # -- uncertainty ------------------------------------------------------
    stage(
        "monte_carlo_record_uncertainty",
        lambda: _unc.monte_carlo_record_uncertainty(bundle.q, n_records=n_mc_records),
        store="monte_carlo",
    )
    stage(
        "bayesian_mcmc",
        lambda: _unc.bayesian_mcmc(bundle.q_ordered, n_draws=mcmc_draws),
        store="mcmc",
    )
    stage(
        "machine_learning", lambda: _ml.run(bundle.q_ordered), store="machine_learning"
    )

    # -- review and output ------------------------------------------------
    trend = None
    if results.get("stats") is not None:
        stats = results["stats"].to_dict()
        trend = {
            "mann_kendall": _pick_mk(stats.get("trend", [])),
            "pettitt": stats.get("change_point", {}),
        }
    stage(
        "cwc_manual_check",
        lambda: _cc.run(
            results.get("qc"), design, fits, outliers_report, pot_result, trend
        ),
        store="checklist",
    )

    want_figures = _cfg.CONFIG.write_figures if write_figures is None else write_figures
    figures: list[Any] = []
    gallery: dict[str, Any] = {}
    if want_figures and design is not None and fits is not None:
        figure_dir = outdir / "figures"
        # The legacy single-panel set goes to an appendix folder: every one of
        # them is a subset of a P-figure, so publishing them at the same level
        # would show the same evidence twice.
        core = (
            stage(
                "visualization",
                lambda: _viz.render_all(
                    bundle,
                    design,
                    fits,
                    pot_result,
                    figure_dir / "appendix",
                    sub_annual,
                ),
            )
            or []
        )
        # The notebook's own P1-P14 suite, driven by the same result objects.
        suite = (
            stage(
                "figure_suite",
                lambda: _fs.render_suite(
                    bundle,
                    design,
                    fits,
                    pot_result,
                    figure_dir,
                    stats=results.get("stats"),
                    lmoments=results.get("lmoments"),
                    autocorrelation=results.get("autocorrelation"),
                    lp3=results.get("lp3"),
                    monte_carlo=results.get("monte_carlo"),
                    mcmc=results.get("mcmc"),
                    machine_learning=results.get("machine_learning"),
                ),
            )
            or []
        )
        gallery = (
            stage(
                "gallery",
                lambda: _fs.write_gallery(suite + core, figure_dir),
            )
            or {}
        )
        figures = suite + core
        results["figures"] = figures
        results["gallery"] = gallery

    report_inputs = {
        k: v
        for k, v in results.items()
        if k not in {"bundle", "sub_annual", "figures", "gallery"}
    }
    artifacts = stage(
        "reporting",
        lambda: _rep.run(
            report_inputs,
            figures,
            outdir,
            station=getattr(_cfg.CONFIG, "station_id", "AK000X6"),
        ),
    )
    if _cfg.CONFIG.build_docs:
        stage(
            "documentation",
            lambda: _doc.build_docs(
                report_inputs,
                figures,
                figures_dir=outdir / "figures",
            ),
        )
    if artifacts is not None:
        results.pop("figures", None)

    total = time.perf_counter() - started
    log.info(
        "pipeline finished in %.1fs: %d stages, %d failed",
        total,
        len(stages),
        sum(1 for s in stages if not s.ok),
    )
    return PipelineResult(results=results, stages=stages, artifacts=artifacts)


def _docs_only(outdir: Path) -> int:
    """Rebuild the GitHub Pages site from an existing run, no re-analysis.

    Useful in CI, where the pipeline has already run and only the docs are
    stale, and on a machine where a full 40,000-draw MCMC is not worth paying
    for to fix a typo in a caption.
    """
    figure_dir = outdir / "figures"
    manifest = json.loads((figure_dir / "figures.json").read_text(encoding="utf-8"))
    records = [
        _viz.FigureRecord(
            name=item["name"],
            path=Path(item["path"]),
            caption=item["caption"],
            html_path=(
                None if item.get("html_path") is None else Path(item["html_path"])
            ),
            section=item.get("section", ""),
            tier=int(item.get("tier", 2)),
            aspect=item.get("aspect", "wide"),
            engine=item.get("engine", "seaborn"),
            question=item.get("question", ""),
        )
        for item in manifest["figures"]
    ]
    results_path = outdir / "results.json"
    results = {}
    if results_path.exists():
        # The JSON dump is enough for the site's headline numbers; anything
        # absent is simply not printed.
        results = json.loads(results_path.read_text(encoding="utf-8"))
    written = _doc.build_docs(results, records, figures_dir=figure_dir)
    print(f"docs: {len(written)} pages in {_cfg.SITE_SRC_DIR}")
    for path in written.values():
        print(f"  {path}")
    print("  mkdocs build --strict   # then publish site/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="where tables, figures and the report are written",
    )
    parser.add_argument(
        "--n-mc-records",
        type=int,
        default=None,
        help="independent simulated records (default from config)",
    )
    parser.add_argument(
        "--mcmc-draws",
        type=int,
        default=None,
        help="draws per MCMC chain (default from config)",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip figure rendering"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="write PNG only, skipping the interactive HTML figures",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="also write the mkdocs GitHub Pages source into site_src/",
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="build the site from the figures already in --outdir, without "
        "re-running the analysis",
    )
    parser.add_argument("--quiet", action="store_true", help="warnings only")
    args = parser.parse_args(argv)

    logging.getLogger(util.LOGGER_NAME).setLevel(
        logging.WARNING if args.quiet else logging.INFO
    )

    if args.docs_only:
        return _docs_only(Path(args.outdir))

    if args.no_html or args.docs:
        overrides = {}
        if args.no_html:
            overrides["write_figure_html"] = False
        if args.docs:
            overrides["build_docs"] = True
        _cfg.CONFIG = dataclasses.replace(_cfg.CONFIG, **overrides)

    outcome = run(
        outdir=args.outdir,
        n_mc_records=args.n_mc_records,
        mcmc_draws=args.mcmc_draws,
        write_figures=False if args.no_figures else None,
    )
    print(
        f"\n{len(outcome.stages)} stages, "
        f"{sum(1 for s in outcome.stages if not s.ok)} failed"
    )
    for record in outcome.stages:
        if not record.ok:
            print(f"  FAILED {record.name}: {record.detail}")
    if outcome.results.get("gallery"):
        counts = outcome.results["gallery"].get("counts", {})
        print(
            f"gallery: {counts.get('figures')} figures "
            f"({counts.get('interactive')} interactive) "
            f"-> {Path(args.outdir) / 'figures' / 'index.html'}"
        )
    if outcome.artifacts:
        print(f"report: {outcome.artifacts.markdown_path}")
        print(f"tables: {len(outcome.artifacts.csv_paths)} CSV files")
    print(json.dumps([s.to_dict() for s in outcome.stages], indent=2))
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    sys.exit(main())
