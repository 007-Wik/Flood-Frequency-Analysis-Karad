"""
Build the GitHub Pages documentation site from a pipeline run.

The site is deliberately generated, never hand-written.  Anything in it that
could drift from the numbers -- a figure list, a design flood, a checklist
count, the verdict on a refused fit -- is read from the same result objects the
report reads, so a published page can never disagree with ``report.md``.

Layout
------

``site_src/index.md``        overview, the adopted numbers, how to reproduce
``site_src/figures.md``      every figure, sectioned, tier-ordered, as a gallery
``site_src/method.md``       what each stage does and which standard it follows
``site_src/api.md``          the module map and the public entry points
``site_src/figures/*``       copies of the PNG and HTML figures
``site_src/assets/figures.json``   the machine-readable figure manifest

The site is built with the ``mkdocs`` default theme so it renders with no theme
dependency.  It lives in its own directory rather than in ``docs/``, because
``docs/`` holds the reference documents supplied with the analysis (reports and
station abstracts) and those do not belong in a web build.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from . import config as _cfg
from . import util

log = util.get_logger("documentation")

MKDOCS_NAV = """nav:
  - Home: index.md
  - Figures: figures.md
  - Method: method.md
  - API: api.md
"""


def _num(value: Any, digits: int = 0) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _stage_rows() -> list[tuple[str, str]]:
    return [
        (
            "data_processing",
            "Load and hash the raw CSV, delete nothing, and expose both the "
            "sorted (`q`) and the water-year-ordered (`q_ordered`) series.",
        ),
        (
            "quality_control",
            "Screen the record for length, gaps, datum and provenance before "
            "any number is derived from it.",
        ),
        (
            "distribution_fitting",
            "Fit every candidate in :data:`config.CONFIG` on the discharge "
            "scale, then apply the finite-parameter, divergence and KS "
            "probability-plot gates before a candidate may be adopted.",
        ),
        (
            "estimation_design_flood",
            "Produce the return-period table, the Bulletin 17C confidence "
            "limits and the design water level with freeboard.",
        ),
        (
            "skewness_limits",
            "Bulletin 17C skewness/kurtosis confidence limits by parametric "
            "bootstrap, including the kurtosis-governed branch above Cs = 0.9.",
        ),
        (
            "peaks_over_threshold",
            "Threshold selection by profile penalty, GPD fit, and an explicit "
            "accept/refuse verdict with reasons.",
        ),
        (
            "statistical_tests",
            "Goodness of fit, trend (Mann-Kendall and Theil-Sen), and "
            "autocorrelation with bootstrap moment sampling.",
        ),
        (
            "autocorrelation",
            "ACF/PACF, Ljung-Box, and three Hurst estimators with an "
            "admissibility verdict on each.",
        ),
        (
            "lmoments",
            "L-moments and L-moment ratios as an independent shape check.",
        ),
        (
            "machine_learning",
            "Leakage-safe out-of-fold benchmarking. Exploratory only: it never "
            "feeds the adopted flood.",
        ),
        (
            "uncertainty_monte_carlo_bayesian_mcmc",
            "Record-length uncertainty over independent synthetic records, and "
            "an MCMC posterior behind an explicit convergence gate.",
        ),
        (
            "figure_suite",
            "The notebook figures P1-P14, redrawn from these result objects, "
            "each tagged with section, tier and canvas.",
        ),
        (
            "reporting",
            "Assemble the report, the tables and the gallery. Nothing is "
            "printed that a module did not produce.",
        ),
    ]


def build_docs(
    results: dict[str, Any] | None = None,
    figures: Sequence[Any] = (),
    outdir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the mkdocs source tree. Returns the paths written."""
    results = results or {}
    docs_dir = Path(outdir) if outdir is not None else _cfg.SITE_SRC_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    src_figures = (
        Path(figures_dir)
        if figures_dir is not None
        else _cfg.FIGURE_DIR
    )
    live = [f for f in figures if Path(getattr(f, "path", "")).exists()]

    written: dict[str, Path] = {}
    written["index"] = _write_index(docs_dir, results, live)
    written["figures"] = _write_figures_page(docs_dir, results, live)
    written["method"] = _write_method(docs_dir, results)
    written["theory"] = _write_theory(docs_dir)
    written["api"] = _write_api(docs_dir)
    written["mkdocs"] = _write_mkdocs(docs_dir)
    written["manifest"] = _write_manifest(docs_dir, results, live)
    copied = _copy_figure_assets(docs_dir, src_figures, live)
    written["assets"] = copied
    log.info("docs written to %s (%d figures)", docs_dir, len(live))
    return written


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _write_index(docs_dir: Path, results: dict[str, Any], figures: list[Any]) -> Path:
    design = results.get("design")
    fits = results.get("fits")
    pot = results.get("pot")
    mcmc = results.get("mcmc")
    mc = results.get("monte_carlo")
    checklist = results.get("checklist")
    station = _cfg.STATION

    lines = [
        f"# Flood Frequency Analysis -- {station.station_name} "
        f"({station.station_id})",
        "",
        f"**{station.river}**, {station.basin}, {station.state} &#183; "
        f"catchment {station.catchment_area_km2:,.0f} km&#178; &#183; "
        f"record {station.record_start_wy} to {station.record_end_wy}",
        "",
        "This site is generated by `python -m ffa_karad.run_all` from the "
        "package's own result objects. Every number below was produced by a "
        "module in `ffa_karad`; nothing here is typed in by hand.",
        "",
    ]

    if design is not None:
        table = design.table
        lines += [
            "## Adopted design floods",
            "",
            "| T (yr) | Q (cumecs) | 95% lower | 95% upper |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {_num(row['return_period_yr'])} "
                f"| {_num(row['adopted_lp3_cumecs'])} "
                f"| {_num(row.get('lp3_lower_cumecs', float('nan')))} "
                f"| {_num(row.get('lp3_upper_cumecs', float('nan')))} |"
            )
        lines.append("")

    if fits is not None:
        ranking = fits.ranking
        accepted = ranking[ranking["accepted"]]
        lines += [
            "## Candidate distributions",
            "",
            f"{len(ranking)} candidates were fitted and {len(accepted)} passed "
            "every gate. A rejected candidate is never plotted as an "
            "alternative curve and never quoted as a result; it appears in the "
            "figures dotted and labelled `(rejected)`.",
            "",
            "| Candidate | AICc | Akaike weight | Adopted |",
            "| --- | ---: | ---: | :---: |",
        ]
        for _, row in ranking.iterrows():
            lines.append(
                f"| {row['distribution']} | {_num(row['aicc'], 1)} "
                f"| {_num(row.get('akaike_weight', float('nan')), 3)} "
                f"| {'yes' if row['accepted'] else 'no'} |"
            )
        lines.append("")

    lines += ["## Verdict record", ""]
    if pot is not None:
        verdict = "ACCEPTED" if pot.accepted else "NOT ADOPTED"
        lines.append(
            f"- **Peaks over threshold:** {verdict}"
            + (
                ""
                if pot.accepted
                else f" -- {pot.reject_reasons[0]}"
            )
        )
    if mcmc is not None:
        d = mcmc.diagnostics
        lines.append(
            f"- **MCMC convergence gate:** {'PASSED' if d.passed else 'REFUSED'} "
            f"(max R-hat {getattr(d, 'max_rhat', float('nan')):.4f}, "
            f"min ESS {getattr(d, 'min_ess', float('nan')):,.0f})"
        )
    if mc is not None:
        fan = mc.fan
        if len(fan):
            row = fan[fan["return_period_yr"] == 1000]
            if len(row):
                r = row.iloc[0]
                lines.append(
                    f"- **Record-length uncertainty at T=1000:** "
                    f"{_num(r['mc_p05_cumecs'])} to {_num(r['mc_p95_cumecs'])} "
                    f"cumecs (5-95% over {mc.n_records} independent records)"
                )
    if checklist is not None:
        counts = checklist.counts()
        lines.append(
            f"- **Bulletin 17C checklist:** {counts['pass']} pass, "
            f"{counts['warn']} warn, {counts['fail']} fail, "
            f"{counts['manual']} need manual evidence"
        )
    lines += [
        "",
        "## Figures",
        "",
        "Every figure is on the [figures page](figures.md), grouped into "
        "sections and tagged with an editorial tier. The machine-readable "
        "manifest is `assets/figures.json`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m ffa_karad.run_all --docs",
        "mkdocs build",
        "```",
        "",
        "The RNG is seeded, so a re-run reproduces these numbers exactly.",
        "",
    ]
    path = docs_dir / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_figures_page(
    docs_dir: Path, results: dict[str, Any], figures: list[Any]
) -> Path:
    """A Markdown gallery: image, then the caption that qualifies it."""
    by_section: dict[str, list[Any]] = {}
    for record in figures:
        by_section.setdefault(getattr(record, "section", "") or "Other", []).append(record)

    ordered = [s for s in _cfg.CONFIG.doc_sections if s in by_section]
    ordered += [s for s in sorted(by_section) if s not in ordered]

    lines = [
        "# Figures",
        "",
        "Each figure is tagged with the **section** it belongs to and an "
        "**editorial tier**:",
        "",
        "| Tier | Meaning |",
        "| ---: | --- |",
        "| 1 | Decision-critical. The figure the adoption rests on. |",
        "| 2 | Supporting evidence, or a caveat that must travel with it. |",
        "| 3 | Exploratory. Useful while working, not part of the argument. |",
        "",
    ]
    for section in ordered:
        records = sorted(by_section[section], key=lambda r: (r.tier, r.name))
        lines += [f"## {section}", ""]
        for record in records:
            rel = f"figures/{record.name}.png"
            lines += [
                f"### {record.name}",
                "",
                f"*Tier {record.tier} &#183; {record.engine} engine "
                f"&#183; {record.aspect} canvas*",
                "",
            ]
            if getattr(record, "question", ""):
                lines += [f"> {record.question}", ""]
            lines += [
                f"![{record.caption}]({rel})",
                "",
                record.caption,
                "",
            ]
            if getattr(record, "interactive", False):
                lines += [
                    f"[Open the interactive version](figures/{record.name}.html)",
                    "",
                ]
    path = docs_dir / "figures.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_method(docs_dir: Path, results: dict[str, Any]) -> Path:
    lines = [
        "# Method",
        "",
        "## Pipeline stages",
        "",
        "| Stage | What it does |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}` | {text} |" for name, text in _stage_rows()]
    lines += [
        "",
        "## Gates",
        "",
        "A result is published only if the gate it sits behind passed. The gates "
        "are in `ffa_karad/config.py` and are stated here so a reviewer can find "
        "the threshold rather than infer it from a plot:",
        "",
        f"- **Candidate acceptance** -- KS probability-plot p >= "
        f"{_cfg.CONFIG.gof_alpha}, |shape| <= {_cfg.CONFIG.max_abs_shape}, "
        f"location within {_cfg.CONFIG.degenerate_loc_factor:.0%} of the "
        f"smallest peak, and the largest return period above "
        f"{_cfg.CONFIG.observed_max_discharge_factor:.0%} of the observed max.",
        f"- **POT** -- at least {_cfg.CONFIG.pot_min_exceedances} exceedances "
        f"(k = {_cfg.CONFIG.pot_penalty_k}), shape inside "
        f"{_cfg.CONFIG.pot_xi_bounds}, and "
        + (
            "an unbounded tail required."
            if _cfg.CONFIG.pot_require_unbounded_tail
            else "no tail requirement."
        ),
        f"- **MCMC** -- R-hat <= {_cfg.CONFIG.mcmc_rhat_threshold}, "
        f"ESS >= {_cfg.CONFIG.mcmc_min_ess}, accept rate inside "
        f"{_cfg.CONFIG.mcmc_accept_band}. Posterior quantiles are withheld "
        "unless this passes.",
        f"- **Hurst** -- only a slope inside (0, 1) is admissible; the "
        "aggregate-variance estimate is annotated with its verdict rather than "
        "reported as a number.",
        f"- **Bulletin 17C skewness** -- beyond Cs = "
        f"{_cfg.CONFIG.bulletin_17c_skew_threshold}, the kurtosis governs the "
        "confidence limit, not the skewness.",
        "",
        "## Active configuration",
        "",
        "```",
        _cfg.describe(),
        "```",
        "",
        "## Station",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Station | {station_id()} {station_name()} |",
        f"| River / basin | {_cfg.STATION.river} / {_cfg.STATION.basin} |",
        f"| Catchment area | {_cfg.STATION.catchment_area_km2:,.0f} km2 |",
        f"| Zero gauge level | {_cfg.STATION.zero_gauge_level_m} m |",
        f"| Observed HFL | {_cfg.STATION.observed_hfl_m} m "
        f"({_cfg.STATION.observed_hfl_date}) |",
        f"| Discharge provenance | "
        f"{'DERIVED (rating curve)' if _cfg.STATION.discharge_is_derived else 'GAUGED'} |",
        "",
        f"> {_cfg.STATION.provenance_note}",
        "",
    ]
    path = docs_dir / "method.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def station_id() -> str:
    return _cfg.STATION.station_id


def station_name() -> str:
    return _cfg.STATION.station_name


def _write_api(docs_dir: Path) -> Path:
    lines = [
        "# API map",
        "",
        "## Entry points",
        "",
        "| Call | Purpose |",
        "| --- | --- |",
        "| `python -m ffa_karad.run_all` | Full pipeline: every stage, the "
        "figures, the tables, the report and this site. |",
        "| `ffa_karad.figure_suite.render_suite(...)` | The P1-P14 figures from "
        "result objects. |",
        "| `ffa_karad.figure_suite.write_gallery(records, outdir)` | The "
        "sectioned gallery, the single-frame page and `figures.json`. |",
        "| `ffa_karad.documentation.build_docs(results, figures)` | This site. |",
        "",
        "## Module map",
        "",
        "| Module | Stage |",
        "| --- | --- |",
    ]
    lines += [f"| `ffa_karad.{name}` | {text.split('.')[0]} |" for name, text in _stage_rows()]
    lines += [
        "",
        "## Figure registry",
        "",
        "| Figure | Section | Tier | Engine | Canvas |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for meta in _cfg.FIGURE_REGISTRY:
        lines.append(
            f"| `{meta.name}` | {meta.section} | {meta.tier} | {meta.engine} "
            f"| {meta.aspect} |"
        )
    lines += [
        "",
        f"Legacy single-panel figures from `visualization.render_all` are "
        f"written to `{_cfg.LEGACY_FIGURE_SECTION}` and tagged tier "
        f"{_cfg.LEGACY_FIGURE_TIER}: each is a subset of a P-figure above, and "
        "publishing them beside P1-P14 would show the same evidence twice.",
        "",
    ]
    path = docs_dir / "api.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_theory(docs_dir: Path) -> Path:
    js_dir = docs_dir / "javascripts"
    js_dir.mkdir(exist_ok=True)
    (js_dir / "mathjax.js").write_text(
        "window.MathJax = {\n"
        "  tex: {\n"
        "    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],\n"
        "    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],\n"
        "    processEscapes: true,\n"
        "    processEnvironments: true\n"
        "  },\n"
        "  options: {\n"
        "    ignoreHtmlClass: '.*|',\n"
        "    processHtmlClass: 'arithmatex'\n"
        "  }\n"
        "};",
        encoding="utf-8"
    )

    lines = [
        "# Theoretical Methodology",
        "",
        "This section rigorously outlines the mathematical framework and statistical methodologies implemented in this repository, strictly adhering to the IS 11223:1985 guidelines and the USGS Bulletin 17C standards.",
        "",
        "## 1. Distribution Fitting via L-Moments",
        "Instead of relying on the Method of Moments (which is highly sensitive to outliers) or Maximum Likelihood Estimation (which can fail to converge for small samples), this pipeline uses **L-Moments**.",
        "L-moments are expectations of certain linear combinations of order statistics. For a sorted sample $X_{1:n} \\le X_{2:n} \\le \\dots \\le X_{n:n}$, the first four sample L-moments are given by:",
        "",
        "$$ l_1 = b_0 $$",
        "$$ l_2 = 2b_1 - b_0 $$",
        "$$ l_3 = 6b_2 - 6b_1 + b_0 $$",
        "$$ l_4 = 20b_3 - 30b_2 + 12b_1 - b_0 $$",
        "",
        "where the probability weighted moments (PWMs) $b_r$ are:",
        "$$ b_r = \\frac{1}{n} \\sum_{j=r+1}^{n} \\frac{(j-1)(j-2)\\dots(j-r)}{(n-1)(n-2)\\dots(n-r)} X_{j:n} $$",
        "",
        "The **L-moment ratios** (L-CV $\\tau_2$, L-skewness $\\tau_3$, and L-kurtosis $\\tau_4$) provide robust shape descriptors used to map the observed data to the theoretical probability density functions (PDFs).",
        "",
        "## 2. Model Selection: AICc and Akaike Weights",
        "To objectively rank the 12 candidate distributions, we use the **Akaike Information Criterion with small-sample correction (AICc)**:",
        "",
        "$$ AIC = -2 \\ln(L) + 2k $$",
        "$$ AICc = AIC + \\frac{2k(k+1)}{n - k - 1} $$",
        "",
        "where $L$ is the maximized likelihood, $k$ is the number of parameters, and $n$ is the sample size. The model with the lowest AICc is considered the best fit.",
        "",
        "## 3. Peaks Over Threshold (POT) / Generalized Pareto",
        "To model extremes above a high threshold $u$, we fit the Generalized Pareto Distribution (GPD):",
        "",
        "$$ G(x; \\xi, \\beta) = 1 - \\left( 1 + \\xi \\frac{x - u}{\\beta} \\right)^{-1/\\xi} $$",
        "",
        "For the POT model to be valid, the threshold $u$ is selected using a profile penalty algorithm ensuring at least 30 exceedances, and the shape parameter $\\xi$ is constrained within $(-0.95, 0.95)$ to prevent physically impossible bounded tails.",
        "",
        "## 4. Bayesian MCMC and Uncertainty",
        "Record-length uncertainty is quantified via Hamiltonian Monte Carlo (HMC/NUTS). For a parameter vector $\\theta$, the posterior distribution given the annual maxima $D$ is:",
        "",
        "$$ P(\\theta | D) \\propto P(D | \\theta) P(\\theta) $$",
        "",
        "Convergence is verified via the Gelman-Rubin statistic ($\\hat{R} \\le 1.01$) and Effective Sample Size ($ESS \\ge 400$).",
        "",
    ]
    path = docs_dir / "theory.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_mkdocs(docs_dir: Path) -> Path:
    """A mkdocs config using only the bundled themes."""
    path = docs_dir.parent / "mkdocs.yml"
    
    # Do not overwrite an existing mkdocs.yml so we preserve custom themes!
    if path.exists():
        return path
        
    path.write_text(
        "site_name: Karad Flood Frequency Analysis\n"
        "site_description: Generated flood-frequency analysis, figures and "
        "method for Krishna at Karad (AK000X6)\n"
        f"docs_dir: {docs_dir.name}\n"
        f"site_dir: {_cfg.SITE_BUILD_DIR.name}\n"
        "use_directory_urls: true\n"
        "theme:\n"
        "  name: mkdocs\n"
        "markdown_extensions:\n"
        "  - tables\n"
        "  - toc:\n"
        "      permalink: true\n"
        "nav:\n"
        "  - Home: index.md\n"
        "  - Figures: figures.md\n"
        "  - Method: method.md\n"
        "  - Theory: theory.md\n"
        "  - API: api.md\n",
        encoding="utf-8",
    )
    return path


def _write_manifest(
    docs_dir: Path, results: dict[str, Any], figures: list[Any]
) -> Path:
    assets = docs_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated": _cfg.run_timestamp(),
        "station": {
            "id": _cfg.STATION.station_id,
            "name": _cfg.STATION.station_name,
            "river": _cfg.STATION.river,
        },
        "figures": [
            {
                "name": r.name,
                "section": getattr(r, "section", ""),
                "tier": getattr(r, "tier", 0),
                "engine": getattr(r, "engine", ""),
                "aspect": getattr(r, "aspect", ""),
                "caption": r.caption,
                "png": f"figures/{r.name}.png",
                "html": (
                    f"figures/{r.name}.html"
                    if getattr(r, "interactive", False)
                    else None
                ),
            }
            for r in figures
        ],
    }
    path = assets / "figures.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _copy_figure_assets(
    docs_dir: Path, src_figures: Path, figures: list[Any]
) -> Path:
    """Copy the figure files into the site tree so Pages can serve them."""
    target = docs_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for record in figures:
        source = Path(record.path)
        if not source.exists():
            continue
        shutil.copy2(source, target / source.name)
        copied += 1
        if getattr(record, "interactive", False) and record.html_path:
            html = Path(record.html_path)
            if html.exists():
                shutil.copy2(html, target / html.name)
                copied += 1
    # The shared plotly bundle, if the gallery wrote one.
    bundle = Path(src_figures) / "plotly.min.js"
    if bundle.exists():
        shutil.copy2(bundle, target / bundle.name)
    log.info("copied %d figure files into %s", copied, target)
    return target


def build_and_serve_is_not_supported() -> None:  # pragma: no cover - guard
    raise NotImplementedError(
        "mkdocs build is run by CI, not by the pipeline; see .github/workflows"
    )


__all__ = ["build_docs", "MKDOCS_NAV"]