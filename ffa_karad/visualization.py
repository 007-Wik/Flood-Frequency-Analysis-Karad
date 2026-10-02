"""
Report figures: probability plots, time series, dependence and design levels.

Every figure is built from the same numbers the tables use and nothing is drawn
by hand.  The module owns three things the earlier notebook got wrong:

1. **The probability axis.**  A flood-frequency curve is plotted on log-x
   probability paper with exceedance probability increasing to the right.  A
   plot on linear axes, or with the axes swapped, produces a curve that looks
   wrong for reasons that have nothing to do with the data.
2. **The observed points on the curve.**  The 57 annual maxima are plotted on
   the fitted curve, so a reviewer can see the fit rather than take it on trust.
3. **The uncertainty on the figure.**  Confidence limits are drawn where they
   exist, and absent where the diagnostic gate refused them.  A curve with no
   interval is a different statement from a curve with a narrow one, and the
   figure must not blur them.

The historical time series uses ``DataBundle.q_ordered``, never
``DataBundle.q``.  ``q`` is sorted ascending; plotted against a year index it
produces a rising line that looks like a trend and is an artefact of the sort
order.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import autocorrelation as _ac
from . import config as _cfg
from . import histogram_ecdf_violin as _hev
from . import seaborn_kde as _kde
from . import util

log = util.get_logger("visualization")

FIGSIZE = (8.0, 5.0)
DPI = 160


@dataclasses.dataclass
class FigureRecord:
    """One written figure, with enough metadata to place it in the gallery.

    ``html_path`` is ``None`` when a figure has no interactive form: matplotlib
    cannot be made interactive without shipping a converter, and inventing one
    would misrepresent a static image as a live plot.
    """

    name: str
    path: Path
    caption: str
    html_path: Path | None = None
    section: str = ""
    tier: int = 2
    aspect: str = "wide"
    engine: str = "seaborn"
    question: str = ""

    @property
    def interactive(self) -> bool:
        return self.html_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "caption": self.caption,
            "html_path": None if self.html_path is None else str(self.html_path),
            "section": self.section,
            "tier": self.tier,
            "aspect": self.aspect,
            "engine": self.engine,
            "question": self.question,
        }


def _save(fig: plt.Figure, outdir: Path, name: str, caption: str) -> FigureRecord:
    """Write one legacy single-panel figure with its registry metadata."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=_cfg.CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    meta = _cfg.figure_meta(name)
    log.info("wrote %s", path.name)
    return FigureRecord(
        name=name,
        path=path,
        caption=caption,
        section=_cfg.LEGACY_FIGURE_SECTION,
        tier=_cfg.LEGACY_FIGURE_TIER,
        aspect=meta.aspect,
        engine="seaborn",
        question="",
    )


# ---------------------------------------------------------------------------
# Probability plots
# ---------------------------------------------------------------------------


def plot_flood_frequency(
    design, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Flood-frequency curve with confidence limits and the observed record.

    ``design`` is the :class:`ffa_karad.estimation_design_flood.DesignFloodResults`
    object.  The x-axis is return period on a log scale, increasing to the
    right, and the y-axis is discharge on a log scale.
    """
    if ax is None:
        ax = plt.subplots(figsize=FIGSIZE)[1]
    table = design.table
    periods = table["return_period_yr"].to_numpy(dtype=float)
    values = table["adopted_lp3_cumecs"].to_numpy(dtype=float)

    grid = np.logspace(np.log10(1.0), np.log10(2.0 * max(periods.max(), 10.0)), 200)
    fit = design.lp3.fit
    from . import skewness_limits as _skew

    curve = np.exp(
        fit.mean_log
        + fit.sd_log
        * np.array([_skew.frequency_factor_exact(float(t), fit.cs_log) for t in grid])
    )
    ax.plot(
        grid,
        curve,
        color="#1f4e79",
        linewidth=1.8,
        label=f"LP3 on ln Q (MoM, Cs = {fit.cs_log:.3f})",
    )

    ax.fill_between(
        periods,
        table["adopted_ci_lower_cumecs"].to_numpy(dtype=float),
        table["adopted_ci_upper_cumecs"].to_numpy(dtype=float),
        color="#1f4e79",
        alpha=0.15,
        label=f"{100 * _cfg.CONFIG.ci_level:g}% confidence band "
        "(log-space param. bootstrap)",
    )
    ax.plot(
        periods,
        values,
        "o",
        color="#1f4e79",
        markersize=5,
        label="design flood (adopted)",
    )

    for column, style, label in (
        ("gumbel_ev1_cumecs", "--", "Gumbel EV1 (cross-check)"),
        ("ln2_cumecs", ":", "LN2 (cross-check)"),
    ):
        if column in table:
            ax.plot(
                periods,
                table[column].to_numpy(dtype=float),
                style,
                color="#555555",
                linewidth=1.3,
                label=label,
            )
    ax.plot(
        [],
        [],
        " ",
        label=f"candidate spread up to "
        f"{table['candidate_spread_percent'].max():.1f}% of the value",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("return period (years, log scale)")
    ax.set_ylabel("discharge (m3/s, log scale)")
    ax.set_title("Flood-frequency curve, LP3 adopted")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3, which="both")
    return ax.get_figure(), ax


def plot_record_on_curve(
    design, bundle, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """The 57 annual maxima plotted on the adopted frequency curve."""
    if ax is None:
        ax = plt.subplots(figsize=FIGSIZE)[1]
    frame = bundle.frame.sort_values("q_cumecs", kind="stable")
    ax.plot(
        frame["return_period_observed_yr"],
        frame["q_cumecs"],
        "o",
        color="#333333",
        markersize=4.5,
        alpha=0.75,
        label=f"observed annual maxima (n = {len(frame)})",
    )
    plot_flood_frequency(design, ax=ax)
    ax.set_title("Observed record against the fitted frequency curve")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize="small")
    return ax.get_figure(), ax


def plot_time_series(
    bundle, design=None, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Annual maxima in water-year order, with the HFL for scale.

    Water-year order, not ascending discharge: the sorted series rises
    monotonically and would look like a century-long trend.
    """
    if ax is None:
        ax = plt.subplots(figsize=(9.0, 4.8))[1]
    frame = bundle.frame.sort_values("wy_start_year", kind="stable")
    ax.plot(
        frame["wy_start_year"],
        frame["q_cumecs"],
        "-o",
        color="#1f4e79",
        markersize=4,
        linewidth=1.2,
        label="annual maximum (m3/s)",
    )
    hfl = design.hfl_check if design is not None else None
    if isinstance(hfl, dict) and "hfl_cumecs" in hfl:
        ax.axhline(
            float(hfl["hfl_cumecs"]),
            color="#b03a2e",
            linestyle="--",
            linewidth=1.3,
            label=f"observed HFL {float(hfl['hfl_cumecs']):,.0f} m3/s",
        )
    ax.set_xlabel("water year (starting year)")
    ax.set_ylabel("annual maximum discharge (m3/s)")
    ax.set_title("Annual maxima in water-year order")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)
    return ax.get_figure(), ax


def plot_acf(bundle, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Sample ACF with the two-standard-error band."""
    if ax is None:
        ax = plt.subplots(figsize=FIGSIZE)[1]
    max_lag = _cfg.CONFIG.acf_max_lag
    values = _ac.acf(bundle.q_ordered, max_lag)
    band = _ac.acf_band(bundle.n_records, _cfg.CONFIG.alpha)
    lags = np.arange(len(values))
    ax.axhspan(
        band[0],
        band[1],
        color="#cccccc",
        alpha=0.6,
        label=f"+/-{_cfg.CONFIG.alpha:g} standard-error band",
    )
    ax.vlines(lags, 0.0, values, color="#1f4e79", linewidth=1.4)
    ax.plot(lags, values, "o", color="#1f4e79", markersize=3.5, label="sample ACF")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("lag (water years)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Autocorrelation of annual maxima, water-year order")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)
    return ax.get_figure(), ax


def plot_hfl_and_levels(
    design, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Design water levels against the observed HFL and the freeboard."""
    if ax is None:
        ax = plt.subplots(figsize=(9.0, 4.8))[1]
    levels = design.levels
    ax.plot(
        levels["return_period_yr"],
        levels["water_level_m"],
        "-o",
        color="#1f4e79",
        markersize=5,
        linewidth=1.6,
        label="design water level (m)",
    )
    if "top_of_structure_m" in levels:
        ax.plot(
            levels["return_period_yr"],
            levels["top_of_structure_m"],
            "--",
            color="#555555",
            linewidth=1.2,
            label=f"level + {_cfg.CONFIG.freeboard_m:g} m freeboard",
        )
    hfl = design.hfl_check
    if isinstance(hfl, dict) and "hfl_water_level_m" in hfl:
        ax.axhline(
            float(hfl["hfl_water_level_m"]),
            color="#b03a2e",
            linestyle="--",
            linewidth=1.4,
            label=f"observed HFL {float(hfl['hfl_water_level_m']):.3f} m",
        )
    freeboard = _cfg.CONFIG.freeboard_return_period
    row = levels[levels["return_period_yr"] == freeboard]
    if len(row):
        ax.plot(
            [freeboard],
            [float(row["water_level_m"].iloc[0])],
            "s",
            color="#1e8449",
            markersize=7,
            label=f"{freeboard:g}-yr design level",
        )
        ax.annotate(
            f"{float(row['water_level_m'].iloc[0]):.3f} m",
            (float(freeboard), float(row["water_level_m"].iloc[0])),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize="small",
        )
    ax.set_xscale("log")
    ax.set_xlabel("return period (years, log scale)")
    ax.set_ylabel("water level (m)")
    ax.set_title("Design water levels against the observed high flood level")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25, which="both")
    return ax.get_figure(), ax


def plot_candidate_comparison(
    fits, periods: Sequence[float] | None = None, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Every accepted candidate distribution on one axis.

    The spread between the curves is the model-form uncertainty, and it is the
    reason a single adopted number is not the whole answer.
    """
    if ax is None:
        ax = plt.subplots(figsize=FIGSIZE)[1]
    periods = list(_cfg.CONFIG.return_periods if periods is None else periods)
    ranking = fits.ranking
    for row in ranking.itertuples():
        if not bool(row.accepted):
            continue
        values = [float(getattr(row, f"Q{int(t)}")) for t in periods]
        ax.plot(
            periods,
            values,
            "-o",
            markersize=4,
            linewidth=1.3,
            label=f"{row.distribution} (AICc {row.aicc:,.1f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("return period (years, log scale)")
    ax.set_ylabel("discharge (m3/s, log scale)")
    ax.set_title("Candidate distributions: model-form spread")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3, which="both")
    return ax.get_figure(), ax


def plot_pot_return_levels(
    pot, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """POT return levels, or the reason they were refused."""
    if ax is None:
        ax = plt.subplots(figsize=FIGSIZE)[1]
    if not pot.accepted:
        ax.text(
            0.5,
            0.5,
            "peaks-over-threshold not adopted\n\n"
            + "\n".join("- " + r for r in pot.reject_reasons),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="small",
            wrap=True,
        )
        ax.set_axis_off()
        ax.get_figure().set_size_inches(9.0, 4.0)
        return ax.get_figure(), ax
    table = pot.return_levels
    ax.plot(
        table["return_period_yr"],
        table["return_level_cumecs"],
        "-o",
        color="#1e8449",
        markersize=5,
        linewidth=1.6,
        label=f"POT (threshold {float(pot.threshold):,.0f} m3/s)",
    )
    if pot.shape_ci:
        low, high = pot.shape_ci
        ax.plot(
            [],
            [],
            " ",
            label=f"profile-likelihood Cs interval " f"[{low:.3f}, {high:.3f}]",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("return period (years, log scale)")
    ax.set_ylabel("discharge (m3/s, log scale)")
    ax.set_title("Peaks over threshold")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3, which="both")
    return ax.get_figure(), ax


# ---------------------------------------------------------------------------
# Render everything
# ---------------------------------------------------------------------------


def render_all(
    bundle,
    design,
    fits,
    pot,
    outdir: str | Path,
    sub_annual: pd.DataFrame | None = None,
) -> list[FigureRecord]:
    """Write every report figure to ``outdir`` and return what was written.

    A figure that fails is recorded as a skipped name rather than allowed to
    abort the run: a missing histogram must not cost the reviewer the
    flood-frequency curve.
    """
    outdir = Path(outdir)
    records: list[FigureRecord] = []
    builders: list[tuple[str, str, Any]] = [
        (
            "flood_frequency",
            "Adopted LP3 flood-frequency curve with confidence "
            "limits and the Gumbel and LN2 cross-checks",
            lambda: plot_flood_frequency(design),
        ),
        (
            "record_on_curve",
            "The 57 observed annual maxima plotted on the " "adopted frequency curve",
            lambda: plot_record_on_curve(design, bundle),
        ),
        (
            "time_series",
            "Annual maxima in water-year order against the observed HFL",
            lambda: plot_time_series(bundle, design),
        ),
        (
            "acf",
            "Autocorrelation of annual maxima with the standard-error band",
            lambda: plot_acf(bundle),
        ),
        (
            "design_levels",
            "Design water levels, observed HFL and freeboard",
            lambda: plot_hfl_and_levels(design),
        ),
        (
            "candidate_comparison",
            "Return-level curves for every accepted candidate " "distribution",
            lambda: plot_candidate_comparison(fits),
        ),
        (
            "pot_return_levels",
            "Peaks-over-threshold return levels, or the reasons "
            "the screen refused them",
            lambda: plot_pot_return_levels(pot),
        ),
        (
            "kde_log",
            "Kernel density of ln Q with the observations as a rug",
            lambda: _kde.plot_kde(
                bundle.q, label=f"KDE of ln Q (n = {bundle.n_records})"
            ),
        ),
        (
            "kde_normality",
            "Empirical density against the fitted log-normal and LP3",
            lambda: _kde.plot_normality_comparison(bundle.q),
        ),
        (
            "histogram_ecdf_violin",
            "Histogram, empirical CDF with fitted families, "
            "and the shape of the record",
            lambda: _hev.plot_all(bundle.q),
        ),
    ]
    if sub_annual is not None and len(sub_annual):
        builders.append(
            (
                "sub_annual_peaks",
                "Sub-annual flood peaks used for peaks-over-threshold",
                lambda: plot_sub_annual(sub_annual),
            )
        )
    for name, caption, builder in builders:
        try:
            figure, _ = builder()
            records.append(_save(figure, outdir, name, caption))
        except Exception as error:  # noqa: BLE001 - report, do not abort
            log.error("figure %s failed: %s", name, error)
            records.append(
                FigureRecord(
                    name=name,
                    path=outdir / f"{name}.png",
                    caption=f"{caption} [FAILED: {error}]",
                )
            )
    return records


def plot_sub_annual(
    sub_annual: pd.DataFrame, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Sub-annual peaks per water year, which is what POT needs."""
    if ax is None:
        ax = plt.subplots(figsize=(9.0, 4.8))[1]
    grouped = (
        sub_annual.groupby("water_year")["q_cumecs"].agg(["count", "max"]).sort_index()
    )
    ax.bar(
        grouped.index.astype(int),
        grouped["max"],
        width=0.8,
        color="#aed6f1",
        edgecolor="#1f4e79",
        linewidth=0.6,
        label="largest peak in the year",
    )
    ax2 = ax.twinx()
    ax2.plot(
        grouped.index.astype(int),
        grouped["count"],
        "-o",
        color="#b03a2e",
        markersize=3.5,
        linewidth=1.1,
        label="number of peaks recorded",
    )
    ax2.set_ylabel("peaks per water year", color="#b03a2e")
    ax.set_xlabel("water year")
    ax.set_ylabel("largest peak (m3/s)")
    ax.set_title("Sub-annual flood peaks")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2, labels1 + labels2, fontsize="small", loc="upper left"
    )
    ax.grid(alpha=0.25, axis="y")
    return ax.get_figure(), ax


__all__ = [
    "FigureRecord",
    "plot_flood_frequency",
    "plot_record_on_curve",
    "plot_time_series",
    "plot_acf",
    "plot_hfl_and_levels",
    "plot_candidate_comparison",
    "plot_pot_return_levels",
    "plot_sub_annual",
    "render_all",
    "FIGSIZE",
    "DPI",
]
