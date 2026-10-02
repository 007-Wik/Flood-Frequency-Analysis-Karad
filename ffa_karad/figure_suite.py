"""
Notebook-faithful figure suite: the fourteen figures P1-P14 of the 2025 notebook.

The notebook these replace drew them from notebook globals that no longer exist
here, and several of them drew things the data does not support.  This module
reproduces each figure from the package's own result objects, keeping the
notebook's layout, colours, panel lettering and titles, and correcting what was
wrong:

============================================  ======================================
Notebook cell                                  What this module does differently
============================================  ======================================
P1 bars coloured by ``np.percentile(Q, ...)``  percentiles are taken on the
                                              *water-year-ordered* series, and the
                                              trend line comes from
                                              ``statistical_tests.theil_sen`` rather
                                              than a slope printed earlier in a cell
P2 QQ panel fit Gumbel/LN2/P3 by hand           uses the package fits, so a rejected
                                              candidate cannot appear
P3 band from a notebook ``bootstrap_quantile_ci``  band is the Bulletin 17C LP3
                                              parametric bootstrap in
                                              ``skewness_limits``
P4 features from ``X_ml`` built elsewhere      features rebuilt through
                                              ``machine_learning.build_features``, so
                                              the leakage screen covers them
P5 ``MC_RESULTS`` from extra simulated years   drawn from the Monte Carlo *over
                                              independent records*, which is what
                                              record-length uncertainty is
P6 posterior quantiles from an unchecked chain withheld unless
                                              ``bayesian_mcmc`` passed its gate
P8 ``BOOTSTRAP_MOMENTS`` never defined         computed by
                                              ``statistical_tests.bootstrap_moment_samples``
P9 Hurst H reported as a number                annotated as inadmissible, because
                                              ``-0.018`` is outside (0, 1) and the
                                              record cannot resolve H = 0.5 at all
P11 POT drawn as though it passed              drawn, but stamped with the refusal
                                              and its reasons
P12 "train R2" and a GPR band                  leakage-safe out-of-fold R2; the GPR
                                              band becomes the fold spread of the
                                              out-of-fold predictions
P13 "12 distributions"                        says how many candidates were
                                              actually fitted
P14 GPR panel                                 GPD/Monte Carlo quantile band
============================================  ======================================

Every figure is written to ``outputs/figures/`` and returned as a
:class:`ffa_karad.visualization.FigureRecord` so the report lists it.  A figure
that cannot be built is recorded as failed and the rest are still written.
"""

from __future__ import annotations

import dataclasses
import html as _html
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import stats as sps

from . import config as _cfg
from . import machine_learning as _ml
from . import peaks_over_threshold as _pot
from . import plot_style as _ps
from . import skewness_limits as _skew
from . import statistical_tests as _st
from . import util
from .visualization import FigureRecord

log = util.get_logger("figure_suite")

pio.templates.default = "plotly_white"


def _esc(text: str) -> str:
    """HTML-escape text going into a generated page."""
    return _html.escape(str(text), quote=True)


#: Shared styling for the static figure pages and the gallery.  Deliberately
#: small and dependency-free: the site has to render from a plain file:// open
#: as well as from GitHub Pages.
_PAGE_CSS = """
:root{--ink:#212121;--mute:#546E7A;--line:#E0E0E0;--bg:#FAFAFC;
--t1:#B71C1C;--t2:#E65100;--t3:#37474F;--accent:#1565C0}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg)}
main{max-width:1400px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:20px;margin:.2em 0}
h2{font-size:17px;margin:1.6em 0 .4em;border-bottom:2px solid var(--line);
padding-bottom:.3em}
.crumbs{font-size:13px;margin-bottom:8px}
.crumbs a{color:var(--accent);text-decoration:none}
img{max-width:100%;height:auto;border:1px solid var(--line);background:#fff;
border-radius:4px}
.cap{color:var(--mute);font-size:13px}
.q{font-style:italic;color:var(--mute);margin:.2em 0 .8em}
.dl a,.links a{color:var(--accent)}
.tag{display:inline-block;font-size:11px;letter-spacing:.04em;
text-transform:uppercase;padding:2px 8px;border-radius:10px;color:#fff;
background:var(--t3)}
.tag.tier-1{background:var(--t1)}.tag.tier-2{background:var(--t2)}
.grid{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.tile{background:#fff;border:1px solid var(--line);border-radius:6px;
padding:12px;display:flex;flex-direction:column;gap:8px}
.tile.tier-3{opacity:.92;border-style:dashed}
.tile .name{font-weight:600;font-size:14px}
.tile .cap{margin:0}
nav.sections a{margin-right:14px}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left}
th{background:#fff}
"""

#: Return periods ticked on every log-x axis, matching the notebook.
T_TICKS = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
T_FINE = np.logspace(np.log10(1.01), 3.0, 400)
#: Notebook decadal palette, used where the notebook hard-coded it.
DECADE_PALETTE = {
    "1960s": "#E3F2FD",
    "1970s": "#64B5F6",
    "1980s": "#1565C0",
    "1990s": "#0D47A1",
    "2000s": "#6A1B9A",
    "2010s": "#B71C1C",
}


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SuiteContext:
    """The numbers every figure needs, prepared once."""

    bundle: Any
    design: Any
    fits: Any
    pot: Any
    stats: Any = None
    lmoments: Any = None
    autocorrelation: Any = None
    lp3: Any = None
    monte_carlo: Any = None
    mcmc: Any = None
    machine_learning: Any = None
    #: Where :func:`render_suite` writes; set there, not by callers.
    outdir: Path = dataclasses.field(default_factory=lambda: Path("outputs/figures"))

    # -- ordered and sorted series ----------------------------------------
    @property
    def q(self) -> np.ndarray:
        """Annual peaks in water-year order."""
        return self.bundle.q_ordered

    @property
    def years(self) -> np.ndarray:
        return self.bundle.frame["wy_start_year"].to_numpy(dtype=int)

    @property
    def q_sorted(self) -> np.ndarray:
        return np.sort(self.q)

    @property
    def n(self) -> int:
        return int(self.q.size)

    @property
    def decades(self) -> np.ndarray:
        return np.array([self._decade_label(y) for y in self.years], dtype=object)

    @staticmethod
    def _decade_label(year: int) -> str:
        return f"{(int(year) // 10) * 10}s"

    # -- plotting positions ------------------------------------------------
    @property
    def pp_weibull(self) -> np.ndarray:
        """Weibull non-exceedance probabilities, ascending with the sorted peaks."""
        return np.arange(1, self.n + 1) / (self.n + 1)

    @property
    def t_observed(self) -> np.ndarray:
        return 1.0 / (1.0 - self.pp_weibull)

    @property
    def trend_line(self) -> np.ndarray:
        fit = _st.theil_sen(self.q)
        return fit["slope_cumecs_per_year"] * self.years + fit["intercept_cumecs"]

    @property
    def sens_slope(self) -> float:
        return float(_st.theil_sen(self.q)["slope_cumecs_per_year"])

    # -- fitted candidates -------------------------------------------------
    def fit(self, name: str):
        """The named candidate fit, or ``None`` when it is not among the fits."""
        try:
            return self.fits.by_name(name)
        except KeyError:
            return None

    def accepted_fits(self) -> list[Any]:
        return [f for f in self.fits.fits if f.accepted]

    def curve(self, fit, return_periods: np.ndarray) -> np.ndarray:
        """Discharge quantiles of ``fit`` at each return period."""
        return np.asarray(
            fit.ppf(1.0 - 1.0 / np.asarray(return_periods, dtype=float)), dtype=float
        )


def _decade_frame(ctx: SuiteContext) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decade_lbl": ctx.decades,
            "discharge_cms": ctx.q,
            "water_year": ctx.years,
        }
    )


def _feature_frame(ctx: SuiteContext) -> pd.DataFrame:
    """Leakage-safe ML features, joined to the decade of each modelled year."""
    frame = _ml.build_features(ctx.q)
    frame = frame.rename(columns={"q": "Q_peak"})
    frame["decade_lbl"] = [
        ctx._decade_label(ctx.years[min(int(t), ctx.n) - 1]) for t in frame["t"]
    ]
    return frame


def _importance(ctx: SuiteContext, model: str = "RandomForest") -> pd.Series:
    """Feature importance of ``model`` as a descending series, empty if none."""
    if ctx.machine_learning is None:
        return pd.Series(dtype=float)
    for score in ctx.machine_learning.scores:
        if score.model == model and score.feature_importance is not None:
            table = score.feature_importance
            # One row per fold per feature; the importance is the mean over folds.
            grouped = table.groupby("feature")["importance"].mean()
            grouped = grouped.sort_values(ascending=False)
            return grouped.rename("importance")
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


#: Pixel canvases for the interactive pages.  Landscape for the wide figures,
#: because a frequency curve or a dashboard that is letterboxed inside a
#: browser tile loses the resolution the data is actually there for.
CANVAS_PX: dict[str, tuple[int, int]] = {
    "wide": (1600, 900),
    "square": (1100, 1100),
    "tall": (1000, 1250),
}


def _rescale_mpl(fig: plt.Figure, aspect: str) -> plt.Figure:
    """Put a matplotlib figure on the registry canvas for its aspect."""
    fig.set_size_inches(*_cfg.figsize_for(aspect))
    return fig


def _rescale_plotly(fig: go.Figure, aspect: str) -> go.Figure:
    """Put a Plotly figure on the registry canvas for its aspect."""
    width, height = CANVAS_PX[aspect]
    fig.update_layout(width=width, height=height, autosize=False)
    return fig


def _save_mpl(
    fig: plt.Figure, outdir: Path, name: str, caption: str, dpi: int | None = None
) -> FigureRecord:
    """Write a matplotlib figure as PNG plus a captioned HTML page.

    The HTML carries the picture, the caption and the registry metadata; it does
    not pretend to be interactive, because the figure itself is not.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = _cfg.figure_meta(name)
    path = outdir / f"{name}.png"
    _rescale_mpl(fig, meta.aspect)
    fig.savefig(path, dpi=_cfg.CONFIG.dpi if dpi is None else dpi, bbox_inches="tight")
    plt.close(fig)
    html_path = None
    if _cfg.CONFIG.write_figure_html:
        html_path = _write_static_page(fig, outdir, name, caption, meta, path)
    log.info("wrote %s%s", path.name, f" and {html_path.name}" if html_path else "")
    return FigureRecord(
        name=name,
        path=path,
        caption=caption,
        html_path=html_path,
        section=meta.section,
        tier=meta.tier,
        aspect=meta.aspect,
        engine=meta.engine,
        question=meta.question,
    )


def _save_plotly(
    fig: go.Figure, outdir: Path, name: str, caption: str, scale: int = 2
) -> FigureRecord:
    """Write a Plotly figure as PNG and as a fully interactive HTML page."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = _cfg.figure_meta(name)
    png_path = outdir / f"{name}.png"
    _rescale_plotly(fig, meta.aspect)
    fig.write_image(str(png_path), scale=scale)
    html_path = None
    if _cfg.CONFIG.write_figure_html:
        html_path = outdir / f"{name}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs=_cfg.CONFIG.plotly_js,
            full_html=True,
            config={"responsive": True, "displaylogo": False, "toImageButtonOptions":
                    {"format": "png", "scale": scale}},
        )
    log.info(
        "wrote %s%s", png_path.name, f" and {html_path.name}" if html_path else ""
    )
    return FigureRecord(
        name=name,
        path=png_path,
        caption=caption,
        html_path=html_path,
        section=meta.section,
        tier=meta.tier,
        aspect=meta.aspect,
        engine=meta.engine,
        question=meta.question,
    )


def _write_static_page(
    fig: plt.Figure,
    outdir: Path,
    name: str,
    caption: str,
    meta: Any,
    png_path: Path,
) -> Path:
    """A standalone HTML page for a static figure: image, caption, download."""
    rel_png = os.path.relpath(png_path, outdir).replace(os.sep, "/")
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_esc(name)}</title>
<style>{_PAGE_CSS}</style></head>
<body><main>
<nav class="crumbs"><a href="index.html">&#8592; All figures</a></nav>
<p class="tag tier-{meta.tier}">{_esc(meta.section)} &#183; tier {meta.tier}
 &#183; {_esc(meta.engine)} &#183; {_esc(meta.aspect)}</p>
<h1>{_esc(name)}</h1>
<p class="q">{_esc(meta.question)}</p>
<img src="{rel_png}" alt="{_esc(caption)}" loading="lazy">
<p class="cap">{_esc(caption)}</p>
<p class="dl"><a href="{rel_png}" download>Download PNG</a></p>
</main></body></html>
"""
    path = outdir / f"{name}.html"
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# P1 -- Interactive time series with trend, anomaly and rolling statistics
# ---------------------------------------------------------------------------


def plot_p1_time_series(ctx: SuiteContext) -> go.Figure:
    q = ctx.q
    years = ctx.years
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[
            "Annual Instantaneous Peak Flood -- Krishna at Karad",
            "5-Year Rolling Mean & +/-1 sigma Band",
            "Standardised Anomaly (Z-score)",
        ],
    )
    p90, p75, p95 = np.percentile(q, [90, 75, 95])
    colors = ["#EF5350" if v > p90 else "#FF9800" if v > p75 else "#1565C0" for v in q]

    fig.add_trace(
        go.Bar(
            x=years,
            y=q,
            name="Annual Peak",
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Q = %{y:.0f} cumecs<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=ctx.trend_line,
            name=f"Sen's Slope ({ctx.sens_slope:.1f} cumecs/yr)",
            line=dict(color="#B71C1C", width=2.5, dash="dash"),
            hovertemplate="Trend: %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=float(q.mean()),
        line_color="#4CAF50",
        line_dash="dot",
        annotation_text=f"Mean = {q.mean():.0f}",
        row=1,
        col=1,
        annotation_font_size=11,
    )
    fig.add_hrect(
        y0=p90, y1=float(q.max()) * 1.05, fillcolor="rgba(239,83,80,0.07)", line_width=0,
        row=1, col=1,
    )
    for year, value in zip(years, q):
        if value >= p95:
            fig.add_annotation(
                x=int(year),
                y=float(value),
                text=f"{value:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowcolor="#B71C1C",
                font=dict(size=9, color="#B71C1C"),
                ay=-28,
                ax=0,
                row=1,
                col=1,
            )

    series = pd.Series(q, index=years)
    r5_mean = series.rolling(5, center=True).mean()
    r5_std = series.rolling(5, center=True).std()
    fig.add_trace(
        go.Scatter(
            x=list(years) + list(years[::-1]),
            y=list(r5_mean + r5_std) + list((r5_mean - r5_std)[::-1]),
            fill="toself",
            fillcolor="rgba(21,101,192,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5-yr +/-1 sigma",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=r5_mean,
            name="5-yr Rolling Mean",
            line=dict(color="#1565C0", width=2.5),
            hovertemplate="%{y:.0f}<extra>5-yr mean</extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=series.rolling(10, center=True).mean(),
            name="10-yr Rolling Mean",
            line=dict(color="#E65100", width=2, dash="dot"),
            hovertemplate="%{y:.0f}<extra>10-yr mean</extra>",
        ),
        row=2,
        col=1,
    )

    z = (q - q.mean()) / q.std(ddof=1)
    fig.add_trace(
        go.Bar(
            x=years,
            y=z,
            name="Z-score",
            marker_color=[
                "#EF5350" if v > 1.5 else "#FF9800" if v > 0 else "#1565C0" for v in z
            ],
            hovertemplate="Z = %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=1.645,
        line_color="#EF5350",
        line_dash="dash",
        annotation_text="Z=1.645 (10% tail)",
        row=3,
        col=1,
    )
    fig.add_hline(y=-1.645, line_color="#1565C0", line_dash="dash", row=3, col=1)

    fig.update_layout(
        height=700,
        title_text="<b>Karad Flood Time Series -- Complete Hydrological Record</b>",
        title_font_size=16,
        hovermode="x unified",
        showlegend=True,
        legend=dict(x=1.01, y=1),
        plot_bgcolor="rgba(248,249,250,0.8)",
    )
    fig.update_yaxes(title_text="Q (cumecs)", row=1, col=1)
    fig.update_yaxes(title_text="Q (cumecs)", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    fig.update_xaxes(title_text="Water Year", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# P2 -- Distribution explorer
# ---------------------------------------------------------------------------


def plot_p2_distribution_explorer(ctx: SuiteContext) -> plt.Figure:
    q = ctx.q
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Karad Annual Flood Peaks -- Distribution Analysis",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    gumbel, ln2, lp3 = ctx.fit("Gumbel"), ctx.fit("LN2"), ctx.fit("LP3")

    ax = axes[0, 0]
    sns.histplot(
        q, bins=12, stat="density", ax=ax, color="#1565C0", alpha=0.45,
        edgecolor="white", linewidth=0.8, label="Observed",
    )
    sns.kdeplot(q, ax=ax, color="#1565C0", linewidth=2.5, label="KDE")
    sns.rugplot(q, ax=ax, color="#B71C1C", height=0.04, alpha=0.6)
    x_r = np.linspace(q.min() * 0.85, q.max() * 1.1, 400)
    for fit, colour, label in (
        (gumbel, "#E65100", "Gumbel fit"),
        (ln2, "#2E7D32", "LN2 fit"),
    ):
        if fit is not None:
            ax.plot(x_r, fit.pdf(x_r), "--", color=colour, lw=2, label=label)
    ax.set_title("(a) Histogram + KDE + Rug", fontweight="bold")
    ax.set_xlabel("Annual Peak (cumecs)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.axvline(q.mean(), color="red", ls=":", alpha=0.7, lw=1.2)

    ax = axes[0, 1]
    log_q = np.log(q)
    sns.histplot(
        log_q, bins=12, stat="density", ax=ax, color="#2E7D32", alpha=0.45,
        edgecolor="white", label="log(Q)",
    )
    sns.kdeplot(log_q, ax=ax, color="#2E7D32", linewidth=2.5)
    lq_r = np.linspace(log_q.min() - 0.2, log_q.max() + 0.2, 300)
    ax.plot(
        lq_r,
        sps.norm.pdf(lq_r, log_q.mean(), log_q.std(ddof=1)),
        "r--",
        lw=2,
        label="Normal fit (log-space)",
    )
    ax.set_title("(b) Log-Space Distribution (LN2 check)", fontweight="bold")
    ax.set_xlabel("ln(Q)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    sns.ecdfplot(q, ax=ax, color="#6A1B9A", linewidth=2.5, label="Empirical CDF")
    x_c = np.linspace(q.min(), q.max(), 300)
    for fit, colour, style, label in (
        (gumbel, "#E65100", "--", "Gumbel"),
        (ln2, "#2E7D32", ":", "LN2"),
        (lp3, "#1565C0", "-.", "P3"),
    ):
        if fit is not None:
            ax.plot(x_c, fit.cdf(x_c), style, color=colour, lw=2, label=label)
    ax.set_title("(c) ECDF vs Theoretical CDFs", fontweight="bold")
    ax.set_xlabel("Q (cumecs)")
    ax.set_ylabel("P(X <= x)")
    ax.legend(fontsize=8)

    decade = _decade_frame(ctx)
    order = sorted(decade["decade_lbl"].unique())
    ax = axes[1, 0]
    sns.violinplot(
        data=decade, x="decade_lbl", y="discharge_cms", hue="decade_lbl", ax=ax,
        order=order, palette="Blues", inner="box", cut=0.5, linewidth=1.2,
        legend=False,
    )
    ax.axhline(q.mean(), color="red", ls="--", alpha=0.7, lw=1.5)
    ax.set_title("(d) Decade-wise Violin Plots", fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Peak Discharge (cumecs)")

    ax = axes[1, 1]
    sns.boxplot(
        data=decade, x="decade_lbl", y="discharge_cms", hue="decade_lbl", ax=ax,
        order=order, palette="Set2", linewidth=1.5, fliersize=0, legend=False,
    )
    sns.stripplot(
        data=decade, x="decade_lbl", y="discharge_cms", ax=ax, order=order,
        color="#37474F", alpha=0.55, size=4, jitter=True,
    )
    ax.set_title("(e) Box + Jitter (decade)", fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Peak Discharge (cumecs)")

    ax = axes[1, 2]
    lim = [float(q.min() * 0.95), float(q.max() * 1.02)]
    for fit, colour, label in (
        (gumbel, "#E65100", "Gumbel"),
        (ln2, "#2E7D32", "LN2"),
        (lp3, "#6A1B9A", "P3"),
    ):
        if fit is None:
            continue
        q_fit = ctx.curve(fit, ctx.t_observed)
        ax.scatter(q_fit, ctx.q_sorted, s=18, color=colour, alpha=0.65, label=label)
    ax.plot(lim, lim, "k--", lw=1.2)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    ax.set_title("(f) Multi-Distribution Q-Q Plot", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# P3 -- Flood frequency curves
# ---------------------------------------------------------------------------


def plot_p3_frequency_curves(ctx: SuiteContext) -> go.Figure:
    fig = go.Figure()
    n_candidates = len(ctx.fits.fits)
    n_accepted = len(ctx.accepted_fits())

    for fit in ctx.fits.fits:
        curve = np.clip(ctx.curve(fit, T_FINE), 0, 3e4)
        name = fit.candidate.name
        fig.add_trace(
            go.Scatter(
                x=T_FINE,
                y=curve,
                name=name if fit.accepted else f"{name} (rejected)",
                line=dict(
                    color=_ps.distribution_color(name),
                    width=2.2 if fit.accepted else 1.0,
                    dash="solid" if fit.accepted else "dot",
                ),
                opacity=1.0 if fit.accepted else 0.45,
                hovertemplate=(
                    f"<b>{name}</b><br>T=%{{x:.1f}} yr"
                    "<br>Q=%{y:.0f} m3/s<extra></extra>"
                ),
            )
        )

    band = ctx.lp3.bootstrap if ctx.lp3 is not None else None
    if band is not None and len(band):
        t_ci = band["return_period_yr"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t_ci, t_ci[::-1]]),
                y=np.concatenate(
                    [
                        band["Q_upper"].to_numpy(dtype=float),
                        band["Q_lower"].to_numpy(dtype=float)[::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(21,101,192,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="LP3 95% parametric bootstrap CI",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=ctx.t_observed,
            y=ctx.q_sorted,
            mode="markers",
            name="Observed (Weibull)",
            marker=dict(
                color="#212121", size=9, symbol="circle",
                line=dict(color="white", width=1.2),
            ),
            hovertemplate=(
                "<b>Observed</b><br>T=%{x:.1f} yr<br>Q=%{y:.0f} m3/s<extra></extra>"
            ),
            zorder=20,
        )
    )

    best = ctx.fits.best_accepted
    best_fit = ctx.fit(best) if best else None
    for period in (100.0, 500.0, 1000.0):
        fig.add_vline(
            x=period,
            line_dash="dot",
            line_color="#90A4AE",
            line_width=1,
            annotation_text=f"T={period:.0f}yr",
            annotation_font=dict(size=10, color="#546E7A"),
        )
        if best_fit is not None:
            value = float(best_fit.quantile(period))
            fig.add_annotation(
                x=float(np.log10(period)),
                y=value,
                text=f"Q={value:.0f}",
                showarrow=False,
                font=dict(size=9, color="#1565C0"),
                bgcolor="white",
                bordercolor="#1565C0",
                borderwidth=1,
            )

    fig.update_xaxes(
        type="log",
        title_text="Return Period T (years)",
        tickvals=list(T_TICKS),
        ticktext=[f"{t:g}" for t in T_TICKS],
        gridcolor="#E0E0E0",
        showgrid=True,
    )
    fig.update_yaxes(title_text="Design Flood Q_T (m3/s)", gridcolor="#E0E0E0")
    fig.update_layout(
        title=dict(
            text=(
                "<b>Flood Frequency Curves -- Krishna at Karad (AK000X6)</b><br>"
                f"<sup>{n_candidates} candidates fitted, {n_accepted} accepted "
                f"every gate -- LP3 {100 * _cfg.CONFIG.ci_level:g}% bootstrap CI "
                f"-- CWC IS 11223:1985 -- N={ctx.n} years</sup>"
            ),
            font=dict(size=15),
        ),
        height=620,
        hovermode="x unified",
        legend=dict(x=1.01, y=1, font=dict(size=10)),
        plot_bgcolor="rgba(250,250,252,1)",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# P4 -- Pairplot, jointplot, correlation heatmap
# ---------------------------------------------------------------------------


def plot_p4a_pairplot(ctx: SuiteContext) -> plt.Figure:
    features = _feature_frame(ctx)
    importance = _importance(ctx)
    top = list(importance.index[:5]) or [c for c in features.columns if c.startswith("lag_")]
    columns = [c for c in [*top, "Q_peak", "decade_lbl"] if c in features.columns]
    grid = sns.pairplot(
        features[columns].dropna(),
        hue="decade_lbl",
        palette="husl",
        diag_kind="kde",
        plot_kws=dict(alpha=0.55, s=25),
        diag_kws=dict(fill=True, alpha=0.4),
        corner=False,
    )
    grid.figure.suptitle(
        "Pairplot: Top-5 Leakage-Safe Features x Peak Flood (Karad)", y=1.02, fontsize=13
    )
    return grid.figure


def plot_p4b_jointplot(ctx: SuiteContext) -> plt.Figure:
    features = _feature_frame(ctx).dropna()
    lag_column = next((c for c in features.columns if c.startswith("lag_")), None)
    if lag_column is None:
        raise ValueError("machine_learning.build_features produced no lag feature")
    grid = sns.jointplot(
        data=features,
        x=lag_column,
        y="Q_peak",
        kind="reg",
        color="#1565C0",
        marginal_kws=dict(bins=12, fill=True),
        scatter_kws=dict(alpha=0.5, s=35),
        height=7,
    )
    rho, pvalue = sps.spearmanr(features[lag_column], features["Q_peak"])
    grid.ax_joint.annotate(
        f"Spearman rho={rho:.3f}  p={pvalue:.3f}",
        xy=(0.05, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        color="#B71C1C",
        bbox=dict(fc="white", ec="#B71C1C", alpha=0.8),
    )
    grid.figure.suptitle(
        f"Joint Distribution: Q(t) vs Q(t-{lag_column.rsplit('_', 1)[-1]}) -- Karad",
        y=1.01,
        fontsize=12,
    )
    return grid.figure


def plot_p4c_heatmap(ctx: SuiteContext) -> plt.Figure:
    features = _feature_frame(ctx)
    importance = _importance(ctx)
    wanted = list(importance.index[:5]) + [
        "Q_peak",
        "mean_5",
        "std_5",
        "expanding_mean",
    ]
    columns = []
    for name in wanted:
        if name in features.columns and name not in columns:
            columns.append(name)
    if len(columns) < 2:
        raise ValueError("too few numeric features for a correlation matrix")
    correlation = features[columns].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        square=True,
        ax=ax,
        cbar_kws=dict(shrink=0.8, label="Pearson r"),
    )
    ax.set_title(
        "Pearson Correlation Matrix -- ML Features & Peak Flood\n(Karad Station)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# P5 -- Monte Carlo uncertainty fan
# ---------------------------------------------------------------------------


def plot_p5_monte_carlo_uncertainty(ctx: SuiteContext) -> go.Figure:
    mc = ctx.monte_carlo
    if mc is None:
        raise ValueError("no Monte Carlo result; run uncertainty first")
    fan = mc.fan
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            f"Monte Carlo Uncertainty Fan (LP3, {mc.n_records:,} records)",
            "Simulated Estimates at Key T",
        ],
    )
    periods = fan["return_period_yr"].to_numpy(dtype=float)
    fig.add_trace(
        go.Scatter(
            x=list(periods) + list(periods[::-1]),
            y=list(fan["mc_p95_cumecs"]) + list(fan["mc_p05_cumecs"])[::-1],
            fill="toself",
            fillcolor="rgba(21,101,192,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% MC interval",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(periods) + list(periods[::-1]),
            y=list(fan["mc_p75_cumecs"]) + list(fan["mc_p25_cumecs"])[::-1],
            fill="toself",
            fillcolor="rgba(21,101,192,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="50% MC interval",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=fan["mc_median_cumecs"],
            name="MC median",
            line=dict(color="#1565C0", width=3),
            mode="lines+markers",
            marker=dict(size=8),
            hovertemplate="T=%{x} yr<br>Median=%{y:.0f} m3/s<extra>MC</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=fan["mc_p05_cumecs"],
            name="5th percentile",
            line=dict(color="#90CAF9", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=fan["mc_p95_cumecs"],
            name="95th percentile",
            line=dict(color="#1A237E", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )

    for period, colour in zip((10.0, 100.0, 1000.0), ("#90CAF9", "#1565C0", "#0D47A1")):
        subset = mc.replicate_samples[
            mc.replicate_samples["return_period_yr"] == period
        ]
        if subset.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=subset["simulated_cumecs"].to_numpy(dtype=float),
                name=f"T={period:g}",
                box_visible=True,
                meanline_visible=True,
                fillcolor=colour,
                opacity=0.65,
                line_color="#0D47A1",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(
        type="log", title_text="Return Period T (years)", row=1, col=1,
        tickvals=[10, 25, 50, 100, 200, 500, 1000],
    )
    fig.update_yaxes(title_text="Q_T (m3/s)", row=1, col=1)
    fig.update_yaxes(title_text="Q_T (m3/s)", row=1, col=2)
    fig.update_layout(
        height=520,
        title_text=(
            "<b>Monte Carlo Record-Length Uncertainty -- LP3 | "
            f"{mc.n_records:,} independent {mc.record_length}-year records</b>"
        ),
        title_font_size=14,
        hovermode="x unified",
        violinmode="group",
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# P6 -- Bayesian MCMC posteriors
# ---------------------------------------------------------------------------


def plot_p6_mcmc_posteriors(ctx: SuiteContext) -> go.Figure:
    mcmc = ctx.mcmc
    if mcmc is None:
        raise ValueError("no MCMC result; run bayesian_mcmc first")
    chains = np.asarray(mcmc.chains, dtype=float)
    flat = chains.reshape(-1, chains.shape[-1])
    names = list(mcmc.parameter_names)
    fit = ctx.lp3.fit if ctx.lp3 is not None else None
    moment_labels = {
        "mean_ln_q": ("Posterior: mean(ln Q)", "Moment estimate"),
        "sd_ln_q": ("Posterior: sd(ln Q)", "Moment estimate"),
        "cs_log": ("Posterior: Cs_log", "Moment estimate"),
    }
    moment_values = (
        {
            "mean_ln_q": fit.mean_log,
            "sd_ln_q": fit.sd_log,
            "cs_log": fit.cs_log,
        }
        if fit is not None
        else {}
    )
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            *(moment_labels.get(n, (f"Posterior: {n}", ""))[0] for n in names[:3]),
            "Joint Posterior (mean, sd)",
            "Predictive Q100",
            "Predictive Q1000",
            f"Trace ({names[0]})",
        ],
    )
    for index, name in enumerate(names[:3]):
        draws = flat[:, index]
        title, _ = moment_labels.get(name, (f"Posterior: {name}", ""))
        fig.add_trace(
            go.Histogram(
                x=draws,
                nbinsx=45,
                histnorm="probability density",
                name=title,
                marker_color=["#1565C0", "#2E7D32", "#6A1B9A"][index],
                opacity=0.7,
            ),
            row=1,
            col=index + 1,
        )
        fig.add_vline(
            x=float(draws.mean()),
            line_color="#B71C1C",
            line_width=2.5,
            annotation_text=f"Mean={draws.mean():.4g}",
            row=1,
            col=index + 1,
        )
        if name in moment_values:
            fig.add_vline(
                x=float(moment_values[name]),
                line_dash="dash",
                line_color="#FF9800",
                annotation_text="MoM",
                row=1,
                col=index + 1,
            )

    fig.add_trace(
        go.Histogram2dContour(
            x=flat[:, 0], y=flat[:, 1], colorscale="Blues", ncontours=15,
            name="Joint posterior",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[flat[:, 0].mean()],
            y=[flat[:, 1].mean()],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name="posterior mean",
        ),
        row=1,
        col=3,
    )

    for column, period in ((1, 100.0), (2, 1000.0)):
        predictive = np.exp(
            flat[:, 0] + flat[:, 1] * np.array(
                [_skew.frequency_factor_exact(period, cs) for cs in flat[:, 2]]
            )
        )
        fig.add_trace(
            go.Histogram(
                x=predictive,
                nbinsx=45,
                histnorm="probability density",
                name=f"Q{period:g} posterior",
                marker_color="#6A1B9A",
                opacity=0.7,
            ),
            row=2,
            col=column,
        )
        fig.add_vline(
            x=float(predictive.mean()),
            line_color="#B71C1C",
            line_width=2.5,
            annotation_text=f"Mean={predictive.mean():.0f}",
            row=2,
            col=column,
        )
        low, high = np.percentile(predictive, [2.5, 97.5])
        fig.add_vrect(
            x0=low, x1=high, fillcolor="rgba(106,27,154,0.12)", line_width=0,
            row=2, col=column,
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(chains.shape[1] // 10),
            y=chains[0, ::10, 0],
            mode="lines",
            name=f"{names[0]} trace (thinned)",
            line=dict(color="#1565C0", width=0.8),
        ),
        row=2,
        col=3,
    )

    diagnostics = mcmc.diagnostics
    verdict = (
        f"<br><sup>gate PASSED: max R-hat {max(diagnostics.r_hat.values()):.4f}, "
        f"min ESS {min(diagnostics.ess.values()):,.0f}</sup>"
        if diagnostics.passed
        else "<br><sup>gate REFUSED the posterior quantiles: "
        + "; ".join(diagnostics.failures[:2])
        + "</sup>"
    )
    fig.update_layout(
        height=620,
        showlegend=False,
        title_text="<b>Bayesian MCMC -- LP3 Parameters & Predictive Posteriors</b>"
        + verdict,
        title_font_size=14,
    )
    return fig


# ---------------------------------------------------------------------------
# P7 -- Q-Q probability plots
# ---------------------------------------------------------------------------


def plot_p7_qq_plots(ctx: SuiteContext, top_n: int = 6) -> go.Figure:
    ranking = ctx.fits.ranking
    order = ranking.sort_values(["accepted", "aicc"], ascending=[False, True])
    names = list(order["distribution"].head(top_n))
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=names,
        horizontal_spacing=0.09,
        vertical_spacing=0.15,
    )
    for index, name in enumerate(names):
        row, column = divmod(index, 3)
        fit = ctx.fit(name)
        if fit is None:
            continue
        q_fit = np.clip(ctx.curve(fit, ctx.t_observed), 0, 2e4)
        colour = _ps.distribution_color(name)
        r2 = float(
            1.0
            - np.sum((ctx.q_sorted - q_fit) ** 2)
            / np.sum((ctx.q_sorted - ctx.q_sorted.mean()) ** 2)
        )
        fig.add_trace(
            go.Scatter(
                x=q_fit,
                y=ctx.q_sorted,
                mode="markers",
                marker=dict(
                    color=colour, size=7, opacity=0.75,
                    line=dict(color="white", width=0.8),
                ),
                name=name,
                showlegend=False,
                hovertemplate=f"Fitted=%{{x:.0f}}<br>Obs=%{{y:.0f}}<br>{name}<extra></extra>",
            ),
            row=row + 1,
            col=column + 1,
        )
        low = float(min(q_fit.min(), ctx.q_sorted.min()) * 0.95)
        high = float(max(q_fit.max(), ctx.q_sorted.max()) * 1.02)
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line=dict(color="#B0BEC5", dash="dash", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=column + 1,
        )
        fig.update_xaxes(title_text="Fitted (m3/s)", row=row + 1, col=column + 1)
        fig.update_yaxes(title_text="Observed (m3/s)", row=row + 1, col=column + 1)
        # The first subplot's axes are named "x"/"y" with no index suffix.
        suffix = "" if index == 0 else str(index + 1)
        fig.add_annotation(
            x=0.05,
            y=0.90,
            xref=f"x{suffix} domain",
            yref=f"y{suffix} domain",
            text=f"R2={r2:.4f}<br>{'accepted' if fit.accepted else 'REJECTED'}",
            showarrow=False,
            font=dict(size=10, color=colour),
            bgcolor="white",
            bordercolor=colour,
            borderwidth=1,
        )
    fig.update_layout(
        height=580,
        title_text=(
            "<b>Q-Q Probability Plots -- Top "
            f"{len(names)} Distributions (AICc ranked)</b>"
        ),
        title_font_size=14,
        plot_bgcolor="rgba(250,250,252,1)",
    )
    return fig


# ---------------------------------------------------------------------------
# P8 -- Skewness and kurtosis
# ---------------------------------------------------------------------------


def plot_p8_skewness_kurtosis(ctx: SuiteContext) -> plt.Figure:
    q = ctx.q
    moments = ctx.stats.moments
    boot = _st.bootstrap_moment_samples(q)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Skewness / Kurtosis / Moment Analysis -- Karad Annual Peaks",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for column, (values, observed, colour, dark, label, title, xlabel) in enumerate(
        (
            (
                boot["skewness"].to_numpy(dtype=float),
                float(moments.skewness_unadjusted),
                "#E65100",
                "#B71C1C",
                "Cs",
                "(a) Bootstrap Skewness CI",
                "Skewness Coefficient (Cs)",
            ),
            (
                boot["kurtosis"].to_numpy(dtype=float),
                float(moments.excess_kurtosis),
                "#6A1B9A",
                "#4A148C",
                "Ck",
                "(b) Bootstrap Kurtosis CI",
                "Excess Kurtosis (Fisher)",
            ),
        )
    ):
        ax = axes[0, column]
        sns.histplot(
            values, bins=50, stat="density", ax=ax, color=colour, alpha=0.6,
            edgecolor="white",
        )
        sns.kdeplot(values, ax=ax, color=dark, lw=2.5)
        ax.axvline(observed, color=dark, lw=2.5, label=f"Observed {label}={observed:.4f}")
        low, high = np.percentile(values, [2.5, 97.5])
        ax.axvspan(low, high, alpha=0.15, color=colour, label=f"95% CI [{low:.3f},{high:.3f}]")
        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="Normal")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    ax = axes[0, 2]
    cs_range = np.linspace(0, 2.5, 100)
    ax.plot(cs_range, 1.5 * cs_range**2, "b-", lw=2, label="Pearson III: Ck=1.5*Cs^2")
    ax.plot(cs_range, 1.5 * cs_range**2, "g--", lw=2, label="LP3 approx")
    ax.axhline(0, color="gray", ls=":", lw=1, label="Normal (Cs=Ck=0)")
    ax.scatter(
        [moments.skewness_unadjusted], [moments.excess_kurtosis], s=150, color="red",
        zorder=10, marker="*", label="Karad data",
    )
    ax.set_xlabel("Skewness Cs")
    ax.set_ylabel("Excess Kurtosis Ck")
    ax.set_title("(c) Cs-Ck Moment Space", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    (osm, osr), (slope, intercept, r) = sps.probplot(q, dist="norm", plot=None)
    ax.scatter(osm, osr, s=25, color="#1565C0", alpha=0.7, label="Observed vs Normal")
    ax.plot(osm, slope * osm + intercept, "r-", lw=2, label=f"Fit (r={r:.3f})")
    (osm_ln, osr_ln), (slope_ln, intercept_ln, r_ln) = sps.probplot(
        np.log(q), dist="norm", plot=None
    )
    twin = ax.twinx()
    twin.scatter(osm_ln, osr_ln, s=25, color="#2E7D32", alpha=0.5, marker="^")
    twin.plot(osm_ln, slope_ln * osm_ln + intercept_ln, "g--", lw=1.5)
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Q (cumecs)", color="#1565C0")
    twin.set_ylabel("ln(Q)", color="#2E7D32")
    ax.set_title(f"(d) Normal Probability Paper  (r={r:.3f})", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    decade = _decade_frame(ctx)
    for label, group in decade.groupby("decade_lbl"):
        if len(group) >= 4:
            sns.kdeplot(
                group["discharge_cms"], ax=ax,
                color=DECADE_PALETTE.get(label, "gray"), linewidth=2.2,
                fill=True, alpha=0.18, label=label, cut=0,
            )
    ax.set_xlabel("Annual Peak (cumecs)")
    ax.set_ylabel("Density")
    ax.set_title("(e) Decadal KDE (non-stationarity check)", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    cv_values = boot["cv"].to_numpy(dtype=float)
    cv_observed = float(q.std(ddof=1) / q.mean())
    sns.histplot(
        cv_values, bins=50, stat="density", ax=ax, color="#00695C", alpha=0.6,
        edgecolor="white",
    )
    ax.axvline(cv_observed, color="#004D40", lw=2.5, label=f"Observed CV={cv_observed:.4f}")
    low, high = np.percentile(cv_values, [2.5, 97.5])
    ax.axvspan(low, high, alpha=0.15, color="#00695C", label=f"95% CI [{low:.3f},{high:.3f}]")
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_title("(f) Bootstrap CV Distribution", fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# P9 -- ACF, PACF, Hurst, Ljung-Box
# ---------------------------------------------------------------------------


def plot_p9_dependence(ctx: SuiteContext) -> go.Figure:
    if ctx.autocorrelation is None:
        raise ValueError("no autocorrelation result; run autocorrelation first")
    ac = ctx.autocorrelation
    frame = ac.acf_frame
    stats_rows = (
        ctx.stats.randomness
        if ctx.stats is not None
        else pd.DataFrame({"test": [], "lag": [], "p_value": []})
    )
    ljung = stats_rows[stats_rows["test"].astype(str).str.contains("Ljung")]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Autocorrelation (ACF)",
            "Partial ACF (PACF)",
            "Lag Plot Q(t) vs Q(t-1)",
            "Hurst R/S Analysis",
            "Ljung-Box p-values",
            "Rolling Variance (Stability)",
        ],
    )
    band = float(frame["band_high"].iloc[0])
    for _, row in frame[frame["lag"] > 0].iterrows():
        fig.add_trace(
            go.Bar(
                x=[int(row["lag"])],
                y=[float(row["acf"])],
                marker_color="#EF5350" if abs(row["acf"]) > band else "#1565C0",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[int(row["lag"])],
                y=[float(row["pacf"])],
                marker_color="#EF5350" if abs(row["pacf"]) > band else "#2E7D32",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    for column in (1, 2):
        fig.add_hline(y=band, line_dash="dash", line_color="#90A4AE", row=1, col=column)
        fig.add_hline(y=-band, line_dash="dash", line_color="#90A4AE", row=1, col=column)
    fig.add_hline(y=0, line_color="#37474F", line_width=1, row=1, col=1)

    q = ctx.q
    fig.add_trace(
        go.Scatter(
            x=q[:-1],
            y=q[1:],
            mode="markers",
            marker=dict(
                color=list(range(q.size - 1)), colorscale="Viridis", size=8,
                opacity=0.75, showscale=True,
                colorbar=dict(title="Time", x=0.66, len=0.45, y=0.75),
            ),
            name="Q(t) vs Q(t-1)",
        ),
        row=1,
        col=3,
    )
    rho, _ = sps.spearmanr(q[:-1], q[1:])
    fig.add_annotation(
        x=0.05, y=0.92, xref="x3 domain", yref="y3 domain",
        text=f"Spearman rho={rho:.3f}", showarrow=False, font=dict(size=10),
        bgcolor="white",
    )

    from . import autocorrelation as _ac

    rs = _ac.rescaled_range_curve(q)
    fig.add_trace(
        go.Scatter(
            x=np.log(rs["lag"].to_numpy(dtype=float)),
            y=np.log(rs["rs_ratio"].to_numpy(dtype=float)),
            mode="markers+lines",
            marker=dict(color="#E65100", size=7),
            line=dict(color="#E65100", width=2),
            name="R/S vs n",
        ),
        row=2,
        col=1,
    )
    slope, intercept = np.polyfit(
        np.log(rs["lag"].to_numpy(dtype=float)),
        np.log(rs["rs_ratio"].to_numpy(dtype=float)),
        1,
    )
    x_fit = np.array(
        [float(np.log(rs["lag"].min())), float(np.log(rs["lag"].max()))]
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=slope * x_fit + intercept,
            mode="lines",
            line=dict(color="#B71C1C", dash="dash", width=2),
            name=f"RS slope H={slope:.3f}",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    h_note = f"H(Powers)={ac.hurst_powers:.3f}"
    if not ac.hurst_interval.get("point_admissible", True):
        h_note += " -- NOT admissible (outside 0-1)"
    h_note += f"; MC s.d. {ac.hurst_verification['monte_carlo_sd']:.3f}"
    fig.add_annotation(
        x=0.02, y=0.98, xref="x4 domain", yref="y4 domain", text=h_note,
        showarrow=False, font=dict(size=10, color="#B71C1C"), bgcolor="white",
        align="left",
    )

    if len(ljung):
        pvalues = ljung["p_value"].to_numpy(dtype=float)
        fig.add_trace(
            go.Bar(
                x=ljung["lag"].to_numpy(dtype=float),
                y=pvalues,
                marker_color=["#EF5350" if p < 0.05 else "#4CAF50" for p in pvalues],
                name="Ljung-Box p",
            ),
            row=2,
            col=2,
        )
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="#B71C1C", annotation_text="alpha=0.05",
        row=2, col=2,
    )

    series = pd.Series(q, index=ctx.years)
    fig.add_trace(
        go.Scatter(
            x=ctx.years,
            y=series.rolling(10, center=True).var(),
            mode="lines+markers",
            line=dict(color="#6A1B9A", width=2),
            marker=dict(size=5),
            name="10-yr Rolling Var",
        ),
        row=2,
        col=3,
    )
    fig.add_hline(
        y=float(q.var()), line_dash="dot", line_color="#9E9E9E",
        annotation_text="Full-record var", row=2, col=3,
    )

    fig.update_layout(
        height=620,
        title_text="<b>Time-Series Structure -- ACF / PACF / Hurst / Ljung-Box</b>",
        title_font_size=14,
        showlegend=False,
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_xaxes(title_text="Q(t-1) cumecs", row=1, col=3)
    fig.update_xaxes(title_text="ln(n)", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=3)
    fig.update_yaxes(title_text="ACF", row=1, col=1)
    fig.update_yaxes(title_text="PACF", row=1, col=2)
    fig.update_yaxes(title_text="Q(t) cumecs", row=1, col=3)
    fig.update_yaxes(title_text="ln(R/S)", row=2, col=1)
    fig.update_yaxes(title_text="p-value", row=2, col=2)
    fig.update_yaxes(title_text="Rolling Var", row=2, col=3)
    return fig


# ---------------------------------------------------------------------------
# P10 -- Distribution ranking
# ---------------------------------------------------------------------------


def plot_p10_distribution_ranking(ctx: SuiteContext) -> go.Figure:
    frame = ctx.fits.ranking.reset_index(drop=True)
    frame.index += 1
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=[
            f"Information Criteria Comparison ({len(frame)} fitted candidates)",
            "Akaike Weights",
        ],
    )
    width = 0.2
    for index, (column, colour) in enumerate(
        zip(
            ("aic", "aicc", "bic", "hqic"),
            ("#1565C0", "#E65100", "#2E7D32", "#6A1B9A"),
        )
    ):
        fig.add_trace(
            go.Bar(
                x=list(np.arange(len(frame)) + index * width - 1.5 * width),
                y=frame[column].tolist(),
                name=column.upper(),
                marker_color=colour,
                width=width * 0.85,
                text=frame["distribution"].tolist(),
                hoverinfo="text+y",
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(
        tickvals=list(np.arange(len(frame))),
        ticktext=[n[:14] for n in frame["distribution"]],
        tickangle=-35,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Information Criterion Value", row=1, col=1)

    accepted = frame[frame["accepted"]]
    if len(accepted):
        delta = accepted["aicc"].to_numpy(dtype=float) - accepted["aicc"].min()
        weights = np.exp(-0.5 * delta)
        weights = weights / weights.sum()
    else:
        accepted = frame.assign(akaike_weight=np.nan)
        weights = np.array([])
    fig.add_trace(
        go.Bar(
            x=accepted["distribution"].tolist(),
            y=list(weights),
            marker_color=[
                _ps.distribution_color(n) for n in accepted["distribution"]
            ],
            text=[f"{w:.4f}" for w in weights],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(tickangle=-35, row=1, col=2)
    fig.update_yaxes(title_text="Akaike Weight (accepted only)", row=1, col=2)
    fig.update_layout(
        height=500,
        barmode="group",
        title_text=(
            "<b>Goodness-of-Fit & Information Criterion Ranking -- "
            f"{len(frame)} Distributions</b>"
        ),
        title_font_size=14,
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# P11 -- POT / GPD
# ---------------------------------------------------------------------------


def plot_p11_pot(ctx: SuiteContext) -> go.Figure:
    pot = ctx.pot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Mean Residual Life Plot",
            "GPD Return Level Plot",
            "Excess Distribution (Fitted vs Empirical)",
            "Threshold Stability (Shape xi)",
        ],
    )
    mrl = _pot.mean_residual_life(ctx.q)
    usable = mrl[mrl["usable"]]
    if len(usable):
        fig.add_trace(
            go.Scatter(
                x=list(usable["threshold"]) + list(usable["threshold"])[::-1],
                y=list(usable["ci_upper"]) + list(usable["ci_lower"])[::-1],
                fill="toself",
                fillcolor="rgba(21,101,192,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=usable["threshold"],
                y=usable["mean_excess"],
                mode="lines+markers",
                line=dict(color="#1565C0", width=2.2),
                marker=dict(size=5),
                name="Mean excess",
            ),
            row=1,
            col=1,
        )
    fig.add_vline(
        x=float(pot.threshold),
        line_dash="dash",
        line_color="#B71C1C",
        annotation_text=f"u={pot.threshold:,.0f}",
        row=1,
        col=1,
    )

    levels = pot.return_levels
    if levels is not None and len(levels):
        fig.add_trace(
            go.Scatter(
                x=levels["return_period_yr"],
                y=levels["return_level_cumecs"],
                mode="lines",
                name="GPD return level",
                line=dict(color="#E65100", width=2.5),
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=ctx.t_observed, y=ctx.q_sorted, mode="markers", name="Observed",
            marker=dict(color="#212121", size=7),
        ),
        row=1,
        col=2,
    )

    exceed = ctx.q[ctx.q > float(pot.threshold)]
    if exceed.size >= 5:
        excess = np.sort(exceed - float(pot.threshold))
        empirical = np.arange(1, excess.size + 1) / (excess.size + 1)
        fitted = sps.genpareto.cdf(
            excess, pot.fit["xi"], loc=0.0, scale=pot.fit["beta"]
        )
        fig.add_trace(
            go.Scatter(
                x=excess, y=empirical, mode="markers", name="Empirical",
                marker=dict(color="#2E7D32", size=7),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=excess, y=fitted, mode="lines", name="GPD CDF",
                line=dict(color="#B71C1C", width=2.2),
            ),
            row=2,
            col=1,
        )

    stability = pot.stability
    if stability is not None and len(stability):
        fig.add_trace(
            go.Scatter(
                x=stability["threshold"],
                y=stability["xi"],
                mode="lines+markers",
                line=dict(color="#6A1B9A", width=2.2),
                marker=dict(size=7),
                name="Shape xi(u)",
            ),
            row=2,
            col=2,
        )
    fig.add_hline(
        y=0, line_dash="dot", line_color="#90A4AE", annotation_text="xi=0",
        row=2, col=2,
    )

    stamp = (
        "<br><sup>POT ACCEPTED</sup>"
        if pot.accepted
        else "<br><sup>POT NOT ADOPTED: " + "; ".join(pot.reject_reasons)[:400] + "</sup>"
    )
    fig.update_xaxes(type="log", title_text="Return Period (yr)", row=1, col=2,
                     tickvals=list(T_TICKS))
    fig.update_xaxes(title_text="Threshold u", row=1, col=1)
    fig.update_xaxes(title_text="Excess (m3/s)", row=2, col=1)
    fig.update_xaxes(title_text="Threshold u", row=2, col=2)
    fig.update_yaxes(title_text="Mean Excess e(u)", row=1, col=1)
    fig.update_yaxes(title_text="Q_T (m3/s)", row=1, col=2)
    fig.update_yaxes(title_text="CDF", row=2, col=1)
    fig.update_yaxes(title_text="Shape xi", row=2, col=2)
    fig.update_layout(
        height=620,
        title_text="<b>Peaks Over Threshold (POT) & Generalised Pareto Analysis</b>"
        + stamp,
        title_font_size=14,
        showlegend=True,
        legend=dict(x=1.01, y=1),
    )
    return fig


# ---------------------------------------------------------------------------
# P12 -- ML performance panel
# ---------------------------------------------------------------------------


def plot_p12_ml_panel(ctx: SuiteContext) -> plt.Figure:
    if ctx.machine_learning is None:
        raise ValueError("no machine-learning result; run machine_learning first")
    mlr = ctx.machine_learning
    predictions = mlr.predictions
    models = list(mlr.scores and [s.model for s in mlr.scores])
    colours = sns.color_palette("husl", len(models))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Machine Learning Model Performance -- Karad Annual Floods\n"
        "(leakage-safe out-of-fold predictions, not train fits)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    ax = axes[0, 0]
    for model, colour in zip(models, colours):
        subset = predictions[predictions["model"] == model]
        if subset.empty:
            continue
        score = next(s for s in mlr.scores if s.model == model)
        ax.scatter(
            subset["actual_cumecs"],
            subset["predicted_cumecs"],
            s=18,
            alpha=0.55,
            color=colour,
            label=f"{model} R2={score.metrics['r2']:.2f}",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, transform=ax.transAxes)
    ax.set_xlabel("Observed (cumecs)")
    ax.set_ylabel("Predicted (cumecs)")
    ax.set_title("(a) All ML: Predicted vs Observed (out-of-fold)", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    importance = _importance(ctx)
    if importance.empty:
        ax.text(0.5, 0.5, "no tree model in the registry", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        # One bar per feature, already in descending order; a categorical
        # barplot would re-sort the categories and drop the rank.
        colours = sns.color_palette("Blues_r", len(importance))
        ax.barh(
            np.arange(len(importance)),
            importance.to_numpy(dtype=float),
            color=colours,
            edgecolor="white",
        )
        ax.set_yticks(np.arange(len(importance)))
        ax.set_yticklabels(list(importance.index))
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Random Forest)")
        ax.set_title("(b) RF Feature Importance", fontweight="bold")
        for index, value in enumerate(importance.values):
            ax.text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=8)

    ax = axes[0, 2]
    best = min(mlr.scores, key=lambda s: s.metrics["rmse_cumecs"])
    subset = predictions[predictions["model"] == best.model].sort_values("year")
    spread = subset.groupby("year")["predicted_cumecs"].std()
    ax.plot(subset["year"], subset["actual_cumecs"], "ko-", lw=1, ms=4, label="Observed")
    ax.plot(
        subset["year"], subset["predicted_cumecs"], "r-", lw=2.2,
        label=f"{best.model} out-of-fold (R2={best.metrics['r2']:.3f})",
    )
    ax.fill_between(
        subset["year"].to_numpy(dtype=float),
        subset["predicted_cumecs"].to_numpy(dtype=float) - spread.reindex(
            subset["year"]
        ).fillna(0.0).to_numpy(dtype=float),
        subset["predicted_cumecs"].to_numpy(dtype=float) + spread.reindex(
            subset["year"]
        ).fillna(0.0).to_numpy(dtype=float),
        alpha=0.2,
        color="#FF9800",
        label="fold spread +/-1sd",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Discharge (cumecs)")
    ax.set_title("(c) Time-Series Reconstruction (out-of-fold)", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    residuals = subset["actual_cumecs"] - subset["predicted_cumecs"]
    sns.residplot(
        x=subset["predicted_cumecs"], y=residuals, ax=ax,
        scatter_kws=dict(alpha=0.6, s=20), line_kws=dict(color="red", lw=2),
        lowess=True,
    )
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"(d) Residual Plot ({best.model})", fontweight="bold")

    ax = axes[1, 1]
    r2_values = [s.metrics["r2"] for s in mlr.scores]
    bars = ax.bar(
        models, r2_values, color=colours[: len(models)],
        edgecolor="white",
    )
    ax.set_ylabel("R2 (out-of-fold)")
    ax.set_title("(e) Out-of-Fold R2 Comparison", fontweight="bold")
    ax.axhline(0.0, color="green", ls="--", lw=1, alpha=0.6)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars.patches, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )

    ax = axes[1, 2]
    rmse_values = [s.metrics["rmse_cumecs"] for s in mlr.scores]
    bars2 = ax.bar(
        models, rmse_values, color=list(colours[::-1])[: len(models)],
        edgecolor="white",
    )
    ax.set_ylabel("RMSE (cumecs)")
    ax.set_title("(f) Out-of-Fold RMSE Comparison", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    persistence = mlr.baseline.get("persistence_rmse_cumecs", float("nan"))
    if np.isfinite(persistence):
        ax.axhline(
            persistence, color="#B71C1C", ls="--", lw=1.2,
            label=f"persistence RMSE {persistence:,.0f}",
        )
        ax.legend(fontsize=8)
    for bar, value in zip(bars2.patches, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_values) * 0.02,
            f"{value:,.0f}",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# P13 -- Publication dashboard
# ---------------------------------------------------------------------------


def plot_p13_dashboard(ctx: SuiteContext) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "Time Series + Trend",
            "Frequency Curves",
            "GoF: AICc Ranking",
            "Design Flood Comparison",
            "Monte Carlo Q100 Distribution",
            "L-moment Ratio Diagram",
            "Decade-wise Discharge",
            "Monte Carlo CI Fan",
            "ML Feature Importance",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.35, 0.35, 0.30],
    )
    q = ctx.q
    p90 = float(np.percentile(q, 90))
    fig.add_trace(
        go.Bar(
            x=list(ctx.years),
            y=list(q),
            name="Annual Peak",
            marker_color=["#EF5350" if v > p90 else "#1565C0" for v in q],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(ctx.years), y=list(ctx.trend_line), name="Trend",
            line=dict(color="red", dash="dash", width=2), showlegend=False,
        ),
        row=1,
        col=1,
    )

    for fit in ctx.accepted_fits():
        fig.add_trace(
            go.Scatter(
                x=T_FINE,
                y=np.clip(ctx.curve(fit, T_FINE), 0, 3e4),
                name=fit.candidate.name,
                line=dict(color=_ps.distribution_color(fit.candidate.name), width=2),
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=ctx.t_observed, y=ctx.q_sorted, mode="markers", name="Observed",
            marker=dict(color="#212121", size=6), showlegend=False,
        ),
        row=1,
        col=2,
    )

    ranked = ctx.fits.ranking[ctx.fits.ranking["accepted"]].sort_values("aicc")
    fig.add_trace(
        go.Bar(
            x=[n[:13] for n in ranked["distribution"]],
            y=ranked["aicc"].tolist(),
            marker_color=[
                _ps.distribution_color(n) for n in ranked["distribution"]
            ],
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    periods = np.array([10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    for fit in ctx.accepted_fits():
        fig.add_trace(
            go.Bar(
                x=[f"T={t:g}" for t in periods],
                y=np.clip(ctx.curve(fit, periods), 0, 2e4).tolist(),
                name=fit.candidate.name[:12],
                marker_color=_ps.distribution_color(fit.candidate.name),
            ),
            row=2,
            col=1,
        )

    if ctx.monte_carlo is not None:
        subset = ctx.monte_carlo.replicate_samples[
            ctx.monte_carlo.replicate_samples["return_period_yr"] == 100.0
        ]
        if not subset.empty:
            sims = subset["simulated_cumecs"].to_numpy(dtype=float)
            fig.add_trace(
                go.Histogram(
                    x=sims, nbinsx=45, name="MC Q100", marker_color="#6A1B9A",
                    opacity=0.7, showlegend=False,
                ),
                row=2,
                col=2,
            )
            fig.add_vline(
                x=float(sims.mean()), line_color="red", line_width=2,
                annotation_text=f"Mean={sims.mean():.0f}", row=2, col=2,
            )

    if ctx.lmoments is not None:
        tau3 = np.linspace(-0.3, 0.7, 200)
        fig.add_trace(
            go.Scatter(
                x=tau3,
                y=0.10701 + 0.11090 * tau3 + 0.84838 * tau3**2
                - 0.06669 * tau3**3 + 0.00567 * tau3**4,
                line=dict(color="#1565C0", width=2),
                showlegend=False,
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=[ctx.lmoments.stats["lskew"]],
                y=[ctx.lmoments.stats["lkurt"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    decade = _decade_frame(ctx)
    for index, (label, group) in enumerate(decade.groupby("decade_lbl")):
        fig.add_trace(
            go.Violin(
                y=group["discharge_cms"].tolist(),
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=_ps.DECADE_COLORS[index % len(_ps.DECADE_COLORS)],
                opacity=0.7,
                line_color="#0D47A1",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    if ctx.monte_carlo is not None:
        fan = ctx.monte_carlo.fan
        fan_periods = fan["return_period_yr"].to_numpy(dtype=float)
        for low, high, alpha, label in (
            ("mc_p05_cumecs", "mc_p95_cumecs", 0.12, "90% CI"),
            ("mc_p25_cumecs", "mc_p75_cumecs", 0.22, "50% CI"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=list(fan_periods) + list(fan_periods)[::-1],
                    y=list(fan[high]) + list(fan[low])[::-1],
                    fill="toself",
                    fillcolor=f"rgba(21,101,192,{alpha})",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=label,
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
        fig.add_trace(
            go.Scatter(
                x=fan_periods,
                y=fan["mc_median_cumecs"],
                name="MC median",
                line=dict(color="#1565C0", width=2.5),
                mode="lines+markers",
                showlegend=False,
            ),
            row=3,
            col=2,
        )

    importance = _importance(ctx)
    if not importance.empty:
        fig.add_trace(
            go.Bar(
                x=importance.values.tolist(),
                y=importance.index.tolist(),
                orientation="h",
                marker_color="#E65100",
                showlegend=False,
            ),
            row=3,
            col=3,
        )

    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_xaxes(type="log", row=3, col=2, tickvals=list(T_TICKS))
    n_accepted = len(ctx.accepted_fits())
    fig.update_layout(
        height=900,
        barmode="group",
        violinmode="group",
        title_text=(
            "<b>Advanced Flood Frequency Analysis -- Krishna at Karad (AK000X6)</b><br>"
            f"<sup>CWC IS 11223:1985 -- {n_accepted} accepted of "
            f"{len(ctx.fits.fits)} candidates -- N={ctx.n} years</sup>"
        ),
        title_font_size=16,
        legend=dict(x=1.01, y=1, font=dict(size=9)),
        plot_bgcolor="rgba(250,250,252,1)",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# P14 -- Comprehensive summary
# ---------------------------------------------------------------------------


def plot_p14_comprehensive(ctx: SuiteContext) -> plt.Figure:
    q = ctx.q
    fig = plt.figure(figsize=(18, 14))
    grid = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f"Comprehensive Statistical Summary -- Krishna at Karad (N={ctx.n} yr)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    decade = _decade_frame(ctx)

    ax = fig.add_subplot(grid[0, :2])
    palette = sns.color_palette("husl", 6)
    offset = 0.0
    for index, (label, group) in enumerate(decade.groupby("decade_lbl")):
        if len(group) < 4:
            continue
        values = group["discharge_cms"].to_numpy(dtype=float)
        kde_x = np.linspace(values.min(), values.max(), 300)
        kde_y = sps.gaussian_kde(values)(kde_x)
        ax.fill_between(
            kde_x, offset, offset + kde_y * 6000, alpha=0.55, color=palette[index % 6],
            label=label,
        )
        ax.plot(kde_x, offset + kde_y * 6000, lw=1.8, color=palette[index % 6])
        offset += 0.8
    ax.set_title("(a) Ridgeline Plot -- Decadal Flood Distributions", fontweight="bold")
    ax.set_xlabel("Annual Peak Discharge (cumecs)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 2:])
    gumbel = ctx.fit("Gumbel")
    y_emp = -np.log(-np.log(1.0 - ctx.pp_weibull))
    y_range = np.linspace(-2.0, 8.0, 300)
    if gumbel is not None:
        location = float(gumbel.params.get("loc", float("nan")))
        scale = float(gumbel.params.get("scale", float("nan")))
        ax.plot(y_range, location + scale * y_range, "r-", lw=2.5, label="Gumbel fit")
    ax.scatter(y_emp, ctx.q_sorted, s=40, color="#1565C0", zorder=5, label="Observed")
    ticks = (1.5, 2, 5, 10, 25, 50, 100, 500, 1000)
    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks([-np.log(-np.log(1 - 1 / t)) for t in ticks])
    twin.set_xticklabels([str(t) for t in ticks], fontsize=8)
    twin.set_xlabel("Return Period T (years)", fontsize=9)
    ax.set_xlabel("Gumbel Reduced Variate y_T")
    ax.set_ylabel("Q (cumecs)")
    ax.set_title("(b) Gumbel Probability Paper", fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, :2])
    order = sorted(decade["decade_lbl"].unique())
    sns.boxplot(
        data=decade, x="decade_lbl", y="discharge_cms", hue="decade_lbl",
        order=order, palette="Blues", ax=ax, linewidth=1.5, fliersize=0,
        legend=False,
    )
    sns.swarmplot(
        data=decade, x="decade_lbl", y="discharge_cms", order=order,
        color="#0D47A1", alpha=0.6, size=4, ax=ax,
    )
    ax.axhline(q.mean(), color="red", ls="--", lw=1.5, alpha=0.7, label=f"Mean={q.mean():.0f}")
    ax.set_title("(c) Decade Box + Swarm Plot", fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Peak Discharge (cumecs)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 2:])
    sns.ecdfplot(q, ax=ax, color="#37474F", linewidth=2.5, label="Empirical ECDF")
    x_cdf = np.linspace(q.min() * 0.9, q.max() * 1.05, 300)
    for name, colour, style in (
        ("Gumbel", "#E65100", "-"),
        ("LN2", "#2E7D32", "--"),
        ("LP3", "#6A1B9A", ":"),
    ):
        fit = ctx.fit(name)
        if fit is not None:
            ax.plot(x_cdf, fit.cdf(x_cdf), lw=2, color=colour, ls=style, label=name)
    ax.set_title("(d) ECDF vs Theoretical CDFs", fontweight="bold")
    ax.set_xlabel("Q (cumecs)")
    ax.set_ylabel("P(X <= x)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[2, :2])
    heat_periods = list(T_TICKS)
    heat_fits = [f for f in ctx.fits.fits if f.candidate.name in ("Gumbel", "LN2", "LP3", "GEV", "Weibull")]
    heat = pd.DataFrame(
        {
            f.candidate.name: np.round(
                np.clip(ctx.curve(f, np.array(heat_periods, dtype=float)), 0, 2e4), 0
            )
            for f in heat_fits
        },
        index=[f"T={t:g}" for t in heat_periods],
    )
    sns.heatmap(
        heat, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5, linecolor="white",
        annot_kws={"size": 8}, ax=ax, cbar_kws=dict(label="Q (cumecs)"),
    )
    ax.set_title("(e) Design Flood Heatmap (cumecs)", fontweight="bold")
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Return Period")

    ax = fig.add_subplot(grid[2, 2:])
    if ctx.monte_carlo is not None:
        fan = ctx.monte_carlo.fan
        periods = fan["return_period_yr"].to_numpy(dtype=float)
        ax.fill_between(
            periods, fan["mc_p05_cumecs"], fan["mc_p95_cumecs"], alpha=0.15,
            color="#FF9800", label="MC 90% interval",
        )
        ax.fill_between(
            periods, fan["mc_p25_cumecs"], fan["mc_p75_cumecs"], alpha=0.25,
            color="#FF9800", label="MC 50% interval",
        )
        ax.plot(periods, fan["mc_median_cumecs"], "o-", color="#E65100", lw=2, ms=4,
                label="MC median")
        ax.plot(periods, fan["observed_lp3_cumecs"], "k^", ms=5, alpha=0.7,
                label="Observed-record LP3")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Return Period (years, log scale)")
    ax.set_ylabel("Q (cumecs, log scale)")
    ax.set_title("(f) Monte Carlo Record-Length Uncertainty", fontweight="bold")
    ax.legend(fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Render everything
# ---------------------------------------------------------------------------


def _builders(ctx: SuiteContext) -> list[tuple[str, str, Any]]:
    """``(name, caption, callable)`` for every figure in the notebook suite."""
    return [
        (
            "P1_time_series_trend_anomaly",
            "Annual peaks in water-year order with the Theil-Sen trend, "
            "rolling mean and standardised anomaly",
            lambda: _save_plotly(plot_p1_time_series(ctx), ctx.outdir,
                                 "P1_time_series_trend_anomaly",
                                 "Annual peaks, trend, rolling mean and anomaly"),
        ),
        (
            "P2_seaborn_distribution",
            "Histogram, KDE, log-space check, ECDF against fitted CDFs, "
            "decadal violin and box plots, multi-distribution Q-Q",
            lambda: _save_mpl(plot_p2_distribution_explorer(ctx), ctx.outdir,
                              "P2_seaborn_distribution",
                              "Distribution explorer for the annual peaks"),
        ),
        (
            "P3_flood_frequency_curves",
            "Every fitted candidate with the Bulletin 17C LP3 bootstrap band "
            "and the observed record",
            lambda: _save_plotly(plot_p3_frequency_curves(ctx), ctx.outdir,
                                 "P3_flood_frequency_curves",
                                 "Flood frequency curves for every candidate"),
        ),
        (
            "P4a_pairplot",
            "Pairplot of the five most important leakage-safe features "
            "against the peak, coloured by decade",
            lambda: _save_mpl(plot_p4a_pairplot(ctx), ctx.outdir, "P4a_pairplot",
                              "Pairplot of the top leakage-safe features"),
        ),
        (
            "P4b_jointplot",
            "Joint distribution of the one-year lag and the peak, with the "
            "Spearman coefficient",
            lambda: _save_mpl(plot_p4b_jointplot(ctx), ctx.outdir, "P4b_jointplot",
                              "Joint distribution of lag-1 and peak discharge"),
        ),
        (
            "P4c_correlation_heatmap",
            "Annotated Pearson correlation matrix of the modelling features "
            "and the peak discharge",
            lambda: _save_mpl(plot_p4c_heatmap(ctx), ctx.outdir,
                              "P4c_correlation_heatmap",
                              "Correlation matrix of the modelling features"),
        ),
        (
            "P5_monte_carlo_uncertainty_fan",
            "Record-length uncertainty from independent simulated records, "
            "with the simulated estimates at 10, 100 and 1000 years",
            lambda: _save_plotly(plot_p5_monte_carlo_uncertainty(ctx), ctx.outdir,
                                 "P5_monte_carlo_uncertainty_fan",
                                 "Monte Carlo record-length uncertainty fan"),
        ),
        (
            "P6_bayesian_mcmc_posteriors",
            "MCMC marginals, joint density, predictive design quantiles and a "
            "trace, stamped with the convergence verdict",
            lambda: _save_plotly(plot_p6_mcmc_posteriors(ctx), ctx.outdir,
                                 "P6_bayesian_mcmc_posteriors",
                                 "Bayesian MCMC posteriors for the LP3 parameters"),
        ),
        (
            "P7_qq_probability_plots",
            "Q-Q plots for the six lowest-AICc candidates with their R-squared "
            "and acceptance state",
            lambda: _save_plotly(plot_p7_qq_plots(ctx), ctx.outdir,
                                 "P7_qq_probability_plots",
                                 "Q-Q probability plots for the best candidates"),
        ),
        (
            "P8_skewness_kurtosis",
            "Non-parametric bootstrap of skewness, kurtosis and CV, the Cs-Ck "
            "moment space, normal probability paper and decadal densities",
            lambda: _save_mpl(plot_p8_skewness_kurtosis(ctx), ctx.outdir,
                              "P8_skewness_kurtosis",
                              "Bootstrap skewness, kurtosis and CV"),
        ),
        (
            "P9_acf_pacf_hurst_lag",
            "ACF, PACF, lag plot, rescaled-range Hurst with its resolution "
            "verdict, Ljung-Box p-values and rolling variance",
            lambda: _save_plotly(plot_p9_dependence(ctx), ctx.outdir,
                                 "P9_acf_pacf_hurst_lag",
                                 "Dependence diagnostics with the Hurst verdict"),
        ),
        (
            "P10_distribution_ranking",
            "AIC, AICc, BIC and HQIC for every candidate, and Akaike weights "
            "over the accepted ones",
            lambda: _save_plotly(plot_p10_distribution_ranking(ctx), ctx.outdir,
                                 "P10_distribution_ranking",
                                 "Information criteria and Akaike weights"),
        ),
        (
            "P11_pot_gpd",
            "Mean residual life, GPD return levels, excess CDF and threshold "
            "stability, stamped with the acceptance verdict",
            lambda: _save_plotly(plot_p11_pot(ctx), ctx.outdir, "P11_pot_gpd",
                                 "Peaks-over-threshold diagnostics and verdict"),
        ),
        (
            "P12_ml_panel",
            "Out-of-fold predicted versus observed, RF feature importance, "
            "reconstruction with fold spread, residuals, R-squared and RMSE",
            lambda: _save_mpl(plot_p12_ml_panel(ctx), ctx.outdir, "P12_ml_panel",
                              "Leakage-safe machine-learning performance panel"),
        ),
        (
            "P13_publication_dashboard",
            "Nine-panel summary: series, frequency curves, ranking, design "
            "floods, Monte Carlo, L-moments, decades and feature importance",
            lambda: _save_plotly(plot_p13_dashboard(ctx), ctx.outdir,
                                 "P13_publication_dashboard",
                                 "Nine-panel publication dashboard"),
        ),
        (
            "P14_comprehensive",
            "Ridgeline by decade, Gumbel probability paper, box and swarm, "
            "ECDF against CDFs, design-flood heatmap and Monte Carlo band",
            lambda: _save_mpl(plot_p14_comprehensive(ctx), ctx.outdir,
                              "P14_comprehensive",
                              "Comprehensive statistical summary"),
        ),
    ]


def render_suite(
    bundle,
    design,
    fits,
    pot,
    outdir: str | Path,
    stats=None,
    lmoments=None,
    autocorrelation=None,
    lp3=None,
    monte_carlo=None,
    mcmc=None,
    machine_learning=None,
) -> list[FigureRecord]:
    """Write the notebook figure suite to ``outdir`` and return what was written.

    Every figure that cannot be built is recorded with ``FAILED`` in its caption
    and the rest are still written: a missing fan chart must not cost the
    reviewer the flood-frequency curve.
    """
    ctx = SuiteContext(
        bundle=bundle,
        design=design,
        fits=fits,
        pot=pot,
        stats=stats,
        lmoments=lmoments,
        autocorrelation=autocorrelation,
        lp3=lp3,
        monte_carlo=monte_carlo,
        mcmc=mcmc,
        machine_learning=machine_learning,
    )
    ctx.outdir = Path(outdir)
    ctx.outdir.mkdir(parents=True, exist_ok=True)
    records: list[FigureRecord] = []
    for name, caption, builder in _builders(ctx):
        try:
            records.append(builder())
        except Exception as error:  # noqa: BLE001 - report, do not abort
            log.error("figure %s failed: %s", name, error)
            meta = _cfg.figure_meta(name)
            records.append(
                FigureRecord(
                    name=name,
                    path=ctx.outdir / f"{name}.png",
                    caption=f"{caption} [FAILED: {error}]",
                    section=meta.section,
                    tier=meta.tier,
                    aspect=meta.aspect,
                    engine=meta.engine,
                    question=meta.question,
                )
            )
    return records


def write_gallery(
    records: list[FigureRecord], outdir: str | Path, title: str = "Figure gallery"
) -> dict[str, Any]:
    """Write the sectioned gallery and the single-frame contact sheet.

    Two pages, because they answer two different questions:

    ``index.html``
        one tile per figure, grouped into the sections of
        :data:`ffa_karad.config.CONFIG.doc_sections`, ordered by tier inside each
        section.  This is the page to read.

    ``all.html``
        every figure at full width in one frame, in tier order, for comparing
        panels against each other and for printing.

    Also writes ``figures.json``, the machine-readable manifest the mkdocs site
    and any downstream consumer read, so no page has to scrape HTML.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    usable = [r for r in records if r.path.exists()]
    sections: dict[str, list[FigureRecord]] = {}
    for record in usable:
        sections.setdefault(record.section or "Other", []).append(record)

    ordered_sections = [s for s in _cfg.CONFIG.doc_sections if s in sections]
    ordered_sections += [s for s in sorted(sections) if s not in ordered_sections]

    _write_index_page(usable, sections, ordered_sections, outdir, title)
    _write_single_frame_page(usable, outdir, title)

    # One shared plotly.js beside the pages keeps the interactive figures
    # working offline and on a file:// open.
    if _cfg.CONFIG.plotly_js == "directory":
        _ensure_plotly_bundle(outdir)

    manifest = {
        "title": title,
        "generated": _cfg.run_timestamp(),
        "figures": [r.to_dict() for r in usable],
        "sections": {
            s: [r.name for r in sorted(sections[s], key=lambda r: (r.tier, r.name))]
            for s in ordered_sections
        },
        "counts": {
            "figures": len(usable),
            "interactive": sum(1 for r in usable if r.interactive),
            "tier1": sum(1 for r in usable if r.tier == 1),
            "tier2": sum(1 for r in usable if r.tier == 2),
            "tier3": sum(1 for r in usable if r.tier == 3),
        },
    }
    (outdir / "figures.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info(
        "wrote gallery: %d figures (%d interactive) across %d sections",
        manifest["counts"]["figures"],
        manifest["counts"]["interactive"],
        len(ordered_sections),
    )
    return manifest


def _rel(record: FigureRecord, outdir: Path, suffix: str) -> str:
    return os.path.relpath(record.path.with_suffix(suffix), outdir).replace(os.sep, "/")


def _tile(record: FigureRecord, outdir: Path) -> str:
    meta = record
    links = [
        f'<a href="{_esc(_rel(record, outdir, ".html"))}">open page</a>',
        f'<a href="{_esc(_rel(record, outdir, ".png"))}" download>PNG</a>',
    ]
    return f"""<figure class="tile tier-{meta.tier}">
<p class="tag tier-{meta.tier}">tier {meta.tier} &#183; {_esc(meta.engine)}</p>
<a href="{_esc(_rel(record, outdir, ".html"))}">
<img src="{_esc(_rel(record, outdir, ".png"))}" alt="{_esc(record.caption)}"
 loading="lazy"></a>
<figcaption>
<p class="name">{_esc(record.name)}</p>
<p class="q">{_esc(record.question)}</p>
<p class="cap">{_esc(record.caption)}</p>
<p class="links">{" &#183; ".join(links)}</p>
</figcaption></figure>"""


def _write_index_page(
    records: list[FigureRecord],
    sections: dict[str, list[FigureRecord]],
    ordered_sections: list[str],
    outdir: Path,
    title: str,
) -> Path:
    nav = " &#183; ".join(
        f'<a href="#{_slug(s)}">{_esc(s)}</a>' for s in ordered_sections
    )
    body = []
    for section in ordered_sections:
        tiles = "\n".join(
            _tile(r, outdir)
            for r in sorted(sections[section], key=lambda r: (r.tier, r.name))
        )
        body.append(
            f'<h2 id="{_slug(section)}">{_esc(section)}</h2>\n'
            f'<div class="grid">\n{tiles}\n</div>'
        )
    n_int = sum(1 for r in records if r.interactive)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{_PAGE_CSS}</style></head>
<body><main>
<h1>{_esc(title)}</h1>
<p class="cap">{len(records)} figures &#183; {n_int} interactive (Plotly) &#183;
generated {_esc(_cfg.run_timestamp())} &#183;
<a href="all.html">single frame</a> &#183;
<a href="figures.json">figures.json</a></p>
<nav class="sections">{nav}</nav>
{chr(10).join(body)}
</main></body></html>
"""
    path = outdir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _write_single_frame_page(
    records: list[FigureRecord], outdir: Path, title: str
) -> Path:
    """Every figure full width, in tier order, in one scrollable frame."""
    blocks = []
    for record in sorted(records, key=lambda r: (r.tier, r.section, r.name)):
        src = _rel(record, outdir, ".html" if record.interactive else ".png")
        blocks.append(
            f'<section id="{_slug(record.name)}">'
            f'<p class="tag tier-{record.tier}">tier {record.tier} &#183; '
            f"{_esc(record.section)}</p>"
            f"<h2>{_esc(record.name)}</h2>"
            f'<p class="q">{_esc(record.question)}</p>'
            + (
                f'<iframe src="{_esc(src)}" loading="lazy" '
                'style="width:100%;height:720px;border:1px solid var(--line);'
                'border-radius:4px;background:#fff"></iframe>'
                if record.interactive
                else f'<img src="{_esc(src)}" alt="{_esc(record.caption)}">'
            )
            + f'<p class="cap">{_esc(record.caption)}</p></section>'
        )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} &#183; all figures</title>
<style>{_PAGE_CSS}
section{{margin:0 0 40px;padding-bottom:24px;border-bottom:1px solid var(--line)}}
</style></head>
<body><main>
<p class="crumbs"><a href="index.html">&#8592; Sectioned gallery</a></p>
<h1>All figures, one frame</h1>
<p class="cap">Ordered by editorial tier, then section. Tier 1 is the
submission; tier 3 is exploratory.</p>
{chr(10).join(blocks)}
</main></body></html>
"""
    path = outdir / "all.html"
    path.write_text(page, encoding="utf-8")
    return path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text)).strip("-").lower()


def _ensure_plotly_bundle(outdir: Path) -> None:
    """Write one ``plotly.min.js`` next to the pages, if it is not there yet."""
    target = outdir / "plotly.min.js"
    if target.exists():
        return
    try:
        bundle = pio.get_plotlyjs()
    except Exception as error:  # noqa: BLE001 - cosmetic, do not fail the run
        log.warning("could not write plotly.min.js: %s", error)
        return
    target.write_text(bundle, encoding="utf-8")
    log.info("wrote %s (%.1f MB)", target.name, len(bundle) / 1e6)


__all__ = [
    "SuiteContext",
    "plot_p1_time_series",
    "plot_p2_distribution_explorer",
    "plot_p3_frequency_curves",
    "plot_p4a_pairplot",
    "plot_p4b_jointplot",
    "plot_p4c_heatmap",
    "plot_p5_monte_carlo_uncertainty",
    "plot_p6_mcmc_posteriors",
    "plot_p7_qq_plots",
    "plot_p8_skewness_kurtosis",
    "plot_p9_dependence",
    "plot_p10_distribution_ranking",
    "plot_p11_pot",
    "plot_p12_ml_panel",
    "plot_p13_dashboard",
    "plot_p14_comprehensive",
    "render_suite",
    "write_gallery",
]