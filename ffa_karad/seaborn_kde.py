"""
Kernel density estimate of the annual-peak record, with an explicit bandwidth.

The 2025 notebook shipped a density built with an unstated ``bw_adjust`` and no
statement of what it was estimating, and the resulting curve was used to argue
the record was log-normal.  A KDE cannot settle that: it is a smoothed version
of 57 points, its width is a free parameter, and a bandwidth small enough to
show structure also shows sampling noise.  So the bandwidth is declared here,
the support is clipped to the data range (``kde_cut = 0``) rather than extended
into a tail the record says nothing about, and the empirical observations are
drawn as a rug so the reader can see how little there is underneath.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display on a reviewer's PC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sps

from . import config as _cfg
from . import util

log = util.get_logger("seaborn_kde")


def kde_frame(
    values: Sequence[float] | np.ndarray, n_grid: int = 400, transform: str = "log"
) -> pd.DataFrame:
    """Grid of ``x``, density and the empirical CDF at each grid point.

    ``transform="log"`` estimates the density of ``ln Q``, which is the
    transform the flood-frequency method assumes; ``"raw"`` estimates the
    density of the discharge itself.  Both are provided because the first
    answers "is the record normal in log space?" and the second answers "how
    much probability mass sits near the design floods?", and they do not look
    the same.
    """
    x = util.as_float_array(values)
    if transform not in {"log", "raw"}:
        raise ValueError(
            f"seaborn_kde: transform must be 'log' or 'raw', got {transform!r}"
        )
    data = np.log(x) if transform == "log" else x
    cut = float(_cfg.CONFIG.kde_cut)
    grid = np.linspace(data.min(), data.max(), int(n_grid))
    # seaborn's estimator is the one used everywhere else, so the numbers here
    # and the curve in :func:`plot_kde` cannot drift apart.  It is drawn on a
    # throwaway axes purely because there is no public "fit only" entry point.
    scratch, ax = plt.subplots()
    try:
        sns.kdeplot(
            data=data,
            bw_adjust=float(_cfg.CONFIG.kde_bw_adjust),
            cut=cut,
            ax=ax,
            warn_singular=False,
        )
        # seaborn returns the Axes; the last artist it drew is the evaluated
        # density curve, which is what the numeric frame has to agree with.
        line = [a for a in ax.get_lines() if a.get_xdata().size >= 2][-1]
        support = np.asarray(line.get_xdata(), dtype=float)
        density = np.asarray(line.get_ydata(), dtype=float)
        grid = np.linspace(grid.min(), grid.max(), int(n_grid))
        density = np.interp(grid, support, density)
    finally:
        plt.close(scratch)
    return pd.DataFrame(
        {
            "x": grid,
            "value_cumecs": np.exp(grid) if transform == "log" else grid,
            "density": density,
            "ecdf": np.searchsorted(np.sort(data), grid, side="right") / data.size,
        }
    )


def plot_kde(
    values: Sequence[float] | np.ndarray,
    ax: plt.Axes | None = None,
    transform: str = "log",
    label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """KDE of the record with a rug of the observations and the mean marked."""
    x = util.as_float_array(values)
    frame = kde_frame(x, transform=transform)
    ax = ax or plt.subplots(figsize=(8.0, 5.0))[1]
    scale = "ln " + "Q" if transform == "log" else "Q"
    values_for_rug = np.log(x) if transform == "log" else x

    sns.kdeplot(
        data=values_for_rug,
        bw_adjust=float(_cfg.CONFIG.kde_bw_adjust),
        cut=float(_cfg.CONFIG.kde_cut),
        fill=True,
        alpha=0.25,
        label=label
        or f"KDE (bw_adjust={_cfg.CONFIG.kde_bw_adjust:g}, "
        f"cut={_cfg.CONFIG.kde_cut:g})",
        ax=ax,
    )
    ax.plot(
        frame["x"],
        frame["density"],
        color="#1f4e79",
        linewidth=1.6,
        label="evaluated density",
    )
    ax.plot(
        values_for_rug,
        np.zeros_like(values_for_rug),
        "|",
        color="#333333",
        markersize=7,
        alpha=_cfg.CONFIG.rug_alpha,
        label="annual maxima",
    )
    ax.axvline(
        float(np.mean(values_for_rug)),
        color="#b03a2e",
        linestyle="--",
        linewidth=1.2,
        label=f"mean {float(np.mean(x)):,.0f} m3/s",
    )
    ax.set_xlabel(f"{scale} (m3/s)" if transform == "raw" else f"{scale}")
    ax.set_ylabel("density")
    ax.set_title(f"Annual-peak density, {x.size} water years")
    ax.legend(frameon=True, fontsize="small")
    ax.grid(alpha=0.25)
    fig = ax.get_figure()
    log.debug("KDE drawn for %d observations in %s space", x.size, transform)
    return fig, ax


def plot_normality_comparison(
    values: Sequence[float] | np.ndarray, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Empirical density against the fitted log-normal and the fitted LP3.

    The three curves use the same data and the same axis so the disagreement at
    the upper tail is visible.  If the record were log-normal, the empirical
    line and the fitted log-normal line would coincide everywhere; the visible
    gap at the right-hand end is the evidence that the fit is not adequate in
    the part of the distribution the design floods depend on.
    """
    from . import skewness_limits as _skew

    x = util.as_float_array(values)
    log_q = np.log(x)
    fit = _skew.lp3_fit(x)
    ax = ax or plt.subplots(figsize=(8.0, 5.0))[1]
    grid = np.linspace(log_q.min(), log_q.max(), 300)

    sns.kdeplot(
        data=log_q,
        bw_adjust=float(_cfg.CONFIG.kde_bw_adjust),
        cut=float(_cfg.CONFIG.kde_cut),
        ax=ax,
        color="#555555",
        label="empirical (KDE)",
        warn_singular=False,
    )
    normal = sps.norm.pdf(grid, fit.mean_log, fit.sd_log)
    ax.plot(
        grid,
        normal,
        color="#1f4e79",
        linewidth=1.8,
        label=f"log-normal fit (ln Q ~ N({fit.mean_log:.3f}, {fit.sd_log:.3f}))",
    )
    lp3 = sps.pearson3.pdf(grid, fit.cs_log, loc=fit.mean_log, scale=fit.sd_log)
    ax.plot(
        grid,
        lp3,
        color="#1e8449",
        linestyle="--",
        linewidth=1.8,
        label=f"LP3 fit (Cs = {fit.cs_log:.3f})",
    )
    ax.set_xlabel("ln Q")
    ax.set_ylabel("density")
    ax.set_title("Annual peaks in log space: empirical against two fitted families")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)
    return ax.get_figure(), ax


def summarise_kde(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Numbers a reviewer needs to judge the density, not just look at it."""
    from scipy import stats as sps

    x = util.as_float_array(values)
    log_q = np.log(x)
    shapiro = sps.shapiro(log_q)
    frame = kde_frame(x)
    return {
        "n": int(x.size),
        "bandwidth_adjust": float(_cfg.CONFIG.kde_bw_adjust),
        "kde_support_cut": float(_cfg.CONFIG.kde_cut),
        "log_q_skewness": float(sps.skew(log_q, bias=False)),
        "log_q_kurtosis_excess": float(sps.kurtosis(log_q, fisher=True, bias=False)),
        "shapiro_wilk_W": float(shapiro.statistic),
        "shapiro_wilk_p": float(shapiro.pvalue),
        "log_q_normal_adequacy": (
            "rejected" if shapiro.pvalue < _cfg.CONFIG.alpha else "not rejected"
        ),
        "max_density_x": float(frame.loc[frame["density"].idxmax(), "x"]),
        "note": (
            "the density is a smoothed view of 57 points; it describes the "
            "body of the record and is not evidence about the tail"
        ),
    }


__all__ = ["kde_frame", "plot_kde", "plot_normality_comparison", "summarise_kde"]
