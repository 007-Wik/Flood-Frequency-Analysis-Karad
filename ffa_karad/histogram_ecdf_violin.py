"""
Three views of the annual-peak record that answer different questions.

* **Histogram** -- how the 57 peaks are distributed.  Counts, not a density, so
  the reader can see that there are 57 points and not lose track of n.
* **ECDF** -- the empirical distribution function as a step function, with the
  fitted LP3 and log-normal curves laid over it.  This is the honest way to show
  fit quality: it makes the gap at the upper tail visible and it cannot be
  smoothed away.
* **Violin** -- the shape of the distribution, which is worth one panel because
  the skewness of the record is what decides whether LP3 is needed.

The 2025 notebook drew a histogram with 30 bins over 57 points, which puts two
or three observations in a bin and produces a shape nobody should interpret.  The
bin count here is derived from Sturges' rule and the *raw* values are always
shown as points on top of the histogram, so no bin can misrepresent anything.
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as _cfg
from . import util

log = util.get_logger("histogram_ecdf_violin")


def sturges_bins(n: int) -> int:
    """Sturges' bin count, ``1 + log2(n)``.

    The rule of thumb for a *sample*: a histogram should not have more bins than
    the data can support.  Thirty bins over 57 observations is a shape drawn out
    of two points per bin, and it was the kind of picture that made the record
    look bimodal in the 2025 notebook.
    """
    return max(1, int(math.ceil(1.0 + math.log2(max(int(n), 2)))))


def histogram_frame(
    values: Sequence[float] | np.ndarray, bins: int | None = None
) -> pd.DataFrame:
    """Bin counts and edges for the annual peaks."""
    x = util.as_float_array(values)
    n_bins = sturges_bins(x.size) if bins is None else int(bins)
    counts, edges = np.histogram(x, bins=n_bins)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_mid": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
        }
    )


def plot_histogram(
    values: Sequence[float] | np.ndarray, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of the annual peaks with every observation shown."""
    x = util.as_float_array(values)
    ax = ax or plt.subplots(figsize=(8.0, 5.0))[1]
    frame = histogram_frame(x)
    ax.bar(
        frame["bin_mid"],
        frame["count"],
        width=(frame["bin_right"] - frame["bin_left"]) * 0.95,
        color="#aed6f1",
        edgecolor="#1f4e79",
        linewidth=0.8,
        label=f"counts (Sturges: {len(frame)} bins for n = {x.size})",
    )
    ax.plot(
        x,
        np.zeros_like(x),
        "|",
        color="#333333",
        markersize=8,
        alpha=_cfg.CONFIG.rug_alpha,
        label="each annual maximum",
    )
    ax.axvline(
        float(np.mean(x)),
        color="#b03a2e",
        linestyle="--",
        linewidth=1.2,
        label=f"mean {np.mean(x):,.0f} m3/s",
    )
    ax.axvline(
        float(np.max(x)),
        color="#7d3c98",
        linestyle=":",
        linewidth=1.5,
        label=f"observed maximum {np.max(x):,.0f} m3/s",
    )
    ax.set_xlabel("annual maximum discharge (m3/s)")
    ax.set_ylabel("number of water years")
    ax.set_title(
        f"Annual-peak histogram, {x.size} water years, {int(x.min()):,}-"
        f"{int(x.max()):,} m3/s"
    )
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25, axis="y")
    return ax.get_figure(), ax


def ecdf_frame(values: Sequence[float] | np.ndarray) -> pd.DataFrame:
    """Step-function ECDF: one row per observation, ascending."""
    x = np.sort(util.as_float_array(values))
    return pd.DataFrame(
        {
            "value_cumecs": x,
            "ecdf": np.arange(1, x.size + 1) / x.size,
            "exceedance": 1.0 - np.arange(1, x.size + 1) / x.size,
        }
    )


def plot_ecdf(
    values: Sequence[float] | np.ndarray, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Empirical CDF against the fitted LP3 and log-normal curves, log-x."""
    from scipy import stats as sps

    from . import skewness_limits as _skew

    x = util.as_float_array(values)
    frame = ecdf_frame(x)
    fit = _skew.lp3_fit(x)
    ax = ax or plt.subplots(figsize=(8.0, 5.0))[1]
    grid = np.linspace(float(np.log(x.min())), float(np.log(x.max())), 400)
    ax.step(
        frame["value_cumecs"],
        frame["ecdf"],
        where="post",
        color="#333333",
        linewidth=1.8,
        label=f"empirical CDF (n = {x.size})",
    )
    ax.plot(
        np.exp(grid),
        sps.pearson3.cdf(grid, fit.cs_log, loc=fit.mean_log, scale=fit.sd_log),
        color="#1e8449",
        linewidth=1.6,
        label=f"LP3 (Cs = {fit.cs_log:.3f})",
    )
    ax.plot(
        np.exp(grid),
        sps.norm.cdf(grid, fit.mean_log, fit.sd_log),
        color="#1f4e79",
        linestyle="--",
        linewidth=1.4,
        label="log-normal (same mean and s.d.)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("discharge (m3/s, log scale)")
    ax.set_ylabel("non-exceedance probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Empirical distribution against fitted families")
    ax.legend(fontsize="small", loc="lower right")
    ax.grid(alpha=0.25, which="both")
    return ax.get_figure(), ax


def plot_violin(
    values: Sequence[float] | np.ndarray, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """Violin of the record in log space, with the quartiles marked."""
    x = util.as_float_array(values)
    ax = ax or plt.subplots(figsize=(4.5, 5.0))[1]
    frame = pd.DataFrame({"ln Q": np.log(x)})
    sns.violinplot(
        data=frame,
        x="ln Q",
        inner="quartile",
        cut=0,
        density_norm="width",
        color="#aed6f1",
        ax=ax,
    )
    ax.set_xlabel("ln Q")
    ax.set_ylabel("")
    ax.set_title(f"Shape of the record (n = {x.size})")
    ax.grid(alpha=0.25, axis="x")
    return ax.get_figure(), ax


def plot_all(
    values: Sequence[float] | np.ndarray, axes: Sequence[plt.Axes] | None = None
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Histogram, ECDF and violin in one figure."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    plot_histogram(values, axes[0])
    plot_ecdf(values, axes[1])
    plot_violin(values, axes[2])
    fig = axes[0].get_figure()
    fig.tight_layout()
    return fig, list(axes)


__all__ = [
    "sturges_bins",
    "histogram_frame",
    "ecdf_frame",
    "plot_histogram",
    "plot_ecdf",
    "plot_violin",
    "plot_all",
]
