"""
The notebook's visual language, as one importable object.

The 2025 baseline notebook carried its palette, theme and layout decisions as
cell-local globals (``COLORS``, ``DECADE_COLORS``, ``sns.set_theme(...)``,
``figsize=(16, 10)``, panel titles of the form ``(a) ...``).  That worked while
the figures lived in one notebook.  It does not survive being generated from a
package that runs headless, from tests, and from a pipeline that has to rebuild
the same figure twice and get the same picture.

So the decisions are kept, but they are kept **here**: one place that defines
the palette, the theme, the panel-labelling convention and the save settings.
Every figure in :mod:`ffa_karad.visualization`,
:mod:`ffa_karad.seaborn_kde` and :mod:`ffa_karad.histogram_ecdf_violin` draws
itself from this module, so a change to the palette changes every figure at
once and no figure can drift into its own private colour scheme.

What is preserved from the notebook
-----------------------------------
- the unified distribution palette, one colour per named distribution, so the
  Gumbel curve is the same blue in every figure;
- the decadal palette, light to dark in time order;
- ``seaborn`` ``whitegrid`` with the ``deep`` palette;
- the ``(a) ... (b) ...`` panel labels and the multi-panel grid sizes;
- ``dpi=180``, ``bbox_inches="tight"``.

What is deliberately not preserved
----------------------------------
The notebook's axis, transform and estimator mistakes.  Its frequency curves
were plotted with discharge on a *linear* y-axis against return periods from
1.01 to 1000, which visually compresses every flood above the 100-year mark
into the top decile of the plot; its dashboard labelled a L-moment axis by
concatenating the string ``"Log"`` onto it; its decadal panels grouped on a
label that never existed in the data.  Those are corrected here rather than
copied.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import util

log = util.get_logger("plot_style")

#: One colour per named distribution, as in the notebook.  ``obs``, ``ci_fill``
#: and ``highlight`` are not distributions but share the vocabulary.
COLORS: dict[str, str] = {
    "Gumbel (EV1)": "#1565C0",
    "Log-Normal (LN2)": "#2E7D32",
    "Pearson III (P3)": "#6A1B9A",
    "Log-Pearson III": "#E65100",
    "GEV": "#B71C1C",
    "Weibull (EV3)": "#00695C",
    "Generalised Pareto": "#37474F",
    "Log-Logistic": "#880E4F",
    "Gen. Logistic": "#1A237E",
    "Burr Type XII": "#33691E",
    "Kappa-4": "#4A148C",
    "Exponential": "#795548",
    "obs": "#212121",
    "ci_fill": "rgba(21,101,192,0.15)",
    "highlight": "#FF6F00",
}

#: Decade colours, lightest first, so a time-ordered legend reads as a ramp.
DECADE_COLORS: list[str] = [
    "#E3F2FD",
    "#90CAF9",
    "#1565C0",
    "#0D47A1",
    "#1A237E",
    "#4A148C",
]

#: Short names the fitting module uses, mapped onto the notebook's palette keys.
NAME_ALIASES: dict[str, str] = {
    "Gumbel": "Gumbel (EV1)",
    "EV1": "Gumbel (EV1)",
    "LN2": "Log-Normal (LN2)",
    "LN3": "Log-Normal (LN2)",
    "LogNormal": "Log-Normal (LN2)",
    "LP3": "Log-Pearson III",
    "Pearson3": "Pearson III (P3)",
    "P3": "Pearson III (P3)",
    "Weibull": "Weibull (EV3)",
    "Lognormal": "Log-Normal (LN2)",
    "Loglogistic": "Log-Logistic",
    "GLO": "Gen. Logistic",
    "GNO": "Gen. Logistic",
    "GEV": "GEV",
}

#: Line style per distribution, mirroring the notebook's dash vocabulary.
LINE_STYLES: dict[str, str] = {
    "Gumbel (EV1)": "-",
    "Log-Normal (LN2)": "--",
    "Pearson III (P3)": "-.",
    "Log-Pearson III": "-",
}

DPI = 180
FIGSIZE_PANEL = (16.0, 10.0)
FIGSIZE_WIDE = (18.0, 14.0)
FIGSIZE_SINGLE = (9.0, 5.5)

_ALARM = "#EF5350"
_WARN = "#FF9800"
_OK = "#4CAF50"
_COOL = "#1565C0"
_COOL_LIGHT = "#90CAF9"
_COOL_DARK = "#0D47A1"
_NEUTRAL = "#37474F"
_NEUTRAL_LIGHT = "#B0BEC5"


def apply_theme() -> None:
    """Apply the notebook's seaborn theme.  Idempotent."""
    sns.set_theme(style="whitegrid", palette="deep")


def distribution_color(name: str) -> str:
    """Colour for a distribution name, matching the notebook's palette.

    Unknown names get a stable colour from the ``husl`` wheel rather than a
    grey, so a newly added candidate is visible rather than invisible.
    """
    key = NAME_ALIASES.get(name, name)
    if key in COLORS:
        return COLORS[key]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    red, green, blue = sns.color_palette("husl", 12)[int(digest[:8], 16) % 12]
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


def distribution_style(name: str) -> str:
    return LINE_STYLES.get(NAME_ALIASES.get(name, name), "-")


def decade_colors(n: int) -> list[str]:
    """``n`` decade colours, interpolating the ramp when more are needed."""
    if n <= len(DECADE_COLORS):
        return DECADE_COLORS[:n]
    return [tuple(color) for color in sns.color_palette("husl", n)]


def decade_label(start_year: int) -> str:
    """``1976`` -> ``"1970s"``."""
    return f"{(int(start_year) // 10) * 10}s"


def bar_color_for_value(value: float, values: np.ndarray) -> str:
    """Colour one annual peak by where it sits in the record.

    Red above the 90th percentile, amber above the 75th, blue otherwise.  This
    is the notebook's rule and it makes the extremes findable without reading
    the axis.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return COLORS["obs"]
    if value > np.percentile(values, 90):
        return _ALARM
    if value > np.percentile(values, 75):
        return _WARN
    return _COOL


def anomaly_color(z: float) -> str:
    """Colour one standardised anomaly."""
    if z > 1.5:
        return _ALARM
    if z > 0.0:
        return _WARN
    return _COOL


def panel_title(letter: str, text: str) -> str:
    """``panel_title("a", "Histogram")`` -> ``"(a) Histogram"``."""
    return f"({letter}) {text}"


@dataclasses.dataclass
class PanelGrid:
    """A matplotlib grid with the notebook's spacing conventions."""

    rows: int
    cols: int
    figsize: tuple[float, float]
    height_ratios: list[float] | None = None
    hspace: float | None = None
    wspace: float | None = None

    def build(self):
        kwargs: dict[str, Any] = {"figsize": self.figsize}
        if self.height_ratios:
            kwargs["gridspec_kw"] = {
                "height_ratios": self.height_ratios,
                "hspace": self.hspace or 0.32,
                "wspace": self.wspace or 0.26,
            }
        fig, axes = plt.subplots(self.rows, self.cols, **kwargs)
        axes = np.atleast_1d(axes)
        return fig, axes

    @staticmethod
    def letter(index: int) -> str:
        """0 -> ``"a"``, 25 -> ``"z"``."""
        return chr(ord("a") + index)


def suptitle(fig: plt.Figure, text: str, subtitle: str | None = None) -> None:
    """Bold suptitle, with an optional smaller subtitle line beneath it."""
    if subtitle:
        fig.suptitle(f"{text}\n{subtitle}", fontsize=14, fontweight="bold", y=0.995)
    else:
        fig.suptitle(text, fontsize=14, fontweight="bold", y=0.99)


def tidy(axes, legend: bool = True, legend_size: int = 8) -> None:
    """Apply the notebook's finishing touches to a set of axes."""
    for ax in np.atleast_1d(axes).ravel():
        if not hasattr(ax, "set_title"):
            continue
        if legend and ax.get_legend() is not None:
            ax.get_legend().set_fontsize(legend_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(9)
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(10)


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Save at the notebook's settings and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def reduced_variate(periods: np.ndarray) -> np.ndarray:
    """Gumbel reduced variate ``y = -ln(-ln(1 - 1/T))``.

    Used by the probability-paper panels, where the x-axis is linear in the
    reduced variate and a second axis converts it back to return period.
    """
    t = np.asarray(periods, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(-np.log(1.0 - 1.0 / t))


def period_to_reduced(period: float) -> float:
    return float(reduced_variate(np.array([period]))[0])


apply_theme()


__all__ = [
    "COLORS",
    "DECADE_COLORS",
    "NAME_ALIASES",
    "LINE_STYLES",
    "DPI",
    "FIGSIZE_PANEL",
    "FIGSIZE_WIDE",
    "FIGSIZE_SINGLE",
    "PanelGrid",
    "anomaly_color",
    "apply_theme",
    "bar_color_for_value",
    "decade_colors",
    "decade_label",
    "distribution_color",
    "distribution_style",
    "panel_title",
    "period_to_reduced",
    "reduced_variate",
    "save",
    "suptitle",
    "tidy",
]
