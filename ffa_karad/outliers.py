"""
Outlier screening: identify candidate low outliers, but do not delete anything.

Three independent screens are run and reported side by side, because none of
them is decisive on its own:

* **Grubbs' test** -- the classical two-sided test for a single outlier in a
  normal sample.  It assumes normality, which annual maxima are not, and it
  cannot tell a *low* observation from an error.  Reported with that caveat.
* **Dixon's Q** -- designed for small samples and, unlike Grubbs, directional.
  The low-outlier critical values are used only for observations below the mean,
  which is the direction relevant to a flood record.
* **Rosenblatt's test (ROS)** -- the recommended screen in Bulletin 17C for
  flood-frequency data.  It works on the *normal scores* of the log record, so
  it does not assume the record is normally distributed.

**Nothing is deleted.**  A low annual maximum is the expected consequence of a
dry year, not a blunder, and deleting it makes the 2- and 5-year quantiles too
large.  The 2025 notebook dropped the lowest peak and reported a 2-year flood of
about 8,000 m3/s on a record whose second-largest value is 5,895 m3/s; that
number was an artefact of the deletion.  Every result here carries
``action_required`` and, where a value is questionable, ``retained``: the record
is reported as measured, and the judgement is left to a reviewer who can see the
water level, the rating curve and the field records.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import data_processing as _dp
from . import util

log = util.get_logger("outliers")


@dataclasses.dataclass
class OutlierResult:
    method: str
    assumptions: str
    flagged: list[int]
    test_statistic: float
    critical_value: float
    p_value: float
    threshold: float | None
    weakest_low_value: float | None
    passed: bool | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class OutlierReport:
    n: int
    grubbs: OutlierResult
    dixon_q: OutlierResult
    ros: pd.DataFrame
    retained: pd.DataFrame
    candidates: list[dict[str, Any]]
    consensus: str
    action_required: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "methods": {
                "grubbs": self.grubbs.to_dict(),
                "dixon_q": self.dixon_q.to_dict(),
            },
            "ros": self.ros.to_dict(orient="records"),
            "retained": self.retained.to_dict(orient="records"),
            "candidates": self.candidates,
            "consensus": self.consensus,
            "action_required": self.action_required,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Grubbs
# ---------------------------------------------------------------------------


def grubbs(
    values: Sequence[float] | np.ndarray, alpha: float | None = None
) -> dict[str, Any]:
    """Two-sided Grubbs' test for a single outlier.

    .. math::
        G = \\frac{\\max_i |x_i - \\bar x|}{s}

    The critical value uses the finite-sample factor
    ``(n-1)/sqrt(n) * sqrt(t^2/(n-2+t^2))`` with
    ``t = t_{alpha/(2n), n-2}``.
    """
    x = util.as_float_array(values)
    alpha = _cfg.CONFIG.alpha if alpha is None else float(alpha)
    n = x.size
    if n < 4:
        return {"ok": False, "note": f"Grubbs' test needs at least 4 values, got {n}"}
    mean = float(x.mean())
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return {"ok": False, "note": "zero dispersion; no outlier is definable"}
    deviation = x - mean
    index = int(np.argmax(np.abs(deviation)))
    g = float(np.max(np.abs(deviation)) / sd)
    t = float(sps.t.ppf(1.0 - alpha / (2.0 * n), n - 2))
    critical = float((n - 1.0) / np.sqrt(n) * np.sqrt(t**2 / (n - 2.0 + t**2)))
    # Grubbs' statistic is a monotone function of the t statistic
    # ``t = G sqrt(n(n-2)) / sqrt((n-1)^2 - G^2 n)``; the two-sided p-value is
    # then ``n * 2 * P(T > t)``, the factor n accounting for the fact that the
    # most extreme of n observations is being tested.  This reproduces the
    # published critical values (n = 10 gives G = 2.2900) and returns 0.05 at
    # the critical value.
    t2 = (g**2 * n * (n - 2.0)) / ((n - 1.0) ** 2 - g**2 * n)
    p = (
        0.0
        if t2 <= 0
        else float(min(max(n * 2.0 * sps.t.sf(np.sqrt(t2), n - 2), 0.0), 1.0))
    )
    return {
        "ok": True,
        "index": index,
        "value": float(x[index]),
        "g": g,
        "critical_g": critical,
        "p_value": p,
        "flagged": bool(g > critical),
        "direction": "high" if deviation[index] > 0 else "low",
    }


# ---------------------------------------------------------------------------
# Dixon Q
# ---------------------------------------------------------------------------

#: Dixon's Q critical values for a *low* outlier, alpha = 0.05, r10.
DIXON_Q_LOW_05: dict[int, float] = {
    3: 0.941,
    4: 0.765,
    5: 0.642,
    6: 0.560,
    7: 0.507,
    8: 0.468,
    9: 0.437,
    10: 0.412,
    15: 0.338,
    20: 0.290,
    25: 0.256,
    30: 0.237,
}


def dixon_q_low(
    values: Sequence[float] | np.ndarray, alpha: float | None = None
) -> dict[str, Any]:
    """Dixon's Q for a low outlier.

    Uses the ``r10`` ratio ``(x2 - x1) / (x(n) - x1)``, which is the form
    Bulletin 17C applies to annual maxima.  Critical values are tabulated for
    ``alpha = 0.05``; any other level is refused rather than interpolated,
    because an interpolated critical value would be a number nobody audited.
    """
    x = np.sort(util.as_float_array(values))
    alpha = _cfg.CONFIG.alpha if alpha is None else float(alpha)
    n = x.size
    if n < 4:
        return {"ok": False, "note": f"Dixon's Q needs at least 4 values, got {n}"}
    if abs(alpha - 0.05) > 1e-12:
        return {
            "ok": False,
            "note": "tabulated low-outlier critical values are for alpha = 0.05 only",
        }
    critical = DIXON_Q_LOW_05.get(n)
    if critical is None:
        return {"ok": False, "note": f"no tabulated Dixon Q critical value for n = {n}"}
    spread = float(x[-1] - x[0])
    if spread <= 0:
        return {"ok": False, "note": "zero dispersion; no outlier is definable"}
    q = float((x[1] - x[0]) / spread)
    return {
        "ok": True,
        "value": float(x[0]),
        "q": q,
        "critical_q": float(critical),
        "p_value": float("nan"),
        "flagged": bool(q > critical),
    }


# ---------------------------------------------------------------------------
# Rosenblatt (normal-score) plot
# ---------------------------------------------------------------------------


def ros_plot(
    values: Sequence[float] | np.ndarray, plotting_position: str | None = None
) -> pd.DataFrame:
    """Rosenblatt plot coordinates for log-transformed annual maxima.

    The record is transformed to ``ln Q``, given normal scores by the selected
    plotting position, and the two are sorted and paired.  On the resulting
    paper, points lying on a straight line indicate a log-normal record; a
    systematic bow indicates skewness, and an isolated point off the line is a
    candidate outlier.
    """
    x = util.as_float_array(values)
    method = plotting_position or _cfg.CONFIG.plotting_position
    n = x.size
    scores = _dp.plotting_position(np.arange(1, n + 1), n, method)
    # Rank i pairs the i-th smallest log value with the i-th smallest normal
    # score; sorting by the value or by its log gives the same order.
    ordered = np.argsort(x, kind="stable")
    return pd.DataFrame(
        {
            "rank": np.arange(1, x.size + 1),
            "value_cumecs": x[ordered],
            "log_value": np.log(x)[ordered],
            "normal_score": np.sort(scores)[ordered],
        }
    )


def ros_outliers(
    values: Sequence[float] | np.ndarray,
    n_candidates: int = 2,
    plotting_position: str | None = None,
) -> pd.DataFrame:
    """Flag the ``n_candidates`` lowest points furthest from the ROS line.

    The reference line is the least-squares fit through the *central* bulk of
    the plot (the points between the first and last quartiles of normal score).
    Fitting through the extremes would let a bad point drag the line onto itself
    and hide the evidence.
    """
    frame = ros_plot(values, plotting_position)
    y = frame["log_value"].to_numpy(dtype=float)
    z = frame["normal_score"].to_numpy(dtype=float)
    interior = (z >= np.quantile(z, 0.25)) & (z <= np.quantile(z, 0.75))
    slope, intercept = np.polyfit(z[interior], y[interior], 1)
    frame["fitted_log_value"] = intercept + slope * z
    frame["residual"] = y - frame["fitted_log_value"]
    frame["std_residual"] = frame["residual"] / (frame["residual"].std(ddof=1) or 1.0)
    frame["is_candidate"] = False
    order = frame["std_residual"].abs().sort_values(ascending=False).index
    frame.loc[order[: max(int(n_candidates), 1)], "is_candidate"] = True
    return frame.sort_values("normal_score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(
    values: Sequence[float] | np.ndarray,
    years: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    n_candidates: int = 2,
) -> OutlierReport:
    """Run all three screens and report candidates without deleting anything."""
    x = util.as_float_array(values)
    if labels is None:
        labels = [
            str(y) for y in (years if years is not None else np.arange(1, x.size + 1))
        ]
    labels = [str(v) for v in labels]
    if len(labels) != x.size:
        raise ValueError(f"outliers: {len(labels)} labels for {x.size} values")

    alpha = _cfg.CONFIG.alpha
    g = grubbs(x, alpha)
    d = dixon_q_low(x, alpha)
    ros = ros_outliers(x, n_candidates=n_candidates)

    lowest = np.argsort(x)[: min(3, x.size)]
    retained = pd.DataFrame(
        {
            "label": [labels[i] for i in lowest],
            "value_cumecs": x[lowest],
            "log_value": np.log(x[lowest]),
            "rank_ascending": np.arange(1, lowest.size + 1),
            "retained": True,
            "reason": (
                "kept in the record: a low annual maximum is a dry year, not "
                "a measurement error, and deleting it inflates the 2- and "
                "5-year quantiles"
            ),
        }
    )

    candidates: list[dict[str, Any]] = []
    if g.get("ok") and g["flagged"]:
        candidates.append(
            {
                "label": labels[g["index"]],
                "value_cumecs": g["value"],
                "detected_by": ["Grubbs"],
                "direction": g["direction"],
                "severity": "two-sided; the test cannot say whether the point is an "
                "error or a genuine dry year",
            }
        )
    if d.get("ok") and d["flagged"]:
        candidates.append(
            {
                "label": labels[int(np.argmin(x))],
                "value_cumecs": float(np.min(x)),
                "detected_by": ["Dixon Q"],
                "direction": "low",
                "severity": "directional low-outlier screen",
            }
        )
    for _, row in ros.iterrows():
        if bool(row["is_candidate"]) and float(row["std_residual"]) < -1.5:
            label = labels[int(np.argmin(np.abs(x - row["value_cumecs"])))]
            candidates.append(
                {
                    "label": label,
                    "value_cumecs": float(row["value_cumecs"]),
                    "detected_by": ["Rosenblatt"],
                    "direction": "low",
                    "severity": f"{row['std_residual']:.2f} s.d. below the ROS line "
                    "fitted through the central bulk",
                }
            )

    ros_mask = ros["is_candidate"] & (
        ros["std_residual"].abs() > _cfg.CONFIG.ros_flag_sd
    )
    ros_flagged = bool(ros_mask.any())
    if ros_flagged:
        signs = set(np.sign(ros.loc[ros_mask, "residual"].to_numpy(dtype=float)))
        ros_tails = (
            "both" if len(signs) > 1 else ("the low" if signs == {-1.0} else "the high")
        )
        ros_detail = f"Rosenblatt flags {ros_tails} tails" + (
            "; flags in both tails at once indicate curvature of the "
            "log record rather than a bad observation"
            if ros_tails == "both"
            else ""
        )
    else:
        ros_tails = ros_detail = "none"
    flags = {
        "Grubbs": bool(g.get("ok") and g["flagged"]),
        "Dixon Q": bool(d.get("ok") and d["flagged"]),
        "Rosenblatt": ros_flagged,
    }
    if all(flags.values()):
        consensus = (
            "all three screens flag a candidate low outlier; review the "
            "water level, rating curve and field record for that year "
            "before any decision is taken"
        )
    elif not any(flags.values()):
        consensus = "no screen flags a low outlier; the record stands as measured"
    else:
        agreeing = [k for k, v in flags.items() if v]
        consensus = (
            f"{', '.join(agreeing)} flags a candidate but the others do "
            "not, which is the usual outcome; a single test on 57 points "
            "is weak evidence"
        )

    notes = [
        "No observation is deleted by this module.  Bulletin 17C screening "
        "exists to protect against a *recording* error, and the lowest annual "
        "maximum in a 57-year flood record is expected to be well below the mean.",
        "The 2025 notebook removed the lowest peak and reported a 2-year flood of "
        "about 8,000 m3/s, larger than the second-largest peak in the record "
        "(5,895 m3/s).  That number came from the deletion, not from the data.",
        f"The lowest value in this record is {float(np.min(x)):,.0f} m3/s "
        f"({labels[int(np.argmin(x))]}), {float(np.min(x) / np.mean(x)):.2f} times "
        "the mean; the second lowest is "
        f"{float(np.sort(x)[1]):,.0f} m3/s.",
        "Grubbs' test assumes normality and cannot distinguish a low observation "
        "from an error.  Dixon's Q is directional but tabulated only to n = 30, "
        "beyond which the critical value is not defined for a record of this "
        "length.  Rosenblatt works on normal scores of the log record and is the "
        "recommended screen for flood-frequency data.",
    ]
    if ros_detail:
        notes.append(ros_detail)
    log.info(
        "outlier screens: Grubbs %s, Dixon Q %s, ROS candidates %d; nothing deleted",
        "flags" if flags["Grubbs"] else "clear",
        "flags" if flags["Dixon Q"] else "clear",
        int(ros["is_candidate"].sum()),
    )
    return OutlierReport(
        n=int(x.size),
        grubbs=OutlierResult(
            method="Grubbs",
            assumptions="normal sample",
            flagged=[int(g["index"])] if g.get("ok") and g["flagged"] else [],
            test_statistic=float(g.get("g", float("nan"))),
            critical_value=float(g.get("critical_g", float("nan"))),
            p_value=float(g.get("p_value", float("nan"))),
            threshold=None,
            weakest_low_value=float(np.min(x)),
            passed=bool(g.get("ok")) and not g.get("flagged", False),
            note=str(g.get("note", "")),
        ),
        dixon_q=OutlierResult(
            method="Dixon Q (r10, low outlier)",
            assumptions="none beyond the tabulated critical value",
            flagged=[] if not (d.get("ok") and d["flagged"]) else [int(np.argmin(x))],
            test_statistic=float(d.get("q", float("nan"))),
            critical_value=float(d.get("critical_q", float("nan"))),
            p_value=float(d.get("p_value", float("nan"))),
            threshold=None,
            weakest_low_value=float(np.min(x)),
            passed=bool(d.get("ok")) and not d.get("flagged", False),
            note=str(d.get("note", "")),
        ),
        ros=ros,
        retained=retained,
        candidates=candidates,
        consensus=consensus,
        action_required=bool(candidates),
        notes=notes,
    )


def summarise(report: OutlierReport) -> str:
    """Plain-text outlier summary for the report."""
    lines = [
        "OUTLIER SCREENING  (nothing deleted)",
        f"  n = {report.n} annual maxima",
        "",
        f"  {'method':<26} {'statistic':>10} {'critical':>10} {'flags?':>8}",
    ]
    for result in (report.grubbs, report.dixon_q):
        flag = "yes" if result.flagged else "no"
        lines.append(
            f"  {result.method:<26} {result.test_statistic:10.3f} "
            f"{result.critical_value:10.3f} {flag:>8}"
        )
        if result.note:
            lines.append(f"      note: {result.note}")
    lines += ["", "  Rosenblatt candidates (normal-score plot):"]
    for row in report.ros[report.ros["is_candidate"]].itertuples():
        lines.append(
            f"    {row.value_cumecs:10,.0f} m3/s  ln Q {row.log_value:6.3f}  "
            f"normal score {row.normal_score:6.3f}  "
            f"{row.std_residual:+6.2f} s.d."
        )
    lines += [
        "",
        "  Lowest three, retained:",
    ]
    for row in report.retained.itertuples():
        lines.append(
            f"    {row.label:<12} {row.value_cumecs:10,.0f} m3/s  "
            f"ln Q {row.log_value:6.3f}"
        )
    lines += ["", f"  VERDICT: {report.consensus}"]
    if report.action_required:
        lines.append(
            "  ACTION: a reviewer must confirm or reject each candidate "
            "against the original water-level and rating-curve records."
        )
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in report.notes]
    return "\n".join(lines)


__all__ = [
    "OutlierResult",
    "OutlierReport",
    "grubbs",
    "dixon_q_low",
    "ros_plot",
    "ros_outliers",
    "run",
    "summarise",
    "DIXON_Q_LOW_05",
]
