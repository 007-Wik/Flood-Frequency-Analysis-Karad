"""Tests for :mod:`ffa_karad.outliers`.

The behaviour these tests protect is negative: nothing gets deleted, and a
synthetic outlier is found by the screen that claims to find it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffa_karad import data_processing as dp
from ffa_karad import outliers as o


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


# ---------------------------------------------------------------------------
# Grubbs
# ---------------------------------------------------------------------------


def test_grubbs_finds_a_planted_high_outlier():
    rng = np.random.default_rng(0)
    x = np.append(rng.normal(100.0, 5.0, 30), 400.0)
    out = o.grubbs(x)
    assert out["ok"] is True
    assert out["flagged"] is True
    assert out["direction"] == "high"
    assert out["index"] == 30
    assert 0.0 <= out["p_value"] <= 1.0


def test_grubbs_is_quiet_on_clean_data():
    rng = np.random.default_rng(1)
    out = o.grubbs(rng.normal(100.0, 5.0, 40))
    assert out["flagged"] is False


def test_grubbs_refuses_short_and_degenerate_input():
    assert o.grubbs([1.0, 2.0, 3.0])["ok"] is False
    assert o.grubbs([5.0] * 10)["ok"] is False


def test_grubbs_p_value_agrees_with_the_critical_value():
    """The p-value and the critical-value comparison must not disagree."""
    rng = np.random.default_rng(2)
    for n in (10, 25, 40):
        x = rng.normal(0.0, 1.0, n)
        out = o.grubbs(x)
        assert bool(out["flagged"]) == bool(out["p_value"] < 0.05)


# ---------------------------------------------------------------------------
# Dixon Q
# ---------------------------------------------------------------------------


def test_dixon_q_flags_a_planted_low_outlier():
    x = np.array([1.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 30.0])
    out = o.dixon_q_low(x)
    assert out["ok"] is True
    assert out["flagged"] is True
    assert out["value"] == pytest.approx(1.0)


def test_dixon_q_refuses_n_above_the_table():
    rng = np.random.default_rng(3)
    out = o.dixon_q_low(rng.normal(0.0, 1.0, 57))
    assert out["ok"] is False
    assert "no tabulated" in out["note"]


def test_dixon_q_refuses_other_alpha():
    out = o.dixon_q_low(np.arange(1.0, 11.0), alpha=0.01)
    assert out["ok"] is False
    assert "alpha = 0.05" in out["note"]


# ---------------------------------------------------------------------------
# Rosenblatt
# ---------------------------------------------------------------------------


def test_ros_line_is_close_for_a_lognormal_sample():
    rng = np.random.default_rng(4)
    x = np.exp(rng.normal(8.0, 0.4, 60))
    frame = o.ros_plot(x)
    assert len(frame) == 60
    assert frame["normal_score"].is_monotonic_increasing
    # On a Rosenblatt plot the normal score is a rank transform of the value,
    # so even a perfect log-normal sample gives a correlation a little below 1
    # (0.98 for n = 60); a skewed or heavy-tailed record falls away from it.
    assert frame["normal_score"].corr(frame["log_value"]) > 0.97


def test_ros_flags_an_planted_low_point(bundle):
    """The lowest real observation must show up as a negative residual."""
    frame = o.ros_outliers(bundle.q, n_candidates=3)
    lowest = frame.loc[frame["value_cumecs"].idxmin()]
    assert bool(lowest["is_candidate"])
    assert lowest["std_residual"] < 0


def test_ros_fit_uses_the_central_bulk():
    """The reference line must ignore the extreme points it is testing."""
    rng = np.random.default_rng(5)
    x = np.exp(rng.normal(8.0, 0.3, 60))
    frame = o.ros_outliers(np.append(x, 1e6))
    z = frame["normal_score"].to_numpy(dtype=float)
    y = frame["log_value"].to_numpy(dtype=float)
    interior = (z >= np.quantile(z, 0.25)) & (z <= np.quantile(z, 0.75))
    slope, intercept = np.polyfit(z[interior], y[interior], 1)
    assert np.allclose(frame["fitted_log_value"], intercept + slope * z)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def test_run_deletes_nothing(bundle):
    report = o.run(bundle.q, labels=bundle.frame["wy_label"])
    retained = report.retained
    assert len(retained) == 3
    assert bool(retained["retained"].all())
    assert retained["value_cumecs"].is_monotonic_increasing
    # The retained set must be exactly the three smallest observations.
    assert np.isclose(sorted(retained["value_cumecs"]), sorted(bundle.q[:3])).all()
    assert "No observation is deleted" in " ".join(report.notes)


def test_run_reports_every_screen(bundle):
    report = o.run(bundle.q, labels=bundle.frame["wy_label"])
    for result in (report.grubbs, report.dixon_q):
        assert result.method
        assert result.assumptions
    assert set(report.ros.columns) >= {"normal_score", "log_value", "std_residual"}
    assert report.consensus


def test_summarise_names_the_lowest_observations(bundle):
    text = o.summarise(o.run(bundle.q, labels=bundle.frame["wy_label"]))
    assert "nothing deleted" in text
    assert "855" in text.replace(",", "")  # the 1965-66 peak, 855 cumecs
