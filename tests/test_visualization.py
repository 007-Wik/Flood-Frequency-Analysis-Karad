"""Tests for the three figure modules.

These check the numbers behind the pictures and the fact that every figure is
written, not the appearance of the pixels: a broken axis or a curve drawn from
the wrong array is caught by the frame assertions below, and a figure that fails
to render is caught by ``render_all``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ffa_karad import data_processing as dp
from ffa_karad import distribution_fitting as df
from ffa_karad import estimation_design_flood as design
from ffa_karad import histogram_ecdf_violin as hev
from ffa_karad import peaks_over_threshold as pot
from ffa_karad import seaborn_kde as kde
from ffa_karad import visualization as viz


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


@pytest.fixture(scope="module")
def design_results(bundle):
    return design.run(bundle.q)


@pytest.fixture(scope="module")
def fits(bundle):
    return df.run(bundle.q)


@pytest.fixture(scope="module")
def pot_results(bundle):
    return pot.run(bundle.q)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Histogram / ECDF / violin
# ---------------------------------------------------------------------------


def test_sturges_bins_is_not_more_than_the_data_supports():
    assert hev.sturges_bins(57) == 7
    assert hev.sturges_bins(1000) <= 12
    assert hev.sturges_bins(4) >= 2


def test_histogram_counts_conserve_the_sample(bundle):
    frame = hev.histogram_frame(bundle.q)
    assert frame["count"].sum() == bundle.n_records
    assert frame["bin_right"].iloc[-1] >= bundle.q.max()
    assert frame["bin_left"].iloc[0] <= bundle.q.min()
    # Contiguous bins: no gaps where observations could fall through.
    assert np.allclose(
        frame["bin_right"].to_numpy()[:-1], frame["bin_left"].to_numpy()[1:]
    )


def test_ecdf_is_a_step_function_from_zero_to_one(bundle):
    frame = hev.ecdf_frame(bundle.q)
    assert len(frame) == bundle.n_records
    assert frame["ecdf"].iloc[0] == pytest.approx(1 / bundle.n_records)
    assert frame["ecdf"].iloc[-1] == pytest.approx(1.0)
    assert frame["value_cumecs"].is_monotonic_increasing
    assert np.allclose(frame["ecdf"] + frame["exceedance"], 1.0)


def test_the_three_panels_draw(bundle):
    fig, axes = hev.plot_all(bundle.q)
    assert len(axes) == 3
    assert all(a.get_xlabel() or a.get_ylabel() for a in axes)


# ---------------------------------------------------------------------------
# KDE
# ---------------------------------------------------------------------------


def test_kde_frame_support_is_clipped_to_the_data(bundle):
    frame = kde.kde_frame(bundle.q, transform="log")
    log_q = np.log(bundle.q)
    assert frame["x"].min() >= log_q.min() - 1e-9
    assert frame["x"].max() <= log_q.max() + 1e-9
    assert (frame["density"] >= 0).all()


def test_kde_frame_reports_cumecs_in_log_mode(bundle):
    frame = kde.kde_frame(bundle.q, transform="log")
    assert np.allclose(frame["value_cumecs"], np.exp(frame["x"]))
    raw = kde.kde_frame(bundle.q, transform="raw")
    assert np.allclose(raw["value_cumecs"], raw["x"])


def test_kde_rejects_an_unknown_transform(bundle):
    with pytest.raises(ValueError, match="transform"):
        kde.kde_frame(bundle.q, transform="sqrt")


def test_kde_summary_is_honest_about_the_bandwidth(bundle):
    summary = kde.summarise_kde(bundle.q)
    assert summary["n"] == bundle.n_records
    assert summary["bandwidth_adjust"] == 1.0
    assert summary["log_q_normal_adequacy"] in {"rejected", "not rejected"}
    assert "not evidence about the tail" in summary["note"]


def test_normality_comparison_draws(bundle):
    fig, ax = kde.plot_normality_comparison(bundle.q)
    assert len(ax.get_legend().get_texts()) == 3


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def test_flood_frequency_plot_uses_the_adopted_table(design_results):
    fig, ax = viz.plot_flood_frequency(design_results)
    table = design_results.table
    line = [ln for ln in ax.get_lines() if ln.get_marker() == "o"][0]
    assert np.allclose(line.get_ydata(), table["adopted_lp3_cumecs"])
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_record_on_curve_plots_every_observation(bundle, design_results):
    fig, ax = viz.plot_record_on_curve(design_results, bundle)
    points = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    observed = points[0]
    assert observed.get_xdata().size == bundle.n_records


def test_time_series_uses_water_year_order(bundle, design_results):
    fig, ax = viz.plot_time_series(bundle, design_results)
    years = ax.get_lines()[0].get_xdata()
    assert np.all(np.diff(years) >= 0), "the x-axis must be the water year, not a sort"
    assert years.min() == bundle.frame["wy_start_year"].min()


def test_candidate_comparison_plots_only_accepted_fits(fits):
    fig, ax = viz.plot_candidate_comparison(fits)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    accepted = set(fits.ranking.loc[fits.ranking["accepted"], "distribution"])
    assert len(labels) == len(accepted)
    assert all(any(name in label for label in labels) for name in accepted)


def test_pot_plot_states_the_refusal_when_gated(pot_results):
    fig, ax = viz.plot_pot_return_levels(pot_results)
    if not pot_results.accepted:
        assert not ax.axison
        for reason in pot_results.reject_reasons:
            assert reason[:30] in " ".join(t.get_text() for t in ax.texts)
    else:
        assert ax.get_xscale() == "log"


def test_design_levels_mark_the_hfl(design_results):
    fig, ax = viz.plot_hfl_and_levels(design_results)
    hfl = float(design_results.hfl_check["hfl_water_level_m"])
    assert any(np.isclose(ln.get_ydata()[0], hfl) for ln in ax.get_lines())


def test_render_all_writes_every_figure(
    bundle, design_results, fits, pot_results, tmp_path
):
    records = viz.render_all(bundle, design_results, fits, pot_results, tmp_path)
    assert len(records) >= 10
    for record in records:
        assert "FAILED" not in record.caption, f"{record.name} failed: {record.caption}"
        assert record.path.exists() and record.path.stat().st_size > 5000


def test_render_all_survives_a_broken_builder(
    bundle, design_results, fits, pot_results, tmp_path, monkeypatch
):
    """One failing figure must not cost the reviewer the flood-frequency curve."""

    def explode(design_arg, ax=None):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(viz, "plot_acf", explode)
    records = viz.render_all(bundle, design_results, fits, pot_results, tmp_path)
    failed = [r for r in records if "FAILED" in r.caption]
    assert [r.name for r in failed] == ["acf"]
    assert (tmp_path / "flood_frequency.png").exists()


def test_sub_annual_panel(bundle):
    frame = pd.DataFrame(
        {
            "water_year": [2019, 2019, 2020],
            "q_cumecs": [1200.0, 5300.0, 6400.0],
        }
    )
    fig, ax = viz.plot_sub_annual(frame)
    assert ax.get_lines() or ax.patches
    assert len(ax.get_legend().get_texts()) >= 2
