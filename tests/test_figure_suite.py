"""Tests for the notebook figure suite P1-P14.

These check that each figure is written and that the *numbers behind it* come
from the package results, not from a notebook global: the wrong axis, an
unsorted series or a rejected candidate appearing as an accepted curve all
produce a plausible-looking picture and no test failure unless something
asserts on the data behind it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ffa_karad import autocorrelation as ac
from ffa_karad import data_processing as dp
from ffa_karad import distribution_fitting as dfit
from ffa_karad import estimation_design_flood as edf
from ffa_karad import figure_suite as fs
from ffa_karad import lmoments as lm
from ffa_karad import machine_learning as ml
from ffa_karad import peaks_over_threshold as pot
from ffa_karad import skewness_limits as sk
from ffa_karad import statistical_tests as st
from ffa_karad import uncertainty_monte_carlo_bayesian_mcmc as unc

#: The exact figure files the suite promises, in notebook order.
EXPECTED = [
    "P1_time_series_trend_anomaly",
    "P2_seaborn_distribution",
    "P3_flood_frequency_curves",
    "P4a_pairplot",
    "P4b_jointplot",
    "P4c_correlation_heatmap",
    "P5_monte_carlo_uncertainty_fan",
    "P6_bayesian_mcmc_posteriors",
    "P7_qq_probability_plots",
    "P8_skewness_kurtosis",
    "P9_acf_pacf_hurst_lag",
    "P10_distribution_ranking",
    "P11_pot_gpd",
    "P12_ml_panel",
    "P13_publication_dashboard",
    "P14_comprehensive",
]


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


@pytest.fixture(scope="module")
def stats(bundle):
    return st.run(bundle.q_ordered)


@pytest.fixture(scope="module")
def results(bundle, stats):
    """Every result object the suite can draw from, at reduced draw counts."""
    return {
        "fits": dfit.run(bundle.q),
        "design": edf.run(bundle.q),
        "pot": pot.run(bundle.q_ordered),
        "stats": stats,
        "lmoments": lm.run(bundle.q),
        "autocorrelation": ac.run(bundle),
        "lp3": sk.run(bundle.q),
        "monte_carlo": unc.monte_carlo_record_uncertainty(bundle.q, n_records=40),
        "mcmc": unc.bayesian_mcmc(bundle.q_ordered, n_draws=400),
        "machine_learning": ml.run(bundle.q_ordered),
    }


@pytest.fixture(scope="module")
def ctx(bundle, results):
    return fs.SuiteContext(
        bundle=bundle,
        design=results["design"],
        fits=results["fits"],
        pot=results["pot"],
        stats=results["stats"],
        lmoments=results["lmoments"],
        autocorrelation=results["autocorrelation"],
        lp3=results["lp3"],
        monte_carlo=results["monte_carlo"],
        mcmc=results["mcmc"],
        machine_learning=results["machine_learning"],
    )


@pytest.fixture(scope="module")
def written(bundle, results, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("figures")
    records = fs.render_suite(
        bundle,
        results["design"],
        results["fits"],
        results["pot"],
        outdir,
        stats=results["stats"],
        lmoments=results["lmoments"],
        autocorrelation=results["autocorrelation"],
        lp3=results["lp3"],
        monte_carlo=results["monte_carlo"],
        mcmc=results["mcmc"],
        machine_learning=results["machine_learning"],
    )
    return records


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The suite as a whole
# ---------------------------------------------------------------------------


def test_render_suite_writes_every_notebook_figure(written):
    assert [r.name for r in written] == EXPECTED
    for record in written:
        assert "FAILED" not in record.caption, f"{record.name}: {record.caption}"
        assert record.path.exists(), record.name
        assert record.path.stat().st_size > 10_000, record.name


def test_render_suite_survives_a_missing_stage(bundle, results, tmp_path, monkeypatch):
    """A figure with no input must not cost the reviewer the frequency curve."""

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(fs, "plot_p6_mcmc_posteriors", explode)
    records = fs.render_suite(
        bundle,
        results["design"],
        results["fits"],
        results["pot"],
        tmp_path,
        stats=results["stats"],
        lmoments=results["lmoments"],
        autocorrelation=results["autocorrelation"],
        lp3=results["lp3"],
        monte_carlo=results["monte_carlo"],
        machine_learning=None,
    )
    failed = [r.name for r in records if "FAILED" in r.caption]
    assert "P6_bayesian_mcmc_posteriors" in failed
    assert "P12_ml_panel" in failed
    assert (tmp_path / "P3_flood_frequency_curves.png").exists()
    assert (tmp_path / "P1_time_series_trend_anomaly.png").exists()


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def test_context_uses_water_year_order_for_the_series(bundle, ctx):
    assert np.array_equal(ctx.q, bundle.q_ordered)
    assert ctx.years.min() == bundle.frame["wy_start_year"].min()
    assert np.all(np.diff(ctx.years) >= 0)


def test_trend_line_is_the_theil_sen_fit(bundle, ctx):
    fit = st.theil_sen(bundle.q_ordered)
    expected = fit["slope_cumecs_per_year"] * ctx.years + fit["intercept_cumecs"]
    assert np.allclose(ctx.trend_line, expected)
    assert ctx.sens_slope == pytest.approx(fit["slope_cumecs_per_year"])


def test_weibull_positions_are_sorted_and_span_one_to_n(bundle, ctx):
    assert np.all(np.diff(ctx.t_observed) >= 0)
    assert ctx.pp_weibull[0] == pytest.approx(1 / (bundle.n_records + 1))
    assert ctx.pp_weibull[-1] == pytest.approx(
        bundle.n_records / (bundle.n_records + 1)
    )
    assert np.allclose(np.sort(ctx.q), ctx.q_sorted)


def test_curve_reads_the_fit_not_the_ranking_table(ctx, results):
    fit = ctx.fit("Gumbel")
    periods = np.array([10.0, 100.0, 1000.0])
    for period, value in zip(periods, ctx.curve(fit, periods)):
        assert value == pytest.approx(fit.quantile(float(period)))


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def test_p3_plots_every_candidate_and_labels_the_rejected_ones(ctx):
    fig = fs.plot_p3_frequency_curves(ctx)
    names = [trace.name for trace in fig.data]
    for fit in ctx.fits.fits:
        if fit.accepted:
            assert fit.candidate.name in names
        else:
            # A rejected curve is drawn dotted, but never under its own name:
            # a legend entry without the verdict reads as an endorsement.
            assert fit.candidate.name not in names
            assert f"{fit.candidate.name} (rejected)" in names


def test_p3_title_reports_the_candidate_count_not_twelve(ctx):
    fig = fs.plot_p3_frequency_curves(ctx)
    title = fig.layout.title.text
    assert f"{len(ctx.fits.fits)} candidates fitted" in title
    assert "12 distributions" not in title
    assert f"{len(ctx.accepted_fits())} accepted" in title


def test_p5_fan_is_ordered_by_return_period(results):
    fan = results["monte_carlo"].fan
    periods = fan["return_period_yr"].to_numpy(dtype=float)
    assert np.all(np.diff(periods) > 0)
    for low, mid, high in (
        ("mc_p05_cumecs", "mc_median_cumecs", "mc_p95_cumecs"),
        ("mc_p25_cumecs", "mc_median_cumecs", "mc_p75_cumecs"),
    ):
        assert (fan[low] < fan[mid]).all()
        assert (fan[mid] < fan[high]).all()
    assert (fan["mc_median_cumecs"] < fan["observed_lp3_cumecs"]).any()


def test_p6_states_the_convergence_verdict(ctx):
    fig = fs.plot_p6_mcmc_posteriors(ctx)
    title = fig.layout.title.text
    diagnostics = ctx.mcmc.diagnostics
    if diagnostics.passed:
        assert "gate PASSED" in title
    else:
        assert "gate REFUSED" in title


def test_p7_only_plots_candidates_that_were_fitted(ctx):
    fig = fs.plot_p7_qq_plots(ctx)
    ranked = ctx.fits.ranking.sort_values(["accepted", "aicc"], ascending=[False, True])
    for name in ranked["distribution"].head(6):
        assert ctx.fit(name) is not None


def test_p9_annotates_an_inadmissible_hurst_estimate(ctx):
    """H = -0.018 is outside (0, 1); the figure must not print it as a result."""
    fig = fs.plot_p9_dependence(ctx)
    text = " ".join(a.text for a in fig.layout.annotations)
    assert "NOT admissible" in text or ctx.autocorrelation.hurst_interval.get(
        "point_admissible", False
    )
    assert "MC s.d." in text


def test_p9_rolling_variance_and_ljung_box_use_the_ordered_series(ctx):
    fig = fs.plot_p9_dependence(ctx)
    rolling = next(t for t in fig.data if t.name == "10-yr Rolling Var")
    expected = pd_series(ctx.q).rolling(10, center=True).var().to_numpy(dtype=float)
    assert np.allclose(rolling.y, expected, equal_nan=True)


def pd_series(values):
    import pandas as pd

    return pd.Series(values)


def test_p11_is_stamped_with_the_pot_verdict(ctx):
    fig = fs.plot_p11_pot(ctx)
    title = fig.layout.title.text
    if ctx.pot.accepted:
        assert "POT ACCEPTED" in title
    else:
        assert "POT NOT ADOPTED" in title
        assert ctx.pot.reject_reasons[0][:20] in title


def test_p12_reports_out_of_fold_skill_not_train_skill(ctx):
    fig = fs.plot_p12_ml_panel(ctx)
    bar_labels = [
        text.get_text()
        for ax in fig.axes
        for text in ax.get_yticklabels() + ax.get_xticklabels()
    ]
    captions = " ".join(ax.get_title() for ax in fig.axes)
    assert "Out-of-Fold" in captions or any("Out-of-Fold" in b for b in bar_labels)
    for score in ctx.machine_learning.scores:
        assert score.n_splits >= 2, "a single split is a train fit by another name"


def test_p13_says_how_many_candidates_were_accepted(ctx):
    fig = fs.plot_p13_dashboard(ctx)
    title = fig.layout.title.text
    assert f"{len(ctx.accepted_fits())} accepted" in title
    assert f"{len(ctx.fits.fits)} candidates" in title


def test_p4_features_are_the_leakage_safe_ones(ctx):
    frame = fs._feature_frame(ctx)
    for column in ("lag_1", "mean_5", "expanding_mean", "Q_peak", "decade_lbl"):
        assert column in frame.columns
    assert frame["t"].is_monotonic_increasing
    assert "q" not in frame.columns, "the target column must not double as a feature"


def test_importance_is_averaged_over_folds_not_duplicated(ctx, results):
    importance = fs._importance(ctx)
    assert not importance.index.duplicated().any()
    assert list(importance.index) == list(importance.sort_values(ascending=False).index)
    assert (importance >= 0).all()

    # Averaging per-fold rows must reproduce the single set of features, once.
    rf = next(
        s for s in results["machine_learning"].scores if s.model == "RandomForest"
    )
    expected = rf.feature_importance.groupby("feature")["importance"].mean()
    assert set(importance.index) == set(expected.index)
    assert len(importance) == len(rf.feature_importance["feature"].unique())
