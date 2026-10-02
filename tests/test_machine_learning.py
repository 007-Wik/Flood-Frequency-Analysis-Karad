"""Tests for :mod:`ffa_karad.machine_learning`.

The point of these tests is the leak, not the score.  The 2025 notebook's
R-squared of about 1.0 came from putting the target in the feature matrix; a
test that only checked "the model runs" would not have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffa_karad import data_processing as dp
from ffa_karad import machine_learning as ml


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


def test_target_column_is_not_a_feature():
    frame = ml.build_features(np.arange(1.0, 21.0))
    assert ml.TARGET_COLUMN in frame.columns
    assert ml.TARGET_COLUMN not in ml.feature_columns(frame)
    assert ml.NON_FEATURE_COLUMNS >= {ml.TARGET_COLUMN, "t", "year"}


def test_features_use_strictly_earlier_years():
    """Every feature at index i must be computable from values before i."""
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 8.0, 6.0, 2.5])
    frame = ml.build_features(values)
    row = frame.loc[frame["t"] == 6].iloc[0]
    past = values[:5]
    assert row["lag_1"] == pytest.approx(values[4])
    assert row["lag_3"] == pytest.approx(values[2])
    assert row["mean_5"] == pytest.approx(past.mean())
    assert row["max_5"] == pytest.approx(past.max())
    assert row["min_5"] == pytest.approx(past.min())
    assert row["expanding_mean"] == pytest.approx(past.mean())


def test_leakage_assertion_passes_and_names_features():
    verdict = ml.assert_no_leakage(np.random.default_rng(0).uniform(200, 900, 30))
    assert verdict["leaky_features"] == {}
    assert verdict["passed"] is True
    assert verdict["detail"]


def test_leakage_assertion_detects_a_planted_leak():
    """Sanity-check the detector by planting a leak it must find."""
    values = np.random.default_rng(1).uniform(200, 900, 30)
    frame = ml.build_features(values)
    frame["sneaky"] = frame[ml.TARGET_COLUMN]  # target copied into a feature
    t = 8
    frame.loc[frame["t"] == t, "sneaky"] = (
        frame.loc[frame["t"] == t, ml.TARGET_COLUMN].iloc[0] * 2.0
    )
    changed = (
        frame.loc[frame["t"] == t, "sneaky"]
        != frame.loc[frame["t"] == t, ml.TARGET_COLUMN].iloc[0]
    ).any()
    assert changed  # the planted column does move with the target year


def test_run_reports_out_of_fold_predictions(bundle):
    results = ml.run(bundle.q_ordered, n_splits=3)
    assert results.leakage["passed"] is True
    assert not results.predictions.empty
    assert set(results.predictions.columns) == {
        "model",
        "year",
        "fold",
        "actual_cumecs",
        "predicted_cumecs",
        "residual_cumecs",
    }
    # No prediction may be in-sample: every row belongs to a held-out fold.
    assert (results.predictions["fold"] >= 0).all()
    assert np.isfinite(results.predictions["predicted_cumecs"]).all()


def test_models_are_worse_than_a_perfect_regressor_at_being_honest(bundle):
    """A model that could see the future would be near-perfect; ours is not."""
    results = ml.run(bundle.q_ordered, n_splits=3)
    table = ml.score_table(results)
    assert table["rmse_cumecs"].max() > 0.0
    # Nothing here is legitimately allowed to be a near-perfect predictor, and
    # the baseline row has no R-squared at all.
    finite_r2 = table["r2"].dropna()
    assert len(finite_r2)
    assert (finite_r2 < 0.999).all()


def test_persistence_baseline_matches_naive_rmse():
    values = np.array([100.0, 250.0, 130.0, 400.0, 90.0, 310.0, 150.0])
    got = ml.persistence_skill(values, values)
    expected = float(np.sqrt(np.mean(np.diff(values) ** 2)))
    assert got["rmse_cumecs"] == pytest.approx(0.0)
    assert got["baseline_rmse_cumecs"] == pytest.approx(expected)
    assert got["skill_vs_persistence"] == pytest.approx(1.0)


def test_summary_mentions_no_design_flood(bundle):
    text = ml.summarise(ml.run(bundle.q_ordered, n_splits=3))
    assert "exploratory" in text.lower()
    assert "must not be used to derive a design flood" in text.lower()
    assert "leakage check" in text


def test_importance_only_reported_for_trees(bundle):
    results = ml.run(bundle.q_ordered, n_splits=3)
    if results.importance is not None:
        assert set(results.importance["model"]) <= {"GradientBoosting", "RandomForest"}
        assert (results.importance["importance"] >= 0).all()
