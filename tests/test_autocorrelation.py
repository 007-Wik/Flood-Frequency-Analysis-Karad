"""Tests for the time-series dependence diagnostics.

Three failure modes are pinned here, each of which produced a plausible
looking but wrong output during development:

1.  An ACF significance band computed as ``z/sqrt(n)`` with ``z`` taken from a
    *confidence* level instead of a significance level.  That returns +/-0.008
    instead of +/-0.26 at N = 57 and flags every lag as significant.
2.  An effective sample size formed from an untruncated ACF sum.  An
    oscillatory ACF makes the sum vanish and the guard divides N by the floor.
3.  A rescaled range normalised by each block's own standard deviation.  That
    ratio is scale-free, so it does not grow with the aggregation lag and the
    fitted slope is ~0.05 for every true H -- a total, silent failure.

The Hurst tests also assert the estimator's measured spread, so the claim
"this record cannot resolve persistence" is checked rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffa_karad import autocorrelation as ac
from ffa_karad import config as _cfg
from ffa_karad import data_processing as dp


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


@pytest.fixture(scope="module")
def results(bundle):
    return ac.run(bundle, n_bootstrap=120)


# ---------------------------------------------------------------------------
# ACF / PACF estimators
# ---------------------------------------------------------------------------


def test_acf_and_pacf_match_statsmodels():
    """Our estimators are the reference implementations, checked lag for lag."""
    check = ac.verify_against_statsmodels(nlags=10, n_series=4)
    assert check.get("ok") is True, check


def test_pacf_second_order_identity():
    """pacf_2 = (r2 - r1^2)/(1 - r1^2) for every series, by construction."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        x = rng.normal(size=200)
        r = ac.acf(x, 2)
        expected = (r[2] - r[1] ** 2) / (1.0 - r[1] ** 2)
        assert ac.pacf(x, 2)[2] == pytest.approx(expected, abs=1e-12)


def test_pacf_second_order_matches_simulated_ar2():
    """The identity must also hold against a process with known coefficients."""
    assert ac.verify_ar2_pacf_identity(a1=0.6, a2=0.25)["ok"]


def test_pacf_of_a_random_walk_dies_after_lag_one():
    """An integrated series has all its dependence at lag 1; deeper lags must
    not inherit it, which is the whole point of partial autocorrelation."""
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(size=400))
    p = ac.pacf(walk, 5)
    assert abs(p[1]) > 0.9
    assert max(abs(p[2:])) < 0.2


def test_acf_band_is_the_normal_significance_band():
    """Guards the confidence-vs-significance slip: +/-1.96/sqrt(57) = +/-0.26."""
    lo, hi = ac.acf_band(57)
    assert hi == pytest.approx(1.959963985 / np.sqrt(57), abs=1e-9)
    assert (lo, hi) == (-hi, hi)
    assert ac.acf_band(57)[1] > 0.2, "band must not collapse at this N"


def test_acf_band_shrinks_as_the_record_grows():
    assert ac.acf_band(10_000)[1] < ac.acf_band(100)[1]


def test_white_noise_acf_stays_inside_the_band():
    rng = np.random.default_rng(99)
    lo, hi = ac.acf_band(57)
    breaches = 0
    for _ in range(400):
        r = ac.acf(rng.normal(size=57), 14)
        breaches += int(np.sum(np.abs(r[1:]) > hi))
    # 400 series x 14 lags = 5600 comparisons; ~5 % exceedance is expected
    assert breaches / 5600 < 0.12


def test_order_guard_rejects_sorted_input():
    """A sorted series must be refused, not silently analysed as a random walk."""
    with pytest.raises(ValueError, match="non-decreasing"):
        ac.acf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        ac.hurst_powers(list(range(1, 60)))


def test_short_series_is_rejected():
    with pytest.raises(ValueError, match="at least 3"):
        ac.acf([1.0, 2.0])


# ---------------------------------------------------------------------------
# Dependence summaries
# ---------------------------------------------------------------------------


def test_geyer_sum_is_never_below_one():
    """Truncation guarantees tau >= 1; the naive sum can drive it to zero."""
    r = np.array([1.0, 0.4, -0.5, 0.2, -0.3, 0.1])
    assert ac.geyer_initial_positive_sequence(r) >= 1.0
    assert ac.integrated_autocorrelation_time(r) >= 1.0


def test_effective_sample_size_is_bounded_by_the_record():
    rng = np.random.default_rng(5)
    out = ac.effective_sample_size(rng.normal(size=200))
    assert 0 < out["n_eff_conservative"] <= out["n"]


def test_effective_sample_size_falls_when_peaks_persist():
    """Persistence must reduce the independent-years count, not inflate it."""
    rng = np.random.default_rng(6)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.85 * x[i - 1] + rng.normal()
    persistent = ac.effective_sample_size(x)
    independent = ac.effective_sample_size(rng.normal(size=400))
    assert persistent["n_eff_conservative"] < independent["n_eff_conservative"]
    assert persistent["n_eff_conservative"] < 100


def test_autocorrelation_length_is_nonnegative():
    assert ac.autocorrelation_length([1.0, 0.3, 0.2, -0.1]) >= 0
    assert ac.autocorrelation_length([1.0, -0.3]) == 0


def test_cross_correlation_peaks_at_lag_zero_for_matched_series():
    rng = np.random.default_rng(7)
    base = rng.normal(size=80)
    matched = base + 0.01 * rng.normal(size=80)
    frame = ac.cross_correlation(base, matched, 5)
    assert int(np.argmax(frame["cross_correlation"])) == 0
    assert frame["cross_correlation"].iloc[0] > 0.99


def test_cross_correlation_detects_a_one_year_offset():
    """Positive lag means b leads a, so b[t] = a[t+1] peaks at lag 1."""
    rng = np.random.default_rng(8)
    base = rng.normal(size=80)
    frame = ac.cross_correlation(base, np.roll(base, -1), 5)
    assert int(np.argmax(frame["cross_correlation"])) == 1


def test_cross_correlation_requires_equal_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        ac.cross_correlation([3.0, 1.0, 2.0, 9.0, 4.0, 7.0], [1.0, 2.0, 3.0, 4.0])


def test_lag_scatter_table_has_the_right_pair_counts():
    rng = np.random.default_rng(9)
    frame = ac.lag_scatter_table(rng.normal(size=50), lags=(1, 2))
    assert len(frame) == (50 - 1) + (50 - 2)


def test_residual_acf_is_labelled(results):
    assert results.residuals is None  # not supplied to run()
    rng = np.random.default_rng(10)
    frame = ac.residual_acf(rng.normal(size=60))
    assert set(frame["series"]) == {"residuals"}


# ---------------------------------------------------------------------------
# Hurst
# ---------------------------------------------------------------------------


def test_fractional_gaussian_noise_reproduces_its_exponent():
    """The simulator must honour the H it is given.

    A generator that scales an independent random walk by n^(H-1/2) leaves
    every rescaled-range ratio unchanged and returns H = 0.5 always, so this
    test is the guard against that.
    """
    rng = np.random.default_rng(2026)
    lags = np.arange(2, 29)
    for h in (0.5, 0.8):
        est = [
            ac.hurst_powers(ac.fractional_gaussian_noise(400, h, rng), lags)
            for _ in range(30)
        ]
        assert np.median(est) == pytest.approx(h, abs=0.05)


def test_fractional_gaussian_noise_rejects_impossible_exponents():
    rng = np.random.default_rng(2027)
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            ac.fractional_gaussian_noise(20, bad, rng)


def test_rescaled_range_uses_the_whole_series_scale():
    """R/S divided by each block's own s.d. would be scale-free and flat."""
    rng = np.random.default_rng(12)
    walk = np.cumsum(rng.normal(size=2000))
    curve = ac.rescaled_range_curve(walk, np.arange(4, 65))
    first = curve.loc[curve["lag"] == 4, "rs_ratio"].iloc[0]
    last = curve.loc[curve["lag"] == 64, "rs_ratio"].iloc[0]
    assert last > 2.0 * first, "R/S must grow with the aggregation lag"
    assert ac.hurst_rescaled_range(walk, np.arange(4, 65)) > 0.3


def test_rescaled_range_curve_has_one_row_per_usable_lag():
    rng = np.random.default_rng(13)
    curve = ac.rescaled_range_curve(rng.normal(size=60), np.arange(4, 29))
    assert (curve["rs_ratio"] > 0).all()
    assert set(curve["lag"]) == set(range(4, 29))


def test_aggregated_variance_curve_matches_its_definition():
    rng = np.random.default_rng(14)
    x = rng.normal(size=40)
    frame = ac.aggregated_variance_curve(x, [1, 2, 3])
    row = frame.set_index("lag").loc[2]
    assert row["aggregated_variance"] == pytest.approx(
        float(np.mean((x[2:] - x[:-2]) ** 2) / 2.0)
    )


def test_hurst_recovery_reports_a_spread_that_matters():
    """At N = 57 the estimator must be shown to be unable to resolve H.

    The Monte-Carlo s.d. is measured, and the resolution verdict is derived
    from the width of the 95 per cent interval about the null rather than from
    a bare comparison with a round number, so it cannot flip on sampling error
    in the s.d. itself.
    """
    check = ac.verify_hurst_recovery(n=57, n_rep=200)
    assert check["ok"]
    assert check["monte_carlo_sd"] > 0.1, "spread suspiciously small at N = 57"
    assert check["half_width_95"] == pytest.approx(
        1.959963985 * check["monte_carlo_sd"] / 2.0, rel=1e-9
    )
    assert check["resolvable_against_0_5"] is False, (
        "if this ever becomes True the report text claiming persistence is "
        "unresolvable must be updated"
    )
    for entry in check["by_truth"].values():
        assert abs(entry["powers_bias"]) < 0.15
        assert entry["rs_bias"] > -0.05, (
            "the uncorrected rescaled range is expected to be upward biased; "
            "a negative bias would mean the reference estimator is broken"
        )


def test_hurst_interval_spans_the_null_when_the_estimate_is_inadmissible(results):
    """A clamp must never make the null look excluded."""
    ci = results.hurst_interval
    assert ci["ok"]
    if not ci["point_admissible"]:
        assert ci["simulation_at_null"] is True
        assert ci["distinguishes_half_from_point"] is False
        assert ci["q_low"] < 0.5 < ci["q_high"]


def test_hurst_interval_width_is_consistent_with_the_measured_spread(results):
    """A suspiciously tight interval is the block-bootstrap failure mode."""
    ci = results.hurst_interval
    assert ci["sd"] > 0.02
    assert ci["sd"] < 0.6


def test_hurst_bootstrap_reports_failure_on_a_constant_series():
    """No dispersion means no estimable exponent; that is a failure, not a 0."""
    out = ac.hurst_bootstrap_ci([5.0] * 20)
    assert out["ok"] is False
    assert "finite" in out["note"]


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def test_run_uses_time_order_and_finishes(results, bundle):
    assert results.lag1 == pytest.approx(ac.acf(bundle.q_ordered, 15)[1])
    assert results.band[1] == pytest.approx(ac.acf_band(bundle.n_records)[1])
    # lags 0..14: the n/4 rule caps the request at 14 for N = 57
    assert len(results.acf_frame) == 15
    assert results.acf_frame["lag"].max() == 14


def test_run_is_deterministic(bundle):
    first = ac.run(bundle, n_bootstrap=30)
    second = ac.run(bundle, n_bootstrap=30)
    assert first.hurst_powers == second.hurst_powers
    assert first.hurst_interval["sd"] == second.hurst_interval["sd"]


def test_effective_sample_size_is_reported_and_plausible(results, bundle):
    assert results.n_eff["n_eff_conservative"] <= bundle.n_records
    assert results.n_eff["n_eff_conservative"] > 5


def test_record_shows_no_lag_one_dependence(results):
    """The conclusion this record supports: peaks are independent at lag 1."""
    assert abs(results.lag1) < results.band[1]
    assert results.exceedance_lag == 0


def test_summary_renders_without_error(results):
    text = ac.summarise(results)
    assert "TIME-SERIES DEPENDENCE" in text
    assert "Hurst" in text
    assert "NOTES:" in text


def test_result_serialises(results):
    payload = results.to_dict()
    assert len(payload["acf"]) == 15
    assert "hurst_bootstrap" in payload
    assert payload["notes"]


def test_config_constants_used_are_declared():
    assert _cfg.CONFIG.acf_alpha < 0.5
    assert _cfg.CONFIG.acf_max_lag >= 1
    assert _cfg.CONFIG.hurst_lag_max > _cfg.CONFIG.hurst_lag_min
