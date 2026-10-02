"""Regression tests for the log-space LP3 correction.

The central claim these tests protect: LP3 is fitted to ``ln Q``, so the LP3
skewness for the Karad record is 0.0956, not the arithmetic skewness of 1.0819.
An earlier version of this pipeline applied an arithmetic-space 0.9 threshold,
concluded the record was unresolvable, and emitted ``adopted_cs = NaN``.  These
tests fail if that mistake is reintroduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffa_karad import config as _cfg
from ffa_karad import data_processing as dp
from ffa_karad import skewness_limits as sl

CS_LOG_EXPECTED = 0.0956
CS_RAW_EXPECTED = 1.0819


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


@pytest.fixture(scope="module")
def fit(bundle):
    return sl.lp3_fit(bundle.q)


def test_lp3_skewness_is_computed_in_log_space(fit):
    assert fit.cs_log == pytest.approx(CS_LOG_EXPECTED, abs=5e-4)
    assert fit.cs_raw == pytest.approx(CS_RAW_EXPECTED, abs=5e-4)


def test_log_skewness_is_far_below_the_historical_threshold(fit):
    """The threshold that triggered the erroneous kurtosis regime is not
    breached in log space."""
    assert fit.cs_log < 0.9
    # and the raw skewness, while real, is NOT what LP3 uses
    assert fit.cs_raw > 0.9


def test_fit_requires_positive_discharges():
    with pytest.raises(ValueError):
        sl.lp3_fit([1.0, 2.0, -3.0, 4.0])


def test_wilson_hilferty_matches_exact_pearson3(fit):
    """Two independent routes to K_T must agree for a small skewness."""
    for t in (2.0, 10.0, 50.0, 100.0, 1000.0):
        exact = sl.frequency_factor_exact(t, fit.cs_log)
        approx = sl.frequency_factor_wilson_hilferty(t, fit.cs_log)
        assert exact == pytest.approx(approx, rel=1e-3)


def test_wilson_hilferty_limit_is_normal(fit):
    """As G -> 0 the closed form must tend to the normal quantile z."""
    for t in (10.0, 100.0):
        from scipy import stats as sps

        z = sps.norm.ppf(1 - 1 / t)
        assert sl.frequency_factor_wilson_hilferty(t, 0.0) == pytest.approx(z)
        assert sl.frequency_factor_wilson_hilferty(t, 1e-9) == pytest.approx(z)


def test_quantiles_increase_with_return_period(fit):
    qs = sl.lp3_quantiles(_cfg.CONFIG.return_periods, fit)
    assert qs["Q_cumecs"].is_monotonic_increasing
    assert (qs["Q_cumecs"] > 0).all()


def test_bootstrap_population_reproduces_fitted_moments(fit):
    check = sl.verify_bootstrap_population(fit, n_draw=50_000)
    assert check["ok"], check


def test_bootstrap_propagates_all_three_parameters(fit):
    """A bootstrap that freezes mean or sd would be reporting no uncertainty."""
    _, params = sl.bootstrap_confidence_limits(
        np.exp(fit.mean_log) * 0 + _q_from_fit(fit), [100.0], n_bootstrap=300
    )
    row = params.set_index("quantity")
    for name in ("Cs_log", "mean_log", "sd_log"):
        assert row.loc[name, "boot_sd"] > 0


def _q_from_fit(fit):
    return np.array(
        [
            float(
                np.exp(
                    fit.mean_log
                    + fit.sd_log * sl.frequency_factor_exact(2.0 + i, fit.cs_log)
                )
            )
            for i in range(57)
        ]
    )


def test_point_estimate_lies_inside_its_own_interval(bundle):
    res = sl.run(bundle.q, n_bootstrap=400)
    boot = res.bootstrap.set_index("return_period_yr")
    for row in res.quantiles.itertuples():
        lo = boot.loc[row.return_period_yr, "Q_lower"]
        hi = boot.loc[row.return_period_yr, "Q_upper"]
        assert lo <= row.Q_cumecs <= hi


def test_skewness_band_is_not_truncated_at_threshold(fit):
    """The band must span the standard-error interval, however far it reaches."""
    band = sl.skewness_band(fit)
    lo, hi = band["cs_log"].min(), band["cs_log"].max()
    assert lo == pytest.approx(fit.cs_log - 1.96 * fit.cs_log_se, abs=1e-3)
    assert hi == pytest.approx(fit.cs_log + 1.96 * fit.cs_log_se, abs=1e-3)


def test_chowdhury_interval_brackets_the_point_estimate(fit):
    frame = sl.chowdhury_confidence_limits(_cfg.CONFIG.return_periods, fit)
    assert (frame["Q_lower"] <= frame["Q_cumecs"]).all()
    assert (frame["Q_cumecs"] <= frame["Q_upper"]).all()
