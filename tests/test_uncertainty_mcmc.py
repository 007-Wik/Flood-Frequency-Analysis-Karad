"""Tests for :mod:`ffa_karad.uncertainty_monte_carlo_bayesian_mcmc`.

The sampler tests run short chains on purpose: the point is that the
diagnostics and the gate respond correctly, not that this module is fast.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ffa_karad import config as _cfg
from ffa_karad import data_processing as dp
from ffa_karad import skewness_limits as _skew
from ffa_karad import uncertainty_monte_carlo_bayesian_mcmc as u


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


# ---------------------------------------------------------------------------
# Monte Carlo over independent records
# ---------------------------------------------------------------------------


def test_monte_carlo_shape_and_columns(bundle):
    out = u.monte_carlo_record_uncertainty(bundle.q, n_records=60, record_length=57)
    assert out.n_records == 60
    assert out.record_length == 57
    for frame in (out.summary, out.coverage, out.exceedance_probability):
        assert len(frame) == len(_cfg.CONFIG.return_periods)
        assert frame["return_period_yr"].is_unique
    assert out.summary["mc_median_cumecs"].gt(0).all()
    assert (out.summary["mc_lower_cumecs"] < out.summary["mc_upper_cumecs"]).all()


def test_monte_carlo_is_deterministic(bundle):
    a = u.monte_carlo_record_uncertainty(bundle.q, n_records=40)
    b = u.monte_carlo_record_uncertainty(bundle.q, n_records=40)
    assert np.allclose(a.summary["mc_median_cumecs"], b.summary["mc_median_cumecs"])


def test_monte_carlo_interval_widens_with_return_period(bundle):
    out = u.monte_carlo_record_uncertainty(bundle.q, n_records=400)
    width = out.summary.set_index("return_period_yr")["mc_relative_width"]
    # A longer record buys less at the extreme tail than near the median.
    assert width.loc[1000.0] > width.loc[2.0]


def test_shorter_records_give_wider_intervals(bundle):
    """Uncertainty must shrink as the record grows.

    Checked on the absolute width at T = 2 and T = 100.  The extreme tail is
    deliberately not asserted: for T = 1000 the re-estimated spread is not
    monotone in record length, because a 10-year record's LP3 tail is governed
    by a couple of parameters estimated from ten points and the estimates
    wander.  That is a real property of the estimator, not a bug to be tuned
    away.
    """
    short = u.monte_carlo_record_uncertainty(bundle.q, n_records=400, record_length=10)
    full = u.monte_carlo_record_uncertainty(bundle.q, n_records=400, record_length=57)

    def width(f):
        return f.summary["mc_upper_cumecs"] - f.summary["mc_lower_cumecs"]

    short_width, full_width = width(short), width(full)
    assert short_width.iloc[0] > full_width.iloc[0]
    assert short_width.iloc[5] > full_width.iloc[5]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_split_r_hat_is_one_for_iid_chains():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 4000))
    assert u.split_r_hat(chains) < 1.01


def test_split_r_hat_detects_separated_chains():
    """Two chains stuck in different places must not look converged."""
    rng = np.random.default_rng(1)
    chains = np.vstack([rng.normal(0.0, 1.0, 3000), rng.normal(20.0, 1.0, 3000)])
    assert u.split_r_hat(chains) > 1.5


def test_ess_is_bounded_by_the_draw_count():
    rng = np.random.default_rng(2)
    ess = u.effective_sample_size(rng.normal(size=(4, 5000)))
    assert 0 < ess <= 4 * 5000


def test_ess_collapses_for_a_stuck_chain():
    """A constant chain carries no information about the target's spread."""
    assert u.effective_sample_size(np.ones((4, 200))) >= 4 * 200  # no false alarm
    rng = np.random.default_rng(3)
    stuck = np.cumsum(rng.normal(scale=1e-9, size=(2, 4000)), axis=1)
    assert u.effective_sample_size(stuck) < 2 * 4000


# ---------------------------------------------------------------------------
# Sampler and gate
# ---------------------------------------------------------------------------


def test_log_posterior_rejects_impossible_parameters(bundle):
    log_q = np.log(bundle.q)
    priors = u._priors(_skew.lp3_fit(bundle.q), 2.0)
    good = u._log_posterior(np.array([7.8, 0.55, 0.15]), log_q, priors)
    assert np.isfinite(good)
    assert not np.isfinite(u._log_posterior(np.array([7.8, -0.1, 0.15]), log_q, priors))
    assert not np.isfinite(u._log_posterior(np.array([7.8, 0.55, 99.0]), log_q, priors))


def test_mcmc_runs_and_reports_diagnostics(bundle):
    out = u.bayesian_mcmc(bundle.q_ordered, n_draws=4000, burn_in=1000)
    assert set(out.parameter_names) == {"mean_ln_q", "sd_ln_q", "cs_log"}
    assert out.chains.shape[1] == 3000
    assert set(out.diagnostics.r_hat) == set(out.parameter_names)
    assert all(v > 0 for v in out.diagnostics.ess.values())
    assert 0.0 < out.diagnostics.accept_rate_mean < 1.0
    assert isinstance(out.diagnostics.passed, bool)


def test_mcmc_withholds_quantiles_when_the_gate_fails(bundle, monkeypatch):
    """A deliberately impossible threshold must suppress the answer."""
    monkeypatch.setattr(
        u._cfg, "CONFIG", dataclasses.replace(u._cfg.CONFIG, mcmc_rhat_threshold=0.5)
    )
    out = u.bayesian_mcmc(bundle.q_ordered, n_draws=2000, burn_in=500)
    assert out.accepted is False
    assert out.diagnostics.failures
    assert out.quantiles.empty
    assert "GATE FAILED" in " ".join(out.notes)


def test_posterior_interval_is_wider_than_a_shrunken_one(bundle):
    """The 2025 notebook's posterior s.d. was 0.49x the analytic s.e."""
    out = u.bayesian_mcmc(bundle.q_ordered, n_draws=6000, burn_in=1000)
    ratios = out.posterior_summary["posterior_sd_over_analytic"].to_numpy()
    assert np.all(np.isfinite(ratios))
    assert (ratios > 0.7).all()


def test_prior_sensitivity_shifts_are_small(bundle):
    out = u.bayesian_mcmc(bundle.q_ordered, n_draws=4000, burn_in=1000)
    shift = out.prior_sensitivity["posterior_mean_shift_sd"].abs()
    assert (shift < 0.5).all()


def test_summarise_states_the_verdict(bundle):
    text = u.summarise(
        u.monte_carlo_record_uncertainty(bundle.q, n_records=50),
        u.bayesian_mcmc(bundle.q_ordered, n_draws=3000, burn_in=1000),
    )
    assert "CONVERGENCE GATE" in text
    assert "independent records" in text
