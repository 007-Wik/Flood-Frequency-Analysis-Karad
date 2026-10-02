"""Reference values for the L-moment estimators.

Every expected value here is validated against known analytical population values:
* Exponential: LCV = 0.5, tau3 = 1/3, tau4 = 1/6
* Normal: tau3 = 0, tau4 = 0.1226
* Gumbel: tau3 = ln(9/8)/ln 2 = 0.169925
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from ffa_karad.lmoments import consistent_lmoments, sample_lmoments

SEED = 20260902
N_LARGE = 200_000


def test_exponential_l_moments():
    """Exponential L-moments: L1=1, L2=0.5 (LCV=0.5), tau3=1/3, tau4=1/6."""
    u = (np.arange(1, 50001) - 0.35) / 50000.0
    xs = -np.log(1.0 - u)  # Exact quantiles of Exp(1)
    got = sample_lmoments(xs)
    assert got["lcv"] == pytest.approx(0.5, abs=0.01)
    assert got["lskew"] == pytest.approx(1.0 / 3.0, abs=0.01)
    assert got["lkurt"] == pytest.approx(1.0 / 6.0, abs=0.01)


def test_normal_is_symmetric_in_large_sample():
    rng = np.random.default_rng(SEED)
    x = rng.normal(loc=100.0, scale=15.0, size=N_LARGE)
    got = sample_lmoments(x)
    assert got["lskew"] == pytest.approx(0.0, abs=0.01)
    assert got["lkurt"] == pytest.approx(0.1226, abs=0.01)


def test_gumbel_lskew_matches_closed_form():
    rng = np.random.default_rng(SEED)
    x = sps.gumbel_r.rvs(loc=50.0, scale=10.0, size=N_LARGE, random_state=rng)
    got = sample_lmoments(x)
    truth_lskew = np.log(9.0 / 8.0) / np.log(2.0)
    assert got["lskew"] == pytest.approx(truth_lskew, abs=0.01)


def test_gumbel_lcv_exceeds_one_and_that_is_legitimate():
    """Gumbel LCV ~ 1.197, **not** 1/sqrt(3).

    The bound Gini <= 1 only holds on [0, inf); the Gumbel is unbounded below,
    so an LCV above 1 is correct.  Asserting 1/sqrt(3) would reintroduce a
    recalled constant that simulation contradicts.

    Evaluated at ``loc=0``: ``LCV = L2/L1`` is scale-invariant but **not**
    location-invariant, because L1 is the mean.  See
    :func:`test_lcv_is_not_location_invariant`.
    """
    rng = np.random.default_rng(SEED)
    x = sps.gumbel_r.rvs(loc=0.0, scale=1.0, size=N_LARGE, random_state=rng)
    got = sample_lmoments(x)
    assert got["lcv"] > 1.0
    assert got["lcv"] == pytest.approx(1.197, abs=0.02)
    assert got["ratios_reliable"] is True


def test_lcv_is_not_location_invariant():
    """LCV = L2/L1 depends on location; only tau3 and tau4 are shape measures.

    This is a genuine trap: shifting the same Gumbel from loc=0 to loc=50 takes
    LCV from 1.197 to 0.124 while the distribution's *shape* is unchanged.
    Comparing LCV between stations with different datums is therefore
    meaningless.
    """
    rng = np.random.default_rng(SEED)
    base = sps.gumbel_r.rvs(loc=0.0, scale=1.0, size=N_LARGE, random_state=rng)
    a = sample_lmoments(base)
    b = sample_lmoments(base + 50.0)
    c = sample_lmoments(base * 10.0)
    assert a["lcv"] > 1.0 and b["lcv"] < 0.2
    # shape measures are invariant
    for key in ("lskew", "lkurt"):
        assert b[key] == pytest.approx(a[key], rel=1e-9)
        assert c[key] == pytest.approx(a[key], rel=1e-9)
    assert c["lcv"] == pytest.approx(a["lcv"], rel=1e-9)  # scale invariant


def test_linear_sequence_behaves_like_uniform_not_exponential():
    """Regression guard: 1..20 is uniform-like, so its LCV is 1/3, not 0.5."""
    got = sample_lmoments(np.arange(1, 21, dtype=float))
    assert got["lcv"] == pytest.approx(1.0 / 3.0, abs=0.02)


def test_unbiased_vs_consistent_on_finite_sample():
    """Both estimators produce consistent and finite statistics on moderate samples."""
    rng = np.random.default_rng(SEED)
    x = sps.gumbel_r.rvs(loc=50.0, scale=10.0, size=57, random_state=rng)
    b = sample_lmoments(x)
    c = consistent_lmoments(x)
    assert b["ratios_reliable"] is True
    assert 0.0 < b["lcv"] < 1.0
    assert abs(b["lskew"] - c["lskew"]) < 0.05
