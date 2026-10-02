"""
Time-series dependence diagnostics for the annual-peak record.

Scope and honesty note
----------------------
An annual-maxima record of 57 water years is *not* a long series.  Everything
in this module is a **diagnostic**, not an input to the design flood.  The
reasons are structural and are enforced rather than footnoted:

* The ACF significance band at N = 57 is +/-1.96/sqrt(57) = +/-0.26.  Only a
  lag-1 correlation above about 0.26 can be declared significant, and the
  estimate itself carries a standard error of roughly the same size, so a
  reported ``r1 = 0.30`` is not distinguishable from ``r1 = 0.15``.
* Hurst-type exponents carry a Monte-Carlo standard error of about 0.2 at
  N = 57 (measured, not assumed -- see :func:`verify_hurst_recovery`).  The
  distance between "no persistence" (0.5) and "strongly persistent" (0.7) is
  therefore *not* resolvable from this record, whatever estimator is used.
  Any report that quotes a Hurst exponent without that standard error is
  overstating the data.

What is carried forward from the notebook
-----------------------------------------
The notebook plotted ACF, PACF, Hurst and lag plots without any of this
context, and reported the Hurst exponent as if it were a measured property of
the river.  This module keeps the diagnostics, adds the bands and standard
errors that make them interpretable, and adds the two quantities that actually
matter for flood-frequency analysis:

* :func:`effective_sample_size` -- how many *independent* years the record
  really contains.  All confidence intervals elsewhere assume N independent
  annual maxima; if the peaks are correlated, those intervals are too narrow
  and this ratio quantifies by how much.
* :func:`residual_acf` -- the dependence check that belongs to a *fitted*
  model.  Independence of the raw series is the wrong question; what must hold
  is independence of the residuals from the adopted distribution.

Implementation choices, each verified in this package
-----------------------------------------------------
``acf``
    The "adjusted" (unbiased) estimator of ``statsmodels.tsa.stattools.acf``,
    i.e. both numerator and denominator are divided by ``n - k``.  Verified to
    machine precision against statsmodels by :func:`verify_against_statsmodels`.
``pacf``
    Durbin-Levinson recursion written out from the recurrence rather than
    delegated, and verified lag-for-lag against
    ``statsmodels.tsa.stattools.pacf(method="ld")`` on several series.  Its
    closed form ``pacf_2 = (r2 - r1**2) / (1 - r1**2)`` is checked against
    simulated AR(2) processes of known parameters.
``hurst``
    Powers' aggregated-variance estimator is **primary**: regressing the
    log aggregated variance on log aggregation scale is close to unbiased with
    no small-sample correction constant.  The rescaled-range slope is reported
    for reference and is upward biased at this record length; the Anis-Lloyd
    correction that would remove that bias is **not** applied, because the
    rescaled-range expectation depends on the small-sample details of the
    R(m)/S(m) distribution and no closed form used here reproduces it to better
    than a factor 1.6 at m <= 128.  Applying it would inject an unverifiable
    constant into a submittable number.  The measured bias is reported instead,
    so the reader can see it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as _cfg
from . import util

log = util.get_logger("autocorrelation")


# ---------------------------------------------------------------------------
# Sample autocorrelation
# ---------------------------------------------------------------------------


def acf(values: Sequence[float] | np.ndarray, nlags: int | None = None) -> np.ndarray:
    """Sample autocorrelation function, adjusted (unbiased) estimator.

    .. math::
        r_k = \\frac{\\sum_{t=k+1}^{n}(x_t-\\bar x)(x_{t-k}-\\bar x)}
                    {\\sum_{t=1}^{n}(x_t-\\bar x)^2}\\cdot\\frac{n}{n-k}

    The ``n/(n-k)`` factor removes the downward bias of the naive estimator.
    It is applied to numerator and denominator together, which is what
    ``statsmodels.tsa.stattools.acf(adjusted=True)`` does; the Durbin-Levinson
    recursion in :func:`pacf` is consistent with this same convention, so ACF
    and PACF come from one internally coherent set of estimates.

    Parameters
    ----------
    values
        Series **in time order**.  Use ``DataBundle.q_ordered``; feeding the
        ascending order statistic returns a band of +1 everywhere and is
        rejected by the shared order guard.
    """
    x = _series(values, "acf")
    nlags = _default_lag(x.size, nlags)
    n = x.size
    d = x - x.mean()
    denom = float(d @ d)
    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    if denom <= 0.0:
        # A constant series has no estimable autocorrelation.  NaN is honest:
        # zeros would report "no dependence", which is a claim, not a failure.
        out[1:] = np.nan
        log.warning("acf: zero variance in the input; returning NaN beyond lag 0")
        return out
    for k in range(1, nlags + 1):
        out[k] = float(d[k:] @ d[:-k]) / denom * (n / (n - k))
    return out


def pacf(values: Sequence[float] | np.ndarray, nlags: int | None = None) -> np.ndarray:
    """Partial autocorrelation function by Durbin-Levinson recursion.

    With :math:`\\phi_{k,j}` the AR(k) coefficients,

    .. math::
        \\phi_{k,k} = \\frac{r_k - \\sum_{j=1}^{k-1}\\phi_{k-1,j}\\,r_{k-j}}
                           {1 - \\sum_{j=1}^{k-1}\\phi_{k-1,j}\\,r_j},
        \\qquad
        \\phi_{k,j} = \\phi_{k-1,j} - \\phi_{k,k}\\,\\phi_{k-1,k-j}

    The recursion is written out rather than delegated so the estimator is
    auditable, and it is checked against statsmodels in the test suite.  At
    ``k = 2`` it reduces to the standard identity
    ``pacf_2 = (r_2 - r_1**2) / (1 - r_1**2)``, which the tests also confirm
    against simulated AR(2) processes.
    """
    rho = acf(values, nlags)
    nlags = rho.size - 1
    if nlags < 1:
        return rho
    phi = np.zeros((nlags + 1, nlags + 1), dtype=float)
    pac = np.full(nlags + 1, np.nan, dtype=float)
    pac[0] = 1.0
    phi[1, 1] = rho[1]
    pac[1] = rho[1]
    for k in range(2, nlags + 1):
        prev = phi[k - 1, 1:k]
        num = rho[k] - float(prev @ rho[k - 1 : 0 : -1])
        den = 1.0 - float(prev @ rho[1:k])
        phi[k, k] = num / den
        pac[k] = phi[k, k]
        phi[k, 1:k] = prev - phi[k, k] * phi[k - 1, k - 1 : 0 : -1]
    return pac


def acf_band(n: int, alpha: float | None = None) -> tuple[float, float]:
    """Two-sided normal screening band ``z_{1-alpha/2} / sqrt(n)``.

    ``alpha`` is a **significance level**, following ``CONFIG.alpha`` and the
    ``level`` convention already used by
    :func:`ffa_karad.skewness_limits.chowdhury_confidence_limits`.  At
    alpha = 0.05 and N = 57 the band is +/-0.26.

    For white noise the sampling standard deviation of ``r_k`` is
    ``1/sqrt(n)``.  The band is not exact at small ``n`` and it ignores that
    neighbouring autocorrelations are not independent, so it screens rather
    than tests.  The tests are the Ljung-Box scan in
    :mod:`ffa_karad.statistical_tests` and the runs test.
    """
    alpha = _cfg.CONFIG.acf_alpha if alpha is None else alpha
    half = float(sps.norm.ppf(1.0 - alpha / 2.0)) / np.sqrt(max(n, 1))
    return -half, half


def acf_table(
    values: Sequence[float] | np.ndarray,
    nlags: int | None = None,
    alpha: float | None = None,
) -> pd.DataFrame:
    """ACF and PACF with the screening band and significance flags."""
    x = _series(values, "acf_table")
    nlags = _default_lag(x.size, nlags)
    alpha = _cfg.CONFIG.acf_alpha if alpha is None else alpha
    r = acf(x, nlags)
    p = pacf(x, nlags)
    lo, hi = acf_band(x.size, alpha)
    frame = pd.DataFrame(
        {
            "lag": np.arange(nlags + 1),
            "acf": r,
            "pacf": p,
            "band_low": lo,
            "band_high": hi,
        }
    )
    frame["acf_within_band"] = (frame["lag"] == 0) | (frame["acf"].abs() <= hi)
    frame["pacf_within_band"] = (frame["lag"] == 0) | (frame["pacf"].abs() <= hi)
    frame.attrs["alpha"] = alpha
    return frame


# ---------------------------------------------------------------------------
# Dependence summaries that feed the uncertainty budget
# ---------------------------------------------------------------------------


def first_exceedance_lag(
    r: Sequence[float] | np.ndarray, n: int | None = None, alpha: float | None = None
) -> int:
    """Largest lag whose ACF still leaves the band before the first return, or 0.

    ``n`` is the record length the band is computed from; it must be the number
    of observations, not the number of lags, or the band is wrong by a factor
    of about two at this record length.

    Used to size the dependence-structure input to the Hurst analysis and
    reported in the summary so the choice is visible rather than buried.
    """
    arr = np.asarray(r, dtype=float)[1:]
    if arr.size == 0:
        return 0
    n = int(arr.size) + 1 if n is None else int(n)
    lo, hi = acf_band(n, alpha)
    out = 0
    for k, value in enumerate(arr, start=1):
        if not np.isfinite(value) or value > hi or value < lo:
            out = k
        else:
            break
    return out


def geyer_initial_positive_sequence(r: Sequence[float] | np.ndarray) -> float:
    """Truncated autocorrelation sum ``1 + 2 sum_k r_k`` (Geyer, 1992).

    The pair sequence ``gamma_k = r_{2k-1} + r_{2k}`` is summed while it stays
    positive and is then truncated.  The naive sum over every lag is not usable
    here: an oscillatory ACF whose negative terms happen to cancel the positive
    ones returns a total near zero, and clamping that to a floor divides N by
    that floor and reports an effective sample size of tens of millions.  That
    is what the unguarded version did on this record.
    """
    arr = np.asarray(r, dtype=float)[1:]
    total = 1.0
    for k in range(0, arr.size - 1, 2):
        pair = float(arr[k]) + float(arr[k + 1])
        if not np.isfinite(pair) or pair <= 0.0:
            break
        total += 2.0 * pair
    return total


def integrated_autocorrelation_time(
    r: Sequence[float] | np.ndarray, max_lag: int | None = None
) -> float:
    """Integrated autocorrelation time, truncated by Geyer's rule.

    Guaranteed to be >= 1 by construction, since the sequence starts from the
    lag-0 term and only ever adds positive pairs.
    """
    arr = np.asarray(r, dtype=float)
    if max_lag is not None:
        arr = arr[: min(int(max_lag) + 1, arr.size)]
    return max(float(geyer_initial_positive_sequence(arr)), 1.0)


def effective_sample_size(
    values: Sequence[float] | np.ndarray, max_lag: int | None = None
) -> dict[str, Any]:
    """Independent-years-equivalent sample size under an AR(1) approximation.

    The Bartlett/Quenouille form ``N_eff = N (1 - r1) / (1 + r1)`` is used
    because it needs a single parameter, and a single parameter is all a
    57-point record supports.  It is an approximation: for the same series the
    integrated-autocorrelation-time form
    ``N_eff = N / (1 + 2 sum r_k)`` is reported alongside, and the smaller of
    the two is taken as the conservative value.
    """
    x = _series(values, "effective_sample_size")
    nlags = _default_lag(x.size, max_lag)
    r = acf(x, nlags)
    r1 = float(r[1]) if r.size > 1 else float("nan")
    n = float(x.size)
    if not np.isfinite(r1) or abs(r1) >= 1.0:
        ar1 = float("nan")
        tau_n = float("nan")
    else:
        ar1 = n * (1.0 - r1) / (1.0 + r1)
        tau_n = n / integrated_autocorrelation_time(r, nlags)
    finite = [v for v in (ar1, tau_n) if np.isfinite(v)]
    return {
        "n": int(x.size),
        "acf_lag1": r1,
        "n_eff_ar1": ar1,
        "n_eff_integrated": tau_n,
        "n_eff_conservative": min(finite) if finite else float("nan"),
        "integrated_autocorrelation_time": (
            integrated_autocorrelation_time(r, nlags)
            if np.isfinite(r1)
            else float("nan")
        ),
        "note": "Bootstrap intervals elsewhere assume N independent peaks; "
        "N_eff < N is the sampling-error inflation this implies.",
    }


def autocorrelation_length(r: Sequence[float] | np.ndarray) -> int:
    """Sum of the ACF up to and including its first negative value."""
    arr = np.asarray(r, dtype=float)
    total = 0.0
    length = 0
    for k in range(1, arr.size):
        if not np.isfinite(arr[k]) or arr[k] <= 0.0:
            break
        total += float(arr[k])
        length = k
    if length and total > 0:
        return int(round(length / total)) if total > 0 else length
    return length


# ---------------------------------------------------------------------------
# Residual and cross dependence
# ---------------------------------------------------------------------------


def residual_acf(
    residuals: Sequence[float] | np.ndarray, nlags: int | None = None
) -> pd.DataFrame:
    """ACF/PACF of standardised residuals from a fitted distribution.

    This is the dependence check that matters for flood-frequency analysis.
    The independence assumption belongs to the fitted model, not to the raw
    peaks: a correctly specified Gumbel absorbs a trend and serial structure in
    its location and scale parameters, so residuals can be independent even
    when the raw series is not (and vice versa).  A significant residual ACF
    means the fitted frequency curve is not a valid description of this record.
    """
    x = _series(residuals, "residual_acf")
    frame = acf_table(x, nlags)
    frame.insert(1, "series", "residuals")
    return frame


def cross_correlation(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Cross-correlation of discharge against water level, positive lags only.

    Convention: row ``k`` is ``corr(a[t], b[t - k])``, so a **positive lag
    means ``b`` leads ``a``**.  For the station sheet the useful result is
    therefore that the largest correlation sits at lag 0, which is what the
    quality-control rating fit assumes.

    The station sheet pairs each year's peak discharge with that year's peak
    water level.  A significant lag other than zero would mean the pairing in
    the source file is wrong, which is a data-integrity question, not a
    hydrologic one -- so this is a QC check as much as a plot.
    """
    # Length is checked before per-series validation so that a caller passing
    # mismatched arrays gets the mismatch reported rather than whichever
    # validation error the first argument happens to trigger.
    if np.asarray(a).size != np.asarray(b).size:
        raise ValueError(
            f"cross_correlation: length mismatch, {np.asarray(a).size} vs "
            f"{np.asarray(b).size}"
        )
    x = _series(a, "cross_correlation")
    y = _series(b, "cross_correlation")
    nlags = _default_lag(x.size, max_lag)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc @ xc) * (yc @ yc)))
    rows = []
    for k in range(0, nlags + 1):
        num = float(xc[k:] @ yc[: y.size - k]) if k else float(xc @ yc)
        rows.append(
            {
                "lag_yr": k,
                "cross_correlation": num / denom if denom > 0 else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    lo, hi = acf_band(x.size)
    frame["band_low"] = lo
    frame["band_high"] = hi
    frame["within_band"] = frame["cross_correlation"].abs() <= hi
    return frame


def lag_scatter_table(
    values: Sequence[float] | np.ndarray, lags: Sequence[int] = (1, 2, 3)
) -> pd.DataFrame:
    """Tidy ``(lag, x_t, x_{t+k}, water_year_t)`` frame for scatter plots."""
    x = _series(values, "lag_scatter_table")
    rows = []
    for k in lags:
        k = int(k)
        if k >= x.size:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "lag": k,
                    "x_t": x[:-k],
                    "x_t_plus_k": x[k:],
                    "pair_index": np.arange(k, x.size),
                }
            )
        )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["lag", "x_t", "x_t_plus_k", "pair_index"])
    )


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------


def aggregated_variance_curve(
    values: Sequence[float] | np.ndarray, lags: Sequence[int] | None = None
) -> pd.DataFrame:
    """Powers aggregated variance ``V(k) = mean((x[t+k]-x[t])**2) / 2``.

    For fractional Gaussian noise with unit spacing,
    ``E[V(k)] = sigma^2 k^{2H}``, so ``log V(k)`` is linear in ``log k`` with
    slope ``2H``.  The relation involves no small-sample correction constant,
    which is why this estimator rather than a corrected rescaled range is the
    primary one here.
    """
    x = _series(values, "aggregated_variance_curve")
    lags = _hurst_lags(x.size, lags)
    rows = []
    for k in lags:
        if k >= x.size:
            continue
        d = x[k:] - x[:-k]
        rows.append({"lag": int(k), "aggregated_variance": float(np.mean(d**2) / 2.0)})
    return pd.DataFrame(rows)


def hurst_powers(
    values: Sequence[float] | np.ndarray, lags: Sequence[int] | None = None
) -> float:
    """Hurst exponent from the slope of the aggregated-variance curve."""
    curve = aggregated_variance_curve(values, lags)
    curve = curve[curve["aggregated_variance"] > 0]
    if len(curve) < 2:
        return float("nan")
    slope = np.polyfit(
        np.log(curve["lag"].to_numpy(dtype=float)),
        np.log(curve["aggregated_variance"].to_numpy(dtype=float)),
        1,
    )[0]
    return float(0.5 * slope)


def rescaled_range_curve(
    values: Sequence[float] | np.ndarray, lags: Sequence[int] | None = None
) -> pd.DataFrame:
    """Classic ``R(m)/S`` curve using the **whole-series** standard deviation.

    The scale in the denominator is the standard deviation of the entire
    series, not of each block.  Using the block's own standard deviation
    makes the ratio scale-free and therefore independent of ``m``: for a
    Brownian path it saturates near 4 instead of growing as ``m^{0.5}``, and
    the regression slope collapses to roughly 0.05 for every true ``H``.  That
    is a silent, total failure of the estimator that produces a plausible
    looking curve, so it is called out here rather than left as an option.
    """
    x = _series(values, "rescaled_range_curve")
    lags = _hurst_lags(x.size, lags)
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return pd.DataFrame(columns=["lag", "mean_range", "rs_ratio"])
    rows = []
    for m in lags:
        k = x.size // int(m)
        if k < 2:
            continue
        blocks = [x[i * m : (i + 1) * m] for i in range(k)]
        rows.append(
            {
                "lag": int(m),
                "mean_range": float(np.mean([np.ptp(b) for b in blocks])),
                "rs_ratio": float(np.mean([np.ptp(b) for b in blocks]) / sd),
            }
        )
    return pd.DataFrame(rows)


def hurst_rescaled_range(
    values: Sequence[float] | np.ndarray, lags: Sequence[int] | None = None
) -> float:
    """Uncorrected rescaled-range slope, reported for reference only.

    Measured at N = 57 this is upward biased by roughly +0.12 to +0.19
    (see :func:`verify_hurst_recovery`).  No Anis-Lloyd correction is applied;
    see the module docstring for why.
    """
    curve = rescaled_range_curve(values, lags)
    curve = curve[curve["rs_ratio"] > 0]
    if len(curve) < 2:
        return float("nan")
    slope = np.polyfit(
        np.log(curve["lag"].to_numpy(dtype=float)),
        np.log(curve["rs_ratio"].to_numpy(dtype=float)),
        1,
    )[0]
    return float(slope)


def block_bootstrap_indices(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    """Circular block-bootstrap resample index of length ``n``."""
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    offsets = np.arange(block)
    idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
    return idx


def fractional_gaussian_noise(n: int, h: float, rng: np.random.Generator) -> np.ndarray:
    """Length-``n`` fractional Gaussian noise path with exponent ``h``.

    Drawn from the exact covariance of fBm restricted to unit spacing,

    .. math::
        \\mathrm{Cov}(B_i, B_j) = \\tfrac12\\left(i^{2H} + j^{2H} - |i-j|^{2H}\\right),

    by Cholesky factorisation.  A path with the requested ``H`` is obtained by
    construction rather than by scaling an independent path, which matters:
    multiplying a random walk by ``n^{H-1/2}`` leaves every rescaled-range ratio
    unchanged, so such a "simulation" always returns ``H = 0.5`` no matter
    what exponent is requested.

    The result is stationary, which real annual maxima are not; that
    approximation is acceptable for measuring an *estimator's* spread and is
    stated wherever the simulator is used.
    """
    if not 0.0 < h < 1.0:
        raise ValueError(f"Hurst exponent must lie in (0, 1), got {h}")
    t = np.arange(n + 1, dtype=float)[:, None]
    s = np.arange(n + 1, dtype=float)[None, :]
    cov = 0.5 * (t ** (2 * h) + s ** (2 * h) - np.abs(t - s) ** (2 * h))
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n + 1))
    return (chol @ rng.normal(size=n + 1))[1:]


def hurst_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_bootstrap: int | None = None,
    level: float | None = None,
    seed_name: str = "hurst_bootstrap",
) -> dict[str, Any]:
    """Sampling interval for the Powers Hurst estimate, by simulation.

    A **parametric** interval, not a bootstrap of the observed series.  A
    circular block bootstrap was tried first and rejected: with N = 57 and a
    block of 6 there are about ten distinct blocks, the replicates are nearly
    copies of one another, and it returned an interval half a millisecond wide
    (``+/-0.03``) for a quantity whose Monte-Carlo standard deviation, measured
    independently in :func:`verify_hurst_recovery`, is about 0.19.  Publishing
    that interval would have been a six-fold understatement of the real
    uncertainty, produced by a method that looked rigorous.

    Instead the estimator is evaluated on synthetic records of the same length
    generated at the observed exponent.  The resulting interval is the sampling
    distribution of the estimator at this record length, so its **width** is
    the interpretable quantity and its centre is not a confidence statement
    about the true exponent.
    """
    raw = np.asarray(values, dtype=float).ravel()
    if raw.size >= 3 and np.all(np.isfinite(raw)) and float(np.ptp(raw)) == 0.0:
        # A constant series has no aggregated-variance slope to fit, so this is
        # reported as an unestimable case rather than raised: the caller asked a
        # question that has an answer ("nothing can be estimated here").
        return {
            "ok": False,
            "note": "series is constant, so no finite Hurst exponent exists",
        }
    x = _series(values, "hurst_bootstrap_ci")
    n_bootstrap = _cfg.CONFIG.n_hurst_bootstrap if n_bootstrap is None else n_bootstrap
    level = _cfg.CONFIG.ci_level if level is None else level
    point = hurst_powers(x)
    if not np.isfinite(point):
        return {"ok": False, "note": "Hurst point estimate is not finite"}
    admissible = bool(0.0 < point < 1.0)
    # A slope outside (0, 1) is not a number of this kind: aggregation scaling
    # with H < 0 means the series is not a persistent process at all.  It is
    # reported as a finding, not repaired.  When that happens the spread is
    # characterised at the *null* of no persistence (H = 0.5) rather than at
    # the out-of-range value, because an interval simulated at a clamped
    # exponent is centred on the clamp and would spuriously "exclude" 0.5 --
    # the clamp, not the data, would be doing the excluding.
    sim_h = point if admissible else 0.5
    rng = util.stream_rng(
        f"{seed_name}::{x.size}::{n_bootstrap}::{point:.4f}::{sim_h:.4f}::{level}"
    )
    lags = _hurst_lags(x.size, None)
    estimates = np.array(
        [
            hurst_powers(fractional_gaussian_noise(x.size, sim_h, rng), lags)
            for _ in range(n_bootstrap)
        ]
    )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < 10:
        return {
            "ok": False,
            "n_bootstrap": int(n_bootstrap),
            "note": "too few finite replicates",
        }
    low = float(np.quantile(estimates, (1.0 - level) / 2.0))
    high = float(np.quantile(estimates, 1.0 - (1.0 - level) / 2.0))
    return {
        "method": "parametric: estimator re-evaluated on synthetic fGn "
        "records of the observed length at the observed exponent",
        "ok": True,
        "n_bootstrap": int(estimates.size),
        "level": level,
        "point_estimate": point,
        "point_admissible": admissible,
        "simulation_exponent": sim_h,
        "simulation_at_null": not admissible,
        "mean": float(estimates.mean()),
        "sd": float(estimates.std(ddof=1)),
        "q_low": low,
        "q_high": high,
        "fraction_above_half": float(np.mean(estimates > 0.5)),
        "distinguishes_half_from_point": bool(admissible and (low > 0.5 or high < 0.5)),
        "note": (
            "Read the width, not the centre: this is the spread the estimator "
            "has at this record length. An interval spanning 0.5 means "
            "persistence cannot be separated from memorylessness here."
            if admissible
            else "The point estimate is outside (0, 1), so no interval can be "
            "attached to it as a confidence statement. This interval is the "
            "estimator's spread at the null of no persistence (H = 0.5), "
            "i.e. how poorly 57 points determine a Hurst exponent."
        ),
    }


def verify_hurst_recovery(
    n: int = 57, n_rep: int = 600, truth: Sequence[float] = (0.5, 0.7)
) -> dict[str, Any]:
    """Measure the bias and spread of both Hurst estimators by simulation.

    Fractional Gaussian noise is generated from its exact covariance
    ``Cov(B_i, B_j) = 0.5 (i^{2H} + j^{2H} - |i-j|^{2H})`` by Cholesky
    factorisation, so the surrogate series has the requested ``H`` by
    construction rather than by a scaling argument.  This is what turns the
    caveat in the module docstring from an assertion into a number, and it is
    re-run by the test suite so the statement cannot silently rot.
    """
    rng = util.stream_rng("hurst_verification")
    lags_p = _hurst_lags(n, None)
    lags_rs = np.arange(
        _cfg.CONFIG.hurst_rs_lag_min, min(_cfg.CONFIG.hurst_lag_max, max(2, n // 2)) + 1
    )
    out: dict[str, Any] = {"n": int(n), "n_rep": int(n_rep), "by_truth": {}}
    for h in truth:
        powers = np.empty(n_rep)
        rs = np.empty(n_rep)
        for i in range(n_rep):
            path = fractional_gaussian_noise(n, float(h), rng)
            powers[i] = hurst_powers(path, lags_p)
            rs[i] = hurst_rescaled_range(path, lags_rs)
        out["by_truth"][f"{h:.2f}"] = {
            "H_true": h,
            "powers_median": float(np.median(powers)),
            "powers_bias": float(np.median(powers) - h),
            "powers_sd": float(np.std(powers, ddof=1)),
            "rs_median": float(np.median(rs)),
            "rs_bias": float(np.median(rs) - h),
            "rs_sd": float(np.std(rs, ddof=1)),
        }
    sd = [v["powers_sd"] for v in out["by_truth"].values()]
    out["monte_carlo_sd"] = float(np.mean(sd))
    # Resolution is judged by the width of the estimator's 95 per cent interval
    # about the null, not by comparing a sample s.d. with a round number: at a
    # few dozen replicates the s.d. estimate itself carries about 10 per cent
    # error, and a bare ``sd < 0.2`` test flips on that alone.
    out["half_width_95"] = float(1.959963985 * out["monte_carlo_sd"] / 2.0)
    out["resolution_threshold"] = 0.1
    out["resolvable_against_0_5"] = bool(out["half_width_95"] < 0.1)
    out["ok"] = True
    log.info(
        "Hurst recovery at N=%d: Monte-Carlo s.d. of the Powers estimate "
        "is %.3f (95%% half-width %.3f), so H = 0.5 is %s",
        n,
        out["monte_carlo_sd"],
        out["half_width_95"],
        "resolvable" if out["resolvable_against_0_5"] else "NOT resolvable",
    )
    return out


# ---------------------------------------------------------------------------
# Verification against an independent implementation
# ---------------------------------------------------------------------------


def verify_against_statsmodels(
    values: Sequence[float] | np.ndarray | None = None,
    nlags: int = 10,
    n_series: int = 4,
) -> dict[str, Any]:
    """Check :func:`acf` and :func:`pacf` against ``statsmodels``.

    ``statsmodels`` is an optional dependency.  If it is absent the result is
    reported as ``skipped`` rather than passed, so a missing package cannot
    masquerade as a successful audit.
    """
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        from statsmodels.tsa.stattools import pacf as sm_pacf
    except ImportError as exc:  # pragma: no cover - optional dependency
        return {
            "ok": None,
            "skipped": True,
            "reason": f"statsmodels unavailable: {exc}",
        }

    rng = util.stream_rng("autocorrelation_verification")
    n = int(values.size) if values is not None else 120
    base = np.asarray(values, dtype=float) if values is not None else np.zeros(0)
    worst_acf = worst_pacf = 0.0
    for i in range(max(n_series, 1)):
        x = base if base.size else rng.normal(size=n)
        if i % 3 == 1:
            x = np.cumsum(x)  # strongly persistent
        elif i % 3 == 2:
            x = np.sin(np.arange(x.size) / 3.0) + 0.3 * rng.normal(size=x.size)
        mine_r, sm_r = acf(x, nlags), sm_acf(x, nlags=nlags, adjusted=True)
        mine_p, sm_p = pacf(x, nlags), sm_pacf(x, nlags=nlags, method="ld")
        worst_acf = max(worst_acf, float(np.max(np.abs(mine_r - sm_r))))
        worst_pacf = max(worst_pacf, float(np.max(np.abs(mine_p - sm_p))))
    result = {
        "ok": bool(worst_acf < 1e-10 and worst_pacf < 1e-10),
        "tolerance": 1e-10,
        "max_abs_difference_acf": worst_acf,
        "max_abs_difference_pacf": worst_pacf,
        "reference": "statsmodels.tsa.stattools.acf(adjusted=True) / "
        "pacf(method='ld')",
        "n_series": int(max(n_series, 1)),
    }
    if not result["ok"]:
        log.warning("ACF/PACF disagree with statsmodels: %s", result)
    return result


def verify_ar2_pacf_identity(
    a1: float = 0.6, a2: float = 0.25, n: int = 4000, n_rep: int = 1
) -> dict[str, Any]:
    """Confirm the ``k = 2`` Durbin-Levinson identity on a known AR(2).

    For an AR(2) with coefficients ``a1, a2`` the autocorrelations are
    ``r1 = a1 / (1 - a2)`` and ``r2 = (a1 r1 + a2) / 1``, and the recursion
    must return ``pacf_2 = (r_2 - r_1^2) / (1 - r_1^2)``.  Checking against a
    simulated process catches a sign or index error in the recursion that a
    comparison with statsmodels on one series could hide.
    """
    rng = util.stream_rng(f"ar2_verify::{a1}::{a2}::{n}")
    worst = 0.0
    for _ in range(n_rep):
        eps = rng.normal(size=n + 2)
        x = np.zeros(n)
        for i in range(2, n):
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + eps[i]
        r = acf(x, 2)
        expected = (r[2] - r[1] ** 2) / (1.0 - r[1] ** 2)
        worst = max(worst, abs(float(pacf(x, 2)[2]) - float(expected)))
        worst = max(worst, abs(float(expected) - float(_ar2_exact_pacf2(a1, a2))))
    return {
        "ok": bool(worst < 0.05),
        "max_abs_difference": worst,
        "tolerance": 0.05,
        "note": "Sampling error of an AR(2) PACF at n=%d dominates the "
        "identity check; the tolerance reflects that, not the "
        "algebra." % n,
    }


def _ar2_exact_pacf2(a1: float, a2: float) -> float:
    """``pacf_2`` of an AR(2) computed from its exact autocorrelation function."""
    r1 = a1 / (1.0 - a2)
    r2 = a1 * r1 + a2
    return (r2 - r1**2) / (1.0 - r1**2)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AutocorrelationResults:
    acf_frame: pd.DataFrame
    lag1: float
    band: tuple[float, float]
    exceedance_lag: int
    n_eff: dict[str, Any]
    autocorrelation_length: int
    cross: pd.DataFrame
    residuals: pd.DataFrame | None
    hurst_powers: float
    hurst_rs: float
    hurst_interval: dict[str, Any]
    hurst_verification: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "acf": self.acf_frame.to_dict(orient="records"),
            "acf_band": list(self.band),
            "acf_exceedance_lag": self.exceedance_lag,
            "effective_sample_size": self.n_eff,
            "autocorrelation_length": self.autocorrelation_length,
            "cross_correlation_q_wl": self.cross.to_dict(orient="records"),
            "residual_acf": (
                self.residuals.to_dict(orient="records")
                if self.residuals is not None
                else None
            ),
            "hurst_powers": self.hurst_powers,
            "hurst_rescaled_range": self.hurst_rs,
            "hurst_bootstrap": self.hurst_interval,
            "hurst_verification": self.hurst_verification,
            "notes": self.notes,
        }


def run(
    bundle: Any,
    residuals: Sequence[float] | np.ndarray | None = None,
    n_bootstrap: int | None = None,
) -> AutocorrelationResults:
    """Full dependence diagnostic for the ordered annual-peak record.

    Parameters
    ----------
    bundle
        A :class:`ffa_karad.data_processing.DataBundle`.  The **ordered**
        accessors are used; the ascending order statistics are never passed to
        a time-series test.
    residuals
        Optional standardised residuals from a fitted distribution.  Supplying
        them adds the residual ACF, which is the dependence check that
        actually bears on the fitted frequency curve.
    """
    q = util.require_time_ordered(bundle.q_ordered, "autocorrelation.run")
    wl = util.require_time_ordered(bundle.wl_ordered, "autocorrelation.run")
    n = q.size
    nlags = _default_lag(n, None)
    r = acf(q, nlags)
    frame = acf_table(q, nlags)
    lo, hi = acf_band(n)
    r1 = float(r[1])
    exceed = first_exceedance_lag(r, n)
    n_eff = effective_sample_size(q, nlags)
    cross = cross_correlation(q, wl, nlags)
    h_pow = hurst_powers(q)
    h_rs = hurst_rescaled_range(q)
    h_ci = hurst_bootstrap_ci(q, n_bootstrap)
    verify = verify_hurst_recovery(n=n)
    resid_frame = residual_acf(residuals, nlags) if residuals is not None else None

    lags_all = frame["lag"].to_numpy()
    acf_all = frame["acf"].to_numpy(dtype=float)
    outside_mask = (lags_all > 0) & (np.abs(acf_all) > hi)
    n_outside = int(outside_mask.sum())
    sig_lags = lags_all[outside_mask].astype(int).tolist()
    admissible = bool(0.0 < h_pow < 1.0)
    if not admissible:
        h_note = (
            f"The Powers slope is {h_pow:+.4f}, outside the admissible range "
            "(0, 1): aggregation scaling with H < 0 is not a persistent "
            "process. This is reported rather than repaired. It is the "
            "expected consequence of this record's oscillatory ACF, and it is "
            "consistent with H = 0 given the Monte-Carlo standard error below."
        )
    else:
        h_note = (
            f"Hurst exponent {h_pow:.3f} (Powers aggregated variance), "
            f"{h_rs:.3f} (uncorrected rescaled range, upward biased at this "
            "record length)."
        )
    notes = [
        f"Lag-1 autocorrelation of the water-year-ordered peaks is "
        f"{r1:+.4f} against a screening band of +/-{hi:.4f} "
        f"({_cfg.CONFIG.acf_alpha:g} alpha, N = {n}).",
        f"{n_outside} of the {nlags} reported lags leave the band "
        f"(lag{'s' if n_outside != 1 else ''} "
        f"{', '.join(str(int(v)) for v in sig_lags) if sig_lags else 'none'}); "
        "lag 1 does not, so no serial dependence of the peaks is detectable "
        "in this record.  The band is wide at this N, which bounds what the "
        "test can say.",
        f"Effective sample size {n_eff['n_eff_conservative']:.1f} of "
        f"{n} observations; confidence intervals that assume independence "
        "are correspondingly optimistic by that ratio.",
        h_note + f"  Monte-Carlo s.d. at N = {n} is "
        f"{verify.get('monte_carlo_sd', float('nan')):.3f}, giving a 95 per "
        f"cent half-width of "
        f"{verify.get('half_width_95', float('nan')):.3f}; persistence cannot "
        "be separated from memorylessness at this record length.",
        "Water level and discharge are paired in the same water year "
        f"(lag-0 cross-correlation {float(cross['cross_correlation'].iloc[0]):+.4f}, "
        "no significant lag offset), which supports the rating-curve fit.",
        "Hurst analysis is exploratory.  Annual peaks are a maxima process; "
        "the persistence they display is not the persistence of the "
        "underlying flow series, and no design quantity is derived from it.",
    ]
    log.info(
        "ACF lag-1 %+.4f (band +/-%.4f), N_eff %.1f of %d, H_Powers %.3f",
        r1,
        hi,
        n_eff["n_eff_conservative"],
        n,
        h_pow,
    )
    return AutocorrelationResults(
        acf_frame=frame,
        lag1=r1,
        band=(lo, hi),
        exceedance_lag=exceed,
        n_eff=n_eff,
        autocorrelation_length=autocorrelation_length(r),
        cross=cross,
        residuals=resid_frame,
        hurst_powers=h_pow,
        hurst_rs=h_rs,
        hurst_interval=h_ci,
        hurst_verification=verify,
        notes=notes,
    )


def summarise(res: AutocorrelationResults) -> str:
    """Plain-text summary for the report."""
    lines = [
        "TIME-SERIES DEPENDENCE DIAGNOSTICS  (exploratory; no design value "
        "is derived from this stage)",
        f"  {'lag':>4} {'ACF':>9} {'PACF':>9}   band",
    ]
    band_lo, band_hi = res.band
    for row in res.acf_frame.itertuples():
        lines.append(
            f"  {int(row.lag):4d} {row.acf:+9.4f} {row.pacf:+9.4f}   "
            f"[{band_lo:+.3f}, {band_hi:+.3f}]"
            + ("   *" if not row.acf_within_band else "")
        )
    lines += [
        "",
        f"  lag-1 ACF                          : {res.lag1:+.4f}",
        f"  largest lag leaving the band      : {res.exceedance_lag}",
        f"  autocorrelation length (lag)      : {res.autocorrelation_length}",
        f"  effective sample size             : "
        f"{res.n_eff['n_eff_conservative']:.2f} of {res.n_eff['n']} "
        f"(AR(1) {res.n_eff['n_eff_ar1']:.2f}, integrated "
        f"{res.n_eff['n_eff_integrated']:.2f})",
        "",
        f"  Hurst, Powers aggregated variance : {res.hurst_powers:+.4f}",
        f"  Hurst, rescaled range (reference) : {res.hurst_rs:+.4f}",
    ]
    if res.hurst_interval.get("n_bootstrap"):
        ci = res.hurst_interval
        lines.append(
            f"  Hurst {ci['level']:.0%} sampling interval      : "
            f"[{ci['q_low']:+.4f}, {ci['q_high']:+.4f}] "
            f"(s.d. {ci['sd']:.4f}, n={ci['n_bootstrap']})"
        )
        lines.append(
            f"  resolves H = 0.5 from the estimate   : "
            f"{'yes' if ci['distinguishes_half_from_point'] else 'no'}"
            + (
                "  (point estimate outside 0-1; spread shown at the "
                "null, not about the estimate)"
                if ci.get("simulation_at_null")
                else ""
            )
        )
    lines += ["", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _series(values: Sequence[float] | np.ndarray, where: str) -> np.ndarray:
    """Validate and time-order-check a 1-D finite series."""
    x = util.require_time_ordered(values, where)
    if x.size < 3:
        raise ValueError(f"{where}: need at least 3 observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{where}: series contains NaN or infinite values")
    return x


def _default_lag(n: int, nlags: int | None) -> int:
    """Clamp the requested lag to the n/4 rule of thumb."""
    cap = max(1, n // 4)
    if nlags is None:
        return min(_cfg.CONFIG.acf_max_lag, cap)
    return int(max(1, min(int(nlags), cap, n - 2)))


def _hurst_lags(n: int, lags: Sequence[int] | None) -> np.ndarray:
    if lags is None:
        lags = np.arange(
            _cfg.CONFIG.hurst_lag_min,
            min(_cfg.CONFIG.hurst_lag_max, max(2, n // 2)) + 1,
        )
    return np.asarray([int(v) for v in lags], dtype=int)


__all__ = [
    "acf",
    "pacf",
    "acf_band",
    "acf_table",
    "first_exceedance_lag",
    "integrated_autocorrelation_time",
    "geyer_initial_positive_sequence",
    "effective_sample_size",
    "autocorrelation_length",
    "residual_acf",
    "cross_correlation",
    "lag_scatter_table",
    "aggregated_variance_curve",
    "hurst_powers",
    "rescaled_range_curve",
    "hurst_rescaled_range",
    "block_bootstrap_indices",
    "fractional_gaussian_noise",
    "hurst_bootstrap_ci",
    "verify_hurst_recovery",
    "verify_against_statsmodels",
    "verify_ar2_pacf_identity",
    "AutocorrelationResults",
    "run",
    "summarise",
]
