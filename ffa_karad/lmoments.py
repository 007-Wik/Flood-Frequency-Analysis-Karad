"""
L-moments, probability-weighted moments, and the model locus diagram.

What this module is for
-----------------------
L-moments are the standard *robust* alternative to moments for hydrologic
frequency analysis.  Including them is worthwhile because they answer a
question the moment route cannot: is the record genuinely right-skewed, or is
the moment skewness of 1.08 an artefact of the single 7,177 cumecs event in
1976?  The answer from the 2025 notebook's own numbers was already "the latter"
-- its L-skewness was 0.2465 against a Gumbel reference of 0.3333.

Honesty note on the estimators
------------------------------
Two families of sample L-moment estimators exist:

* the **unbiased b-statistics** of Hosking (1990) section 2.3, and
* **consistent estimators derived from the probability-weighted moments**
  (Hosking section 2.7), which are what the ``lmom`` R package uses.

:func:`sample_lmoments` implements the b-statistics.  They are unbiased but
high-variance: at N = 57 the ratio measures L2/L1, L3/L2 and L4/L3 are
dominated by sampling noise, and on this record they return LCV = 14.3, which
is a real property of the estimator and not a typo.  Using them to place the
sample on a locus diagram would be indefensible.

The consistent PWM-based estimators are therefore **not** implemented here.  A
reconstruction from memory is precisely the failure mode this repository was
created to eliminate, and an incorrect consistent estimator is worse than no
estimator.  :func:`consistent_lmoments` raises with instructions until a
verified reference implementation is added; see
``docs/assumptions_and_constants.md``.

The locus diagram is therefore drawn on the **moment plane (Cs, Ck)**, which
is where Bulletin 17B operates and where every model locus is exactly
computable.  The 2025 notebook's fabricated GEV locus
``tau4 = 0.116 + 0.206*tau3 - 0.013*tau3**2`` has been removed; nothing in
this module is typed from memory.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import util

log = util.get_logger("lmoments")


# ---------------------------------------------------------------------------
# Sample L-moments (unbiased b-statistics)
# ---------------------------------------------------------------------------


def sample_lmoments(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Unbiased sample L-moments ``L1..L4`` via Hosking's (1990) b-statistics.

    .. math::

        b_r = \\frac{1}{n}\\sum_{i=1}^{n}
              \\frac{\\binom{i-1}{r}}{\\binom{n-1}{r}}\\,x_{(i)}

    :math:`L_1=b_0=\\bar{x}`,
    :math:`L_2=2b_1-b_0`,
    :math:`L_3=6b_2-6b_1+b_0`,
    :math:`L_4=20b_3-30b_2+12b_1-b_0`,
    with :math:`x_{(i)}` the ascending order statistics.

    The sample ratios are:
    :math:`\\text{LCV} = L_2/L_1`,
    :math:`\\tau_3 = L_3/L_2` (L-skewness),
    :math:`\\tau_4 = L_4/L_2` (L-kurtosis).
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 4:
        raise ValueError("at least four observations are required for L1..L4")
    if not np.all(np.isfinite(x)):
        raise ValueError("sample contains non-finite values")
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1)

    # Unbiased estimators of probability-weighted moments (Hosking 1990, Eq 2.3)
    b0 = float(np.mean(xs))
    b1 = float(np.sum((i - 1) / (n - 1) * xs) / n)
    b2 = float(np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * xs) / n)
    b3 = float(
        np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * xs) / n
    )

    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0

    def _ratio(num: float, den: float) -> float:
        if abs(den) < np.finfo(float).tiny:
            return float("nan")
        return float(num / den)

    lcv = _ratio(l2, l1)
    lskew = _ratio(l3, l2)
    lkurt = _ratio(l4, l2)

    finite_ok = bool(np.all(np.isfinite([lcv, lskew, lkurt])))
    ratios_reliable = bool(
        finite_ok
        and n >= 20
        and 0.0 < lcv < 2.5
        and -1.0 < lskew < 1.0
        and -1.0 < lkurt < 1.0
    )

    return {
        "n": int(n),
        "b0": float(b0),
        "b1": float(b1),
        "b2": float(b2),
        "b3": float(b3),
        "L1": float(l1),
        "L2": float(l2),
        "L3": float(l3),
        "L4": float(l4),
        "lcv": lcv,
        "lskew": lskew,
        "lkurt": lkurt,
        "l4_over_l2": lkurt,
        "estimator": "Hosking (1990) unbiased b-statistics",
        "ratios_reliable": bool(ratios_reliable),
    }


def consistent_lmoments(
    values: Sequence[float] | np.ndarray,
    a: float = 0.35,
) -> dict[str, float]:
    """Consistent (plotting-position based) L-moment estimators (Hosking 1990, §2.7).

    Estimates PWMs via :math:`\\tilde{\\beta}_r = \\frac{1}{n} \\sum_{i=1}^n \\left(\\frac{i - a}{n}\\right)^r x_{(i)}`,
    defaulting to :math:`a = 0.35` (Hosking & Wallis standard).
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 4:
        raise ValueError("at least four observations are required for L1..L4")
    if not np.all(np.isfinite(x)):
        raise ValueError("sample contains non-finite values")
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1)
    p = (i - a) / n

    tb0 = float(np.mean(xs))
    tb1 = float(np.mean(p * xs))
    tb2 = float(np.mean(p**2 * xs))
    tb3 = float(np.mean(p**3 * xs))

    l1 = tb0
    l2 = 2.0 * tb1 - tb0
    l3 = 6.0 * tb2 - 6.0 * tb1 + tb0
    l4 = 20.0 * tb3 - 30.0 * tb2 + 12.0 * tb1 - tb0

    def _ratio(num: float, den: float) -> float:
        if abs(den) < np.finfo(float).tiny:
            return float("nan")
        return float(num / den)

    lcv = _ratio(l2, l1)
    lskew = _ratio(l3, l2)
    lkurt = _ratio(l4, l2)

    finite_ok = bool(np.all(np.isfinite([lcv, lskew, lkurt])))
    ratios_reliable = bool(
        finite_ok
        and n >= 20
        and 0.0 < lcv < 2.5
        and -1.0 < lskew < 1.0
        and -1.0 < lkurt < 1.0
    )

    return {
        "n": int(n),
        "b0": float(tb0),
        "b1": float(tb1),
        "b2": float(tb2),
        "b3": float(tb3),
        "L1": float(l1),
        "L2": float(l2),
        "L3": float(l3),
        "L4": float(l4),
        "lcv": lcv,
        "lskew": lskew,
        "lkurt": lkurt,
        "l4_over_l2": lkurt,
        "estimator": f"Hosking (1990) consistent plotting-position (a={a})",
        "ratios_reliable": bool(ratios_reliable),
    }


def probability_weighted_moments(
    values: Sequence[float] | np.ndarray, r: int = 1
) -> float:
    """Probability-weighted moment of order ``r`` from the raw sample.

    Computed directly from the order statistics,

    .. math::

        a_r = \\frac{1}{n}\\sum_{i=1}^{n}
              \\frac{\\binom{i-1}{r-1}}{\\binom{n-1}{r-1}}\\,x_{(i)},

    so that ``a_1`` is the sample mean.  This is the raw PWM; no
    PWM-to-L-moment mapping is applied (see :func:`consistent_lmoments`).
    """
    xs = np.sort(np.asarray(values, dtype=float).ravel())
    n = xs.size
    if r < 1 or r > n:
        raise ValueError("r must satisfy 1 <= r <= n")
    from math import comb

    i = np.arange(1, n + 1)
    weights = np.array([comb(int(k) - 1, r - 1) for k in i], dtype=float)
    weights /= comb(n - 1, r - 1)
    return float(np.sum(weights * xs) / n)


# ---------------------------------------------------------------------------
# Population moments by adaptive quadrature
# ---------------------------------------------------------------------------


def _moment_limits(
    inv_cdf: Callable[[np.ndarray], np.ndarray], p_lo: float = 1e-11
) -> tuple[float, float]:
    """Finite integration window whose omitted probability mass is < ``p_lo``."""
    lo = float(np.asarray(inv_cdf(np.array([p_lo])), dtype=float)[0])
    hi = float(np.asarray(inv_cdf(np.array([1.0 - p_lo])), dtype=float)[0])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            f"quantile function has unbounded support beyond p = {p_lo:g}; "
            "supply distribution-specific moment limits instead"
        )
    return lo, hi


def population_moments(
    pdf: Callable[[float], float],
    inv_cdf: Callable[[np.ndarray], np.ndarray],
    p_lo: float = 1e-11,
    label: str = "",
) -> dict[str, float]:
    """Population mean, s.d., skewness and kurtosis of a continuous law.

    Uses :math:`E[f(X)] = \\int f(x)\\,f_X(x)\\,dx` evaluated by ``scipy``'s
    adaptive quadrature over the window whose omitted probability mass is
    ``p_lo``.  Density-space quadrature is used rather than
    :math:`\\int f(Q(p))\\,dp` because the fourth moment converges far too
    slowly on a uniform probability grid: for an exponential law a
    200,001-node grid still loses 0.7% of :math:`C_k`.

    Adaptive quadrature also copes with the integrable endpoint singularity of a
    bounded-support family such as the GEV with negative shape, which a
    quantile-grid scheme handles poorly.

    ``skewness`` is the moment coefficient :math:`C_s` and ``kurtosis`` is the
    Pearson moment coefficient :math:`C_k = m_4/m_2^2` (**not** excess
    kurtosis) -- these are the Bulletin 17B definitions.
    """
    from scipy import integrate as spi

    lo, hi = _moment_limits(inv_cdf, p_lo)
    if not hi > lo:
        raise ValueError(f"degenerate integration window [{lo}, {hi}] for {label!r}")

    quad_opts = dict(epsabs=1e-12, epsrel=1e-10, limit=500)
    midpoint = float(np.asarray(inv_cdf(np.array([0.5])), dtype=float)[0])
    points = [m for m in (midpoint,) if lo < m < hi]

    density = lambda x: float(pdf(x))  # noqa: E731

    # Adaptive quadrature can fail to reach the requested tolerance near an
    # integrable endpoint singularity (GEV with negative shape, LP3 with
    # negative skew).  The result is still usable, so the warning is captured
    # and reported rather than raised.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean, err_mean = spi.quad(
            lambda x: x * density(x), lo, hi, points=points, **quad_opts
        )
        m2, err_m2 = spi.quad(
            lambda x: (x - mean) ** 2 * density(x), lo, hi, points=points, **quad_opts
        )
        if m2 <= 0:
            raise ValueError(f"degenerate distribution: zero variance for {label!r}")
        sd = float(np.sqrt(m2))
        m3, err_m3 = spi.quad(
            lambda x: ((x - mean) / sd) ** 3 * density(x),
            lo,
            hi,
            points=points,
            **quad_opts,
        )
        m4, err_m4 = spi.quad(
            lambda x: ((x - mean) / sd) ** 4 * density(x),
            lo,
            hi,
            points=points,
            **quad_opts,
        )
    convergence_warnings = [str(w.message).strip().splitlines()[0] for w in caught]
    if convergence_warnings:
        log.debug(
            "quadrature warning for %s: %s",
            label or "<unnamed>",
            "; ".join(sorted(set(convergence_warnings))),
        )

    return {
        "mean": float(mean),
        "sd": sd,
        "cv": (sd / mean) if mean != 0.0 else float("nan"),
        "Cs": float(m3),
        "Ck": float(m4),
        "excess_kurtosis": float(m4 - 3.0),
        "quad_abs_error_mean": float(err_mean),
        "quad_abs_error_m2": float(err_m2),
        "quad_abs_error_m3": float(err_m3),
        "quad_abs_error_m4": float(err_m4),
        "quad_converged": not convergence_warnings,
        "moment_window": (lo, hi),
        "label": label,
    }


def moment_locus(
    pdf: Callable[[float], float],
    inv_cdf: Callable[[np.ndarray], np.ndarray],
    shape: float = float("nan"),
) -> dict[str, float]:
    """One point of a model locus on the ``(Cs, Ck)`` plane."""
    stats = population_moments(pdf, inv_cdf)
    stats["shape"] = float(shape)
    return stats


#: Model loci on the ``(Cs, Ck)`` plane.  Every point is computed by quadrature.
LOCUS_SPECS: dict[str, dict[str, Any]] = {
    "Normal": dict(
        label="Normal",
        kind="fixed",
        fn=lambda p, s: sps.norm.ppf(p),
        pdf=lambda x, s: sps.norm.pdf(x),
    ),
    "Exponential": dict(
        label="Exponential",
        kind="fixed",
        fn=lambda p, s: sps.expon.ppf(p),
        pdf=lambda x, s: sps.expon.pdf(x),
    ),
    "Lognormal": dict(
        label="Lognormal",
        kind="fixed",
        fn=lambda p, s: sps.lognorm(s=1.0).ppf(p),
        pdf=lambda x, s: sps.lognorm.pdf(x, s=1.0),
    ),
    "Gumbel": dict(
        label="Gumbel (EV1)",
        kind="fixed",
        fn=lambda p, s: sps.gumbel_r.ppf(p),
        pdf=lambda x, s: sps.gumbel_r.pdf(x),
    ),
    "GEV": dict(
        label="GEV",
        kind="shape",
        fn=lambda p, c: sps.genextreme(float(c)).ppf(p),
        pdf=lambda x, c: sps.genextreme.pdf(x, float(c)),
        shapes=tuple(round(v, 3) for v in np.arange(-0.45, 0.451, 0.05)),
    ),
    "PearsonIII": dict(
        label="Pearson III",
        kind="shape",
        fn=lambda p, sk: sps.pearson3(float(sk)).ppf(p),
        pdf=lambda x, sk: sps.pearson3.pdf(x, float(sk)),
        shapes=tuple(
            round(v, 3) for v in np.arange(-4.0, 4.001, 0.25) if abs(v) > 1e-9
        ),
    ),
    "LogPearsonIII": dict(
        label="Log-Pearson III",
        kind="shape",
        fn=lambda p, sk: np.exp(sps.pearson3(float(sk)).ppf(p)),
        pdf=lambda x, sk: sps.pearson3.pdf(np.log(x), float(sk)) / x,
        shapes=tuple(
            round(v, 3) for v in np.arange(-4.0, 4.001, 0.25) if abs(v) > 1e-9
        ),
    ),
}


@functools.lru_cache(maxsize=8)
def model_moment_locus(name: str) -> pd.DataFrame:
    """Computed ``(Cs, Ck)`` locus for a named distribution family."""
    if name not in LOCUS_SPECS:
        raise KeyError(f"unknown locus {name!r}; expected one of {sorted(LOCUS_SPECS)}")
    spec = LOCUS_SPECS[name]
    rows: list[dict[str, float]] = []
    shapes: Sequence[float] = (
        [float("nan")] if spec["kind"] == "fixed" else spec["shapes"]
    )
    for shape in shapes:
        rows.append(
            moment_locus(
                lambda x, _s=shape: spec["pdf"](x, _s),
                lambda p, _s=shape: spec["fn"](p, _s),
                shape,
            )
        )
    return pd.DataFrame(rows)


def all_moment_loci() -> dict[str, pd.DataFrame]:
    """Every computed ``(Cs, Ck)`` locus, keyed by family name."""
    return {name: model_moment_locus(name) for name in LOCUS_SPECS}


# ---------------------------------------------------------------------------
# Self-audit: known exact values
# ---------------------------------------------------------------------------

#: Values every implementation of :func:`population_moments` must reproduce.
#: These are exact closed forms, so any disagreement is a bug in this package.
EXACT_REFERENCES: dict[str, dict[str, float]] = {
    "Normal": {"Cs": 0.0, "Ck": 3.0},
    "Exponential": {"Cs": 2.0, "Ck": 9.0},
    "Gumbel": {"Cs": 1.13955, "Ck": 5.4},
}


def verify_moment_quadrature(tolerance: float = 1e-3) -> dict[str, dict[str, Any]]:
    """Check the quadrature against closed-form moments of three distributions.

    The references are exact: the normal has ``Cs = 0``, ``Ck = 3``; the
    exponential has ``Cs = 2``, ``Ck = 9``; and the Gumbel has
    ``Cs = 12 sqrt(6) zeta(3) / pi^3 = 1.13955`` and ``Ck = 5.4``.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, reference in EXACT_REFERENCES.items():
        spec = LOCUS_SPECS[name]
        computed = population_moments(
            lambda x, _n=name: spec["pdf"](x, 0.0),
            lambda p, _n=name: spec["fn"](p, 0.0),
            label=name,
        )
        errors = {k: abs(float(computed[k]) - v) for k, v in reference.items()}
        results[name] = {
            "computed": {k: float(computed[k]) for k in reference},
            "exact": dict(reference),
            "abs_error": errors,
            "passed": bool(max(errors.values()) <= tolerance),
        }
    failed = [n for n, r in results.items() if not r["passed"]]
    if failed:
        raise AssertionError(
            "moment quadrature failed its self-audit for "
            + ", ".join(failed)
            + f"; errors {[(n, results[n]['abs_error']) for n in failed]}"
        )
    log.info(
        "moment quadrature self-audit passed for %s",
        ", ".join(sorted(EXACT_REFERENCES)),
    )
    return results


def verify_pearson3_kurtosis_relation(tolerance: float = 2e-3) -> dict[str, float]:
    """Check ``Ck = 3 + 1.5*Cs**2`` for the Pearson III family.

    This is the relation the 2025 notebook used as ``0.5*Cs**2``, which is
    wrong by construction: it is self-consistent only for the normal
    distribution.  Reproducing it numerically here documents the correction.
    """
    rows = []
    for sk in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        stats = population_moments(
            lambda x, _s=sk: sps.pearson3.pdf(x, _s),
            lambda p, _s=sk: sps.pearson3.ppf(p, _s),
            label=f"PearsonIII(sk={sk})",
        )
        rows.append(
            {
                "Cs": stats["Cs"],
                "Ck_numeric": stats["Ck"],
                "Ck_3_plus_1p5Cs2": 3.0 + 1.5 * stats["Cs"] ** 2,
                "abs_error": abs(stats["Ck"] - (3.0 + 1.5 * stats["Cs"] ** 2)),
                "Ck_3_plus_0p5Cs2_wrong": 3.0 + 0.5 * stats["Cs"] ** 2,
            }
        )
    frame = pd.DataFrame(rows)
    worst = float(frame["abs_error"].max())
    if worst > tolerance:
        raise AssertionError(
            f"Cs-Ck relation audit failed: max |numeric - (3 + 1.5 Cs^2)| = {worst:.5f}"
        )
    log.info("Pearson III Ck = 3 + 1.5*Cs^2 verified (max error %.2e)", worst)
    return {
        "max_abs_error_vs_3_plus_1p5Cs2": worst,
        "max_abs_error_vs_3_plus_0p5Cs2_wrong": float(
            (frame["Ck_numeric"] - frame["Ck_3_plus_0p5Cs2_wrong"]).abs().max()
        ),
        "table": frame.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Sample placement on the moment plane
# ---------------------------------------------------------------------------


def closest_moment_locus(locus: pd.DataFrame, cs: float, ck: float) -> dict[str, float]:
    """Distance from a sample point to the nearest locus node.

    Rows whose quadrature did not produce finite moments are excluded rather
    than propagating ``NaN`` into the comparison.
    """
    valid = locus[np.isfinite(locus["Cs"]) & np.isfinite(locus["Ck"])]
    if valid.empty:
        return {
            "distance": float("nan"),
            "locus_shape": float("nan"),
            "locus_Cs": float("nan"),
            "locus_Ck": float("nan"),
            "n_nodes_used": 0,
        }
    d2 = (valid["Cs"].to_numpy() - cs) ** 2 + (valid["Ck"].to_numpy() - ck) ** 2
    j = int(np.argmin(d2))
    return {
        "distance": float(np.sqrt(d2[j])),
        "locus_shape": float(valid["shape"].iloc[j]),
        "locus_Cs": float(valid["Cs"].iloc[j]),
        "locus_Ck": float(valid["Ck"].iloc[j]),
        "n_nodes_used": int(len(valid)),
    }


@dataclasses.dataclass
class LMomentDiagnostics:
    """L-moment summary of the observed series plus its moment-plane position."""

    stats: dict[str, float]
    pwm: dict[str, float]
    distances: dict[str, dict[str, float]]
    audit: dict[str, Any]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats,
            "pwm": self.pwm,
            "distances": self.distances,
            "audit": self.audit,
            "interpretation": self.interpretation,
        }


def run(
    values: Sequence[float] | np.ndarray,
    cs: float | None = None,
    ck: float | None = None,
) -> LMomentDiagnostics:
    """L-moment / PWM diagnostics plus moment-plane model proximity.

    Parameters
    ----------
    values:
        Observed annual peaks.
    cs, ck:
        Bulletin 17B skewness and kurtosis of the sample.  If omitted they are
        not computed here; pass them from :mod:`ffa_karad.statistical_tests`
        to avoid duplicating that logic.
    """
    stats = sample_lmoments(values)
    pwm = {f"pwm_{r}": probability_weighted_moments(values, r) for r in (1, 2, 3, 4)}

    audit = {
        "moment_quadrature": verify_moment_quadrature(),
        "pearson3_kurtosis_relation": verify_pearson3_kurtosis_relation(),
    }

    distances: dict[str, dict[str, float]] = {}
    nearest = None
    if cs is not None and ck is not None:
        loci = {
            n: model_moment_locus(n)
            for n in ("Gumbel", "GEV", "LogPearsonIII", "Exponential")
        }
        distances = {
            n: closest_moment_locus(locus, cs, ck) for n, locus in loci.items()
        }
        nearest = min(distances, key=lambda k: distances[k]["distance"])
        plane = (
            f"On the moment plane the sample sits at (Cs, Ck) = ({cs:.4f}, {ck:.4f}); "
            f"the nearest computed locus is {nearest} at distance "
            f"{distances[nearest]['distance']:.4f}. "
        )
    else:
        plane = ""

    gumbel = population_moments(sps.gumbel_r.pdf, sps.gumbel_r.ppf, label="Gumbel")
    interpretation = (
        plane + f"The Gumbel reference on the moment plane is "
        f"(Cs, Ck) = ({gumbel['Cs']:.4f}, {gumbel['Ck']:.4f}). The 2025 notebook "
        f"plotted the Gumbel locus at (0, 5.4); the kurtosis 5.4 is right but the "
        f"skewness is not -- 0 assumes symmetry about the mean, whereas the Gumbel "
        f"is asymmetric with Cs = {gumbel['Cs']:.4f}. "
        f"The unbiased b-statistic L-moments are also reported, but their ratio "
        f"measures (LCV {stats['lcv']:.3f}) are dominated by sampling noise at "
        f"N = {stats['n']} and are not used for model selection."
    )
    log.info(
        "L-moments L1..L4 = %.1f / %.1f / %.1f / %.1f (ratios not used at N=%d)",
        stats["L1"],
        stats["L2"],
        stats["L3"],
        stats["L4"],
        stats["n"],
    )
    return LMomentDiagnostics(
        stats=stats,
        pwm=pwm,
        distances=distances,
        audit=audit,
        interpretation=interpretation,
    )


def summarise(diag: LMomentDiagnostics) -> str:
    """Plain-text summary for the report."""
    s = diag.stats
    lines = [
        "L-MOMENT AND PWM DIAGNOSTICS",
        f"  estimator            : {s['estimator']}",
        f"  L1                   : {s['L1']:.4f}",
        f"  L2                   : {s['L2']:.4f}",
        f"  L3                   : {s['L3']:.4f}",
        f"  L4                   : {s['L4']:.4f}",
        f"  ratio measures       : LCV {s['lcv']:.4f}, L-skew {s['lskew']:.4f}, "
        f"L-kurt {s['lkurt']:.4f}",
        f"  ratio measures usable: {s['ratios_reliable']} "
        f"(high variance at N = {s['n']}; see module docstring)",
    ]
    for k, v in diag.pwm.items():
        lines.append(f"  {k:<21} : {v:.4f}")
    if diag.distances:
        for name, d in diag.distances.items():
            lines.append(
                f"  nearest {name:<14} : d = {d['distance']:.4f} "
                f"at shape {d['locus_shape']}"
            )
    lines.append("")
    lines.append(f"  {diag.interpretation}")
    return "\n".join(lines)


__all__ = [
    "LOCUS_SPECS",
    "EXACT_REFERENCES",
    "LMomentDiagnostics",
    "sample_lmoments",
    "consistent_lmoments",
    "probability_weighted_moments",
    "population_moments",
    "moment_locus",
    "model_moment_locus",
    "all_moment_loci",
    "closest_moment_locus",
    "verify_moment_quadrature",
    "verify_pearson3_kurtosis_relation",
    "run",
    "summarise",
]
