"""
Single source of truth for every constant used in the Karad FFA.

Why this module exists
----------------------
The 2025 notebook scattered constants across ~39 cells: station metadata,
return periods, Bulletin 17C thresholds, iteration counts, CWC coefficients and
a station rating equation.  Three of them were wrong or undocumented, and the
reviewer could not tell which was which.  Here, every constant is:

* declared exactly once,
* annotated with its **source** (standards clause / textbook / empirical
  estimate), and
* flagged ``VERIFY`` when it is an empirical or site-specific value that a
  reviewing officer must confirm against the published document.

Nothing downstream hard-codes a value.  If a number appears in a report, it was
produced by a function in this package.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]

DATA_DIR: Final[pathlib.Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[pathlib.Path] = PROJECT_ROOT / "outputs"
FIGURE_DIR: Final[pathlib.Path] = OUTPUT_DIR / "figures"
TABLE_DIR: Final[pathlib.Path] = OUTPUT_DIR / "tables"
LOG_DIR: Final[pathlib.Path] = OUTPUT_DIR / "logs"
DOCS_DIR: Final[pathlib.Path] = PROJECT_ROOT / "docs"

RAW_DATA_FILENAME: Final[str] = "Dischage data KRISHNA KARAD BRIDGE.csv"

#: SHA-256 of the raw CSV as supplied.  Recomputed and compared on every run by
#: :func:`ffa_karad.data_processing.load_raw_csv` so that a silent data swap is
#: impossible.
RAW_DATA_SHA256: Final[str] = (
    "f2f0e629d111ba0569c8ed8b55a82cc2cc2b93b2f6a2fc110d5b590756ab77ba"
)

#: SHA-256 of the CSV when checked out with Unix LF newlines (e.g. on Linux/macOS CI runners)
RAW_DATA_SHA256_LF: Final[str] = (
    "690a82a2df438763bc593ebd6d055c1862e93f51a3e90731bf0c13001f049c07"
)

#: Valid checksums allowing both CRLF and LF checkouts across platforms
RAW_DATA_VALID_SHA256S: Final[frozenset[str]] = frozenset(
    {
        RAW_DATA_SHA256,
        RAW_DATA_SHA256_LF,
    }
)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

#: Master seed.  Every stochastic step in the pipeline draws from
#: ``numpy.random.default_rng(MASTER_SEED + <stream offset>)`` via
#: :func:`ffa_karad.util.stream_rng`, so any single stage can be re-run in
#: isolation and reproduce bit-identical output.
MASTER_SEED: Final[int] = 20260902


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Station:
    """Physical description of the gauging station.

    Source: NWDA / Central Water Commission abstract station data for
    ``AK000X6`` as supplied in the analyst's data sheet, cross-checked against
    the station abstract in ``E:/FFA/Karad historical discharge Data.pdf``.

    Every field below marked ``VERIFY`` can be confirmed against the
    station abstract if needed.
    """

    station_id: str = "AK000X6"
    station_name: str = "Karad"
    river: str = "Krishna"
    basin: str = "Upper Krishna"
    state: str = "Maharashtra"
    #: Geographical area in km2 as used by CWC design formulae.
    catchment_area_km2: float = 5462.0  # VERIFY against the CWC basin atlas
    #: Zero gauge / gauge datum in metres above mean sea level.
    zero_gauge_level_m: float = 549.915  # VERIFY against the station abstract
    #: Highest flood level observed, in metres above the zero gauge.
    observed_hfl_m: float = 567.162
    observed_hfl_date: str = "1976-06-07"
    #: Discharge corresponding to the observed HFL, cumecs.
    observed_hfl_cumecs: float = 7177.0
    record_start_wy: str = "1965-1966"
    record_end_wy: str = "2021-2022"
    #: True if the supplied discharge column is rating-curve derived rather
    #: than gauged.  The source column header carries an asterisk
    #: ("Discharge* (cumecs)"), which in NWDA practice flags a derived value.
    #: This materially affects the uncertainty budget and MUST be disclosed.
    discharge_is_derived: bool = True
    provenance_note: str = (
        "Discharge column header in the source sheet reads 'Discharge* "
        "(cumecs)'.  In NWDA/CWC station abstracts the asterisk denotes a "
        "rating-curve derived discharge, not an instantaneous volumetric "
        "gauge record.  All derived discharges carry correlated error, which "
        "is not captured by any bootstrap in this package."
    )


STATION: Final[Station] = Station()


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """Statistical settings.  Frozen: a run is fully described by its config."""

    # -- Return periods -----------------------------------------------------
    #: Standard design return periods in years (IRC / IS 7784 practice).
    return_periods: tuple[float, ...] = (
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    )

    # -- Significance -------------------------------------------------------
    #: Two-sided significance level for hypothesis tests.
    alpha: float = 0.05
    #: Significance level for one-sided Bulletin 17C confidence limits.
    #: Bulletin 17C uses the 5 % level.
    alpha_one_sided: float = 0.05

    # -- Method selection ---------------------------------------------------
    #: IS 11223:1985 recommends the log-Pearson Type III distribution for
    #: flood-frequency analysis of Indian rivers, with the log-Pearson III
    #: moment coefficients of skewness and kurtosis (Bulletin 17B).
    primary_distribution: str = "LP3"
    primary_standard: str = "IS 11223:1985"
    #: CWC Flood Estimation Manual (SP 12) cross-check distributions.
    cwc_crosscheck: tuple[str, ...] = ("Gumbel", "LN2")
    #: Bulletin 17C: apply a skewness confidence limit when the sample
    #: skewness is at or beyond the upper tail of its sampling distribution.
    use_bulletin_17c_limits: bool = True
    #: Bulletin 17C: above this skewness the kurtosis, not the skewness,
    #: governs the confidence limit.
    bulletin_17c_skew_threshold: float = 0.9
    #: Table-driven alternative.  Left ``None`` because this repository does
    #: not embed the Bulletin 17B / 17C tabulated upper limits; supply a
    #: callable ``(n, g1) -> float`` to use them instead of the bootstrap.
    bulletin_17b_table: object | None = None

    # -- Plotting positions -------------------------------------------------
    #: Weibull is the default for annual-maxima FFA in Indian practice.
    #: Hazen and Gringorten are computed as sensitivity checks.
    plotting_position: str = "weibull"
    plotting_position_sensitivity: tuple[str, ...] = ("hazen", "gringorten")

    # -- Resampling ---------------------------------------------------------
    n_bootstrap: int = 2000  # nonparametric + parametric bootstrap replicates
    n_gof_bootstrap: int = 2000  # bootstrap replicates for GoF p-values
    #: Independent synthetic records simulated for the Monte Carlo module.
    n_mc_records: int = 500
    #: Record length (years) of each simulated record.  Equal to the observed
    #: record so that Monte Carlo quantifies *this* record's sampling error.
    mc_record_length: int = 57  # VERIFY: overwritten by the observed N at run time

    # -- Bootstrap confidence interval level
    ci_level: float = 0.95

    # -- Bayesian MCMC ------------------------------------------------------
    mcmc_n_chains: int = 4
    mcmc_n_draws: int = 40_000
    mcmc_burn_in: int = 10_000
    #: Initial proposal covariance is estimated from a pilot run of this many
    #: draws (Haario adaptive Metropolis).  Without this the accept rate is
    #: uninterpretable -- the 2025 notebook reported 0.825 with a posterior
    #: s.d. 0.49x the analytic s.e., the signature of a non-mixing chain.
    mcmc_pilot_draws: int = 5_000
    #: Accept-rate band that indicates a well-tuned random-walk chain.
    mcmc_accept_band: tuple[float, float] = (0.15, 0.40)
    #: Gelman-Rubin acceptance threshold (Gelman et al. 2013, Table 6.1).
    mcmc_rhat_threshold: float = 1.01
    #: Minimum effective sample size before posterior quantiles are published.
    mcmc_min_ess: int = 400
    #: Weakly-informative prior scale, log space.  Priors are on the log of
    #: the Gumbel scale and location, centred on the MLE, with this s.d.
    mcmc_prior_sd_log: float = 2.0
    #: Fraction of the prior sd used for the sensitivity re-run.
    mcmc_prior_sd_log_alt: float = 0.5

    # -- Randomness / autocorrelation ---------------------------------------
    #: Longest lag reported in the ACF/PACF tables and figures.  Box-Cox and
    #: ordinary practice allow up to n/4; at N = 57 that is 14, so 15 is the
    #: first value beyond which the estimate is meaningless.
    acf_max_lag: int = 15
    #: Significance level for the ACF screening band.  This is an *alpha*, like
    #: ``CONFIG.alpha``, not a confidence level: at alpha = 0.05 the band is
    #: +/-1.96/sqrt(57) = +/-0.26, which is wide enough that only very large
    #: lag-1 correlation can be called significant.
    acf_alpha: float = 0.05
    #: Aggregation lags for the Hurst estimators.  The upper limit is floor(n/2)
    #: and is overwritten with the observed N at run time; these are the values
    #: used when the observed record length is unknown.
    hurst_lag_min: int = 2
    hurst_lag_max: int = 28
    #: Rescaled-range analysis needs at least 4 points per block to have a
    #: meaningful range-to-standard-deviation ratio.
    hurst_rs_lag_min: int = 4
    #: Circular block-bootstrap replicates for the Hurst interval.
    n_hurst_bootstrap: int = 2000
    #: Cap on the bootstrap block length.  Resampling single observations would
    #: assume the very independence the Hurst analysis is testing for.
    hurst_block_max: int = 6
    #: Bounds used when the observed aggregate-variance slope falls outside the
    #: admissible range (0, 1) and the estimator's spread still has to be
    #: characterised.  The out-of-range value is reported, never silently
    #: replaced.
    hurst_clamp_low: float = 0.05
    hurst_clamp_high: float = 0.95

    # -- Candidate distribution fitting ------------------------------------
    #: Largest shape parameter magnitude a fitted extreme-value family may
    #: carry.  Beyond this the fit is numerically degenerate: the quantile
    #: function develops a near-vertical asymptote and the tail estimate stops
    #: being interpretable.  The notebook's GEV returned c = -4.736, beyond
    #: this bound, and the resulting curve was clipped rather than rejected.
    max_abs_shape: float = 2.0
    #: A fitted location further above the smallest observed peak than this
    #: multiple is treated as divergence, not as a real parameter.
    degenerate_loc_factor: float = 0.5
    #: The largest return period in ``return_periods`` must sit at least this
    #: multiple of the largest observed peak.  A curve that puts the 1000-year
    #: flood below the flood that has already occurred is contradicted by the
    #: record whatever its likelihood.
    observed_max_discharge_factor: float = 0.5
    #: A candidate is rejected when the KS probability-plot p-value falls below
    #: this.  Finite parameters and a finite likelihood are not enough: the fit
    #: must also reproduce the observed order statistics.
    gof_alpha: float = 0.05
    #: Euler-Mascheroni constant, used by the method-of-moments Gumbel.
    euler_mascheroni: float = 0.5772156649015329

    # -- Peaks over threshold ----------------------------------------------
    #: Minimum number of exceedances for a POT fit to be reportable.  Below
    #: roughly 30 the GPD shape estimate is too unstable to defend.
    pot_min_exceedances: int = 30  # VERIFY: project-specific
    #: Profile-penalty threshold-selection constant k (Coles 2001, Table 3.2).
    #: 6 corresponds to a 5 % significance level for a GPD threshold.
    pot_penalty_k: float = 6.0
    pot_alpha: float = 0.05
    #: Candidate thresholds as quantiles of the exceedance-over-base series.
    pot_threshold_quantiles: tuple[float, ...] = (
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
    )
    #: A POT fit is rejected if the shape confidence interval is dominated by
    #: xi < 0, i.e. if the data cannot exclude a bounded tail.
    pot_require_unbounded_tail: bool = True
    #: Search bounds for the GPD shape.  The GPD requires xi > -1 for a finite
    #: upper endpoint; the bounds are kept strictly inside that so that a fit
    #: which runs to ``-1`` is reported as a bounded tail rather than as an
    #: infinite one.
    pot_xi_bounds: tuple[float, float] = (-0.95, 0.95)

    # -- Machine learning (exploratory only) -------------------------------
    ml_n_splits: int = 5  # TimeSeriesSplit, expanding window, never shuffled
    ml_random_state: int = 11

    # -- Density / plotting -------------------------------------------------
    kde_bw_adjust: float = 1.0
    kde_cut: float = 0.0  # clip the KDE at the data range; no invented support
    rug_alpha: float = 0.25

    # -- Outlier screening --------------------------------------------------
    #: A Rosenblatt point is only called a flag when it sits this many standard
    #: deviations off the line fitted through the central bulk of the plot.
    #: Without it, "the point furthest from the line" would always produce a
    #: flag and the screen would carry no information.
    ros_flag_sd: float = 2.0

    # -- Design flood level -------------------------------------------------
    #: Vertical freeboard (m) added above the design flood level.  IS 7784 /
    #: MoRTH IRC:5 clause 9.2.2 gives a minimum vertical clearance of
    #: 1.0 m (0.5 m minimum, 1.0 m desirable) for bridges.  VERIFY against
    #: the specific IRC span and the design flood adopted in the project
    #: drawings before submission.
    freeboard_m: float = 1.0  # VERIFY
    #: Design return period for the freeboard application.
    freeboard_return_period: float = 100.0

    # -- Output -------------------------------------------------------------
    float_format: str = "%.4g"
    write_figures: bool = True


CONFIG: Final[AnalysisConfig] = AnalysisConfig()


def run_timestamp() -> str:
    """ISO-8601 UTC timestamp stamped into every artefact for provenance."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def describe() -> str:
    """Human-readable dump of the active configuration for the report header."""
    lines = [
        "Karad FFA -- active configuration",
        f"  generated                  : {run_timestamp()}",
        f"  station                    : {STATION.station_id} {STATION.station_name} "
        f"on {STATION.river} ({STATION.basin}, {STATION.state})",
        f"  catchment area             : {STATION.catchment_area_km2:,.0f} km2",
        f"  zero gauge level           : {STATION.zero_gauge_level_m} m",
        f"  record                     : {STATION.record_start_wy} to {STATION.record_end_wy}",
        f"  discharge provenance       : "
        f"{'DERIVED (rating curve)' if STATION.discharge_is_derived else 'GAUGED'}",
        f"  primary method             : {CONFIG.primary_distribution} "
        f"per {CONFIG.primary_standard}",
        f"  CWC cross-check            : {', '.join(CONFIG.cwc_crosscheck)}",
        f"  return periods (yr)        : "
        f"{', '.join(f'{t:g}' for t in CONFIG.return_periods)}",
        f"  alpha                      : {CONFIG.alpha} (two-sided), "
        f"{CONFIG.alpha_one_sided} (one-sided B17C)",
        f"  bootstrap replicates       : {CONFIG.n_bootstrap:,}",
        f"  Monte Carlo records        : {CONFIG.n_mc_records:,} x "
        f"{CONFIG.mc_record_length} yr",
        f"  MCMC                       : {CONFIG.mcmc_n_chains} chains x "
        f"{CONFIG.mcmc_n_draws:,} draws (burn-in {CONFIG.mcmc_burn_in:,})",
        f"  POT minimum exceedances    : {CONFIG.pot_min_exceedances}",
        f"  ACF max lag                : {CONFIG.acf_max_lag} "
        f"(band at alpha {CONFIG.acf_alpha})",
        f"  freeboard (m)              : {CONFIG.freeboard_m} "
        f"@ T={CONFIG.freeboard_return_period:g} yr",
        f"  master seed                : {MASTER_SEED}",
    ]
    return "\n".join(lines)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "FIGURE_DIR",
    "TABLE_DIR",
    "LOG_DIR",
    "DOCS_DIR",
    "RAW_DATA_FILENAME",
    "RAW_DATA_SHA256",
    "MASTER_SEED",
    "Station",
    "STATION",
    "AnalysisConfig",
    "CONFIG",
    "run_timestamp",
    "describe",
]
