# Notebook → repository map

Source: `E:\FFA\Result\FFA_Karad_Advanced_CWC_V2.ipynb` — 67 cells (39 code, 28 markdown),
nbformat 4.5, kernel `Python 3`.

Every notebook section is accounted for below. "Disposition" states what the
refactored code does with it: **carried** (reimplemented, corrected), **split**
(became more than one module), **replaced** (superseded by a better method), or
**dropped** (removed because it was wrong, uninformative, or unsafe).

---

| Nb cell(s) | Notebook section | New module | Disposition | What changed |
|---|---|---|---|---|
| 0 | Title + station banner | `config.STATION` | carried | Station metadata moved into one frozen dataclass; the `Discharge*` asterisk is now an explicit QC finding. |
| 1–2 | S1 Installation & Imports | `packages.py`, `requirements.txt` | carried | Optional, opt-in install. Writes `outputs/environment_manifest.json`. |
| 3–4 | S2 Data Loading & QC | `data_processing.py`, `quality_control.py` | split | CSV-first with embedded fallback; verbatim water-year labels; **added** stage–discharge rating-curve test, seasonality check, provenance warning. |
| 5–6 | S3 Descriptive stats & L-moments | `data_processing.py`, `lmoments.py` | split | **Dropped** the "Mode (3-yr)" statistic as uninformative (see `corrections_from_v2.md` §1). L-moments kept but ratio measures flagged unreliable at N=57. |
| 7–8 | S4a Normality | `statistical_tests.normality_battery` | carried | Anderson–Darling p-value replaced by parametric bootstrap (scipy's table assumes known parameters). |
| 7–9 | S4b Independence (runs, Ljung–Box, Durbin–Watson) | `statistical_tests.randomness_battery`, `autocorrelation.py` | split | **Fixed the crashing cell** `c4bba4e6` (`float | function` → `TypeError`). DW now reports an asymptotic p-value and is labelled approximate. |
| 7–10 | S4c Stationarity (ADF/KPSS) | `statistical_tests.stationarity_battery` | carried | Both regressions reported, p-values from `statsmodels`. |
| 7–11 | S4d Trend | `statistical_tests.trend_battery` | carried | **Added** Hamed–Rao modified MK and trend-free pre-whitening. Sen slope relabelled `cumecs/yr` (was `cm/yr`). |
| 12–13 | S5a Skewness & kurtosis | `statistical_tests.moment_coefficients`, `secondary_skewness_estimators` | carried | **Fixed Ck**: it is `m₄/m₂² = 3.4683`, not the 0.6253 printed by the notebook. Excess kurtosis 0.4683 is correct and is now labelled as such. **Fixed** the fabricated Pearson III locus `0.5·Cs²` → verified `3 + 1.5·Cs²`. |
| 12–14 | S5b Mode estimator | — | **dropped** | `rolling(3).mean().mode()` is tie-broken arbitrarily by `idxmax`; removed. |
| 12–14 | S5c "Kelvin–Boltzmann" analogy | — | **dropped** | A physics analogy for a statistical quantity is not a method. |
| 12–15 | S5d Skewness sampling distribution | `skewness_limits.py` | **replaced** | Replaced by the Bulletin 17C decision procedure with bootstrap-derived limits and an acceptance gate. |
| 16–17 | S6 ACF / PACF / Hurst | `autocorrelation.py` | carried | Added residual ACF against fitted models; Hurst CI from the standard `1/H` interval; documented as exploratory only. |
| 18–20 | S7 Extended distribution fitting (12 candidates) | `distribution_fitting.py` | **replaced** | **Removed GPD** (POT method, not a block-maxima family). **Removed free-location LN2** (CWC/IS 11223 require 2 parameters). **Fixed the LP3 Jacobian** so AIC/BIC/AICc/HQIC are comparable. Added degenerate-fit guards and bootstrap GoF p-values. |
| 21–22 | S8a Quantile estimation | `estimation_design_flood.py` | carried | **Removed the hard-coded report literals** that contradicted the computed Gumbel quantiles at 4 return periods. |
| 21–23 | S8b Plotting positions & multipliers | `data_processing.plotting_position`, `estimation_design_flood` | carried | Weibull default with Hazen and Gringorten as declared sensitivity checks. |
| 24–25 | S9 POT / GPD | `peaks_over_threshold.py` | **replaced** | **Added** Jenkinson–Collison threshold, profile-penalty likelihood, threshold-stability CI, shape CI, and an acceptance gate that blocks the result if the tail cannot be shown unbounded or `n_exceed < 30`. |
| 26–28 | S10 Machine learning | `machine_learning.py` | **replaced** | **Removed target leakage**: `rank_norm = Q.rank()/N` is monotone in the target; rolling features used `rolling(5)` without `.shift(1)` so they contained the current year. Added `TimeSeriesSplit`, scaling inside the pipeline, a persistence baseline, and an explicit statement that ML cannot produce a design flood. |
| 29–30 | S11a Monte Carlo | `uncertainty_monte_carlo_bayesian_mcmc.monte_carlo_record_uncertainty` | carried | Rebuilt as *independent synthetic N-year records* so it measures record-length uncertainty, and extended with a return-period exceedance probability. |
| 29–31 | S11b Bayesian MCMC | `uncertainty_monte_carlo_bayesian_mcmc.bayesian_mcmc` | **replaced** | The notebook's accept rate of 0.825 and posterior s.d. 0.49× the analytic s.e. indicated a non-mixing chain. Replaced with adaptive Metropolis (pilot covariance), 4 chains, split-R̂, ESS, trace diagnostics, a prior-sensitivity check, and a **hard gate**: results are refused unless R̂ ≤ 1.01, ESS ≥ 400 and accept rate in band. |
| 32–33 | S12 Colour system & setup | `visualization.py` (`PALETTE`) | carried | Plotly made optional-import-safe so the pipeline runs headless. |
| 34–35 | P1 Interactive time series | `visualization.plot_time_series` | carried | Fixed the axis bug where the string `"Log"` was concatenated onto the numeric L-moment axis. |
| 36–37 | P2 Distribution explorer (KDE/hist/ECDF/violin) | `seaborn_kde.py`, `histogram_ecdf_violin.py` | split | **Fixed** the `floc` inconsistency (panel used `floc=0`, table used `floc=True`). |
| 38–39 | P3 Interactive FFC, all distributions | `visualization.plot_frequency_curves` | carried | Curves drawn from the same fitted objects as the tables; diverged fits rejected upstream rather than clipped. |
| 40–43 | P4 Pairplot / jointplot / heatmap | `visualization.plot_correlation_panels` | carried | **Fixed** cell `8c701863` (unterminated string literal → `SyntaxError`); removed the meaningless upper-triangle mask; removed duplicated columns that produced a 2× correlation block. |
| 44–45 | P5 Monte Carlo fan | `visualization.plot_monte_carlo_fan` | carried | `add_vrect` now scoped to the intended subplot column instead of spanning all panels. |
| 46–47 | P6 MCMC posteriors | `visualization.plot_mcmc_diagnostics` | carried | Plots the actual diagnostic traces and gate verdicts, not only posteriors. |
| 48–49 | P7 Q–Q probability plots | `visualization.plot_qq_grid` | carried | Same fits as the tables. |
| 50–51 | P8 Skewness & kurtosis | `visualization.plot_moment_plane` | carried | **Removed** the fabricated GEV τ-locus; all loci computed by quadrature and self-audited against exact values. |
| 52–53 | P9 ACF/PACF/Hurst/lag | `visualization.plot_autocorrelation` | carried | **Fixed** ridgeline y-offsets so ridge polygons no longer sit on the ±1.96 bounds. |
| 54–55 | P10 Distribution ranking | `visualization.plot_model_ranking` | carried | Ranks on corrected, comparable ICs. |
| 56–57 | P11 POT stability & return levels | `visualization.plot_pot` | carried | Gate verdict shown on the figure. |
| 58–59 | P12 ML performance panel | `visualization.plot_ml_performance` | carried | Shows out-of-fold metrics against the persistence baseline, not in-sample fit. |
| 60–61 | P13 Publication dashboard | `visualization.plot_summary_dashboard` | carried | Every panel bound to a pipeline result object. |
| 62–63 | P14 Decade trends / probability paper / anomalies | `visualization.plot_decade_and_probability_paper` | carried | Probability paper uses the Weibull plotting position consistently. |
| 64–65 | S13a Final design flood table | `estimation_design_flood.py` | carried | **Added** HFL check (adopted Q must exceed the observed HFL) and the design flood *level* via the rating curve plus freeboard. |
| 64–66 | S13b Engineering recommendations (hard-coded text) | `reporting.py` | **replaced** | **Fixed** cell `46d29ac9` (`UnicodeEncodeError` on a Windows console). All recommendation text is now generated from result objects; no numeric literals. |

---

## Sections intentionally not carried forward

| Notebook item | Reason |
|---|---|
| "Mode (3-yr)" = 1175.33 cumecs | `rolling(3).mean()` is unimodal for any right-skewed record; the value is an `idxmax` tie-break artefact. Statistically vacuous. |
| Kelvin–Boltzmann skewness analogy | Analogy, not a method. |
| Gumbel locus at (Cs, Ck) = (0, 5.4) | The Gumbel is asymmetric: Cs = 1.13955 (computed, not asserted). |
| Pearson III locus `Ck = 3 + 0.5·Cs²` | Correct relation is `3 + 1.5·Cs²`; verified numerically to 1.2e-5. The notebook's form is off by up to 9.0 in Ck. |
| GEV τ₄ locus `0.116 + 0.206τ₃ − 0.013τ₃²` | Fabricated; no such closed form exists. Replaced by computed GEV loci. |
| GPD as a block-maxima candidate | GPD is the POT tail model, not a distribution for annual maxima. Including it invites a bounded upper tail. |
| LN2 with free location | CWC/IS 11223 define LN2 with two parameters. A free location is a 3-parameter LN3. |
| Hard-coded design-flood literals in the report | Contradicted the computed values at T = 25, 100, 200 and 1000 yr. |
| `rank_norm` as an ML feature | Monotone transform of the target; guarantees R² ≈ 1 without any skill. |
| Exponential accepted as a fitted candidate | KS probability-plot p-value 9.4e-05: the fitted CDF does not reproduce the observed order statistics. Now gated out by `distribution_fitting.gof_alpha`. |
| Sorted `DataBundle.q` fed to the time-ordered analyses | `q` is the ascending order-statistic array. Passing it where a chronological series is required silently produced an RMSE of 178 from a model that had simply memorised the rank order. `util.require_time_ordered` now rejects non-decreasing input, ties included. |
| ML on sorted `q`, and `q` in the feature matrix | Two separate leakage paths; both closed and both covered by tests. |

---

## New material with no notebook counterpart

| Module | Addition |
|---|---|
| `quality_control.py` | Stage–discharge rating-curve consistency test; derived-discharge provenance warning. |
| `skewness_limits.py` | Bulletin 17C decision procedure; bootstrap confidence limits; kurtosis-governed regime. |
| `peaks_over_threshold.py` | Jenkinson–Collison threshold; profile-penalty likelihood; shape CI; acceptance gate. |
| `uncertainty_monte_carlo_bayesian_mcmc.py` | Adaptive Metropolis with a pilot covariance; split-R̂; ESS; prior sensitivity; convergence gate. |
| `estimation_design_flood.py` | HFL check; design flood level by rating-curve inversion; freeboard. |
| `outliers.py` | Grubbs, directional Dixon Q, Rosenblatt normal-score screening, with no deletion path. |
| `cwc_manual_check.py` | Clause-by-clause Bulletin 17C checklist with `pass` / `fail` / `warn` / `manual` / `not applicable` statuses. Clauses the data cannot settle are `manual`, never `pass`. |
| `reporting.py` | Literal-free report generation. Each table carries its own qualification; a refused analysis is printed as refused; numbers are rendered without scientific notation. Writes `report.md`, CSV tables with provenance sidecars, and `results.json`. |
| `run_all.py` | Pipeline driver: 18 stages in dependency order, each recorded with its runtime and traceback. A failing stage does not abort the run. |
| `tests/` | Unit tests for every formula that could silently be wrong. 130 tests. |
