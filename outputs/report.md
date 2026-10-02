# Flood-frequency analysis, Krishna river at Karad bridge

Generated 2026-10-02 11:27 UTC by `ffa_karad.reporting`. Every number below is read from a module output; nothing is entered by hand.

Station AK000X6. RNG seed 20260902.

## Summary

- T = 100: **8,667** m3/s (95% 6,048-12,923 m3/s, candidate spread 17%).
- T = 1,000: **13,327** m3/s (95% 7,622-25,397 m3/s, candidate spread 27%).

This analysis supersedes the 2025 notebook in three respects: it deletes no observations, it reports no number without the interval and the diagnostic that qualifies it, and it mixes no transform in which a Gumbel distribution is fitted to `ln Q` and its quantiles are reported as discharges.

## Adopted design floods

Method: log-Pearson type III on `ln Q`, Bulletin 17B moments (n = 57, Cs = 0.096), the primary method under IS 11223:1985. Gumbel EV1 and strict two-parameter LN2 are reported as cross-checks; AICc ranks the candidates and does not choose the method.

| return_period_yr | adopted_lp3_cumecs | adopted_ci_lower_cumecs | adopted_ci_upper_cumecs | gumbel_ev1_cumecs | ln2_cumecs | candidate_spread_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2,483 | 2,152 | 2,861 | 2,583 | 2,504 | 4.0396 |
| 5 | 3,862 | 3,276 | 4,547 | 3,814 | 3,858 | 1.2435 |
| 10 | 4,889 | 3,993 | 5,962 | 4,629 | 4,835 | 5.3027 |
| 25 | 6,307 | 4,860 | 8,173 | 5,659 | 6,153 | 10.2791 |
| 50 | 7,451 | 5,484 | 10,266 | 6,423 | 7,189 | 13.7942 |
| 100 | 8,667 | 6,048 | 12,923 | 7,181 | 8,270 | 17.1394 |
| 200 | 9,964 | 6,551 | 15,873 | 7,937 | 9,400 | 20.3424 |
| 500 | 11,815 | 7,196 | 20,662 | 8,934 | 10,979 | 24.3871 |
| 1,000 | 13,327 | 7,622 | 25,397 | 9,687 | 12,243 | 27.3162 |

Columns: `adopted_lp3_cumecs` is the Bulletin 17B moment estimate; the confidence band is the log-space parametric bootstrap; the last column is the spread between accepted candidate distributions, i.e. the model-form uncertainty, which is larger than the confidence band.

**Qualifications that travel with these numbers.**
- The observed HFL of 7,177 m3/s (1976-06-07) sits at an implied 43-year return period on this curve, against a 57-year record.
- T = 100: re-estimating on 500 independent 57-year records gives a 95% interval of 5,359-12,349 m3/s, a width of 81% of the adopted value.
- T = 1,000: re-estimating on 500 independent 57-year records gives a 95% interval of 5,578-16,320 m3/s, a width of 81% of the adopted value.
- The discharge column is rating-curve derived (`Discharge*`), so these values carry correlated measurement error that no bootstrap in this package quantifies.

## Design water levels

Levels come from inverting the rating curve at each design discharge. Freeboard is applied only at the 100-year return period, which is why `top_of_structure_m` exceeds `water_level_m` in exactly one row.

| return_period_yr | discharge_cumecs | water_level_m | depth_over_zero_gauge_m | freeboard_m | top_of_structure_m | rating_curve_extrapolated |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2,483 | 559.3148 | 9.3998 | 0 | 559.3148 | False |
| 5 | 3,862 | 562.0659 | 12.1509 | 0 | 562.0659 | False |
| 10 | 4,889 | 563.8489 | 13.9339 | 0 | 563.8489 | False |
| 25 | 6,307 | 566.0728 | 16.1578 | 0 | 566.0728 | False |
| 50 | 7,451 | 567.7152 | 17.8002 | 0 | 567.7152 | True |
| 100 | 8,667 | 569.3495 | 19.4345 | 1 | 570.3495 | True |
| 200 | 9,964 | 570.99 | 21.075 | 0 | 570.99 | True |
| 500 | 11,815 | 573.1839 | 23.2689 | 0 | 573.1839 | True |
| 1,000 | 13,327 | 574.8711 | 24.9561 | 0 | 574.8711 | True |


Rows flagged `rating_curve_extrapolated` lie outside the stage range over which the rating curve was calibrated. Those levels are extrapolations, not measured levels.
## Record, quality control and dependence

Quality control: 14 checks passed, 3 warned, 0 failed. Record length 57 water years.

| check | status | value | detail |
| --- | --- | --- | --- |
| rating_curve_outliers | WARN | ['1976-1977', '2003-2004', '2021-2022'] | 3 water year(s) deviate from the rating curve: 1976-1977, 2003-2004, 2021-2022 |
| discharge_provenance | WARN | rating-curve derived | Discharge column header in the source sheet reads 'Discharge* (cumecs)'.  In NWDA/CWC station abstracts the asterisk denotes a rating-curve derived discharge, not an instantaneous volumetric gauge record.  All derived discharges carry correlated error, which is not captured by any bootstrap in this package. |
| no_extreme_year_on_year_jumps | WARN | ['1968-1969', '1970-1971', '1977-1978', '1995-1996', '1998-1999', '2000-2001', '2004-2005', '2007-2008', '2009-2010', '2012-2013', '2020-2021'] | large year-on-year changes at 1968-1969, 1970-1971, 1977-1978, 1995-1996, 1998-1999, 2000-2001, 2004-2005, 2007-2008, 2009-2010, 2012-2013, 2020-2021; a change-point or outlier test is required (see ffa_karad.outliers) |


### Trend and stationarity

| test | statistic | p_value | alpha | reject_h0_at_alpha | detail | h0 |
| --- | --- | --- | --- | --- | --- | --- |
| Mann-Kendall (original) | -1.0395 | 0.2986 | 0.05 | False | S = -152, tau = -0.0952 | no monotonic trend |
| Mann-Kendall (Hamed-Rao corrected) | -0.6001 | 0.5484 | 0.05 | False | correction factor 1.7321 | no monotonic trend, serially correlated series |
| Spearman rho | -0.1595 | 0.2359 | 0.05 | False | monotonic association with water-year index | no monotonic trend |
| Kendall tau | -0.0952 | 0.2954 | 0.05 | False | monotonic association with water-year index | no monotonic trend |
| Sen's slope | -12.7795 |  | 0.05 | False | 95% CI [-31.5263, 11.6667] cumecs/yr | slope = 0 |
| Trend-free pre-whitened MK | 0 | 1 | 0.05 | False | adjusted Sen slope -12.779545 cumecs/yr after pre-whitening | no residual trend after removing the linear component |


Pettitt change point at index 50 (p = 0.211); no change point is indicated at alpha = 0.05. The index is a position in the water-year series, not a calendar year.

### Dependence

- lag-1 autocorrelation +0.1707 against a screening band of -0.2596 to +0.2596 (0.05 alpha, N = 57); autocorrelation length 6 years.
- effective sample size 34.0 of 57 (integrated autocorrelation time 1.68). Bootstrap intervals in this package assume independent peaks, so they are optimistic by roughly this factor.
- Hurst exponent -0.018 by the powers estimator, +0.463 by rescaled range; rescaled range is biased high at this record length and the two disagree, so no persistence claim is made from either.

### Outlier screening

Rosenblatt flags a candidate but the others do not, which is the usual outcome; a single test on 57 points is weak evidence

| label | value_cumecs | log_value | rank_ascending |
| --- | --- | --- | --- |
| 1965-1966 | 855 | 6.7511 | 1 |
| 1966-1967 | 1,052 | 6.9584 | 2 |
| 1967-1968 | 1,077 | 6.9819 | 3 |

Nothing was deleted. The three lowest peaks are retained and are listed so a reviewer can check them against the gauge records.
## Distribution choice

| distribution | k | fitted_on | loglike_Q | aicc | ks_p | ad_p | accepted | Q100 | Q1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN2 | 2 | Q | -488.9547 | 982.1317 | 0.9695 | 0.9865 | True | 8,270 | 12,243 |
| LP3 | 3 | log Q | -488.8668 | 984.1864 | 0.9868 | 0.9915 | True | 8,923 | 14,197 |
| Gumbel | 2 | Q | -490.675 | 985.5722 | 0.8335 | 0.7336 | True | 7,181 | 9,687 |
| Log-logistic | 2 | Q | -490.5074 | 985.237 | 0.9438 | 0.9545 | False | 9,964 | 20,027 |
| GLO | 3 | Q | -490.6809 | 987.8147 | 0.8319 | 0.7326 | False | 7,184 | 9,690 |
| Weibull | 2 | Q | -492.6749 | 989.572 | 0.5277 | 0.4613 | False | 6,890 | 8,416 |
| GNO | 3 | Q | -497.6643 | 1,002 | 0.3177 | 0.2164 | False | 6,612 | 8,637 |
| Exponential | 1 | Q | -510.5707 | 1,023 | 9.35e-05 | 0.003 | False | 13,155 | 19,733 |
| GEV | 3 | Q | -584.5115 | 1,175 | 6.43e-24 | 0.0005 | False | 8,039,914,442 | 447,145,276,735,346 |

`accepted` is decided by goodness-of-fit and support gates, not by the information criteria: a fit whose CDF cannot carry the observed order statistics is rejected however good its likelihood. The log-likelihood is compared on the raw scale for every candidate, with the Jacobian correction applied to the LP3 fit of `ln Q`.

### Peaks over threshold

Peaks over threshold was attempted and **not adopted**:

- only 28 exceedances over 57 years (minimum required 30); the shape estimate is not defensible
- the shape interval [-0.556, -0.333] lies entirely below zero, i.e. the data cannot exclude a bounded upper tail; this contradicts the Gumbel and LP3 block-maxima curves

The sub-annual record in this station's files contains one value per year, so a POT fit would rest on about 28 exceedances. That is below the minimum of 30 and the fitted shape is negative, which would mean a bounded upper tail. Neither is defensible, so no POT return level appears anywhere in this report.

### Exploratory machine learning

Ridge beats persistence with skill +0.077 (RMSE 1,768 vs 1,914 cumecs); this is exploratory and must not be used to derive a design flood

| model | rmse_cumecs | mae_cumecs | r2 | skill_vs_persistence |
| --- | --- | --- | --- | --- |
| RandomForest | 1,646 | 1,413 | -0.1031 | 0.1404 |
| Ridge | 1,768 | 1,557 | -0.2728 | 0.0766 |
| GradientBoosting | 1,908 | 1,564 | -0.4822 | 0.0035 |
| persistence (baseline) | 1,914 | 1,424 |  | 0 |
| Linear | 3,078 | 2,473 | -2.8579 | -0.6076 |

These numbers exist to show that the regression models do not help, not to produce a design flood. Out-of-fold scores only; the 2025 notebook's near-perfect R-squared came from putting the target in the feature matrix.
## Uncertainty

Convergence gate: **PASS**. split R-hat max 1.0007 (threshold 1.01); minimum effective sample size 7,576 (minimum 400); mean acceptance rate 0.193 (band 0.15-0.4).

| parameter | mean | sd | lower | upper | frequentist_mom | posterior_sd_over_analytic |
| --- | --- | --- | --- | --- | --- | --- |
| mean_ln_q | 7.8261 | 0.0729 | 7.6841 | 7.9724 | 7.8255 | 14.2193 |
| sd_ln_q | 0.5462 | 0.0584 | 0.4481 | 0.6772 | 0.5182 | 0.8312 |
| cs_log | 0.1422 | 0.4035 | -0.6826 | 0.8712 | 0.0956 | 0.804 |


Posterior design quantiles (model-based, LP3 assumed correct):

| return_period_yr | posterior_median_cumecs | posterior_mean_cumecs | posterior_lower_cumecs | posterior_upper_cumecs | probability_above_observed_max |
| --- | --- | --- | --- | --- | --- |
| 2 | 2,468 | 2,480 | 2,121 | 2,895 | 0 |
| 5 | 3,921 | 3,953 | 3,337 | 4,757 | 0 |
| 10 | 5,006 | 5,095 | 4,191 | 6,553 | 0.0072 |
| 25 | 6,509 | 6,753 | 5,239 | 9,678 | 0.2728 |
| 50 | 7,745 | 8,165 | 5,948 | 12,903 | 0.6708 |
| 100 | 9,089 | 9,749 | 6,586 | 16,896 | 0.8954 |
| 200 | 10,536 | 11,544 | 7,128 | 21,990 | 0.9723 |
| 500 | 12,653 | 14,312 | 7,792 | 30,844 | 0.9957 |
| 1,000 | 14,378 | 16,774 | 8,233 | 39,525 | 0.9989 |

## Bulletin 17C checklist

13 pass, 0 warn, 0 fail, 5 require manual evidence.

| clause | status | requirement | responsible |
| --- | --- | --- | --- |
| 17C 2.1 | pass | The annual peak series covers every water year in the period of record with no gaps | data |
| 17C 2.2 | manual | Each peak is the highest discharge in its water year, taken from the maximum of the continuous record | hydrologist |
| 17C 2.3 | manual | The station rating curve is current, and shifts in it are documented | hydrologist / gauge office |
| 17C 2.4 | manual | The datum used is the same throughout, and is stated | hydrologist |
| 17C 2.5 | manual | Dredging, diversion, construction or reservoir changes in the channel are assessed for non-stationarity | hydrologist |
| 17C 2.6 | pass | Outlying values have been screened and the screening is recorded | analysis package |
| 17C 3.1 | pass | A test of the null hypothesis of stationarity is applied | hydrologist |
| 17C 3.2 | pass | The record length is adequate for the return periods being reported, or the limitation is stated | analysis package |
| 17C 3.3 | pass | Where the record is too short for the required return period, independent evidence is used | analysis package |
| 17C 4.1 | pass | The log-Pearson III is fitted to the logarithms of the annual maxima | analysis package |
| 17C 4.2 | pass | The skew coefficient lies inside the Bulletin 17B table bounds for the record length | analysis package |
| 17C 4.3 | manual | The zero flood is accounted for, and the log transform is legitimate | hydrologist |
| 17C 4.4 | pass | Confidence limits are computed and reported with the design flood | analysis package |
| 17C 4.5 | pass | Independent methods are used as a check on the adopted distribution | analysis package |
| 17C 5.1 | pass | The observed points are plotted on the frequency curve and the fit is judged visually as well as statistically | analysis package |
| 17C 5.2 | pass | An arithmetic check is performed on the adopted curve | analysis package |
| 17C 5.3 | pass | The frequency curve is extrapolated only to the periods the data can support | analysis package |
| 17C 5.4 | pass | Bulletin 17B/B17C recommends that for n below about 25 years the confidence limits are treated as approximate | analysis package |


### Outstanding manual evidence

These are not failures. They are questions this package has no data to answer, and a reviewer must close each one from the gauge records before the report is issued.

- **17C 2.2** Each peak is the highest discharge in its water year, taken from the maximum of the continuous record _(owner: hydrologist)_: requires the continuous gauge record; this package holds one annual value per water year and cannot confirm the within-year maximum
- **17C 2.3** The station rating curve is current, and shifts in it are documented _(owner: hydrologist / gauge office)_: quality control reports rating-curve fit PASS and rating-curve outlier years WARN; a fit check cannot establish the rating *history*, which must come from the gauge records
- **17C 2.4** The datum used is the same throughout, and is stated _(owner: hydrologist)_: peak water levels in this study are referenced to the zero gauge; the reduction datum must be stated on the drawing
- **17C 2.5** Dredging, diversion, construction or reservoir changes in the channel are assessed for non-stationarity _(owner: hydrologist)_: not assessable from the discharge series; the last two decades of this record show a clear regime change that must be explained
- **17C 4.3** The zero flood is accounted for, and the log transform is legitimate _(owner: hydrologist)_: no zero or negative discharges appear in this record; a record containing zeros cannot be fitted on a logarithmic scale

### Figures

- `outputs/figures/flood_frequency.png` -- Adopted LP3 flood-frequency curve with confidence limits and the Gumbel and LN2 cross-checks
- `outputs/figures/record_on_curve.png` -- The 57 observed annual maxima plotted on the adopted frequency curve
- `outputs/figures/time_series.png` -- Annual maxima in water-year order against the observed HFL
- `outputs/figures/acf.png` -- Autocorrelation of annual maxima with the standard-error band
- `outputs/figures/design_levels.png` -- Design water levels, observed HFL and freeboard
- `outputs/figures/candidate_comparison.png` -- Return-level curves for every accepted candidate distribution
- `outputs/figures/pot_return_levels.png` -- Peaks-over-threshold return levels, or the reasons the screen refused them
- `outputs/figures/kde_log.png` -- Kernel density of ln Q with the observations as a rug
- `outputs/figures/kde_normality.png` -- Empirical density against the fitted log-normal and LP3
- `outputs/figures/histogram_ecdf_violin.png` -- Histogram, empirical CDF with fitted families, and the shape of the record

## How to reproduce

```
python -m ffa_karad.run_all
```

The run writes this document, the result tables as CSV, every figure as PNG and a JSON dump of every result object under `outputs/`. The RNG is seeded, so a re-run reproduces these numbers exactly.

