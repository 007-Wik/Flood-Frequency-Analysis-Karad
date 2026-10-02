# 🌊 Flood Frequency Analysis: Krishna River at Karad (AK000X6)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10 | 3.11 | 3.12" />
  <img src="https://img.shields.io/badge/CI-passing-2EA44F?style=for-the-badge&logo=github-actions&logoColor=white" alt="CI: Passing" />
  <img src="https://img.shields.io/badge/tests-130%20passed-success?style=for-the-badge&logo=pytest&logoColor=white" alt="130 Tests Passing" />
  <img src="https://img.shields.io/badge/coverage-83%25-brightgreen?style=for-the-badge&logo=codecov&logoColor=white" alt="83% Coverage" />
  <img src="https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge&logo=python&logoColor=white" alt="Code Style: Black" />
  <img src="https://img.shields.io/badge/imports-isort-1674b1?style=for-the-badge&logo=python&logoColor=white" alt="Imports: isort" />
  <img src="https://img.shields.io/badge/linter-flake8-informational?style=for-the-badge" alt="Linter: Flake8" />
  <img src="https://img.shields.io/badge/docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/Standard-IS%2011223%3A1985-007ACC?style=for-the-badge" alt="IS 11223:1985" />
  <img src="https://img.shields.io/badge/Guideline-CWC%20%7C%20USGS%2017C-blueviolet?style=for-the-badge" alt="CWC & USGS 17C" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License" />
</p>

A modular, production-grade Python package for Flood Frequency Analysis (FFA) implementing **IS 11223:1985** and **Central Water Commission (CWC)** methodologies, with cross-checks against **USGS Bulletin 17C**.

---

## 📍 1. Station & Record Overview

- 🏷️ **Station Code**: AK000X6
- 🗺️ **Location**: Krishna River at Karad Bridge, Maharashtra, India
- 📐 **Catchment Area**: 5,462.0 km²
- 📏 **Zero of Gauge (ZG)**: 549.915 m above MSL
- 🌊 **Observed Highest Flood Level (HFL)**: 567.162 m (Discharge: 7,177 m³/s on 1976-06-07)
- 📅 **Record Length**: 57 Water Years (1965–1966 to 2021–2022)
- ✅ **Record Completeness**: 100% (zero missing annual instantaneous peaks)
- 🎯 **Primary Design Method**: Log-Pearson Type III (LP3) fitted to $\ln(Q)$ per IS 11223:1985
- 🔍 **CWC Cross-Checks**: Gumbel (EV1, Method of Moments & Maximum Likelihood) and strict 2-Parameter Log-Normal (LN2, `floc=0`)

---

## 📊 2. Adopted Design Flood Results

Reproduce the complete pipeline with one command:

```bash
python -m ffa_karad.run_all
```

### 📈 Adopted Design Floods (LP3 on $\ln Q$, Bulletin 17B moments, `n = 57`, $C_{s,\log} = +0.0956$):

| Return Period $T$ (yr) | Adopted $Q$ (m³/s) | 95% Confidence Interval (m³/s) | Gumbel EV1 (m³/s) | LN2 (m³/s) | Candidate Spread |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | **7,451** | 5,484 – 10,266 | 6,423 | 7,189 | 14% |
| **100** | **8,667** | 6,048 – 12,923 | 7,181 | 8,270 | 17% |
| **500** | **11,815** | 7,196 – 20,662 | 8,934 | 10,979 | 24% |
| **1,000** | **13,327** | 7,622 – 25,397 | 9,687 | 12,243 | 27% |

> **⚠️ Engineering Insight**: Every design flood at or beyond the record length exceeds the observed HFL discharge of 7,177 m³/s, which the fitted curve places at a 43-year return period. The full report, with all engineering qualifications, is written to [`outputs/report.md`](file:///outputs/report.md).

### 🛡️ What the Analysis Refuses (Diagnostic Gates)
- ❌ **Peaks Over Threshold (POT)**: Attempted and gated out: only 28 exceedances over 57 years against the required minimum of 30, and the fitted shape interval $[-0.556, -0.333]$ lies entirely below zero (bounded upper tail).
- ❌ **Bayesian MCMC Posterior Quantiles**: Withheld unless convergence diagnostics pass (split $\hat{R} \le 1.01$, $\text{ESS} \ge 400$, acceptance rate $\in [0.15, 0.40]$). Current run passes at $\hat{R} = 1.0007$ and $\text{ESS} = 7,576$.
- ❌ **No Observation Is Ever Deleted**: Grubbs, directional Dixon Q, and Rosenblatt tests are recorded; the three lowest peaks are retained and catalogued for gauge record verification.

---

## 🎨 3. Publication-Grade Visualizations

The package generates rich, multi-panel **Plotly (interactive HTML + static high-res PNG)** and **Seaborn/Matplotlib** figures saved directly to `outputs/figures/`:

| Figure ID | Visual Title & Description | Engine |
| :---: | :--- | :---: |
| **P1** | **Interactive Time Series**: Annual peak bars colored by percentile, Sen's slope trend, rolling mean/band, standardized Z-score anomalies | Plotly |
| **P2** | **Distribution Explorer**: 6-panel Seaborn suite (Hist+KDE+rug, log-space histogram, ECDF with theoretical fits, decadal violins, box+strip, multi-dist Q-Q) | Seaborn |
| **P3** | **Interactive Flood Frequency Curves**: All 12 candidate distributions with 95% bootstrap CI band, observed Weibull plotting positions, return period grid | Plotly |
| **P4** | **Hydrologic Feature Structure**: Pairplot (`P4a`), Jointplot (`P4b`), and Clustered Correlation Heatmap (`P4c`) | Seaborn |
| **P5** | **Monte Carlo Uncertainty Fan**: 50% and 90% confidence envelopes with return-period posterior insets | Plotly |
| **P6** | **Bayesian MCMC Posteriors**: 6-panel parameter marginals, 2D joint density, $Q_{100}/Q_{500}$ posteriors, and MCMC trace chains | Plotly |
| **P7** | **Q-Q Probability Grid**: Multi-distribution quantile-quantile diagnostics for top candidates | Plotly |
| **P8** | **Skewness & Kurtosis Moments**: Bootstrap $C_s$, bootstrap $C_k$, moment plane with verified $C_k = 3 + 1.5 C_s^2$ locus, normal probability paper | Seaborn |
| **P9** | **Serial Dependence & Memory**: ACF ($\pm 1.96/\sqrt{N}$ band), PACF, lag plot, Hurst rescaled range, Ljung-Box $p$-values, rolling variance | Plotly |
| **P10** | **Distribution Ranking**: Information criteria comparison (AIC, BIC, AICc, HQIC) and Akaike weights | Plotly |
| **P11** | **POT / GPD Analysis**: Mean Residual Life (MRL) plot, GPD return level curve, threshold stability | Plotly |
| **P12** | **Machine Learning Evaluation**: 6-panel ML panel with out-of-fold predictions, RF feature importances, residuals, time-series reconstructions | Seaborn |
| **P13** | **Interactive Publication Dashboard**: 9-panel comprehensive executive dashboard | Plotly |
| **P14** | **Comprehensive Synthesis Panel**: 12-panel decadal ridgelines, Gumbel probability paper, CDF comparisons, quantile heatmaps | Seaborn |

---

## 🔬 4. Scientific Foundations & Methodology

1. **📐 LP3 Skewness Resolution**:
   - Station log-space skewness: $C_{s,\log} = \mathbf{+0.0956}$ ($SE = 0.3244$), well below the Bulletin 17C/IS 11223 threshold of $0.90$.
   - Raw arithmetic skewness: $C_{s,\text{raw}} = \mathbf{+1.0819}$ (raw asymmetry diagnostic; must **not** be confused with log-space skewness).
2. **⚖️ Unbiased L-Moments**:
   - Hosking (1990) unbiased $b$-statistics: $L_1 = 2856.58\text{ m}^3/\text{s}$, $L_2 = 834.17\text{ m}^3/\text{s}$, $LCV = 0.2920$, $\tau_3 = 0.2465$, $\tau_4 = 0.1306$.
3. **📐 Log-Likelihood Jacobian Correction**:
   - Rigorous comparison between raw and log-space distribution candidates via the transformation Jacobian:
     $$\ln L_Q = \ln L_Y - \sum_{i=1}^n \ln Q_i$$
4. **🔒 Zero Magic Numbers**:
   - Every parameter, standard coefficient, return period, and tolerance is centrally maintained in `ffa_karad/config.py` as the single source of truth.

---

## 📁 5. Repository Structure

```
FFA/
├── .github/workflows/ci.yml                     # 🤖 Multi-OS (Ubuntu, Windows, macOS) CI pipeline
├── .pre-commit-config.yaml                      # 🪝 Git pre-commit hooks (Black, isort, flake8)
├── Dockerfile                                   # 🐳 Production container image
├── docker-compose.yml                           # 🐳 Local multi-container orchestration
├── pyproject.toml                               # 📦 PEP 518/621 package metadata & tool configs
├── requirements.txt                             # 📌 Pinned runtime dependencies
├── data/                                        # 📂 Observed hydrologic data
│   ├── Dischage data KRISHNA KARAD BRIDGE.csv  # 💧 Raw 57-yr observed peaks (verified SHA-256)
│   └── runoff data.xlsx                        # 💧 Supplementary runoff data
├── references/                                  # 📚 Standards, literature & station abstracts
├── notebooks/legacy_v2/                         # 📓 Baseline research notebooks & legacy outputs
├── ffa_karad/                                   # 🐍 Core production Python package
│   ├── __init__.py                             # Package root
│   ├── __main__.py                             # Direct module entry point (`python -m ffa_karad`)
│   ├── config.py                               # Single source of truth (metadata, standards, seeds)
│   ├── cli.py                                  # Stage-by-stage command line interface
│   ├── run_all.py                              # Master end-to-end pipeline driver
│   ├── util.py                                 # Deterministic RNG streams, hashing, I/O helpers
│   ├── data_processing.py                      # Data loading, order statistics, plotting positions
│   ├── quality_control.py                      # 14-check validation battery & rating curve QC
│   ├── lmoments.py                             # Hosking unbiased PWMs & theoretical loci
│   ├── statistical_tests.py                    # Bulletin 17B moments, Mann-Kendall trend, Pettitt test
│   ├── autocorrelation.py                      # ACF, screening band, ESS, Hurst estimators
│   ├── skewness_limits.py                      # LP3 fitting, Wilson-Hilferty, Chowdhury, bootstrap CIs
│   ├── distribution_fitting.py                 # 12 candidate fits, GoF gates, AICc/BIC/HQIC ranking
│   ├── bootstrap_confidence_intervals.py       # Parametric bootstrap confidence bands
│   ├── peaks_over_threshold.py                 # POT fit with exceedance-count & tail-shape gates
│   ├── estimation_design_flood.py              # Adopted design floods, levels, HFL consistency
│   ├── outliers.py                             # Grubbs, Dixon Q, Rosenblatt screening (no deletion)
│   ├── uncertainty_monte_carlo_bayesian_mcmc.py# Record-length MC + gated adaptive-Metropolis MCMC
│   ├── machine_learning.py                     # Leakage-safe exploratory ML vs persistence baseline
│   ├── cwc_manual_check.py                     # Bulletin 17C clause-by-clause checklist
│   ├── visualization.py                        # Publication-grade Plotly & Seaborn figure builders
│   └── reporting.py                            # Markdown report, CSV tables, JSON dump
├── outputs/                                     # 📤 Pipeline generated outputs
│   ├── report.md                               # Generated engineering report
│   ├── results.json                            # Serialized calculation results
│   ├── figures/                                # 17+ publication-grade figures (.png + .html)
│   └── tables/                                 # Structured CSV tables with provenance sidecars
└── tests/                                       # 🧪 Unit and integration test suite (130 tests)
```

---

## 🚀 6. Installation & Quickstart

Requires **Python 3.10+** (tested on Python 3.10, 3.11, and 3.12 across Linux, Windows, and macOS).

```bash
# Clone the repository
git clone https://github.com/your-username/FFA.git
cd FFA

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## ⚙️ 7. Running the Analysis

### 🔄 Master Pipeline
```bash
# Run the complete end-to-end analysis
python -m ffa_karad.run_all

# Or via the installed console entrypoint
ffa-karad --outdir outputs
```

### 🎛️ Stage-by-Stage CLI
```bash
# Run specific stages
python -m ffa_karad.cli --stages qc stats skew fit pot design ml mcmc

# Fast run without generating figures
python -m ffa_karad.run_all --no-figures
```

The RNG is deterministically seeded (`MASTER_SEED = 20260902`), ensuring bitwise-reproducible tables and bootstrap intervals across executions.

---

## 🧪 8. Verification & Test Suite

Run the full pytest suite:

```bash
pytest tests/ -v
```

- **130 / 130 tests passing (100% green)**
- **83% code coverage**
- Automated validation checks:
  - Strict preservation of observation counts ($N = 57$)
  - Verification of temporal water-year order (rejection of sorted vectors)
  - Zero data leakage in ML feature pipelines
  - Strict diagnostic gating on POT and MCMC quantiles

---

## 🐳 9. Docker & Containerized Execution

Build and run using Docker:

```bash
# Build the Docker image
docker build -t ffa-karad .

# Run the analysis inside the container
docker run --rm -v $(pwd)/outputs:/app/outputs ffa-karad
```

Or with Docker Compose:

```bash
docker compose up
```

---

## 📋 10. Engineering Standards & References

- 📜 **IS 11223:1985**: Indian Standard Guidelines for Fixation of Spillway Capacity
- 📜 **CWC Flood Estimation Reports**: Krishna Basin Sub-zone 3(h)
- 📜 **IRC:5-2015**: Standard Specifications and Code of Practice for Road Bridges
- 📜 **USGS Bulletin 17C**: Guidelines for Determining Flood Flow Frequency (2018)
- 📜 **Hosking, J. R. M. (1990)**: L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *J. R. Statist. Soc. B*, 52(1), 105–124.

