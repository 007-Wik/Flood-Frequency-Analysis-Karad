# Flood Frequency Analysis: Krishna River at Karad (AK000X6)

```text
                     ███████╗███████╗ █████╗     ██╗  ██╗ █████╗ ██████╗  █████╗ ██████╗ 
                     ██╔════╝██╔════╝██╔══██╗    ██║ ██╔╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗
                     █████╗  █████╗  ███████║    █████╔╝ ███████║██████╔╝███████║██║  ██║
                     ██╔══╝  ██╔══╝  ██╔══██║    ██╔═██╗ ██╔══██║██╔══██╗██╔══██║██║  ██║
                     ██║     ██║     ██║  ██║    ██║  ██╗██║  ██║██║  ██║██║  ██║██████╔╝
                     ╚═╝     ╚═╝     ╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
                         Flood Frequency Analysis · Krishna Basin · CWC & IS 11223
```

<p align="center">
  <!-- GitHub Stats -->
  <a href="https://github.com/007-Wik/Flood-Frequency-Analysis-Karad/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/007-Wik/Flood-Frequency-Analysis-Karad/ci.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=CI%20Build" alt="CI Status" />
  </a>
  <a href="https://github.com/007-Wik/Flood-Frequency-Analysis-Karad/stargazers">
    <img src="https://img.shields.io/github/stars/007-Wik/Flood-Frequency-Analysis-Karad?style=for-the-badge&color=gold&logo=github&logoColor=white" alt="GitHub Stars" />
  </a>
  <a href="https://github.com/007-Wik/Flood-Frequency-Analysis-Karad/network/members">
    <img src="https://img.shields.io/github/forks/007-Wik/Flood-Frequency-Analysis-Karad?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Forks" />
  </a>
  <a href="https://github.com/007-Wik/Flood-Frequency-Analysis-Karad/issues">
    <img src="https://img.shields.io/github/issues/007-Wik/Flood-Frequency-Analysis-Karad?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Issues" />
  </a>
  <br>
  
  <!-- Python & Code Quality -->
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Versions" />
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=python&logoColor=white" alt="Code Style: Black" />
  </a>
  <a href="https://pycqa.github.io/isort/">
    <img src="https://img.shields.io/badge/Imports-isort-1674b1?style=for-the-badge&logo=python&logoColor=white" alt="Imports: isort" />
  </a>
  <a href="https://flake8.pycqa.org/">
    <img src="https://img.shields.io/badge/Linter-Flake8-informational?style=for-the-badge&logo=python&logoColor=white" alt="Linter: Flake8" />
  </a>
  <img src="https://img.shields.io/badge/Tests-130%20Passed-success?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests" />
  <img src="https://img.shields.io/badge/Coverage-83%25-brightgreen?style=for-the-badge&logo=codecov&logoColor=white" alt="Coverage" />
  <br>
  
  <!-- Tech Stack & Libraries -->
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white" alt="Markdown" />
  <br>
  
  <!-- Project Standards & Environment -->
  <a href="https://github.com/007-Wik/Flood-Frequency-Analysis-Karad/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/007-Wik/Flood-Frequency-Analysis-Karad?style=for-the-badge&color=2EA44F" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/Standard-IS%2011223%3A1985-007ACC?style=for-the-badge" alt="IS 11223:1985" />
  <img src="https://img.shields.io/badge/Guideline-CWC%20%7C%20USGS%2017C-blueviolet?style=for-the-badge" alt="CWC & USGS 17C" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
</p>

---

## 1. Quickstart & Execution

### Python Setup
```bash
git clone https://github.com/007-Wik/Flood-Frequency-Analysis-Karad.git
cd Flood-Frequency-Analysis-Karad
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Run the complete end-to-end analysis
python -m ffa_karad.run_all
```

### Docker (Containerized Build)
```bash
# Build the Docker image
docker build -t ffa-karad .

# Run the analysis inside the container
docker run --rm -v $(pwd)/outputs:/app/outputs ffa-karad
```

---

## 2. Project Details & Results

<details>
<summary><strong>System Architecture & Methodology Flow</strong></summary>

```text
+-----------------------------------------------------------------------------+
|                     INPUT DATA: KRISHNA AT KARAD (AK000X6)                  |
|          57-Year Annual Maximum Discharge Record (1965-66 to 2021-22)       |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                       STAGE 1: DATA INTEGRITY & AUDIT                       |
|   +-----------------------+  +----------------------+  +------------------+ |
|   |   14-Point QC Check   |  |  Temporal Order QC   |  | Rating Curve QC  | |
|   |   Zero gaps or dups   |  | Reject sorted vector |  |  HFL vs Peak Q   | |
|   +-----------------------+  +----------------------+  +------------------+ |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     STAGE 2: STATISTICAL & MOMENT ANALYSIS                  |
|   +-----------------------+  +----------------------+  +------------------+ |
|   | Hosking L-Moments     |  | Bulletin 17B Moments |  | Serial Memory    | |
|   | Unbiased b-statistics |  | Log-space moments    |  | ACF & Hurst R/S  | |
|   | tau3=0.2465,tau4=0.13 |  | Cs(log) = +0.0956    |  | Lag-1 r = +0.171 | |
|   +-----------------------+  +----------------------+  +------------------+ |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                    STAGE 3: CANDIDATE DISTRIBUTION FITTING                  |
|   +-----------------------+  +----------------------+  +------------------+ |
|   | 12 Standard Families  |  | Information Metric   |  | Diagnostic Gates | |
|   | LP3, EV1, LN2, GEV,   |  | AICc / BIC / HQIC    |  | Jacobian Log-Lik | |
|   | Weibull, Gamma, etc.  |  | Model Selection Rank |  | floc=0 on LN2    | |
|   +-----------------------+  +----------------------+  +------------------+ |
+--------------------------------------┬--------------------------------------+
                                       |
               +-----------------------+-----------------------+
               |                                               |
               v                                               v
+------------------------------+               +------------------------------+
|  STAGE 4A: POT / GPD BRANCH  |               |  STAGE 4B: BAYESIAN MCMC UQ  |
|  - Mean Residual Life (MRL)  |               |  - Adaptive Metropolis MCMC  |
|  - Tail Shape & Stability    |               |  - Convergence Gating:       |
|  - Gate: REJECTED            |               |    R-hat <= 1.01, ESS >= 400 |
|    (N_exc = 28 < 30 min)     |               |    (PASSED: R-hat = 1.0007)  |
+--------------┬---------------+               +--------------┬---------------+
               |                                              |
               +-----------------------+----------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                   STAGE 5: ADOPTED DESIGN FLOOD ESTIMATION                  |
|   +---------------------------------------------------------------------+   |
|   | Primary Design Method: Log-Pearson Type III (IS 11223:1985)         |   |
|   | Q50 = 7,451 m3/s  |  Q100 = 8,667 m3/s  |  Q1000 = 13,327 m3/s      |   |
|   | 95% Parametric Bootstrap Confidence Intervals                       |   |
|   +---------------------------------------------------------------------+   |
|   | CWC Comparative Cross-Checks: Gumbel EV1 (Raw Scale) & LN2 (floc=0) |   |
|   | USGS Bulletin 17C Checklist Audit (13 Passed, 5 Manual Field Verifs)|   |
|   +---------------------------------------------------------------------+   |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                       STAGE 6: REPRODUCIBLE ARTIFACTS                       |
|   +-----------------------+  +----------------------+  +------------------+ |
|   |  Engineering Report   |  | CSV Tables & Schemas |  | Multi-Panel Figs | |
|   |  (outputs/report.md)  |  | Provenance sidecars  |  | Plotly & Seaborn | |
|   +-----------------------+  +----------------------+  +------------------+ |
+-----------------------------------------------------------------------------+
```
</details>

<details>
<summary><strong>Station Overview & Adopted Results</strong></summary>

- **Station Code**: AK000X6 (Krishna River at Karad Bridge, Maharashtra, India)
- **Catchment Area**: 5,462.0 km² | **Zero of Gauge**: 549.915 m (MSL)
- **Observed HFL**: 567.162 m (7,177 m³/s on 1976-06-07)
- **Record**: 57 Water Years (1965–66 to 2021–22) | **100% Complete**

**Adopted Design Floods (LP3 on $\ln Q$, `n = 57`)**:

| Return Period $T$ | Adopted $Q$ (m³/s) | 95% Confidence Interval | Gumbel EV1 | LN2 | Spread |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **50-yr** | **7,451** | 5,484 – 10,266 | 6,423 | 7,189 | 14% |
| **100-yr** | **8,667** | 6,048 – 12,923 | 7,181 | 8,270 | 17% |
| **1,000-yr** | **13,327** | 7,622 – 25,397 | 9,687 | 12,243 | 27% |
</details>

<details>
<summary><strong>Repository Structure & Tests</strong></summary>

```
FFA/
├── ffa_karad/                                   # Core production Python package
├── data/                                        # Observed hydrologic data
├── outputs/                                     # Pipeline generated outputs (Reports, Figures, Tables)
├── tests/                                       # Unit and integration test suite (130 tests)
├── .github/workflows/ci.yml                     # Multi-OS CI pipeline
├── Dockerfile & docker-compose.yml              # Container infrastructure
└── pyproject.toml / requirements.txt            # Package metadata & dependencies
```

- **Verification**: Run `pytest tests/ -v`. **130 / 130 tests passing (100% green)** at 83% coverage.
</details>

<details>
<summary><strong>Engineering Standards</strong></summary>

- **IS 11223:1985**: Indian Standard Guidelines for Fixation of Spillway Capacity
- **CWC Flood Estimation Reports**: Krishna Basin Sub-zone 3(h)
- **USGS Bulletin 17C**: Guidelines for Determining Flood Flow Frequency (2018)
</details>

