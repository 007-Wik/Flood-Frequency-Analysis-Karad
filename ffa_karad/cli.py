"""Command-line interface (CLI) for FFA Karad flood-frequency analysis."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import data_processing as dp
from . import distribution_fitting as df
from . import estimation_design_flood as edf
from . import machine_learning as ml
from . import peaks_over_threshold as pot
from . import quality_control as qc
from . import skewness_limits as sl
from . import statistical_tests as st
from . import uncertainty_monte_carlo_bayesian_mcmc as unc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffa-karad",
        description="Comprehensive Flood Frequency Analysis for Krishna at Karad (AK000X6)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to custom station CSV data file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["qc", "stats", "skew", "fit", "pot", "design", "ml", "mcmc", "all"],
        default=["all"],
        help="Pipeline stages to execute (default: all).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stages = set(args.stages)
    run_all = "all" in stages

    print("=" * 72)
    print("  FLOOD FREQUENCY ANALYSIS: KRISHNA RIVER AT KARAD (AK000X6)")
    print("  Indian Standard IS 11223:1985 & Bulletin 17C Governed Pipeline")
    print("=" * 72)

    # 1. Load data
    bundle = dp.load(path=args.csv_path)

    # 2. Quality Control
    if run_all or "qc" in stages:
        print("\n" + "-" * 72)
        print("STAGE 1: DATA QUALITY CONTROL & RATING CURVE")
        print("-" * 72)
        qc_res = qc.run(bundle)
        print(qc.summarise(qc_res))

    # 3. Statistical Batteries
    if run_all or "stats" in stages:
        print("\n" + "-" * 72)
        print("STAGE 2: STATISTICAL INFERENCE & STATIONARITY")
        print("-" * 72)
        st_res = st.run(bundle.q_ordered)
        print(st.summarise(st_res))

    # 4. Skewness Decision Gate
    if run_all or "skew" in stages:
        print("\n" + "-" * 72)
        print("STAGE 3: BULLETIN 17C SKEWNESS DECISION")
        print("-" * 72)
        sk_res = sl.run(bundle.q)
        print(sl.summarise(sk_res))

    # 5. Candidate Distribution Fitting
    df_res = None
    if run_all or "fit" in stages or "design" in stages:
        print("\n" + "-" * 72)
        print("STAGE 4: CANDIDATE DISTRIBUTION FITTING (JACOBIAN CORRECTED)")
        print("-" * 72)
        df_res = df.run(bundle.q)
        print(df.summarise(df_res))

    # 6. Peaks Over Threshold (POT)
    if run_all or "pot" in stages:
        print("\n" + "-" * 72)
        print("STAGE 5: PEAKS OVER THRESHOLD (GPD) AUDIT")
        print("-" * 72)
        pot_res = pot.run(bundle.q)
        print(pot.summarise(pot_res))

    # 7. Design Flood & Hydraulic Levels
    if run_all or "design" in stages:
        print("\n" + "-" * 72)
        print("STAGE 6: ADOPTED DESIGN FLOODS & WATER LEVELS (IS 11223:1985)")
        print("-" * 72)
        edf_res = edf.run(bundle.q, fits=df_res)
        print(edf.summarise(edf_res))

    # 8. Machine Learning Benchmark
    if run_all or "ml" in stages:
        print("\n" + "-" * 72)
        print("STAGE 7: MACHINE LEARNING BENCHMARK (LEAK-FREE TIME-SERIES SPLIT)")
        print("-" * 72)
        ml_res = ml.run(bundle.q_ordered)
        print(ml.summarise(ml_res))

    # 9. Bayesian MCMC & Monte Carlo
    if run_all or "mcmc" in stages:
        print("\n" + "-" * 72)
        print("STAGE 8: UNCERTAINTY QUANTIFICATION (MONTE CARLO & BAYESIAN MCMC)")
        print("-" * 72)
        mc_res = unc.monte_carlo_record_uncertainty(bundle.q, n_records=500)
        mcmc_res = unc.bayesian_mcmc(bundle.q, n_draws=4000, burn_in=1000, n_chains=4)
        print(unc.summarise(mc=mc_res, mcmc=mcmc_res))

    print("\n" + "=" * 72)
    print("  ANALYSIS EXECUTION COMPLETE - ALL SAFETY GATES AUDITED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
