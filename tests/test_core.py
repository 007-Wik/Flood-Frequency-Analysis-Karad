"""Core test battery verifying ingestion, L-moments, and LP3 fitting."""

import numpy as np

from ffa_karad import config
from ffa_karad import data_processing as dp
from ffa_karad import lmoments as lm
from ffa_karad import quality_control as qc
from ffa_karad import skewness_limits as sl
from ffa_karad import statistical_tests as st


def test_data_ingestion_and_sha256():
    bundle = dp.load()
    assert bundle.n_records == 57
    assert len(bundle.q) == 57
    assert len(bundle.years) == 57
    assert bundle.source_sha256 in config.RAW_DATA_VALID_SHA256S
    assert np.isclose(np.mean(bundle.q), 2856.58, atol=0.1)


def test_quality_control_battery():
    bundle = dp.load()
    report = qc.run(bundle)
    assert report.admissible
    assert not report.blocking
    assert report.n_fail == 0


def test_unbiased_lmoments():
    bundle = dp.load()
    lmo = lm.sample_lmoments(bundle.q)
    assert np.isclose(lmo["L1"], 2856.58, atol=0.1)
    assert np.isclose(lmo["L2"], 834.17, atol=0.1)
    assert np.isclose(lmo["lcv"], 0.2920, atol=0.001)
    assert np.isclose(lmo["lskew"], 0.2465, atol=0.001)
    assert np.isclose(lmo["lkurt"], 0.1306, atol=0.001)


def test_moments_and_log_skewness():
    bundle = dp.load()
    m_raw = st.moment_coefficients(bundle.q)
    assert np.isclose(m_raw.skewness_adjusted_b17b, 1.0819, atol=0.001)
    assert np.isclose(m_raw.kurtosis_moment_b17b, 3.4683, atol=0.001)

    m_log = st.moment_coefficients(bundle.log_q)
    assert np.isclose(m_log.skewness_adjusted_b17b, 0.0956, atol=0.001)


def test_lp3_design_floods():
    bundle = dp.load()
    fit = sl.lp3_fit(bundle.q)
    assert np.isclose(fit.cs_log, 0.0956, atol=0.001)
    q100 = sl.lp3_quantile(100.0, fit.mean_log, fit.sd_log, fit.cs_log)
    assert np.isclose(q100, 8667.0, atol=25.0)
