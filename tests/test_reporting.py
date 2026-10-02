"""Tests for :mod:`ffa_karad.reporting` and the pipeline driver.

The property worth protecting is that the report cannot print a number the
analysis did not produce, and cannot print a gated result as if it were
adopted.  A report that quietly omits a refusal is worse than no report.
"""

from __future__ import annotations

import json

import pytest

from ffa_karad import cwc_manual_check as cc
from ffa_karad import data_processing as dp
from ffa_karad import distribution_fitting as df
from ffa_karad import estimation_design_flood as edf
from ffa_karad import outliers as ol
from ffa_karad import peaks_over_threshold as pot
from ffa_karad import quality_control as qc
from ffa_karad import reporting as rep
from ffa_karad import run_all


@pytest.fixture(scope="module")
def results():
    bundle = dp.load(force_embedded=True)
    q = bundle.q
    trend = {
        "mann_kendall": {"tau": -0.095, "p_value": 0.299},
        "pettitt": {
            "change_point_index": 50,
            "p_value": 0.211,
            "reject_h0_at_alpha": False,
        },
    }
    return {
        "qc": qc.run(bundle),
        "design": edf.run(q),
        "fits": df.run(q),
        "outliers": ol.run(
            bundle.q_ordered,
            years=list(bundle.frame["wy_start_year"]),
            labels=list(bundle.frame["wy_label"]),
        ),
        "pot": pot.run(bundle.q_ordered),
        "checklist": cc.run(
            qc.run(bundle),
            edf.run(q),
            df.run(q),
            ol.run(q, labels=list(bundle.frame["wy_label"])),
            pot.run(bundle.q_ordered),
            trend,
        ),
    }


def test_report_names_the_method_and_its_standard(results):
    text = rep.build_markdown(results)
    assert "log-Pearson type III" in text
    assert "IS 11223:1985" in text
    assert "AK000X6" in text


def test_report_carries_the_confidence_band_with_the_headline(results):
    text = rep.build_markdown(results)
    table = results["design"].table.set_index("return_period_yr")
    q100 = float(table.loc[100.0, "adopted_lp3_cumecs"])
    low = float(table.loc[100.0, "adopted_ci_lower_cumecs"])
    high = float(table.loc[100.0, "adopted_ci_upper_cumecs"])
    assert f"{q100:,.0f}" in text
    assert f"{low:,.0f}" in text and f"{high:,.0f}" in text


def test_report_never_uses_scientific_notation_for_discharges(results):
    """`%.4g` prints the 50-year upper limit as 1.027e+04, which a reviewer
    reading a government document will read as a typo.  Probabilities keep
    scientific notation, where it belongs."""
    text = rep.build_markdown(results)
    discharge_columns = (
        "adopted_lp3_cumecs",
        "adopted_ci_lower_cumecs",
        "adopted_ci_upper_cumecs",
        "gumbel_ev1_cumecs",
        "ln2_cumecs",
        "discharge_cumecs",
        "water_level_m",
        "depth_over_zero_gauge_m",
        "top_of_structure_m",
    )
    for line in text.splitlines():
        if not line.startswith("|") or not any(c in line for c in discharge_columns):
            continue
        for cell in line.strip("|").split("|"):
            cell = cell.strip()
            if not cell:
                continue
            try:
                float(cell.replace(",", ""))
            except ValueError:
                continue
            assert "e" not in cell.lower(), f"scientific notation in {line!r}"


def test_report_states_the_refused_pot_instead_of_omitting_it(results):
    assert results["pot"].accepted is False
    text = rep.build_markdown(results)
    assert "not adopted" in text
    assert "no POT return level appears" in text


def test_report_lists_manual_evidence_as_outstanding_work(results):
    text = rep.build_markdown(results)
    assert "Outstanding manual evidence" in text
    assert "17C 2.3" in text
    assert "17C 2.5" in text


def test_report_says_nothing_is_adopted_when_design_failed():
    """A withheld adoption must be stated at the top, not left implied."""
    design = edf.run(dp.load(force_embedded=True).q)
    design.adopted = False
    design.reject_reasons = ["LP3 failed the HFL consistency check"]
    text = rep.build_markdown({"design": design})
    assert "No value is adopted" in text
    assert "HFL consistency check" in text


def test_report_survives_missing_stages():
    text = rep.build_markdown({})
    assert "nothing can be adopted" in text
    assert "not available" in text or "_" in text


def test_run_writes_report_tables_and_json(results, tmp_path):
    artifacts = rep.run(results, (), tmp_path, station="AK000X6")
    assert artifacts.markdown_path.exists()
    assert artifacts.json_path is not None and artifacts.json_path.exists()
    assert "design_floods" in artifacts.csv_paths
    assert "checklist" in artifacts.csv_paths
    sidecar = artifacts.csv_paths["design_floods"].with_suffix(".meta.json")
    assert sidecar.exists()
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["station"] == "AK000X6"
    assert payload["design"]["adopted"] is True


def test_run_all_is_reproducible(tmp_path):
    """Two short runs must agree exactly; the RNG is seeded."""
    first = run_all.run(
        outdir=tmp_path / "a", n_mc_records=8, mcmc_draws=400, write_figures=False
    )
    second = run_all.run(
        outdir=tmp_path / "b", n_mc_records=8, mcmc_draws=400, write_figures=False
    )
    assert first.ok and second.ok, [s.detail for s in first.stages if not s.ok]
    a = (tmp_path / "a" / "tables" / "design_floods.csv").read_text()
    b = (tmp_path / "b" / "tables" / "design_floods.csv").read_text()
    assert a == b


def test_run_all_records_every_stage(tmp_path):
    outcome = run_all.run(
        outdir=tmp_path, n_mc_records=5, mcmc_draws=300, write_figures=False
    )
    names = [s.name for s in outcome.stages]
    for expected in (
        "data_processing",
        "quality_control",
        "estimation_design_flood",
        "monte_carlo_record_uncertainty",
        "bayesian_mcmc",
        "machine_learning",
        "cwc_manual_check",
        "reporting",
    ):
        assert expected in names
    assert outcome.artifacts is not None
    assert outcome.artifacts.markdown_path.exists()


def test_run_all_tolerates_a_failing_stage(tmp_path, monkeypatch):
    """One broken stage must not cost the whole report."""

    def boom(bundle):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(run_all._ml, "run", boom)
    outcome = run_all.run(
        outdir=tmp_path, n_mc_records=5, mcmc_draws=300, write_figures=False
    )
    assert not outcome.ok
    failed = [s for s in outcome.stages if not s.ok]
    assert [s.name for s in failed] == ["machine_learning"]
    assert "simulated stage failure" in failed[0].detail
    assert outcome.artifacts is not None
    assert outcome.artifacts.markdown_path.exists()


def test_run_all_exits_zero_on_success(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        run_all,
        "run",
        lambda **kwargs: run_all.PipelineResult(results={}, stages=[], artifacts=None),
    )
    assert run_all.main(["--outdir", str(tmp_path), "--quiet"]) == 0
