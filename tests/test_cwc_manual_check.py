"""Tests for :mod:`ffa_karad.cwc_manual_check`.

The behaviour worth protecting: an item the data cannot settle is never
reported as a pass, and nothing is adopted while a clause has failed.
"""

from __future__ import annotations

import pytest

from ffa_karad import cwc_manual_check as cc
from ffa_karad import data_processing as dp
from ffa_karad import distribution_fitting as df
from ffa_karad import estimation_design_flood as design
from ffa_karad import outliers as ol
from ffa_karad import peaks_over_threshold as pot
from ffa_karad import quality_control as qc
from ffa_karad import statistical_tests as st


@pytest.fixture(scope="module")
def bundle():
    return dp.load(force_embedded=True)


@pytest.fixture(scope="module")
def checklist(bundle):
    trend = {
        "mann_kendall": st.mann_kendall(bundle.q_ordered),
        "pettitt": st.pettitt_test(bundle.q_ordered),
    }
    return cc.run(
        qc.run(bundle),
        design.run(bundle.q),
        df.run(bundle.q),
        ol.run(bundle.q),
        pot.run(bundle.q),
        trend,
    )


def test_every_clause_has_evidence_and_an_owner(checklist):
    frame = cc.to_frame(checklist)
    assert len(frame) >= 15
    assert frame["evidence"].str.len().min() > 20
    assert frame["responsible"].str.len().min() > 3
    assert set(frame["status"]) <= {cc.PASS, cc.FAIL, cc.WARN, cc.MANUAL, cc.N_A}


def test_unsettable_items_are_manual_not_passed(checklist):
    by_clause = {i.clause: i for i in checklist.items}
    for clause in ("17C 2.2", "17C 2.3", "17C 2.4", "17C 2.5", "17C 4.3"):
        assert by_clause[clause].status == cc.MANUAL, clause
    assert by_clause["17C 2.2"].responsible == "hydrologist"


def test_missing_inputs_produce_manual_items_not_passes():
    """With no data at all, nothing data-dependent may be scored as a pass.

    Two clauses describe properties of the package rather than of the record --
    that the observed points are plotted on the curve, and that the quantile
    function is arithmetically verified -- and those stay as they are.
    """
    bare = cc.run()
    data_dependent = [i for i in bare.items if i.clause not in {"17C 5.1", "17C 5.2"}]
    assert all(item.status == cc.MANUAL for item in data_dependent)
    assert not bare.failures
    assert cc.to_frame(bare)["clause"].tolist()[1] == "17C 2.2"


def test_refused_pot_counts_as_handling_the_short_record(checklist):
    by_clause = {i.clause: i for i in checklist.items}
    item = by_clause["17C 3.3"]
    assert item.status == cc.PASS
    assert "refused" in item.evidence


def test_a_detected_change_point_goes_to_manual(bundle):
    """A significant Pettitt result must not be scored as a clean pass."""
    trend = {
        "mann_kendall": st.mann_kendall(bundle.q_ordered),
        "pettitt": dict(st.pettitt_test(bundle.q_ordered), reject_h0_at_alpha=True),
    }
    result = cc.run(qc.run(bundle), design.run(bundle.q), trend=trend)
    by_clause = {i.clause: i for i in result.items}
    assert by_clause["17C 3.1"].status == cc.MANUAL


def test_counts_add_up(checklist):
    counts = checklist.counts()
    assert sum(counts.values()) == len(checklist.items)


def test_summarise_lists_outstanding_work(checklist):
    text = cc.summarise(checklist)
    assert "AWAITING MANUAL EVIDENCE" in text
    assert "not failures" in text
    assert "rating curve" in text
