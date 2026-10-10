"""Attainment + inflation + PLO roll-up tests (deterministic, no network)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from skillleak import attainment as at
from skillleak import clo
from test_clo import scan

SAMPLE = (ROOT / "sample_data/marks_ee357_synthetic.csv").read_bytes()


def test_parse_and_compute():
    sheet, warns = at.parse_marks_csv(SAMPLE)
    assert len(sheet["students"]) == 40
    assert any("no CLO tag" in w for w in warns)
    rows = {r["clo"]: r for r in at.compute_attainment(sheet, 0.5, 0.6)}
    assert set(rows) == {"CLO-1", "CLO-2", "CLO-3", "CLO-4", "CLO-5", "CLO-6"}
    assert rows["CLO-3"]["max_marks"] == 40          # Final Q5 (10) + Lab exam (30)
    assert rows["CLO-5"]["class_attainment"] >= 0.9
    strict = {r["clo"]: r for r in at.compute_attainment(sheet, 0.9, 0.9)}
    assert not strict["CLO-2"]["attained"]


def test_inflation_flag(monkeypatch):
    run = scan(monkeypatch)
    res = clo.analyze(run["alignment"])
    sheet, _ = at.parse_marks_csv(SAMPLE)
    inf = {r["clo"]: r for r in at.inflation_check(at.compute_attainment(sheet), res["rows"])}
    assert inf["CLO-5"]["flag"] == "Potential attainment-validity concern"
    assert "C1" in inf["CLO-5"]["note"] or "C2" in inf["CLO-5"]["note"]
    assert inf["CLO-6"]["flag"] == ""                      # attained and measured at level
    assert inf["CLO-2"]["flag"] == "Not checked"  # fake reply has no CLO-2


def test_plo_rollup_and_marks_by_level(monkeypatch):
    run = scan(monkeypatch)
    res = clo.analyze(run["alignment"])
    plos = {p["plo"]: p for p in at.plo_rollup(res["rows"])}
    assert plos["PLO-2"]["evidence"] == "Weak" and plos["PLO-2"]["weak_clos"] == ["CLO-5"]
    assert plos["PLO-5"]["evidence"] == "Supported"
    mbl = at.marks_by_level(run["alignment"])
    assert mbl[1] == 10 and mbl[3] == 10 and 2 in mbl   # A1(5,C1)+A6(5,C1); A5 C3 (10)


@pytest.mark.parametrize("bad, msg", [
    ("Student,Q1\nS01,5\n", "CLO"),
    ("Student\nCLO\nMAX\n", "student column"),
    ("Student,Q1\nCLO,CLO-1\nMAX,5\n", "No student rows"),
])
def test_bad_sheets(bad, msg):
    with pytest.raises(ValueError, match=msg):
        at.parse_marks_csv(bad)


def test_messy_values_are_handled():
    sheet, warns = at.parse_marks_csv("Student,Q1,Q2\nCLO,CLO 1,clo-1/CLO-2\nMAX,5,x\nS01,9,abc\nS02,,3\n")
    assert any("capped" in w for w in warns) and any("no valid maximum" in w for w in warns)
    rows = at.compute_attainment(sheet)
    assert [r["clo"] for r in rows] == ["CLO 1"]   # Q2 ignored (no max); "CLO 1" and "clo-1" merge
    assert rows[0]["max_marks"] == 5
