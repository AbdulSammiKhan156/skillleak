"""Run with:  python -m pytest -q   (no API key or internet needed)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skillleak import pipeline, clo
from skillleak.extract import load_sample_documents
from skillleak.llm import LLMConfig, parse_json
from skillleak.report import run_markdown, lab_markdown
from skillleak.scoring import verify_quote, demand_weight, score_run, coverage_index
from fake_llm import fake_call_json

CFG = LLMConfig(provider="Google Gemini", api_key="test", model="fake")


def make_run(monkeypatch, **kw):
    monkeypatch.setattr(pipeline, "call_json", fake_call_json)
    monkeypatch.setattr(clo, "call_json", fake_call_json)
    import skillleak.bloom as _bloom
    monkeypatch.setattr(_bloom, "call_json", fake_call_json)
    return pipeline.run_scan(load_sample_documents(), CFG, **kw)


def test_sample_documents_load():
    docs = load_sample_documents()
    assert [d.doc_id for d in docs] == ["SYL1", "LAB1", "ASM1", "JD1", "JD2", "JD3", "JD4"]
    assert all(len(d.text) > 200 and not d.warning for d in docs)


def test_verify_quote():
    src = "Monitor network health using NMS dashboards and SNMP-based alarms, and respond."
    assert verify_quote("monitor network health using NMS dashboards", src) == "exact"
    assert verify_quote("Monitor network health using NMS dashboards and SNMP based alarms and respond", src) == "exact"
    assert verify_quote("Monitor network health using the NMS dashboards and SNMP-based alarms", src) == "approx"
    assert verify_quote("Configure BGP peering with upstream providers", src) == "not_found"
    assert verify_quote("", src) == "not_found"


def test_demand_weight():
    assert [demand_weight(n, 4) for n in range(5)] == [0, 2, 3, 4, 5]
    assert demand_weight(1, 1) == 5


def test_scoring_end_to_end(monkeypatch):
    run = make_run(monkeypatch)
    rows = {r["name"]: r for r in score_run(run)}
    ts = rows["Troubleshoot network connectivity faults"]
    assert (ts["D"], ts["T"], ts["P"], ts["A"]) == (5, 1, 0, 0)
    assert ts["status"] == "Phantom competence risk" and ts["priority"] == 10
    assert "Remember" in ts["diagnosis"]

    rs = rows["Configure routing and switching"]
    assert rs["status"] == "Covered" and rs["priority"] == 0 and rs["D"] == 4

    doc = rows["Document incidents and write incident reports"]
    assert doc["jd_count"] == 2  # hallucinated JD2 quote rejected
    assert doc["status"] == "Industry blind spot" and doc["priority"] == 3 * 3
    assert run["meta"]["quotes_rejected"] == 3  # 1 industry quote + 1 invented exam item + 1 invented CLO
    assert 0 < coverage_index(list(rows.values())) < 100


def test_lab_and_reports(monkeypatch, tmp_path):
    run = make_run(monkeypatch)
    lab = pipeline.generate_lab(run, "C1", CFG, feedback="Use GNS3")
    assert lab["_meta"]["revised_with_feedback"] and lab["review"]["status"] == "Pending review"
    run["labs"]["C1"] = lab
    md = run_markdown(run)
    assert "Phantom competence risk" in md and "Fault Hunt" in md
    assert "| Fault isolation | 50 |" in lab_markdown(lab)
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    p = pipeline.save_run(run)
    assert pipeline.load_run(p)["labs"]["C1"]["title"] == lab["title"]


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}
