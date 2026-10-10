"""CLO-assessment alignment tests (no network)."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skillleak import pipeline, clo
from skillleak.extract import load_sample_documents, build_manual_documents
from skillleak.llm import LLMConfig
from skillleak.report import alignment_markdown
from fake_llm import fake_call_json

CFG = LLMConfig(provider="Google Gemini", api_key="test", model="fake")


def scan(monkeypatch, docs=None, **kw):
    monkeypatch.setattr(pipeline, "call_json", fake_call_json)
    monkeypatch.setattr(clo, "call_json", fake_call_json)
    import skillleak.bloom as _bloom
    monkeypatch.setattr(_bloom, "call_json", fake_call_json)
    return pipeline.run_scan(docs or load_sample_documents("public"), CFG, **kw)


def findings(al):
    return {r["id"]: r["finding"] for r in clo.analyze(al)["rows"]}


def test_alignment_findings(monkeypatch):
    run = scan(monkeypatch)
    al = run["alignment"]
    f = findings(al)
    assert f["CLO-1"] == "Potential mismatch"          # C2 CLO, only a C1 recall question
    assert f["CLO-3"] == "No apparent mismatch detected"  # lab exam reaches C3 despite a C2 question
    assert f["CLO-5"] == "Potential mismatch"          # C4 CLO, only "list the steps" (C1)
    assert f["CLO-6"] == "No apparent mismatch detected"
    assert f["CLO-9"] == "Insufficient evidence"
    res = clo.analyze(al)
    # invented item A7 is rejected by quote verification and never counts
    assert any(i["id"] == "A7" for i in res["rejected_items"])
    a7 = next(m for m in res["mapping_rows"] if m["item_id"] == "A7")
    assert not a7["included"] and not a7["quote_verified"]
    # unknown CLO mapping dropped; SNMP question is unmapped
    assert all(m["clo_id"] != "CLO-404" for m in al["mappings"])
    assert [i["id"] for i in res["unmapped_items"]] == ["A6"]
    sm = res["summary"]
    assert (sm["clos"], sm["mismatch"], sm["aligned"], sm["insufficient"]) == (5, 2, 2, 1)
    assert "could not be matched to the source" in next(r for r in res["rows"] if r["id"] == "CLO-9")["explanation"]
    row3 = next(r for r in res["rows"] if r["id"] == "CLO-3")
    assert "below the CLO level" in row3["explanation"]


def test_faculty_overrides(monkeypatch):
    al = scan(monkeypatch)["alignment"]
    al["overrides"] = {"clo_level": {"CLO-1": 1}, "item_level": {"A4": 4}, "exclude": ["CLO-3|A3"]}
    f = findings(al)
    assert f["CLO-1"] == "No apparent mismatch detected"
    assert f["CLO-5"] == "No apparent mismatch detected"
    assert f["CLO-3"] == "Potential mismatch"


def test_marks_and_inferred_rules(monkeypatch):
    al = copy.deepcopy(scan(monkeypatch)["alignment"])
    # give the lab exam marks: CLO-3 now has 5 marks at level vs 10 below -> Needs review
    for it in al["items"]:
        if it["id"] == "A3":
            it["marks"] = 5
    assert findings(al)["CLO-3"] == "Needs review"
    # an inferred (not stated) level never produces a hard mismatch
    for c in al["clos"]:
        if c["id"] == "CLO-5":
            c["stated_level"], c["inferred_level"] = None, 4
    assert findings(al)["CLO-5"] == "Needs review"
    # low-confidence content match only -> Needs review
    for m in al["mappings"]:
        if m["clo_id"] == "CLO-6":
            m["basis"], m["confidence"] = "content_match", "low"
    assert findings(al)["CLO-6"] == "Needs review"


def test_alignment_only_scan_and_manual_input(monkeypatch):
    docs = build_manual_documents("CLO-1: Explain routing. (C2)", "Q1 [CLO-1]: Define routing.")
    ok = pipeline.can_run(docs)
    assert ok["alignment"] and not ok["industry"]
    run = scan(monkeypatch, docs=docs)
    assert run["meta"]["analyses"] == ["alignment"] and run["competencies"] == []
    assert "CLO-Assessment Alignment Report" in alignment_markdown(run)


def test_improvement_and_student_pack(monkeypatch):
    run = scan(monkeypatch)
    res = clo.analyze(run["alignment"])
    imp = clo.generate_aligned_item(run["alignment"], res, "CLO-5", CFG)
    assert imp["bloom_level"] == 4 and imp["_meta"]["target_level"] == 4
    pack = pipeline.generate_student_pack(run, "C1", CFG)
    assert pack["steps"] and pack["_meta"]["competency_id"] == "C1"


def test_no_documents_errors():
    import pytest
    with pytest.raises(ValueError):
        pipeline.run_scan(build_manual_documents("", "", "only jobs"), CFG)


def test_malformed_model_output_is_tolerated(monkeypatch):
    junk = {"clos": [{"id": "CLO-1", "statement": None, "stated_level": "high"}, "not a dict?"][:1],
            "assessment_items": [{"id": "A1", "doc_id": "NOPE", "quote": 5, "marks": "five", "demand_level": "x"}],
            "mappings": [{"clo_id": "CLO-1", "item_id": "A1", "confidence": "very"}]}
    monkeypatch.setattr(clo, "call_json", lambda *a, **k: junk)
    al = clo.extract_alignment(build_manual_documents("CLO-1: Explain routing.", "Q1: Define routing."), CFG)
    res = clo.analyze(al)
    assert res["rows"][0]["finding"] in clo.FINDINGS
    assert al["items"][0]["marks"] is None and al["items"][0]["demand_level"] is None
    assert al["mappings"][0]["confidence"] == "low"


def test_call_json_retries_then_errors(monkeypatch):
    import pytest
    from skillleak import llm
    replies = iter(["not json at all", "[1, 2, 3]"])
    monkeypatch.setattr(llm, "_call_raw", lambda *a, **k: next(replies))
    with pytest.raises(llm.LLMError):
        llm.call_json(CFG, "s", "u")
    replies2 = iter(["oops", '{"ok": true}'])
    monkeypatch.setattr(llm, "_call_raw", lambda *a, **k: next(replies2))
    assert llm.call_json(CFG, "s", "u") == {"ok": True}


def test_two_rater_verification(monkeypatch):
    run = scan(monkeypatch)
    al = run["alignment"]
    assert al["verifier"]["status"] == "ok"
    assert al["verifier"]["requested"] == 6           # invented item A7 is never sent for rating
    res = clo.analyze(al)
    ag = res["agreement"]
    assert ag["n"] == 6 and abs(ag["exact"] - 4 / 6) < 1e-9 and ag["within_one"] == 1.0
    r5 = next(r for r in res["rows"] if r["id"] == "CLO-5")
    assert r5["finding"] == "Potential mismatch" and r5["severity"] == "high"   # C4 vs at most C2
    r1 = next(r for r in res["rows"] if r["id"] == "CLO-1")
    assert r1["severity"] == "low"                                              # one-level gap
    # make the raters disagree on the lab exam: one says C3, other C2 -> CLO-3 needs review
    for it in al["items"]:
        if it["id"] == "A3":
            it["demand_level_2"] = 2
    assert findings(al)["CLO-3"] == "Needs review"
    # faculty override resolves the disagreement
    al["overrides"]["item_level"]["A3"] = 3
    assert findings(al)["CLO-3"] == "No apparent mismatch detected"


def test_verifier_failure_is_non_blocking(monkeypatch):
    def boom(cfg, system, user, max_tokens=8000):
        if system == clo.prompts.BLOOM_RATER_SYSTEM:
            raise RuntimeError("quota exceeded")
        return fake_call_json(cfg, system, user, max_tokens)
    monkeypatch.setattr(pipeline, "call_json", fake_call_json)
    monkeypatch.setattr(clo, "call_json", boom)
    import skillleak.bloom as bloom
    monkeypatch.setattr(bloom, "call_json", boom)
    run = pipeline.run_scan(load_sample_documents("public"), CFG)
    assert run["alignment"]["verifier"]["status"].startswith("failed")
    assert findings(run["alignment"])["CLO-5"] == "Potential mismatch"


def test_quick_check_rules(monkeypatch):
    import skillleak.bloom as bloom
    import fake_llm
    monkeypatch.setattr(clo, "call_json", fake_call_json)
    monkeypatch.setattr(bloom, "call_json", fake_call_json)
    r = clo.quick_check("Diagnose faults", "List the steps", CFG, stated_level=4)
    assert (r["r1"], r["r2"], r["verdict"]) == (1, 2, "Potential mismatch")
    r = clo.quick_check("Diagnose faults", "List the steps", CFG)            # level inferred -> never a hard mismatch
    assert r["verdict"] == "Needs review" and r["target"] == 4
    monkeypatch.setitem(fake_llm.QUICK, "item_level", 4)                   # raters 4 and 2 straddle C3
    assert clo.quick_check("x", "y", CFG, stated_level=3)["verdict"] == "Needs review"
    monkeypatch.setitem(fake_llm.RATER, "Q", 3)
    assert clo.quick_check("x", "y", CFG, stated_level=3)["verdict"] == "No apparent mismatch detected"
    monkeypatch.setitem(fake_llm.QUICK, "relevant_to_clo", False)
    assert clo.quick_check("x", "y", CFG, stated_level=3)["verdict"] == "Needs review"


def test_aligned_mark_share_rule(monkeypatch):
    """One 2-mark C4 item + 98 marks of C1/C2 must not pass as aligned."""
    al = copy.deepcopy(scan(monkeypatch)["alignment"])
    al["items"].append({"id": "B1", "doc_id": "ASM1", "doc_category": "assessment", "label": "Tiny Q",
                        "item_type": "exam question", "quote": "List the steps of the bottom-up troubleshooting approach.",
                        "verified": "exact", "marks": 2, "demand_level": 4, "demand_level_2": 4, "demand_reason": ""})
    for it in al["items"]:
        if it["id"] == "A4":
            it["marks"] = 98
    al["mappings"].append({"clo_id": "CLO-5", "item_id": "B1", "basis": "explicit_tag", "confidence": "high", "reason": ""})
    row = next(r for r in clo.analyze(al)["rows"] if r["id"] == "CLO-5")
    assert row["finding"] == "Needs review" and abs(row["marks_share_at_level"] - 0.02) < 1e-9
    assert "Aligned-mark share is 2%" in row["explanation"]
    # an at-level item with unknown marks makes the share indecisive -> not forced to review on share alone
    for it in al["items"]:
        if it["id"] == "B1":
            it["marks"] = None
    row = next(r for r in clo.analyze(al)["rows"] if r["id"] == "CLO-5")
    assert row["finding"] == "No apparent mismatch detected" and "incomplete" in row["explanation"]
