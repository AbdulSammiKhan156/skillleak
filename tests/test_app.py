"""Smoke-test the Streamlit UI with a fixture scan (no API key needed)."""
import copy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
FIX = json.loads((ROOT / "tests/fixtures/fixture_run.json").read_text())


def _app(run=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    if run is not None:
        at.session_state["run"] = copy.deepcopy(run)
        at.session_state["replay"] = True
    return at.run()


def test_app_renders_without_run():
    at = _app()
    assert not at.exception
    assert any("Run SkillLeak analysis" in b.label for b in at.button)
    # analysis button disabled without an API key
    btn = next(b for b in at.button if "Run SkillLeak analysis" in b.label)
    assert btn.disabled


def test_app_renders_with_run():
    at = _app(FIX)
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    subs = " ".join(s.value for s in at.subheader)
    assert "Skill Leakage Map" in subs and "CLO → assessment alignment" in subs
    assert "Potential mismatch" in md and "Phantom competence risk" in md
    assert "Fault Hunt" in md and "Replaying a saved scan" in md
    assert "not official PEC" in md


def test_app_alignment_only_run():
    run = copy.deepcopy(FIX)
    run["competencies"], run["labs"], run["student_packs"] = [], {}, {}
    at = _app(run)
    assert not at.exception
    assert any("no industry skill-gap analysis" in i.value for i in at.info)


def test_app_old_run_without_alignment():
    run = copy.deepcopy(FIX)
    run.pop("alignment")
    run.pop("student_packs")
    at = _app(run)
    assert not at.exception
    assert any("no CLO" in i.value for i in at.info)


def test_manual_input_mode():
    at = _app()
    at.radio[0].set_value("Enter CLOs & questions manually").run()
    assert not at.exception
    at.text_area[0].input("CLO-1: Explain routing. (C2)")
    at.text_area[1].input("Q1 [CLO-1] (5): Define routing.").run()
    assert not at.exception
    assert any("documents loaded" in s.value for s in at.subheader)


def test_corrections_and_reset_buttons():
    at = _app(FIX)
    btn = next(b for b in at.button if b.label == "Apply corrections and re-check")
    btn.click().run()
    assert not at.exception
    assert at.session_state["run"]["alignment"]["overrides"] == {"clo_level": {}, "item_level": {}, "exclude": []}
    next(b for b in at.button if b.label == "Reset corrections").click().run()
    assert not at.exception


def test_clo_pick_defaults_to_largest_mismatch():
    at = _app(FIX)
    sel = next(s for s in at.selectbox if s.label == "CLO")
    assert sel.value == "CLO-5"   # C4 CLO assessed only at C1
    md = " ".join(m.value for m in at.markdown)
    assert "Proposed" in md        # saved aligned-item draft is shown


def test_attainment_and_validation_tabs():
    at = _app(FIX)
    assert not at.exception
    warn = " ".join(w.value for w in at.warning)
    assert "CLO-5" in warn and "measure it only up to" in warn          # inflation warning on the sample marks
    md = " ".join(m.value for m in at.markdown)
    assert "Potential attainment-validity concern" in md and "PLO assessment-evidence" in " ".join(s.value for s in at.subheader)
    assert "Two independent AI ratings" in " ".join(c.value for c in at.caption)
    assert any("Keyword baseline" in str(d.value) for d in at.dataframe)  # saved benchmark results shown


def test_attainment_without_scan():
    at = _app()
    assert not at.exception
    assert any("Run (or load) a scan" in i.value for i in at.info)


def test_quick_check_in_app(monkeypatch):
    import sys as _s
    _s.path.insert(0, str(ROOT / "tests"))
    from fake_llm import fake_call_json
    import skillleak.clo as clo
    import skillleak.bloom as bloom
    monkeypatch.setattr(clo, "call_json", fake_call_json)
    monkeypatch.setattr(bloom, "call_json", fake_call_json)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    at = _app()
    next(b for b in at.button if b.label == "Check alignment").click().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Potential mismatch" in md
    next(b for b in at.button if "Suggest an aligned version" in b.label).click().run()
    assert not at.exception
    assert "Packet Tracer topology" in " ".join(m.value for m in at.markdown)
