"""The SkillLeak pipeline: documents -> competencies -> evidence map -> lab."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Callable

from . import prompts
from .extract import Document
from .llm import LLMConfig, call_json
from .scoring import verify_quote, is_verified, score_run
from .clo import extract_alignment

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _fmt_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"=== {d.doc_id} | type: {d.category} | file: {d.name} ===\n{d.text}" for d in docs
    )


def can_run(docs: list[Document]) -> dict:
    """Which analyses the supplied documents allow."""
    has = {c: any(d.category == c and d.text for d in docs)
           for c in ("syllabus", "practice", "assessment", "industry")}
    return {
        "alignment": has["syllabus"] and (has["assessment"] or has["practice"]),
        "industry": has["industry"] and (has["syllabus"] or has["practice"] or has["assessment"]),
        "has": has,
    }


def run_scan(docs: list[Document], cfg: LLMConfig,
             progress: Callable[[str, float], None] = lambda m, p: None,
             do_alignment: bool = True, do_industry: bool = True,
             verify: bool = True) -> dict:
    ok = can_run(docs)
    do_alignment = do_alignment and ok["alignment"]
    do_industry = do_industry and ok["industry"]
    if not (do_alignment or do_industry):
        raise ValueError("Not enough documents. CLO alignment needs a syllabus plus assessments or labs; "
                         "the industry gap analysis needs job descriptions plus at least one course document.")
    jds = [d for d in docs if d.category == "industry" and d.text]
    course = [d for d in docs if d.category != "industry" and d.text]
    by_id = {d.doc_id: d for d in docs}
    steps = (1 if do_alignment else 0) + (2 if do_industry else 0)
    step = 0

    alignment = None
    if do_alignment:
        step += 1
        progress(f"Step {step}/{steps} - AI extracts CLOs and assessment items, rates cognitive demand "
                 "and maps items to CLOs" + (" (then a second, independent AI rater re-rates every item)"
                                             if verify else "") + "...", 0.05)
        alignment = extract_alignment(docs, cfg, verify=verify)

    competencies, ind = [], {}
    if do_industry:
        step += 1
        progress(f"Step {step}/{steps} - AI extracts employer-required competencies from job descriptions...",
                 0.35 if do_alignment else 0.1)
        ind = call_json(cfg, prompts.INDUSTRY_SYSTEM,
                        prompts.INDUSTRY_USER.format(docs=_fmt_docs(jds)), max_tokens=10000)
        for i, c in enumerate(ind.get("competencies", []) or [], start=1):
            cid = f"C{i}"
            evid = []
            for e in c.get("evidence", []) or []:
                d = by_id.get(e.get("doc_id"))
                if d is None or d.category != "industry":
                    continue
                evid.append({"doc_id": d.doc_id, "quote": e.get("quote", ""),
                             "verified": verify_quote(e.get("quote", ""), d.text)})
            competencies.append({"id": cid, "orig_id": c.get("id", cid), "name": c.get("name", cid),
                                 "description": c.get("description", ""),
                                 "industry_evidence": evid, "course_evidence": []})

        step += 1
        progress(f"Step {step}/{steps} - AI searches course documents for evidence of each competency "
                 "and rates Bloom level...", 0.65 if do_alignment else 0.45)
        comp_list = "\n".join(f"- {c['id']}: {c['name']} - {c['description']}" for c in competencies)
        mapped = call_json(cfg, prompts.COURSE_SYSTEM,
                           prompts.COURSE_USER.format(competencies=comp_list, docs=_fmt_docs(course)),
                           max_tokens=16000)
        comp_by_id = {c["id"]: c for c in competencies}
        for m in mapped.get("mappings", []) or []:
            comp = comp_by_id.get(m.get("competency_id"))
            if not comp:
                continue
            for e in m.get("evidence", []) or []:
                d = by_id.get(e.get("doc_id"))
                if d is None or d.category == "industry":
                    continue
                try:
                    bloom = max(1, min(6, int(e.get("bloom_level", 1))))
                except (TypeError, ValueError):
                    bloom = 1
                comp["course_evidence"].append({
                    "doc_id": d.doc_id, "category": d.category, "quote": e.get("quote", ""),
                    "bloom_level": bloom, "reason": e.get("reason", ""),
                    "verified": verify_quote(e.get("quote", ""), d.text),
                })

    progress("Verifying every quote against the source documents and scoring...", 0.9)
    all_ev = [e for c in competencies for e in c["industry_evidence"] + c["course_evidence"]]
    if alignment:
        all_ev += alignment["items"] + alignment["clos"]
    run = {
        "meta": {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "model": cfg.label(),
            "documents_hash": hashlib.sha256(
                "".join(d.text for d in docs).encode()).hexdigest()[:12],
            "quotes_total": len(all_ev),
            "quotes_rejected": sum(1 for e in all_ev if not is_verified(e)),
            "analyses": [a for a, on in (("alignment", do_alignment), ("industry", do_industry)) if on],
        },
        "job_family": ind.get("job_family", "Entry-level role") if do_industry else "",
        "documents": [d.to_dict() for d in docs],
        "competencies": competencies,
        "alignment": alignment,
        "labs": {},
        "student_packs": {},
    }
    progress("Done.", 1.0)
    return run


def _quotes(comp: dict, run: dict, categories: set[str]) -> str:
    names = {d["doc_id"]: d["name"] for d in run["documents"]}
    ev = [e for e in comp["industry_evidence"] if "industry" in categories] + \
         [e for e in comp["course_evidence"] if e["category"] in categories]
    ev = [e for e in ev if is_verified(e)]
    if not ev:
        return "  (none found)"
    out = []
    for e in ev:
        bl = f" [Bloom {e['bloom_level']}]" if "bloom_level" in e else ""
        out.append(f"  - {e['doc_id']} ({names.get(e['doc_id'], '')}){bl}: \"{e['quote']}\"")
    return "\n".join(out)


def generate_lab(run: dict, comp_id: str, cfg: LLMConfig, feedback: str = "") -> dict:
    comp = next(c for c in run["competencies"] if c["id"] == comp_id)
    row = next((r for r in score_run(run) if r["id"] == comp_id), None)
    syl = [d for d in run["documents"] if d["category"] == "syllabus"]
    course_name = syl[0]["text"].splitlines()[0:6] if syl else []
    course_name = " ".join(l for l in course_name if l.strip().startswith("#")).strip("# ") \
        or (syl[0]["name"] if syl else "the programme")
    practice_docs = [d for d in run["documents"] if d["category"] == "practice"]
    existing = "\n".join(
        l.strip() for d in practice_docs for l in d["text"].splitlines()
        if l.strip().startswith("#") or l.lower().startswith("objective")
    )[:2500] or "(no practice documents provided)"
    fb = (f"\nTeacher feedback on the previous draft - revise the lab to address it:\n{feedback}\n"
          if feedback else "")

    user = prompts.LAB_USER.format(
        course=course_name, job_family=run.get("job_family", ""),
        competency=f"{comp['name']} - {comp['description']}",
        industry_quotes=_quotes(comp, run, {"industry"}),
        course_evidence=_quotes(comp, run, {"syllabus", "practice", "assessment"}),
        diagnosis=row["diagnosis"] if row else "",
        existing_practice=existing, feedback=fb,
    )
    lab = call_json(cfg, prompts.LAB_SYSTEM, user)
    lab["_meta"] = {"competency_id": comp_id, "model": cfg.label(),
                    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "revised_with_feedback": bool(feedback)}
    lab["review"] = {"status": "Pending review", "note": ""}
    return lab


def generate_student_pack(run: dict, comp_id: str, cfg: LLMConfig) -> dict:
    comp = next(c for c in run["competencies"] if c["id"] == comp_id)
    out = call_json(cfg, prompts.STUDENT_SYSTEM, prompts.STUDENT_USER.format(
        skill=comp["name"], description=comp.get("description", ""),
        quotes=_quotes(comp, run, {"industry"}),
        course=_quotes(comp, run, {"syllabus", "practice", "assessment"})))
    out["_meta"] = {"competency_id": comp_id, "model": cfg.label(),
                    "created_at": dt.datetime.now().isoformat(timespec="seconds")}
    return out


# ---------------------------------------------------------------------------
# Saving and replaying runs (backup for live demos)
# ---------------------------------------------------------------------------
def save_run(run: dict, name: str | None = None) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    name = name or f"scan_{run['meta']['created_at'].replace(':', '-')}.json"
    path = RUNS_DIR / name
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def list_saved_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
