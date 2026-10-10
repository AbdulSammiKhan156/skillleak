"""CLO-assessment alignment check (Outcome-Based Education support).

AI step  : extract CLOs + assessment items, rate each item's cognitive demand, map items to CLOs.
Code step: verify every quote against the source, apply faculty corrections, and decide the
           finding for each CLO with transparent rules (no AI-generated scores).

Findings are decision-support signals for faculty review. They are NOT official accreditation
judgements and do not certify PEC (or any other) compliance.
"""
from __future__ import annotations

import datetime as dt

from . import prompts
from .llm import LLMConfig, call_json
from .prompts import BLOOM
from .scoring import verify_quote, is_verified, _norm
from .bloom import rate_items_blind

FINDINGS = ["Potential mismatch", "Needs review", "Insufficient evidence", "No apparent mismatch detected"]
CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


def _lvl(x) -> int | None:
    try:
        v = int(x)
        return v if 1 <= v <= 6 else None
    except (TypeError, ValueError):
        return None


def level_name(v: int | None) -> str:
    return f"C{v} {BLOOM[v]}" if v in BLOOM else "not stated"


# ---------------------------------------------------------------------------
# AI extraction
# ---------------------------------------------------------------------------
def extract_alignment(docs: list, cfg: LLMConfig, verify: bool = True) -> dict:
    """docs: list of Document objects (syllabus / practice / assessment).

    verify=True adds an independent second rating of every verified item from a blind rater
    (it does not see the CLOs), so findings do not rest on a single model judgement."""
    course_docs = [d for d in docs if d.category in ("syllabus", "practice", "assessment") and d.text]
    by_id = {d.doc_id: d for d in course_docs}
    text = "\n\n".join(f"=== {d.doc_id} | type: {d.category} | file: {d.name} ===\n{d.text}"
                       for d in course_docs)
    raw = call_json(cfg, prompts.CLO_SYSTEM, prompts.CLO_USER.format(docs=text), max_tokens=16000)

    clos = []
    seen = set()
    for c in raw.get("clos", []) or []:
        cid = str(c.get("id") or f"CLO-{len(clos) + 1}").strip()
        if cid in seen:
            continue
        seen.add(cid)
        d = by_id.get(c.get("doc_id"))
        stmt = str(c.get("statement") or "")
        verified = "not_found"
        if d is not None:
            verified = verify_quote(stmt, d.text)
        if verified == "not_found":  # statement may come from another doc
            for dd in course_docs:
                v = verify_quote(stmt, dd.text)
                if v != "not_found":
                    d, verified = dd, v
                    break
        clos.append({
            "id": cid, "doc_id": d.doc_id if d else str(c.get("doc_id", "")),
            "statement": stmt, "verified": verified,
            "stated_level": _lvl(c.get("stated_level")),
            "stated_level_text": str(c.get("stated_level_text") or ""),
            "inferred_level": _lvl(c.get("inferred_level")),
            "inference_reason": str(c.get("inference_reason") or ""),
            "plo": str(c.get("plo") or ""),
        })

    items = []
    item_ids = set()
    for it in raw.get("assessment_items", []) or []:
        iid = str(it.get("id") or f"A{len(items) + 1}").strip()
        if iid in item_ids:
            continue
        item_ids.add(iid)
        d = by_id.get(it.get("doc_id"))
        q = str(it.get("quote") or "")
        marks = it.get("marks")
        try:
            marks = float(marks) if marks not in (None, "") else None
        except (TypeError, ValueError):
            marks = None
        items.append({
            "id": iid, "doc_id": d.doc_id if d else str(it.get("doc_id", "")),
            "doc_category": d.category if d else "",
            "label": str(it.get("label") or iid), "item_type": str(it.get("item_type") or ""),
            "quote": q, "verified": verify_quote(q, d.text) if d else "not_found",
            "marks": marks, "demand_level": _lvl(it.get("demand_level")),
            "demand_reason": str(it.get("demand_reason") or ""),
        })

    clo_ids = {c["id"] for c in clos}
    mappings = []
    for m in raw.get("mappings", []) or []:
        if m.get("clo_id") in clo_ids and m.get("item_id") in item_ids:
            conf = str(m.get("confidence") or "low").lower()
            mappings.append({
                "clo_id": m["clo_id"], "item_id": m["item_id"],
                "basis": "explicit_tag" if m.get("basis") == "explicit_tag" else "content_match",
                "confidence": conf if conf in CONF_ORDER else "low",
                "reason": str(m.get("reason") or ""),
            })

    verifier = {"status": "off"}
    if verify:
        to_rate = [{"id": it["id"], "text": f"[{it['item_type'] or 'item'}] {it['quote']}"}
                   for it in items if is_verified(it)]
        try:
            ratings = rate_items_blind(to_rate, cfg) if to_rate else {}
            for it in items:
                r = ratings.get(it["id"])
                if r:
                    it["demand_level_2"] = r["level"]
                    it["demand_reason_2"] = r["reason"]
            verifier = {"status": "ok", "rated": len(ratings), "requested": len(to_rate)}
        except Exception as exc:  # second opinion is best-effort; never block the analysis
            verifier = {"status": f"failed: {str(exc)[:200]}"}

    return {
        "verifier": verifier,
        "course": str(raw.get("course") or ""),
        "clos": clos, "items": items, "mappings": mappings,
        "missing_information": [str(x) for x in raw.get("missing_information", []) or []],
        "overrides": {"clo_level": {}, "item_level": {}, "exclude": []},
        "improvements": {},
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": cfg.label(),
    }


# ---------------------------------------------------------------------------
# Deterministic analysis (re-run instantly after faculty corrections)
# ---------------------------------------------------------------------------
def _key(m: dict) -> str:
    return f"{m['clo_id']}|{m['item_id']}"


def target_level(clo: dict, overrides: dict) -> tuple[int | None, str]:
    o = _lvl(overrides.get("clo_level", {}).get(clo["id"]))
    if o:
        return o, "faculty"
    if clo.get("stated_level"):
        return clo["stated_level"], "stated"
    if clo.get("inferred_level"):
        return clo["inferred_level"], "inferred"
    return None, "missing"


def item_level(item: dict, overrides: dict) -> tuple[int | None, int | None, str]:
    """Return (low, high, source). Two AI ratings give a range; faculty override fixes it."""
    o = _lvl(overrides.get("item_level", {}).get(item["id"]))
    if o:
        return o, o, "faculty"
    r1, r2 = _lvl(item.get("demand_level")), _lvl(item.get("demand_level_2"))
    vals = [v for v in (r1, r2) if v]
    if not vals:
        return None, None, "ai"
    return min(vals), max(vals), ("ai x2" if len(vals) == 2 else "ai")


def level_range(lo: int | None, hi: int | None) -> str:
    if lo is None:
        return "not rated"
    return level_name(lo) if lo == hi else f"C{lo}-C{hi} ({BLOOM[lo]}-{BLOOM[hi]})"


def rater_agreement(al: dict) -> dict:
    """Agreement between the two independent AI ratings on verified items."""
    pairs = [(i.get("demand_level"), i.get("demand_level_2")) for i in al.get("items", [])
             if is_verified(i) and _lvl(i.get("demand_level")) and _lvl(i.get("demand_level_2"))]
    n = len(pairs)
    if not n:
        return {"n": 0, "exact": None, "within_one": None}
    exact = sum(1 for a, b in pairs if int(a) == int(b))
    near = sum(1 for a, b in pairs if abs(int(a) - int(b)) <= 1)
    return {"n": n, "exact": exact / n, "within_one": near / n}


def analyze(al: dict) -> dict:
    ov = al.get("overrides", {"clo_level": {}, "item_level": {}, "exclude": []})
    excluded = set(ov.get("exclude", []))
    items = {i["id"]: i for i in al["items"]}
    rows, mapping_rows = [], []

    for clo in al["clos"]:
        tgt, tgt_src = target_level(clo, ov)
        ms = [m for m in al["mappings"] if m["clo_id"] == clo["id"]]
        usable = []
        for m in ms:
            it = items.get(m["item_id"])
            lo, hi, lvl_src = item_level(it, ov) if it else (None, None, "ai")
            ok_quote = bool(it) and is_verified(it)
            include = ok_quote and _key(m) not in excluded and hi is not None
            mapping_rows.append({
                "key": _key(m), "clo_id": clo["id"], "item_id": m["item_id"],
                "item_label": it["label"] if it else m["item_id"],
                "quote": it["quote"] if it else "", "doc_id": it["doc_id"] if it else "",
                "marks": it["marks"] if it else None,
                "demand": hi, "demand_lo": lo, "demand_hi": hi, "demand_src": lvl_src,
                "demand_range": level_range(lo, hi),
                "raters_disagree": lo is not None and lo != hi and lvl_src != "faculty",
                "demand_reason": it["demand_reason"] if it else "",
                "demand_reason_2": it.get("demand_reason_2", "") if it else "",
                "basis": m["basis"], "confidence": m["confidence"], "reason": m["reason"],
                "quote_verified": ok_quote, "included": include,
                "meets_level": (tgt is not None and lo is not None and lo >= tgt),
                "maybe_meets_level": (tgt is not None and hi is not None and hi >= tgt),
            })
            if include:
                usable.append((m, it, lo, hi))

        at = [u for u in usable if tgt is not None and u[2] >= tgt]           # both raters agree
        possible = [u for u in usable if tgt is not None and u[3] >= tgt]     # at least one rater
        below = [u for u in usable if tgt is not None and u[3] < tgt]
        max_demand = max((u[3] for u in usable), default=None)
        # Aligned-mark share = marks on mapped items meeting the CLO level / all mapped marks
        # (over items whose marks are stated; coverage is reported so a partial picture is visible)
        marked = [u for u in usable if u[1]["marks"] is not None]
        share = None
        if marked:
            tot = sum(u[1]["marks"] for u in marked)
            share = (sum(u[1]["marks"] for u in marked if tgt is not None and u[2] >= tgt) / tot) if tot else None
        marks_coverage = f"{len(marked)}/{len(usable)}" if usable else "0/0"
        # The share rule can only be trusted if no unmarked item reaches the level (an unmarked item below the
        # level could only lower the true share further; an unmarked item at the level makes it unknowable).
        unmarked_at_level = any(u[1]["marks"] is None and tgt is not None and u[2] >= tgt for u in usable)
        share_decisive = share is not None and not unmarked_at_level

        severity = ""
        reasons = []
        if tgt is None:
            finding = "Needs review"
            reasons.append("No taxonomy level stated or inferable for this CLO; faculty should set the intended level.")
        elif not usable:
            finding = "Insufficient evidence"
            reasons.append("No verified assessment item in the supplied documents could be linked to this CLO.")
        elif not possible:
            gap = tgt - (max_demand or 0)
            if tgt_src == "inferred":
                finding = "Needs review"
                reasons.append(f"All mapped items demand at most {level_name(max_demand)}, below the inferred level "
                               f"{level_name(tgt)}. The CLO level was not stated, so this is not flagged as a mismatch.")
            else:
                finding = "Potential mismatch"
                severity = "high" if gap >= 2 else "low"
                reasons.append(f"Every mapped item demands at most {level_name(max_demand)} - {gap} level(s) below the "
                               f"{'faculty-set' if tgt_src == 'faculty' else 'stated'} CLO level {level_name(tgt)}"
                               + (" (both AI raters agree)." if any(u[1].get("demand_level_2") for u in usable) else "."))
                if gap == 1:
                    reasons.append("A one-level gap is a common point of disagreement between raters - treat as low severity.")
        elif not at:
            finding = "Needs review"
            reasons.append(f"The two AI raters disagree on whether any mapped item reaches {level_name(tgt)}; "
                           "faculty should decide.")
        elif all(CONF_ORDER[u[0]["confidence"]] <= 1 and u[0]["basis"] != "explicit_tag" for u in at):
            finding = "Needs review"
            reasons.append("Only low-confidence content matches reach the CLO level; faculty should confirm the mapping.")
        elif share_decisive and share < 0.5:
            finding = "Needs review"
            reasons.append(f"Aligned-mark share is {share:.0%}: items at {level_name(tgt)} or above carry under half of "
                           "the marks linked to this CLO, so most of its marks are earned below its level.")
        else:
            finding = "No apparent mismatch detected"
            reasons.append(f"{len(at)} mapped item(s) demand {level_name(tgt)} or higher.")
        if usable and len(marked) < len(usable):
            reasons.append(f"Marks are stated for only {marks_coverage} mapped item(s); the aligned-mark share uses those"
                           + (" and is incomplete because an item at the CLO level has no stated marks."
                              if unmarked_at_level else "."))
        if below and finding == "No apparent mismatch detected":
            reasons.append(f"{len(below)} other mapped item(s) sit below the CLO level - acceptable if they assess "
                           "prerequisite knowledge.")
        if tgt_src == "inferred":
            reasons.append("CLO level was inferred by the AI (not stated in the document).")
        if clo.get("verified") == "not_found":
            reasons.append("CLO statement could not be matched to the source text - check extraction.")

        rows.append({
            "id": clo["id"], "statement": clo["statement"], "plo": clo.get("plo", ""),
            "target": tgt, "target_src": tgt_src, "max_demand": max_demand,
            "n_items": len(usable), "n_at_level": len(at), "marks_share_at_level": share,
            "marks_coverage": marks_coverage,
            "finding": finding, "severity": severity, "explanation": " ".join(reasons),
            "action": recommended_action(finding, tgt),
            "clo_verified": clo.get("verified"),
        })

    mapped_items = {m["item_id"] for m in al["mappings"]}
    unmapped = [i for i in al["items"] if i["id"] not in mapped_items and is_verified(i)]
    rejected_items = [i for i in al["items"] if not is_verified(i)]
    n = len(rows)
    ok = sum(1 for r in rows if r["finding"] == "No apparent mismatch detected")
    return {
        "rows": rows, "mapping_rows": mapping_rows, "unmapped_items": unmapped,
        "rejected_items": rejected_items,
        "agreement": rater_agreement(al),
        "summary": {
            "clos": n,
            "with_evidence": sum(1 for r in rows if r["n_items"] > 0),
            "mismatch": sum(1 for r in rows if r["finding"] == "Potential mismatch"),
            "insufficient": sum(1 for r in rows if r["finding"] == "Insufficient evidence"),
            "review": sum(1 for r in rows if r["finding"] == "Needs review"),
            "aligned": ok,
            "alignment_indicator": (100.0 * ok / n) if n else 0.0,
        },
    }


def recommended_action(finding: str, tgt: int | None) -> str:
    if finding == "Potential mismatch":
        return (f"Add or revise an assessment item that requires students to perform the CLO at "
                f"{level_name(tgt)} or higher (use 'Generate aligned assessment item').")
    if finding == "Insufficient evidence":
        return "Check whether this CLO is assessed in documents not supplied; otherwise add an assessment item for it."
    if finding == "Needs review":
        return "Faculty to confirm the CLO level and the item mappings (edit them in the correction panel)."
    return "No action suggested from the supplied documents."


def clo_for_text(al: dict | None, quote: str) -> list[str]:
    """CLO ids whose statement overlaps a syllabus quote (links skill gaps to CLOs)."""
    if not al or not quote:
        return []
    q = _norm(quote)
    out = []
    for c in al.get("clos", []):
        s = _norm(c["statement"])
        if s and (s in q or q in s or verify_quote(quote, c["statement"]) != "not_found"):
            out.append(c["id"])
    return out


# ---------------------------------------------------------------------------
# AI: improved assessment item for a flagged CLO
# ---------------------------------------------------------------------------
def generate_aligned_item(al: dict, analysis: dict, clo_id: str, cfg: LLMConfig) -> dict:
    row = next(r for r in analysis["rows"] if r["id"] == clo_id)
    maps = [m for m in analysis["mapping_rows"] if m["clo_id"] == clo_id and m["included"]]
    items_txt = "\n".join(f"- {m['item_label']} (demand {level_name(m['demand'])}, marks {m['marks']}): \"{m['quote']}\""
                          for m in maps) or "- (no assessment item currently mapped)"
    out = call_json(cfg, prompts.ALIGN_Q_SYSTEM, prompts.ALIGN_Q_USER.format(
        clo_id=clo_id, statement=row["statement"], level=level_name(row["target"]),
        items=items_txt, finding=f"{row['finding']}: {row['explanation']}"))
    out["_meta"] = {"model": cfg.label(), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "target_level": row["target"]}
    return out


# ---------------------------------------------------------------------------
# Quick check (one CLO + one question) - same two-rater rule as the full scan
# ---------------------------------------------------------------------------
def quick_check(clo_text: str, question: str, cfg: LLMConfig, stated_level: int | None = None) -> dict:
    from .bloom import rate_items_blind
    a = call_json(cfg, prompts.QUICK_SYSTEM, prompts.QUICK_USER.format(
        clo=clo_text.strip(), level=f"C{stated_level}" if stated_level else "not stated - infer it",
        item=question.strip()), max_tokens=2000)
    b = rate_items_blind([{"id": "Q", "text": question.strip()}], cfg).get("Q", {})
    r1, r2 = _lvl(a.get("item_level")), _lvl(b.get("level"))
    tgt = stated_level or _lvl(a.get("clo_level"))
    vals = [v for v in (r1, r2) if v]
    lo, hi = (min(vals), max(vals)) if vals else (None, None)
    relevant = a.get("relevant_to_clo") is not False
    if tgt is None or lo is None:
        verdict = "Needs review"
        why = "The model did not return usable levels."
    elif not relevant:
        verdict = "Needs review"
        why = "The item may not assess what this CLO describes: " + str(a.get("relevance_reason") or "")
    elif hi < tgt:
        verdict = "Potential mismatch" if stated_level else "Needs review"
        why = (f"Both raters place the item at most at {level_name(hi)}, below {level_name(tgt)} "
               f"({tgt - hi} level gap)." + ("" if stated_level else " CLO level was inferred, not stated."))
    elif lo < tgt:
        verdict = "Needs review"
        why = f"The raters disagree ({level_range(lo, hi)}) on whether the item reaches {level_name(tgt)}."
    else:
        verdict = "No apparent mismatch detected"
        why = f"Both raters place the item at {level_range(lo, hi)}, meeting {level_name(tgt)}."
    return {"target": tgt, "target_src": "stated" if stated_level else "inferred",
            "clo_level_reason": str(a.get("clo_level_reason") or ""),
            "r1": r1, "r1_reason": str(a.get("item_level_reason") or ""),
            "r2": r2, "r2_reason": str(b.get("reason") or ""),
            "relevant": relevant, "relevance_reason": str(a.get("relevance_reason") or ""),
            "verdict": verdict, "why": why, "lo": lo, "hi": hi}


def quick_rewrite(clo_text: str, question: str, target: int, cfg: LLMConfig) -> dict:
    return call_json(cfg, prompts.ALIGN_Q_SYSTEM, prompts.ALIGN_Q_USER.format(
        clo_id="CLO", statement=clo_text.strip(), level=level_name(target),
        items=f"- \"{question.strip()}\"", finding="Quick check"))
