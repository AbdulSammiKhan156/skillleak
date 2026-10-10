"""Deterministic, transparent scoring. No LLM is involved in any number shown.

Gap priority = D x (3 - (T + P + A))
  D  industry demand weight 1-5 = ceil(5 x share of job descriptions with a verified quote)
  T  1 if the syllabus has verified evidence of the competency
  P  1 if labs/assignments have verified evidence at Bloom level >= Apply (3)
  A  1 if exams/rubrics have verified evidence at Bloom level >= Apply (3)
Only quotes that were found in the source documents count.
"""
from __future__ import annotations

import math
import re
import unicodedata

from .prompts import BLOOM

APPLY_LEVEL = 3


# ---------------------------------------------------------------------------
# Quote verification (anti-hallucination guard)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _ngrams(words: list[str], n: int = 2) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def verify_quote(quote: str, source_text: str) -> str:
    """Return 'exact', 'approx' or 'not_found'."""
    q, t = _norm(quote), _norm(source_text)
    if not q:
        return "not_found"
    if q in t:
        return "exact"
    qw = q.split()
    if len(qw) < 4:
        return "not_found"
    qg = _ngrams(qw)
    tg = _ngrams(t.split())
    if qg and len(qg & tg) / len(qg) >= 0.75:
        return "approx"
    return "not_found"


def is_verified(ev: dict) -> bool:
    return ev.get("verified") in ("exact", "approx")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _max_bloom(evidence: list[dict], category: str) -> int:
    levels = [int(e.get("bloom_level") or 0) for e in evidence
              if e.get("category") == category and is_verified(e)]
    return max(levels, default=0)


def demand_weight(n_jd: int, total_jd: int) -> int:
    if n_jd <= 0 or total_jd <= 0:
        return 0
    return max(1, min(5, math.ceil(5 * n_jd / total_jd)))


def score_competency(comp: dict, total_jd: int,
                     available: frozenset = frozenset({"syllabus", "practice", "assessment"})) -> dict:
    jd_ids = {e["doc_id"] for e in comp.get("industry_evidence", []) if is_verified(e)}
    D = demand_weight(len(jd_ids), total_jd)

    ev = comp.get("course_evidence", [])
    b_syl = _max_bloom(ev, "syllabus")
    b_prac = _max_bloom(ev, "practice")
    b_asm = _max_bloom(ev, "assessment")

    T = 1 if b_syl > 0 else 0
    P = 1 if b_prac >= APPLY_LEVEL else 0
    A = 1 if b_asm >= APPLY_LEVEL else 0
    evidence_score = T + P + A
    priority = D * (3 - evidence_score)
    missing = [c for c in ("syllabus", "practice", "assessment") if c not in available]

    if evidence_score == 3:
        status = "Covered"
    elif missing:
        status = "Cannot fully evaluate"  # some document types were not supplied
    elif evidence_score == 0:
        status = "Industry blind spot"
    elif T == 1:
        status = "Phantom competence risk"
    else:
        status = "Practised but not in syllabus"

    return {
        "id": comp["id"], "name": comp["name"],
        "jd_count": len(jd_ids), "D": D, "T": T, "P": P, "A": A,
        "bloom_syllabus": b_syl, "bloom_practice": b_prac, "bloom_assessment": b_asm,
        "evidence_score": evidence_score, "priority": priority, "status": status,
        "missing_categories": missing,
        "diagnosis": diagnose(T, b_prac, b_asm, missing),
    }


def _level(b: int) -> str:
    return f"{BLOOM[b]} ({b})" if b in BLOOM else "none"


def diagnose(T: int, b_prac: int, b_asm: int, missing: list | None = None) -> str:
    missing = missing or []
    parts = []
    if "syllabus" in missing:
        parts.append("No syllabus was supplied, so teaching coverage cannot be evaluated.")
    elif not T:
        parts.append("Not identified in the supplied syllabus or learning outcomes.")
    if "practice" in missing:
        parts.append("No lab manual or assignments were supplied, so practice cannot be evaluated.")
    elif b_prac == 0:
        parts.append("No lab, assignment or project in the supplied material asks students to practise it.")
    elif b_prac < APPLY_LEVEL:
        parts.append(f"Practice found only at {_level(b_prac)} level - students are not required to perform it.")
    if "assessment" in missing:
        parts.append("No exams or rubrics were supplied, so assessment cannot be evaluated.")
    elif b_asm == 0:
        parts.append("No exam, quiz or rubric in the supplied material assesses it.")
    elif b_asm < APPLY_LEVEL:
        parts.append(f"Assessed only at {_level(b_asm)} level - recall, not demonstration.")
    return " ".join(parts) or "Taught, practised and assessed at Apply level or above in the supplied material."


def score_run(run: dict) -> list[dict]:
    docs = run["documents"]
    total_jd = sum(1 for d in docs if d["category"] == "industry")
    available = frozenset(d["category"] for d in docs if d["category"] != "industry" and d.get("text"))
    rows = [score_competency(c, total_jd, available) for c in run.get("competencies", [])]
    rows = [r for r in rows if r["D"] > 0]  # drop competencies with no verified employer quote
    rows.sort(key=lambda r: (-r["priority"], -r["D"], r["name"]))
    return rows


def coverage_index(rows: list[dict]) -> float:
    """Demand-weighted share of required evidence present (prototype indicator)."""
    total = sum(r["D"] * 3 for r in rows)
    have = sum(r["D"] * r["evidence_score"] for r in rows)
    return 100.0 * have / total if total else 0.0
