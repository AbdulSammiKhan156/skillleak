"""CLO attainment from a marks sheet, cross-checked against assessment alignment.

Deterministic - no AI. The cross-check flags a *potential attainment-validity concern*: a CLO
can be "attained" under the configured thresholds while every mapped item in the supplied
documents demands less than the CLO's stated cognitive level. This is a prompt to check the
evidence - it does not prove that attainment is inflated (the CLO may be assessed elsewhere).

Marks sheet format (CSV, wide):
    Student, Mid Q1, Mid Q2, ...
    CLO,     CLO-1,  CLO-2,  ...      <- CLO tag per question (blank = not mapped; "CLO-1/CLO-2" allowed)
    MAX,     5,      10,     ...      <- maximum marks per question
    S01,     4,      7,      ...
Thresholds differ between institutions, so both are parameters.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from .clo import level_name


def _norm_clo(x: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(x).lower())


def parse_marks_csv(data: bytes | str) -> tuple[dict, list[str]]:
    """Return ({'questions': [...], 'clo': {q: [clo,...]}, 'max': {q: float}, 'students': DataFrame}, warnings)."""
    warnings: list[str] = []
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError("The marks sheet needs a student column and at least one question column.")
    first = df.columns[0]
    questions = [c for c in df.columns[1:] if str(c).strip()]
    key = df[first].str.strip().str.upper()
    clo_rows, max_rows = df[key == "CLO"], df[key == "MAX"]
    if clo_rows.empty or max_rows.empty:
        raise ValueError("The marks sheet must contain a 'CLO' row (CLO tag per question) and a 'MAX' row "
                         "(maximum marks per question). Download the template for the format.")
    clo_map, max_map = {}, {}
    for q in questions:
        tags = [t.strip() for t in re.split(r"[/;,|]", str(clo_rows.iloc[0][q])) if t.strip()]
        clo_map[q] = tags
        try:
            max_map[q] = float(max_rows.iloc[0][q])
        except ValueError:
            max_map[q] = 0.0
            warnings.append(f"Question '{q}' has no valid maximum mark; it is ignored.")
    students = df[(key != "CLO") & (key != "MAX")].copy()
    if students.empty:
        raise ValueError("No student rows found below the CLO and MAX rows.")
    blanks = 0
    for q in questions:
        col = pd.to_numeric(students[q].str.strip().replace("", "0"), errors="coerce")
        blanks += int(col.isna().sum())
        col = col.fillna(0.0)
        over = col > max_map[q]
        if over.any() and max_map[q] > 0:
            warnings.append(f"{int(over.sum())} mark(s) in '{q}' exceed the maximum and were capped.")
            col = col.clip(upper=max_map[q])
        students[q] = col
    if blanks:
        warnings.append(f"{blanks} non-numeric mark(s) were treated as 0.")
    untagged = [q for q in questions if not clo_map[q]]
    if untagged:
        warnings.append(f"{len(untagged)} question(s) have no CLO tag and are excluded from CLO attainment: "
                        + ", ".join(untagged))
    return {"questions": questions, "clo": clo_map, "max": max_map, "students": students,
            "id_col": first}, warnings


def compute_attainment(sheet: dict, student_threshold: float = 0.5, class_threshold: float = 0.6) -> list[dict]:
    """Per CLO: average score, share of students reaching student_threshold, attained flag."""
    by_clo: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    for q, tags in sheet["clo"].items():
        if sheet["max"].get(q, 0) <= 0:
            continue
        for t in tags:
            k = _norm_clo(t)
            labels.setdefault(k, t)
            by_clo.setdefault(k, []).append(q)
    st = sheet["students"]
    rows = []
    for k, qs in by_clo.items():
        mx = sum(sheet["max"][q] for q in qs)
        got = st[qs].sum(axis=1)
        pct = got / mx if mx else got * 0
        share = float((pct >= student_threshold).mean()) if len(pct) else 0.0
        rows.append({"clo_key": k, "clo": labels[k], "questions": qs, "max_marks": mx,
                     "n_students": int(len(pct)), "avg_score": float(pct.mean()) if len(pct) else 0.0,
                     "class_attainment": share, "attained": share >= class_threshold})
    rows.sort(key=lambda r: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", r["clo_key"])])
    return rows


def inflation_check(att_rows: list[dict], alignment_rows: list[dict] | None) -> list[dict]:
    """Attach the alignment finding to each attainment row and flag potential attainment-validity concerns."""
    al = {_norm_clo(r["id"]): r for r in (alignment_rows or [])}
    out = []
    for a in att_rows:
        r = al.get(a["clo_key"])
        flag, note = "", ""
        if r is None:
            flag = "Not checked"
            note = "No matching CLO in the alignment analysis."
        elif r["finding"] == "Potential mismatch" and a["attained"]:
            flag = "Potential attainment-validity concern"
            note = (f"Attained under the configured thresholds ({a['class_attainment']:.0%} of students), but the "
                    f"assessment evidence supplied reaches at most {level_name(r['max_demand'])} while the CLO states "
                    f"{level_name(r['target'])}. Check whether the CLO is assessed at its level elsewhere (e.g. a lab or "
                    "project not in the supplied documents) before relying on this figure.")
        elif r["finding"] == "Potential mismatch":
            flag = "Not attained + level mismatch"
            note = "Not attained, and the items used are below the CLO level - fix the assessment before acting on the number."
        elif r["finding"] in ("Needs review", "Insufficient evidence"):
            flag = "Validity unclear"
            note = f"Alignment finding: {r['finding']}. Confirm before relying on the attainment figure."
        elif a["attained"]:
            note = "Attained, and measured by items at the CLO level."
        else:
            flag = "Not attained"
            note = "Measured at the CLO level but below the class threshold - a teaching/learning issue rather than an assessment design issue."
        out.append({**a, "intended": r["target"] if r else None, "measured_up_to": r["max_demand"] if r else None,
                    "finding": r["finding"] if r else "", "plo": r.get("plo", "") if r else "",
                    "flag": flag, "note": note})
    return out


def plo_rollup(alignment_rows: list[dict], att: list[dict] | None = None) -> list[dict]:
    """Group CLOs by PLO: what evidence actually supports each programme outcome."""
    att_by = {a["clo_key"]: a for a in (att or [])}
    groups: dict[str, list[dict]] = {}
    for r in alignment_rows:
        plo = (r.get("plo") or "").strip() or "No PLO stated"
        m = re.match(r"(PLO[\s-]*\d+)", plo, flags=re.I)
        key = m.group(1).upper().replace(" ", "-") if m else plo
        groups.setdefault(key, []).append(r)
    out = []
    for plo, rs in groups.items():
        weak = [r for r in rs if r["finding"] in ("Potential mismatch", "Insufficient evidence")]
        atts = [att_by[_norm_clo(r["id"])]["class_attainment"] for r in rs if _norm_clo(r["id"]) in att_by]
        out.append({
            "plo": plo, "plo_label": rs[0].get("plo", "") or plo,
            "clos": [r["id"] for r in rs],
            "highest_intended": max((r["target"] or 0 for r in rs), default=0) or None,
            "highest_evidenced": max((r["max_demand"] or 0 for r in rs), default=0) or None,
            "weak_clos": [r["id"] for r in weak],
            "evidence": ("Weak" if len(weak) == len(rs) else "Partial" if weak else "Supported"),
            "avg_attainment": (sum(atts) / len(atts)) if atts else None,
        })
    out.sort(key=lambda x: (x["plo"] == "No PLO stated",
                            [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x["plo"])]))
    return out


def marks_by_level(alignment: dict) -> dict[int, float]:
    """Total marks per cognitive level across verified items with stated marks (higher rating when two)."""
    from .clo import item_level
    from .scoring import is_verified
    ov = alignment.get("overrides", {})
    totals: dict[int, float] = {}
    for it in alignment.get("items", []):
        if not is_verified(it) or it.get("marks") is None:
            continue
        _, hi, _ = item_level(it, ov)
        if hi:
            totals[hi] = totals.get(hi, 0.0) + float(it["marks"])
    return totals


TEMPLATE = ("Student,Mid Q1,Mid Q2,Final Q1,Lab Exam\n"
            "CLO,CLO-1,CLO-2,CLO-3,CLO-2/CLO-3\n"
            "MAX,5,10,10,20\n"
            "S01,4,7,6,15\n"
            "S02,5,5,8,12\n")
