"""Bloom's-taxonomy rating helpers.

* rate_items_blind(): independent LLM rater (sees item text only, not the CLO).
  Used as the second rater in the CLO check and evaluated by the benchmark.
* keyword_baseline(): the classic verb-list rule, used ONLY as a comparison baseline
  in the validation benchmark (it shows why a context-aware AI rater is needed).
"""
from __future__ import annotations

import re

from . import prompts
from .llm import LLMConfig, call_json

LEVELS = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}

# Original (1956) Bloom labels used by many public datasets -> revised level number.
ORIGINAL_TO_REVISED = {"knowledge": 1, "comprehension": 2, "application": 3, "analysis": 4,
                       "evaluation": 5, "synthesis": 6,
                       "remember": 1, "understand": 2, "apply": 3, "analyze": 4, "analyse": 4,
                       "evaluate": 5, "create": 6}


def _lvl(x) -> int | None:
    try:
        v = int(x)
        return v if 1 <= v <= 6 else None
    except (TypeError, ValueError):
        return None


def rate_items_blind(items: list[dict], cfg: LLMConfig, batch: int = 25) -> dict[str, dict]:
    """items: [{"id":..., "text":...}] -> {id: {"level": int|None, "reason": str}}"""
    out: dict[str, dict] = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        listing = "\n".join(f"- {it['id']}: {it['text']}" for it in chunk)
        raw = call_json(cfg, prompts.BLOOM_RATER_SYSTEM, prompts.BLOOM_RATER_USER.format(items=listing),
                        max_tokens=6000)
        valid = {it["id"] for it in chunk}
        for r in raw.get("ratings", []) or []:
            if isinstance(r, dict) and str(r.get("id")) in valid:
                out[str(r["id"])] = {"level": _lvl(r.get("level")), "reason": str(r.get("reason") or "")}
    return out


# ---------------------------------------------------------------------------
# Keyword baseline (comparison only)
# ---------------------------------------------------------------------------
VERBS = {
    1: "define list name state identify label recall recognize recognise memorize memorise match "
       "repeat record locate select quote who when where tell",
    2: "explain describe summarize summarise classify discuss interpret paraphrase compare contrast "
       "illustrate outline give restate translate distinguish report review why",
    3: "apply calculate compute solve use demonstrate configure implement execute operate show "
       "complete construct determine find modify prepare sketch convert predict",
    4: "analyze analyse examine investigate diagnose differentiate categorize categorise infer "
       "deduce inspect test troubleshoot break debug isolate relate organize organise",
    5: "evaluate judge justify assess critique argue defend appraise recommend rate prioritize "
       "prioritise conclude support decide criticize criticise weigh verify",
    6: "design create develop formulate compose plan propose invent devise generate build "
       "construct produce write program synthesize synthesise assemble",
}
_VERB_LEVEL = {}
for _lv, _ws in VERBS.items():
    for _w in _ws.split():
        _VERB_LEVEL.setdefault(_w, _lv)


def keyword_baseline(text: str) -> int:
    """Return the level of the first Bloom verb found (classic rule-based approach)."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    for t in toks:
        if t in _VERB_LEVEL:
            return _VERB_LEVEL[t]
        if t.endswith("s") and t[:-1] in _VERB_LEVEL:
            return _VERB_LEVEL[t[:-1]]
    if toks and toks[0] in ("what", "which"):
        return 1
    if toks and toks[0] == "how":
        return 2
    return 2
