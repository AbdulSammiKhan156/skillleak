"""Validation benchmark: how well does the Bloom rater agree with a public labelled set?

Data: 600 questions labelled with Bloom levels (100 per level) from the MIT-licensed
BloomNet repository (github.com/macabdul9/BloomNet, data/BloomTaxonomy.csv). The repository does
not document the annotators or labelling procedure, so label provenance is unverified. Original
1956 labels are mapped approximately to the revised scale (Synthesis -> 6, Evaluation -> 5);
the two taxonomies order these levels differently, so C5/C6 comparisons are the least reliable.

Two raters are compared on the SAME questions:
  * keyword baseline - classic first-verb rule (deterministic, no AI)
  * SkillLeak's independent AI rater - the prompt used as the second rater in the CLO check
Metrics: exact accuracy, within-one-level accuracy, macro-F1, confusion matrix.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import pandas as pd

from .bloom import LEVELS, ORIGINAL_TO_REVISED, keyword_baseline, rate_items_blind
from .llm import LLMConfig

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "data" / "bloom_questions_600.csv"
RESULTS = ROOT / "eval" / "results"
SOURCE = "BloomNet repository data (github.com/macabdul9/BloomNet, MIT licence), 600 questions, 100 per level"


def load_dataset() -> list[dict]:
    df = pd.read_csv(DATA, encoding="utf-8")
    out = []
    for i, r in df.iterrows():
        lv = ORIGINAL_TO_REVISED.get(str(r["Label"]).strip().lower())
        if lv and str(r["Text"]).strip():
            out.append({"id": f"Q{i + 1}", "text": str(r["Text"]).strip(), "gold": lv})
    return out


def stratified_sample(rows: list[dict], per_level: int, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for lv in sorted(LEVELS):
        pool = [r for r in rows if r["gold"] == lv]
        out += rnd.sample(pool, min(per_level, len(pool)))
    return out


def metrics(gold: list[int], pred: list[int | None]) -> dict:
    pairs = [(g, p) for g, p in zip(gold, pred)]
    n = len(pairs)
    answered = [(g, p) for g, p in pairs if p]
    exact = sum(1 for g, p in pairs if p == g)
    within = sum(1 for g, p in pairs if p and abs(p - g) <= 1)
    f1s = []
    for lv in LEVELS:
        tp = sum(1 for g, p in pairs if g == lv and p == lv)
        fp = sum(1 for g, p in pairs if g != lv and p == lv)
        fn = sum(1 for g, p in pairs if g == lv and p != lv)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    conf = [[sum(1 for g, p in pairs if g == gl and p == pl) for pl in LEVELS] for gl in LEVELS]
    return {"n": n, "answered": len(answered), "accuracy": exact / n if n else 0.0,
            "within_one": within / n if n else 0.0, "macro_f1": sum(f1s) / len(f1s),
            "confusion": conf}


def run(rater: str, per_level: int = 20, cfg: LLMConfig | None = None, seed: int = 7,
        save: bool = True) -> dict:
    rows = stratified_sample(load_dataset(), per_level, seed)
    gold = [r["gold"] for r in rows]
    if rater == "keyword":
        pred = [keyword_baseline(r["text"]) for r in rows]
        label = "Keyword baseline (first Bloom verb)"
    elif rater == "ai":
        if cfg is None:
            raise ValueError("An LLM configuration is required for the AI rater.")
        ratings = rate_items_blind([{"id": r["id"], "text": r["text"]} for r in rows], cfg, batch=30)
        pred = [ratings.get(r["id"], {}).get("level") for r in rows]
        label = f"SkillLeak AI rater ({cfg.label()})"
    else:
        raise ValueError("rater must be 'keyword' or 'ai'")
    res = {"rater": rater, "label": label, "per_level": per_level, "seed": seed, "source": SOURCE,
           "created_at": dt.datetime.now().isoformat(timespec="seconds"), **metrics(gold, pred),
           "examples": [{"text": r["text"], "gold": g, "pred": p} for r, g, p in zip(rows, gold, pred)
                        if p != g][:15]}
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        name = f"{rater}_{per_level * 6}q_{res['created_at'].replace(':', '-')}.json"
        (RESULTS / name).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def saved_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    out = []
    for p in sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append({**json.loads(p.read_text(encoding="utf-8")), "_file": p.name})
        except (OSError, ValueError):
            continue
    return out
