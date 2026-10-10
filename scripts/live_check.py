"""Pre-submission reliability audit against a REAL AI provider.

    python scripts/live_check.py                 # full sample workflow + Quick Check + exports
    python scripts/live_check.py --benchmark 10  # also run the AI rater on 60 benchmark questions

Reads LLM_PROVIDER / LLM_MODEL / *_API_KEY from .env (same as the app). Nothing here is mocked:
every AI step calls your configured provider. A report is written to live_check_report.md and the
analysis is saved to runs/ so it can be replayed in the app if the internet fails on stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skillleak import attainment as attn  # noqa: E402
from skillleak import benchmark  # noqa: E402
from skillleak.clo import analyze, level_name, quick_check  # noqa: E402
from skillleak.export import course_file_pack  # noqa: E402
from skillleak.extract import load_sample_documents, job_sources  # noqa: E402
from skillleak.llm import PROVIDERS, LLMConfig, list_models  # noqa: E402
from skillleak.pipeline import RUNS_DIR, save_run, run_scan  # noqa: E402
from skillleak.report import alignment_markdown, run_markdown  # noqa: E402
from skillleak.scoring import score_run  # noqa: E402

KEYS = {"Google Gemini": "GEMINI_API_KEY", "Anthropic Claude": "ANTHROPIC_API_KEY",
        "OpenAI-compatible": "OPENAI_API_KEY"}

QUICK_CASES = [
    ("clear C1 recall", "Diagnose and resolve common connectivity faults in small routed and switched networks.", 4,
     "List the steps of the bottom-up troubleshooting approach.", "Potential mismatch"),
    ("genuine C4 diagnosis", "Diagnose and resolve common connectivity faults in small routed and switched networks.", 4,
     "PC1 (192.168.1.10/24, gateway 192.168.1.254) cannot ping the server at 10.0.0.5, but PC2 on the same switch can. "
     "Using the show ip interface brief and show running-config outputs in Figure 3, identify the fault, justify your "
     "diagnosis from the evidence, and give the commands that fix it.", "No apparent mismatch detected"),
    ("borderline", "Configure static and dynamic routing (OSPF) and VLAN-based switching on network devices.", 3,
     "Explain the steps you would follow to configure OSPF area 0 on a router.", "either"),
]

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        log(f"[PASS] {name}  ({time.time() - t0:.1f}s)")
        return out
    except Exception as exc:  # report and continue
        log(f"[FAIL] {name}  ({time.time() - t0:.1f}s): {type(exc).__name__}: {str(exc)[:400]}")
        lines.append("```\n" + traceback.format_exc()[-1500:] + "\n```")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0, help="questions per level for the AI benchmark (0 = skip)")
    ap.add_argument("--jobs", choices=["public", "synthetic"], default="public")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    prov = os.environ.get("LLM_PROVIDER", "Google Gemini")
    if prov not in PROVIDERS:
        sys.exit(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
    cfg = LLMConfig(prov, os.environ.get(KEYS[prov], ""), os.environ.get("LLM_MODEL") or PROVIDERS[prov]["default_model"],
                    os.environ.get("OPENAI_BASE_URL", ""))
    if not cfg.api_key:
        sys.exit(f"Set {KEYS[prov]} in .env first (never commit .env).")

    log(f"# SkillLeak live check - {dt.datetime.now():%Y-%m-%d %H:%M}")
    log(f"Provider/model: {cfg.label()}  (all AI steps below are REAL calls, nothing mocked)\n")

    models = step("1. API key works (list models)", lambda: list_models(cfg))
    if models is not None and cfg.model not in models:
        log(f"   ! model '{cfg.model}' not in the list returned for this key - set LLM_MODEL to one of: "
            + ", ".join(models[:12]))

    docs = load_sample_documents(a.jobs)
    run = step("2. Full sample analysis (CLO alignment + second rater + industry)",
               lambda: run_scan(docs, cfg, lambda m, p: print("   ..", m)))
    if run:
        run["meta"]["input"] = f"live_check / sample course / {a.jobs} job postings"
        al = run.get("alignment") or {}
        res = analyze(al) if al else None
        m = run["meta"]
        log(f"   quotes returned {m['quotes_total']}, rejected by source check {m['quotes_rejected']}")
        if res:
            ag = res["agreement"]
            log(f"   CLOs {len(al['clos'])} | items {len(al['items'])} | mappings {len(al['mappings'])} | "
                f"second rater: {al.get('verifier', {}).get('status')}"
                + (f" | rater agreement exact {ag['exact']:.0%}, within one {ag['within_one']:.0%} (n={ag['n']})"
                   if ag['n'] else ""))
            for r in res["rows"]:
                share = f"{r['marks_share_at_level']:.0%}" if r["marks_share_at_level"] is not None else "-"
                if r.get("marks_coverage") and r["marks_share_at_level"] is not None and \
                        r["marks_coverage"].split("/")[0] != r["marks_coverage"].split("/")[1]:
                    share += f" (marks on {r['marks_coverage']})"
                log(f"   {r['id']:<6} intended {level_name(r['target']):<14} highest {level_name(r['max_demand']):<14} "
                    f"aligned-mark share {share:<18} -> {r['finding']}{' (' + r['severity'] + ')' if r['severity'] else ''}")
            c5 = next((r for r in res["rows"] if r["id"].replace(" ", "-").upper() == "CLO-5"), None)
            log("   DEMO STORY CHECK - CLO-5 flagged as Potential mismatch: "
                + ("YES" if c5 and c5["finding"] == "Potential mismatch" else
                   f"NO (got: {c5['finding'] if c5 else 'CLO-5 not found'}) - adjust the demo script"))
        rows = score_run(run)
        log(f"   industry competencies with verified employer quotes: {len(rows)}")
        for r in rows[:5]:
            log(f"   gap {r['priority']:>2}  D={r['D']} ({r['jd_count']} postings)  {r['status']:<28} {r['name']}")
        path = RUNS_DIR / "demo_scan.json"
        name = "demo_scan.json" if not path.exists() else f"live_check_{dt.datetime.now():%Y%m%d_%H%M}.json"
        saved = save_run(run, name)
        log(f"   saved for offline replay: runs/{saved.name}")

        if res:
            def att():
                sheet, _ = attn.parse_marks_csv((ROOT / "sample_data/marks_ee357_synthetic.csv").read_bytes())
                return attn.inflation_check(attn.compute_attainment(sheet), res["rows"])
            arows = step("3. Attainment-validity check on sample marks", att)
            for x in arows or []:
                if x["flag"]:
                    log(f"   {x['clo']}: {x['class_attainment']:.0%} attained -> {x['flag']}")
        step("4. Markdown reports", lambda: len(alignment_markdown(run)) + len(run_markdown(run)))
        step("5. Excel course-file pack", lambda: len(course_file_pack(run, sources=job_sources(a.jobs))))

    log("\n6. Quick check (real model)")
    for label, clo_txt, lvl, q, expect in QUICK_CASES:
        r = step(f"   quick check - {label}", lambda: quick_check(clo_txt, q, cfg, lvl))
        if r:
            ok = expect == "either" or r["verdict"] == expect
            log(f"     rater1 {level_name(r['r1'])} | rater2 {level_name(r['r2'])} | verdict {r['verdict']}"
                + ("" if ok else f"   <-- expected {expect}; review before demoing this case"))

    if a.benchmark:
        log("")
        r = step(f"7. AI rater benchmark ({a.benchmark * 6} questions)", lambda: benchmark.run("ai", a.benchmark, cfg))
        k = benchmark.run("keyword", a.benchmark)
        if r:
            log(f"   AI rater: {r['accuracy']:.1%} exact, {r['within_one']:.1%} within one level "
                f"(answered {r['answered']}/{r['n']})")
        log(f"   keyword baseline, same questions: {k['accuracy']:.1%} exact, {k['within_one']:.1%} within one")
        log("   Labels: BloomNet repository data, annotation provenance undocumented; original taxonomy mapped "
            "approximately to C1-C6.")

    (ROOT / "live_check_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {ROOT / 'live_check_report.md'}")


if __name__ == "__main__":
    main()
