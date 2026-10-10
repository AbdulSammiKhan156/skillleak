"""Run the Bloom-rating validation benchmark from the command line.

    python eval/run_benchmark.py --rater keyword --per-level 100      # no key needed (all 600 questions)
    python eval/run_benchmark.py --rater ai --per-level 20             # 120 questions, uses your API key

The AI rater reads provider/model/key from .env (LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY ...).
Results are saved to eval/results/ and shown in the app's Validation tab.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillleak import benchmark  # noqa: E402
from skillleak.llm import PROVIDERS, LLMConfig  # noqa: E402

KEYS = {"Google Gemini": "GEMINI_API_KEY", "Anthropic Claude": "ANTHROPIC_API_KEY",
        "OpenAI-compatible": "OPENAI_API_KEY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", choices=["keyword", "ai"], default="keyword")
    ap.add_argument("--per-level", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = None
    if a.rater == "ai":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        prov = os.environ.get("LLM_PROVIDER", "Google Gemini")
        cfg = LLMConfig(provider=prov, api_key=os.environ.get(KEYS.get(prov, ""), ""),
                        model=os.environ.get("LLM_MODEL") or PROVIDERS[prov]["default_model"],
                        base_url=os.environ.get("OPENAI_BASE_URL", ""))
        if not cfg.api_key:
            sys.exit(f"Set {KEYS.get(prov)} in .env first.")
    r = benchmark.run(a.rater, a.per_level, cfg, a.seed)
    print(f"{r['label']}: n={r['n']}  accuracy={r['accuracy']:.1%}  within-one-level={r['within_one']:.1%}  "
          f"macro-F1={r['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
