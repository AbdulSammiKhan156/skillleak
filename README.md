# SkillLeak AI

**Find curriculum gaps. Verify assessment alignment. Prepare graduates for real-world engineering.**

AICON'26 Build With AI · Domain: EdTech & Workforce

SkillLeak is a curriculum intelligence and employability-gap analysis tool built first for Pakistani engineering education. It helps engineering departments catch two kinds of problem before they become bigger ones:

- CLO–assessment alignment issues
- gaps between the curriculum and what industry asks for

It is a **decision-support and audit-assistance tool**. It does not replace faculty judgement, professional curriculum review or any official accreditation process. It does **not** certify or guarantee PEC, HEC or any other compliance.

## The two problems it addresses

**A. Course quality and OBE alignment.** Engineering programmes use Outcome-Based Education (OBE):

- each Course Learning Outcome (CLO) has a cognitive level (C1–C6) and maps to a Programme Learning Outcome (PLO)
- course files must show that assessments actually measure each CLO

A common silent problem is a CLO that says *"Diagnose and resolve connectivity faults (C4)"* while the only exam question tagged to it is *"List the steps of the bottom-up troubleshooting approach"* (C1). SkillLeak flags this as a **potential mismatch** and shows the reasoning and evidence.

**B. Graduate employability.** Syllabi can miss practical skills that employers ask for, such as troubleshooting, monitoring, packet analysis and documentation. SkillLeak compares what the course teaches, practises and assesses against requirements quoted from public Pakistani job postings. It then proposes a practical fix.

## Pipeline: Problem → Input → AI → Output → Impact

| Stage | What happens |
|---|---|
| Input | Course outline with CLOs, lab manual, exam papers/quizzes, job postings (PDF/TXT/MD), or CLOs and questions typed in manually |
| AI | An LLM (Gemini, Claude or any OpenAI-compatible model) does three things: (1) extracts CLOs, levels, PLOs and assessment items, (2) rates each item's cognitive demand on Bloom's taxonomy by *what the student must do*, (3) maps items to CLOs and employer competencies to course evidence. It also drafts aligned assessment items, practical labs and student practice plans. |
| Guard-rails (code) | Every AI quote is checked against the source text, and unmatched claims are excluded. Findings and scores come from fixed, documented rules. Faculty can correct levels and mappings, and findings recompute instantly. |
| Output | CLO→assessment alignment table with findings and evidence, an industry Skill Leakage Map, a Gap-to-Lab generator with teacher review, Student Mode practice plans, and Markdown reports |
| Impact | Faster, evidence-backed course-file review, and assessments and labs that make students demonstrate the skills employers ask for (SDG 4 and SDG 8) |

## What makes it different

- **Two AI ratings as a disagreement detector.** Each assessment item is rated once in context and once by a *blind* pass that sees only the item text. A mismatch is flagged only when both ratings are below the CLO level; split ratings go to faculty review. Both passes use the same model, so agreement shows consistency, not proven accuracy. This design responds to published evidence that zero-shot LLMs reach roughly 0.72–0.73 accuracy on Bloom classification, with errors mostly between adjacent levels (arXiv:2511.10903).
- **Attainment-validity check.** It cross-checks CLO attainment computed from a marks sheet against assessment alignment, and flags a *potential attainment-validity concern*. For example: *"CLO-5 is attained by 100% of students under the thresholds, but the supplied assessments measure it only up to C1–C2 while the CLO states C4."* This is a prompt to check, not proof; the CLO may be assessed elsewhere.
- **Aligned-mark share** for each CLO: marks at or above the CLO level ÷ marks linked to it. Also **PLO assessment-evidence coverage** for one course, and a **marks-by-cognitive-level** profile of the paper.
- **Benchmark harness.** A public 600-question labelled set (MIT-licensed BloomNet repository; annotation provenance undocumented) compares the AI rater with a keyword baseline on the same questions. The keyword baseline result is included: 62.7% exact, 73.8% within one level on all 600. The original-taxonomy labels are mapped approximately to C1–C6.
- **Course-file pack (Excel).** It contains the CLO alignment, the CLO × item matrix, attainment, PLO evidence, industry gaps, sources and proposed items.
- **Quick check.** Type any CLO and exam question and get a two-rater verdict in seconds, plus an aligned rewrite.

## Features (tabs)

1. **Documents**: sample course file plus job postings, file upload, or **manual CLO/question entry** (fallback). Shows a provenance table for job data.
0. **Quick check** (on the Documents tab): one CLO and one question → a two-rater verdict and a suggested aligned rewrite.
2. **CLO Alignment**:
   - per-CLO findings: *Potential mismatch / Needs review / Insufficient evidence / No apparent mismatch detected*
   - evidence quotes, marks, mapping basis (explicit CLO tag vs content match) and AI confidence label
   - **Generate aligned assessment item**, with a before/after level comparison
   - **Faculty corrections** (override CLO level or item demand, exclude a mapping)
   - unmapped items, missing information, and rejected AI quotes
3. **Attainment & PLO** (no AI, works offline on saved scans):
   - CLO attainment from a marks CSV, with configurable thresholds
   - the attainment-validity flag for each CLO
   - PLO assessment-evidence coverage
   - Excel course-file pack
4. **Industry Gaps**: taught / practised / assessed vs employer demand, with linked CLOs and gap priority.
5. **Evidence**: verbatim quotes behind every finding, plus rejected AI claims.
6. **Gap-to-Lab**: AI-drafted lab, 100-point rubric and matching exam question. A teacher approves, rejects or requests changes, and the AI revises from that feedback.
7. **Student Mode**: a self-practice plan for a skill gap: why it matters (grounded only in employer quotes), steps, free tools, expected output and a self-check list.
8. **Validation**: the Bloom-rating benchmark (AI rater vs keyword baseline, confusion matrix). You can run it from the app or from the CLI.
9. **Method**: every rule, formula and limitation.

### CLO finding rules (deterministic, in `skillleak/clo.py`)

Each CLO is judged on **all** of its mapped, verified items together:

- **Potential mismatch**: every mapped item demands less than the *stated* (or faculty-set) CLO level.
- **Needs review**: any of these:
  - the CLO level was only AI-inferred
  - only low-confidence content matches reach the level
  - items at the level carry less than 50% of the mapped marks
  - no level is available
- **Insufficient evidence**: no verified assessment item is linked.
- **No apparent mismatch detected**: at least one mapped item reaches the CLO level.

The **Aligned CLOs indicator** is the number of CLOs with "No apparent mismatch detected" divided by the number of CLOs detected. It is a prototype heuristic, not a PEC score.

### Industry gap scoring (in `skillleak/scoring.py`)

```
Gap priority = D × (3 − (T + P + A))           range 0-15
D = ceil(5 × share of postings with a verified quote)
T = syllabus evidence; P / A = practice / assessment evidence at Bloom ≥ Apply (3)
```

If a document type is not supplied, a competency is labelled **Cannot fully evaluate** rather than "not taught".

## Data provenance

| Data | Status |
|---|---|
| `sample_data/01_syllabus.md`, `02_lab_manual.md`, `03_assessments.md` | **Synthetic demo data.** A fictional EE-357 course in the style of a Pakistani OBE course file. Written for this demo and designed to contain typical issues for the AI to find. |
| `sample_data/marks_ee357_synthetic.csv` | **Synthetic.** Fictional marks for 40 fictional students, generated with a fixed seed for the demo. |
| `eval/data/bloom_questions_600.csv` | Public benchmark: 600 questions with Bloom labels (annotators and procedure not documented in the source repository), 100 per level, from the MIT-licensed [BloomNet](https://github.com/macabdul9/BloomNet) repository (`data/BloomTaxonomy.csv`). The licence file is in `eval/data/`. Used only for evaluation. |
| `sample_data/04-07_job_*.md` | **Synthetic.** Fictional job descriptions, labelled as such in the files and in the app. |
| `sample_data/public_jobs/*.md` + `sources.json` | **Public postings, sourced.** Selected lines quoted from 4 Pakistani network-engineering postings (ROZEE.PK, getpakjob.com), with URL, employer as listed, location, posted date and access date (10 Oct 2026). **All four are closed or expired.** They document what employers asked for, not current vacancies. Two ask for 2–5 years' experience. ROZEE.PK marks three of the employers "Unverified". |

No private, university or personal data is included.

## Run locally

```bash
cd skillleak
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # then add ONE provider key (or paste it in the sidebar)
streamlit run app.py
```

**Getting a free key:** Google AI Studio → *Get API key*. Select "Google Gemini" in the sidebar, then click **Test key & list models** and copy a valid model name into the *Model* box.

## Environment variables (all optional if you paste a key in the sidebar)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `Google Gemini`, `Anthropic Claude` or `OpenAI-compatible` |
| `LLM_MODEL` | Model name, e.g. a current Gemini Flash model |
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Key for the chosen provider |
| `OPENAI_BASE_URL` | For OpenAI-compatible providers (Groq, OpenRouter, ...) |

Never commit `.env` (it is git-ignored). On Streamlit Cloud, put these under *Secrets*.

## Live check with your real API key (do this before presenting)

```bash
python scripts/live_check.py                 # sample workflow, quick check (3 cases), attainment, exports
python scripts/live_check.py --benchmark 10  # also scores the AI rater on 60 benchmark questions
```

Every AI step in this script calls your real provider; nothing is mocked. It prints PASS/FAIL for each step and a "demo story check" (is CLO-5 flagged?). It writes `live_check_report.md` and saves the analysis to `runs/` for offline replay.

## Validation benchmark

```bash
python eval/run_benchmark.py --rater keyword --per-level 100   # no key: 600 questions
python eval/run_benchmark.py --rater ai --per-level 20         # your key: 120 questions (same seed)
```

Results are saved in `eval/results/` and shown in the Validation tab.

## Tests (no key or internet needed)

```bash
pip install pytest
python -m pytest -q
```

44 tests use canned model replies (no real AI provider is called). Nine of them check the provider adapters against the documented Gemini, Claude and OpenAI request and response formats, retries on rate limits and server errors, and a full pipeline run over a real local HTTP server. Together the tests, which live only in `tests/`, to check:

- extraction
- quote verification (including invented quotes, CLOs and exam items)
- the two-rater rules and a failing second rater
- attainment and the validity check
- the aligned-mark share rule (a 2-mark C4 item among 98 low-level marks)
- the PLO roll-up and malformed marks sheets
- the quick check
- CLO finding rules
- faculty overrides
- marks and inferred-level rules
- malformed model output
- JSON retry
- scoring and reports
- the Streamlit UI: replay, alignment-only scans, old scans, manual input and correction buttons

## Deploy (Streamlit Community Cloud)

1. Push to a public GitHub repo.
2. Go to share.streamlit.io → New app → main file `app.py`.
3. Add your key and settings under Secrets, then deploy.

## Demo backup

Every live analysis is saved to `runs/`. After one successful live run, rename the newest file to `runs/demo_scan.json` and commit it. If Wi-Fi or the API fails on stage, use **Sidebar → Saved scans → Load saved scan**. The app labels it as a replay, with its timestamp and model.

## Limitations

- Findings reflect only the supplied documents. "Not identified" is not proof that a skill is never taught.
- Bloom ratings and mappings come from an LLM and can be wrong. Each one is shown with its quote and reason, and faculty can correct it.
- AI confidence labels are the model's own judgement, not calibrated probabilities.
- Scores and indicators are prototype heuristics, not validated or official measures.
- No OCR for scanned PDFs. Each document is truncated to 15,000 characters.
- Public job postings are dated, closed listings.
- The CLO analysis does not yet use PLO attainment data, student results or affective/psychomotor domains.

## Disclosure

- **Built during AICON'26** (9–10 Oct 2026) with an AI coding assistant (Claude).
- **Runtime AI:** the user-selected LLM API.
- **Libraries:** Streamlit, PyMuPDF, Plotly, pandas, requests, python-dotenv.
- **Prior work:** curriculum–job-market alignment is an existing research area. SkillLeak's contribution is combining these pieces:
  - Bloom-level-aware CLO–assessment checking with source-verified evidence and faculty correction
  - a four-way taught / practised / assessed / required evidence chain
  - AI-drafted, teacher-reviewed interventions
