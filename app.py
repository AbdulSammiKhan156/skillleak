"""SkillLeak AI - curriculum intelligence for Pakistani engineering education (Streamlit app).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skillleak import attainment as attn
from skillleak import benchmark
from skillleak.clo import (FINDINGS, analyze, clo_for_text, generate_aligned_item, level_name,
                           quick_check, quick_rewrite)
from skillleak.export import course_file_pack
from skillleak.extract import (CATEGORIES, JOB_SETS, build_documents, build_manual_documents,
                               job_sources, load_sample_documents)
from skillleak.llm import PROVIDERS, LLMConfig, list_models
from skillleak.pipeline import (can_run, generate_lab, generate_student_pack, list_saved_runs,
                                load_run, run_scan, save_run)
from skillleak.prompts import BLOOM
from skillleak.report import alignment_markdown, lab_markdown, run_markdown, student_markdown
from skillleak.scoring import APPLY_LEVEL, coverage_index, is_verified, score_run

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="SkillLeak AI", page_icon="🔎", layout="wide")

# Status palette (fixed meanings, always paired with an icon + label)
GOOD, WARN, SERIOUS, CRIT, MUTED = "#0ca30c", "#fab219", "#ec835a", "#d03b3b", "#8a8984"
STATUS_COLOR = {"Covered": GOOD, "Practised but not in syllabus": WARN,
                "Phantom competence risk": SERIOUS, "Industry blind spot": CRIT,
                "Cannot fully evaluate": MUTED}
STATUS_ICON = {"Covered": "✅", "Practised but not in syllabus": "🟡",
               "Phantom competence risk": "👻", "Industry blind spot": "🚫",
               "Cannot fully evaluate": "❔"}
FINDING_STYLE = {  # finding -> (css class, icon)
    "Potential mismatch": ("f-bad", "⚠️"),
    "Needs review": ("f-warn", "🔍"),
    "Insufficient evidence": ("f-muted", "❔"),
    "No apparent mismatch detected": ("f-good", "✅"),
}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.sl-hero h1 {margin-bottom: 0; font-size: 2.1rem;}
.sl-hero p {color: #52514e; margin-top: .2rem; font-size: 1.05rem;}
.sl-cards {display:grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap:.6rem; margin:.6rem 0 1rem 0;}
.sl-card {background:#fcfcfb; border:1px solid #e4e3df; border-radius:10px; padding:.65rem .8rem; font-size:.88rem;}
.sl-card b {display:block; color:#184f95; margin-bottom:.2rem; font-size:.9rem;}
table.sl-matrix {border-collapse:separate; border-spacing:0 4px; width:100%; font-size:.9rem;}
table.sl-matrix th {text-align:left; color:#52514e; font-weight:600; padding:4px 8px;
  border-bottom:1px solid #e4e3df;}
table.sl-matrix td {padding:7px 8px; background:#fcfcfb; border-top:1px solid #efeeea;
  border-bottom:1px solid #efeeea; vertical-align:top;}
.cell {display:inline-block; white-space:nowrap; border-radius:6px; padding:2px 8px; font-weight:600; font-size:.8rem;}
.c-good, .f-good {background:#e3f5e3; color:#0b5f0b;} .c-warn, .f-warn {background:#fff1cf; color:#7a5200;}
.c-none, .f-bad {background:#fbe3e3; color:#9b2222;} .f-muted {background:#efeeea; color:#52514e;}
.badge {display:inline-block; border-radius:6px; padding:1px 7px; font-size:.75rem; font-weight:600;
  margin:0 4px 2px 0; background:#efeeea; color:#52514e;}
.quote {border-left:3px solid #86b6ef; padding:.35rem .7rem; margin:.35rem 0 .1rem 0;
  background:#f7f9fc; font-size:.9rem;}
.quote.bad {border-left-color:#d03b3b; background:#fdf3f3; text-decoration: line-through;}
.reason {color:#52514e; font-size:.8rem; margin:0 0 .6rem .8rem;}
.replay {background:#fff7e6; border:1px solid #fab219; border-radius:8px; padding:.5rem .8rem; margin-bottom:.8rem;}
.disclaimer {background:#eef4fc; border-radius:8px; padding:.45rem .8rem; font-size:.85rem; color:#184f95; margin-bottom:.6rem;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("run", None)
ss.setdefault("run_path", None)
ss.setdefault("replay", False)


# ---------------------------------------------------------------------------
# Sidebar - model settings and saved scans
# ---------------------------------------------------------------------------
def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


KEY_NAMES = {"Google Gemini": "GEMINI_API_KEY", "Anthropic Claude": "ANTHROPIC_API_KEY",
             "OpenAI-compatible": "OPENAI_API_KEY"}

with st.sidebar:
    st.header("⚙️ AI model")
    default_provider = _secret("LLM_PROVIDER") or "Google Gemini"
    providers = list(PROVIDERS)
    provider = st.selectbox("Provider", providers,
                            index=providers.index(default_provider) if default_provider in providers else 0)
    env_key = _secret(KEY_NAMES[provider])
    api_key = st.text_input("API key", type="password", value="",
                            placeholder="Loaded from secrets" if env_key else "Paste your key",
                            help="Never stored. Leave empty to use the key from .env / Streamlit secrets.")
    api_key = api_key or env_key
    model = st.text_input("Model", value=_secret("LLM_MODEL") or PROVIDERS[provider]["default_model"])
    base_url = ""
    if provider == "OpenAI-compatible":
        base_url = st.text_input("Base URL", value=_secret("OPENAI_BASE_URL")
                                 or PROVIDERS[provider]["default_base_url"],
                                 help="Works with OpenAI, Groq, OpenRouter and other compatible APIs.")
    cfg = LLMConfig(provider=provider, api_key=api_key, model=model.strip(), base_url=base_url)
    st.caption("🔑 Key status: " + ("ready" if api_key else "missing"))
    if st.button("Test key & list models", width="stretch", disabled=not api_key):
        try:
            models = list_models(cfg)
            st.success(f"Key works. {len(models)} models available.")
            st.code("\n".join(models[:40]) or "(none listed)", language=None)
        except Exception as exc:
            st.error(str(exc)[:600])

    st.divider()
    st.header("💾 Saved scans")
    st.caption("Every live scan is saved automatically. Load one as a backup if the "
               "internet or API fails during the demo. Replays are clearly labelled.")
    saved = list_saved_runs()
    if saved:
        pick = st.selectbox("Saved scan", saved, format_func=lambda p: p.name)
        if st.button("Load saved scan", width="stretch"):
            try:
                ss.run, ss.run_path, ss.replay = load_run(pick), pick, True
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load this scan: {exc}")
    else:
        st.caption("No saved scans yet.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="sl-hero"><h1>🔎 SkillLeak AI</h1>
<p><b>Find curriculum gaps. Verify assessment alignment. Prepare graduates for real-world engineering.</b><br>
Curriculum intelligence for Pakistani engineering departments - a decision-support tool for faculty,
not a replacement for faculty judgement or official accreditation review.</p></div>
""", unsafe_allow_html=True)

if ss.run and ss.replay:
    m = ss.run["meta"]
    st.markdown(f"<div class='replay'>⏪ <b>Replaying a saved scan</b> from {html.escape(m['created_at'])} "
                f"(model: {html.escape(m['model'])}). These results were produced earlier - run a new scan "
                "on the Documents tab for a live analysis.</div>", unsafe_allow_html=True)

tab_docs, tab_clo, tab_att, tab_map, tab_ev, tab_lab, tab_student, tab_val, tab_method = st.tabs(
    ["① Documents", "② CLO Alignment", "③ Attainment & PLO", "④ Industry Gaps", "⑤ Evidence",
     "⑥ Gap-to-Lab", "⑦ Student Mode", "⑧ Validation", "ⓘ Method"])


def safe_error(exc: Exception) -> str:
    msg = str(exc)
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "***")
    return msg[:800]


def fmt_marks(v) -> str:
    if v is None:
        return "not stated"
    return str(int(v)) if float(v).is_integer() else str(v)


LEVEL_OPTS = [""] + [f"C{i} {BLOOM[i]}" for i in range(1, 7)]


def _opt_to_level(v) -> int | None:
    if isinstance(v, str) and v.startswith("C") and v[1:2].isdigit():
        return int(v[1])
    return None


# ---------------------------------------------------------------------------
# Tab 1 - documents and scan
# ---------------------------------------------------------------------------
with tab_docs:
    st.markdown("""
<div class="sl-cards">
<div class="sl-card"><b>1 · The problem</b>Course files can list CLOs at one cognitive level while exams test a lower one,
and syllabi can miss practical skills employers ask for. Checking this by hand is slow.</div>
<div class="sl-card"><b>2 · What we analyse</b>Course outline + CLOs, lab manuals, exam papers/quizzes, and employer job postings.</div>
<div class="sl-card"><b>3 · How the AI works</b>An LLM extracts CLOs, assessment items and employer competencies, rates cognitive
demand on Bloom's taxonomy, and maps items to CLOs and skills.</div>
<div class="sl-card"><b>4 · The evidence</b>Every AI claim carries a quote that code checks against the source. Findings and
scores are computed by transparent rules, not by the AI.</div>
<div class="sl-card"><b>5 · What next</b>Faculty correct mappings, generate an aligned assessment item or a practical lab,
and students get a self-practice plan.</div>
</div>""", unsafe_allow_html=True)

    with st.expander("⚡ Quick check - type any CLO and exam question and see if they match", expanded=False):
        q1, q2 = st.columns([3, 1])
        qc_clo = q1.text_input("CLO", value="Diagnose and resolve common connectivity faults in small routed and "
                                             "switched networks.", key="qc_clo")
        qc_lvl = q2.selectbox("Stated level", ["infer"] + [f"C{i} {BLOOM[i]}" for i in range(1, 7)], index=4,
                              key="qc_lvl")
        qc_q = st.text_area("Exam question / task", value="List the steps of the bottom-up troubleshooting approach.",
                            height=70, key="qc_q")
        if st.button("Check alignment", disabled=not (api_key and qc_clo.strip() and qc_q.strip()), key="qc_go"):
            with st.spinner("Two independent AI raters are reading the question..."):
                try:
                    ss["qc_res"] = quick_check(qc_clo, qc_q, cfg, _opt_to_level(qc_lvl))
                    ss["qc_input"] = (qc_clo, qc_q)
                    ss.pop("qc_fix", None)
                except Exception as exc:
                    st.error(safe_error(exc))
        if not api_key:
            st.caption("Add an API key in the sidebar to use the quick check.")
        qr = ss.get("qc_res")
        if qr and ss.get("qc_input") == (qc_clo, qc_q):
            cls, icon = FINDING_STYLE[qr["verdict"]]
            st.markdown(f"<span class='cell {cls}'>{icon} {qr['verdict']}</span> &nbsp; {html.escape(qr['why'])}",
                        unsafe_allow_html=True)
            v = st.columns(3)
            v[0].metric("CLO level", level_name(qr["target"]), help=qr["clo_level_reason"] or qr["target_src"])
            v[1].metric("Rater 1 (sees CLO)", level_name(qr["r1"]), help=qr["r1_reason"])
            v[2].metric("Rater 2 (blind)", level_name(qr["r2"]), help=qr["r2_reason"])
            if qr["verdict"] != "No apparent mismatch detected" and qr["target"]:
                if st.button("✨ Suggest an aligned version", key="qc_fix_btn", disabled=not api_key):
                    with st.spinner("Drafting an item at the CLO's level..."):
                        try:
                            ss["qc_fix"] = quick_rewrite(qc_clo, qc_q, qr["target"], cfg)
                        except Exception as exc:
                            st.error(safe_error(exc))
                fx = ss.get("qc_fix")
                if fx:
                    st.markdown(f"**Suggested ({level_name(_opt_to_level('C' + str(fx.get('bloom_level'))))}, "
                                f"{fx.get('marks', '?')} marks):**")
                    st.markdown(f"<div class='quote'>{html.escape(str(fx.get('revised_item', '')))}</div>",
                                unsafe_allow_html=True)
                    st.caption(str(fx.get("why_it_measures_the_clo", "")) + " · AI draft - faculty to review.")

    source = st.radio("Input", ["Sample course file + job postings",
                                "Upload documents",
                                "Enter CLOs & questions manually"], horizontal=True)
    docs = []
    job_set = None
    if source.startswith("Sample"):
        job_set = st.selectbox("Job postings to compare against", list(JOB_SETS),
                               format_func=lambda k: JOB_SETS[k])
        docs = load_sample_documents(job_set)
        st.caption("Course documents (EE-357 Computer Communication Networks) are **synthetic demo data**, "
                   "written in the style of a Pakistani OBE course file - not a real university course.")
        src = job_sources(job_set)
        ss["job_src"] = src
        prov = pd.DataFrame([{"Job title": r["title"], "Employer": r["employer"], "Location": r["location"],
                              "Posted": r["posted"], "Status": r["status"], "Provenance": r["provenance"],
                              "Source": r["source_url"] or "-", "Accessed": r["accessed"]} for r in src])
        with st.expander("Job posting provenance", expanded=(job_set == "public")):
            st.dataframe(prov, hide_index=True, width="stretch",
                         column_config={"Source": st.column_config.LinkColumn("Source")})
            if job_set == "public":
                st.caption("Selected lines quoted from public postings on ROZEE.PK / getpakjob.com, checked "
                           "accessible on 10 Oct 2026. All four postings are closed; they document what these "
                           "employers asked for, not current vacancies. Two ask for 2-5 years' experience. "
                           "Employers are shown as listed on the job site (some marked 'Unverified' there).")
            else:
                st.caption("Fictional job descriptions written for this demo. Not real employers or vacancies.")
    elif source.startswith("Upload"):
        cols = st.columns(4)
        files = {}
        for col, (cat, label) in zip(cols, CATEGORIES.items()):
            with col:
                up = st.file_uploader(label, type=["pdf", "txt", "md"], accept_multiple_files=True,
                                      key=f"up_{cat}")
                files[cat] = [(f.name, f.getvalue()) for f in up or []]
        if any(files.values()):
            docs = build_documents({k: v for k, v in files.items() if v})
        st.caption("Use only documents you are permitted to share with the selected model API. "
                   "Scanned PDFs need OCR, which this prototype does not perform.")
    else:
        st.caption("Fallback when documents are not available as files. The same AI analysis runs on the text you enter.")
        c1, c2 = st.columns(2)
        clo_text = c1.text_area("CLOs (one per line, with taxonomy level if known)", height=180,
                                placeholder="CLO-1: Explain fundamental routing concepts. (C2)\n"
                                            "CLO-2: Configure dynamic routing in a simulated network. (C3)\n"
                                            "CLO-3: Diagnose and resolve common connectivity failures. (C4)")
        items_text = c2.text_area("Assessment items (one per line; tag CLOs and marks if known)", height=180,
                                  placeholder="Mid Q1 (5 marks) [CLO-1]: Define a routing protocol and state its purpose.\n"
                                              "Lab exam (10 marks) [CLO-2]: Configure OSPF on three routers in Packet Tracer and verify convergence.\n"
                                              "Final Q3 (5 marks) [CLO-3]: List the steps of the bottom-up troubleshooting approach.")
        jd_text = st.text_area("Optional: employer requirements (paste text from a real posting and include its source URL)",
                               height=100)
        docs = build_manual_documents(clo_text, items_text, jd_text)

    if docs:
        st.subheader(f"{len(docs)} documents loaded")
        df = pd.DataFrame([{"ID": d.doc_id, "Type": CATEGORIES[d.category].split(" (")[0],
                            "File": d.name, "Characters": len(d.text),
                            "Warning": d.warning or "-"} for d in docs])
        st.dataframe(df, hide_index=True, width="stretch")
        for d in docs:
            if d.warning:
                st.warning(f"{d.doc_id} - {d.name}: {d.warning}")
        with st.expander("Preview document text"):
            by_doc = {d.doc_id: d for d in docs}
            did = st.selectbox("Document", list(by_doc), format_func=lambda i: f"{i} - {by_doc[i].name}")
            st.text(by_doc[did].text[:6000])

        ok = can_run(docs)
        c1, c2 = st.columns(2)
        do_al = c1.checkbox("CLO–assessment alignment check", value=ok["alignment"], disabled=not ok["alignment"],
                            help="Needs a syllabus/CLO list plus exam questions or graded labs.")
        do_ind = c2.checkbox("Industry skill-gap analysis", value=ok["industry"], disabled=not ok["industry"],
                             help="Needs job descriptions plus at least one course document.")
        do_verify = c1.checkbox("Independent second AI rater (recommended)", value=True, disabled=not do_al,
                                help="A second, blind AI pass re-rates every assessment item without seeing the CLOs. "
                                     "A mismatch is only flagged when both ratings fall below the CLO level.")
        if not ok["alignment"]:
            c1.caption("Add a syllabus/CLOs and assessments to enable.")
        if not ok["industry"]:
            c2.caption("Add job descriptions and a course document to enable.")
        ready = (do_al or do_ind) and bool(api_key)
        if st.button("🚀 Run SkillLeak analysis", type="primary", disabled=not ready):
            with st.status("Analysing documents...", expanded=True) as status:
                bar = st.progress(0.0)

                def progress(msg, frac):
                    status.write(msg)
                    bar.progress(frac)

                try:
                    run = run_scan(docs, cfg, progress, do_alignment=do_al, do_industry=do_ind, verify=do_verify)
                    run["meta"]["input"] = source + (f" / {JOB_SETS[job_set]}" if job_set else "")
                    path = save_run(run)
                    ss.run, ss.run_path, ss.replay = run, path, False
                    status.update(label="Analysis complete - open ② CLO Alignment or ④ Industry Gaps",
                                  state="complete")
                except Exception as exc:  # network, quota, malformed model output...
                    status.update(label="Analysis failed", state="error")
                    st.error(safe_error(exc))
                    st.caption("Tip: use 'Test key & list models' in the sidebar, try again, or load a saved scan.")
        if not api_key:
            st.caption("Add an API key in the sidebar to run a live analysis, or load a saved scan.")
    elif source.startswith("Enter"):
        st.info("Enter at least the CLOs and the assessment items above.")


# ---------------------------------------------------------------------------
# Helpers for result tabs
# ---------------------------------------------------------------------------
def need_run() -> bool:
    if not ss.run:
        st.info("Run an analysis on the ① Documents tab (or load a saved scan from the sidebar) to see results.")
        return False
    return True


def cell(bloom: int, is_taught: bool = False) -> str:
    if bloom == 0:
        return "<span class='cell c-none'>✗ not identified</span>"
    name = BLOOM.get(bloom, "?")
    if is_taught or bloom >= APPLY_LEVEL:
        return f"<span class='cell c-good'>✓ {name}</span>"
    return f"<span class='cell c-warn'>~ {name} only</span>"


def persist():
    if ss.run_path:
        try:
            save_run(ss.run, Path(ss.run_path).name)
        except Exception as exc:
            st.warning(f"Could not save changes to disk: {exc}")


def doc_names(run) -> dict:
    return {d["doc_id"]: d["name"] for d in run["documents"]}


# ---------------------------------------------------------------------------
# Tab 2 - CLO alignment
# ---------------------------------------------------------------------------
with tab_clo:
    if need_run():
        run = ss.run
        al = run.get("alignment")
        if not al:
            st.info("This scan has no CLO–assessment analysis (it was run without a syllabus + assessments, "
                    "or saved by an older version). Run a new analysis with the alignment check enabled.")
        else:
            res = analyze(al)
            sm = res["summary"]
            names = doc_names(run)
            st.markdown("<div class='disclaimer'>ℹ️ Decision support for faculty and OBE coordinators. Findings are "
                        "potential issues to review - not official PEC or accreditation judgements.</div>",
                        unsafe_allow_html=True)
            if al.get("course"):
                st.markdown(f"**Course:** {html.escape(al['course'])}")
            k = st.columns(6)
            k[0].metric("CLOs detected", sm["clos"])
            k[1].metric("Have evidence", sm["with_evidence"], help="CLOs with at least one verified mapped assessment item.")
            k[2].metric("⚠️ Mismatches", sm["mismatch"], help="Potential mismatches: every mapped item is below the stated CLO level.")
            k[3].metric("❔ No evidence", sm["insufficient"], help="Insufficient evidence: no assessment item linked.")
            k[4].metric("🔍 Needs review", sm["review"])
            k[5].metric("Aligned CLOs", f"{sm['aligned']}/{sm['clos']}",
                        help="Number of detected CLOs with 'No apparent mismatch detected' / all detected CLOs. "
                             "A prototype heuristic from the supplied documents - not a PEC score.")

            ag = res["agreement"]
            vs = al.get("verifier", {}).get("status", "off")
            if ag["n"]:
                st.caption(f"🤝 Two independent AI ratings per item: exact agreement {ag['exact']:.0%}, within one level "
                           f"{ag['within_one']:.0%} (n={ag['n']} items). Where they differ, SkillLeak uses the range: "
                           "a mismatch needs both ratings below the CLO level.")
            elif vs.startswith("failed"):
                st.caption("⚠️ The second AI rater failed for this scan, so findings rest on a single rating. "
                           + html.escape(vs[:160]))
            elif vs == "off":
                st.caption("Single AI rating per item (second rater was switched off for this scan).")

            st.subheader("CLO → assessment alignment")
            st.caption("Intended level = level stated in the course file (or set by faculty; AI-inferred only if none "
                       "is stated). Highest demand = most demanding mapped assessment item, rated by the AI on "
                       "Bloom's taxonomy. Aligned-mark share = marks on mapped items at or above the CLO level ÷ all "
                       "mapped marks - an assessment-alignment indicator, not evidence of student mastery. "
                       "Findings use fixed rules - see ⓘ Method.")
            body = []
            for r in res["rows"]:
                cls, icon = FINDING_STYLE[r["finding"]]
                src = {"stated": "stated", "faculty": "faculty-set", "inferred": "AI-inferred",
                       "missing": "missing"}[r["target_src"]]
                share = (f"{r['marks_share_at_level']:.0%}" if r["marks_share_at_level"] is not None else "-") + \
                    (f"<br><span class='badge'>marks on {r['marks_coverage']}</span>"
                     if r.get("marks_coverage") and r["marks_share_at_level"] is not None
                     and r["marks_coverage"].split("/")[0] != r["marks_coverage"].split("/")[1] else "")
                body.append(
                    f"<tr><td style='white-space:nowrap'><b>{html.escape(r['id'])}</b></td><td>{html.escape(r['statement'])}"
                    + (f"<br><span class='badge'>{html.escape(r['plo'])}</span>" if r["plo"] else "") + "</td>"
                    f"<td>{level_name(r['target'])}<br><span class='badge'>{src}</span></td>"
                    f"<td>{level_name(r['max_demand']) if r['max_demand'] else '-'}</td>"
                    f"<td>{r['n_at_level']}/{r['n_items']}</td><td>{share}</td>"
                    f"<td><span class='cell {cls}'>{icon} {r['finding']}</span>"
                    + (f"<br><span class='badge'>severity: {r['severity']}</span>" if r.get("severity") else "")
                    + "</td></tr>")
            st.markdown("<table class='sl-matrix'><tr><th>CLO</th><th>Statement</th><th>Intended level</th>"
                        "<th>Highest demand</th><th>Items at level</th><th>Aligned-mark share</th><th>Finding</th></tr>"
                        + "".join(body) + "</table>", unsafe_allow_html=True)

            mbl = attn.marks_by_level(al)
            if mbl:
                tot = sum(mbl.values())
                low = sum(v for k2, v in mbl.items() if k2 <= 2)
                st.subheader("Where the marks go")
                st.caption(f"Marks of verified assessment items by cognitive demand (higher rating when the two AI "
                           f"raters differ). **{low / tot:.0%} of rated marks are at C1-C2 (recall/understanding)**; "
                           f"{(tot - low) / tot:.0%} at C3 or above. Items without stated marks are not counted.")
                ramp = {1: "#cde2fb", 2: "#9ec5f4", 3: "#6da7ec", 4: "#3987e5", 5: "#256abf", 6: "#184f95"}
                figm = go.Figure(go.Bar(
                    x=[f"C{k2} {BLOOM[k2]}" for k2 in range(1, 7)], y=[mbl.get(k2, 0) for k2 in range(1, 7)],
                    marker_color=[ramp[k2] for k2 in range(1, 7)], marker_line_width=0,
                    text=[f"{mbl.get(k2, 0):g}" if mbl.get(k2) else "" for k2 in range(1, 7)], textposition="outside",
                    hovertemplate="%{x}: %{y} marks<extra></extra>"))
                figm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor="#fcfcfb",
                                   paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Marks", gridcolor="#ecebe7"),
                                   font=dict(color="#0b0b0b"), showlegend=False)
                st.plotly_chart(figm, width="stretch")

            st.subheader("Inspect a CLO")
            order = {f: i for i, f in enumerate(FINDINGS)}
            rows_sorted = sorted(res["rows"], key=lambda r: (order[r["finding"]],
                                                             -((r["target"] or 0) - (r["max_demand"] or 0))))
            by_clo = {r["id"]: r for r in rows_sorted}
            cid = st.selectbox("CLO", list(by_clo), key="clo_pick",
                               format_func=lambda i: f"{FINDING_STYLE[by_clo[i]['finding']][1]} {i} - {by_clo[i]['finding']}")
            r = by_clo[cid]
            st.markdown(f"**{html.escape(r['id'])}:** {html.escape(r['statement'])}")
            st.info(f"**Why:** {r['explanation']}\n\n**Suggested review action:** {r['action']}")
            maps = [m for m in res["mapping_rows"] if m["clo_id"] == cid]
            if not maps:
                st.caption("No assessment item was mapped to this CLO.")
            for m in maps:
                badges = (f"<span class='badge'>{html.escape(m['item_label'])}</span>"
                          f"<span class='badge'>{html.escape(names.get(m['doc_id'], m['doc_id']))}</span>"
                          f"<span class='badge'>marks: {fmt_marks(m['marks'])}</span>"
                          f"<span class='badge'>demand: {html.escape(m['demand_range'])}"
                          f"{' (faculty)' if m['demand_src'] == 'faculty' else ''}"
                          f"{' · raters differ' if m['raters_disagree'] else ''}</span>"
                          f"<span class='badge'>{'explicit CLO tag' if m['basis'] == 'explicit_tag' else 'content match'}"
                          f" · AI confidence: {m['confidence']}</span>")
                if not m["quote_verified"]:
                    badges += "<span class='badge'>⛔ quote not found in source - excluded</span>"
                elif not m["included"]:
                    badges += "<span class='badge'>excluded by faculty</span>"
                else:
                    badges += ("<span class='badge'>✔ meets CLO level</span>" if m["meets_level"]
                               else "<span class='badge'>↓ below CLO level</span>")
                st.markdown(f"{badges}<div class='quote{'' if m['included'] else ' bad'}'>“{html.escape(m['quote'])}”</div>"
                            f"<div class='reason'>Rater 1: {html.escape(m['demand_reason'])}"
                            + (f" · Rater 2 (blind): {html.escape(m['demand_reason_2'])}" if m['demand_reason_2'] else "")
                            + f" · Mapping: {html.escape(m['reason'])}</div>", unsafe_allow_html=True)

            # Aligned assessment generator
            st.markdown("#### ✨ Improve the assessment")
            imp = al.setdefault("improvements", {}).get(cid)
            if st.button("Generate aligned assessment item" if not imp else "↻ Generate another version",
                         key=f"gen_imp_{cid}", disabled=not api_key or r["target"] is None):
                with st.spinner("AI is drafting an assessment item at the CLO's level..."):
                    try:
                        al["improvements"][cid] = generate_aligned_item(al, res, cid, cfg)
                        persist()
                        st.rerun()
                    except Exception as exc:
                        st.error(safe_error(exc))
            if not api_key:
                st.caption("Add an API key in the sidebar to generate an improved item.")
            imp = al.get("improvements", {}).get(cid)
            if imp:
                before = max((m["demand"] for m in maps if m["included"] and m["demand"]), default=None)
                cmp = pd.DataFrame([
                    {"": "Intended CLO level", "Level": level_name(r["target"]), "Source": r["target_src"]},
                    {"": "Current mapped items (highest)", "Level": level_name(before) if before else "none",
                     "Source": "AI rating of supplied items"},
                    {"": "Proposed item", "Level": level_name(_opt_to_level(f"C{imp.get('bloom_level')}")),
                     "Source": "AI estimate - faculty to confirm"},
                ])
                st.dataframe(cmp, hide_index=True, width="stretch")
                st.markdown(f"**Proposed {html.escape(str(imp.get('item_type', 'item')))}** "
                            f"({imp.get('marks', '?')} marks)")
                st.markdown(f"<div class='quote'>{html.escape(str(imp.get('revised_item', '')))}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"**Why it measures the CLO:** {imp.get('why_it_measures_the_clo', '')}")
                if imp.get("marking_guide"):
                    st.dataframe(pd.DataFrame(imp["marking_guide"]), hide_index=True, width="stretch")
                if imp.get("expected_student_evidence"):
                    st.markdown("**Expected student evidence:**\n" +
                                "\n".join(f"- {x}" for x in imp["expected_student_evidence"]))
                st.caption("AI-drafted. The proposed level is the model's estimate; add the item to the paper and "
                           "re-run the analysis to check the new mapping.")

            # Faculty corrections
            with st.expander("✏️ Faculty corrections (fix CLO levels or mappings, then re-check)"):
                ov = al.setdefault("overrides", {"clo_level": {}, "item_level": {}, "exclude": []})
                clo_df = pd.DataFrame([{
                    "CLO": c["id"], "Statement": c["statement"],
                    "Extracted level": level_name(c["stated_level"]) if c["stated_level"]
                    else (f"{level_name(c['inferred_level'])} (inferred)" if c["inferred_level"] else "none"),
                    "Faculty level": (f"C{ov['clo_level'][c['id']]} {BLOOM[int(ov['clo_level'][c['id']])]}"
                                      if ov["clo_level"].get(c["id"]) else ""),
                } for c in al["clos"]])
                clo_ed = st.data_editor(clo_df, hide_index=True, width="stretch", key="clo_ed",
                                        disabled=["CLO", "Statement", "Extracted level"],
                                        column_config={"Faculty level": st.column_config.SelectboxColumn(
                                            "Faculty level", options=LEVEL_OPTS)})
                map_df = pd.DataFrame([{
                    "Key": m["key"], "CLO": m["clo_id"], "Item": m["item_label"],
                    "AI demand": m["demand_range"] if m["demand_src"] != "faculty" else "",
                    "Faculty demand": (f"C{ov['item_level'][m['item_id']]} {BLOOM[int(ov['item_level'][m['item_id']])]}"
                                       if ov["item_level"].get(m["item_id"]) else ""),
                    "Include": m["key"] not in ov["exclude"],
                    "Item ID": m["item_id"],
                } for m in res["mapping_rows"]])
                map_ed = st.data_editor(map_df, hide_index=True, width="stretch", key="map_ed",
                                        disabled=["Key", "CLO", "Item", "AI demand", "Item ID"],
                                        column_config={"Faculty demand": st.column_config.SelectboxColumn(
                                            "Faculty demand", options=LEVEL_OPTS),
                                            "Key": None})
                b1, b2 = st.columns(2)
                if b1.button("Apply corrections and re-check", type="primary"):
                    new_ov = {"clo_level": {}, "item_level": {}, "exclude": []}
                    for _, row in clo_ed.iterrows():
                        lv = _opt_to_level(row["Faculty level"])
                        if lv:
                            new_ov["clo_level"][row["CLO"]] = lv
                    for _, row in map_ed.iterrows():
                        lv = _opt_to_level(row["Faculty demand"])
                        if lv:
                            new_ov["item_level"][row["Item ID"]] = lv
                        if not bool(row["Include"]):
                            new_ov["exclude"].append(row["Key"])
                    al["overrides"] = new_ov
                    persist()
                    st.rerun()
                if b2.button("Reset corrections"):
                    al["overrides"] = {"clo_level": {}, "item_level": {}, "exclude": []}
                    persist()
                    st.rerun()
                st.caption("Corrections change only the rule-based findings; the AI is not re-run. "
                           "Corrections are saved with the scan.")

            c1, c2 = st.columns(2)
            with c1:
                with st.expander(f"Assessment items not mapped to any CLO ({len(res['unmapped_items'])})"):
                    for it in res["unmapped_items"]:
                        st.markdown(f"<span class='badge'>{html.escape(it['label'])}</span>"
                                    f"<span class='badge'>{level_name(it['demand_level'])}</span>"
                                    f"<div class='quote'>“{html.escape(it['quote'])}”</div>", unsafe_allow_html=True)
                    if not res["unmapped_items"]:
                        st.caption("None.")
            with c2:
                with st.expander(f"Missing information & rejected AI quotes "
                                 f"({len(al.get('missing_information', [])) + len(res['rejected_items'])})"):
                    for x in al.get("missing_information", []):
                        st.markdown(f"- {x}")
                    for it in res["rejected_items"]:
                        st.markdown(f"<span class='badge'>{html.escape(it['label'])}</span>"
                                    f"<div class='quote bad'>“{html.escape(it['quote'])}”</div>",
                                    unsafe_allow_html=True)
                    if not al.get("missing_information") and not res["rejected_items"]:
                        st.caption("None reported.")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ CLO alignment report (Markdown)", alignment_markdown(run),
                               file_name="skillleak_clo_alignment.md", mime="text/markdown",
                               width="stretch")
            try:
                d2.download_button("⬇️ Course-file pack (Excel)", course_file_pack(run, sources=ss.get("job_src")),
                                   file_name="skillleak_course_file_pack.xlsx", width="stretch",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                d2.caption(f"Excel export unavailable: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 - Attainment & PLO (deterministic; works offline on saved scans)
# ---------------------------------------------------------------------------
with tab_att:
    st.markdown("<div class='disclaimer'>ℹ️ Attainment is computed from the marks sheet with the thresholds you set "
                "(policies differ between institutions). No AI is used on this tab; it cross-checks the numbers "
                "against the CLO alignment findings.</div>", unsafe_allow_html=True)
    msrc = st.radio("Marks sheet", ["Sample: EE-357 marks (synthetic, 40 students)", "Upload CSV"], horizontal=True)
    marks_bytes = None
    if msrc.startswith("Sample"):
        marks_bytes = (Path(__file__).parent / "sample_data" / "marks_ee357_synthetic.csv").read_bytes()
        st.caption("Synthetic marks generated for this demo (fictional students). Not real student data.")
    else:
        up = st.file_uploader("Marks sheet (.csv): first row = question names, then a CLO row and a MAX row",
                              type=["csv"], key="marks_up")
        st.download_button("Download CSV template", attn.TEMPLATE, file_name="marks_template.csv", mime="text/csv")
        st.caption("Use anonymised student IDs only. Do not upload names or other personal data.")
        if up:
            marks_bytes = up.getvalue()
    t1, t2 = st.columns(2)
    s_thr = t1.slider("Student attains a CLO at ≥ (% of that CLO's marks)", 30, 90, 50, 5) / 100
    c_thr = t2.slider("CLO counted as attained when ≥ (% of students)", 30, 100, 60, 5) / 100

    att_rows, plo_rows = None, None
    if marks_bytes:
        try:
            sheet, warns = attn.parse_marks_csv(marks_bytes)
            for w in warns:
                st.caption(f"⚠️ {w}")
            att_rows = attn.compute_attainment(sheet, s_thr, c_thr)
        except Exception as exc:
            st.error(f"Could not read the marks sheet: {exc}")
    al_rows = None
    if ss.run and ss.run.get("alignment"):
        al_rows = analyze(ss.run["alignment"])["rows"]
    if att_rows is not None:
        rows_i = attn.inflation_check(att_rows, al_rows)
        if not al_rows:
            st.info("Run (or load) a scan with the CLO alignment check to add validity flags to these numbers.")
        n_inf = sum(1 for a in rows_i if a["flag"] == "Potential attainment-validity concern")
        k = st.columns(4)
        k[0].metric("CLOs with marks", len(rows_i))
        k[1].metric("Reported as attained", sum(1 for a in rows_i if a["attained"]))
        k[2].metric("⚠️ Validity concerns", n_inf,
                    help="Attained under your thresholds, but the supplied assessment evidence for the CLO sits below "
                         "its stated level. A prompt to check - not proof that attainment is inflated.")
        k[3].metric("Students", rows_i[0]["n_students"] if rows_i else 0)
        if n_inf:
            worst = max((a for a in rows_i if a["flag"] == "Potential attainment-validity concern"),
                        key=lambda a: (a["intended"] or 0) - (a["measured_up_to"] or 0))
            st.warning(f"**{worst['clo']}** is attained by **{worst['class_attainment']:.0%}** of students under your "
                       f"thresholds - but the supplied assessments measure it only up to "
                       f"**{level_name(worst['measured_up_to'])}** while the CLO states **{level_name(worst['intended'])}**. "
                       "Unless it is assessed at its level elsewhere, this figure may not reflect the intended competence.")
        FLAG_STYLE = {"Potential attainment-validity concern": ("f-bad", "⚠️"), "Not attained + level mismatch": ("f-bad", "⛔"),
                      "Validity unclear": ("f-warn", "🔍"), "Not attained": ("f-warn", "📉"),
                      "Not checked": ("f-muted", "❔"), "": ("f-good", "✅")}
        body = []
        for a in rows_i:
            cls, icon = FLAG_STYLE.get(a["flag"], ("f-muted", "•"))
            body.append(
                f"<tr><td style='white-space:nowrap'><b>{html.escape(a['clo'])}</b></td>"
                f"<td>{html.escape(', '.join(a['questions']))}</td>"
                f"<td>{level_name(a['intended']) if a['intended'] else '-'}</td>"
                f"<td>{level_name(a['measured_up_to']) if a['measured_up_to'] else '-'}</td>"
                f"<td>{a['avg_score']:.0%}</td><td><b>{a['class_attainment']:.0%}</b> "
                f"{'(attained)' if a['attained'] else '(not attained)'}</td>"
                f"<td><span class='cell {cls}'>{icon} {html.escape(a['flag'] or 'OK')}</span>"
                f"<div class='reason' style='margin:4px 0 0 0'>{html.escape(a['note'])}</div></td></tr>")
        st.markdown("<table class='sl-matrix'><tr><th>CLO</th><th>Questions</th><th>Intended</th><th>Measured up to</th>"
                    "<th>Avg score</th><th>Class attainment</th><th>Validity check</th></tr>" + "".join(body)
                    + "</table>", unsafe_allow_html=True)
        st.caption(f"Class attainment = share of students scoring ≥ {s_thr:.0%} of the CLO's marks; a CLO counts as "
                   f"attained when that share is ≥ {c_thr:.0%}.")

    if al_rows:
        plo_rows = attn.plo_rollup(al_rows, att_rows)
        st.subheader("PLO assessment-evidence coverage")
        st.caption("For each PLO: the CLOs mapped to it in this course, the highest cognitive level any verified "
                   "assessment reaches for them, and how many have alignment problems. This covers one course's "
                   "documents only - it is not a programme-level PLO attainment result.")
        EV = {"Supported": ("f-good", "✅"), "Partial": ("f-warn", "🟡"), "Weak": ("f-bad", "⚠️")}
        body = []
        for p in plo_rows:
            cls, icon = EV[p["evidence"]]
            body.append(
                f"<tr><td><b>{html.escape(p['plo_label'])}</b></td><td>{html.escape(', '.join(p['clos']))}</td>"
                f"<td>{level_name(p['highest_intended'])}</td><td>{level_name(p['highest_evidenced'])}</td>"
                f"<td>{'-' if p['avg_attainment'] is None else format(p['avg_attainment'], '.0%')}</td>"
                f"<td><span class='cell {cls}'>{icon} {p['evidence']}</span>"
                + (f"<br><span class='badge'>weak: {', '.join(p['weak_clos'])}</span>" if p["weak_clos"] else "")
                + "</td></tr>")
        st.markdown("<table class='sl-matrix'><tr><th>PLO</th><th>CLOs</th><th>Highest intended</th>"
                    "<th>Highest evidenced</th><th>Avg attainment</th><th>Evidence</th></tr>" + "".join(body)
                    + "</table>", unsafe_allow_html=True)
    if ss.run and (att_rows or al_rows):
        try:
            st.download_button("⬇️ Course-file pack with attainment & PLO evidence coverage (Excel)",
                               course_file_pack(ss.run, attainment_rows=attn.inflation_check(att_rows, al_rows)
                                                if att_rows else None, plo_rows=plo_rows,
                                                sources=ss.get("job_src")),
                               file_name="skillleak_course_file_pack.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.caption(f"Excel export unavailable: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 - industry gap map
# ---------------------------------------------------------------------------
def need_industry(run) -> bool:
    if not run.get("competencies"):
        st.info("This scan has no industry skill-gap analysis. Add job descriptions on the ① Documents tab "
                "and run again with 'Industry skill-gap analysis' enabled.")
        return False
    return True


with tab_map:
    if need_run() and need_industry(ss.run):
        run = ss.run
        rows = score_run(run)
        n_jd = sum(1 for d in run["documents"] if d["category"] == "industry")
        counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_COLOR}
        al = run.get("alignment")

        st.markdown(f"**Job family:** {html.escape(run.get('job_family', ''))} &nbsp;·&nbsp; "
                    f"{n_jd} job descriptions &nbsp;·&nbsp; {len(rows)} employer-required competencies "
                    f"(with a verified employer quote)")
        k = st.columns(5)
        k[0].metric("Evidence coverage", f"{coverage_index(rows):.0f}%",
                    help="Demand-weighted share of Taught/Practised/Assessed evidence found in the supplied documents. "
                         "Prototype indicator.")
        k[1].metric("✅ Covered", counts["Covered"])
        k[2].metric("👻 Partially covered", counts["Phantom competence risk"] + counts["Practised but not in syllabus"],
                    help="Some evidence found, but not taught, practised and assessed at Apply level or above.")
        k[3].metric("🚫 Not identified", counts["Industry blind spot"],
                    help="Required in the postings; not identified anywhere in the supplied course material.")
        k[4].metric("AI quotes rejected", f"{run['meta']['quotes_rejected']}/{run['meta']['quotes_total']}",
                    help="Quotes the AI produced that could not be found in the source documents. They are excluded.")
        if counts["Cannot fully evaluate"]:
            st.warning(f"{counts['Cannot fully evaluate']} competencies cannot be fully evaluated because some "
                       "document types (syllabus, labs or assessments) were not supplied.")

        st.subheader("Skill Leakage Map")
        st.caption("✓ verified evidence at Apply level or higher · ~ evidence only at Remember/Understand level · "
                   "✗ not identified in the provided course material (which is not proof it is never taught).")
        body = []
        for r in rows:
            sc = STATUS_COLOR[r["status"]]
            comp = next(c for c in run["competencies"] if c["id"] == r["id"])
            linked = sorted({cid for e in comp["course_evidence"] if e["category"] == "syllabus" and is_verified(e)
                             for cid in clo_for_text(al, e["quote"])})
            body.append(
                f"<tr><td><b>{html.escape(r['name'])}</b>"
                + "".join(f" <span class='badge'>{c}</span>" for c in linked) + "</td>"
                f"<td>{r['jd_count']}/{n_jd} postings · D={r['D']}</td>"
                f"<td>{cell(r['bloom_syllabus'], True)}</td><td>{cell(r['bloom_practice'])}</td>"
                f"<td>{cell(r['bloom_assessment'])}</td>"
                f"<td><span style='color:{sc}'>●</span> {STATUS_ICON[r['status']]} {r['status']}</td>"
                f"<td><b>{r['priority']}</b></td></tr>")
        st.markdown("<table class='sl-matrix'><tr><th>Competency employers require (linked CLOs)</th><th>Demand</th>"
                    "<th>Taught</th><th>Practised</th><th>Assessed</th><th>Status</th><th>Gap priority</th></tr>"
                    + "".join(body) + "</table>", unsafe_allow_html=True)

        gaps = [r for r in rows if r["priority"] > 0]
        if gaps:
            st.subheader("Where to act first")
            st.caption("Gap priority = D × (3 − (T + P + A)), range 0-15. Computed in Python, not by the AI. "
                       f"D = ceil(5 × share of the {n_jd} analysed postings that mention the skill) - a relative weight "
                       "within this small sample, not a measure of national labour-market demand.")
            g = list(reversed(gaps))
            fig = go.Figure()
            for status in STATUS_COLOR:
                sub = [r for r in g if r["status"] == status]
                if not sub:
                    continue
                fig.add_bar(
                    y=[r["name"] for r in sub], x=[r["priority"] for r in sub], orientation="h",
                    name=f"{STATUS_ICON[status]} {status}", marker_color=STATUS_COLOR[status],
                    marker_line_width=0,
                    customdata=[[r["D"], r["T"], r["P"], r["A"]] for r in sub],
                    hovertemplate="<b>%{y}</b><br>Priority %{x}<br>D=%{customdata[0]} · T=%{customdata[1]} "
                                  "· P=%{customdata[2]} · A=%{customdata[3]}<extra></extra>")
            fig.update_layout(height=max(260, 38 * len(g) + 90), barmode="overlay", bargap=0.35,
                              margin=dict(l=10, r=20, t=10, b=40), plot_bgcolor="#fcfcfb",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0b0b0b"),
                              xaxis=dict(title="Gap priority", range=[0, 15.5], gridcolor="#ecebe7",
                                         dtick=3, zeroline=False),
                              yaxis=dict(categoryorder="array", categoryarray=[r["name"] for r in g]),
                              legend=dict(orientation="h", y=-0.18, x=0))
            st.plotly_chart(fig, width="stretch")

        st.download_button("⬇️ Download full evidence report (Markdown)", run_markdown(run),
                           file_name="skillleak_report.md", mime="text/markdown")


# ---------------------------------------------------------------------------
# Tab 4 - evidence explorer
# ---------------------------------------------------------------------------
with tab_ev:
    if need_run() and need_industry(ss.run):
        run = ss.run
        rows = score_run(run)
        comps = {c["id"]: c for c in run["competencies"]}
        names = doc_names(run)
        al = run.get("alignment")
        if not rows:
            st.info("No competency had a verified employer quote.")
        else:
            by_row = {x["id"]: x for x in rows}
            rid = st.selectbox("Competency", list(by_row),
                               format_func=lambda i: f"{STATUS_ICON[by_row[i]['status']]} {by_row[i]['name']}  "
                                                     f"(priority {by_row[i]['priority']})")
            r = by_row[rid]
            c = comps[r["id"]]
            st.markdown(f"**{html.escape(c['name'])}** - {html.escape(c.get('description', ''))}")
            st.info(f"**Diagnosis:** {r['diagnosis']}")
            st.markdown(f"`D={r['D']}` `T={r['T']}` `P={r['P']}` `A={r['A']}` → "
                        f"**priority = {r['D']} × (3 − {r['evidence_score']}) = {r['priority']}**")

            def show(evs, with_bloom=True):
                ok = [e for e in evs if is_verified(e)]
                if not ok:
                    st.markdown("<div class='reason'>Not identified in the provided material.</div>",
                                unsafe_allow_html=True)
                for e in ok:
                    badges = f"<span class='badge'>{html.escape(names.get(e['doc_id'], e['doc_id']))}</span>"
                    if with_bloom:
                        lvl = e["bloom_level"]
                        badges += f"<span class='badge'>Bloom {lvl} · {BLOOM[lvl]}</span>"
                    for cid in (clo_for_text(al, e["quote"]) if e.get("category") == "syllabus" else []):
                        badges += f"<span class='badge'>{cid}</span>"
                    badges += (f"<span class='badge'>{'✔ quote found in source' if e['verified'] == 'exact' else '≈ quote near-exact in source'}"
                               "</span>")
                    st.markdown(f"{badges}<div class='quote'>“{html.escape(e['quote'])}”</div>"
                                + (f"<div class='reason'>{html.escape(e.get('reason', ''))}</div>" if e.get("reason") else ""),
                                unsafe_allow_html=True)

            cols = st.columns(4)
            sections = [("🏭 Employers require", c["industry_evidence"], False),
                        ("📘 Taught", [e for e in c["course_evidence"] if e["category"] == "syllabus"], True),
                        ("🛠️ Practised", [e for e in c["course_evidence"] if e["category"] == "practice"], True),
                        ("📝 Assessed", [e for e in c["course_evidence"] if e["category"] == "assessment"], True)]
            for col, (title, evs, wb) in zip(cols, sections):
                with col:
                    st.markdown(f"#### {title}")
                    show(evs, wb)

            rejected = [e for e in c["industry_evidence"] + c["course_evidence"] if not is_verified(e)]
            with st.expander(f"🛡️ AI claims rejected by source verification ({len(rejected)})"):
                st.caption("These quotes were produced by the AI but could not be found in the named document, "
                           "so they do not count towards any score.")
                for e in rejected:
                    st.markdown(f"<span class='badge'>{html.escape(e['doc_id'])}</span>"
                                f"<div class='quote bad'>“{html.escape(e['quote'])}”</div>", unsafe_allow_html=True)
                if not rejected:
                    st.caption("None for this competency.")


# ---------------------------------------------------------------------------
# Tab 5 - Gap-to-Lab generator with teacher review
# ---------------------------------------------------------------------------
def render_lab(lab: dict):
    st.markdown(f"### 🧪 {html.escape(lab.get('title', 'Practical lab'))}")
    m = st.columns(3)
    m[0].markdown(f"**Bloom target**<br>{html.escape(str(lab.get('bloom_target', '')))}", unsafe_allow_html=True)
    m[1].markdown(f"**Duration**<br>{lab.get('duration_minutes', '?')} min", unsafe_allow_html=True)
    m[2].markdown(f"**Tools**<br>{html.escape(', '.join(map(str, lab.get('tools', []))))}", unsafe_allow_html=True)
    st.markdown(f"**Learning outcome:** {lab.get('learning_outcome', '')}")
    st.markdown(f"**Scenario:** {lab.get('scenario', '')}")
    a, b = st.columns(2)
    with a:
        if lab.get("instructor_setup"):
            st.markdown("**Instructor setup**")
            st.markdown("\n".join(f"- {s}" for s in lab["instructor_setup"]))
        st.markdown("**Student tasks**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(lab.get("student_tasks", []), 1)))
    with b:
        st.markdown("**Evidence students submit**")
        st.markdown("\n".join(f"- {s}" for s in lab.get("evidence_to_submit", [])))
        st.markdown(f"**How this closes the gap:** {lab.get('how_it_closes_gap', '')}")
        if lab.get("assessment_variant"):
            st.markdown(f"**Matching assessment question:** {lab['assessment_variant']}")
    rub = lab.get("rubric", [])
    if rub:
        total = sum(int(x.get("points", 0) or 0) for x in rub if str(x.get("points", "0")).isdigit())
        st.markdown(f"**Marking rubric** (total {total} points)")
        st.dataframe(pd.DataFrame([{"Criterion": x.get("criterion"), "Points": x.get("points"),
                                    "Excellent": x.get("excellent"), "Satisfactory": x.get("satisfactory"),
                                    "Needs improvement": x.get("needs_improvement")} for x in rub]),
                     hide_index=True, width="stretch")


with tab_lab:
    if need_run() and need_industry(ss.run):
        run = ss.run
        gaps = [r for r in score_run(run) if r["priority"] > 0]
        if not gaps:
            st.success("No evidence gaps found for the supplied documents.")
        else:
            by_gap = {x["id"]: x for x in gaps}
            gid = st.selectbox("Choose a gap to repair (sorted by priority)", list(by_gap),
                               format_func=lambda i: f"{STATUS_ICON[by_gap[i]['status']]} {by_gap[i]['name']} - "
                                                     f"priority {by_gap[i]['priority']}")
            r = by_gap[gid]
            st.info(f"**Why it is flagged:** {r['diagnosis']}")
            lab = run["labs"].get(r["id"])
            can_gen = bool(api_key)
            if st.button("✨ Generate practical lab + rubric" if not lab else "↻ Generate a new version",
                         type="primary", disabled=not can_gen):
                with st.spinner("AI is designing a lab grounded in the employer quotes and the diagnosis..."):
                    try:
                        run["labs"][r["id"]] = generate_lab(run, r["id"], cfg)
                        persist()
                        st.rerun()
                    except Exception as exc:
                        st.error(safe_error(exc))
            if not can_gen:
                st.caption("Add an API key in the sidebar to generate labs.")

            lab = run["labs"].get(r["id"])
            if lab:
                if lab.get("_meta", {}).get("revised_with_feedback"):
                    st.caption("🔁 This version was revised using teacher feedback.")
                render_lab(lab)

                st.divider()
                st.markdown("#### 👩‍🏫 Teacher review (human in the loop)")
                rv = lab.setdefault("review", {"status": "Pending review", "note": ""})
                opts = ["Pending review", "Approved", "Changes requested", "Rejected"]
                c1, c2 = st.columns([1, 2])
                with c1:
                    status = st.radio("Decision", opts, index=opts.index(rv.get("status", opts[0]))
                                      if rv.get("status") in opts else 0, key=f"rv_{r['id']}")
                with c2:
                    note = st.text_area("Reviewer note / requested changes", value=rv.get("note", ""),
                                        key=f"note_{r['id']}",
                                        placeholder="e.g. Use GNS3 instead of Packet Tracer; shorten to 2 hours")
                b1, b2, b3 = st.columns(3)
                if b1.button("💾 Save review", width="stretch"):
                    lab["review"] = {"status": status, "note": note}
                    persist()
                    st.success(f"Saved: {status}")
                if b2.button("🔁 Revise with this feedback", width="stretch",
                             disabled=not (can_gen and note.strip())):
                    with st.spinner("Revising the lab with the teacher's feedback..."):
                        try:
                            new = generate_lab(run, r["id"], cfg, feedback=note.strip())
                            run["labs"][r["id"]] = new
                            persist()
                            st.rerun()
                        except Exception as exc:
                            st.error(safe_error(exc))
                b3.download_button("⬇️ Download lab (Markdown)", lab_markdown(lab),
                                   file_name=f"lab_{r['id']}.md", mime="text/markdown",
                                   width="stretch")


# ---------------------------------------------------------------------------
# Tab 6 - Student mode
# ---------------------------------------------------------------------------
with tab_student:
    st.caption("For students: turn a skill gap found in your course material into a self-practice plan "
               "using free tools.")
    if need_run() and need_industry(ss.run):
        run = ss.run
        gaps = [r for r in score_run(run) if r["priority"] > 0]
        if not gaps:
            st.success("No skill gaps found for the supplied documents.")
        else:
            by_gap = {x["id"]: x for x in gaps}
            sid = st.selectbox("Skill to work on", list(by_gap), key="student_pick",
                               format_func=lambda i: f"{by_gap[i]['name']}  ({by_gap[i]['jd_count']} postings ask for it)")
            packs = run.setdefault("student_packs", {})
            if st.button("🎓 Create my practice plan" if sid not in packs else "↻ Create a new plan",
                         type="primary", disabled=not api_key):
                with st.spinner("Building a practice plan grounded in the employer postings..."):
                    try:
                        packs[sid] = generate_student_pack(run, sid, cfg)
                        persist()
                        st.rerun()
                    except Exception as exc:
                        st.error(safe_error(exc))
            if not api_key:
                st.caption("Add an API key in the sidebar to create a practice plan.")
            p = packs.get(sid)
            if p:
                st.markdown(f"### 🎯 {html.escape(str(p.get('skill', '')))}")
                st.markdown(f"**Why it matters:** {p.get('why_it_matters', '')}")
                st.markdown(f"**Exercise:** {p.get('exercise_title', '')} · ⏱️ {p.get('time_needed', '')} · "
                            f"🧰 {', '.join(map(str, p.get('tools', [])))}")
                a, b = st.columns(2)
                with a:
                    st.markdown("**Steps**")
                    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(p.get("steps", []), 1)))
                with b:
                    st.markdown("**Expected output**")
                    st.markdown("\n".join(f"- {s}" for s in p.get("expected_output", [])))
                    st.markdown("**Self-check**")
                    for i, s in enumerate(p.get("self_check", [])):
                        st.checkbox(str(s), key=f"sc_{sid}_{i}")
                    st.markdown("**Next steps**")
                    st.markdown("\n".join(f"- {s}" for s in p.get("next_steps", [])))
                st.download_button("⬇️ Download practice plan (Markdown)", student_markdown(p),
                                   file_name=f"practice_{sid}.md", mime="text/markdown")
                st.caption("AI-generated study plan. Employer reasons are based only on the quoted postings.")


# ---------------------------------------------------------------------------
# Tab 8 - Validation benchmark
# ---------------------------------------------------------------------------
with tab_val:
    st.markdown("### How accurate is the AI's Bloom rating?")
    st.caption(f"Test set: {benchmark.SOURCE}. The repository does not document who labelled the questions or how, "
               "so treat these as labels of unverified annotation provenance. Labels use the original (1956) taxonomy and "
               "are mapped approximately to the revised scale (Knowledge→C1 … Analysis→C4, Evaluation→C5, Synthesis→C6); "
               "the original order places Synthesis before Evaluation, so C5/C6 results are the least comparable. "
               "Separately, a published study of a similar 600-sentence set reports zero-shot LLMs at about 0.72-0.73 "
               "accuracy with most errors between adjacent levels (arXiv:2511.10903) - a different evaluation, cited for "
               "context only. The keyword baseline and the AI rater are scored on the same stratified questions.")
    results = benchmark.saved_results()
    latest = {}
    for r in results:
        latest.setdefault((r["rater"], r["n"]), r)
    if latest:
        tbl = pd.DataFrame([{"Rater": r["label"], "Questions": r["n"], "Exact accuracy": f"{r['accuracy']:.1%}",
                             "Within one level": f"{r['within_one']:.1%}", "Macro-F1": f"{r['macro_f1']:.2f}",
                             "Run at": r["created_at"]} for r in latest.values()])
        st.dataframe(tbl, hide_index=True, width="stretch")
        pick = st.selectbox("Confusion matrix for", list(latest),
                            format_func=lambda k: f"{latest[k]['label']} - {k[1]} questions")
        r = latest[pick]
        lv = [f"C{i} {BLOOM[i]}" for i in range(1, 7)]
        figc = go.Figure(go.Heatmap(z=r["confusion"], x=lv, y=lv, colorscale=[[0, "#f7f9fc"], [1, "#184f95"]],
                                    text=r["confusion"], texttemplate="%{text}", showscale=False,
                                    hovertemplate="Expert: %{y}<br>Predicted: %{x}<br>%{z} questions<extra></extra>"))
        figc.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Predicted",
                           yaxis_title="Expert label", yaxis_autorange="reversed", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figc, width="stretch")
        if r.get("examples"):
            with st.expander("Examples of disagreements"):
                st.dataframe(pd.DataFrame([{"Question": e["text"], "Expert": e["gold"], "Predicted": e["pred"]}
                                           for e in r["examples"]]), hide_index=True, width="stretch")
    else:
        st.info("No benchmark results yet.")
    st.markdown("#### Run the AI rater on the benchmark")
    per = st.select_slider("Questions per level", options=[5, 10, 20, 50], value=10)
    st.caption(f"{per * 6} questions, stratified, fixed seed - the same questions every time. Uses your API key "
               f"(about {max(1, per * 6 // 30)} model call(s)). Results are saved and shown above.")
    if st.button("▶ Run AI rater benchmark", disabled=not api_key):
        with st.spinner("Rating benchmark questions..."):
            try:
                rr = benchmark.run("ai", per, cfg)
                kb = benchmark.run("keyword", per, save=True)
                st.success(f"AI rater: {rr['accuracy']:.1%} exact, {rr['within_one']:.1%} within one level · "
                           f"keyword baseline on the same questions: {kb['accuracy']:.1%} / {kb['within_one']:.1%}")
                st.rerun()
            except Exception as exc:
                st.error(safe_error(exc))
    if not api_key:
        st.caption("Add an API key in the sidebar to run the AI rater. The keyword baseline needs no key and is "
                   "already included above.")


# ---------------------------------------------------------------------------
# Tab 7 - method and limitations
# ---------------------------------------------------------------------------
with tab_method:
    st.markdown("""
### Part A - CLO–assessment alignment (Outcome-Based Education support)
1. **Extraction (AI).** The LLM extracts each CLO exactly as written, the taxonomy level stated in the course file
   (C1-C6), the mapped PLO, and every assessment item (exam/quiz question, lab exam, graded lab task) with marks.
2. **Cognitive demand (AI).** Each item is rated on Bloom's revised taxonomy by **what the student must do**,
   not its topic: "List the troubleshooting steps" is *Remember (C1)*; diagnosing an unknown fault is *Analyze (C4)*.
3. **Mapping (AI).** Items are mapped to CLOs (many-to-many). Explicit CLO tags in the paper are used where present;
   otherwise content matching, with the model's own confidence label (high/medium/low - not a calibrated probability).
4. **Verification (code).** Every CLO statement and item quote is checked against the source text; unmatched items are
   excluded.
5. **Findings (code, fixed rules), per CLO, using all mapped items together:**
   - ⚠️ **Potential mismatch** - every mapped item demands less than the *stated* (or faculty-set) CLO level.
   - 🔍 **Needs review** - the level was only inferred, the two ratings straddle the level, only low-confidence
     matches reach it, the **aligned-mark share** is under 50%, or no level is available.
   - ❔ **Insufficient evidence** - no verified assessment item could be linked to the CLO.
   - ✅ **No apparent mismatch detected** - mapped items reach the CLO level (both ratings) and carry at least half
     of the CLO's stated marks.
   - **Aligned-mark share** = marks on mapped items at or above the CLO level ÷ all mapped marks (items with stated
     marks). An assessment-alignment indicator - not evidence that students mastered the CLO.
   - **Quote verified** means the text exists in the source; the level and mapping remain AI judgements for review.
6. **Alignment indicator** = CLOs with "No apparent mismatch detected" ÷ all detected CLOs. A prototype heuristic,
   **not a PEC score or accreditation rating**.
7. **Faculty corrections** - levels and mappings can be corrected; findings are recomputed instantly by the rules.

### Two AI ratings = a disagreement detector, not proof of accuracy
Published work finds zero-shot LLMs reach only about 0.72-0.73 accuracy on Bloom classification, with most errors
between adjacent levels (arXiv:2511.10903). Each verified item is therefore rated twice: once in context (with the
CLO and full documents) and once by a **blind** pass that sees only the extracted item text. Both passes use the same
model, so their errors can be correlated - agreement shows *consistency*, not correctness. The rule is conservative:
a **Potential mismatch** needs *both* ratings below the CLO level; if they straddle it, the CLO goes to
**Needs review**; a one-level gap is **low severity**. Each rating's reason is shown and faculty can override it.
Note that a question's wording alone may not reveal its true demand (e.g. whether fault data is supplied), which is
another reason the final call stays with faculty.

### Attainment-validity check (③ Attainment & PLO - no AI)
From a marks sheet (CLO tag + maximum marks per question), SkillLeak computes CLO attainment with your thresholds,
then cross-checks each CLO against its alignment finding. A CLO that is **attained under the thresholds while the
supplied assessment evidence sits below its stated level** gets a *potential attainment-validity concern*. This is a
prompt to check, not proof: the CLO may be assessed at its level in a lab or project that was not supplied.
**PLO assessment-evidence coverage** lists, per PLO, the mapped CLOs, the highest level any verified assessment
reaches and which CLOs have alignment problems - for one course only, not programme-level PLO attainment.

### Part B - Industry skill-gap analysis
1. **Competency extraction (AI)** from job postings, with a verbatim quote per posting.
2. **Evidence mapping (AI)** in syllabus, labs and assessments, with Bloom level of what students must do.
3. **Verification (code)** - unmatched quotes are rejected and never counted.
4. **Scoring (code)**: D = ceil(5 × share of the analysed postings with a verified quote) - relative to that small
   sample only (one job set is analysed at a time; synthetic and real postings are never mixed), not national demand; T = syllabus evidence;
   P/A = practice/assessment evidence at Bloom ≥ Apply (3). **Gap priority = D × (3 − (T + P + A))**.
5. **Gap-to-Lab and Student Mode (AI + human)** - AI drafts a lab/rubric or a self-practice plan; teachers review.

### Status labels (industry)
✅ Covered · 👻 Phantom competence risk (in syllabus, not practised/assessed at Apply level) ·
🟡 Practised but not in syllabus · 🚫 Not identified in the provided material · ❔ Cannot fully evaluate (document type missing)

### Limitations (please read)
- **Not an accreditation tool of record.** SkillLeak does not certify or guarantee compliance with PEC, HEC or any
  accreditation requirement; it flags potential issues for faculty review.
- Findings reflect **only the documents supplied**. "Not identified" is not proof a skill is never taught.
- Bloom ratings and mappings come from an LLM and can be wrong; every one is shown with its quote and reason and can
  be corrected by faculty.
- Scores are prototype heuristics, not validated measures. Model confidence labels are not calibrated probabilities.
- Scanned PDFs need OCR (not implemented). Long documents are truncated to 15,000 characters each.
- Public job postings are dated, closed listings used as evidence of documented requirements, not live demand.

### Data & responsible AI
Course documents in the sample are **synthetic demo data**. Public job excerpts are quoted with source URLs and access
dates. API keys come from environment variables / Streamlit secrets or the sidebar and are never written to disk.
Uploaded documents are sent only to the selected model API.
""")
