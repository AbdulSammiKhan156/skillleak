"""Document loading and text extraction.

Supports PDF (via PyMuPDF), plain text and Markdown. Scanned PDFs (images only)
yield no text; we report that instead of failing silently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# Document categories map onto the four evidence types.
CATEGORIES = {
    "syllabus": "Taught (syllabus / course outline)",
    "practice": "Practised (labs, assignments, projects)",
    "assessment": "Assessed (exams, quizzes, rubrics)",
    "industry": "Industry required (job descriptions)",
}

MAX_CHARS_PER_DOC = 15000  # keeps prompts within cheap-model context limits


@dataclass
class Document:
    doc_id: str          # short id used in prompts, e.g. "SYL1", "LAB1", "JD2"
    name: str            # original file name
    category: str        # one of CATEGORIES keys
    text: str
    truncated: bool = False
    warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Return (text, warning). Warning is empty when extraction looks fine."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "", "PyMuPDF is not installed, so PDFs cannot be read."
        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        except Exception as exc:  # corrupted or encrypted PDF
            return "", f"Could not open PDF: {exc}"
        text = _clean(text)
        if len(text) < 40:
            return text, ("Very little text found. This may be a scanned PDF; "
                          "OCR is not implemented in this prototype.")
        return text, ""
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return _clean(data.decode(enc)), ""
            except UnicodeDecodeError:
                continue
        return "", "Could not decode the text file."
    return "", f"Unsupported file type '{suffix}'. Use PDF, TXT or MD."


PREFIX = {"syllabus": "SYL", "practice": "LAB", "assessment": "ASM", "industry": "JD"}


def build_documents(files_by_category: dict[str, list[tuple[str, bytes]]]) -> list[Document]:
    """files_by_category: {category: [(filename, bytes), ...]}"""
    docs: list[Document] = []
    for category, files in files_by_category.items():
        for i, (name, data) in enumerate(files, start=1):
            text, warning = extract_text(name, data)
            truncated = len(text) > MAX_CHARS_PER_DOC
            if truncated:
                text = text[:MAX_CHARS_PER_DOC]
                warning = (warning + " " if warning else "") + \
                    f"Document truncated to the first {MAX_CHARS_PER_DOC:,} characters."
            docs.append(Document(
                doc_id=f"{PREFIX[category]}{i}", name=name, category=category,
                text=text, truncated=truncated, warning=warning,
            ))
    return docs


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
PUBLIC_JOBS_DIR = SAMPLE_DIR / "public_jobs"
COURSE_FILES = {
    "syllabus": ["01_syllabus.md"],
    "practice": ["02_lab_manual.md"],
    "assessment": ["03_assessments.md"],
}
SYNTHETIC_JOB_FILES = [
    "04_job_noc_engineer.md",
    "05_job_junior_network_engineer.md",
    "06_job_network_support_bank.md",
    "07_job_field_engineer_telecom.md",
]
# Backwards-compatible layout name used by older code/tests
SAMPLE_LAYOUT = {**COURSE_FILES, "industry": SYNTHETIC_JOB_FILES}

JOB_SETS = {
    "public": "Public Pakistani job postings (4, sourced excerpts)",
    "synthetic": "Synthetic job descriptions (4, fictional)",
}


def job_sources(job_set: str = "public") -> list[dict]:
    """Provenance records for the bundled job descriptions."""
    if job_set == "public":
        return json.loads((PUBLIC_JOBS_DIR / "sources.json").read_text(encoding="utf-8"))
    return [{"file": f, "title": f.split("_job_")[-1].replace("_", " ").replace(".md", "").title(),
             "employer": "Fictional", "location": "-", "posted": "-", "status": "-",
             "source_url": "", "accessed": "-", "provenance": "Synthetic demo data"}
            for f in SYNTHETIC_JOB_FILES]


def load_sample_documents(job_set: str = "synthetic") -> list[Document]:
    files = {cat: [(n, (SAMPLE_DIR / n).read_bytes()) for n in names]
             for cat, names in COURSE_FILES.items()}
    if job_set == "public":
        files["industry"] = [(r["file"], (PUBLIC_JOBS_DIR / r["file"]).read_bytes())
                             for r in job_sources("public")]
    elif job_set == "synthetic":
        files["industry"] = [(n, (SAMPLE_DIR / n).read_bytes()) for n in SYNTHETIC_JOB_FILES]
    return build_documents(files)


def build_manual_documents(clo_text: str, items_text: str, jd_text: str = "") -> list[Document]:
    """Fallback: CLOs, assessment items (and optional job text) typed or pasted by faculty."""
    files = {}
    if clo_text.strip():
        files["syllabus"] = [("manual_clos.txt", ("Course Learning Outcomes (entered manually)\n"
                                                  + clo_text.strip()).encode())]
    if items_text.strip():
        files["assessment"] = [("manual_assessment_items.txt", ("Assessment items (entered manually)\n"
                                                               + items_text.strip()).encode())]
    if jd_text.strip():
        files["industry"] = [("manual_job_requirements.txt", jd_text.strip().encode())]
    return build_documents(files)
