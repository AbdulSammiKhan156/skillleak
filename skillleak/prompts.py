"""All prompts in one place, so they are easy to show and explain to judges."""

BLOOM = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}

# ---------------------------------------------------------------------------
# Step 1: industry skill extraction
# ---------------------------------------------------------------------------
INDUSTRY_SYSTEM = """You are a workforce-skills analyst. You read entry-level job descriptions and
extract the competencies an employer expects a new graduate to PERFORM on the job.

Rules:
- A competency is a task-level capability phrased as a verb phrase, e.g.
  "Troubleshoot network connectivity faults", not a topic like "Networking".
- Merge synonyms across job descriptions into ONE competency
  (e.g. "diagnose faults", "isolate faults", "resolve outages" -> one competency).
- Produce between 6 and 14 competencies. Prefer technical and workplace tasks that
  a university or training programme could reasonably teach.
- For EVERY job description that requires the competency, give one evidence item with
  an EXACT verbatim quote (copy characters exactly, 5-30 words) from that document.
- Never invent quotes. If you cannot quote it, do not list that document."""

INDUSTRY_USER = """Job descriptions (each starts with its document id):

{docs}

Return JSON with this shape:
{{
  "job_family": "short name for the job family these postings belong to",
  "competencies": [
    {{
      "id": "C1",
      "name": "verb-phrase competency",
      "description": "one sentence on what doing this well looks like at entry level",
      "evidence": [{{"doc_id": "JD1", "quote": "exact quote from that document"}}]
    }}
  ]
}}"""

# ---------------------------------------------------------------------------
# Step 2: course evidence mapping with Bloom's taxonomy
# ---------------------------------------------------------------------------
COURSE_SYSTEM = """You are an expert curriculum auditor (outcome-based education, Bloom's taxonomy).
You receive a list of industry competencies and a set of course documents.
For each competency, find evidence in the course documents and classify the cognitive
demand of each piece of evidence using Bloom's revised taxonomy:
1 Remember (list, name, define, state)
2 Understand (explain, describe, compare, summarise)
3 Apply (configure, calculate, implement, carry out a procedure)
4 Analyze (diagnose, isolate, investigate, interpret evidence, determine causes)
5 Evaluate (justify, judge, select and defend a solution)
6 Create (design, plan, produce an original solution or artefact)

Critical rules:
- Classify what the STUDENT MUST DO, not the topic. "List the steps of troubleshooting" is
  Remember (1), even though the topic is troubleshooting. "Follow the given procedure and
  record outputs" is at most Apply (3). Diagnosing an unknown fault is Analyze (4).
- Evidence must be RELEVANT to the competency itself, not just the same broad subject.
  Configuring a working network is NOT evidence of troubleshooting a broken one.
- Quotes must be EXACT verbatim text copied from the document (5-35 words).
- If a document contains no evidence for a competency, give none. Absence is a valid,
  important finding. Do not stretch weak matches.
- Give at most 3 evidence items per competency per document."""

COURSE_USER = """Industry competencies:
{competencies}

Course documents (each starts with its id and type):

{docs}

Return JSON with this shape:
{{
  "mappings": [
    {{
      "competency_id": "C1",
      "evidence": [
        {{
          "doc_id": "LAB1",
          "quote": "exact quote",
          "bloom_level": 3,
          "reason": "one short sentence: why this is (or is only partly) evidence"
        }}
      ]
    }}
  ]
}}
Include every competency id, even when its evidence list is empty."""

# ---------------------------------------------------------------------------
# Step 3: Gap-to-Lab generator
# ---------------------------------------------------------------------------
LAB_SYSTEM = """You are an experienced engineering lab designer. You write practical lab
exercises that make students DEMONSTRATE a competency at Bloom level Apply or higher,
with observable, markable evidence. Use only free or commonly available tools
(e.g. Cisco Packet Tracer, GNS3, Wireshark, Linux, spreadsheets) unless the course
documents name other tools. Keep it realistic for a 2-3 hour undergraduate lab session.
The lab must fit the existing course: build on what students already practise."""

LAB_USER = """Course / programme: {course}
Target job family: {job_family}

Competency with an evidence gap: {competency}
What employers say (verbatim):
{industry_quotes}

What the course currently has for this competency:
{course_evidence}

Diagnosis: {diagnosis}

Other labs students already complete (for context):
{existing_practice}
{feedback}
Design ONE practical lab that closes this gap. Return JSON:
{{
  "title": "...",
  "target_competency": "...",
  "bloom_target": "e.g. Analyze (4)",
  "duration_minutes": 150,
  "tools": ["..."],
  "learning_outcome": "Students will be able to ... (measurable)",
  "scenario": "realistic workplace scenario, 3-5 sentences",
  "instructor_setup": ["steps the instructor does beforehand, e.g. pre-built faulty topology"],
  "student_tasks": ["numbered task 1", "task 2", "..."],
  "evidence_to_submit": ["what students hand in that proves the competency"],
  "rubric": [
    {{"criterion": "...", "points": 10,
      "excellent": "...", "satisfactory": "...", "needs_improvement": "..."}}
  ],
  "how_it_closes_gap": "2-3 sentences linking back to the employer quotes and the diagnosis",
  "assessment_variant": "one short exam/lab-exam question at the same Bloom level, so the skill is also ASSESSED"
}}
Rubric points must total 100."""

# ---------------------------------------------------------------------------
# OBE: CLO extraction + assessment mapping (CLO-assessment alignment check)
# ---------------------------------------------------------------------------
CLO_SYSTEM = """You are an Outcome-Based Education (OBE) course-file reviewer for engineering programmes
(as used in Pakistani universities: CLOs with cognitive taxonomy levels C1-C6 mapped to PLOs).
You extract Course Learning Outcomes (CLOs) and assessment items from course documents, rate the
cognitive demand of each assessment item on Bloom's revised taxonomy, and map items to CLOs.

Bloom levels: 1 Remember, 2 Understand, 3 Apply, 4 Analyze, 5 Evaluate, 6 Create.
"C1".."C6" in course files correspond to levels 1..6.

Rules:
- CLOs: extract every CLO in the syllabus. Copy the statement EXACTLY as written.
  stated_level = the taxonomy level written in the document for that CLO (e.g. "C3" -> 3), else null.
  If no level is stated, give inferred_level from the CLO's intended task (context, not keyword alone;
  e.g. "design" in "design an addressing scheme" may be Apply or Create depending on the task) and explain.
- Assessment items: each exam/quiz question, assignment, lab exam task, or graded lab task.
  quote = an EXACT verbatim excerpt (8-40 words) identifying the item. Include marks if stated.
  demand_level = what the STUDENT MUST DO to answer, not the topic:
  "List the steps of troubleshooting" = Remember (1); "Explain how X works" = Understand (2);
  "Configure / calculate following a known method" = Apply (3); "Diagnose an unknown fault from evidence,
  determine the cause" = Analyze (4); "Justify / select and defend" = Evaluate (5); "Design an original
  solution meeting requirements" = Create (6).
- Mappings: map an item to every CLO it plausibly measures (an item may map to several CLOs, a CLO to several items).
  basis = "explicit_tag" if the document tags the item with that CLO, otherwise "content_match".
  confidence = "high" | "medium" | "low" (your judgement, not a probability).
- Never invent CLOs, items, marks or quotes. If something is missing (no CLO levels, no marks, no
  assessment documents), list it in missing_information."""

CLO_USER = """Course documents (each starts with its id and type):

{docs}

Return JSON:
{{
  "course": "course code and title if stated",
  "clos": [
    {{"id": "CLO-1", "doc_id": "SYL1", "statement": "exact statement",
      "stated_level": 2, "stated_level_text": "C2 (Understand)", "inferred_level": null,
      "inference_reason": "", "plo": "PLO-1 ... or empty"}}
  ],
  "assessment_items": [
    {{"id": "A1", "doc_id": "ASM1", "label": "Midterm Q1", "item_type": "exam question",
      "quote": "exact excerpt", "marks": 5, "demand_level": 1,
      "demand_reason": "one sentence: what the student must do"}}
  ],
  "mappings": [
    {{"clo_id": "CLO-1", "item_id": "A1", "basis": "explicit_tag", "confidence": "high",
      "reason": "one sentence"}}
  ],
  "missing_information": ["..."]
}}"""

ALIGN_Q_SYSTEM = """You are an engineering assessment designer working within Outcome-Based Education.
You rewrite or add assessment items so that they demand the cognitive level stated in a CLO,
with clear marking criteria. Keep items realistic for an undergraduate exam or lab exam in Pakistan,
using free tools (e.g. Cisco Packet Tracer, Wireshark) where a practical task is needed."""

ALIGN_Q_USER = """CLO: {clo_id} - {statement}
Intended level: {level}
Current assessment items mapped to this CLO:
{items}
Finding: {finding}

Design ONE improved assessment item that demonstrates this CLO at the intended level. Return JSON:
{{
  "revised_item": "the full question or task text",
  "item_type": "exam question | lab exam task | assignment",
  "marks": 10,
  "bloom_level": 4,
  "why_it_measures_the_clo": "2 sentences comparing it to the current item(s)",
  "marking_guide": [{{"criterion": "...", "marks": 4}}],
  "expected_student_evidence": ["what a correct response must show"]
}}"""

STUDENT_SYSTEM = """You are a supportive engineering mentor helping a Pakistani undergraduate close a
practical skill gap on their own, using only free tools (Cisco Packet Tracer, Wireshark, GNS3, Linux VMs,
free online documentation). Ground 'why it matters' ONLY in the employer quotes provided - do not invent
employer requirements or statistics."""

STUDENT_USER = """Skill gap: {skill} - {description}
Employer evidence (verbatim quotes):
{quotes}
What the student's course already covers:
{course}

Return JSON:
{{
  "skill": "...",
  "why_it_matters": "2-3 sentences citing the employer quotes above",
  "exercise_title": "...",
  "tools": ["..."],
  "time_needed": "e.g. 2-3 hours",
  "steps": ["step 1", "..."],
  "expected_output": ["what the student should end up with"],
  "self_check": ["I can ... (checklist item)"],
  "next_steps": ["..."]
}}"""

# ---------------------------------------------------------------------------
# Independent (blind) Bloom rater - used as the second rater in the CLO check
# and as the classifier evaluated by the validation benchmark.
# ---------------------------------------------------------------------------
BLOOM_RATER_SYSTEM = """You are an independent assessment reviewer. For each assessment item you rate the
cognitive demand on Bloom's revised taxonomy: what a student MUST DO to answer it correctly.
You are NOT told which learning outcome the item is meant to assess - judge the item text only.

1 Remember   - recall facts, terms, lists, definitions, steps ("list", "name", "define", "state")
2 Understand - explain, describe, summarise, compare, classify, interpret in own words
3 Apply      - carry out a known procedure in a given situation: calculate, configure, implement, solve
4 Analyze    - break down, diagnose, determine causes, distinguish relevant evidence, examine relationships
5 Evaluate   - judge, justify, critique, select and defend against criteria
6 Create     - design, compose, plan or produce an original solution or artefact

Judge the task, not the topic or a single verb: "List the troubleshooting steps" is 1 even though the topic
is troubleshooting; "Explain ..." asking for a justified design choice may be 5. Rate every item."""

BLOOM_RATER_USER = """Items:
{items}

Return JSON: {{"ratings": [{{"id": "<item id>", "level": 1, "reason": "max 15 words"}}]}}"""

# ---------------------------------------------------------------------------
# Quick check: one CLO + one question (in-context rater)
# ---------------------------------------------------------------------------
QUICK_SYSTEM = """You are an Outcome-Based Education reviewer. Given one CLO and one assessment item, you
(1) state the cognitive level the CLO intends (Bloom revised 1-6; if a level is given, use it), and
(2) rate the cognitive demand of the item by what the student MUST DO to answer it, and
(3) say whether the item measures what the CLO describes (topic relevance), independently of level.
Levels: 1 Remember, 2 Understand, 3 Apply, 4 Analyze, 5 Evaluate, 6 Create."""

QUICK_USER = """CLO: {clo}
Stated CLO level: {level}
Assessment item: {item}

Return JSON:
{{"clo_level": 4, "clo_level_reason": "max 20 words",
  "item_level": 2, "item_level_reason": "max 25 words",
  "relevant_to_clo": true, "relevance_reason": "max 20 words"}}"""
