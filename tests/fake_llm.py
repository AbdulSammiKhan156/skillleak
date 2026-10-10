"""Canned model replies used ONLY by the automated tests (no network).

These let us test extraction, quote verification, scoring and the UI without an
API key. They are never used by the app itself.
"""
from skillleak import prompts

INDUSTRY = {
    "job_family": "Entry-level network engineering",
    "competencies": [
        {"id": "C1", "name": "Troubleshoot network connectivity faults",
         "description": "Isolate and resolve faults layer by layer.",
         "evidence": [
             {"doc_id": "JD1", "quote": "Troubleshoot connectivity, latency and packet-loss issues across routers and switches"},
             {"doc_id": "JD2", "quote": "Diagnose and resolve network faults reported by customers"},
             {"doc_id": "JD3", "quote": "troubleshoot branch connectivity outages and restore service quickly"},
             {"doc_id": "JD4", "quote": "Troubleshoot link and service faults on site using a structured fault-isolation process"}]},
        {"id": "C2", "name": "Configure routing and switching",
         "description": "Configure VLANs, static routes and OSPF.",
         "evidence": [
             {"doc_id": "JD1", "quote": "Perform basic configuration changes on routers and switches (VLANs, static routes, OSPF)"},
             {"doc_id": "JD2", "quote": "including VLANs, inter-VLAN routing and OSPF"},
             {"doc_id": "JD4", "quote": "configuring interfaces, VLANs and static routes"}]},
        {"id": "C3", "name": "Document incidents and write incident reports",
         "description": "Record faults, root cause and corrective action.",
         "evidence": [
             {"doc_id": "JD1", "quote": "Log, update and close incident tickets with a clear description of the fault, root cause and corrective action"},
             {"doc_id": "JD3", "quote": "write a short incident report for outages longer than 30 minutes"},
             # deliberately hallucinated quote - must be rejected by verification
             {"doc_id": "JD2", "quote": "Maintain the ITIL incident register for all enterprise customers"}]},
    ],
}

COURSE = {
    "mappings": [
        {"competency_id": "C1", "evidence": [
            {"doc_id": "SYL1", "quote": "Network troubleshooting methodology - the layered (bottom-up, top-down, divide-and-conquer) approach to fault isolation",
             "bloom_level": 2, "reason": "Lecture topic only."},
            {"doc_id": "ASM1", "quote": "List the steps of the bottom-up troubleshooting approach.",
             "bloom_level": 1, "reason": "Recall of steps, no diagnosis required."}]},
        {"competency_id": "C2", "evidence": [
            {"doc_id": "SYL1", "quote": "Configure static and dynamic routing (OSPF) and VLAN-based switching on network devices.",
             "bloom_level": 3, "reason": "Stated CLO."},
            {"doc_id": "LAB1", "quote": "Enable OSPF process 1 on all routers, advertise the given networks in area 0",
             "bloom_level": 3, "reason": "Hands-on configuration."},
            {"doc_id": "ASM1", "quote": "configure static routing between two routers and configure two VLANs on the switch",
             "bloom_level": 3, "reason": "Practical lab exam."}]},
        {"competency_id": "C3", "evidence": []},
    ]
}

LAB = {
    "title": "Fault Hunt: Diagnosing a Broken Campus Network",
    "target_competency": "Troubleshoot network connectivity faults",
    "bloom_target": "Analyze (4)", "duration_minutes": 150,
    "tools": ["Cisco Packet Tracer"],
    "learning_outcome": "Students will isolate and correct three injected faults and justify each diagnosis with evidence.",
    "scenario": "You are on shift in a NOC. Users report outages.",
    "instructor_setup": ["Inject a wrong mask, a missing route and a VLAN mismatch."],
    "student_tasks": ["Reproduce the fault", "Isolate the layer", "Fix and verify"],
    "evidence_to_submit": ["Incident ticket for each fault"],
    "rubric": [{"criterion": "Fault isolation", "points": 50, "excellent": "a", "satisfactory": "b", "needs_improvement": "c"},
               {"criterion": "Incident ticket", "points": 50, "excellent": "a", "satisfactory": "b", "needs_improvement": "c"}],
    "how_it_closes_gap": "Students must diagnose, not list steps.",
    "assessment_variant": "Given show-command outputs, identify the fault.",
}


CLO = {
    "course": "EE-357 Computer Communication Networks",
    "clos": [
        {"id": "CLO-1", "doc_id": "SYL1", "statement": "Explain the functions of each layer of the OSI and TCP/IP reference models.",
         "stated_level": 2, "stated_level_text": "C2 (Understand)", "plo": "PLO-1 Engineering Knowledge"},
        {"id": "CLO-3", "doc_id": "SYL1", "statement": "Configure static and dynamic routing (OSPF) and VLAN-based switching on network devices.",
         "stated_level": 3, "stated_level_text": "C3 (Apply)", "plo": "PLO-5 Modern Tool Usage"},
        {"id": "CLO-5", "doc_id": "SYL1", "statement": "Diagnose and resolve common connectivity faults in small routed and switched networks.",
         "stated_level": 4, "stated_level_text": "C4 (Analyze)", "plo": "PLO-2 Problem Analysis"},
        {"id": "CLO-6", "doc_id": "SYL1", "statement": "Describe basic network security mechanisms including firewalls and access control lists.",
         "stated_level": 2, "stated_level_text": "C2 (Understand)", "plo": "PLO-1"},
        {"id": "CLO-9", "doc_id": "SYL1", "statement": "Evaluate wireless network designs for coverage.",
         "stated_level": None, "inferred_level": 5, "inference_reason": "evaluate"},
    ],
    "assessment_items": [
        {"id": "A1", "doc_id": "ASM1", "label": "Midterm Q1", "item_type": "exam question",
         "quote": "List the seven layers of the OSI model and state one function of each layer.", "marks": 5,
         "demand_level": 1, "demand_reason": "Recall."},
        {"id": "A2", "doc_id": "ASM1", "label": "Final Q5", "item_type": "exam question",
         "quote": "Explain the purpose of VLAN trunking and describe the role of the 802.1Q tag.", "marks": 10,
         "demand_level": 2, "demand_reason": "Explain."},
        {"id": "A3", "doc_id": "ASM1", "label": "Lab exam", "item_type": "lab exam",
         "quote": "configure static routing between two routers and configure two VLANs on the switch", "marks": None,
         "demand_level": 3, "demand_reason": "Configure."},
        {"id": "A4", "doc_id": "ASM1", "label": "Final Q2", "item_type": "exam question",
         "quote": "List the steps of the bottom-up troubleshooting approach.", "marks": 5,
         "demand_level": 1, "demand_reason": "Recall of steps."},
        {"id": "A5", "doc_id": "ASM1", "label": "Final Q3", "item_type": "exam question",
         "quote": "Describe how an extended access control list filters traffic and write an ACL statement",
         "marks": 10, "demand_level": 3, "demand_reason": "Write ACL."},
        {"id": "A6", "doc_id": "ASM1", "label": "Final Q4", "item_type": "exam question",
         "quote": "What is SNMP? Name its main components.", "marks": 5, "demand_level": 1, "demand_reason": "Recall."},
        {"id": "A7", "doc_id": "ASM1", "label": "Invented", "item_type": "exam question",
         "quote": "Troubleshoot the broken OSPF adjacency shown in Figure 7 and justify your fix.", "marks": 20,
         "demand_level": 4, "demand_reason": "Hallucinated item."},
    ],
    "mappings": [
        {"clo_id": "CLO-1", "item_id": "A1", "basis": "explicit_tag", "confidence": "high", "reason": "tag"},
        {"clo_id": "CLO-3", "item_id": "A2", "basis": "explicit_tag", "confidence": "high", "reason": "tag"},
        {"clo_id": "CLO-3", "item_id": "A3", "basis": "explicit_tag", "confidence": "high", "reason": "tag"},
        {"clo_id": "CLO-5", "item_id": "A4", "basis": "explicit_tag", "confidence": "high", "reason": "tag"},
        {"clo_id": "CLO-5", "item_id": "A7", "basis": "content_match", "confidence": "high", "reason": "fake"},
        {"clo_id": "CLO-6", "item_id": "A5", "basis": "explicit_tag", "confidence": "high", "reason": "tag"},
        {"clo_id": "CLO-404", "item_id": "A1", "basis": "content_match", "confidence": "low", "reason": "unknown clo"},
    ],
    "missing_information": ["No marks stated for the lab examination."],
}

ALIGN = {"revised_item": "Given a Packet Tracer topology with two injected faults, diagnose and fix them.",
         "item_type": "lab exam task", "marks": 10, "bloom_level": 4,
         "why_it_measures_the_clo": "Requires diagnosis.", "marking_guide": [{"criterion": "Fault found", "marks": 5}],
         "expected_student_evidence": ["ping output"]}

STUDENT = {"skill": "Troubleshoot network connectivity faults", "why_it_matters": "Postings ask for it.",
           "exercise_title": "Break and fix", "tools": ["Packet Tracer"], "time_needed": "2 hours",
           "steps": ["Build", "Break", "Fix"], "expected_output": ["Working network"],
           "self_check": ["I can isolate a fault by layer"], "next_steps": ["Try GNS3"]}


RATER = {"Q": 2, "A1": 1, "A2": 2, "A3": 3, "A4": 2, "A5": 2, "A6": 1, "A7": 4}


QUICK = {"clo_level": 4, "clo_level_reason": "diagnose", "item_level": 1, "item_level_reason": "list",
         "relevant_to_clo": True, "relevance_reason": "same topic"}


def fake_call_json(cfg, system, user, max_tokens=8000):
    if system == prompts.QUICK_SYSTEM:
        return dict(QUICK)
    if system == prompts.BLOOM_RATER_SYSTEM:
        import re
        ids = re.findall(r"^- (\S+?):", user, flags=re.M)
        return {"ratings": [{"id": i, "level": RATER.get(i, 2), "reason": "fake"} for i in ids]}
    if system == prompts.CLO_SYSTEM:
        return CLO
    if system == prompts.ALIGN_Q_SYSTEM:
        return dict(ALIGN)
    if system == prompts.STUDENT_SYSTEM:
        return dict(STUDENT)
    if system == prompts.INDUSTRY_SYSTEM:
        return INDUSTRY
    if system == prompts.COURSE_SYSTEM:
        return COURSE
    if system == prompts.LAB_SYSTEM:
        return dict(LAB)
    raise AssertionError("unexpected prompt")
