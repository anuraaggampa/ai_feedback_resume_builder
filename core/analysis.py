"""
analysis.py — Hybrid ATS Narrative + Skill/Gaps JSON
----------------------------------------------------
Outputs:
1) An ATS-style markdown comparison (your original style)
2) A skill coverage snapshot
3) A JSON blob used by interview + resume_builder
"""

import re
import json
from typing import List, Dict, Any

from .skills_map import load_spacy_model
from .openai_client import chat_completion

# Initialize spaCy model
nlp = load_spacy_model()

# ============================================================
# HARD / SOFT SKILL GROUND TRUTH LISTS
# (Must stay in sync with resume_builder.py & interview.py)
# ============================================================

HARD_SKILL_EXAMPLES = [
    "Python",
    "SQL",
    "Power BI",
    "Data analysis",
    "EDA",
    "Statistical analysis",
    "Hypothesis testing",
    "A/B testing",
    "Data pipelines",
    "ETL",
    "ELT",
    "Data cleaning",
    "Automation workflows",
    "Advanced Excel",
    "Power Query",
    "Dashboarding",
    "KPI design",
    "Forecasting",
    "Regression",
    "Machine learning fundamentals",
    "Database systems",
    "Reporting",
    "Product analytics",
]

SOFT_SKILL_EXAMPLES = [
    "Communication",
    "Presentation skills",
    "Decision-making",
    "Stakeholder management",
    "Consulting mindset",
    "Leadership",
    "Collaboration",
    "Problem solving",
    "Critical thinking",
    "Time management",
    "Task prioritization",
    "Attention to detail",
]


# ============================================================
# ORIGINAL ATS-STYLE NARRATIVE PROMPT (you liked this)
# ============================================================

def build_ats_markdown_prompt(jd_text: str, resume_text: str) -> str:
    return f"""
You are an expert ATS hiring assistant.

Compare the JOB DESCRIPTION and RESUME.

Produce a clear MARKDOWN report with these sections:

### JD_Requirements
- Hard skills
- Soft skills
- Tools / technologies
- Key responsibilities

### Resume_Coverage
- Skills that are well-covered
- Tools/tech that are clearly mentioned
- Types of projects / impact shown

### Gaps_and_Risks
- Skills required but missing or weak
- Responsibilities not reflected
- Seniority / leadership mismatch if any

### Overall_Comment
- 3–5 bullet summary of how well this resume fits the JD.

Avoid hallucinating skills not clearly in the resume.

--- JOB DESCRIPTION (JD) ---
{jd_text}

--- RESUME ---
{resume_text}
"""


def generate_ats_markdown(jd_text: str, resume_text: str) -> str:
    """
    Reproduce the original narrative analysis you liked.
    """
    prompt = build_ats_markdown_prompt(jd_text, resume_text)
    return chat_completion(
        system_msg="You are an expert ATS hiring assistant.",
        user_msg=prompt,
        temperature=0.2,
    )


# ============================================================
# RULE-BASED KEYWORD EXTRACTION (spaCy)
# ============================================================

def extract_keywords_spacy(text: str) -> List[str]:
    """Extract noun phrases, verbs, and adjectives from JD/Resume."""
    doc = nlp(text.lower())
    tokens = []

    # Noun chunks
    for chunk in doc.noun_chunks:
        tokens.append(chunk.text.strip())

    # Individual tokens
    for token in doc:
        if token.pos_ in ("VERB", "NOUN", "PROPN", "ADJ"):
            tokens.append(token.text.strip())

    cleaned = [
        t for t in tokens
        if len(t) > 2 and not re.match(r"^\W+$", t)
    ]
    return list(set(cleaned))


def match_skills(extracted: List[str], reference_list: List[str]) -> List[str]:
    """Substring match between extracted keywords and known skills."""
    found = []
    for skill in reference_list:
        skill_low = skill.lower()
        for token in extracted:
            if skill_low in token.lower():
                found.append(skill)
                break
    return sorted(list(set(found)))


# ============================================================
# LLM-ASSISTED SKILL MAPPING (canonical)
# ============================================================

def llm_match_known_skills(
    text: str,
    hard_ref: List[str],
    soft_ref: List[str],
) -> Dict[str, List[str]]:
    """
    Use an LLM to map nuanced JD/Resume wording back to our canonical
    HARD_SKILL_EXAMPLES and SOFT_SKILL_EXAMPLES.

    Returns:
        {
          "hard_skills": [...canonical names from hard_ref...],
          "soft_skills": [...canonical names from soft_ref...]
        }
    """
    system_msg = (
        "You are an expert at reading job descriptions and resumes and "
        "identifying which skills are actually present.\n"
        "You must ONLY choose from the provided canonical skill lists."
    )

    user_msg = f"""
You are given a text (either a Job Description or a Resume).

TEXT:
\"\"\"{text}\"\"\"


You also have two lists of canonical skills:

HARD_SKILLS = {hard_ref}
SOFT_SKILLS = {soft_ref}

Your task:
1. Carefully read the text and infer which of these canonical skills are demonstrated,
   even if the wording is slightly different (e.g., "building dashboards in Power BI"
   should count as "Power BI" and "Dashboarding").
2. Do NOT invent skills that are not in the lists.
3. Be conservative: only mark a skill as present if there is clear evidence.

Return a JSON object with this exact structure:

{{
  "hard_skills": ["<pick from HARD_SKILLS>", "..."],
  "soft_skills": ["<pick from SOFT_SKILLS>", "..."]
}}
"""

    try:
        raw = chat_completion(system_msg, user_msg, temperature=0.0)
        parsed = json.loads(raw)
    except Exception:
        # If LLM or JSON parsing fails, fall back gracefully
        return {"hard_skills": [], "soft_skills": []}

    hard = [
        s for s in parsed.get("hard_skills", [])
        if isinstance(s, str) and s in hard_ref
    ]
    soft = [
        s for s in parsed.get("soft_skills", [])
        if isinstance(s, str) and s in soft_ref
    ]

    return {
        "hard_skills": sorted(list(set(hard))),
        "soft_skills": sorted(list(set(soft))),
    }


# ============================================================
# ROLE TYPE DETECTION (LLM-based)
# ============================================================

def detect_role_type(jd_text: str, extracted_keywords: List[str]) -> str:
    """
    LLM-based role type detection: decide if the role is 'technical' or 'non-technical'.
    Temperature = 0.0, return exactly one of the two.
    """
    system_msg = (
        "You are an expert hiring manager. "
        "Your ONLY job is to classify a role as either 'technical' or 'non-technical'."
    )

    user_msg = f"""
Read the Job Description and the extracted keywords and decide if the role is
primarily technical or non-technical.

Definitions:
- "technical" → data/analytics/engineering/ML/software-heavy responsibilities, tools, coding, pipelines, etc.
- "non-technical" → roles centered on pure business consulting, stakeholder management, communication, strategy, with minimal hands-on coding.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

EXTRACTED KEYWORDS:
{extracted_keywords}

Answer with EXACTLY one word:
either
  technical
or
  non-technical
(no punctuation, no explanation).
"""

    try:
        resp = chat_completion(
            system_msg=system_msg,
            user_msg=user_msg,
            temperature=0.0,
        )
        ans = resp.strip().lower()

        if "non-technical" in ans or "non technical" in ans or ans == "non":
            return "non-technical"
        if "technical" in ans:
            return "technical"
    except Exception:
        # Fallback heuristic
        tech_indicators = [
            "python", "sql", "analytics", "model", "pipeline",
            "dashboard", "data", "statistics", "forecast", "excel",
        ]
        tech_hits = sum(
            1 for t in extracted_keywords if any(x in t for x in tech_indicators)
        )
        soft_indicators = [
            "consult", "stakeholder", "presentation", "communication",
            "leadership", "collaboration",
        ]
        soft_hits = sum(
            1 for t in extracted_keywords if any(x in t for x in soft_indicators)
        )
        if soft_hits > tech_hits:
            return "non-technical"

    return "technical"


# ============================================================
# LLM GAP ANALYSIS (for interview/resume tailoring)
# ============================================================

def llm_build_gap_analysis(
    jd_text: str,
    resume_text: str,
    jd_hard: List[str],
    jd_soft: List[str],
    resume_hard: List[str],
    resume_soft: List[str],
    missing_hard: List[str],
    missing_soft: List[str],
    role_type: str,
) -> Dict[str, Any]:
    """
    Ask the LLM to build a structured, interview-oriented gap analysis.
    """

    system_msg = (
        "You are a senior data science hiring manager and interview coach. "
        "You analyse JD vs resume gaps and design interview focus areas."
    )

    user_msg = f"""
You are given:

1) JOB DESCRIPTION (JD)
2) RESUME
3) Canonical skill lists (hard vs soft)
4) Which JD skills the resume already covers
5) Which JD skills are still missing
6) The detected ROLE TYPE (technical vs non-technical)

Your job:
- Write a **detailed, but concise** gap analysis that can drive interview questions
  and targeted resume improvements.
- Focus on **practical use**: what topics to probe, what clusters to ask about,
  and where the candidate needs to demonstrate depth or breadth.

JD_TEXT:
\"\"\"{jd_text}\"\"\"

RESUME_TEXT:
\"\"\"{resume_text}\"\"\"

JD_HARD_SKILLS: {jd_hard}
JD_SOFT_SKILLS: {jd_soft}

RESUME_HARD_SKILLS: {resume_hard}
RESUME_SOFT_SKILLS: {resume_soft}

MISSING_HARD_SKILLS: {missing_hard}
MISSING_SOFT_SKILLS: {missing_soft}

ROLE_TYPE: {role_type.upper()}

Return ONLY a JSON object in this exact structure:

{{
  "hard_gap_explanation": "2–4 sentence explanation of the hard-skill gaps and what should be probed or clarified.",
  "soft_gap_explanation": "2–4 sentence explanation of the soft-skill gaps and what should be probed.",
  "interview_focus_areas": [
    "Short bullet-style description of a priority interview area",
    "..."
  ],
  "suggested_interview_questions": [
    "Concrete behavioural / project-based interview question tailored to the above gaps",
    "..."
  ],
  "skill_clusters_to_probe": [
    {{
      "type": "hard" or "soft",
      "cluster": "e.g., 'SQL + Power BI + dashboarding'",
      "reason": "Why this cluster matters and what it will reveal"
    }},
    {{
      "type": "hard" or "soft",
      "cluster": "...",
      "reason": "..."
    }}
  ]
}}
"""

    try:
        raw = chat_completion(system_msg, user_msg, temperature=0.15)
        payload = json.loads(raw)
    except Exception:
        # Fallback simple structure
        hard_msg = (
            "The resume does not fully cover all JD hard skills. The reviewer/interviewer should probe: "
            + (", ".join(missing_hard) if missing_hard else "any advanced technical areas that seem light.")
        )
        soft_msg = (
            "The resume may under-emphasize some JD soft skills. The interviewer should explore: "
            + (", ".join(missing_soft) if missing_soft else "stakeholder communication, ownership, and consulting mindset.")
        )
        return {
            "hard_gap_explanation": hard_msg,
            "soft_gap_explanation": soft_msg,
            "interview_focus_areas": [
                "Clarify depth of hands-on experience in missing or lightly-covered hard skills.",
                "Validate soft skills such as stakeholder communication, leadership, and consulting mindset.",
            ],
            "suggested_interview_questions": [],
            "skill_clusters_to_probe": [],
        }

    # Clean / sanitize
    def _clean_list(x):
        return [str(v).strip() for v in (x or []) if isinstance(v, str) and str(v).strip()]

    hard_gap_explanation = str(payload.get("hard_gap_explanation", "")).strip()
    soft_gap_explanation = str(payload.get("soft_gap_explanation", "")).strip()
    interview_focus_areas = _clean_list(payload.get("interview_focus_areas", []))
    suggested_interview_questions = _clean_list(payload.get("suggested_interview_questions", []))
    skill_clusters_raw = payload.get("skill_clusters_to_probe", []) or []

    clusters_clean: List[Dict[str, str]] = []
    for item in skill_clusters_raw:
        if not isinstance(item, dict):
            continue
        c_type = str(item.get("type", "hard")).strip()
        cluster = str(item.get("cluster", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if cluster:
            clusters_clean.append(
                {"type": c_type, "cluster": cluster, "reason": reason}
            )

    return {
        "hard_gap_explanation": hard_gap_explanation,
        "soft_gap_explanation": soft_gap_explanation,
        "interview_focus_areas": interview_focus_areas,
        "suggested_interview_questions": suggested_interview_questions,
        "skill_clusters_to_probe": clusters_clean,
    }


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def analyze_jd_vs_resume(jd_text: str, resume_text: str) -> str:
    """
    Returns:
    - Markdown report that starts with your ORIGINAL ATS-style analysis,
      and then appends:
        • Skill Coverage Snapshot
        • Interview-oriented gap notes
        • JSON Summary used by agents
    """

    # 1) Original ATS narrative
    ats_md = generate_ats_markdown(jd_text, resume_text)

    # 2) Rule-based + LLM skill extraction
    jd_keywords = extract_keywords_spacy(jd_text)
    resume_keywords = extract_keywords_spacy(resume_text)

    jd_hard = match_skills(jd_keywords, HARD_SKILL_EXAMPLES)
    jd_soft = match_skills(jd_keywords, SOFT_SKILL_EXAMPLES)

    resume_hard = match_skills(resume_keywords, HARD_SKILL_EXAMPLES)
    resume_soft = match_skills(resume_keywords, SOFT_SKILL_EXAMPLES)

    try:
        jd_llm = llm_match_known_skills(jd_text, HARD_SKILL_EXAMPLES, SOFT_SKILL_EXAMPLES)
        resume_llm = llm_match_known_skills(resume_text, HARD_SKILL_EXAMPLES, SOFT_SKILL_EXAMPLES)

        jd_hard = sorted(set(jd_hard) | set(jd_llm["hard_skills"]))
        jd_soft = sorted(set(jd_soft) | set(jd_llm["soft_skills"]))

        resume_hard = sorted(set(resume_hard) | set(resume_llm["hard_skills"]))
        resume_soft = sorted(set(resume_soft) | set(resume_llm["soft_skills"]))
    except Exception:
        # If LLM mapping fails, keep spaCy-only
        pass

    missing_hard = sorted(set(jd_hard) - set(resume_hard))
    missing_soft = sorted(set(jd_soft) - set(resume_soft))

    coverage_hard = round(
        100 * (len(jd_hard) - len(missing_hard)) / max(1, len(jd_hard)), 1
    )
    coverage_soft = round(
        100 * (len(jd_soft) - len(missing_soft)) / max(1, len(jd_soft)), 1
    )

    role_type = detect_role_type(jd_text, jd_keywords)

    gap_analysis = llm_build_gap_analysis(
        jd_text=jd_text,
        resume_text=resume_text,
        jd_hard=jd_hard,
        jd_soft=jd_soft,
        resume_hard=resume_hard,
        resume_soft=resume_soft,
        missing_hard=missing_hard,
        missing_soft=missing_soft,
        role_type=role_type,
    )

    analysis_json = {
        "jd_hard_skills": jd_hard,
        "jd_soft_skills": jd_soft,
        "resume_hard_skills": resume_hard,
        "resume_soft_skills": resume_soft,
        "missing_hard_skills": missing_hard,
        "missing_soft_skills": missing_soft,
        "coverage": {
            "hard_percent": coverage_hard,
            "soft_percent": coverage_soft,
        },
        "role_type": role_type,
        "gap_analysis": gap_analysis,
    }

    json_blob = json.dumps(analysis_json, indent=2)

    # ---------- visible markdown part (user sees only this) ----------
    extra_md = f"""

---

## Skill Coverage Snapshot

**Role Type Detected:** **{role_type.upper()}**

**JD Hard Skills:** {", ".join(jd_hard) if jd_hard else "None"}  
**Resume Hard Skills:** {", ".join(resume_hard) if resume_hard else "None"}  
**Missing Hard:** {", ".join(missing_hard) if missing_hard else "None"}  
**Hard Skill Coverage:** **{coverage_hard}%**

**JD Soft Skills:** {", ".join(jd_soft) if jd_soft else "None"}  
**Resume Soft Skills:** {", ".join(resume_soft) if resume_soft else "None"}  
**Missing Soft:** {", ".join(missing_soft) if missing_soft else "None"}  
**Soft Skill Coverage:** **{coverage_soft}%**

---

## Interview-Oriented Gap Notes

**Hard-skill gaps (what to probe):**  
{gap_analysis.get("hard_gap_explanation", "Hard-skill gap analysis unavailable.")}

**Soft-skill gaps (what to probe):**  
{gap_analysis.get("soft_gap_explanation", "Soft-skill gap analysis unavailable.")}

**Priority interview focus areas:**  
{chr(10).join(f"- {item}" for item in gap_analysis.get("interview_focus_areas", [])) or "- (No specific focus areas generated.)"}

**Skill clusters to probe:**  
{chr(10).join(
    f"- **[{c.get('type', 'hard').upper()}]** {c.get('cluster', '')} — {c.get('reason', '').strip()}"
    for c in gap_analysis.get("skill_clusters_to_probe", [])
) or "- (No explicit clusters generated.)"}
"""

    # ---------- INTERNAL JSON BLOCK (for app.py only) ----------
    # We append it after a clear marker. The app will strip this before displaying.
    json_block = f"\nJSON_BLOCK_START\n{json_blob}\nJSON_BLOCK_END\n"

    # ats_md is your original ATS-style markdown part
    return ats_md + extra_md + json_block

