"""
analysis.py — Hybrid ATS Narrative + Skill/Gaps JSON
----------------------------------------------------
Outputs:
1) An ATS-style markdown comparison (your original style)
2) A skill coverage snapshot
3) A JSON blob used by interview + resume_builder
"""
import pprint
import re
import json
from typing import List, Dict, Any

from .skills_map import load_spacy_model
from .openai_client import chat_completion

import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError:
    Sentence = None
    SequenceTagger = None

# Ensure required NLTK models are available ONCE at import time.
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)


# Ensure the standard punkt tokenizer is available.
# Newer NLTK versions may also use punkt_tab internally,
# but downloading "punkt" will install everything needed.



# Initialize spaCy model

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
# HYBRID KEYWORD / PHRASE EXTRACTION HELPERS
# (spaCy + NLTK + Flair)
# ============================================================

_flair_tagger = None

def get_flair_tagger():
    """
    Lazy-load Flair NER tagger if installed.
    If Flair is not installed or load fails, return None and skip Flair.
    """
    global _flair_tagger
    if _flair_tagger is not None:
        return _flair_tagger

    if SequenceTagger is None:
        _flair_tagger = None
        return _flair_tagger

    try:
        _flair_tagger = SequenceTagger.load("ner")
    except Exception:
        _flair_tagger = None
    return _flair_tagger


def _normalize_phrase(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _dedupe_case_insensitive(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            result.append(it)
    return result


def extract_nltk_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases using NLTK. If NLTK's data (punkt or tagger) is somehow missing,
    fail gracefully (empty list) instead of trying to download on every call.
    """
    # Sentence tokenization
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        return []  # fail-safe

    tokens = [word_tokenize(sent) for sent in sentences]

    # POS tagging
    try:
        tagged = [pos_tag(t) for t in tokens]
    except LookupError:
        return []  # fail-safe

    grammar = r"""
        NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}
            {<NNP>+}
    """
    chunker = RegexpParser(grammar)

    noun_phrases: List[str] = []
    for sent in tagged:
        tree = chunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join(word for word, tag in subtree.leaves())
                if phrase and phrase not in noun_phrases:
                    noun_phrases.append(phrase)

    return noun_phrases





def extract_flair_phrases(text: str) -> List[str]:
    """
    Flair NER spans as extra candidates.
    Best-effort: if Flair not installed or fails, returns [].
    """
    tagger = get_flair_tagger()
    if tagger is None:
        return []

    text = (text or "").strip()
    if not text:
        return []

    sentence = Sentence(text)
    try:
        tagger.predict(sentence)
    except Exception:
        return []

    phrases: List[str] = []
    for span in sentence.get_spans("ner"):
        phrases.append(_normalize_phrase(span.text))
    return phrases

nlp = load_spacy_model()

def extract_keywords_spacy(text: str) -> List[str]:
    """
    Hybrid extractor using:
    - spaCy noun chunks + key tokens
    - NLTK noun phrases
    - Flair NER spans (if available)

    Then deduplicate (case-insensitive).
    """
    text = (text or "").strip()
    if not text:
        return []

    # --- spaCy base layer ---
    doc = nlp(text.lower())
    spacy_terms: List[str] = []

    # noun chunks
    for chunk in doc.noun_chunks:
        spacy_terms.append(chunk.text.strip())

    # important single tokens
    for token in doc:
        if token.pos_ in ("VERB", "NOUN", "PROPN", "ADJ"):
            spacy_terms.append(token.text.strip())

    spacy_terms = [
        t for t in spacy_terms
        if len(t) > 2 and not re.match(r"^\W+$", t)
    ]

    # --- NLTK phrases ---
    nltk_terms = extract_nltk_noun_phrases(text)

    # --- Flair phrases (optional) ---
    flair_terms = extract_flair_phrases(text)

    combined = spacy_terms + nltk_terms + flair_terms
    # combined = spacy_terms
    combined = [_normalize_phrase(t) for t in combined]
    final_terms = _dedupe_case_insensitive(combined)

    return final_terms


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

# nlp = load_spacy_model()

def extract_keywords_spacy(text: str) -> List[str]:
    """
    Extract keywords / phrases from JD/Resume using:
    - spaCy noun chunks + key tokens
    - NLTK noun phrases
    - Flair NER spans (if available)

    Then deduplicate (case-insensitive).
    """
    text = (text or "").strip()
    if not text:
        return []

    # --- spaCy base layer ---
    doc = nlp(text.lower())
    spacy_terms: List[str] = []

    # noun chunks
    for chunk in doc.noun_chunks:
        spacy_terms.append(chunk.text.strip())

    # important single tokens
    for token in doc:
        if token.pos_ in ("VERB", "NOUN", "PROPN", "ADJ"):
            spacy_terms.append(token.text.strip())

    spacy_terms = [
        t for t in spacy_terms
        if len(t) > 2 and not re.match(r"^\W+$", t)
    ]

    # --- NLTK phrases ---
    nltk_terms = extract_nltk_noun_phrases(text)

    # --- Flair phrases (optional) ---
    flair_terms = extract_flair_phrases(text)

    # combine all three + dedupe
    combined = spacy_terms + nltk_terms + flair_terms
    combined = [ _normalize_phrase(t) for t in combined ]
    final_terms = _dedupe_case_insensitive(combined)

    return final_terms



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
    Ask the LLM to build a structured, interview-oriented gap analysis
    AND dynamically infer higher-level competencies from JD vs Resume.
    """

    system_msg = (
        "You are a senior data science hiring manager and interview coach. "
        "You analyse JD vs resume gaps, infer higher-level competencies, "
        "and design interview focus areas."
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
- Additionally, infer **HIGH-LEVEL COMPETENCIES** required by the JD and shown (or missing)
  in the resume.

Definition of competencies (dynamic, NOT a fixed list):
- Short noun-phrases that describe capability themes, e.g.:
  - product lifecycle ownership
  - root cause analysis
  - business understanding
  - experimentation mindset
  - metrics ownership
  - stakeholder management
  - requirement gathering
  - cross-functional collaboration
- They are **not just tools** (e.g. SQL, Python) and **not generic soft skills** like
  "communication" alone.
- You must infer them from the JD and resume wording, not from any static list.

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
  "competency_gap_explanation": "2–4 sentence explanation of the higher-level competency gaps (e.g. product lifecycle, root cause analysis, business understanding, stakeholder management) and what to probe.",

  "jd_competencies": [
    "Short competency phrase inferred from the JD (e.g. 'product lifecycle ownership')",
    "..."
  ],
  "resume_competencies": [
    "Short competency phrase clearly demonstrated in the resume",
    "..."
  ],
  "missing_competencies": [
    "Competency that seems important in the JD but is weak or missing in the resume",
    "..."
  ],

  "interview_focus_areas": [
    "Short bullet-style description of a priority interview area",
    "...",
  ],
  "suggested_interview_questions": [
    "Concrete behavioural / project-based interview question tailored to the above gaps",
    "...",
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
            "competency_gap_explanation": "",
            "jd_competencies": [],
            "resume_competencies": [],
            "missing_competencies": [],
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
    competency_gap_explanation = str(payload.get("competency_gap_explanation", "")).strip()

    jd_competencies = _clean_list(payload.get("jd_competencies", []))
    resume_competencies = _clean_list(payload.get("resume_competencies", []))
    missing_competencies = _clean_list(payload.get("missing_competencies", []))

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
        "competency_gap_explanation": competency_gap_explanation,
        "jd_competencies": jd_competencies,
        "resume_competencies": resume_competencies,
        "missing_competencies": missing_competencies,
        "interview_focus_areas": interview_focus_areas,
        "suggested_interview_questions": suggested_interview_questions,
        "skill_clusters_to_probe": clusters_clean,
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
# JD THEMES ("AXES") + GENERIC COVERAGE SCORING
# ============================================================

def llm_extract_jd_axes(jd_text: str) -> List[Dict[str, Any]]:
    """
    Use the LLM to extract 3–6 major 'axes' (themes) from the JD.

    Each axis is a generic capability theme we can score resumes against,
    e.g. "AI/ML enablement", "Stakeholder management", "Experimentation & metrics".
    This is fully generic and works for any role.
    """
    system_msg = (
        "You read job descriptions and identify 3–6 major capability themes "
        "that matter when evaluating candidates. Themes must be generic, not "
        "company-specific."
    )

    user_msg = f"""
Read the JOB DESCRIPTION below and extract 3–6 major themes ("axes") that matter when evaluating candidates.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

For each theme, return a JSON list with objects of this shape:

[
  {{
    "name": "short axis name",
    "description": "1–2 line explanation in plain language",
    "keywords": ["word1", "word2", "short phrase 3"],
    "importance": 0.0   // float between 0 and 1 (higher = more critical in this JD)
  }},
  ...
]

Rules:
- Themes must be domain-agnostic (e.g. "AI/ML enablement", "Stakeholder management",
  "Experimentation & metrics"), not phrases like "Meta internal tools".
- 'keywords' should be lower-case unigrams or short phrases that appear in the JD
  or are obvious synonyms.
- importance must roughly sum to about the number of axes
  (e.g. for 4 axes, values might be 0.8, 0.9, 0.6, 0.7).

Return ONLY valid JSON (no backticks, no comments).
"""

    try:
        raw = chat_completion(system_msg, user_msg, temperature=0.2)
        axes = json.loads(raw)
    except Exception:
        return []

    clean_axes: List[Dict[str, Any]] = []
    for a in axes or []:
        if not isinstance(a, dict):
            continue

        name = str(a.get("name", "")).strip()
        if not name:
            continue

        desc = str(a.get("description", "")).strip()
        keywords = [
            str(k).strip().lower()
            for k in (a.get("keywords") or [])
            if isinstance(k, str) and str(k).strip()
        ]
        importance = float(a.get("importance", 0.5) or 0.5)
        # clamp between 0 and 1
        importance = max(0.0, min(1.0, importance))

        clean_axes.append(
            {
                "name": name,
                "description": desc,
                "keywords": keywords,
                "importance": importance,
            }
        )

    return clean_axes


def score_axes_for_resume(
    axes: List[Dict[str, Any]],
    resume_text: str,
) -> List[Dict[str, Any]]:
    """
    For each JD axis, compute a simple coverage score based on keyword hits.

    coverage = (# of distinct axis keywords found in resume) / (# axis keywords)
    weighted_score = coverage * importance

    This is totally generic: axes can be AI, finance, marketing, whatever.
    """
    resume_low = resume_text.lower()

    scored: List[Dict[str, Any]] = []
    for axis in axes:
        kws = axis.get("keywords") or []
        if not kws:
            coverage = 0.0
        else:
            hits = 0
            seen = set()
            for kw in kws:
                kw_low = str(kw).lower().strip()
                if not kw_low or kw_low in seen:
                    continue
                if kw_low in resume_low:
                    hits += 1
                    seen.add(kw_low)
            coverage = hits / max(1, len(kws))

        importance = float(axis.get("importance", 0.5) or 0.5)
        importance = max(0.0, min(1.0, importance))
        weighted = coverage * importance

        scored.append(
            {
                "name": axis.get("name", ""),
                "description": axis.get("description", ""),
                "keywords": axis.get("keywords", []),
                "importance": importance,
                "coverage": round(coverage, 3),
                "weighted_score": round(weighted, 3),
            }
        )

    return scored

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
        • JSON Summary used by agents (hidden from user)
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
    jd_competencies = gap_analysis.get("jd_competencies", []) or []
    resume_competencies = gap_analysis.get("resume_competencies", []) or []
    missing_competencies = gap_analysis.get("missing_competencies", []) or []
    competency_gap_explanation = gap_analysis.get("competency_gap_explanation", "")

    # 🔹 Extract generic JD axes + score how well the resume covers each
    # NOTE: We keep these ONLY in JSON for internal use (LLM, weighting),
    #       we do NOT show them to the end user.
    jd_axes = llm_extract_jd_axes(jd_text)
    axis_scores = score_axes_for_resume(jd_axes, resume_text)

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
        "competency_analysis": {
            "jd_competencies": jd_competencies,
            "resume_competencies": resume_competencies,
            "missing_competencies": missing_competencies,
            "competency_gap_explanation": competency_gap_explanation,
        },
        # 🔹 NEW fields that resume + interview can now use internally
        "jd_axes": jd_axes,
        "axis_scores": axis_scores,
        "gap_analysis": gap_analysis,
    }

    json_blob = json.dumps(analysis_json, indent=2)
# DEBUG OUTPUT
    print("\n================= ANALYSIS_JSON DEBUG =================")
    print(json_blob)
    print("========================================================\n")
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

**JD Core Competencies (inferred):** {", ".join(jd_competencies) if jd_competencies else "None"}  
**Resume Core Competencies:** {", ".join(resume_competencies) if resume_competencies else "None"}  
**Missing / Under-emphasized Competencies:** {", ".join(missing_competencies) if missing_competencies else "None"}  

**Competency Gap Notes:**  
{competency_gap_explanation or "-"}

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
    json_block = f"\nJSON_BLOCK_START\n{json_blob}\nJSON_BLOCK_END\n"

    # ats_md is your original ATS-style markdown part
    return ats_md + extra_md + json_block


