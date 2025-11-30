import json
from typing import List, Dict

from .openai_client import chat_completion


# ============================================================
# DUPLICATE GUARD
# ============================================================

def _normalize(text: str) -> str:
    """Normalize whitespace for simple equality checks."""
    return " ".join(text.strip().split())


def is_duplicate_question(text: str, interview_qa: List[Dict[str, str]]) -> bool:
    """Return True if this question text was already asked before."""
    norm = _normalize(text)
    for qa in interview_qa:
        q = qa.get("question", "")
        if q and _normalize(q) == norm:
            return True
    return False


def fallback_simple_question() -> str:
    """
    Simple, safe fallback that still helps tailoring,
    if a duplicate question is generated.
    """
    return (
        "Think of a project that matches this JD well. "
        "What was it about, and what was your role in it?"
    )


# ============================================================
# PROMPT BUILDER — ENHANCV-STYLE, STORY + IMPACT (KPIs)
# ============================================================

def build_next_question_prompt(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    num_questions_asked: int,
    max_questions: int,
) -> str:
    """
    Ask the LLM to act as an Enhancv-style resume coach and decide the next best question,
    or say DONE if we have enough info.

    New strategy (as per design):
    - We are NOT trying to cover every missing skill in the interview.
    - Missing skills will be handled later in the resume/PDF generation.
    - Interview focuses on:
        • Current role & responsibilities
        • 1–2 strongest/high-impact projects
        • KPIs / metrics / business impact (woven into project questions)
        • Ownership, decision-making, stakeholder stories
        • Evidence for core competencies (product lifecycle, RCA, business understanding, etc.)
    - Now also: JD THEMES / AXES (from the analysis) to decide WHICH areas to deepen.
    
    Phases:
    - Q1–3: Understand current role + responsibilities + strongest project + impact
    - Q4–(max_questions-1): Deep dive on 1–2 key projects (problem → actions → tools → impact)
    - Final question (optional): Anything else they want highlighted
    """

    return f"""
You are an expert resume coach, similar to the Enhancv "Tailor your resume" assistant.

Your job:
Ask the NEXT single question that will help you tailor and strengthen the candidate's resume
for THIS JD. The interview is NOT for testing theory or ticking off every JD skill.
Instead, it is for collecting strong material for high-impact bullets.

Your goal is to capture missing information that improves:

- Impact (clear business results, metrics, KPIs, before/after changes)
- Role context (team size, scope, responsibilities, level of ownership)
- Project stories (problem → actions → tools → outcome)
- Leadership / ownership / stakeholder signals
- Evidence for core competencies (e.g. product lifecycle ownership, root cause analysis,
  business understanding, experimentation mindset, stakeholder management, cross-functional collaboration)
- Missing achievements that belong in the resume
- Coverage of the most important JD THEMES / AXES that are currently weak in the resume

You have:
- The JD text.
- The candidate's current resume.
- An analysis of JD vs resume (gaps, coverage, role_type, gap_analysis JSON, dynamic competency analysis,
  and a **JD Theme Coverage (Axes)** section listing 3–6 key themes with importance and coverage scores).
- A history of previous interview Q&A pairs.

You have already asked {num_questions_asked} questions.
The maximum allowed is {max_questions}.

IMPORTANT:
- You do NOT need to cover every missing skill in the interview.
- Missing skills (from analysis_text) will be woven into the resume later.
- Use JD / gaps only to decide which projects or responsibilities to dig deeper into.
- Give extra priority to clarifying core COMPETENCIES that appear missing or weak in the analysis,
  especially items under "Missing / Under-emphasized Competencies" or similar sections.

--------------------------------
HOW TO USE JD THEMES (AXES)
--------------------------------
In the JD vs RESUME analysis, there may be a section like:

"JD Theme Coverage (Axes)"
- Each axis has:
  - a name (e.g. "AI/ML enablement", "KPI governance", "Stakeholder management"),
  - an IMPORTANCE score (0–1),
  - a COVERAGE score (0–1),
  - and a weighted score.

You MUST:
- Scan the JD Theme Coverage (Axes) section.
- Treat axes with **high importance** and **low coverage** as PRIORITY areas for your questions.
- When choosing what to ask about next, favour questions that:
  - Elicit stories, tools, and impact that support these weak-but-important axes.
  - Help the future resume show stronger evidence for those themes.
- Axes with already high coverage do NOT need many more questions; you can move on.

Example:
- If an axis like "AI/ML enablement" or "Metrics & experimentation" is high-importance but low-coverage:
  - Ask about projects where the candidate worked with data scientists, models, experiments, or KPIs.
- If an axis like "Stakeholder management" is high-importance but low-coverage:
  - Ask about how they worked with business partners, PMs, leaders, or cross-functional teams.

--------------------------------
HOW TO THINK ABOUT COMPETENCIES
--------------------------------
Competencies are higher-level capability themes, not individual tools or generic soft skills.
They describe how the candidate works in real situations.

Typical competency themes include:
- Product lifecycle ownership (ideation → requirements → implementation → rollout → metrics)
- Root cause analysis and problem investigation
- Business understanding and linking work to KPIs or revenue/risk
- Experimentation and learning mindset (A/B tests, hypothesis-driven work)
- Metrics ownership and impact orientation
- Stakeholder management and expectation-setting
- Requirement gathering and problem framing
- Cross-functional collaboration with PM/Eng/Design/Business

When a competency is missing or unclear in the analysis:
- Ask for a concrete story where that competency SHOULD show up.
- Guide the candidate to explain:
    1) The context / problem
    2) What they personally did (not just the team)
    3) Tools / methods they used (SQL, analysis, experiments, frameworks, etc.)
    4) The impact in simple, business language
    5) Any reflection or learning

Your questions should make it easy for the candidate to tell these stories in simple language.

--------------------------------
QUESTION PHASES
--------------------------------
Think in phases like a professional resume builder:

PHASE 1 (Questions 1–3): "Current role & strongest project, with impact"
- If you have asked FEWER THAN 3 questions so far, your question should:
  - Clarify what the candidate does in their CURRENT (or most recent) role.
  - Ask about their main responsibilities in simple language.
  - Quickly move to their **strongest/high-impact, JD-relevant project**.
  - While asking about that project, ALSO ask what changed or improved
    (metrics, KPIs, time saved, accuracy improved, risk reduced, revenue/profit impact etc.).
  - If the analysis suggests missing competencies like product lifecycle, root cause analysis,
    or business understanding, you can gently steer the question to touch those.
  - Where possible, pick a project that can improve coverage for a high-importance / low-coverage JD axis.

  Example styles:
    • "In your current role, what do you mainly work on day to day, and who do you work with most?"
    • "What is one project in your current role that best matches this JD, and what changed because of it?"
    • "For that project, what problem were you solving, and how did things improve in the end?"

PHASE 2 (Questions 4–(max_questions-1)): "Deep dive into 1–2 key projects"
- From question 4 up to the second-to-last question, your questions should:
  - Deep dive into 1–2 key JD-relevant projects (prefer ones with strong tools/skills overlap with the JD).
  - For each project, zoom into:
      - The problem/goal
      - The candidate's personal role and decisions
      - The tools / methods they used
      - The outcome, with metrics/KPIs where possible
  - Where relevant, explicitly target:
      - Missing competencies from the analysis (e.g. product lifecycle, root cause analysis,
        experimentation mindset, stakeholder management).
      - High-importance JD axes that currently have low coverage.
  - Keep questions simple and concrete, one or two ideas per question.

  Example styles:
    • "For that project, what was the main KPI or metric you were trying to improve?"
    • "What specific steps did you take to solve that problem, and which tools did you use most?"
    • "How big was the data or scale you were dealing with in that project?"
    • "In that project, how did you work with stakeholders or other teams?"
    • "Can you walk me through how you identified the root cause of the problem and what actions you took?"

PHASE 3 (Last question = when you are at or near the max): "Anything else to highlight"
- For the final question (or when you feel the story is complete), you can ask:
    • "Is there any other project or achievement you’d like to make sure appears clearly on your resume for this JD?"

--------------------------------
STYLE & FORMAT RULES
--------------------------------
- You MUST either:
  1. Ask exactly ONE next question (1–2 sentences, friendly and clear), OR
  2. Reply with exactly the single word: DONE

- If you already have enough information to write a strong, JD-tailored, impact-focused resume:
  - Reply with DONE (no extra text).

- Keep questions user-friendly:
  - Avoid heavy, multi-clause corporate questions.
  - Prefer plain language like:
      "What was the project mainly about?"
      "Which tools did you use most there?"
      "What metric were you trying to improve?"
  - You can use a small 2-step pattern in one sentence:
      "What was the project about, and what was your role?"

- Tie questions back to JD requirements, JD THEMES (AXES), and competencies in a light way:
  - Use the JD vs resume analysis (including competency gaps AND JD Theme Coverage / Axes)
    to choose WHICH project or aspect to ask about,
    but do not try to exhaustively cover every missing skill.
  - Focus on getting rich stories, numbers, competency evidence, and coverage for high-importance / low-coverage axes,
    not a skill checklist.

--------------------------------
CONTEXT
--------------------------------
JOB DESCRIPTION (JD):
{jd_text}

CURRENT RESUME:
{resume_text}

JD vs RESUME ANALYSIS (may contain missing skills, coverage, role_type, gap_analysis JSON, competency_analysis,
and a JD Theme Coverage (Axes) section):
{analysis_text}

INTERVIEW_QA SO FAR:
{json.dumps(interview_qa, indent=2)}

--------------------------------
YOUR OUTPUT
--------------------------------
- Return ONLY the next question text, OR the single token DONE.
No explanations, no JSON, no extra commentary.
"""



# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def get_next_interview_question(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    num_questions_asked: int,
    max_questions: int,
) -> str:
    """
    Call the LLM and get the next Enhancv-style tailoring question.

    Behaviour:
    - Q1–3: current role & responsibilities + strongest/high-impact project + impact (KPIs) woven into the same questions.
    - Q4–(max_questions-1): deep dive into 1–2 key projects (problem → actions → tools → impact).
    - Last question: optional "anything else to highlight" wrap-up.
    - No banned questions. Avoid repeats. Missing skills are used later in resume generation.
    """

    # If we've already hit or exceeded max questions, force DONE
    if num_questions_asked >= max_questions:
        return "DONE"

    prompt = build_next_question_prompt(
        jd_text=jd_text,
        resume_text=resume_text,
        analysis_text=analysis_text,
        interview_qa=interview_qa,
        num_questions_asked=num_questions_asked,
        max_questions=max_questions,
    )

    raw = chat_completion(
        system_msg=(
            "You are an Enhancv-style resume tailoring assistant. "
            "Ask one targeted, story-and-impact-focused question or reply DONE."
        ),
        user_msg=prompt,
        temperature=0.3,
    ).strip()

    text = raw.strip()

    # Normalize DONE / finished conditions
    upper = text.upper()
    if upper in ["DONE", "FINISHED", "NO MORE QUESTIONS"]:
        return "DONE"

    # Guard against exact duplicate question
    if is_duplicate_question(text, interview_qa):
        return fallback_simple_question()

    return text
