import json
import io
import re
from typing import List, Dict, Tuple

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from docx import Document

from .openai_client import chat_completion


# ==============================
# JD-FIT RESUME GENERATION PROMPT
# ==============================


def build_jd_fit_resume_prompt(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    previous_resume: str = "",
    judge_feedback: str = "",
) -> str:
    """
    Build the prompt for the JD-fit resume writer.

    IMPORTANT:
    The output MUST follow the SAME STRUCTURE as your Enhancv-style resume:
      1) Name
      2) Title
      3) Contact line
      4) Summary
      5) Experience
      6) Education
      7) Key Achievements
      8) Skills
      9) Projects

    We are changing ONLY the content, not the section order or top-level layout.
    """
    previous_block = ""
    if previous_resume:
        previous_block = f"""
PREVIOUS_ATTEMPT_RESUME (Markdown):
{previous_resume}

"""

    feedback_block = ""
    if judge_feedback:
        feedback_block = f"""
JUDGE_FEEDBACK:
{judge_feedback}

When you rewrite, address the above feedback explicitly.
"""

    return f"""
You are an ATS-optimized resume writer with an Enhancv-style approach.

Your goal:
Create a JD-FIT RESUME that positions the candidate as a strong, authentic
match for the JOB DESCRIPTION, without inventing any fake experience.

You are given:
1. The JOB DESCRIPTION (JD).
2. The original RESUME (with its own sections: name, title, summary, experience,
   education, key achievements, skills, projects).
3. An ANALYSIS comparing JD vs RESUME.
4. A list of INTERVIEW_QA pairs (questions and the candidate's answers).
{previous_block}
{feedback_block}

VERY IMPORTANT – HOW TO USE THIS INPUT:

1. Identity & Header
   - Infer the candidate's name from the original resume.
     * If you are not sure, use "Your Name".
   - For the TITLE line:
     * Prefer using the EXACT job title from the JD (e.g., "Equity Research Data Analyst")
       if it honestly fits the candidate's domain and level.
     * If the JD title is clearly mismatched or too senior, choose a close, truthful
       variant that still uses the main JD keywords (e.g., "Senior Data Scientist" → 
       "Data Scientist – Financial / Market Analytics").
   - Infer contact details (phone, email, location) only if clearly present.
     * If you are not sure, omit missing fields instead of hallucinating.

2. Sections of the Original Resume
   Infer these sections:
   - Summary
   - Experience (one or more roles)
   - Education
   - Key Achievements
   - Skills
   - Projects (if any)

3. How to Rewrite Each Section
   - Keep factual content the same:
     * Same companies, titles, dates, tools actually used.
     * Same real achievements and responsibilities.
   - Rewrite wording to:
     * Better match the JD's language and seniority.
     * Emphasize impact, measurable results, and ownership.
     * Use strong, varied action verbs (designed, implemented, led,
       optimized, automated, improved, analyzed, engineered, etc.).
     * Avoid generic verbs like "worked on", "helped with", "responsible for"
       unless unavoidable.

4. Use INTERVIEW_QA to Enrich
   - Add missing metrics (≈numbers) where the candidate described impact.
   - Add responsibilities and achievements that were said verbally but not clearly
     written in the original resume.
   - Clarify tools, methods, and leadership elements that are relevant for the JD.
   - Surface achievements the candidate is proud of but not fully documented.

5. Authenticity
   - Do NOT invent:
     * New companies,
     * New roles,
     * Tools they never mentioned,
     * Fake metrics or outcomes that are not supported.
   - You may:
     * Rephrase,
     * Reorganize,
     * Add approximate metrics with "≈" when the direction and order-of-magnitude
       are implied but not exact.

6. ATS Standards
   - Single-column text structure.
   - No tables, emojis, icons, or 2-column layouts.
   - Use headings and bullet points.
   - Use JD keywords only where they match real experience.
   - The TITLE line should be JD-aligned as described above to help ATS filters,
     but must remain truthful to the candidate's background.

STRUCTURE REQUIREMENT (CRITICAL):

Your output MUST follow EXACTLY this section order and heading structure in markdown:

1) First line: the candidate's NAME as a markdown heading, e.g.:

# YOUR NAME

2) Second line: the candidate's TITLE, e.g.:

Data Science Analyst

   - This TITLE should normally reuse the JD's job title (or a very close variant)
     when it honestly fits the candidate.

3) Third line: contact line with phone, email, and location on one line.
   Example:

📞 +91-XXXXXXXXXX  |  ✉️ your.email@example.com  |  Hyderabad, India

(Emoji are optional; keep this as simple text if needed, but keep the idea of a
single contact line.)

Then the following sections in this exact order with these headings:

Summary
<one or two short paragraphs; do not use bullets here>

Experience
For each role:
- Company Name | Location
- Job Title | Dates
- 3–6 bullet points.
  - Put the MOST JD-RELEVANT, high-impact bullets FIRST (top 2–3 bullets).
  - Group more routine or less relevant tasks lower in the list.
  - Where possible, include ≈metrics (%, count, time saved, revenue impact, quality
    improvement, etc.).
  - Highlight tools, platforms, and methods that overlap with the JD
    (e.g., Excel, Python, SQL, Power BI, Bloomberg, financial modeling).

Education
- Degree | Institution | Location (if known) | Years (if clearly known)

Key Achievements
For each major achievement:
- Short achievement title on one line (e.g., "Reduced manual data prep by ≈60% via PySpark automation")
- 1–3 bullet points describing the context, actions, and measurable impact.

Skills
- A single line (or couple of lines) listing skills, separated by commas or slashes:
  Programming languages, tools, platforms, analytics skills, domain knowledge,
  and soft skills.
- Ensure JD-relevant skills are clearly visible and appear FIRST in the list.
- For Excel-related skills, prefer the concise label "Advanced Excel" as the
  main skill name; detailed formulas and features (INDEX-MATCH, XLOOKUP, ARRAY
  formulas, Pivot Tables, Power Query, etc.) can be mentioned inside bullets
  under Experience or Projects.

Projects
For each project:
- Project Name | Dates (if known)
- 1–3 bullet points describing:
  - The problem or goal,
  - What was built or analyzed,
  - Tools and methods used,
  - Measurable outcome or impact (≈metrics if needed).

YOU MUST RESPECT THIS STRUCTURE AND SECTION ORDER.
You are allowed to:
- Change and rewrite the content inside sections,
- Improve wording, add metrics, and tailor to the JD,
- But you must keep:
  - The same section names,
  - The same ordering: Summary → Experience → Education → Key Achievements → Skills → Projects.

---

JOB_DESCRIPTION (JD):
{jd_text}

---

ORIGINAL_RESUME:
{resume_text}

---

JD_vs_RESUME_ANALYSIS:
{analysis_text}

---

INTERVIEW_QA (list of objects with 'question' and 'answer'):
{json.dumps(interview_qa, indent=2)}
"""



def generate_single_jd_fit_resume(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    previous_resume: str = "",
    judge_feedback: str = "",
) -> str:
    """
    One-shot JD-fit resume generation (or refinement if previous_resume/judge_feedback provided).
    """
    prompt = build_jd_fit_resume_prompt(
        jd_text=jd_text,
        resume_text=resume_text,
        analysis_text=analysis_text,
        interview_qa=interview_qa,
        previous_resume=previous_resume,
        judge_feedback=judge_feedback,
    )
    return chat_completion(
        system_msg=(
            "You are an ATS-optimized resume writer that MUST follow the exact requested "
            "section structure and order."
        ),
        user_msg=prompt,
        temperature=0.25,
    )


# ==============================
# LLM-AS-JUDGE SCORING
# ==============================


def _extract_json_from_text(raw: str) -> str:
    """
    Helper to pull the first JSON object from a response.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")
    return raw[start : end + 1]


def score_resume_fit(
    jd_text: str,
    resume_markdown: str,
    analysis_text: str,
) -> Tuple[float, str]:
    """
    LLM-as-judge: score closeness between JD and updated resume.
    Returns (score_0_to_1, rationale).
    """
    prompt = f"""
You are an extremely strict hiring evaluator.

Your task:
Evaluate how well the UPDATED_RESUME matches the JOB_DESCRIPTION (JD).

Consider:
- Skills match (must-have and nice-to-have)
- Tools / technologies overlap
- Level and responsibilities
- Domain / problem-space relevance
- Signals of impact, ownership, and outcomes that the JD cares about

Output STRICTLY in this JSON format:

{{
  "score": 0.0,
  "rationale": "short explanation"
}}

Scoring guidelines:
- 0.90–1.00: Excellent match
- 0.80–0.89: Strong but with some gaps
- 0.60–0.79: Partial match
- < 0.60: Weak match

JOB_DESCRIPTION:
{jd_text}

JD_VS_RESUME_ANALYSIS:
{analysis_text}

UPDATED_RESUME (Markdown):
{resume_markdown}
"""
    raw = chat_completion(
        system_msg="You are a strict JD–resume fit judge.",
        user_msg=prompt,
        temperature=0.0,
    )

    try:
        json_str = _extract_json_from_text(raw)
        data = json.loads(json_str)
        score = float(data.get("score", 0.0))
        rationale = str(data.get("rationale", "")).strip()
    except Exception:
        score = 0.0
        rationale = "Could not parse judge response."

    score = max(0.0, min(1.0, score))
    return score, rationale

def generate_tailoring_feedback(
    jd_text: str,
    original_resume: str,
    updated_resume: str,
) -> str:
    """
    Use the LLM to explain to the user WHY their resume was tailored this way
    (title alignment, skills re-ranking, and experience refocus).
    """
    prompt = f"""
You are a friendly resume coach similar to Enhancv.

Explain to the candidate, in three short sections, why their resume was tailored
for the given job description.

Use the JOB DESCRIPTION, ORIGINAL_RESUME, and UPDATED_RESUME to be as specific
as possible, but keep the tone simple and encouraging.

Output STRICTLY in markdown with these exact headings:

### Why we aligned your Title
<1–3 conversational sentences>

### Why we re-ranked your Skills
<1–3 conversational sentences>

### Why we refocused your Experience
<1–3 conversational sentences>

Focus on:
- ATS filters on job titles and keywords,
- pulling JD-relevant skills to the top of the Skills section,
- reordering experience bullets so the most relevant, high-impact points appear first.

JOB DESCRIPTION:
{jd_text}

---

ORIGINAL_RESUME (before tailoring):
{original_resume}

---

UPDATED_RESUME (after tailoring):
{updated_resume}
"""
    return chat_completion(
        system_msg=(
            "You are a concise, friendly resume coach who explains tailoring "
            "decisions in plain language."
        ),
        user_msg=prompt,
        temperature=0.3,
    )



def generate_refined_resume_with_llm_judge(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    target_score: float = 0.85,
    max_attempts: int = 3,
) -> Dict[str, str]:
    """
    Loop:
    1) Generate JD-fit resume.
    2) LLM judge scores closeness to JD.
    3) If score < target_score and attempts < max_attempts:
         use judge feedback to regenerate.

    Returns dict with:
      - "resume": final markdown
      - "score": final score (0–1)
      - "attempts": number of attempts used
      - "judge_rationale": last rationale
    """
    current_resume = ""
    last_rationale = ""
    final_score = 0.0
    attempt = 0

    for attempt in range(1, max_attempts + 1):
        current_resume = generate_single_jd_fit_resume(
            jd_text=jd_text,
            resume_text=resume_text,
            analysis_text=analysis_text,
            interview_qa=interview_qa,
            previous_resume=current_resume,
            judge_feedback=last_rationale,
        )

        score, rationale = score_resume_fit(
            jd_text=jd_text,
            resume_markdown=current_resume,
            analysis_text=analysis_text,
        )
        final_score = score
        last_rationale = rationale

        if score >= target_score:
            break

    return {
        "resume": current_resume,
        "score": final_score,
        "attempts": attempt,
        "judge_rationale": last_rationale,
    }


# ==============================
# MARKDOWN → PDF / DOCX EXPORT
# ==============================


def _md_to_simple_html(text: str) -> str:
    """
    Very small markdown → HTML converter for bold (**text**).
    This is enough for things like **Location:** etc.
    """
    def repl(m):
        return f"<b>{m.group(1)}</b>"

    return re.sub(r"\*\*(.+?)\*\*", repl, text)


def generate_pdf_from_markdown(md_text: str) -> bytes:
    """
    Convert markdown resume text into a more structured, resume-looking PDF:

    - Big centered name at top (# line)
    - Clean section headings (Summary, Experience, Education, etc.)
    - Real bullet lists instead of raw "- " lines
    - Basic bold for things like **Location:**
    """
    buffer = io.BytesIO()

    # Base document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "NameStyle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    contact_style = ParagraphStyle(
        "ContactStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=12,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        alignment=TA_LEFT,
    )
    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        leftIndent=12,
        spaceAfter=1,
    )

    # Known section headings in your template
    SECTION_HEADINGS = {
        "Summary",
        "Professional Summary",
        "Experience",
        "Education",
        "Key Achievements",
        "Skills",
        "Projects",
    }

    lines = md_text.splitlines()
    story = []

    name_set = False
    contact_set = False
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        # Blank line → small spacing
        if not line:
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Name line: first markdown heading "# ..."
        if line.startswith("# ") and not name_set:
            name_text = line[2:].strip()
            story.append(Paragraph(_md_to_simple_html(name_text), name_style))
            name_set = True
            i += 1
            continue

        # Contact line: assume the line following the title or a line that looks like contact info
        if name_set and not contact_set and (
            line.startswith("📞")
            or "@" in line
            or "Location:" in line
            or "Hyderabad" in line  # loosely matches your style
        ):
            story.append(Paragraph(_md_to_simple_html(line), contact_style))
            contact_set = True
            i += 1
            continue

        # Section headings
        if line in SECTION_HEADINGS:
            story.append(Spacer(1, 6))
            story.append(Paragraph(line, section_style))
            story.append(Spacer(1, 2))
            i += 1
            continue

        # Bullet block: gather consecutive "- " lines into one ListFlowable
        if line.startswith("- "):
            bullet_items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                bullet_text = lines[i].strip()[2:].strip()
                bullet_html = _md_to_simple_html(bullet_text)
                bullet_items.append(ListItem(Paragraph(bullet_html, bullet_style)))
                i += 1
            if bullet_items:
                story.append(ListFlowable(bullet_items, bulletType="bullet", start="•"))
            continue

        # Lines that look like "Role | Company | Location | Dates"
        if "|" in line:
            story.append(Spacer(1, 2))
            story.append(Paragraph(_md_to_simple_html(line), normal_style))
            story.append(Spacer(1, 1))
            i += 1
            continue

        # Default: paragraph
        story.append(Paragraph(_md_to_simple_html(line), normal_style))
        i += 1

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_docx_from_markdown(md_text: str) -> bytes:
    """
    Convert markdown text into a basic Word document.
    """
    doc = Document()
    for line in md_text.split("\n"):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
