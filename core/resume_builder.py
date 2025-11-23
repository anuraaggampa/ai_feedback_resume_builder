"""
resume_builder.py — NEXT-GEN HYBRID WEIGHTING VERSION
-----------------------------------------------------
This file combines:
✔ Your original ATS-safe structure  
✔ New JSON-analysis + skill-cluster engine  
✔ Hybrid weighting (LLM recommended + User override)  
✔ ≥90% JD skill coverage  
✔ Interview QA enrichment  
✔ Lato-based clean PDF export  
✔ Strict markdown output  
"""

import io
import json
import re
from typing import Dict, List, Tuple

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
from core.openai_client import chat_completion


# ============================================================
# LATO FONT REGISTRATION (fallback → Helvetica)
# ============================================================

try:
    pdfmetrics.registerFont(TTFont("Lato", "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("Lato-Bold", "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"))
    DEFAULT_FONT = "Lato"
except Exception:
    DEFAULT_FONT = "Helvetica"


# ============================================================
# HARD + SOFT SKILL ANCHORS (GROUNDING FOR LLM)
# ============================================================

HARD_SKILL_EXAMPLES = [
    "Python", "SQL", "Power BI", "Data analysis", "EDA", "Statistical analysis",
    "Hypothesis testing", "A/B testing", "Data pipelines", "ETL", "ELT",
    "Data cleaning", "Automation workflows", "Advanced Excel", "Power Query",
    "Dashboarding", "KPI design", "Forecasting", "Regression",
    "Machine learning fundamentals", "Database systems", "Reporting",
    "Product analytics"
]

SOFT_SKILL_EXAMPLES = [
    "Communication", "Presentation skills", "Decision-making",
    "Stakeholder management", "Consulting mindset", "Leadership",
    "Collaboration", "Problem solving", "Critical thinking",
    "Time management", "Task prioritization", "Attention to detail"
]

# ============================================================
# SECTION TITLES (for markdown + PDF rendering)
# ============================================================

SECTION_HEADINGS = {
    "Summary",
    "Professional Summary",
    "Experience",
    "Education",
    "Key Achievements",
    "Skills",
    "Projects",
}


def enforce_bold_section_titles(md_text: str) -> str:
    """
    Ensure that section titles like 'Summary', 'Experience', etc.
    are always bold in the markdown output.
    """
    lines = md_text.splitlines()
    out_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip the main name heading and contact line
        if stripped.startswith("# "):
            out_lines.append(line)
            continue

        # If the line is exactly a known section title, convert to bold
        if stripped in SECTION_HEADINGS:
            out_lines.append(f"**{stripped}**")
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


# ============================================================
# SKILLS SECTION NORMALIZATION
# ============================================================

def _split_skill_items(text: str) -> List[str]:
    """
    Split a skills line like:
      'Advanced Excel + SQL + Python | Power BI + Dashboarding'
    into ['Advanced Excel', 'SQL', 'Python', 'Power BI', 'Dashboarding', ...]
    """
    tmp = text.replace("|", ",").replace("+", ",")
    parts = [p.strip(" ,") for p in tmp.split(",")]
    return [p for p in parts if p]


def _rewrite_skills_block(lines: List[str]) -> List[str]:
    """
    Take lines inside the Skills section and normalize patterns like:

      Technical Skills:
      Advanced Excel + SQL + Python | Power BI + Dashboarding...

      Soft Skills:
      Communication + Presentation skills + Stakeholder management | ...

    into:

      - Technical: Advanced Excel, SQL, Python, Power BI, Dashboarding, ...
      - Soft: Communication, Presentation skills, Stakeholder management, ...
    """
    tech_items: List[str] = []
    soft_items: List[str] = []
    other_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Remove optional bullet prefix
        l = stripped.lstrip("-").strip()

        low = l.lower()
        if low.startswith("technical skills:"):
            content = l.split(":", 1)[1].strip()
            tech_items.extend(_split_skill_items(content))
        elif low.startswith("soft skills:"):
            content = l.split(":", 1)[1].strip()
            soft_items.extend(_split_skill_items(content))
        else:
            other_lines.append(line)

    result: List[str] = []

    # Deduplicate while preserving order
    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    if tech_items:
        tech_line = "- Technical: " + ", ".join(dedupe(tech_items))
        result.append(tech_line)
    if soft_items:
        soft_line = "- Soft: " + ", ".join(dedupe(soft_items))
        result.append(soft_line)

    # Include any other lines that we did not parse (for safety)
    result.extend(other_lines)
    return result


def normalize_skills_section(md_text: str) -> str:
    """
    Post-process the markdown to:
    - Keep only one main section heading: 'Skills'
    - Convert inner 'Technical Skills:' / 'Soft Skills:' lines with + and |
      into clean comma-separated lists under bullets:

        - Technical: ...
        - Soft: ...

    This keeps the section ATS-friendly and avoids multiple 'titles'.
    """
    lines = md_text.splitlines()
    out: List[str] = []
    in_skills = False
    buffer_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        base = stripped.strip("*")  # handle **Skills**

        if not in_skills:
            # Detect start of Skills section
            if base == "Skills":
                in_skills = True
                buffer_lines = []
                out.append(line)  # keep the Skills heading as-is (bold or plain)
            else:
                out.append(line)
        else:
            # Inside Skills: detect if this line is actually the start of the next section
            next_base = stripped.strip("*")
            if next_base in SECTION_HEADINGS and next_base != "Skills":
                # We hit a new section → flush the normalized skills block first
                processed = _rewrite_skills_block(buffer_lines)
                out.extend(processed)
                buffer_lines = []
                in_skills = False
                out.append(line)
            else:
                buffer_lines.append(line)

    # If file ended while still inside Skills section → flush buffered block
    if in_skills:
        processed = _rewrite_skills_block(buffer_lines)
        out.extend(processed)

    return "\n".join(out)


# ============================================================
# NEXT-GEN PROMPT BUILDER (HYBRID WEIGHTING + SKILL COVERAGE)
# ============================================================

def build_jd_fit_resume_prompt(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    user_weighting: Dict[str, int],
    previous_resume: str = "",
    judge_feedback: str = "",
) -> str:
    """
    Builds the FULL LLM prompt for JD-fit resume generation.
    Hybrid weighting uses:
      - LLM-inferred role type (from analysis)
      - User override slider (Home → weighting section)
    """

    previous_block = f"\nPREVIOUS_ATTEMPT_RESUME:\n{previous_resume}\n" if previous_resume else ""
    feedback_block = f"\nJUDGE_FEEDBACK:\n{judge_feedback}\n" if judge_feedback else ""

    return f"""
You are an ATS-optimized resume writer with strict structure rules.

===============================================================
### HARD + SOFT SKILL GROUNDING (DO NOT IGNORE)
Use these lists as the base dictionary:

HARD_SKILL_EXAMPLES:
{HARD_SKILL_EXAMPLES}

SOFT_SKILL_EXAMPLES:
{SOFT_SKILL_EXAMPLES}

Your job:
- Identify JD hard skills  
- Identify JD soft skills  
- Ensure ≥ 90% coverage in the FINAL RESUME  

===============================================================
### HYBRID WEIGHTING (CRITICAL)

You MUST combine two signals:

1) **LLM ROLE-TYPE from analysis**
   - If JD is technical → prioritize hard skills (approx 70/30)
   - If JD is non-technical → prioritize soft skills (approx 75/25)

2) **USER OVERRIDE CONTROL**
   User selected:
   - HARD = {user_weighting.get('hard', 70)}%
   - SOFT = {user_weighting.get('soft', 30)}%

Your FINAL weighting MUST blend:
- 50% LLM-inferred role type
- 50% User override

Example final weighting calculation:
final_hard = average(LLM_hard_percent, user_hard_percent)
final_soft = 100 - final_hard

You MUST apply final_hard / final_soft in:
- Summary ordering  
- Top bullets in Experience  
- Skills ordering  
- Projects emphasis  
- Achievement clustering  

===============================================================
### USE CLUSTERS (NOT INDIVIDUAL SKILLS)

Correct examples:
- SQL + Python + Power BI  
- ETL + pipelines + automation  
- Communication + stakeholder mgmt + decision-making  
- Leadership + collaboration + presentation  

Wrong:
- Just “SQL”  
- Just “Communication”

Always use skill clusters.

===============================================================
### USE INTERVIEW_QA TO PATCH MISSING INFO

Rules:
- If candidate revealed a missing tool → add to resume  
- If candidate gave metrics → convert to measurable bullets  
- If a story clarifies ownership/impact → rewrite as an achievement  
- DO NOT fabricate anything.

INTERVIEW DATA:
{json.dumps(interview_qa, indent=2)}

===============================================================
### OUTPUT STRUCTURE (STRICT)

The final resume MUST follow EXACTLY:

# NAME
TITLE
Contact Line

Summary
Experience
Education
Key Achievements
Skills
Projects

Rules:
- No new sections  
- No reordering  
- No tables  
- No icons except contact line emojis  
- All bullets must start with strong verbs  
- Summary must use the weighted skill clusters  

===============================================================
JOB_DESCRIPTION:
{jd_text}

ORIGINAL_RESUME:
{resume_text}

JD_vs_RESUME_ANALYSIS:
{analysis_text}

{previous_block}
{feedback_block}

Your output MUST be pure markdown, with NO commentary.
"""


# ============================================================
# JD-FIT RESUME GENERATION (HYBRID + CLUSTER MODEL)
# ============================================================

def generate_single_jd_fit_resume(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    user_weighting: Dict[str, int],
    previous_resume: str = "",
    judge_feedback: str = "",
) -> str:
    """Generate a *single pass* JD-fit resume (one-shot)."""

    prompt = build_jd_fit_resume_prompt(
        jd_text=jd_text,
        resume_text=resume_text,
        analysis_text=analysis_text,
        interview_qa=interview_qa,
        user_weighting=user_weighting,
        previous_resume=previous_resume,
        judge_feedback=judge_feedback,
    )

    raw_md = chat_completion(
        system_msg="You are an ATS-optimized resume writer. Follow the exact structure.",
        user_msg=prompt,
        temperature=0.25,
    )

    # 1) Enforce bold section titles in markdown so both UI and PDF stay consistent
    md_with_bold = enforce_bold_section_titles(raw_md)

    # 2) Normalize Skills section formatting (commas, one main heading, clean bullets)
    md_normalized = normalize_skills_section(md_with_bold)

    return md_normalized


# ============================================================
# STRICT LLM-AS-JUDGE SCORING
# ============================================================

def _extract_json_from_text(raw: str) -> str:
    """Extract first JSON object from a judge response."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")
    return raw[start:end+1]


def score_resume_fit(
    jd_text: str,
    resume_markdown: str,
    analysis_text: str,
) -> Tuple[float, str]:
    """Strict JD–resume fit judge. Outputs score + rationale."""

    prompt = f"""
You are an extremely strict hiring evaluator.

Evaluate how well the UPDATED_RESUME matches the JOB_DESCRIPTION.

Consider:
- Must-have skills
- Tools / technologies
- Seniority & responsibility match
- Domain relevance
- Impact, ownership, metrics

Output ONLY this JSON:
{{
  "score": 0.0,
  "rationale": "..."
}}

JOB_DESCRIPTION:
{jd_text}

JD_vs_RESUME_ANALYSIS:
{analysis_text}

UPDATED_RESUME:
{resume_markdown}
"""

    raw = chat_completion(
        system_msg="You are a strict JD–resume fit judge.",
        user_msg=prompt,
        temperature=0.0,
    )

    try:
        payload = json.loads(_extract_json_from_text(raw))
        score = float(payload.get("score", 0.0))
        rationale = str(payload.get("rationale", "")).strip()
    except Exception:
        return 0.0, "Judge response parse error."

    score = max(0.0, min(1.0, score))
    return score, rationale


# ============================================================
# MULTI-PASS REFINEMENT WITH LLM JUDGE
# ============================================================

def generate_refined_resume_with_llm_judge(
    jd_text: str,
    resume_text: str,
    analysis_text: str,
    interview_qa: List[Dict[str, str]],
    user_weighting: Dict[str, int],
    target_score: float = 0.92,
    max_attempts: int = 3,
) -> Dict[str, str]:
    """
    Repeated improvement loop:
      1. Generate JD-fit resume
      2. Judge scores it
      3. If score < threshold → refine using judge rationale
    """

    current = ""
    rationale = ""
    final_score = 0.0

    for attempt in range(1, max_attempts + 1):

        current = generate_single_jd_fit_resume(
            jd_text=jd_text,
            resume_text=resume_text,
            analysis_text=analysis_text,
            interview_qa=interview_qa,
            user_weighting=user_weighting,
            previous_resume=current,
            judge_feedback=rationale,
        )

        score, rationale = score_resume_fit(
            jd_text=jd_text,
            resume_markdown=current,
            analysis_text=analysis_text,
        )
        final_score = score

        if score >= target_score:
            break

    return {
        "resume": current,
        "score": final_score,
        "attempts": attempt,
        "judge_rationale": rationale,
    }


# ============================================================
# EXPLANATION BLOCK (WHY WE MADE THESE CHANGES)
# ============================================================

def generate_tailoring_feedback(
    jd_text: str,
    original_resume: str,
    updated_resume: str,
) -> str:
    """Explain WHY the resume changed (ATS title, skills ordering, experience refocus)."""

    prompt = f"""
You are a friendly resume coach like Enhancv.

Explain in **three short blocks** why the resume was rewritten.

Output in EXACT markdown:

### Why we aligned your Title
(2–3 sentences)

### Why we re-ranked your Skills
(2–3 sentences)

### Why we refocused your Experience
(2–3 sentences)

Be specific but encouraging.

JOB_DESCRIPTION:
{jd_text}

ORIGINAL_RESUME:
{original_resume}

UPDATED_RESUME:
{updated_resume}
"""

    return chat_completion(
        system_msg="You are a concise resume coach.",
        user_msg=prompt,
        temperature=0.3,
    )


# ============================================================
# MARKDOWN → SIMPLE HTML (very tiny converter)
# ============================================================

def _md_to_simple_html(text: str) -> str:
    """Convert simple markdown (**bold**) to HTML."""
    def repl(m):
        return f"<b>{m.group(1)}</b>"
    return re.sub(r"\*\*(.+?)\*\*", repl, text)


# ============================================================
# PDF GENERATION (LATO / CLEAN ATS STYLE)
# ============================================================

def generate_pdf_from_markdown(md_text: str) -> bytes:
    """
    Convert markdown resume into a clean ATS-friendly PDF using Lato.

    Features:
    - Big bold name
    - Single-column structure
    - Clean spacing
    - Bullets rendered correctly
    - Sections clearly separated
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    font_name = DEFAULT_FONT

    name_style = ParagraphStyle(
        "NameStyle",
        parent=styles["Title"],
        fontName=f"{font_name}-Bold" if font_name != "Helvetica" else font_name,
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    contact_style = ParagraphStyle(
        "ContactStyle",
        parent=styles["Normal"],
        fontName=font_name,
        alignment=TA_CENTER,
        fontSize=10,
        leading=12,
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontName=f"{font_name}-Bold" if font_name != "Helvetica" else font_name,
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        alignment=TA_LEFT,
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=13,
        spaceAfter=3,
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=13,
        leftIndent=12,
        spaceAfter=2,
    )

    def normalize_heading(line: str) -> str:
        """
        Strip markdown markers (###, **, etc.) so we can
        correctly detect section titles.
        """
        clean = line.strip()

        # Remove leading markdown hashes (##, ###, etc.)
        while clean.startswith("#"):
            clean = clean[1:].lstrip()

        # Remove outer bold/italic markers like **Summary** or *Summary*
        clean = clean.strip("*").strip("_").strip()

        return clean

    lines = md_text.splitlines()
    story = []

    name_set = False
    contact_set = False

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Skip blank lines
        if not line:
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Name line (# ...)
        if line.startswith("# ") and not name_set:
            story.append(Paragraph(_md_to_simple_html(line[2:].strip()), name_style))
            name_set = True
            i += 1
            continue

        # Contact line
        if name_set and not contact_set and (
            "@" in line or "📞" in line or "|" in line
        ):
            story.append(Paragraph(_md_to_simple_html(line), contact_style))
            contact_set = True
            i += 1
            continue

        # SECTION headings (robust to markdown formatting)
        normalized = normalize_heading(line)
        if normalized in SECTION_HEADINGS:
            story.append(Paragraph(_md_to_simple_html(normalized), section_style))
            i += 1
            continue

        # Bullets
        if line.startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                text = lines[i].strip()[2:].strip()
                bullets.append(
                    ListItem(Paragraph(_md_to_simple_html(text), bullet_style))
                )
                i += 1
            if bullets:
                story.append(
                    ListFlowable(bullets, bulletType="bullet", start="•")
                )
            continue

        # Lines like: "Role | Company | Location | Date"
        if "|" in line:
            story.append(Paragraph(_md_to_simple_html(line), normal_style))
            i += 1
            continue

        # Normal paragraph
        story.append(Paragraph(_md_to_simple_html(line), normal_style))
        i += 1

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ============================================================
# DOCX EXPORT
# ============================================================

def generate_docx_from_markdown(md_text: str) -> bytes:
    """Convert markdown resume into basic .docx."""
    doc = Document()
    for line in md_text.split("\n"):
        doc.add_paragraph(line)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
