from .openai_client import chat_completion


def build_analysis_prompt(jd_text: str, resume_text: str) -> str:
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


def analyze_jd_vs_resume(jd_text: str, resume_text: str) -> str:
    prompt = build_analysis_prompt(jd_text, resume_text)
    return chat_completion(
        system_msg="You are an expert ATS hiring assistant.",
        user_msg=prompt,
        temperature=0.2,
    )
