# core/resume_extractor.py

import json
from typing import Any, Dict

from .openai_client import chat_completion
from .resume_model import ResumeModel


EXTRACTION_SYSTEM_MSG = (
    "You convert resumes into a strict JSON structure. "
    "You must not add commentary or extra text, only return JSON."
)


EXTRACTION_USER_TEMPLATE = """
You will receive a full resume in Markdown format.

Your job is to extract it into the following JSON schema EXACTLY:

{{
  "name": "",
  "title": "",
  "email": "",
  "phone": "",
  "location": "",
  "linkedin": null or string,
  "summary": "",
  "skills_inline": "",
  "experience": [
    {{
      "role": "",
      "company": "",
      "location": "",
      "start": "",
      "end": "",
      "bullets": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "location": "",
      "year": ""
    }}
  ],
  "achievements": [],
  "projects": [
    {{
      "name": "",
      "bullets": []
    }}
  ]
}}

IMPORTANT RULES:
- Preserve the actual text of bullets and sections as much as possible.
- Do NOT invent achievements or skills that are not clearly stated.
- If a field is missing in the resume, return an empty string, null, or [] as appropriate.
- The 'skills_inline' field should be a single line of comma- or bullet-separated skills.
- The 'bullets' arrays must contain plain text bullet statements (no markdown symbols).

Return ONLY valid JSON. Do NOT wrap it in ```json``` or any other formatting.

RESUME (MARKDOWN):

{resume_markdown}
"""


def _call_extraction_llm(resume_markdown: str) -> Dict[str, Any]:
    """
    Call the LLM once to convert markdown → JSON dict.
    Raises ValueError if JSON cannot be parsed.
    """
    user_msg = EXTRACTION_USER_TEMPLATE.format(resume_markdown=resume_markdown)

    raw = chat_completion(
        system_msg=EXTRACTION_SYSTEM_MSG,
        user_msg=user_msg,
        temperature=0.1,
    )

    # Attempt to parse JSON directly. If model adds noise, try to trim.
    raw_stripped = raw.strip()

    try:
        return json.loads(raw_stripped)
    except json.JSONDecodeError:
        # Very simple JSON object extraction (in case of minor wrapping)
        start = raw_stripped.find("{")
        end = raw_stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find JSON object in LLM output: {raw_stripped[:200]}")

        snippet = raw_stripped[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM output: {e}") from e


def extract_resume_to_json(resume_markdown: str) -> ResumeModel:
    """
    High-level helper:
    - Sends the JD-fit markdown resume to the LLM extractor
    - Parses the JSON
    - Validates / normalizes into ResumeModel
    """
    if not resume_markdown or not resume_markdown.strip():
        raise ValueError("Empty resume markdown passed to extractor.")

    payload = _call_extraction_llm(resume_markdown)

    # Basic defensive cleanup to avoid None vs "" issues
    def _ensure_list(value, item_type=str):
        if not value:
            return []
        if isinstance(value, list):
            return [item_type(v) if not isinstance(v, item_type) else v for v in value]
        return []

    # Normalization hook (optional, but future-proof if you want to tweak structure)
    payload = dict(payload or {})

    # Ensure keys exist at least with sane defaults; pydantic has defaults too,
    # but this avoids KeyError for obvious fields if LLM omits them.
    payload.setdefault("name", "")
    payload.setdefault("title", "")
    payload.setdefault("email", "")
    payload.setdefault("phone", "")
    payload.setdefault("location", "")
    payload.setdefault("linkedin", None)
    payload.setdefault("summary", "")
    payload.setdefault("skills_inline", "")

    payload.setdefault("experience", [])
    payload.setdefault("education", [])
    payload.setdefault("achievements", [])
    payload.setdefault("projects", [])

    # Ensure list-type fields are lists
    payload["achievements"] = _ensure_list(payload.get("achievements", []), str)

    # Finally, validate with ResumeModel (will fill remaining defaults)
    return ResumeModel(**payload)
