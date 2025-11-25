# core/resume_model.py

from typing import List, Optional
from pydantic import BaseModel, Field


class ExperienceItem(BaseModel):
    role: str = ""
    company: str = ""
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    degree: str = ""
    institution: str = ""
    location: str = ""
    year: str = ""


class ProjectItem(BaseModel):
    name: str = ""
    bullets: List[str] = Field(default_factory=list)


class ResumeModel(BaseModel):
    """
    Canonical structured representation of a resume used by:
    - JD-fit export
    - DOCX templating
    - PDF generation
    - Language refinement (clichés, complexity, etc.)
    """

    # Header
    name: str = ""
    title: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: Optional[str] = None  # optional

    # Top sections
    summary: str = ""
    skills_inline: str = ""  # single-line, ATS-friendly skills string

    # Body
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
