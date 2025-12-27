import streamlit as st
import json  # ✅ needed for json.loads

from core.pdf_utils import extract_text_from_pdf
from core.analysis import analyze_jd_vs_resume
from core.interview import get_next_interview_question
from pathlib import Path  # NEW
from core.openai_client import reset_usage_log, get_usage_summary_with_cost
from core.resume_builder import (
    generate_refined_resume_with_llm_judge,
    generate_pdf_from_markdown,      # keep for now (fallback)
    generate_docx_from_markdown,     # keep for now (fallback)
    generate_tailoring_feedback,
    generate_pdf_export,             # NEW
    generate_docx_export,            # NEW
)
from core.analysis import assert_nltk_ready
from core.skills_map import load_spacy_model, compute_keyword_freqs, make_wordcloud_html
assert_nltk_ready()
MAX_QUESTIONS = 8

@st.cache_resource
def get_nlp():
    return load_spacy_model()

nlp = get_nlp()

# ---------------- Session State ----------------

DEFAULT_WEIGHTS = {
    "hard": 70,
    "soft": 30
}

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "analysis_json" not in st.session_state:
    st.session_state.analysis_json = None
if "analysis_full" not in st.session_state:
    st.session_state.analysis_full = None

if "user_weighting" not in st.session_state:
    st.session_state.user_weighting = DEFAULT_WEIGHTS.copy()

if "tailored_resume" not in st.session_state:
    st.session_state.tailored_resume = None
if "final_score" not in st.session_state:
    st.session_state.final_score = None
if "judge_rationale" not in st.session_state:
    st.session_state.judge_rationale = None
if "tailoring_feedback" not in st.session_state:
    st.session_state.tailoring_feedback = None

if "interview_qa" not in st.session_state:
    st.session_state.interview_qa = []
if "num_questions_asked" not in st.session_state:
    st.session_state.num_questions_asked = 0
if "interview_chat" not in st.session_state:
    st.session_state.interview_chat = []
if "interview_done" not in st.session_state:
    st.session_state.interview_done = False
if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "page" not in st.session_state:
    st.session_state.page = "Home"


def reset_interview():
    st.session_state.interview_qa = []
    st.session_state.num_questions_asked = 0
    st.session_state.interview_chat = []
    st.session_state.interview_done = False
    st.session_state.last_question = None


# ---------------- Layout / Navigation ----------------

st.set_page_config(
    page_title="Super Resume Tailor – JD-Fit Builder",
    page_icon="🧠",
    layout="wide",
)

st.sidebar.title("Super Resume Tailor")
page_choice = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "💬 Interview"],
    index=0 if st.session_state.page == "Home" else 1,
)
st.session_state.page = "Home" if page_choice.startswith("🏠") else "Interview"

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Flow:**\n"
    "1. Home → Upload & Analyze\n"
    "2. Interview → Answer AI questions\n"
    "3. Home → Generate JD-fit resume (LLM judge refined)"
)


# =====================================================
#                     HOME PAGE
# =====================================================

if st.session_state.page == "Home":

    st.title("🧠 Super Resume Tailor – JD-Fit Resume Builder")
    st.caption("Upload JD & resume → Analyze → (Optional) interview → Build ATS-optimized resume")

    # -------- 1) Upload JD + Resume --------
    st.subheader("1️⃣ Upload Job Description & Resume")
    col_jd, col_res = st.columns(2)

    with col_jd:
        st.markdown("**Job Description**")
        jd_file = st.file_uploader("Upload JD (PDF) – or paste manually", type=["pdf"])
        jd_text_manual = st.text_area("Or paste JD text", height=180)

    with col_res:
        st.markdown("**Candidate Resume**")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    jd_text_final = ""
    if jd_file is not None:
        jd_text_final = extract_text_from_pdf(jd_file)
    elif jd_text_manual.strip():
        jd_text_final = jd_text_manual.strip()

    resume_text_final = ""
    if resume_file is not None:
        resume_text_final = extract_text_from_pdf(resume_file)

    with st.expander("Preview Parsed Text"):
        st.markdown("### JD Preview")
        st.text_area("JD", jd_text_final, height=140)
        st.markdown("### Resume Preview")
        st.text_area("Resume", resume_text_final, height=140)

    # -------- 2) Analyze JD vs Resume --------
    st.subheader("2️⃣ Analyze JD–Resume Match")

    analyze_btn = st.button(
        "🔍 Run JD–Resume Analysis",
        type="primary",
        disabled=not (jd_text_final and resume_text_final),
    )

    if analyze_btn:
        st.session_state.jd_text = jd_text_final
        st.session_state.resume_text = resume_text_final

        reset_interview()
        st.session_state.tailored_resume = None
        st.session_state.final_score = None
        st.session_state.judge_rationale = None
        st.session_state.tailoring_feedback = None

        with st.spinner("Analyzing skills and gaps..."):
            analysis_md = analyze_jd_vs_resume(jd_text_final, resume_text_final)

        # Keep the full analysis (with JSON) for internal use
        st.session_state.analysis_full = analysis_md

        # ---------- Extract JSON from the JSON_BLOCK markers ----------
        try:
            start_marker = "JSON_BLOCK_START"
            end_marker = "JSON_BLOCK_END"

            start_idx = analysis_md.index(start_marker)
            end_idx = analysis_md.index(end_marker, start_idx)

            # visible markdown is everything before the marker
            visible_md = analysis_md[:start_idx].rstrip()

            # extract the JSON text between the markers
            json_block = analysis_md[start_idx:end_idx]
            brace_start = json_block.index("{")
            brace_end = json_block.rindex("}")
            json_str = json_block[brace_start:brace_end + 1]

            st.session_state.analysis = visible_md
            st.session_state.analysis_json = json.loads(json_str)

        except Exception:
            # Fallback: show full analysis text but drop JSON parsing
            st.session_state.analysis = analysis_md
            st.session_state.analysis_json = None
            st.warning(
                "Could not parse structured analysis JSON. "
                "Hybrid weighting will fall back to defaults."
            )

    # display analysis results
    if st.session_state.analysis:
        st.markdown("### 🧩 JD vs Resume Analysis Report")
        st.markdown(st.session_state.analysis)

        # Skill maps
        st.markdown("### ☁️ Skill Maps (Word Clouds)")
        jd_freqs = compute_keyword_freqs(nlp, st.session_state.jd_text)
        resume_freqs = compute_keyword_freqs(nlp, st.session_state.resume_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### JD Skills Map")
            st.markdown(make_wordcloud_html(jd_freqs), unsafe_allow_html=True)
        with col2:
            st.markdown("#### Resume Skills Map")
            st.markdown(make_wordcloud_html(resume_freqs), unsafe_allow_html=True)

        st.info(
            "You can now optionally go to the Interview page to fill missing details."
        )

        if st.button("💬 Go to Interview"):
            st.session_state.page = "Interview"
            st.rerun()

    # -------- 3) Hybrid Weighting Controls --------
    if st.session_state.analysis_json:

        st.subheader("3️⃣ Hybrid Skill Weighting (Auto + Override)")

        role_type = st.session_state.analysis_json.get("role_type", "technical")

        auto_hard = 70 if role_type == "technical" else 25
        auto_soft = 30 if role_type == "technical" else 75

        st.markdown(f"**Auto-detected role type:** `{role_type.upper()}`")

        st.markdown("#### Auto weighting suggestion:")
        st.write(f"Hard Skill Weight: **{auto_hard}%**")
        st.write(f"Soft Skill Weight: **{auto_soft}%**")

        st.markdown("#### Override weights (optional):")
        hard_w = st.slider("Hard Skill Weight (%)", 0, 100, auto_hard)
        soft_w = 100 - hard_w

        st.session_state.user_weighting = {
            "hard": hard_w,
            "soft": soft_w
        }

    # -------- 4) Generate Resume --------
    st.subheader("4️⃣ Generate JD-Fit Resume")

    gen_btn = st.button(
        "✨ Generate JD-Fit Resume (LLM judge refined)",
        type="primary",
        disabled=not (
            st.session_state.analysis
            and st.session_state.resume_text
            and st.session_state.jd_text
        ),
    )

    if gen_btn:

        # 🔹 Reset token/cost log for this run
        reset_usage_log()

        with st.spinner("Generating & refining your resume..."):
            result = generate_refined_resume_with_llm_judge(
                jd_text=st.session_state.jd_text,
                resume_text=st.session_state.resume_text,
                analysis_text=st.session_state.analysis,
                interview_qa=st.session_state.interview_qa,
                user_weighting=st.session_state.user_weighting,
                target_score=0.92,
                max_attempts=3,
            )

        # ✅ store generated resume text + judge metadata
        st.session_state.tailored_resume = result["resume"]
        st.session_state.final_score = result["score"]
        st.session_state.judge_rationale = result["judge_rationale"]

        # ✅ structured resume model for template/PDF export (may be None)
        st.session_state.resume_model = result.get("resume_model")

        # ✅ tailoring feedback stays as-is
        st.session_state.tailoring_feedback = generate_tailoring_feedback(
            jd_text=st.session_state.jd_text,
            original_resume=st.session_state.resume_text,
            updated_resume=st.session_state.tailored_resume,
        )

        # 🔹 After generation: compute and print cost to terminal only
        usage_summary = get_usage_summary_with_cost()
        print("\n========== RESUME GENERATION COST ==========")
        print(f"Prompt tokens: {usage_summary['total_prompt_tokens']}")
        print(f"Completion tokens: {usage_summary['total_completion_tokens']}")
        print(f"Total tokens: {usage_summary['total_tokens']}")
        print(f"Estimated API cost (USD): ${usage_summary['total_cost_usd']:.6f}")
        print("============================================\n")

    # -------- Display resume + feedback --------
    if st.session_state.tailored_resume:

        st.markdown("### 5️⃣ Your JD-Fit Resume")
        st.markdown(st.session_state.tailored_resume)

        if st.session_state.final_score is not None:
            pct = round(st.session_state.final_score * 100, 2)
            st.markdown(f"**LLM Judge Match Score:** `{pct}%`")

            if st.session_state.judge_rationale:
                with st.expander("Judge Rationale"):
                    st.markdown(st.session_state.judge_rationale)

        if st.session_state.tailoring_feedback:
            st.markdown("### 🔍 What changed and why")
            st.markdown(st.session_state.tailoring_feedback)

        # ✅ LinkedIn suggestion ONLY if missing in structured resume
        resume_model = st.session_state.get("resume_model")
        linkedin_missing = (
            resume_model is None
            or not getattr(resume_model, "linkedin", None)
        )

        if linkedin_missing:
            st.info(
                "💡 Optional: Add your LinkedIn profile URL — resumes with LinkedIn links "
                "have a higher chance of getting interview callbacks."
            )
            linkedin_manual = st.text_input(
                "LinkedIn Profile URL (optional)",
                key="linkedin_url",
                placeholder="https://www.linkedin.com/in/your-profile"
            )

            # Inject into structured model if available
            if resume_model is not None and linkedin_manual:
                if not resume_model.linkedin:
                    resume_model.linkedin = linkedin_manual
        else:
            linkedin_manual = st.session_state.get("linkedin_url")

        # downloads
        colA, colB, colC = st.columns(3)

        with colA:
            st.download_button(
                "⬇️ Markdown",
                data=st.session_state.tailored_resume.encode("utf-8"),
                file_name="jd_fit_resume.md",
                mime="text/markdown",
            )

        # 🔹 Template-aware PDF export with fallback
        # with colB:
        #     template_path = Path("templates/resume_template_base.docx")
        #     pdf_data = generate_pdf_export(
        #         resume_model=st.session_state.get("resume_model"),
        #         md_text=st.session_state.tailored_resume,
        #         template_path=str(template_path),
        #     )
        #     st.download_button(
        #         "⬇️ PDF (Template)",
        #         data=pdf_data,
        #         file_name="jd_fit_resume.pdf",
        #         mime="application/pdf",
        #     )

        # 🔹 Template-aware DOCX export with fallback
        with colC:
            template_path = Path("templates/resume_template_base.docx")
            docx_data = generate_docx_export(
                resume_model=st.session_state.get("resume_model"),
                md_text=st.session_state.tailored_resume,
                template_path=str(template_path),
            )
            st.download_button(
                "⬇️ Word (.docx)",
                data=docx_data,
                file_name="jd_fit_resume.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )


# =====================================================
#                   INTERVIEW PAGE
# =====================================================

else:

    st.title("💬 Super Resume Tailor – Interview Mode")
    st.caption(
        "AI will ask up to 7 questions to gather deeper experience and missing skills."
    )

    if (
        not st.session_state.analysis
        or not st.session_state.jd_text
        or not st.session_state.resume_text
    ):
        st.warning(
            "Please upload JD/Resume and run the analysis first."
        )
    else:

        st.info(
            "These questions are based on JD gaps, resume content, and hybrid skill weighting."
        )

        # Kick off first question
        if (
            not st.session_state.interview_chat
            and not st.session_state.interview_done
            and st.session_state.num_questions_asked == 0
        ):
            q = get_next_interview_question(
                jd_text=st.session_state.jd_text,
                resume_text=st.session_state.resume_text,
                analysis_text=st.session_state.analysis,
                interview_qa=st.session_state.interview_qa,
                num_questions_asked=st.session_state.num_questions_asked,
                max_questions=MAX_QUESTIONS,
            )

            if q == "DONE":
                st.session_state.interview_done = True
                st.session_state.interview_chat.append({
                    "role": "assistant",
                    "content": "We already have enough information! Return to Home and generate your resume."
                })
            else:
                st.session_state.num_questions_asked += 1
                st.session_state.last_question = q
                st.session_state.interview_chat.append({"role": "assistant", "content": q})

        # Render chat history
        for msg in st.session_state.interview_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        if not st.session_state.interview_done:
            user_input = st.chat_input("Your answer...")
            if user_input:

                st.session_state.interview_chat.append(
                    {"role": "user", "content": user_input}
                )

                if st.session_state.last_question:
                    st.session_state.interview_qa.append({
                        "question": st.session_state.last_question,
                        "answer": user_input,
                    })

                # max question reached
                if st.session_state.num_questions_asked >= MAX_QUESTIONS:
                    st.session_state.interview_done = True
                    st.session_state.interview_chat.append({
                        "role": "assistant",
                        "content": "Thanks! Return to Home and click Generate Resume."
                    })
                else:
                    q = get_next_interview_question(
                        jd_text=st.session_state.jd_text,
                        resume_text=st.session_state.resume_text,
                        analysis_text=st.session_state.analysis,
                        interview_qa=st.session_state.interview_qa,
                        num_questions_asked=st.session_state.num_questions_asked,
                        max_questions=MAX_QUESTIONS,
                    )
                    if q == "DONE":
                        st.session_state.interview_done = True
                        st.session_state.interview_chat.append({
                            "role": "assistant",
                            "content": "Thanks! I have enough info now!"
                        })
                    else:
                        st.session_state.num_questions_asked += 1
                        st.session_state.last_question = q
                        st.session_state.interview_chat.append(
                            {"role": "assistant", "content": q}
                        )

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()
