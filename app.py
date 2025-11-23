import streamlit as st

from core.pdf_utils import extract_text_from_pdf
from core.analysis import analyze_jd_vs_resume
from core.interview import get_next_interview_question
from core.resume_builder import (
    generate_refined_resume_with_llm_judge,
    generate_pdf_from_markdown,
    generate_docx_from_markdown,
)
from core.skills_map import load_spacy_model, compute_keyword_freqs, make_wordcloud_html

MAX_QUESTIONS = 7


@st.cache_resource
def get_nlp():
    return load_spacy_model()


nlp = get_nlp()

# ------------- Streamlit Config -------------

st.set_page_config(
    page_title="Super Resume Tailor – JD-Fit Builder",
    page_icon="🧠",
    layout="wide",
)

# ------------- Session State -------------

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "tailored_resume" not in st.session_state:
    st.session_state.tailored_resume = None
if "final_score" not in st.session_state:
    st.session_state.final_score = None
if "judge_rationale" not in st.session_state:
    st.session_state.judge_rationale = None

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


# ------------- Sidebar Navigation -------------

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
    "3. Home → Generate JD-fit resume (LLM judge-refined)"
)

# ------------- HOME PAGE -------------

if st.session_state.page == "Home":
    st.title("🧠 Super Resume Tailor – JD-Fit Resume Builder")
    st.caption(
        "Upload JD & Resume → Analyze → (Optional) dynamic interview → "
        "Generate an ATS-friendly JD-fit resume, refined by an LLM judge (>85% JD match, up to 3 attempts)."
    )

    # Step 1 – Uploads
    st.subheader("1️⃣ Upload Job Description & Resume")
    col_jd, col_res = st.columns(2)

    with col_jd:
        st.markdown("**Job Description**")
        jd_file = st.file_uploader(
            "Upload JD (PDF) – or use text input below",
            type=["pdf"],
            key="jd_file",
        )
        jd_text_manual = st.text_area(
            "Or paste JD text here",
            height=180,
            placeholder="Paste the JD from LinkedIn / careers page...",
        )

    with col_res:
        st.markdown("**Candidate Resume**")
        resume_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            key="resume_file",
        )

    jd_text_final = ""
    if jd_file is not None:
        jd_text_final = extract_text_from_pdf(jd_file)
    elif jd_text_manual.strip():
        jd_text_final = jd_text_manual.strip()

    resume_text_final = ""
    if resume_file is not None:
        resume_text_final = extract_text_from_pdf(resume_file)

    with st.expander("Preview Parsed Text (optional)"):
        st.markdown("**Job Description Text**")
        st.text_area("JD Preview", jd_text_final, height=140)
        st.markdown("**Resume Text**")
        st.text_area("Resume Preview", resume_text_final, height=140)

    # Step 2 – Analysis
    st.subheader("2️⃣ Analyze JD–Resume Match (no changes yet)")
    analyze_btn = st.button(
        "🔍 Analyze JD–Resume Match",
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

        with st.spinner("Analyzing skills, tools, responsibilities and gaps..."):
            analysis_report = analyze_jd_vs_resume(jd_text_final, resume_text_final)
        st.session_state.analysis = analysis_report

    if st.session_state.analysis:
        st.markdown("### 🧩 JD vs Resume Report")
        st.markdown(st.session_state.analysis)

        st.markdown("### ☁️ Skills Map (Word Cloud Style)")
        jd_freqs = compute_keyword_freqs(nlp, st.session_state.jd_text)
        resume_freqs = compute_keyword_freqs(nlp, st.session_state.resume_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### JD Skills Map")
            st.caption("Keywords from the JD. Bigger = more frequent / important.")
            st.markdown(make_wordcloud_html(jd_freqs), unsafe_allow_html=True)
        with col2:
            st.markdown("#### Candidate Skills Map")
            st.caption("Keywords from your resume. Bigger = more frequently mentioned.")
            st.markdown(make_wordcloud_html(resume_freqs), unsafe_allow_html=True)

        st.info(
            "This step only analyzes the fit. Your resume is not changed yet. "
            "You can optionally go to the Interview page so the AI can ask you "
            "targeted follow-up questions."
        )

        if st.button("💬 Go to Interview (optional but recommended)"):
            st.session_state.page = "Interview"
            st.rerun()

    # Step 4 – Generate JD-fit resume with LLM judge refinement
    st.subheader("4️⃣ Generate JD-Fit, ATS-Friendly Resume (LLM judge-refined)")

    generate_btn = st.button(
        "✨ Generate JD-Fit Resume (refine up to 3 attempts)",
        type="primary",
        disabled=not (st.session_state.analysis and st.session_state.resume_text and st.session_state.jd_text),
    )

    if generate_btn:
        with st.spinner("Building and refining your JD-fit resume..."):
            result = generate_refined_resume_with_llm_judge(
                jd_text=st.session_state.jd_text,
                resume_text=st.session_state.resume_text,
                analysis_text=st.session_state.analysis,
                interview_qa=st.session_state.interview_qa,
                target_score=0.85,
                max_attempts=3,
            )
        st.session_state.tailored_resume = result["resume"]
        st.session_state.final_score = result["score"]
        st.session_state.judge_rationale = result["judge_rationale"]

    if st.session_state.tailored_resume:
        st.markdown("### 5️⃣ Your JD-Fit, ATS-Friendly Resume")
        st.markdown(st.session_state.tailored_resume)

        if st.session_state.final_score is not None:
            pct = round(st.session_state.final_score * 100, 1)
            st.markdown(f"**LLM Judge Match Score:** `{pct}%` (target ≥ 85%)")
            if st.session_state.judge_rationale:
                with st.expander("Judge Rationale (Why this score?)"):
                    st.markdown(st.session_state.judge_rationale)

        st.markdown("### ✅ ATS-Friendly Checklist")
        st.markdown(
            "- Single-column layout (recreate in Word/Docs).\n"
            "- Clear section headings (Summary, Skills, Experience, etc.).\n"
            "- Bullet points with strong action verbs and impact.\n"
            "- JD keywords integrated where they match your real experience.\n"
            "- No tables, text boxes, or multi-column layouts."
        )

        # Download options: Markdown, PDF, Word
        colA, colB, colC = st.columns(3)

        with colA:
            st.download_button(
                "⬇️ Download as Markdown",
                data=st.session_state.tailored_resume.encode("utf-8"),
                file_name="jd_fit_resume.md",
                mime="text/markdown",
            )

        with colB:
            pdf_data = generate_pdf_from_markdown(st.session_state.tailored_resume)
            st.download_button(
                "⬇️ Download as PDF",
                data=pdf_data,
                file_name="jd_fit_resume.pdf",
                mime="application/pdf",
            )

        with colC:
            docx_data = generate_docx_from_markdown(st.session_state.tailored_resume)
            st.download_button(
                "⬇️ Download as Word",
                data=docx_data,
                file_name="jd_fit_resume.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )

# ------------- INTERVIEW PAGE -------------

else:
    st.title("💬 Super Resume Tailor – Interview")
    st.caption(
        "The AI will ask up to 7 targeted questions.\n"
        "- First 3: understand your previous roles.\n"
        "- Next ones: tailor your resume to this JD."
    )

    if not st.session_state.analysis or not st.session_state.jd_text or not st.session_state.resume_text:
        st.warning(
            "You need to first go to the Home page, upload JD & resume, and run "
            "the JD–Resume analysis. Then come back here."
        )
    else:
        st.info(
            "These questions are based on your current resume, the JD, and the gaps we detected. "
            "Try to mention real metrics, tools, and responsibilities."
        )

        # Kick off first question if interview hasn't started
        if (
            not st.session_state.interview_chat
            and not st.session_state.interview_done
            and st.session_state.num_questions_asked == 0
        ):
            ai_q = get_next_interview_question(
                jd_text=st.session_state.jd_text,
                resume_text=st.session_state.resume_text,
                analysis_text=st.session_state.analysis,
                interview_qa=st.session_state.interview_qa,
                num_questions_asked=st.session_state.num_questions_asked,
                max_questions=MAX_QUESTIONS,
            )
            if ai_q.upper() == "DONE":
                st.session_state.interview_done = True
                st.session_state.interview_chat.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Looks like your resume already has enough detail for this JD. "
                            "You can go back to the Home page and click "
                            "**“✨ Generate JD-Fit Resume”**."
                        ),
                    }
                )
            else:
                st.session_state.num_questions_asked += 1
                st.session_state.last_question = ai_q
                st.session_state.interview_chat.append(
                    {"role": "assistant", "content": ai_q}
                )

        # Render chat
        for msg in st.session_state.interview_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not st.session_state.interview_done:
            user_input = st.chat_input("Type your answer here...")
            if user_input:
                st.session_state.interview_chat.append(
                    {"role": "user", "content": user_input}
                )
                if st.session_state.last_question:
                    st.session_state.interview_qa.append(
                        {
                            "question": st.session_state.last_question,
                            "answer": user_input,
                        }
                    )

                if st.session_state.num_questions_asked >= MAX_QUESTIONS:
                    st.session_state.interview_done = True
                    st.session_state.interview_chat.append(
                        {
                            "role": "assistant",
                            "content": (
                                "✅ Thanks! I now have enough information "
                                f"(we reached the limit of {MAX_QUESTIONS} questions). "
                                "Go back to the Home page and click "
                                "**“✨ Generate JD-Fit Resume”**."
                            ),
                        }
                    )
                else:
                    ai_q = get_next_interview_question(
                        jd_text=st.session_state.jd_text,
                        resume_text=st.session_state.resume_text,
                        analysis_text=st.session_state.analysis,
                        interview_qa=st.session_state.interview_qa,
                        num_questions_asked=st.session_state.num_questions_asked,
                        max_questions=MAX_QUESTIONS,
                    )
                    if ai_q.upper() == "DONE":
                        st.session_state.interview_done = True
                        st.session_state.interview_chat.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "✅ Thank you, I have enough information to tailor "
                                    "your resume to this JD. Go back to the Home page "
                                    "and click **“✨ Generate JD-Fit Resume”**."
                                ),
                            }
                        )
                    else:
                        st.session_state.num_questions_asked += 1
                        st.session_state.last_question = ai_q
                        st.session_state.interview_chat.append(
                            {"role": "assistant", "content": ai_q}
                        )

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()
