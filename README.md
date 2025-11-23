
# 🧠 Super Resume Tailor — AI-Powered JD-Fit Resume Builder  
**Interactive • ATS-Optimized • LLM-Refined • PDF Export**

This project is a fully AI-driven **Job Description–aware resume generator**.  
It reads a candidate's resume, analyzes a job description, asks targeted questions like a hiring manager, and **rewrites the entire resume** into a clean, ATS-safe structure.

🚀 Designed for real-world use: HR, LinkedIn job applications, job seekers, and automated resume tailoring workflows.

---

## 🎯 Key Features

### ✅ 1. **Job Description Analysis**
- Extracts required skills, responsibilities, and key competencies.
- Highlights gaps vs. the candidate's resume.

### ✅ 2. **AI Interview Engine**
- Asks up to 7 dynamic LLM-generated questions.
- First 3 explore past roles and responsibilities.
- Next 4 tailor skills, impact metrics, tools, and leadership experience.
- Adapts questions using context from JD + Resume.

### ✅ 3. **JD-Fit Resume Generation**
- Rewrites entire resume into a **clean, ATS-friendly, Enhancv-style format**:
  - Name  
  - Title  
  - Contact line  
  - Summary  
  - Experience  
  - Education  
  - Key Achievements  
  - Skills  
  - Projects  
- No hallucinations — strictly uses resume + interview answers.

### ✅ 4. **LLM-as-Judge Refinement Loop**
- Each generated resume is scored for JD-fit.
- If score < 0.85 → auto-improves using judge feedback.
- Up to 3 refinement attempts.

### ✅ 5. **Professional Exports**
- Export final resume as:
  - **PDF** (nice layout with headings & bullets)
  - **DOCX** (editable)

### ✅ 6. **Streamlit Web UI**
- Upload Resume (PDF or DOCX)
- Upload Job Description
- View JD vs. Resume Analysis
- Chat-style Interview
- Generate Final Resume + Download

---

## 🏗️ System Architecture

```
           ┌──────────────────────────────────────────────┐
           │                User Interface                 │
           │                (Streamlit App)                │
           └───────────────┬──────────────────────────────┘
                           │
                           ▼
        ┌────────────────────────────────────────────────────────┐
        │                   Core Logic (Python)                  │
        │  - Resume parser (Docling / PyPDF2 fallback)           │
        │  - JD parser                                           │
        │  - JD vs Resume analysis                               │
        │  - Interview engine                                    │
        │  - Resume generator                                    │
        │  - LLM as Judge (score + refine loop)                  │
        └───────────────┬────────────────────────────────────────┘
                        │
                        ▼
             ┌────────────────────────┐
             │  OpenAI API (Chat)     │
             │  - Content Rewriting   │
             │  - Skill Extraction    │
             │  - Question Generation │
             │  - Resume Scoring      │
             └────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────────────┐
        │               Output Formatter (PDF/DOCX)               │
        │  - Markdown → PDF (ReportLab custom styling)            │
        │  - Markdown → DOCX                                      │
        └────────────────────────────────────────────────────────┘
```

---

## 📂 Folder Structure

```
resume_builder/
│── app.py                    # Streamlit front-end
│── .env                      # Your OpenAI API Key
│── requirements.txt
│── README.md
│
├── core/
│   ├── resume_builder.py     # JD-fit Resume Generator
│   ├── interview.py          # LLM dynamic interview
│   ├── analysis.py           # JD vs Resume gap analysis
│   ├── openai_client.py      # Wrapper for OpenAI Chat API
│   └── utils.py              # Helpers
│
└── archive/                  # old versions or backups
```

---

## 📦 Installation

### 1️⃣ Clone the project
```bash
git clone <your_repo_url>
cd resume_builder
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts.activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add your OpenAI API key
Create `.env` at root:

```
OPENAI_API_KEY="your_api_key_here"
```

### 5️⃣ Run Streamlit App
```bash
streamlit run app.py
```

---

## 🧪 How It Works — End-to-End Flow

### **Step 1 → Upload Resume + Job Description**
App extracts text using Docling or PyPDF2.

### **Step 2 → JD vs. Resume Gap Analysis**
You see:
- Missing skills  
- Strengths  
- Keyword match rate  
- Word cloud  

### **Step 3 → Interactive AI Interview**
LLM asks:
- Role context  
- Metrics  
- Hard problems solved  
- Leadership  
- JD-specific gaps  

### **Step 4 → JD-Fit Resume Generation**
Model rewrites the entire resume:
- Strong action verbs
- Measurable achievements
- Skills aligned to JD
- No invented experience

### **Step 5 → LLM-as-Judge (Auto-Improve)**
1. Model writes the resume  
2. Judge scores match  
3. If score < 0.85 → refine  
4. Repeat up to 3 attempts  

### **Step 6 → Download Final Resume**
- PDF (professionally formatted)
- DOCX (editable)

---

## 📄 Sample Inputs / Outputs

### Sample JD-fit resume generated  
(Uses your uploaded file)

🔗 [Download Sample Output](sandbox:/mnt/data/jd_fit_resume.pdf)

### Sample Original Resume Used  
🔗 [Download Original Resume](sandbox:/mnt/data/test.pdf)

---

## 🎨 Why This Project Stands Out

✔ Uses **LLM as interviewer**  
✔ Uses **LLM as writer**  
✔ Uses **LLM as evaluator**  
✔ Uses **multi-stage reasoning**  
✔ ATS-focused  
✔ Fully modular architecture  
✔ PDF + DOCX export  
✔ Professional UI  
✔ Great for LinkedIn / GitHub portfolio

---

## 🛠️ Future Improvements

- Add a “beautiful theme mode” with HTML → PDF (WeasyPrint)
- Additional ATS-scoring model
- Multi-file portfolio parser
- Recruiter feedback generator
- LinkedIn auto-fill integration

---

## 🤝 Contributing

Pull requests are welcome!  
Open an issue if you’d like features added.

---

## ⭐ If you like this project, give it a star!

Your support helps the project grow.
