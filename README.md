# 🎯 TalentMatch ML — Intelligent Resume Screening, Skill Extraction & Candidate Ranking System
**Future Interns Machine Learning Internship — Task 3 Submission**  
**Track Code:** `ML` | **CIN:** `FIT/AUG26/ML10465` | **Repository:** `FUTURE_ML_03`

---

## 📌 Executive Summary
In technical recruitment and HR operations, manual resume screening is often time-consuming, prone to cognitive bias, and susceptible to superficial keyword stuffing.

**TalentMatch ML** is an explainable, decision-support Machine Learning pipeline that screens and ranks candidate resumes against structured job descriptions. The system combines:
1. **Multi-Format Document Parsing** (`.pdf`, `.docx`, `.txt`) with automatic **PII anonymization** (masking names, emails, phone numbers, and demographics) to ensure non-discriminatory candidate evaluation.
2. **Controlled Skill Taxonomy (200+ competencies)** with robust alias mapping (e.g., `k8s` -> `kubernetes`, `sklearn` -> `scikit-learn`).
3. **Multi-Tier Hybrid Scoring Architecture**:
   - **Tier 1 (40% Weight):** Hard Skill Overlap (Mandatory vs. Preferred Skills).
   - **Tier 2 (30% Weight):** Dense Semantic Similarity via `SentenceTransformer` (`all-MiniLM-L6-v2`).
   - **Tier 3 (20% Weight):** TF-IDF Lexical Cosine Similarity.
   - **Tier 4 (10% Weight):** Experience & Educational Seniority Alignment.
4. **Transparent Skill-Gap Diagnostics**: Explicitly itemizes missing mandatory skills, preferred bonuses, and actionable hiring recommendations.

---

## 🏗️ System Architecture & Hybrid Matching Pipeline

```
┌────────────────────────────────────────────────────────┐
│           Multi-Format Resumes (PDF/DOCX/TXT)          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Document Parser & PII Anonymization Layer         │
│   - Masks Email, Phone, URLs, Demographic Cues         │
│   - Extracts Experience Years & Education Level        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│    Controlled Skill Extraction & Alias Resolution      │
│   (200+ Taxonomy: ML, Cloud, DevOps, Web, Databases)   │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Tier 1: Skill Overlap │      │  Tier 2: Dense Vectors │
│  (Mandatory/Preferred) │      │  (SentenceTransformer) │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            ├───────────────┬───────────────┤
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Tier 3: TF-IDF Cosine │      │  Tier 4: Experience    │
│  (Lexical Context)     │      │  Alignment Factor      │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Composite Candidate Ranking (0 - 100%)        │
│   - Rank Order & Score Breakdown                       │
│   - Missing Skill Gap Diagnostics & Severity (Low/Mod) │
│   - Decision-Support Hiring Recommendation Narrative   │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation Benchmark & Candidate Ranking Results
Rankings evaluated against the target role: **Senior Machine Learning Engineer (NLP & MLOps)** (`job_senior_ml_engineer.json`):

| Rank | Candidate Profile | Composite Match Score | Skill Match | Semantic Sim | Lexical Sim | Exp (Yrs) | Skill Gap | Decision Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | `candidate_01_lead_ml_engineer` | **77.9%** | 86.7% | 76.8% | 45.4% | 6.5 yrs | **Low** | **Strong Match:** Meets all mandatory competencies. Advance to technical round. |
| 🥈 | `candidate_02_data_scientist` | **40.5%** | 40.0% | 47.9% | 27.2% | 4.0 yrs | Moderate | **Review:** Missing PyTorch, NLP, AWS, Docker. |
| 🥉 | `candidate_06_devops_cloud_architect` | **40.0%** | 30.0% | 46.2% | 34.0% | 8.0 yrs | High | **Skill Gap:** Strong cloud/DevOps, lacks core ML/NLP frameworks. |
| 4 | `candidate_03_junior_ml_intern` | **38.5%** | 40.0% | 51.7% | 29.8% | 2.0 yrs | Moderate | **Junior:** High potential, lacks MLOps/cloud production experience. |
| 5 | `candidate_04_fullstack_lead` | **33.2%** | 20.0% | 43.1% | 27.2% | 7.0 yrs | High | **Mismatch:** Full stack web profile. |
| 6 | `candidate_08_java_backend_dev` | **28.7%** | 10.0% | 35.8% | 25.1% | 5.5 yrs | High | **Mismatch:** Java backend profile. |
| 7 | `candidate_07_bi_data_analyst` | **25.8%** | 10.0% | 33.1% | 19.3% | 4.5 yrs | High | **Mismatch:** BI / Tableau profile. |
| 8 | `candidate_05_frontend_dev` | **17.8%** | 0.0% | 18.2% | 15.6% | 3.0 yrs | High | **Mismatch:** Frontend React profile. |

---

## 🔍 Detailed Skill-Gap Diagnostics for Top Candidates

### 🥇 Rank 1: Alex Rivera (`candidate_01_lead_ml_engineer.docx`) — **77.9% Match**
- **Matched Mandatory (8/8):** `python`, `pytorch`, `scikit-learn`, `machine learning`, `natural language processing`, `docker`, `aws`, `model deployment`
- **Missing Mandatory (0):** *None*
- **Matched Preferred (5/6):** `transformers`, `huggingface`, `kubernetes`, `mlops`, `langchain`
- **Gap Severity:** `Low`

### 🥈 Rank 2: Sarah Chen (`candidate_02_data_scientist.txt`) — **40.5% Match**
- **Matched Mandatory (4/8):** `python`, `scikit-learn`, `machine learning`, `aws`
- **Missing Mandatory (4/8):** `pytorch`, `natural language processing`, `docker`, `model deployment`
- **Gap Severity:** `Moderate`

---

## 🛡️ Fairness, Bias Mitigation & Ethical AI
1. **PII Stripping:** Contact details (emails, phone numbers, addresses, social handles) are masked before feature processing.
2. **Protected Attribute Isolation:** Gender, age, ethnicity, and personal background are never utilized in model scoring.
3. **Decision-Support Scope:** Clearly designated as an assistance tool for initial filtering. Final hiring decisions must always incorporate human interview stages.

---

## 🖥️ Interactive Streamlit Dashboard
Recruiters and hiring managers can launch the dashboard (`dashboard/app.py`) to:
- Select from built-in job descriptions or paste custom job requirements.
- Inspect live candidate leaderboards with score distributions.
- View deep-dive **Skill Gap Cards** with green (matched) and red (missing) badges.
- Export candidate ranking reports directly to CSV.

---

## 📁 Repository Structure
```
FUTURE_ML_03/
├── README.md                                  # Comprehensive Documentation
├── requirements.txt                           # Dependencies
├── .gitignore                                 # Ignore Rules
├── data/
│   ├── README.md                              # Dataset & Schema Details
│   ├── resumes/                               # Candidate Resumes (DOCX & TXT)
│   └── job_descriptions/                      # Structured Role Requirements (JSON)
├── notebooks/
│   └── 03_resume_screening_system.ipynb      # Fully Executed Jupyter Notebook
├── src/
│   ├── taxonomy.py                            # 200+ Skills & Alias Mapping
│   ├── parser.py                              # Document Parsing & PII Anonymizer
│   ├── matcher.py                             # Hybrid Scoring & Skill-Gap Engine
│   ├── data_generator.py                      # Multi-Format Resume Generator
│   ├── visualize.py                           # Leaderboard & Diagnostic Plotter
│   └── pipeline.py                            # End-to-End Execution Pipeline
├── dashboard/
│   └── app.py                                 # Streamlit Screening Web App
├── models/
│   └── talent_matcher_engine.pkl              # Serialized Matcher Pipeline
├── outputs/
│   ├── figures/
│   │   ├── candidate_ranking_leaderboard.png
│   │   ├── score_component_breakdown.png
│   │   └── skill_gap_matrix.png
│   └── metrics/
│       └── candidate_screening_rankings.csv
└── reports/
    └── candidate_screening_decision_report.md # Executive HR Decision Report
```

---

## 🚀 How to Run

### 1. Setup Environment
```bash
git clone https://github.com/<your-username>/FUTURE_ML_03.git
cd FUTURE_ML_03
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
```bash
python src/pipeline.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook notebooks/03_resume_screening_system.ipynb
```
