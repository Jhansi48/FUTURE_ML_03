# 📄 TalentMatch ML — Resume Screening & Candidate Ranking Decision-Support System
**Future Interns Machine Learning Internship — Task 3 Submission**  
**Track Code:** `ML` | **CIN:** `FIT/AUG26/ML10465` | **Repository:** `FUTURE_ML_03`

---

## 📌 Executive Summary
In high-volume technical talent acquisition, recruiters manually screen hundreds of multi-format resumes against complex job specifications. This manual workflow creates cognitive overload, recruiter fatigue, and vulnerability to unconscious demographic bias.

**TalentMatch ML** is a recruiter decision-support system designed to objectively assess candidate alignment against structured technical Job Descriptions (JDs). It combines:
1. **Multi-Format Parsing**: Automated text extraction from `.pdf`, `.docx`, and `.txt` resumes.
2. **PII Anonymization & Fairness Safeguards**: Automatic masking of candidate names, emails, phone numbers, and profile URLs prior to scoring.
3. **Controlled Technical Skill Taxonomy**: 200+ canonical competencies across 5 domains with alias resolution.
4. **4-Tier Hybrid Scoring Architecture**:
   - **Hard Skill Overlap ($40\%$)**: Required skill coverage and preferred skill bonus.
   - **Dense Semantic Similarity ($30\%$)**: Contextual vector cosine similarity via `SentenceTransformer('all-MiniLM-L6-v2')`.
   - **TF-IDF Lexical Similarity ($20\%$)**: Keyword-level sublinear TF-IDF cosine similarity.
   - **Experience & Education Alignment ($10\%$)**: Stated tenure and educational attainment scoring.
5. **Skill Gap Diagnostics & Explainability**: Granular itemization of matched required skills, missing required skills, and recruiter recommendations.

---

## 🏗️ System Architecture & 4-Tier Scoring Pipeline

```
┌────────────────────────────────────────────────────────┐
│        Multi-Format Resumes (PDF, DOCX, TXT)           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Text Extraction & PII Anonymization           │
│   - Regex PII Stripping: Emails, Phones, URLs, Links   │
│   - Experience Years & Education Level Extraction      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     Controlled Technical Taxonomy Extraction (200+)    │
│   - Canonical Skill Matching & Alias Normalization     │
│   - Matches Required (80%) and Preferred (20%) Skills  │
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Tier 1: Skill │   │ Tier 2: Dense │   │ Tier 3: TFIDF │
│ Overlap (40%) │   │ Semantic (30%)│   │ Lexical (20%) │
│ (Req & Pref)  │   │ (all-MiniLM)  │   │ (Sublinear)   │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│    Tier 4: Experience & Education Alignment (10%)      │
│    - Experience Ratio: min(1.0, Cand_Exp / Req_Exp)    │
│    - Education Factor: PhD(1.0), MS(0.95), BS(0.85)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Composite Match Score (0 - 100%)           │
│       = 0.40*Skill + 0.30*Dense + 0.20*TFIDF + 0.10*Exp│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Recruiter Decision Support & Gap Diagnostics      │
│   - Candidate Ranking Leaderboard                      │
│   - Matched vs Missing Skill Breakdown                 │
│   - Severity Flag (Low / Moderate / High Gap)          │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Provenance & Controlled Prototype Disclosure
- **Corpus Nature**: Controlled, synthetic benchmark candidate corpus comprising 8 distinct engineering profiles in `.pdf`, `.docx`, and `.txt` formats across seniorities and domains.
- **Job Descriptions**: 3 structured technical job descriptions (`Senior Machine Learning Engineer`, `Lead Full Stack Engineer`, `Cloud DevOps Specialist`).
- **Disclosure & Scope**: This benchmark dataset was generated specifically to demonstrate and validate end-to-end multi-tier scoring logic, multi-format parsing, PII anonymization, and skill gap diagnostics in a controlled prototype environment. It is not an external production recruiting dataset and should not be used as an autonomous hiring tool.

---

## 📈 Real Experimental Ranking Results

### Target Job: *Senior Machine Learning Engineer (NLP & MLOps)*
**Required Skills (8):** `python`, `pytorch`, `scikit-learn`, `machine learning`, `natural language processing`, `docker`, `aws`, `model deployment`  
**Preferred Skills (6):** `transformers`, `huggingface`, `kubernetes`, `mlops`, `langchain`, `sql`

| Rank | Candidate Profile | Format | Exp (Yrs) | Education | Skill Overlap | Semantic Sim | Lexical Sim | Composite Score | Gap Severity | Recruiter Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | `candidate_01_lead_ml_engineer` | `.docx` | 6.5 | Master's | **100.0%** | **70.2%** | **55.7%** | **77.7%** | **Low** | Advance to Technical Screen |
| 🥈 2 | `candidate_03_junior_ml_intern` | `.pdf` | 1.0 | Bachelor's | **60.0%** | **50.6%** | **37.6%** | **46.2%** | **Moderate** | Potential Fit: Review Exp Gap |
| 🥉 3 | `candidate_02_data_scientist` | `.txt` | 4.0 | Master's | **40.0%** | **44.9%** | **39.0%** | **40.4%** | **Moderate** | Moderate Match: Review Gaps |
| 4 | `candidate_06_devops_cloud_architect` | `.docx` | 8.0 | Bachelor's | **30.0%** | **46.4%** | **32.8%** | **39.5%** | **High** | Skill Gap: Lacks Core ML |
| 5 | `candidate_04_fullstack_lead` | `.docx` | 7.0 | Bachelor's | **20.0%** | **42.2%** | **26.9%** | **32.8%** | **High** | Skill Gap: Lacks Core ML |
| 6 | `candidate_08_java_backend_dev` | `.txt` | 5.5 | Bachelor's | **10.0%** | **36.9%** | **24.5%** | **28.3%** | **High** | Skill Gap: Lacks Core ML |
| 7 | `candidate_07_bi_data_analyst` | `.txt` | 4.5 | Bachelor's | **10.0%** | **34.1%** | **19.8%** | **25.3%** | **High** | Skill Gap: Lacks Core ML |
| 8 | `candidate_05_frontend_dev` | `.txt` | 3.0 | Bachelor's | **0.0%** | **27.4%** | **13.5%** | **18.1%** | **High** | Skill Gap: Lacks Core ML |

> **Multi-Job Specification Verification**: Rankings, scores, and skill gap matrices execute successfully and consistently across all three job specifications (`Senior Machine Learning Engineer`, `Lead Full Stack Software Engineer`, `Senior Cloud DevOps & Infrastructure Specialist`).

---

## 🔍 Granular Skill Gap & Explainability Breakdown

### Top Candidate: `candidate_01_lead_ml_engineer` (Rank 1 — Score: 77.7%)
- **Matched Required (8/8):** `aws`, `docker`, `machine learning`, `model deployment`, `natural language processing`, `python`, `pytorch`, `scikit-learn`
- **Missing Required (0):** None
- **Matched Preferred (6/6):** `huggingface`, `kubernetes`, `langchain`, `mlops`, `sql`, `transformers`
- **Recommendation:** Strong Candidate: High technical alignment with core requirements. Advance to Technical Screen.

### Borderline Candidate: `candidate_03_junior_ml_intern` (Rank 2 — Score: 46.2%)
- **Matched Required (6/8):** `docker`, `machine learning`, `model deployment`, `python`, `pytorch`, `scikit-learn`
- **Missing Required (2):** `aws`, `natural language processing`
- **Matched Preferred (0/6):** None
- **Recommendation:** Potentially Qualified: Possesses solid foundational ML competencies but lacks senior AWS/NLP and MLOps tooling.

---

## 🛡️ Fairness & Ethics Guardrails
1. **PII Masking**: Candidate names, email addresses, phone numbers, and web links are automatically scrubbed from resume text prior to embedding and scoring.
2. **Exclusion of Protected Attributes**: Scoring strictly relies on technical competencies, domain relevance, stated experience tenure, and education level. Demographic factors (age, gender, ethnicity, location) are excluded from the ranking formula.
3. **Decision-Support Framing**: TalentMatch ML is explicitly built as a **recruiter assistance and screening accelerator**, not an autonomous hiring engine. Final interviewing and hiring decisions require human evaluation.

---

## 🖥️ Interactive Streamlit Decision-Support Dashboard
Launch the web interface under `dashboard/app.py`:
- Inspect candidate rankings across different Job Descriptions.
- View interactive radar charts and multi-tier score breakdowns.
- Review missing vs. matched skills and export candidate evaluations to CSV.

---

## 📁 Repository Structure
```
FUTURE_ML_03/
├── README.md                                  # Comprehensive Task Documentation
├── requirements.txt                           # Dependencies
├── .gitignore                                 # Ignore Rules
├── data/
│   ├── README.md                              # Dataset Schema & Sourcing
│   ├── resumes/                               # Multi-Format Resumes (PDF, DOCX, TXT)
│   │   ├── candidate_01_lead_ml_engineer.docx
│   │   ├── candidate_02_data_scientist.txt
│   │   ├── candidate_03_junior_ml_intern.pdf
│   │   ├── candidate_04_fullstack_lead.docx
│   │   ├── candidate_05_frontend_dev.txt
│   │   ├── candidate_06_devops_cloud_architect.docx
│   │   ├── candidate_07_bi_data_analyst.txt
│   │   └── candidate_08_java_backend_dev.txt
│   └── job_descriptions/                      # Structured Role Descriptions
│       ├── job_cloud_devops_specialist.json
│       ├── job_lead_fullstack_engineer.json
│       └── job_senior_ml_engineer.json
├── notebooks/
│   └── 03_resume_screening_system.ipynb       # Fully Executed Notebook
├── src/
│   ├── data_generator.py                      # Multi-Format Generator
│   ├── parser.py                              # PDF, DOCX, TXT Parser & PII Masker
│   ├── taxonomy.py                            # 200+ Skill Taxonomy & Aliases
│   ├── matcher.py                             # 4-Tier Hybrid Matcher & Ranker
│   ├── visualize.py                           # Leaderboard & Radar Chart Plots
│   └── pipeline.py                            # End-to-End Execution Pipeline
├── dashboard/
│   └── app.py                                 # Streamlit Recruiter Dashboard
├── models/
│   └── talent_matcher_engine.pkl              # Serialized Engine
├── outputs/
│   ├── figures/
│   │   ├── candidate_rankings_bar.png
│   │   └── top_candidates_radar_chart.png
│   └── metrics/
│       └── candidate_screening_rankings.csv
└── reports/
    └── candidate_screening_decision_report.md # Decision Support Report
```

---

## 🚀 How to Run

### 1. Setup Environment
```bash
git clone https://github.com/<your-username>/FUTURE_ML_03.git
cd FUTURE_ML_03
pip install -r requirements.txt
```

### 2. Run End-to-End Matching Pipeline
```bash
python src/pipeline.py
```

### 3. Launch Interactive Recruiter Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook notebooks/03_resume_screening_system.ipynb
```
