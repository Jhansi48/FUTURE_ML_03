# 📄 TalentMatch ML — Resume Screening & Candidate Ranking Decision-Support System
**Future Interns Machine Learning Internship — Task 3 Submission**  
**Track Code:** `ML` | **CIN:** `FIT/AUG26/ML10465` | **Repository:** `FUTURE_ML_03`

---

## 📌 1. Project Title & Overview
**TalentMatch ML** is an interpretable, multi-tier candidate screening and decision-support platform designed to assist technical recruiters in evaluating heterogeneous, multi-format resumes against structured job specifications.

---

## 💼 2. Business Problem
In modern high-volume talent acquisition, recruiters manually inspect hundreds of heterogeneous resumes (`.pdf`, `.docx`, `.txt`) against evolving technical job requirements. This manual screening workflow causes:
- **Cognitive Overload & Inconsistent Evaluation**: Evaluating candidate skill coverage manually across varying formats leads to subjective grading and inconsistent shortlisting.
- **Unconscious Demographic Bias**: Reviewing identifiable demographic information (names, contact details) introduces vulnerability to unconscious bias.
- **Unclear Skill Gap Diagnostics**: Simple keyword search tools miss semantic domain depth and fail to differentiate between core mandatory technical competencies and optional preferred skills.

---

## 🎯 3. Objectives
1. **Multi-Format Ingestion**: Ingest and parse candidate documents across `.pdf`, `.docx`, and `.txt` formats.
2. **PII Reduction & Anonymization**: Automatically redact personally identifiable information (`[NAME_MASKED]`, `[EMAIL_MASKED]`, `[PHONE_MASKED]`, `[SOCIAL_PROFILE_MASKED]`, `[URL_MASKED]`) before feature extraction.
3. **Structured Competency Taxonomy**: Extract technical skills against a curated taxonomy covering 107 canonical skills and 32 aliases across 5 domains.
4. **4-Tier Hybrid Scoring Architecture**: Compute transparent, weighted match scores combining taxonomy overlap (40%), dense semantic similarity (30%), TF-IDF lexical match (20%), and experience alignment (10%).
5. **Skill Gap Diagnostics & Decision Support**: Deliver actionable candidate gap diagnostics and recruiter recommendations with explicit human-in-the-loop governance.

---

## ✨ 4. Key Features
- **Multi-Format Parsing**: Robust text extraction for PDF (`pypdf`), DOCX (`python-docx`), and plain text (`.txt`).
- **PII Anonymization**: Pre-scoring redaction of personal names, emails, phone numbers, and profile URLs.
- **Domain-Specific Skill Taxonomy**: Resolves acronyms, frameworks, and tools (e.g. `k8s` → `kubernetes`, `sklearn` → `scikit-learn`).
- **Dense Semantic Embeddings**: Utilizes `SentenceTransformer('all-MiniLM-L6-v2')` to evaluate contextual domain alignment beyond exact keyword matching.
- **Multi-Role Flexibility**: Evaluates candidate pools across distinct technical specifications (DevOps, Full Stack, Machine Learning).
- **Interactive Recruiter Workspace**: Clean, light enterprise SaaS Streamlit interface with candidate profile inspector, gap matrix, and one-click CSV export.

---

## 🏗️ 5. System Architecture
```
┌────────────────────────────────────────────────────────┐
│        Multi-Format Resumes (PDF, DOCX, TXT)           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Text Extraction & PII Anonymization           │
│   - Name Masking: [NAME_MASKED]                        │
│   - Regex PII Stripping: Emails, Phones, URLs, Links   │
│   - Experience Tenure & Education Level Extraction     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Controlled Technical Taxonomy Extraction          │
│   - 107 Canonical Skills + 32 Aliases (5 Domains)      │
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
│    - Informational Degree Attainment Factor            │
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
│   - Canonical Fit Taxonomy (Strong / Moderate / Gap)   │
└────────────────────────────────────────────────────────┘
```

---

## 📊 6. Dataset Provenance & Controlled Prototype Disclosure
- **Corpus Nature**: Controlled, synthetic benchmark candidate corpus comprising **8 distinct engineering profiles** in `.pdf`, `.docx`, and `.txt` formats across junior, mid-level, and senior engineering roles.
- **Job Descriptions**: 3 structured technical job specifications:
  1. `Senior Cloud DevOps & Infrastructure Specialist` (`JOB-DO-003`)
  2. `Lead Full Stack Software Engineer` (`JOB-FS-002`)
  3. `Senior Machine Learning Engineer (NLP & MLOps)` (`JOB-ML-001`)
- **Scope & Limitations**: This synthetic dataset was created specifically to validate multi-format parsing, PII redaction, skill taxonomy extraction, and multi-tier hybrid scoring in a controlled prototype environment. It is not an empirical population recruiting dataset and is not intended for autonomous hiring.

---

## 🛡️ 7. PII Handling & Anonymization
Before feature extraction, vector embedding, or scoring takes place, the resume text undergoes automated PII reduction:
- **Personal Names**: Replaced with `[NAME_MASKED]`.
- **Email Addresses**: Replaced with `[EMAIL_MASKED]`.
- **Phone Numbers**: Replaced with `[PHONE_MASKED]`.
- **Social Profiles & URLs**: Replaced with `[SOCIAL_PROFILE_MASKED]` and `[URL_MASKED]`.

Technical skills, project descriptions, stated experience years, and educational degree attainments are preserved for objective scoring.

---

## 📚 8. Skill Taxonomy
The taxonomy module (`src/taxonomy.py`) contains **107 canonical skills** and **32 aliases** structured across 5 core domains:
1. **Machine Learning & AI** (24 skills): `pytorch`, `tensorflow`, `scikit-learn`, `xgboost`, `transformers`, `huggingface`, `langchain`, `rag`, `mlops`, etc.
2. **Data Engineering & Databases** (23 skills): `python`, `sql`, `postgresql`, `apache spark`, `apache kafka`, `redis`, `snowflake`, `bigquery`, etc.
3. **Cloud & DevOps** (22 skills): `aws`, `docker`, `kubernetes`, `terraform`, `ci/cd`, `github actions`, `prometheus`, `grafana`, `linux`, `bash`, etc.
4. **Software Engineering & Web** (24 skills): `typescript`, `javascript`, `react`, `node.js`, `next.js`, `fastapi`, `graphql`, `rest api`, etc.
5. **Analytics & BI** (14 skills): `pandas`, `numpy`, `tableau`, `power bi`, `a/b testing`, `statistical analysis`, `business intelligence`, etc.

---

## 🧮 9. Scoring Methodology & Mathematical Formula
The **Composite Match Score** is an interpretable linear combination of four evaluation pillars:

$$\text{Composite Match Score} = 0.40 \cdot S_{\text{skill}} + 0.30 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{lexical}} + 0.10 \cdot S_{\text{exp\_edu}}$$

Where:
- **$S_{\text{skill}}$ (Hard Skill Overlap)**:
  $$S_{\text{skill}} = 0.80 \cdot \left(\frac{|\text{Matched Required}|}{|\text{Total Required}|}\right) + 0.20 \cdot \left(\frac{|\text{Matched Preferred}|}{|\text{Total Preferred}|}\right)$$
- **$S_{\text{semantic}}$ (Dense Semantic Fit)**: Cosine similarity of contextual document vectors from `SentenceTransformer('all-MiniLM-L6-v2')`.
- **$S_{\text{lexical}}$ (TF-IDF Lexical Similarity)**: Cosine similarity of sublinear term-frequency inverse document frequency vectors.
- **$S_{\text{exp\_edu}}$ (Experience & Education Fit)**: Prototype heuristic evaluating experience tenure ratio $\min(1.0, \text{Exp} / \text{Req\_Exp})$ combined with educational degree attainment.

---

## 🏷️ 10. Canonical Fit Taxonomy
To ensure 100% consistency across charts, metrics, profile badges, and tables, TalentMatch ML uses three canonical fit tiers:
- **Strong Overall Match** ($\ge 75.0\%$): High multi-factor alignment; core mandatory competencies satisfied with strong domain contextual alignment.
- **Moderate Match** ($40.0\% - 74.9\%$): Partial technical alignment; review specific missing competencies with hiring manager.
- **High Technical Gap** ($< 40.0\%$): Substantial core skill gaps; profile aligns primarily with adjacent domains.

---

## 📈 11. Reproducible Experimental Results Across All 3 Roles

Source of Truth: `outputs/metrics/rankings_all_roles.csv` generated by `python src/pipeline.py`.

### Role 1: Senior Cloud DevOps & Infrastructure Specialist (`JOB-DO-003`)
- **Required (6)**: `aws`, `kubernetes`, `docker`, `terraform`, `ci/cd`, `linux`
- **Preferred (6)**: `python`, `prometheus`, `grafana`, `github actions`, `bash`, `ansible`

| Rank | Candidate Profile | Format | Exp (Yrs) | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | `candidate_06_devops_cloud_architect` | `.docx` | 8.0 | **75.2%** | Strong Overall Match | 100.0% | 70.1% | 23.2% | None (All Satisfied) |
| 🥈 2 | `candidate_01_lead_ml_engineer` | `.docx` | 6.5 | **56.0%** | Moderate Match | 70.0% | 55.1% | 8.2% | `terraform` |
| 🥉 3 | `candidate_04_fullstack_lead` | `.docx` | 7.0 | **43.9%** | Moderate Match | 43.3% | 52.1% | 6.9% | `kubernetes`, `linux`, `terraform` |
| 4 | `candidate_08_java_backend_dev` | `.txt` | 5.5 | **31.1%** | High Technical Gap | 26.7% | 35.2% | 1.8% | `aws`, `ci/cd`, `linux`, `terraform` |
| 5 | `candidate_02_data_scientist` | `.txt` | 4.0 | **26.0%** | High Technical Gap | 16.7% | 30.5% | 1.9% | `ci/cd`, `docker`, `kubernetes`, `linux`, `terraform` |
| 6 | `candidate_03_junior_ml_intern` | `.pdf` | 1.0 | **25.4%** | High Technical Gap | 30.0% | 28.5% | 2.7% | `aws`, `ci/cd`, `kubernetes`, `terraform` |
| 7 | `candidate_07_bi_data_analyst` | `.txt` | 4.5 | **19.6%** | High Technical Gap | 3.3% | 28.2% | 1.1% | `aws`, `ci/cd`, `docker`, `kubernetes`, `linux`, `terraform` |
| 8 | `candidate_05_frontend_dev` | `.txt` | 3.0 | **15.1%** | High Technical Gap | 0.0% | 24.4% | 0.0% | `aws`, `ci/cd`, `docker`, `kubernetes`, `linux`, `terraform` |

---

### Role 2: Lead Full Stack Software Engineer (`JOB-FS-002`)
- **Required (7)**: `typescript`, `javascript`, `react`, `node.js`, `postgresql`, `rest api`, `docker`
- **Preferred (6)**: `next.js`, `graphql`, `redis`, `aws`, `ci/cd`, `system design`

| Rank | Candidate Profile | Format | Exp (Yrs) | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | `candidate_04_fullstack_lead` | `.docx` | 7.0 | **72.8%** | Moderate Match | 100.0% | 65.3% | 18.3% | None (All Satisfied) |
| 🥈 2 | `candidate_01_lead_ml_engineer` | `.docx` | 6.5 | **39.7%** | High Technical Gap | 32.9% | 52.0% | 5.5% | `javascript`, `node.js`, `react`, `rest api`, `typescript` |
| 🥉 3 | `candidate_08_java_backend_dev` | `.txt` | 5.5 | **38.8%** | High Technical Gap | 37.6% | 43.8% | 5.5% | `javascript`, `node.js`, `react`, `typescript` |
| 4 | `candidate_05_frontend_dev` | `.txt` | 3.0 | **36.1%** | High Technical Gap | 37.6% | 42.3% | 7.9% | `docker`, `node.js`, `postgresql`, `rest api` |
| 5 | `candidate_06_devops_cloud_architect` | `.docx` | 8.0 | **34.7%** | High Technical Gap | 18.1% | 53.9% | 8.7% | `javascript`, `node.js`, `postgresql`, `react`, `rest api`, `typescript` |
| 6 | `candidate_02_data_scientist` | `.txt` | 4.0 | **25.0%** | High Technical Gap | 14.8% | 33.9% | 2.5% | `docker`, `javascript`, `node.js`, `react`, `rest api`, `typescript` |
| 7 | `candidate_07_bi_data_analyst` | `.txt` | 4.5 | **16.1%** | High Technical Gap | 0.0% | 23.8% | 0.6% | `docker`, `javascript`, `node.js`, `postgresql`, `react`, `rest api`, `typescript` |
| 8 | `candidate_03_junior_ml_intern` | `.pdf` | 1.0 | **14.4%** | High Technical Gap | 11.4% | 18.5% | 1.9% | `javascript`, `node.js`, `postgresql`, `react`, `rest api`, `typescript` |

---

### Role 3: Senior Machine Learning Engineer (NLP & MLOps) (`JOB-ML-001`)
- **Required (8)**: `python`, `pytorch`, `scikit-learn`, `machine learning`, `natural language processing`, `docker`, `aws`, `model deployment`
- **Preferred (6)**: `transformers`, `huggingface`, `kubernetes`, `mlops`, `langchain`, `sql`

| Rank | Candidate Profile | Format | Exp (Yrs) | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | `candidate_01_lead_ml_engineer` | `.docx` | 6.5 | **76.0%** | Strong Overall Match | 100.0% | 72.2% | 22.7% | None (All Satisfied) |
| 🥈 2 | `candidate_03_junior_ml_intern` | `.pdf` | 1.0 | **43.9%** | Moderate Match | 60.0% | 43.1% | 13.3% | `aws`, `natural language processing` |
| 🥉 3 | `candidate_02_data_scientist` | `.txt` | 4.0 | **39.2%** | High Technical Gap | 43.3% | 35.8% | 6.7% | `docker`, `model deployment`, `natural language processing`, `pytorch` |
| 4 | `candidate_06_devops_cloud_architect` | `.docx` | 8.0 | **39.0%** | High Technical Gap | 33.3% | 49.3% | 6.8% | `machine learning`, `model deployment`, `natural language processing`, `pytorch`, `scikit-learn` |
| 5 | `candidate_04_fullstack_lead` | `.docx` | 7.0 | **30.3%** | High Technical Gap | 20.0% | 39.7% | 4.1% | `machine learning`, `model deployment`, `natural language processing`, `python`, `pytorch`, `scikit-learn` |
| 6 | `candidate_08_java_backend_dev` | `.txt` | 5.5 | **24.8%** | High Technical Gap | 13.3% | 30.7% | 3.4% | `aws`, `machine learning`, `model deployment`, `natural language processing`, `python`, `pytorch`, `scikit-learn` |
| 7 | `candidate_07_bi_data_analyst` | `.txt` | 4.5 | **23.2%** | High Technical Gap | 13.3% | 26.9% | 1.5% | `aws`, `docker`, `machine learning`, `model deployment`, `natural language processing`, `pytorch`, `scikit-learn` |
| 8 | `candidate_05_frontend_dev` | `.txt` | 3.0 | **14.8%** | High Technical Gap | 0.0% | 22.0% | 1.9% | `aws`, `docker`, `machine learning`, `model deployment`, `natural language processing`, `python`, `pytorch`, `scikit-learn` |

---

## ⚖️ 12. Responsible AI, Privacy & Decision Support
1. **Zero Demographic Attribute Bias**: Protected demographic attributes (gender, race, age, religion) are excluded from scoring features.
2. **Pre-Scoring PII Redaction**: Personal names, contact details, and URLs are masked prior to document embedding and feature evaluation.
3. **Prototype Disclosure**: This controlled prototype does not conduct an empirical statistical population fairness audit due to its synthetic benchmark size (8 profiles). Production deployment requires continuous fairness monitoring and audits.
4. **Human-in-the-Loop Mandate**: TalentMatch ML is explicitly built as a **recruiter decision-support tool**, NOT an autonomous hiring engine. The Composite Match Score is an alignment heuristic, and final shortlisting requires human recruiter validation.

---

## 📁 13. Repository Structure
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
│   └── 03_resume_screening_system.ipynb       # Fully Executed Jupyter Notebook
├── src/
│   ├── data_generator.py                      # Multi-Format Synthetic Data Generator
│   ├── parser.py                              # Multi-Format Parser & PII Anonymizer
│   ├── taxonomy.py                            # Technical Skill Taxonomy & Aliases
│   ├── matcher.py                             # 4-Tier Hybrid Matcher & Ranker
│   ├── visualize.py                           # Leaderboard & Diagnostic Plotter
│   └── pipeline.py                            # End-to-End Multi-Role Execution Pipeline
├── dashboard/
│   └── app.py                                 # Streamlit Recruiter Intelligence Dashboard
├── models/
│   └── talent_matcher_engine.pkl              # Serialized Matching Engine
├── outputs/
│   ├── figures/
│   │   ├── candidate_ranking_leaderboard.png
│   │   ├── score_component_breakdown.png
│   │   └── skill_gap_matrix.png
│   └── metrics/
│       ├── candidate_screening_rankings.csv
│       ├── rankings_all_roles.csv             # Consolidated Source of Truth
│       ├── rankings_job_do_003.csv
│       ├── rankings_job_fs_002.csv
│       └── rankings_job_ml_001.csv
└── reports/
    └── candidate_screening_decision_report.md # Executive HR Decision-Support Report
```

---

## 🚀 14. Execution Instructions

### 1. Setup Environment
```bash
git clone https://github.com/Jhansi48/FUTURE_ML_03.git
cd FUTURE_ML_03
pip install -r requirements.txt
```

### 2. Run End-to-End Multi-Role Pipeline
```bash
python src/pipeline.py
```

### 3. Launch Interactive Recruiter Dashboard
```bash
python -m streamlit run dashboard/app.py
```

### 4. Execute Jupyter Notebook
```bash
python -m nbconvert --to notebook --execute --inplace notebooks/03_resume_screening_system.ipynb
```

---

## 💡 15. Limitations & Future Roadmap
- **Domain Scope**: Current taxonomy covers 5 core software and data domains. Future iterations can integrate dynamic LLM-driven skill ontologies.
- **Dataset Scaling**: Expand from synthetic benchmark profiles to large-scale, ethically sourced industry recruiting corpora with formal disparate impact auditing.
- **Multimodal Evaluation**: Support portfolio link verification, GitHub contribution parsing, and structured work sample assessments.
