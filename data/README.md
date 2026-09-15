# Dataset Documentation: TalentMatch ML Benchmark Corpus

## 📌 Overview & Data Provenance
This dataset contains a controlled, reproducible prototype candidate corpus designed to demonstrate and validate multi-format parsing, PII redaction, skill taxonomy extraction, and multi-tier hybrid scoring.

> **Disclosure**: This is a controlled synthetic benchmark containing 8 candidate profiles across 3 standard engineering domains. It is specifically designed for prototype evaluation and testing, not production autonomous hiring.

---

## 📄 Multi-Format Resume Profiles (`data/resumes/`)
The corpus contains candidate resumes across three heterogeneous formats:

| Filename | Format | Candidate Title | Target Seniority | Experience Stated | Highest Education |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `candidate_01_lead_ml_engineer.docx` | `.docx` | Senior Machine Learning & MLOps Engineer | Lead / Senior | 6.5 Years | Master's Degree |
| `candidate_02_data_scientist.txt` | `.txt` | Data Scientist & Quantitative Analyst | Mid-Level | 4.0 Years | Master's Degree |
| `candidate_03_junior_ml_intern.pdf` | `.pdf` | Junior Machine Learning Developer | Junior / Entry | 1.0 Years | Bachelor's Degree |
| `candidate_04_fullstack_lead.docx` | `.docx` | Lead Full Stack Engineer | Lead / Senior | 7.0 Years | Bachelor's Degree |
| `candidate_05_frontend_dev.txt` | `.txt` | Frontend React Developer | Mid-Level | 3.0 Years | Bachelor's Degree |
| `candidate_06_devops_cloud_architect.docx` | `.docx` | Principal Cloud DevOps Architect | Principal / Senior | 8.0 Years | Bachelor's Degree |
| `candidate_07_bi_data_analyst.txt` | `.txt` | Business Intelligence & Data Analyst | Mid-Level | 4.5 Years | Bachelor's Degree |
| `candidate_08_java_backend_dev.txt` | `.txt` | Senior Java Backend Developer | Senior | 5.5 Years | Bachelor's Degree |

---

## 🎯 Job Description Specifications (`data/job_descriptions/`)
Three structured JSON specifications defining mandatory requirements, preferred competencies, and experience thresholds:

1. **`job_cloud_devops_specialist.json` (`JOB-DO-003`)**:
   - Title: *Senior Cloud DevOps & Infrastructure Specialist*
   - Minimum Experience: 4.0 Years
   - Required Skills: `aws`, `kubernetes`, `docker`, `terraform`, `ci/cd`, `linux`
   - Preferred Skills: `python`, `prometheus`, `grafana`, `github actions`, `bash`, `ansible`

2. **`job_lead_fullstack_engineer.json` (`JOB-FS-002`)**:
   - Title: *Lead Full Stack Software Engineer*
   - Minimum Experience: 5.0 Years
   - Required Skills: `typescript`, `javascript`, `react`, `node.js`, `postgresql`, `rest api`, `docker`
   - Preferred Skills: `next.js`, `graphql`, `redis`, `aws`, `ci/cd`, `system design`

3. **`job_senior_ml_engineer.json` (`JOB-ML-001`)**:
   - Title: *Senior Machine Learning Engineer (NLP & MLOps)*
   - Minimum Experience: 4.0 Years
   - Required Skills: `python`, `pytorch`, `scikit-learn`, `machine learning`, `natural language processing`, `docker`, `aws`, `model deployment`
   - Preferred Skills: `transformers`, `huggingface`, `kubernetes`, `mlops`, `langchain`, `sql`
