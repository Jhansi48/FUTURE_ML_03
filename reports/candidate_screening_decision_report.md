# TalentMatch ML — Candidate Screening & Decision-Support Report

## Executive Summary & Data Provenance
TalentMatch ML is an objective, interpretable candidate screening decision-support prototype.
The evaluation is conducted on a controlled synthetic corpus of **8 multi-format candidate profiles** (PDF, DOCX, and TXT) across **3 structured job descriptions**.

> **Human-in-the-Loop Disclaimer**: TalentMatch ML is explicitly built as a recruiter decision-support tool, NOT an autonomous hiring engine. The Composite Match Score represents multi-factor alignment against prototype heuristic weights and must be validated through structured human interviews.

---

## Evaluation Methodology & Scoring Architecture
The scoring formula evaluates candidate compatibility across four interpretable pillars:

$$\text{Composite Match Score} = 0.40 \cdot S_{\text{skill}} + 0.30 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{lexical}} + 0.10 \cdot S_{\text{exp\_edu}}$$

1. **Hard Skill Overlap (40%)**: Structured mandatory (80%) and preferred (20%) technical competency extraction against taxonomy.
2. **Dense Semantic Similarity (30%)**: Contextual vector cosine similarity via `SentenceTransformer('all-MiniLM-L6-v2')`.
3. **TF-IDF Lexical Similarity (20%)**: Sublinear term-frequency inverse document frequency cosine similarity.
4. **Experience & Education Fit (10%)**: Experience tenure ratio against stated job requirements combined with educational degree attainment heuristic.

---

## Canonical Fit Taxonomy
- **Strong Technical Fit** (Score $\ge 75.0\%$): Core mandatory competencies satisfied with high contextual semantic alignment.
- **Moderate Fit** (Score $40.0\% - 74.9\%$): Partial technical alignment; review specific skill gaps with hiring manager.
- **High Technical Gap** (Score $< 40.0\%$): Substantial core skill gaps; profile aligns primarily with adjacent domains.

---

## Multi-Role Screening Results

### Target Role: Senior Cloud DevOps & Infrastructure Specialist (`JOB-DO-003`)
- **Minimum Experience**: 4.0+ Years
- **Mandatory Skills**: aws, kubernetes, docker, terraform, ci/cd, linux
- **Preferred Skills**: python, prometheus, grafana, github actions, bash, ansible
- **Top Match Candidate**: `candidate_06_devops_cloud_architect` (Score: **75.2%**, Strong Technical Fit)

| Rank | Candidate Profile | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Experience | Education | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `candidate_06_devops_cloud_architect` | **75.2%** | Strong Technical Fit | 100.0% | 70.1% | 23.2% | 8.0 yrs | Bachelor's Degree | None (All Satisfied) |
| #2 | `candidate_01_lead_ml_engineer` | **56.0%** | Moderate Fit | 70.0% | 55.1% | 8.2% | 6.5 yrs | Master's Degree | terraform |
| #3 | `candidate_04_fullstack_lead` | **43.9%** | Moderate Fit | 43.3% | 52.1% | 6.9% | 7.0 yrs | Bachelor's Degree | kubernetes, linux, terraform |
| #4 | `candidate_08_java_backend_dev` | **31.1%** | High Technical Gap | 26.7% | 35.2% | 1.8% | 5.5 yrs | Bachelor's Degree | amazon web services, ci/cd, linux, terraform |
| #5 | `candidate_02_data_scientist` | **26.0%** | High Technical Gap | 16.7% | 30.5% | 1.9% | 4.0 yrs | Master's Degree | ci/cd, docker, kubernetes, linux, terraform |
| #6 | `candidate_03_junior_ml_intern` | **25.4%** | High Technical Gap | 30.0% | 28.5% | 2.7% | 1.0 yrs | Bachelor's Degree | amazon web services, ci/cd, kubernetes, terraform |
| #7 | `candidate_07_bi_data_analyst` | **19.6%** | High Technical Gap | 3.3% | 28.2% | 1.1% | 4.5 yrs | Bachelor's Degree | amazon web services, ci/cd, docker, kubernetes, linux, terraform |
| #8 | `candidate_05_frontend_dev` | **15.1%** | High Technical Gap | 0.0% | 24.4% | 0.0% | 3.0 yrs | Bachelor's Degree | amazon web services, ci/cd, docker, kubernetes, linux, terraform |


### Target Role: Lead Full Stack Software Engineer (`JOB-FS-002`)
- **Minimum Experience**: 5.0+ Years
- **Mandatory Skills**: typescript, javascript, react, node.js, postgresql, rest api, docker
- **Preferred Skills**: next.js, graphql, redis, aws, ci/cd, system design
- **Top Match Candidate**: `candidate_04_fullstack_lead` (Score: **72.8%**, Moderate Fit)

| Rank | Candidate Profile | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Experience | Education | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `candidate_04_fullstack_lead` | **72.8%** | Moderate Fit | 100.0% | 65.3% | 18.3% | 7.0 yrs | Bachelor's Degree | None (All Satisfied) |
| #2 | `candidate_01_lead_ml_engineer` | **39.7%** | High Technical Gap | 32.9% | 52.0% | 5.5% | 6.5 yrs | Master's Degree | javascript, node.js, react, rest api, typescript |
| #3 | `candidate_08_java_backend_dev` | **38.8%** | High Technical Gap | 37.6% | 43.8% | 5.5% | 5.5 yrs | Bachelor's Degree | javascript, node.js, react, typescript |
| #4 | `candidate_05_frontend_dev` | **36.1%** | High Technical Gap | 37.6% | 42.3% | 7.9% | 3.0 yrs | Bachelor's Degree | docker, node.js, postgresql, rest api |
| #5 | `candidate_06_devops_cloud_architect` | **34.7%** | High Technical Gap | 18.1% | 53.9% | 8.7% | 8.0 yrs | Bachelor's Degree | javascript, node.js, postgresql, react, rest api, typescript |
| #6 | `candidate_02_data_scientist` | **25.0%** | High Technical Gap | 14.8% | 33.9% | 2.5% | 4.0 yrs | Master's Degree | docker, javascript, node.js, react, rest api, typescript |
| #7 | `candidate_07_bi_data_analyst` | **16.1%** | High Technical Gap | 0.0% | 23.8% | 0.6% | 4.5 yrs | Bachelor's Degree | docker, javascript, node.js, postgresql, react, rest api, typescript |
| #8 | `candidate_03_junior_ml_intern` | **14.4%** | High Technical Gap | 11.4% | 18.5% | 1.9% | 1.0 yrs | Bachelor's Degree | javascript, node.js, postgresql, react, rest api, typescript |


### Target Role: Senior Machine Learning Engineer (NLP & MLOps) (`JOB-ML-001`)
- **Minimum Experience**: 4.0+ Years
- **Mandatory Skills**: python, pytorch, scikit-learn, machine learning, natural language processing, docker, aws, model deployment
- **Preferred Skills**: transformers, huggingface, kubernetes, mlops, langchain, sql
- **Top Match Candidate**: `candidate_01_lead_ml_engineer` (Score: **76.0%**, Strong Technical Fit)

| Rank | Candidate Profile | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Experience | Education | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `candidate_01_lead_ml_engineer` | **76.0%** | Strong Technical Fit | 100.0% | 72.2% | 22.7% | 6.5 yrs | Master's Degree | None (All Satisfied) |
| #2 | `candidate_03_junior_ml_intern` | **43.9%** | Moderate Fit | 60.0% | 43.1% | 13.3% | 1.0 yrs | Bachelor's Degree | amazon web services, natural language processing |
| #3 | `candidate_02_data_scientist` | **39.2%** | High Technical Gap | 43.3% | 35.8% | 6.7% | 4.0 yrs | Master's Degree | docker, model deployment, natural language processing, pytorch |
| #4 | `candidate_06_devops_cloud_architect` | **39.0%** | High Technical Gap | 33.3% | 49.3% | 6.8% | 8.0 yrs | Bachelor's Degree | machine learning, model deployment, natural language processing, pytorch, scikit-learn |
| #5 | `candidate_04_fullstack_lead` | **30.3%** | High Technical Gap | 20.0% | 39.7% | 4.1% | 7.0 yrs | Bachelor's Degree | machine learning, model deployment, natural language processing, python, pytorch, scikit-learn |
| #6 | `candidate_08_java_backend_dev` | **24.8%** | High Technical Gap | 13.3% | 30.7% | 3.4% | 5.5 yrs | Bachelor's Degree | amazon web services, machine learning, model deployment, natural language processing, python, pytorch, scikit-learn |
| #7 | `candidate_07_bi_data_analyst` | **23.2%** | High Technical Gap | 13.3% | 26.9% | 1.5% | 4.5 yrs | Bachelor's Degree | amazon web services, docker, machine learning, model deployment, natural language processing, pytorch, scikit-learn |
| #8 | `candidate_05_frontend_dev` | **14.8%** | High Technical Gap | 0.0% | 22.0% | 1.9% | 3.0 yrs | Bachelor's Degree | amazon web services, docker, machine learning, model deployment, natural language processing, python, pytorch, scikit-learn |


---

## Privacy & Responsible Use Safeguards
1. **PII Reduction & Anonymization**: Candidate names (`[NAME_MASKED]`), email addresses (`[EMAIL_MASKED]`), phone numbers (`[PHONE_MASKED]`), and social profile URLs are scrubbed before feature extraction.
2. **Exclusion of Demographic Attributes**: Gender, race, age, and personal identifiers are excluded from ranking features.
3. **Prototype Limitations**: As a controlled synthetic prototype with 8 candidate profiles, this benchmark does not perform a statistical population fairness audit. Real-world deployment requires representative data, continuous bias auditing, and human oversight.
