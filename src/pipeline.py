"""
TalentMatch ML - End-to-End Multi-Role Pipeline Execution Script
Parses multi-format resumes, extracts skills against canonical taxonomy,
computes hybrid match scores and skill gaps across all 3 Job Descriptions,
saves metrics, figures, and comprehensive decision-support reports.
"""

import os
import sys
import glob
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import generate_sample_resumes, generate_sample_job_descriptions
from src.parser import parse_resume
from src.matcher import TalentMatcher
from src.visualize import (
    plot_candidate_leaderboard,
    plot_score_component_breakdown,
    plot_skill_gap_matrix
)

def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resumes_dir = os.path.join(base_dir, "data", "resumes")
    jds_dir = os.path.join(base_dir, "data", "job_descriptions")
    models_dir = os.path.join(base_dir, "models")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 75)
    print("STEP 1: Ingesting & Generating Multi-Format Resumes & Job Descriptions")
    print("=" * 75)
    generate_sample_resumes(resumes_dir)
    generate_sample_job_descriptions(jds_dir)
    
    resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))
    print(f"Total Candidate Resume Files Ingested: {len(resume_files)}")
    
    print("\n" + "=" * 75)
    print("STEP 2: Parsing Resumes, Anonymizing PII & Extracting Features")
    print("=" * 75)
    candidate_profiles = []
    for fpath in sorted(resume_files):
        parsed = parse_resume(fpath)
        candidate_profiles.append(parsed)
        # Verify PII name removal
        has_masked_name = "[NAME_MASKED]" in parsed["anonymized_text"]
        print(f"-> Parsed: {parsed['candidate_id']:<36} | Exp: {parsed['experience_years']} yrs | Edu: {parsed['education_level']:<22} | PII Anonymized: {has_masked_name}")
        
    print("\n" + "=" * 75)
    print("STEP 3: Initializing TalentMatch Hybrid Scoring Engine")
    print("=" * 75)
    matcher = TalentMatcher(use_dense_embeddings=True)
    
    # Save matcher payload
    joblib.dump(matcher, os.path.join(models_dir, "talent_matcher_engine.pkl"))
    print(f"TalentMatcher engine serialized to: {os.path.join(models_dir, 'talent_matcher_engine.pkl')}")
    
    jd_files = sorted(glob.glob(os.path.join(jds_dir, "*.json")))
    all_role_rankings = []
    
    print("\n" + "=" * 75)
    print("STEP 4: Multi-Role Candidate Ranking & Evaluation Across All 3 JDs")
    print("=" * 75)
    
    report_sections = []
    
    for jd_file in jd_files:
        with open(jd_file, "r", encoding="utf-8") as f:
            jd_data = json.load(f)
            
        role_title = jd_data["title"]
        job_id = jd_data.get("job_id", "JOB")
        print(f"\n>>> Evaluating Target Role: {role_title} ({job_id})")
        print(f"    Required Skills ({len(jd_data['required_skills'])}): {jd_data['required_skills']}")
        print(f"    Preferred Skills ({len(jd_data['preferred_skills'])}): {jd_data['preferred_skills']}")
        
        rankings_df = matcher.rank_candidates(candidate_profiles, jd_data)
        rankings_df["job_id"] = job_id
        rankings_df["job_title"] = role_title
        
        for idx, row in rankings_df.iterrows():
            print(f"    Rank {row['Rank']}: {row['candidate_id']:<35} | Score: {row['composite_score']:>5.1f}% | Fit: {row['fit_category']:<20} | Req Matched: {len(row['matched_required'])}/{len(jd_data['required_skills'])}")
            
        # Save individual role CSV
        role_csv_filename = f"rankings_{job_id.lower().replace('-', '_')}.csv"
        rankings_df.to_csv(os.path.join(metrics_dir, role_csv_filename), index=False)
        all_role_rankings.append(rankings_df)
        
        # Build Markdown section for this role
        top_cand = rankings_df.iloc[0]
        report_sections.append(f"""### Target Role: {role_title} (`{job_id}`)
- **Minimum Experience**: {jd_data['min_experience_years']}+ Years
- **Mandatory Skills**: {', '.join(jd_data['required_skills'])}
- **Preferred Skills**: {', '.join(jd_data['preferred_skills'])}
- **Top Match Candidate**: `{top_cand['candidate_id']}` (Score: **{top_cand['composite_score']:.1f}%**, {top_cand['fit_category']})

| Rank | Candidate Profile | Composite Match (%) | Fit Category | Technical Overlap (%) | Semantic Fit (%) | Lexical Match (%) | Experience | Education | Missing Mandatory Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
""" + "\n".join([f"| #{row['Rank']} | `{row['candidate_id']}` | **{row['composite_score']:.1f}%** | {row['fit_category']} | {row['skill_score_pct']:.1f}% | {row['semantic_similarity_pct']:.1f}% | {row['lexical_similarity_pct']:.1f}% | {row['experience_years']} yrs | {row['education_level']} | {', '.join(row['missing_required']) if row['missing_required'] else 'None (All Satisfied)'} |" for _, row in rankings_df.iterrows()]) + "\n")
    
    # Save Consolidated CSV for all roles
    consolidated_df = pd.concat(all_role_rankings, ignore_index=True)
    consolidated_csv_path = os.path.join(metrics_dir, "rankings_all_roles.csv")
    consolidated_df.to_csv(consolidated_csv_path, index=False)
    print(f"\n[Artifact] Consolidated multi-role rankings saved to: {consolidated_csv_path}")
    
    # Primary benchmark rankings for default visualizations (Senior Machine Learning Engineer)
    primary_df = all_role_rankings[0]
    with open(jd_files[0], "r", encoding="utf-8") as f:
        primary_jd = json.load(f)
        
    print("\n" + "=" * 75)
    print("STEP 5: Generating Decision-Support Diagnostic Visualizations")
    print("=" * 75)
    plot_candidate_leaderboard(primary_df, primary_jd["title"], os.path.join(figures_dir, "candidate_ranking_leaderboard.png"))
    plot_score_component_breakdown(primary_df, os.path.join(figures_dir, "score_component_breakdown.png"))
    
    # Prepare skill coverage objects for matrix
    matrix_profiles = []
    for c in candidate_profiles:
        cand_skills = matcher.extract_skills_from_text(c["anonymized_text"])
        matrix_profiles.append({
            "candidate_id": c["candidate_id"],
            "matched_skills": list(cand_skills)
        })
    plot_skill_gap_matrix(matrix_profiles, primary_jd["required_skills"], os.path.join(figures_dir, "skill_gap_matrix.png"))
    print(f"Diagnostic figures saved to: {figures_dir}")
    
    # Write Comprehensive Multi-Role Decision Support Report
    report_path = os.path.join(reports_dir, "candidate_screening_decision_report.md")
    report_body = f"""# TalentMatch ML — Candidate Screening & Decision-Support Report

## Executive Summary & Data Provenance
TalentMatch ML is an objective, interpretable candidate screening decision-support prototype.
The evaluation is conducted on a controlled synthetic corpus of **8 multi-format candidate profiles** (PDF, DOCX, and TXT) across **3 structured job descriptions**.

> **Human-in-the-Loop Disclaimer**: TalentMatch ML is explicitly built as a recruiter decision-support tool, NOT an autonomous hiring engine. The Composite Match Score represents multi-factor alignment against prototype heuristic weights and must be validated through structured human interviews.

---

## Evaluation Methodology & Scoring Architecture
The scoring formula evaluates candidate compatibility across four interpretable pillars:

$$\\text{{Composite Match Score}} = 0.40 \\cdot S_{{\\text{{skill}}}} + 0.30 \\cdot S_{{\\text{{semantic}}}} + 0.20 \\cdot S_{{\\text{{lexical}}}} + 0.10 \\cdot S_{{\\text{{exp\\_edu}}}}$$

1. **Hard Skill Overlap (40%)**: Structured mandatory (80%) and preferred (20%) technical competency extraction against taxonomy.
2. **Dense Semantic Similarity (30%)**: Contextual vector cosine similarity via `SentenceTransformer('all-MiniLM-L6-v2')`.
3. **TF-IDF Lexical Similarity (20%)**: Sublinear term-frequency inverse document frequency cosine similarity.
4. **Experience & Education Fit (10%)**: Experience tenure ratio against stated job requirements combined with educational degree attainment heuristic.

---

## Canonical Fit Taxonomy (Composite Alignment)
- **Strong Overall Match** (Score $\\ge 75.0\\%$): High multi-factor alignment; core mandatory competencies satisfied with strong domain contextual alignment.
- **Moderate Match** (Score $40.0\\% - 74.9\\%$): Partial technical/semantic alignment; review specific skill gaps with hiring manager.
- **High Technical Gap** (Score $< 40.0\\%$): Substantial core skill gaps; profile aligns primarily with adjacent domains.

---

## Multi-Role Screening Results

""" + "\n\n".join(report_sections) + """

---

## Privacy & Responsible Use Safeguards
1. **PII Reduction & Anonymization**: Candidate names (`[NAME_MASKED]`), email addresses (`[EMAIL_MASKED]`), phone numbers (`[PHONE_MASKED]`), and social profile URLs are scrubbed before feature extraction.
2. **Exclusion of Demographic Attributes**: Gender, race, age, and personal identifiers are excluded from ranking features.
3. **Prototype Limitations**: As a controlled synthetic prototype with 8 candidate profiles, this benchmark does not perform a statistical population fairness audit. Real-world deployment requires representative data, continuous bias auditing, and human oversight.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_body)
    print(f"Executive Candidate Screening Report written to: {report_path}")
    print("\nEnd-to-End Multi-Role Pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()
