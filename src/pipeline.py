"""
TalentMatch ML - End-to-End Pipeline Execution Script
Parses multi-format resumes, extracts skills against canonical taxonomy,
computes hybrid match scores and skill gaps, ranks candidates, saves metrics & figures.
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
    
    print("=" * 70)
    print("STEP 1: Ingesting & Generating Multi-Format Resumes & Job Descriptions")
    print("=" * 70)
    generate_sample_resumes(resumes_dir)
    generate_sample_job_descriptions(jds_dir)
    
    resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))
    print(f"Total Candidate Resume Files Ingested: {len(resume_files)}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Parsing Resumes, Anonymizing PII & Extracting Features")
    print("=" * 70)
    candidate_profiles = []
    for fpath in sorted(resume_files):
        parsed = parse_resume(fpath)
        candidate_profiles.append(parsed)
        print(f"-> Parsed: {parsed['candidate_id']:<36} | Exp: {parsed['experience_years']} yrs | Edu: {parsed['education_level']}")
        
    print("\n" + "=" * 70)
    print("STEP 3: Initializing TalentMatch Hybrid Scoring Engine")
    print("=" * 70)
    matcher = TalentMatcher(use_dense_embeddings=True)
    
    # Load primary benchmark Job Description: Senior ML Engineer
    primary_jd_path = os.path.join(jds_dir, "job_senior_ml_engineer.json")
    with open(primary_jd_path, "r", encoding="utf-8") as f:
        primary_jd = json.load(f)
        
    print(f"Target Role: {primary_jd['title']}")
    print(f"Required Skills ({len(primary_jd['required_skills'])}): {primary_jd['required_skills']}")
    print(f"Preferred Skills ({len(primary_jd['preferred_skills'])}): {primary_jd['preferred_skills']}")
    
    print("\n" + "=" * 70)
    print("STEP 4: Candidate Ranking & Skill Gap Analysis")
    print("=" * 70)
    rankings_df = matcher.rank_candidates(candidate_profiles, primary_jd)
    
    for idx, row in rankings_df.iterrows():
        print(f"Rank {row['Rank']}: {row['candidate_id']:<35} | Score: {row['composite_score']:>5.1f}% | Gap: {row['gap_severity']:<8} | Matched: {len(row['matched_required'])}/{len(primary_jd['required_skills'])}")
        
    # Save CSV
    ranking_csv_path = os.path.join(metrics_dir, "candidate_screening_rankings.csv")
    rankings_df.to_csv(ranking_csv_path, index=False)
    print(f"\nCandidate rankings saved to: {ranking_csv_path}")
    
    # Save matcher payload
    joblib.dump(matcher, os.path.join(models_dir, "talent_matcher_engine.pkl"))
    print(f"TalentMatcher engine serialized to: {os.path.join(models_dir, 'talent_matcher_engine.pkl')}")
    
    print("\n" + "=" * 70)
    print("STEP 5: Generating Decision-Support Diagnostic Visualizations")
    print("=" * 70)
    plot_candidate_leaderboard(rankings_df, primary_jd["title"], os.path.join(figures_dir, "candidate_ranking_leaderboard.png"))
    plot_score_component_breakdown(rankings_df, os.path.join(figures_dir, "score_component_breakdown.png"))
    
    # Prepare skill coverage objects for matrix
    matrix_profiles = []
    for c in candidate_profiles:
        cand_skills = matcher.extract_skills_from_text(c["anonymized_text"])
        matrix_profiles.append({
            "candidate_id": c["candidate_id"],
            "matched_skills": list(cand_skills)
        })
    plot_skill_gap_matrix(matrix_profiles, primary_jd["required_skills"], os.path.join(figures_dir, "skill_gap_matrix.png"))
    
    # Write Business HR Decision Support Report
    report_path = os.path.join(reports_dir, "candidate_screening_decision_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# TalentMatch ML — Candidate Screening & Decision-Support Report

## Target Role: {primary_jd['title']}
- **Role ID**: `{primary_jd['job_id']}`
- **Minimum Experience**: {primary_jd['min_experience_years']} Years
- **Mandatory Competencies**: {', '.join(primary_jd['required_skills'])}
- **Preferred Competencies**: {', '.join(primary_jd['preferred_skills'])}

## Executive Candidate Ranking Leaderboard
| Rank | Candidate ID | Composite Match Score (%) | Skill Match (%) | Semantic Sim (%) | Experience (Yrs) | Gap Severity | Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
""" + "\n".join([f"| {row['Rank']} | `{row['candidate_id']}` | **{row['composite_score']:.1f}%** | {row['skill_score_pct']:.1f}% | {row['semantic_similarity_pct']:.1f}% | {row['experience_years']} yrs | {row['gap_severity']} | {row['recommendation'].split(':')[0]} |" for _, row in rankings_df.iterrows()]) + f"""

## Key Candidate Profiles & Skill-Gap Breakdown

### 🥇 Rank 1: `{rankings_df.iloc[0]['candidate_id']}` (Score: {rankings_df.iloc[0]['composite_score']:.1f}%)
- **Matched Mandatory Skills**: `{', '.join(rankings_df.iloc[0]['matched_required'])}`
- **Missing Mandatory Skills**: `None`
- **Bonus Preferred Skills**: `{', '.join(rankings_df.iloc[0]['matched_preferred'])}`
- **Operational Recommendation**: {rankings_df.iloc[0]['recommendation']}

### 🥈 Rank 2: `{rankings_df.iloc[1]['candidate_id']}` (Score: {rankings_df.iloc[1]['composite_score']:.1f}%)
- **Matched Mandatory Skills**: `{', '.join(rankings_df.iloc[1]['matched_required'])}`
- **Missing Mandatory Skills**: `{', '.join(rankings_df.iloc[1]['missing_required'])}`
- **Operational Recommendation**: {rankings_df.iloc[1]['recommendation']}

## Ethical AI & Fairness Safeguards
1. **Zero-PII Evaluation**: Candidate names, email addresses, phone numbers, and demographic references are scrubbed prior to feature extraction.
2. **Decision-Support Design**: This system serves exclusively to assist recruiters with initial technical filtering. Final hiring decisions must always incorporate human interview stages.
""")
    print(f"Executive Candidate Screening Report written to: {report_path}")
    print("\nPipeline execution complete successfully!")

if __name__ == "__main__":
    run_pipeline()
