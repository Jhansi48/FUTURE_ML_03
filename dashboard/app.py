"""
TalentMatch ML - Interactive Candidate Screening & Skill-Gap Analysis Dashboard
Streamlit web application for recruiters and hiring managers to evaluate candidates,
inspect skill gaps, and explore multi-tier candidate ranking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.parser import parse_resume
from src.matcher import TalentMatcher

st.set_page_config(page_title="TalentMatch ML", page_icon="🎯", layout="wide")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resumes_dir = os.path.join(base_dir, "data", "resumes")
jds_dir = os.path.join(base_dir, "data", "job_descriptions")

@st.cache_resource
def get_matcher():
    return TalentMatcher(use_dense_embeddings=True)

matcher = get_matcher()

# Header
st.title("🎯 TalentMatch ML — Candidate Screening & Skill Gap System")
st.markdown("Decision-support Machine Learning platform for objective, skill-based candidate screening, semantic job-fit matching, and transparent skill-gap diagnostics.")

# Sidebar: Job Description Selector
st.sidebar.header("📋 Job Description Configuration")
jd_files = glob.glob(os.path.join(jds_dir, "*.json"))

jd_options = {}
for jdf in jd_files:
    with open(jdf, "r", encoding="utf-8") as f:
        data = json.load(f)
        jd_options[data["title"]] = data

selected_jd_title = st.sidebar.selectbox("Select Target Job Role:", list(jd_options.keys()))
active_jd = jd_options[selected_jd_title]

st.sidebar.markdown(f"**Required Experience:** `{active_jd['min_experience_years']} Years`")
st.sidebar.markdown(f"**Mandatory Skills ({len(active_jd['required_skills'])}):**")
st.sidebar.write(", ".join(active_jd['required_skills']))
st.sidebar.markdown(f"**Preferred Skills ({len(active_jd['preferred_skills'])}):**")
st.sidebar.write(", ".join(active_jd['preferred_skills']))

# Main Dashboard
resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))

if not resume_files:
    st.warning("No candidate resumes found. Please run `python src/pipeline.py` first.")
    st.stop()

# Parse all resumes
candidate_profiles = [parse_resume(f) for f in sorted(resume_files)]
rankings_df = matcher.rank_candidates(candidate_profiles, active_jd)

# Top KPI Summary Cards
st.subheader(f"📊 Candidate Screening Summary — {active_jd['title']}")
col1, col2, col3, col4 = st.columns(4)

total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 80])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 65) & (rankings_df["composite_score"] < 80)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 65])

col1.metric("Total Resumes Screened", total_cands)
col2.metric("Strong Technical Matches (≥80%)", strong_matches, "Ready for Interview")
col3.metric("Review / Moderate Matches", moderate_matches, "Review Missing Skills")
col4.metric("High Skill Gaps (<65%)", high_gaps, "Skill Mismatch")

# Visual Leaderboard
st.divider()
st.subheader("🏆 Candidate Ranking Leaderboard")

fig_bar = px.bar(
    rankings_df.sort_values(by="composite_score", ascending=True),
    x="composite_score",
    y="candidate_id",
    orientation="h",
    color="composite_score",
    color_continuous_scale="Viridis",
    title="Candidate Composite Match Score Leaderboard (%)",
    labels={"composite_score": "Composite Match Score (%)", "candidate_id": "Candidate Profile"}
)
fig_bar.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Strong Fit (80%)")
fig_bar.add_vline(x=65, line_dash="dash", line_color="orange", annotation_text="Review Cutoff (65%)")
st.plotly_chart(fig_bar, use_container_width=True)

# Detailed Candidate Table
st.dataframe(
    rankings_df[[
        "Rank", "candidate_id", "composite_score", "gap_severity",
        "skill_score_pct", "semantic_similarity_pct", "experience_years", "education_level"
    ]].style.highlight_max(subset=["composite_score"], color="#d4edda"),
    use_container_width=True
)

# Deep-Dive Candidate Inspector
st.divider()
st.subheader("🔍 Deep-Dive Candidate Skill-Gap Inspector")

selected_cand_id = st.selectbox(
    "Select Candidate Profile to Inspect:",
    rankings_df["candidate_id"].tolist()
)

cand_row = rankings_df[rankings_df["candidate_id"] == selected_cand_id].iloc[0]

cand_col1, cand_col2 = st.columns(2)

with cand_col1:
    st.markdown(f"### Profile: `{cand_row['candidate_id']}`")
    st.markdown(f"**Rank:** `#{cand_row['Rank']}` | **Match Score:** `{cand_row['composite_score']:.1f}%` | **Gap Severity:** `{cand_row['gap_severity']}`")
    st.markdown(f"**Stated Experience:** `{cand_row['experience_years']} Years` | **Education:** `{cand_row['education_level']}`")
    
    st.markdown("#### 💡 Decision-Support Recommendation:")
    st.info(cand_row["recommendation"])
    
    st.markdown("#### 🎯 Score Breakdown by Evaluation Pillar:")
    breakdown_df = pd.DataFrame({
        "Evaluation Pillar": ["Structured Skill Match", "Dense Semantic Fit", "Lexical Text Overlap", "Experience Fit"],
        "Score (%)": [cand_row["skill_score_pct"], cand_row["semantic_similarity_pct"], cand_row["lexical_similarity_pct"], min(100.0, (cand_row["experience_years"]/active_jd["min_experience_years"])*100)]
    })
    st.dataframe(breakdown_df, use_container_width=True)

with cand_col2:
    st.markdown("#### 🛠️ Skill Competency Gap Analysis:")
    
    st.markdown("**✅ Matched Mandatory Skills:**")
    if cand_row["matched_required"]:
        for s in cand_row["matched_required"]:
            st.success(f"✓ {s}")
    else:
        st.write("*None*")
        
    st.markdown("**❌ Missing Mandatory Skills (Skill Gap):**")
    if cand_row["missing_required"]:
        for s in cand_row["missing_required"]:
            st.error(f"✗ {s}")
    else:
        st.info("🎉 All mandatory skills satisfied!")
        
    if cand_row["matched_preferred"]:
        st.markdown("**⭐ Matched Preferred Bonus Skills:**")
        st.write(", ".join(cand_row["matched_preferred"]))

st.divider()
st.caption("🔒 **Fairness & Privacy Note:** TalentMatch ML automatically masks candidate PII (names, contact info, demographics) during evaluation. This system is designed strictly as an objective decision-support assistant.")
