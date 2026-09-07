"""
TalentMatch ML - Interactive Candidate Screening & Skill-Gap Analysis Dashboard
Decision-support Streamlit platform for objective, skill-based candidate screening,
semantic job-fit matching, and transparent skill-gap diagnostics.
"""

import os
import glob
import json
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.parser import parse_resume
from src.matcher import TalentMatcher

# Configure Page Layout & Styling
st.set_page_config(
    page_title="TalentMatch ML — Candidate Screening Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Light Theme CSS
st.markdown("""
<style>
    /* Global Typography & Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1F2933;
    }
    
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Global Canvas */
    .stApp {
        background-color: #F8F7F3 !important;
        color: #1F2933 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EBF0EA !important;
        border-right: 1px solid #D5DDD6 !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #1F2933 !important;
        margin-top: 0 !important;
        margin-bottom: 8px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: #53636A !important;
    }

    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #1F2933 !important;
    }

    section[data-testid="stSidebar"] label p {
        color: #1F2933 !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] code {
        background-color: #FFFFFF !important;
        color: #075E5B !important;
        border: 1px solid #D5DDD6 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 12px 0 !important;
        border-color: #D5DDD6 !important;
    }

    /* Structured Section Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        margin-bottom: 16px;
    }

    /* Section Number Badges */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 22px;
        margin-bottom: 12px;
    }
    
    .section-num {
        background-color: #087F7B;
        color: #FFFFFF;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1F2933;
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* Visual Workflow Steps */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04);
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .workflow-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        min-width: 110px;
    }
    
    .workflow-step-num {
        font-size: 0.70rem;
        font-weight: 800;
        color: #087F7B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .workflow-step-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1F2933;
        margin-top: 2px;
    }
    
    .workflow-arrow {
        color: #087F7B;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card-label {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #53636A;
        margin-bottom: 6px;
    }
    
    .metric-card-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1F2933;
        line-height: 1.25;
        word-break: break-word;
    }
    
    .metric-card-sub {
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid #F1F3EE;
    }

    /* Status Badges */
    .badge-urgent {
        background-color: #FBEAE5;
        color: #D95D39;
        border: 1px solid #F4C7BA;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-medium {
        background-color: #FFF4D8;
        color: #D99A24;
        border: 1px solid #F7DE98;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #EAF3ED;
        color: #5B8C72;
        border: 1px solid #BFDEC7;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-teal {
        background-color: #E7F4F2;
        color: #075E5B;
        border: 1px solid #B8E2DC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }

    .badge-neutral {
        background-color: #EBF0EA;
        color: #1F2933;
        border: 1px solid #D5DDD6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }

    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #1F2933 !important;
        background-color: #FFFFFF !important;
    }

    /* Custom Benchmark Table Styling */
    .benchmark-table-container {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 12px;
        overflow-x: auto;
        margin-bottom: 8px;
    }
    
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.88rem;
    }
    
    .benchmark-table th {
        background-color: #F8F7F3;
        color: #1F2933;
        font-weight: 700;
        padding: 10px 14px;
        border-bottom: 1px solid #D5DDD6;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .benchmark-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F1F3EE;
        color: #1F2933;
    }
    
    .benchmark-table tr:last-child td {
        border-bottom: none;
    }
    
    .benchmark-table tr.top-candidate-row td {
        background-color: #E7F4F2 !important;
        color: #075E5B !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model Setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resumes_dir = os.path.join(base_dir, "data", "resumes")
jds_dir = os.path.join(base_dir, "data", "job_descriptions")

@st.cache_resource
def load_matcher():
    return TalentMatcher(use_dense_embeddings=True)

matcher = load_matcher()

# Product Header Section
st.markdown("""
<div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #D5DDD6;">
    <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
        <h1 style="font-size: 2.1rem; font-weight: 900; color: #075E5B; margin: 0; letter-spacing: -0.03em;">
            TALENTMATCH ML
        </h1>
        <span style="font-size: 1.15rem; font-weight: 700; color: #1F2933; letter-spacing: -0.01em;">
            RESUME SCREENING & CANDIDATE RANKING PLATFORM
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #53636A; font-weight: 500; margin-top: 4px; margin-bottom: 10px;">
        Objective technical resume evaluation, multi-tier semantic matching & skill-gap diagnostics
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-teal">MULTI-FORMAT PARSING</span>
        <span class="badge-teal">4-TIER HYBRID SCORING</span>
        <span class="badge-teal">PII ANONYMIZATION</span>
        <span class="badge-neutral">CONTROLLED BENCHMARK</span>
        <span class="badge-neutral">DECISION-SUPPORT PROTOTYPE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Visual Workflow Pipeline Section
st.markdown("""
<div class="workflow-container">
    <div class="workflow-step">
        <span class="workflow-step-num">Step 1</span>
        <span class="workflow-step-name">Multi-Format Resumes</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 2</span>
        <span class="workflow-step-name">PII Anonymization</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 3</span>
        <span class="workflow-step-name">Skill Taxonomy (200+)</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 4</span>
        <span class="workflow-step-name">4-Tier Hybrid Match</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 5</span>
        <span class="workflow-step-name">Recruiter Decision Support</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("""
<div style="padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid #D5DDD6;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #075E5B; letter-spacing: -0.02em;">
        TALENTMATCH ML
    </div>
    <div style="font-size: 0.80rem; font-weight: 600; color: #53636A;">
        Recruiter Decision Support
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### Target Role Configuration")

jd_files = glob.glob(os.path.join(jds_dir, "*.json"))
if not jd_files:
    st.error("⚠️ No Job Description JSON files found in data/job_descriptions.")
    st.stop()

jd_options = {}
for jdf in sorted(jd_files):
    with open(jdf, "r", encoding="utf-8") as f:
        data = json.load(f)
        jd_options[data["title"]] = data

selected_jd_title = st.sidebar.selectbox("Select Target Job Specification:", list(jd_options.keys()))
active_jd = jd_options[selected_jd_title]

st.sidebar.markdown(f"**Required Experience:** `{active_jd['min_experience_years']} Years`")
st.sidebar.markdown(f"**Mandatory Competencies ({len(active_jd['required_skills'])}):**")
req_pills = " ".join([f"<code style='background:#FFFFFF; color:#075E5B; border:1px solid #D5DDD6; padding:2px 6px; border-radius:4px; font-size:0.78rem; display:inline-block; margin-bottom:3px;'>{s}</code>" for s in active_jd['required_skills']])
st.sidebar.markdown(req_pills, unsafe_allow_html=True)

st.sidebar.markdown(f"**Preferred Competencies ({len(active_jd['preferred_skills'])}):**")
pref_pills = " ".join([f"<code style='background:#FFFFFF; color:#53636A; border:1px solid #D5DDD6; padding:2px 6px; border-radius:4px; font-size:0.78rem; display:inline-block; margin-bottom:3px;'>{s}</code>" for s in active_jd['preferred_skills']])
st.sidebar.markdown(pref_pills, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Hybrid Scoring Formula")
st.sidebar.markdown("""
- **Hard Skill Overlap:** `40%`
- **Dense Semantic Fit:** `30%`
- **TF-IDF Lexical Match:** `20%`
- **Experience / Education:** `10%`
""")

# Ingest and Score Resumes
resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))
if not resume_files:
    st.warning("⚠️ No candidate resumes found in `data/resumes/`. Please run `python src/pipeline.py`.")
    st.stop()

candidate_profiles = [parse_resume(f) for f in sorted(resume_files)]
rankings_df = matcher.rank_candidates(candidate_profiles, active_jd)

# Section 01: Screening Summary KPIs
st.markdown(f"""
<div class="section-header">
    <span class="section-num">01</span>
    <h3 class="section-title">Screening Overview — {active_jd['title']}</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 75.0])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 40.0) & (rankings_df["composite_score"] < 75.0)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 40.0])

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div>
            <div class="metric-card-label">Total Resumes Screened</div>
            <div class="metric-card-value" style="color: #075E5B;">{total_cands} Profiles</div>
        </div>
        <div class="metric-card-sub" style="color: #075E5B;">
            Multi-Format: <strong>PDF, DOCX, TXT</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div>
            <div class="metric-card-label">Strong Technical Fit (≥75%)</div>
            <div class="metric-card-value" style="color: #5B8C72;">{strong_matches} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #5B8C72;">
            Status: <strong>Ready for Technical Screen</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div>
            <div class="metric-card-label">Review / Moderate (40-74%)</div>
            <div class="metric-card-value" style="color: #D99A24;">{moderate_matches} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #D99A24;">
            Status: <strong>Review Missing Competencies</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div>
            <div class="metric-card-label">High Technical Gaps (&lt;40%)</div>
            <div class="metric-card-value" style="color: #D95D39;">{high_gaps} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #D95D39;">
            Status: <strong>Core Competency Mismatch</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Section 02: Candidate Ranking Leaderboard
st.markdown("""
<div class="section-header">
    <span class="section-num">02</span>
    <h3 class="section-title">Candidate Ranking Leaderboard</h3>
</div>
""", unsafe_allow_html=True)

tab_chart, tab_table = st.tabs(["Leaderboard Visual Chart", "Structured Candidate Data Table"])

with tab_chart:
    sorted_chart_df = rankings_df.sort_values(by="composite_score", ascending=True)
    
    bar_colors = []
    for s in sorted_chart_df["composite_score"]:
        if s >= 75.0:
            bar_colors.append("#087F7B")
        elif s >= 40.0:
            bar_colors.append("#D99A24")
        else:
            bar_colors.append("#D95D39")
            
    fig = go.Figure(go.Bar(
        x=sorted_chart_df["composite_score"],
        y=[f"#{r} {c}" for r, c in zip(sorted_chart_df["Rank"], sorted_chart_df["candidate_id"])],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="#D5DDD6", width=1)
        ),
        text=[f"{s:.1f}%" for s in sorted_chart_df["composite_score"]],
        textposition="outside",
        cliponaxis=False
    ))
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=40, t=30, b=20),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#53636A"),
        xaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            gridcolor="#F1F3EE",
            zerolinecolor="#D5DDD6",
            tickfont=dict(size=10, color="#53636A")
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#1F2933", weight=600)
        ),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_table:
    def render_candidate_table(df):
        rows_html = []
        for _, row in df.iterrows():
            is_top = (row["Rank"] == 1)
            row_class = ' class="top-candidate-row"' if is_top else ''
            
            # Gap Badge
            if row["gap_severity"] == "Low":
                gap_badge = '<span class="badge-success">LOW GAP</span>'
            elif row["gap_severity"] == "Moderate":
                gap_badge = '<span class="badge-medium">MODERATE</span>'
            else:
                gap_badge = '<span class="badge-urgent">HIGH GAP</span>'
                
            cells = [
                f"<td style='font-weight:800; text-align:center;'>#{row['Rank']}</td>",
                f"<td><code>{row['candidate_id']}</code></td>",
                f"<td style='font-weight:800; color:#075E5B;'>{row['composite_score']:.1f}%</td>",
                f"<td>{gap_badge}</td>",
                f"<td>{row['skill_score_pct']:.1f}%</td>",
                f"<td>{row['semantic_similarity_pct']:.1f}%</td>",
                f"<td>{row['experience_years']} yrs</td>",
                f"<td>{row['education_level']}</td>"
            ]
            rows_html.append(f"<tr{row_class}>{''.join(cells)}</tr>")
            
        return f"""
        <div class="benchmark-table-container">
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th style="text-align:center;">Rank</th>
                        <th>Candidate Profile</th>
                        <th>Composite Match</th>
                        <th>Gap Severity</th>
                        <th>Skill Overlap</th>
                        <th>Semantic Fit</th>
                        <th>Experience</th>
                        <th>Education</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
    st.markdown(render_candidate_table(rankings_df), unsafe_allow_html=True)

# Section 03: Deep-Dive Candidate Inspector & Skill Gap Diagnostics
st.markdown("""
<div class="section-header">
    <span class="section-num">03</span>
    <h3 class="section-title">Candidate Deep-Dive & Skill Gap Diagnostics</h3>
</div>
""", unsafe_allow_html=True)

selected_cand_id = st.selectbox(
    "Select Candidate Profile to Inspect:",
    rankings_df["candidate_id"].tolist()
)

cand_row = rankings_df[rankings_df["candidate_id"] == selected_cand_id].iloc[0]

c_col1, c_col2 = st.columns([1, 1])

with c_col1:
    st.markdown(f"""
    <div class="content-box">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; border-bottom: 1px solid #F1F3EE; padding-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #075E5B;">
                <code>{cand_row['candidate_id']}</code>
            </div>
            <div>
                <span class="badge-neutral">Rank #{cand_row['Rank']}</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px;">
            <div><span style="color: #53636A; font-size: 0.82rem;">Composite Match:</span><br><strong style="font-size: 1.1rem; color: #075E5B;">{cand_row['composite_score']:.1f}%</strong></div>
            <div><span style="color: #53636A; font-size: 0.82rem;">Gap Severity:</span><br><strong style="font-size: 1.0rem; color: #1F2933;">{cand_row['gap_severity']}</strong></div>
            <div><span style="color: #53636A; font-size: 0.82rem;">Stated Experience:</span><br><strong>{cand_row['experience_years']} Years</strong></div>
            <div><span style="color: #53636A; font-size: 0.82rem;">Highest Education:</span><br><strong>{cand_row['education_level']}</strong></div>
        </div>
        <div style="margin-bottom: 12px;">
            <div style="font-size: 0.84rem; font-weight: 700; color: #1F2933; margin-bottom: 4px;">Decision-Support Recommendation:</div>
            <div style="background-color: #F8F7F3; border: 1px solid #D5DDD6; border-radius: 8px; padding: 10px 12px; font-size: 0.85rem; color: #1F2933; line-height: 1.4;">
                {cand_row['recommendation']}
            </div>
        </div>
        <div style="font-size: 0.84rem; font-weight: 700; color: #1F2933; margin-bottom: 6px;">Evaluation Pillar Scores:</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
            <div style="background:#FFFFFF; border:1px solid #D5DDD6; border-radius:6px; padding:6px 10px;">
                <span style="font-size:0.75rem; color:#53636A;">Structured Skill:</span> <strong>{cand_row['skill_score_pct']:.1f}%</strong>
            </div>
            <div style="background:#FFFFFF; border:1px solid #D5DDD6; border-radius:6px; padding:6px 10px;">
                <span style="font-size:0.75rem; color:#53636A;">Dense Semantic:</span> <strong>{cand_row['semantic_similarity_pct']:.1f}%</strong>
            </div>
            <div style="background:#FFFFFF; border:1px solid #D5DDD6; border-radius:6px; padding:6px 10px;">
                <span style="font-size:0.75rem; color:#53636A;">TF-IDF Lexical:</span> <strong>{cand_row['lexical_similarity_pct']:.1f}%</strong>
            </div>
            <div style="background:#FFFFFF; border:1px solid #D5DDD6; border-radius:6px; padding:6px 10px;">
                <span style="font-size:0.75rem; color:#53636A;">Exp/Edu Fit:</span> <strong>{cand_row['exp_edu_fit_pct']:.1f}%</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_col2:
    matched_req_html = " ".join([f"<span class='badge-success' style='margin-right:4px; margin-bottom:4px;'>✓ {s}</span>" for s in cand_row['matched_required']]) if cand_row['matched_required'] else "<span style='color:#53636A; font-style:italic;'>None identified</span>"
    missing_req_html = " ".join([f"<span class='badge-urgent' style='margin-right:4px; margin-bottom:4px;'>✗ {s}</span>" for s in cand_row['missing_required']]) if cand_row['missing_required'] else "<span class='badge-success'>🎉 All mandatory competencies satisfied!</span>"
    matched_pref_html = " ".join([f"<span class='badge-teal' style='margin-right:4px; margin-bottom:4px;'>★ {s}</span>" for s in cand_row['matched_preferred']]) if cand_row['matched_preferred'] else "<span style='color:#53636A; font-style:italic;'>None identified</span>"
    
    st.markdown(f"""
    <div class="content-box">
        <div style="font-size: 1.05rem; font-weight: 800; color: #1F2933; margin-bottom: 12px; border-bottom: 1px solid #F1F3EE; padding-bottom: 8px;">
            Granular Skill Competency Matrix
        </div>
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.82rem; font-weight: 800; color: #5B8C72; text-transform: uppercase; margin-bottom: 6px;">
                ✅ Matched Mandatory Skills ({len(cand_row['matched_required'])} / {len(active_jd['required_skills'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {matched_req_html}
            </div>
        </div>
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.82rem; font-weight: 800; color: #D95D39; text-transform: uppercase; margin-bottom: 6px;">
                ❌ Missing Mandatory Skills ({len(cand_row['missing_required'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {missing_req_html}
            </div>
        </div>
        <div>
            <div style="font-size: 0.82rem; font-weight: 800; color: #075E5B; text-transform: uppercase; margin-bottom: 6px;">
                ⭐ Matched Preferred Bonus Skills ({len(cand_row['matched_preferred'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {matched_pref_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Section 04: Data Governance & Ethical AI Notice
st.markdown("""
<div class="section-header">
    <span class="section-num">04</span>
    <h3 class="section-title">Data Provenance & Ethical AI Governance</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #FFFFFF; border: 1px solid #D5DDD6; border-radius: 14px; padding: 18px 22px; margin-top: 6px;">
    <h4 style="color: #1F2933; margin-top: 0; font-size: 0.96rem; font-weight: 700;">
        Controlled Prototype Scope & Ethical AI Safeguards
    </h4>
    <p style="font-size: 0.86rem; color: #53636A; line-height: 1.5; margin-bottom: 8px;">
        <strong style="color: #1F2933;">Benchmark Corpus:</strong> Controlled, synthetic candidate resume dataset (8 profiles across PDF, DOCX, and TXT formats) curated to validate multi-format parsing, PII anonymization, 4-tier hybrid scoring, and skill-gap extraction in an objective prototype environment.
    </p>
    <p style="font-size: 0.86rem; color: #53636A; line-height: 1.5; margin-bottom: 8px;">
        <strong style="color: #1F2933;">PII Anonymization:</strong> Automatically scrubs candidate names, email addresses, phone numbers, and profile links prior to feature extraction to mitigate demographic and unconscious recruiter bias.
    </p>
    <p style="font-size: 0.86rem; color: #53636A; line-height: 1.5; margin-bottom: 0;">
        <strong style="color: #1F2933;">Decision-Support Notice:</strong> Because this is a curated synthetic prototype dataset, these results should not be interpreted as representative of production recruiting performance. Real-world deployment requires validation on an independently audited candidate pool and must serve strictly as a decision-support filter rather than an autonomous hiring authority.
    </p>
</div>
""", unsafe_allow_html=True)

