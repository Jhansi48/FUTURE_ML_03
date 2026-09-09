"""
TalentMatch ML - Recruiter Intelligence & Candidate Screening Workspace
Interactive Streamlit decision-support platform for objective, skill-based candidate screening,
semantic job-fit matching, and transparent skill-gap diagnostics.
"""

import os
import glob
import json
import sys
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.parser import parse_resume
from src.matcher import TalentMatcher

# Helper function to render HTML cleanly without markdown code block indentation issues
def html(content: str):
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)

# Configure Page Layout & Styling
st.set_page_config(
    page_title="TalentMatch ML — Recruiter Intelligence Workspace",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Light Theme CSS
html("""
<style>
    /* Global Typography & Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1F2933;
    }
    
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.8rem !important;
        max-width: 1440px !important;
    }

    /* Global Canvas */
    .stApp {
        background-color: #F8F7F2 !important;
        color: #1F2933 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EBEFEA !important;
        border-right: 1px solid #D5DDD6 !important;
        min-width: 350px !important;
        max-width: 420px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #1F2933 !important;
        font-weight: 800 !important;
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
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] code {
        background-color: #FFFFFF !important;
        color: #075E5B !important;
        border: 1px solid #D5DDD6 !important;
        padding: 2px 7px !important;
        border-radius: 5px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 14px 0 !important;
        border-color: #D5DDD6 !important;
    }

    /* Structured Section Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #D8E0D8;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 3px 12px rgba(31, 41, 51, 0.04), 0 1px 3px rgba(31, 41, 51, 0.02);
        margin-bottom: 18px;
    }

    /* Hero Banner */
    .hero-container {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(31, 41, 51, 0.04), 0 1px 3px rgba(31, 41, 51, 0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .hero-title-group {
        display: flex;
        flex-direction: column;
    }

    .hero-badge-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .hero-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EAF3ED;
        color: #2E7D5B;
        border: 1px solid #BFDEC7;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #2E7D5B;
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 900;
        color: #075E5B;
        margin: 0;
        line-height: 1.15;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1.0rem;
        font-weight: 600;
        color: #53636A;
        margin-top: 4px;
    }

    /* Visual Workflow Steps */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.03);
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 10px;
        border-radius: 8px;
        background: #F8F7F2;
        border: 1px solid #E5EBE5;
    }
    
    .workflow-step-num {
        background-color: #087F7B;
        color: #FFFFFF;
        font-size: 0.70rem;
        font-weight: 800;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .workflow-step-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #1F2933;
    }
    
    .workflow-arrow {
        color: #087F7B;
        font-size: 1.1rem;
        font-weight: 800;
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
        background: #087F7B;
        color: #FFFFFF;
        font-size: 0.76rem;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    
    .section-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #1F2933;
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D8E0D8;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 10px rgba(31, 41, 51, 0.03), 0 1px 3px rgba(31, 41, 51, 0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }

    .metric-accent-line {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .metric-card-label {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #53636A;
        margin-bottom: 6px;
    }
    
    .metric-card-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #1F2933;
        line-height: 1.2;
    }
    
    .metric-card-sub {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid #F1F3EE;
    }

    /* Status Badges & Chips */
    .badge-urgent {
        background-color: #FBEAE5;
        color: #D95D39;
        border: 1px solid #F4C7BA;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.78rem;
        display: inline-block;
    }
    
    .badge-medium {
        background-color: #FFF4D8;
        color: #D99A24;
        border: 1px solid #F7DE98;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.78rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #EAF3ED;
        color: #2E7D5B;
        border: 1px solid #BFDEC7;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.78rem;
        display: inline-block;
    }
    
    .badge-teal {
        background-color: #E7F4F2;
        color: #075E5B;
        border: 1px solid #B8E2DC;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    .badge-neutral {
        background-color: #EBF0EA;
        color: #1F2933;
        border: 1px solid #D5DDD6;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }

    .skill-chip {
        background-color: #FFFFFF;
        border: 1px solid #D5DDD6;
        color: #1F2933;
        padding: 4px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin: 2px;
    }

    .skill-chip-req {
        background-color: #EAF3ED;
        border: 1px solid #BFDEC7;
        color: #2E7D5B;
    }

    .skill-chip-miss {
        background-color: #FBEAE5;
        border: 1px solid #F4C7BA;
        color: #D95D39;
    }

    .skill-chip-pref {
        background-color: #FFF4D8;
        border: 1px solid #F7DE98;
        color: #B57E12;
    }

    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DDD6 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #1F2933 !important;
        background-color: #FFFFFF !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
        background-color: #FFFFFF !important;
        color: #1F2933 !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #E7F4F2 !important;
        color: #075E5B !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #1F2933 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: #53636A !important;
        padding: 8px 16px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #075E5B !important;
        border-bottom-color: #087F7B !important;
    }

    /* Custom Benchmark Table Styling */
    .benchmark-table-container {
        background: #FFFFFF;
        border: 1px solid #D8E0D8;
        border-radius: 12px;
        overflow-x: auto;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.03);
    }
    
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.88rem;
    }
    
    .benchmark-table th {
        background-color: #F8F7F2;
        color: #1F2933;
        font-weight: 800;
        padding: 12px 14px;
        border-bottom: 2px solid #D5DDD6;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .benchmark-table td {
        padding: 12px 14px;
        border-bottom: 1px solid #F1F3EE;
        color: #1F2933;
        vertical-align: middle;
        background-color: #FFFFFF;
    }
    
    .benchmark-table tr:last-child td {
        border-bottom: none;
    }
    
    .benchmark-table tr.top-candidate-row td {
        background-color: #EBF5F3 !important;
        color: #075E5B !important;
        font-weight: 700;
    }

    /* Mini Pillar Cards */
    .pillar-card {
        background: #F8F7F2;
        border: 1px solid #D8E0D8;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: left;
    }

    .pillar-weight {
        font-size: 0.70rem;
        font-weight: 800;
        color: #087F7B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .pillar-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #1F2933;
        margin-top: 2px;
    }

    .pillar-score {
        font-size: 1.25rem;
        font-weight: 800;
        color: #075E5B;
        margin-top: 4px;
    }

    /* Download Button Polish */
    div.stDownloadButton > button {
        background-color: #087F7B !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid #075E5B !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 6px rgba(8, 127, 123, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #075E5B !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(8, 127, 123, 0.3) !important;
    }

    div.stDownloadButton > button p {
        color: #FFFFFF !important;
    }
</style>
""")

# Data & Model Setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resumes_dir = os.path.join(base_dir, "data", "resumes")
jds_dir = os.path.join(base_dir, "data", "job_descriptions")

@st.cache_resource
def load_matcher():
    return TalentMatcher(use_dense_embeddings=True)

matcher = load_matcher()

# 1. Hero & Visual Pipeline
html("""
<div class="hero-container">
    <div class="hero-title-group">
        <div class="hero-badge-row">
            <span class="hero-status-pill">
                <span class="pulse-dot"></span>
                Screening Engine • Ready
            </span>
            <span class="badge-neutral" style="font-size: 0.72rem; font-weight: 700;">FIT/AUG26/ML10465</span>
        </div>
        <h1 class="hero-title">TALENTMATCH ML</h1>
        <div class="hero-subtitle">
            Resume Intelligence & Candidate Decision-Support Platform
        </div>
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-teal">4-TIER HYBRID SCORING</span>
        <span class="badge-teal">PII ANONYMIZATION</span>
        <span class="badge-neutral">CONTROLLED BENCHMARK</span>
        <span class="badge-neutral">DECISION SUPPORT ONLY</span>
    </div>
</div>
""")

html("""
<div class="workflow-container">
    <div class="workflow-step">
        <span class="workflow-step-num">1</span>
        <span class="workflow-step-name">Multi-Format Ingestion</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">2</span>
        <span class="workflow-step-name">PII Privacy Filter</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">3</span>
        <span class="workflow-step-name">Skill Taxonomy (200+)</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">4</span>
        <span class="workflow-step-name">4-Tier Hybrid Match</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">5</span>
        <span class="workflow-step-name">Candidate Ranking</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">6</span>
        <span class="workflow-step-name">Recruiter Insights</span>
    </div>
</div>
""")

# Sidebar Configuration
html("""
<div style="padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid #D5DDD6;">
    <div style="font-size: 1.2rem; font-weight: 900; color: #075E5B; letter-spacing: -0.02em;">
        TALENTMATCH ML
    </div>
    <div style="font-size: 0.80rem; font-weight: 600; color: #53636A;">
        Recruiter Intelligence Workspace
    </div>
</div>
""")

st.sidebar.markdown("#### 🎯 Target Role Configuration")

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

st.sidebar.markdown(f"**Target Role ID:** `{active_jd.get('job_id', 'N/A')}`")
st.sidebar.markdown(f"**Min Required Experience:** `{active_jd['min_experience_years']} Years`")

st.sidebar.markdown(f"**Mandatory Competencies ({len(active_jd['required_skills'])}):**")
req_pills = " ".join([f"<span class='skill-chip skill-chip-req'>{s}</span>" for s in active_jd['required_skills']])
st.sidebar.markdown(req_pills, unsafe_allow_html=True)

st.sidebar.markdown(f"**Preferred Competencies ({len(active_jd['preferred_skills'])}):**")
pref_pills = " ".join([f"<span class='skill-chip skill-chip-pref'>{s}</span>" for s in active_jd['preferred_skills']])
st.sidebar.markdown(pref_pills, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚖️ Hybrid Scoring Architecture")
st.sidebar.markdown("""
- **Hard Skill Overlap:** `40%` (Mandatory & Preferred)
- **Dense Semantic Fit:** `30%` (Contextual all-MiniLM)
- **TF-IDF Lexical Match:** `20%` (Sublinear Keyword TF-IDF)
- **Experience & Education:** `10%` (Tenure & Degree Level)
""")

# Ingest and Score Resumes
resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))
if not resume_files:
    st.warning("⚠️ No candidate resumes found in `data/resumes/`. Please run `python src/pipeline.py`.")
    st.stop()

candidate_profiles = [parse_resume(f) for f in sorted(resume_files)]
rankings_df = matcher.rank_candidates(candidate_profiles, active_jd)

# Quick CSV Export in Sidebar
csv_data = rankings_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Export Rankings to CSV",
    data=csv_data,
    file_name=f"talentmatch_rankings_{active_jd.get('job_id', 'role').lower()}.csv",
    mime="text/csv",
    use_container_width=True
)

# Section 01: Screening Command Center
html(f"""
<div class="section-header">
    <span class="section-num">01</span>
    <h3 class="section-title">Screening Command Center — {active_jd['title']}</h3>
</div>
""")

col1, col2, col3, col4 = st.columns(4)

total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 75.0])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 40.0) & (rankings_df["composite_score"] < 75.0)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 40.0])

with col1:
    html(f"""
    <div class="metric-card">
        <div class="metric-accent-line" style="background-color: #087F7B;"></div>
        <div>
            <div class="metric-card-label">Total Resumes Ingested</div>
            <div class="metric-card-value" style="color: #075E5B;">{total_cands} Profiles</div>
        </div>
        <div class="metric-card-sub" style="color: #075E5B;">
            Formats: <strong>PDF, DOCX, TXT</strong>
        </div>
    </div>
    """)

with col2:
    html(f"""
    <div class="metric-card">
        <div class="metric-accent-line" style="background-color: #5B8C72;"></div>
        <div>
            <div class="metric-card-label">Strong Technical Fits (≥75%)</div>
            <div class="metric-card-value" style="color: #2E7D5B;">{strong_matches} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #2E7D5B;">
            Action: <strong>Immediate Technical Screen</strong>
        </div>
    </div>
    """)

with col3:
    html(f"""
    <div class="metric-card">
        <div class="metric-accent-line" style="background-color: #D99A24;"></div>
        <div>
            <div class="metric-card-label">Review / Moderate (40-74%)</div>
            <div class="metric-card-value" style="color: #B57E12;">{moderate_matches} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #B57E12;">
            Action: <strong>Review Specific Competency Gaps</strong>
        </div>
    </div>
    """)

with col4:
    html(f"""
    <div class="metric-card">
        <div class="metric-accent-line" style="background-color: #D95D39;"></div>
        <div>
            <div class="metric-card-label">High Technical Gaps (&lt;40%)</div>
            <div class="metric-card-value" style="color: #BF4320;">{high_gaps} Candidates</div>
        </div>
        <div class="metric-card-sub" style="color: #BF4320;">
            Action: <strong>Domain / Competency Mismatch</strong>
        </div>
    </div>
    """)

# Section 02: Candidate Leaderboard & Rankings
html("""
<div class="section-header">
    <span class="section-num">02</span>
    <h3 class="section-title">Candidate Ranking Leaderboard</h3>
</div>
""")

tab_chart, tab_table = st.tabs(["📊 Visual Ranking Chart", "📋 Structured Candidate Data Table"])

with tab_chart:
    sorted_chart_df = rankings_df.sort_values(by="composite_score", ascending=True)
    
    bar_colors = []
    for s in sorted_chart_df["composite_score"]:
        if s >= 75.0:
            bar_colors.append("#2E7D5B")
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
        text=[f"  <b>{s:.1f}%</b> ({g})" for s, g in zip(sorted_chart_df["composite_score"], sorted_chart_df["gap_severity"])],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Composite Match: %{x:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=60, t=25, b=20),
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#53636A"),
        xaxis=dict(
            range=[0, 108],
            ticksuffix="%",
            gridcolor="#F1F3EE",
            zerolinecolor="#D5DDD6",
            tickfont=dict(size=11, color="#53636A")
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1F2933", weight=600)
        ),
        height=340
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_table:
    def render_candidate_table(df):
        rows_html = []
        for _, row in df.iterrows():
            is_top = (row["Rank"] == 1)
            row_class = ' class="top-candidate-row"' if is_top else ''
            
            # Rank medal
            if row["Rank"] == 1:
                rank_badge = "🥇 #1"
            elif row["Rank"] == 2:
                rank_badge = "🥈 #2"
            elif row["Rank"] == 3:
                rank_badge = "🥉 #3"
            else:
                rank_badge = f"#{row['Rank']}"
            
            # Gap Badge
            if row["gap_severity"] == "Low":
                gap_badge = '<span class="badge-success">LOW GAP</span>'
            elif row["gap_severity"] == "Moderate":
                gap_badge = '<span class="badge-medium">MODERATE</span>'
            else:
                gap_badge = '<span class="badge-urgent">HIGH GAP</span>'
                
            cells = [
                f"<td style='font-weight:800; text-align:center;'>{rank_badge}</td>",
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
    html(render_candidate_table(rankings_df))

# Why This Ranking Insight Box
top_row = rankings_df.iloc[0]
html(f"""
<div style="background-color: #FFFFFF; border: 1px solid #D8E0D8; border-radius: 12px; padding: 14px 18px; margin-top: 6px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(31, 41, 51, 0.02);">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
        <span style="font-size: 0.88rem; font-weight: 800; color: #075E5B; text-transform: uppercase; letter-spacing: 0.04em;">💡 Recruiter Decision Context: Why This Ranking?</span>
    </div>
    <div style="font-size: 0.86rem; color: #53636A; line-height: 1.45;">
        Candidates are ranked strictly via the 4-tier objective scoring architecture (<strong>40% Hard Skills</strong>, <strong>30% Dense Semantic Similarity</strong>, <strong>20% TF-IDF Keyword Match</strong>, <strong>10% Experience & Education</strong>). 
        Top-ranked profile <code>{top_row['candidate_id']}</code> attained the highest composite score (<strong>{top_row['composite_score']:.1f}%</strong>) by satisfying <strong>{len(top_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory competencies</strong> and demonstrating the strongest contextual domain alignment ({top_row['semantic_similarity_pct']:.1f}% semantic fit).
    </div>
</div>
""")

# Section 03: Candidate Deep-Dive & Skill Gap Diagnostics
html("""
<div class="section-header">
    <span class="section-num">03</span>
    <h3 class="section-title">Candidate Deep-Dive & Profile Analysis Workspace</h3>
</div>
""")

selected_cand_id = st.selectbox(
    "Select Candidate Profile to Inspect:",
    rankings_df["candidate_id"].tolist()
)

cand_row = rankings_df[rankings_df["candidate_id"] == selected_cand_id].iloc[0]

# Extract format from files
cand_file_ext = "DOCX"
for f in resume_files:
    if cand_row['candidate_id'] in os.path.basename(f):
        cand_file_ext = os.path.splitext(f)[1].replace(".", "").upper()
        break

c_col1, c_col2 = st.columns([1, 1])

with c_col1:
    # Color badge based on score
    if cand_row['composite_score'] >= 75.0:
        score_color = "#2E7D5B"
        gap_pill = '<span class="badge-success">LOW GAP</span>'
    elif cand_row['composite_score'] >= 40.0:
        score_color = "#B57E12"
        gap_pill = '<span class="badge-medium">MODERATE GAP</span>'
    else:
        score_color = "#BF4320"
        gap_pill = '<span class="badge-urgent">HIGH GAP</span>'

    html(f"""
    <div class="content-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; border-bottom: 1px solid #F1F3EE; padding-bottom: 10px;">
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #075E5B;">
                    <code>{cand_row['candidate_id']}</code>
                </div>
                <div style="font-size: 0.78rem; color: #53636A; font-weight: 600; margin-top: 2px;">
                    Parsed Format: <span class="badge-neutral" style="font-size: 0.72rem; padding: 2px 6px;">.{cand_file_ext}</span> • PII Masked
                </div>
            </div>
            <div style="display: flex; gap: 6px; align-items: center;">
                <span class="badge-teal" style="font-weight: 800;">Rank #{cand_row['Rank']}</span>
                {gap_pill}
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;">
            <div style="background: #F8F7F2; border: 1px solid #E5EBE5; border-radius: 8px; padding: 10px 12px;">
                <span style="color: #53636A; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">Composite Match</span><br>
                <strong style="font-size: 1.45rem; color: {score_color};">{cand_row['composite_score']:.1f}%</strong>
            </div>
            <div style="background: #F8F7F2; border: 1px solid #E5EBE5; border-radius: 8px; padding: 10px 12px;">
                <span style="color: #53636A; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">Stated Experience</span><br>
                <strong style="font-size: 1.25rem; color: #1F2933;">{cand_row['experience_years']} Years</strong> 
                <span style="font-size: 0.76rem; color: #53636A;">(Req: {active_jd['min_experience_years']}y)</span>
            </div>
            <div style="background: #F8F7F2; border: 1px solid #E5EBE5; border-radius: 8px; padding: 10px 12px;">
                <span style="color: #53636A; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">Highest Education</span><br>
                <strong style="font-size: 1.05rem; color: #1F2933;">{cand_row['education_level']}</strong>
            </div>
            <div style="background: #F8F7F2; border: 1px solid #E5EBE5; border-radius: 8px; padding: 10px 12px;">
                <span style="color: #53636A; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">Gap Severity</span><br>
                <strong style="font-size: 1.05rem; color: #1F2933;">{cand_row['gap_severity']}</strong>
            </div>
        </div>

        <div style="font-size: 0.82rem; font-weight: 800; color: #1F2933; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
            4-Tier Evaluation Pillar Breakdown:
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px;">
            <div class="pillar-card">
                <div class="pillar-weight">Tier 1 • Weight 40%</div>
                <div class="pillar-name">Hard Skill Overlap</div>
                <div class="pillar-score">{cand_row['skill_score_pct']:.1f}%</div>
            </div>
            <div class="pillar-card">
                <div class="pillar-weight">Tier 2 • Weight 30%</div>
                <div class="pillar-name">Dense Semantic Fit</div>
                <div class="pillar-score">{cand_row['semantic_similarity_pct']:.1f}%</div>
            </div>
            <div class="pillar-card">
                <div class="pillar-weight">Tier 3 • Weight 20%</div>
                <div class="pillar-name">TF-IDF Lexical Match</div>
                <div class="pillar-score">{cand_row['lexical_similarity_pct']:.1f}%</div>
            </div>
            <div class="pillar-card">
                <div class="pillar-weight">Tier 4 • Weight 10%</div>
                <div class="pillar-name">Experience & Education</div>
                <div class="pillar-score">{cand_row['exp_edu_fit_pct']:.1f}%</div>
            </div>
        </div>

        <div>
            <div style="font-size: 0.82rem; font-weight: 800; color: #1F2933; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Operational Recommendation:
            </div>
            <div style="background-color: #F8F7F2; border: 1px solid #D5DDD6; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #1F2933; line-height: 1.45;">
                {cand_row['recommendation']}
            </div>
        </div>
    </div>
    """)

with c_col2:
    matched_req_html = " ".join([f"<span class='skill-chip skill-chip-req'>✓ {s}</span>" for s in cand_row['matched_required']]) if cand_row['matched_required'] else "<span style='color:#53636A; font-style:italic; font-size:0.85rem;'>None identified</span>"
    missing_req_html = " ".join([f"<span class='skill-chip skill-chip-miss'>✗ {s}</span>" for s in cand_row['missing_required']]) if cand_row['missing_required'] else "<span class='badge-success' style='font-size:0.82rem;'>🎉 All mandatory competencies satisfied!</span>"
    matched_pref_html = " ".join([f"<span class='skill-chip skill-chip-pref'>★ {s}</span>" for s in cand_row['matched_preferred']]) if cand_row['matched_preferred'] else "<span style='color:#53636A; font-style:italic; font-size:0.85rem;'>None identified</span>"
    
    # Recruiter Insight synthesis
    if cand_row['composite_score'] >= 75.0 and len(cand_row['missing_required']) == 0:
        insight_bg = "#EAF3ED"
        insight_border = "#BFDEC7"
        insight_title_color = "#2E7D5B"
        insight_text = f"Candidate satisfies <strong>all {len(active_jd['required_skills'])} mandatory technical skills</strong> with strong domain vector alignment ({cand_row['semantic_similarity_pct']:.1f}%). Recommended to advance immediately to technical screen."
    elif cand_row['composite_score'] >= 40.0:
        insight_bg = "#FFF4D8"
        insight_border = "#F7DE98"
        insight_title_color = "#B57E12"
        missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "none"
        insight_text = f"Candidate displays partial technical alignment ({len(cand_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory skills). Primary gaps to assess: <strong>{missing_str}</strong>. Suitable for hiring manager review or secondary role routing."
    else:
        insight_bg = "#FBEAE5"
        insight_border = "#F4C7BA"
        insight_title_color = "#BF4320"
        missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "core skills"
        insight_text = f"Candidate demonstrates low alignment for this specific role (missing: <strong>{missing_str}</strong>). Profile indicates primary background in adjacent software domains."

    html(f"""
    <div class="content-box">
        <div style="font-size: 1.05rem; font-weight: 800; color: #1F2933; margin-bottom: 14px; border-bottom: 1px solid #F1F3EE; padding-bottom: 8px;">
            Granular Competency Matrix & Gap Breakdown
        </div>
        
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.80rem; font-weight: 800; color: #2E7D5B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                ✅ Matched Mandatory Skills ({len(cand_row['matched_required'])} / {len(active_jd['required_skills'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {matched_req_html}
            </div>
        </div>
        
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.80rem; font-weight: 800; color: #BF4320; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                ❌ Missing Mandatory Skills ({len(cand_row['missing_required'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {missing_req_html}
            </div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.80rem; font-weight: 800; color: #B57E12; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                ⭐ Matched Preferred Bonus Skills ({len(cand_row['matched_preferred'])}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {matched_pref_html}
            </div>
        </div>

        <div style="background-color: {insight_bg}; border: 1px solid {insight_border}; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.82rem; font-weight: 800; color: {insight_title_color}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">
                🧠 Recruiter Intelligence Insight
            </div>
            <div style="font-size: 0.85rem; color: #1F2933; line-height: 1.45;">
                {insight_text}
            </div>
        </div>
    </div>
    """)

# Section 04: Model Governance & Data Provenance
html("""
<div class="section-header">
    <span class="section-num">04</span>
    <h3 class="section-title">Model Governance & Data Provenance</h3>
</div>
""")

gov_col1, gov_col2, gov_col3 = st.columns(3)

with gov_col1:
    html("""
    <div class="content-box" style="height: 100%;">
        <div style="font-size: 0.90rem; font-weight: 800; color: #075E5B; margin-bottom: 8px;">
            📊 Controlled Benchmark Scope
        </div>
        <div style="font-size: 0.84rem; color: #53636A; line-height: 1.45;">
            Controlled, synthetic candidate resume dataset (8 profiles across PDF, DOCX, and TXT formats) curated to validate multi-format parsing, PII anonymization, 4-tier hybrid scoring, and skill-gap extraction in an objective prototype environment.
        </div>
    </div>
    """)

with gov_col2:
    html("""
    <div class="content-box" style="height: 100%;">
        <div style="font-size: 0.90rem; font-weight: 800; color: #075E5B; margin-bottom: 8px;">
            🛡️ Zero-PII Fairness Safeguards
        </div>
        <div style="font-size: 0.84rem; color: #53636A; line-height: 1.45;">
            Candidate names, email addresses, phone numbers, and profile URLs are automatically scrubbed from resume text prior to feature extraction to mitigate demographic, gender, and unconscious recruiter bias.
        </div>
    </div>
    """)

with gov_col3:
    html("""
    <div class="content-box" style="height: 100%;">
        <div style="font-size: 0.90rem; font-weight: 800; color: #075E5B; margin-bottom: 8px;">
            ⚖️ Decision-Support Notice
        </div>
        <div style="font-size: 0.84rem; color: #53636A; line-height: 1.45;">
            TalentMatch ML is explicitly engineered as a recruiter decision-support tool, not an autonomous hiring engine. Real-world deployment requires production validation, and human evaluation remains mandatory for all hiring decisions.
        </div>
    </div>
    """)
