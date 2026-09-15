"""
TalentMatch ML - Recruiter Intelligence & Candidate Screening Platform
Interactive decision-support platform for objective, skill-based candidate screening,
dense semantic job-fit matching, and transparent skill-gap diagnostics.
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
    page_title="TalentMatch ML — Recruiter Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sophisticated Lavender / Off-White Enterprise SaaS Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #25233A;
    }
    
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Primary Background: Clean Light Lavender-Gray */
    .stApp {
        background-color: #F3F0F9 !important;
        color: #25233A !important;
    }
    
    /* Sidebar: Cohesive Lavender */
    section[data-testid="stSidebar"] {
        background-color: #EAE5F3 !important;
        border-right: 1px solid #DDD7EA !important;
        min-width: 350px !important;
        max-width: 400px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #25233A !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
        margin-bottom: 6px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: #6E6A7B !important;
    }

    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #25233A !important;
    }

    section[data-testid="stSidebar"] label p {
        color: #25233A !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 12px 0 !important;
        border-color: #DDD7EA !important;
    }

    /* Light, Professional Candidate ID & Role Chips - No Terminal/Code Appearance */
    .saas-candidate-chip {
        display: inline-block;
        background-color: #EDE9FE;
        color: #5B46D6;
        border: 1px solid #DDD6FE;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }

    .saas-id-badge {
        display: inline-block;
        background-color: #FFFFFF;
        color: #5B46D6;
        border: 1px solid #DDD7EA;
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.82rem;
        font-weight: 700;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }

    code {
        background-color: #EDE9FE !important;
        color: #5B46D6 !important;
        border: 1px solid #DDD6FE !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }

    /* Native Card Containers: White with Lavender-tinted border and soft shadow */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DDD7EA !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(107, 91, 214, 0.05), 0 1px 3px rgba(37, 35, 58, 0.03) !important;
        padding: 16px 18px !important;
        margin-bottom: 4px !important;
    }

    /* Metric Labels - Strong Dark Charcoal Contrast Everywhere */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #25233A !important;
        opacity: 1 !important;
    }

    /* Metric Values - Large, Clear, Primary Purple */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] span,
    div[data-testid="stMetricValue"] p {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #6B5BD6 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        font-size: 1.35rem !important;
    }

    /* Captions styling - High Readability Charcoal */
    .stCaption,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: #4A4658 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DDD7EA !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #25233A !important;
        background-color: #FFFFFF !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
        background-color: #FFFFFF !important;
        color: #25233A !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #EDE9FE !important;
        color: #6B5BD6 !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #25233A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Primary Purple Export Button */
    div.stDownloadButton > button {
        background-color: #6B5BD6 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid #5B4BD1 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 6px rgba(107, 91, 214, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #5848C2 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(107, 91, 214, 0.35) !important;
    }

    div.stDownloadButton > button p {
        color: #FFFFFF !important;
    }

    /* Progress bar color */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #6B5BD6 !important;
    }

    /* Custom CSS Utility Classes for Badges and Chips */
    .saas-badge-pill {
        display: inline-block;
        background-color: #EDE9FE;
        color: #5B46D6;
        border: 1px solid #DDD6FE;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .saas-badge-success {
        display: inline-block;
        background-color: #E6F4EA;
        color: #137333;
        border: 1px solid #CEEAD6;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .saas-badge-warning {
        display: inline-block;
        background-color: #FEF7E0;
        color: #B06000;
        border: 1px solid #FEEFC3;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .saas-badge-danger {
        display: inline-block;
        background-color: #FCE8E6;
        color: #C5221F;
        border: 1px solid #FAD2CF;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .chip-req {
        display: inline-block;
        background-color: #EDE9FE;
        color: #5B46D6;
        border: 1px solid #DDD6FE;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    .chip-pref {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    .chip-miss {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    .section-header {
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .section-title {
        color: #25233A;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 0.03em;
        margin-bottom: 2px;
    }

    .section-sub {
        color: #6E6A7B;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }

    /* Light, Professional Leaderboard Table */
    .saas-table-container {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #DDD7EA;
        border-radius: 8px;
        background-color: #FFFFFF;
    }

    .saas-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        text-align: left;
    }

    .saas-table th {
        background-color: #F6F3FB;
        color: #25233A;
        font-weight: 800;
        padding: 10px 14px;
        border-bottom: 1px solid #DDD7EA;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }

    .saas-table td {
        padding: 10px 14px;
        color: #25233A;
        border-bottom: 1px solid #F0ECF7;
        vertical-align: middle;
    }

    .saas-table tr:hover {
        background-color: #FAF8FE;
    }

    .saas-table tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# Helper taxonomy functions for 100% consistent terminology across all sections
def get_fit_category(score: float) -> str:
    """Returns canonical fit category based on composite score."""
    if score >= 75.0:
        return "Strong Overall Match"
    elif score >= 40.0:
        return "Moderate Match"
    else:
        return "High Technical Gap"

def get_fit_badge_class(score: float) -> str:
    if score >= 75.0:
        return "saas-badge-success"
    elif score >= 40.0:
        return "saas-badge-warning"
    else:
        return "saas-badge-danger"

def get_fit_color(score: float) -> str:
    if score >= 75.0:
        return "#4C9073"
    elif score >= 40.0:
        return "#D39A2B"
    else:
        return "#D45B61"

# Data & Model Setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resumes_dir = os.path.join(base_dir, "data", "resumes")
jds_dir = os.path.join(base_dir, "data", "job_descriptions")

@st.cache_resource
def load_matcher():
    return TalentMatcher(use_dense_embeddings=True)

matcher = load_matcher()

# Ingest and Score Resumes
resume_files = glob.glob(os.path.join(resumes_dir, "*.*"))
if not resume_files:
    st.warning("⚠️ No candidate resumes found in `data/resumes/`. Please run `python src/pipeline.py`.")
    st.stop()

candidate_profiles = [parse_resume(f) for f in sorted(resume_files)]

jd_files = glob.glob(os.path.join(jds_dir, "*.json"))
if not jd_files:
    st.error("⚠️ No Job Description JSON files found in data/job_descriptions.")
    st.stop()

jd_options = {}
for jdf in sorted(jd_files):
    with open(jdf, "r", encoding="utf-8") as f:
        data = json.load(f)
        jd_options[data["title"]] = data

# ==================================================
# SIDEBAR NAVIGATION & CONFIGURATION
# ==================================================
st.sidebar.markdown("### 🎯 TalentMatch ML")
st.sidebar.caption("Recruiter Intelligence Workspace")
st.sidebar.markdown("---")

st.sidebar.markdown("#### TARGET ROLE")
selected_jd_title = st.sidebar.selectbox("Select Role Specification:", list(jd_options.keys()), label_visibility="collapsed")
active_jd = jd_options[selected_jd_title]

# Compute rankings for selected JD
rankings_df = matcher.rank_candidates(candidate_profiles, active_jd)
rankings_df["fit_category"] = rankings_df["composite_score"].apply(get_fit_category)

total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 75.0])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 40.0) & (rankings_df["composite_score"] < 75.0)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 40.0])

st.sidebar.markdown("---")
st.sidebar.markdown("#### ROLE SNAPSHOT")
snap_col1, snap_col2 = st.sidebar.columns(2)
with snap_col1:
    st.sidebar.markdown(f"**Role ID:** <span class='saas-id-badge'>{active_jd.get('job_id', 'N/A')}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Mandatory:** `{len(active_jd['required_skills'])} Skills`")
with snap_col2:
    st.sidebar.markdown(f"**Experience:** `{active_jd['min_experience_years']}+ Yrs`")
    st.sidebar.markdown(f"**Preferred:** `{len(active_jd['preferred_skills'])} Skills`")

st.sidebar.markdown("---")
st.sidebar.markdown("#### MANDATORY SKILLS")
req_chips_sidebar = "".join([f"<span class='chip-req'>✓ {s}</span>" for s in active_jd['required_skills']])
st.sidebar.markdown(f"<div>{req_chips_sidebar}</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### PREFERRED SKILLS")
pref_chips_sidebar = "".join([f"<span class='chip-pref'>★ {s}</span>" for s in active_jd['preferred_skills']])
st.sidebar.markdown(f"<div>{pref_chips_sidebar}</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### SCORING MODEL")
st.sidebar.markdown("""
- **40%** Technical Skill Overlap
- **30%** Dense Semantic Fit
- **20%** TF-IDF Keyword Match
- **10%** Experience & Education
""")

st.sidebar.markdown("---")
# Quick CSV Export in Sidebar
csv_data = rankings_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Export Rankings CSV",
    data=csv_data,
    file_name=f"talentmatch_rankings_{active_jd.get('job_id', 'role').lower()}.csv",
    mime="text/csv",
    use_container_width=True
)

# ==================================================
# MAIN DASHBOARD CONTENT
# ==================================================

# 1. TOP PRODUCT HEADER
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.80rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:2px;'>RECRUITER INTELLIGENCE PLATFORM</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#25233A; font-weight:800; font-size:2.1rem; margin:0 0 4px 0;'>🎯 TalentMatch ML</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4A4658; font-size:0.95rem; margin-bottom:10px;'>Resume intelligence • Candidate ranking • Skill-gap diagnostics</p>", unsafe_allow_html=True)
    st.markdown(
        "<div>"
        "<span class='saas-badge-pill'>✓ PII Anonymized</span>"
        "<span class='saas-badge-pill'>✓ Multi-format Parsing</span>"
        "<span class='saas-badge-pill'>✓ Hybrid Scoring</span>"
        "<span class='saas-badge-pill'>✓ Decision Support</span>"
        "</div>",
        unsafe_allow_html=True
    )

with head_col2:
    with st.container(border=True):
        st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.75rem; letter-spacing:0.06em; margin-bottom:2px;'>DECISION SUPPORT</p>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#25233A; margin:0 0 2px 0;'>Prototype</h4>", unsafe_allow_html=True)
        st.caption("Human-in-the-loop evaluation")

st.markdown("")

# ==================================================
# TARGET ROLE SUMMARY CARD
# ==================================================
with st.container(border=True):
    st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:2px;'>TARGET ROLE</p>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:#25233A; font-weight:800; font-size:1.45rem; margin:0 0 8px 0;'>{active_jd['title']}</h2>", unsafe_allow_html=True)
    
    tr_meta1, tr_meta2, tr_meta3 = st.columns([1, 1, 2])
    with tr_meta1:
        st.markdown(f"**Role ID:** <span class='saas-id-badge'>{active_jd.get('job_id', 'N/A')}</span>", unsafe_allow_html=True)
    with tr_meta2:
        st.markdown(f"**Experience:** `{active_jd['min_experience_years']}+ Years`")
    with tr_meta3:
        st.markdown(f"**Domain:** `{active_jd.get('domain', 'Engineering & Technology')}`")
        
    st.divider()
    
    tr_s1, tr_s2 = st.columns(2)
    with tr_s1:
        st.markdown(f"**MANDATORY TECHNICAL COMPETENCIES ({len(active_jd['required_skills'])})**")
        req_chips = "".join([f"<span class='chip-req'>✓ {s}</span>" for s in active_jd['required_skills']])
        st.markdown(f"<div style='margin-top:4px;'>{req_chips}</div>", unsafe_allow_html=True)
    with tr_s2:
        st.markdown(f"**PREFERRED COMPETENCIES ({len(active_jd['preferred_skills'])})**")
        pref_chips = "".join([f"<span class='chip-pref'>★ {s}</span>" for s in active_jd['preferred_skills']])
        st.markdown(f"<div style='margin-top:4px;'>{pref_chips}</div>", unsafe_allow_html=True)

st.markdown("")

# ==================================================
# SECTION 01: SCREENING OVERVIEW (4 KPI CARDS)
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>01. SCREENING OVERVIEW</div><div class='section-sub'>Candidate-pool alignment based on prototype heuristic composite match scores.</div></div>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    with st.container(border=True):
        st.metric("Total Resumes", f"{total_cands}")
        st.caption("All parsed candidate files")

with kpi2:
    with st.container(border=True):
        st.metric("Strong Overall Match", f"{strong_matches}")
        st.caption("Score ≥ 75%")

with kpi3:
    with st.container(border=True):
        st.metric("Moderate Match", f"{moderate_matches}")
        st.caption("Score 40%–74.9%")

with kpi4:
    with st.container(border=True):
        st.metric("High Technical Gap", f"{high_gaps}")
        st.caption("Score < 40%")

st.caption("📌 *Thresholds are prototype decision-support bands, not validated hiring cutoffs.*")
st.markdown("")

# ==================================================
# SECTION 02: CANDIDATE RANKING (Chart + Top Match)
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>02. CANDIDATE RANKING</div><div class='section-sub'>Technical compatibility across skills, semantic relevance and experience.</div></div>", unsafe_allow_html=True)

rank_col1, rank_col2 = st.columns([60, 40])

top_row = rankings_df.iloc[0]

with rank_col1:
    with st.container(border=True):
        st.markdown("<h4 style='color:#25233A; margin:0 0 8px 0;'>Candidate Compatibility Ranking</h4>", unsafe_allow_html=True)
        
        sorted_chart_df = rankings_df.sort_values(by="composite_score", ascending=True)
        bar_colors = [get_fit_color(s) for s in sorted_chart_df["composite_score"]]
            
        fig = go.Figure(go.Bar(
            x=sorted_chart_df["composite_score"],
            y=[f"#{r} {c}" for r, c in zip(sorted_chart_df["Rank"], sorted_chart_df["candidate_id"])],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#DDD7EA", width=1)
            ),
            text=[f"  <b>{s:.1f}%</b> ({get_fit_category(s)})" for s in sorted_chart_df["composite_score"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Composite Match: %{x:.1f}%<br>Fit: %{text}<extra></extra>"
        ))
        
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=10, r=90, t=10, b=20),
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#4A4658"),
            xaxis=dict(
                range=[0, 115],
                ticksuffix="%",
                gridcolor="#F3F0F9",
                zerolinecolor="#DDD7EA",
                tickfont=dict(size=11, color="#4A4658")
            ),
            yaxis=dict(
                tickfont=dict(size=12, color="#25233A", weight=600)
            ),
            height=340
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with rank_col2:
    with st.container(border=True):
        st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.75rem; letter-spacing:0.06em; margin-bottom:2px;'>🏆 TOP MATCH</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-bottom:6px;'><span class='saas-badge-pill'>Rank #{top_row['Rank']}</span><span class='saas-candidate-chip'>{top_row['candidate_id']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='{get_fit_badge_class(top_row['composite_score'])}'>{get_fit_category(top_row['composite_score'])}</span>", unsafe_allow_html=True)
        st.divider()
        
        tc_m1, tc_m2, tc_m3 = st.columns(3)
        with tc_m1:
            st.metric("Composite Match", f"{top_row['composite_score']:.1f}%")
            st.metric("Experience", f"{top_row['experience_years']} Yrs")
        with tc_m2:
            st.metric("Mandatory Skills", f"{len(top_row['matched_required'])} / {len(active_jd['required_skills'])}")
            st.metric("Semantic Fit", f"{top_row['semantic_similarity_pct']:.1f}%")
        with tc_m3:
            st.metric("Preferred Skills", f"{len(top_row['matched_preferred'])} / {len(active_jd['preferred_skills'])}")
            
        st.divider()
        st.markdown(f"<p style='color:#25233A; font-size:0.88rem; line-height:1.4;'><strong>Recommendation:</strong> {top_row['recommendation']}</p>", unsafe_allow_html=True)

st.markdown("")

# ==================================================
# SECTION 03: RECRUITER INSIGHT
# ==================================================
with st.container(border=True):
    st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.75rem; letter-spacing:0.06em; margin-bottom:2px;'>💡 RECRUITER INSIGHT</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#25233A; margin:0 0 6px 0;'>WHY THIS CANDIDATE RANKS FIRST</h4>", unsafe_allow_html=True)
    
    insight_summary = f"Candidate <strong>#{top_row['Rank']} (<span class='saas-candidate-chip'>{top_row['candidate_id']}</span>)</strong> satisfies <strong>{len(top_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory technical competencies</strong> with a dense semantic fit of <strong>{top_row['semantic_similarity_pct']:.1f}%</strong> and an overall composite match score of <strong>{top_row['composite_score']:.1f}%</strong>."
        
    st.markdown(f"<p style='color:#25233A; font-size:0.92rem; margin-bottom:14px;'>{insight_summary}</p>", unsafe_allow_html=True)
    
    ins_c1, ins_c2, ins_c3, ins_c4 = st.columns(4)
    with ins_c1:
        with st.container(border=True):
            st.metric("Technical Skill Overlap", f"{top_row['skill_score_pct']:.1f}%")
            st.progress(min(1.0, top_row['skill_score_pct'] / 100.0))
            st.caption("Weight: 40% • Taxonomy overlap")
    with ins_c2:
        with st.container(border=True):
            st.metric("Semantic Fit", f"{top_row['semantic_similarity_pct']:.1f}%")
            st.progress(min(1.0, top_row['semantic_similarity_pct'] / 100.0))
            st.caption("Weight: 30% • MiniLM-L6 vector")
    with ins_c3:
        with st.container(border=True):
            st.metric("TF-IDF Match", f"{top_row['lexical_similarity_pct']:.1f}%")
            st.progress(min(1.0, top_row['lexical_similarity_pct'] / 100.0))
            st.caption("Weight: 20% • Keyword cosine")
    with ins_c4:
        with st.container(border=True):
            st.metric("Exp / Edu", f"{top_row['exp_edu_fit_pct']:.1f}%")
            st.progress(min(1.0, top_row['exp_edu_fit_pct'] / 100.0))
            st.caption("Weight: 10% • Tenure heuristic")

st.markdown("")

# ==================================================
# SECTION 04: CANDIDATE PROFILE DEEP DIVE
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>04. CANDIDATE PROFILE</div><div class='section-sub'>Individual candidate inspection and structured competency diagnostics.</div></div>", unsafe_allow_html=True)

selected_cand_id = st.selectbox(
    "Select Candidate Profile to Inspect:",
    rankings_df["candidate_id"].tolist()
)

cand_row = rankings_df[rankings_df["candidate_id"] == selected_cand_id].iloc[0]

cand_file_ext = "DOCX"
for f in resume_files:
    if cand_row['candidate_id'] in os.path.basename(f):
        cand_file_ext = os.path.splitext(f)[1].replace(".", "").upper()
        break

# Candidate Profile Card Container
with st.container(border=True):
    # Top Profile Header with Badges
    st.markdown(f"<div style='margin-bottom:6px;'><span class='saas-candidate-chip' style='font-size:1.05rem; padding:4px 12px;'>{cand_row['candidate_id']}</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-bottom:12px;'>"
        f"<span class='saas-badge-pill'>#{cand_row['Rank']} Ranked</span>"
        f"<span class='saas-badge-pill'>{cand_row['composite_score']:.1f}% Match</span>"
        f"<span class='{get_fit_badge_class(cand_row['composite_score'])}'>{get_fit_category(cand_row['composite_score'])}</span>"
        "<span class='saas-badge-pill'>PII Masked</span>"
        f"<span class='saas-badge-pill'>.{cand_file_ext}</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.divider()
    
    c_col1, c_col2 = st.columns([1, 1])
    
    with c_col1:
        st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.78rem; letter-spacing:0.06em; margin-bottom:4px;'>CANDIDATE SUMMARY</p>", unsafe_allow_html=True)
        
        m_row1, m_row2 = st.columns(2)
        with m_row1:
            st.metric("COMPOSITE MATCH", f"{cand_row['composite_score']:.1f}%")
            st.metric("EDUCATION", cand_row['education_level'])
        with m_row2:
            st.metric("EXPERIENCE", f"{cand_row['experience_years']} Years", f"Req: {active_jd['min_experience_years']}y")
            st.metric("FIT CATEGORY", get_fit_category(cand_row['composite_score']))
        
        st.divider()
        st.markdown(
            "<div style='background-color:#F3F0F9; border:1px solid #DDD7EA; border-radius:8px; padding:12px 14px;'>"
            "<p style='color:#6B5BD6; font-weight:800; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:2px;'>OPERATIONAL RECOMMENDATION</p>"
            f"<p style='color:#25233A; font-size:0.88rem; margin:0; line-height:1.4;'>{cand_row['recommendation']}</p>"
            "</div>",
            unsafe_allow_html=True
        )

    with c_col2:
        st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.78rem; letter-spacing:0.06em; margin-bottom:4px;'>COMPETENCY MATRIX</p>", unsafe_allow_html=True)
        
        # Matched Mandatory
        st.markdown(f"**✓ MATCHED MANDATORY SKILLS ({len(cand_row['matched_required'])} / {len(active_jd['required_skills'])})**")
        if cand_row['matched_required']:
            m_chips = "".join([f"<span class='chip-req'>✓ {s}</span>" for s in cand_row['matched_required']])
            st.markdown(f"<div style='margin-bottom:8px;'>{m_chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#6E6A7B; font-size:0.85rem; font-style:italic;'>None identified</p>", unsafe_allow_html=True)
            
        # Missing Mandatory
        st.markdown(f"**✕ MISSING MANDATORY SKILLS ({len(cand_row['missing_required'])})**")
        if cand_row['missing_required']:
            miss_chips = "".join([f"<span class='chip-miss'>✕ {s}</span>" for s in cand_row['missing_required']])
            st.markdown(f"<div style='margin-bottom:8px;'>{miss_chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-bottom:8px;'><span class='saas-badge-success'>✓ All mandatory competencies satisfied.</span></div>", unsafe_allow_html=True)
            
        # Matched Preferred
        st.markdown(f"**★ MATCHED PREFERRED SKILLS ({len(cand_row['matched_preferred'])})**")
        if cand_row['matched_preferred']:
            pref_m_chips = "".join([f"<span class='chip-pref'>★ {s}</span>" for s in cand_row['matched_preferred']])
            st.markdown(f"<div style='margin-bottom:4px;'>{pref_m_chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#6E6A7B; font-size:0.85rem; font-style:italic;'>None identified</p>", unsafe_allow_html=True)

st.markdown("")

# ==================================================
# SECTION 05: SKILL GAP ANALYSIS & FULL TABLE
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>05. SKILL GAP ANALYSIS</div><div class='section-sub'>Candidate evaluation diagnostics and skill gap resolution.</div></div>", unsafe_allow_html=True)

# Four Metric Cards
gap_c1, gap_c2, gap_c3, gap_c4 = st.columns(4)
mand_pct = (len(cand_row['matched_required']) / len(active_jd['required_skills'])) * 100.0 if active_jd['required_skills'] else 100.0
pref_pct = (len(cand_row['matched_preferred']) / len(active_jd['preferred_skills'])) * 100.0 if active_jd['preferred_skills'] else 100.0

with gap_c1:
    with st.container(border=True):
        st.metric("MANDATORY SKILL COVERAGE", f"{mand_pct:.1f}%")
        st.caption(f"{len(cand_row['matched_required'])} / {len(active_jd['required_skills'])} mandatory skills")
with gap_c2:
    with st.container(border=True):
        st.metric("PREFERRED SKILL COVERAGE", f"{pref_pct:.1f}%")
        st.caption(f"{len(cand_row['matched_preferred'])} / {len(active_jd['preferred_skills'])} preferred skills")
with gap_c3:
    with st.container(border=True):
        st.metric("SEMANTIC FIT", f"{cand_row['semantic_similarity_pct']:.1f}%")
        st.caption("Dense vector alignment")
with gap_c4:
    with st.container(border=True):
        st.metric("EXPERIENCE / EDUCATION", f"{cand_row['exp_edu_fit_pct']:.1f}%")
        st.caption(f"{cand_row['experience_years']} years exp • {cand_row['education_level']}")

# Matched vs Missing Skills Cards Side-by-Side
side_col1, side_col2 = st.columns(2)
with side_col1:
    with st.container(border=True):
        st.markdown("<h5 style='color:#25233A; margin:0 0 6px 0;'>MATCHED SKILLS</h5>", unsafe_allow_html=True)
        all_matched = cand_row['matched_required'] + cand_row['matched_preferred']
        if all_matched:
            all_m_chips = "".join([f"<span class='chip-req'>✓ {s}</span>" for s in cand_row['matched_required']] + [f"<span class='chip-pref'>★ {s}</span>" for s in cand_row['matched_preferred']])
            st.markdown(f"<div>{all_m_chips}</div>", unsafe_allow_html=True)
        else:
            st.caption("No matching skills detected.")

with side_col2:
    with st.container(border=True):
        st.markdown("<h5 style='color:#25233A; margin:0 0 6px 0;'>MISSING SKILLS</h5>", unsafe_allow_html=True)
        if cand_row['missing_required']:
            all_miss_chips = "".join([f"<span class='chip-miss'>✕ {s}</span>" for s in cand_row['missing_required']])
            st.markdown(f"<div>{all_miss_chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='saas-badge-success'>✓ All mandatory competencies satisfied.</span>", unsafe_allow_html=True)

# Recommended Action Card
with st.container(border=True):
    st.markdown("<p style='color:#6B5BD6; font-weight:800; font-size:0.75rem; letter-spacing:0.06em; margin-bottom:2px;'>RECOMMENDED ACTION</p>", unsafe_allow_html=True)
    if cand_row['composite_score'] >= 75.0 and len(cand_row['missing_required']) == 0:
        action_text = f"Strong overall match. Candidate satisfies <strong>all {len(active_jd['required_skills'])} mandatory technical skills</strong> with strong domain vector alignment ({cand_row['semantic_similarity_pct']:.1f}%). Recommended for recruiter review and technical screening."
    elif cand_row['composite_score'] >= 40.0:
        missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "none"
        action_text = f"Moderate match. Candidate satisfies <strong>{len(cand_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory technical skills</strong> (missing: <strong>{missing_str}</strong>). Recommended for recruiter review and hiring manager evaluation."
    else:
        missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "core skills"
        action_text = f"High technical gap. Candidate satisfies only <strong>{len(cand_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory technical skills</strong> (missing: <strong>{missing_str}</strong>). Profile indicates primary background in adjacent domains."
    st.markdown(f"<p style='color:#25233A; font-size:0.90rem; margin:0;'>{action_text}</p>", unsafe_allow_html=True)

st.markdown("")

# Clean Light Leaderboard Table (100% Light Enterprise Styling, Zero Indent for Clean HTML Parsing, Full Responsive Wrap)
with st.container(border=True):
    st.markdown("<h5 style='color:#25233A; margin:0 0 10px 0;'>Candidate Leaderboard Summary</h5>", unsafe_allow_html=True)
    
    table_rows_html = "".join([
        f"<tr>"
        f"<td style='font-weight:700; color:#6B5BD6;'>#{r['Rank']}</td>"
        f"<td style='white-space:normal; overflow-wrap:anywhere; word-break:break-word;'><span class='saas-candidate-chip'>{r['candidate_id']}</span></td>"
        f"<td><strong>{r['composite_score']:.1f}%</strong></td>"
        f"<td><span class='{get_fit_badge_class(r['composite_score'])}'>{r['fit_category']}</span></td>"
        f"<td>{len(r['matched_required'])} / {len(active_jd['required_skills'])}</td>"
        f"<td>{r['semantic_similarity_pct']:.1f}%</td>"
        f"<td>{r['experience_years']} yrs</td>"
        f"</tr>"
        for _, r in rankings_df.iterrows()
    ])
    
    table_html = (
        "<div class='saas-table-container'>"
        "<table class='saas-table'>"
        "<thead><tr>"
        "<th>Rank</th>"
        "<th>Candidate</th>"
        "<th>Composite Match</th>"
        "<th>Fit Category</th>"
        "<th>Mandatory Skills</th>"
        "<th>Semantic Fit</th>"
        "<th>Experience</th>"
        "</tr></thead>"
        f"<tbody>{table_rows_html}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown("")

# ==================================================
# SECTION 06: SCORING ARCHITECTURE (4 REAL CARDS)
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>06. SCORING ARCHITECTURE</div><div class='section-sub'>Transparent multi-factor candidate evaluation heuristic.</div></div>", unsafe_allow_html=True)

arch_c1, arch_c2, arch_c3, arch_c4 = st.columns(4)

with arch_c1:
    with st.container(border=True):
        st.markdown("<h2 style='color:#6B5BD6; font-weight:800; margin:0 0 4px 0;'>40%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-weight:800; font-size:0.90rem; margin-bottom:4px;'>TECHNICAL SKILL OVERLAP</p>", unsafe_allow_html=True)
        st.caption("Mandatory skills = 80% of this component. Preferred skills = 20% of this component.")

with arch_c2:
    with st.container(border=True):
        st.markdown("<h2 style='color:#6B5BD6; font-weight:800; margin:0 0 4px 0;'>30%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-weight:800; font-size:0.90rem; margin-bottom:4px;'>DENSE SEMANTIC FIT</p>", unsafe_allow_html=True)
        st.caption("Contextual vector similarity using SentenceTransformer (all-MiniLM-L6-v2).")

with arch_c3:
    with st.container(border=True):
        st.markdown("<h2 style='color:#6B5BD6; font-weight:800; margin:0 0 4px 0;'>20%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-weight:800; font-size:0.90rem; margin-bottom:4px;'>TF-IDF KEYWORD MATCH</p>", unsafe_allow_html=True)
        st.caption("Sublinear term-frequency inverse document frequency cosine similarity.")

with arch_c4:
    with st.container(border=True):
        st.markdown("<h2 style='color:#6B5BD6; font-weight:800; margin:0 0 4px 0;'>10%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-weight:800; font-size:0.90rem; margin-bottom:4px;'>EXPERIENCE & EDUCATION</p>", unsafe_allow_html=True)
        st.caption("Experience tenure ratio against job requirements with informational degree heuristic.")

st.caption("📌 *Note: Composite Match Score is a transparent heuristic ranking score for decision support; it is not a hiring probability or calibrated confidence score. Weights are prototype design choices and have not been statistically calibrated against hiring outcomes.*")

st.markdown("")

# ==================================================
# SECTION 07: PRIVACY & RESPONSIBLE USE (3 REAL CARDS)
# ==================================================
st.markdown("<div class='section-header'><div class='section-title'>07. PRIVACY & RESPONSIBLE USE</div><div class='section-sub'>Transparency, privacy protection and responsible decision support.</div></div>", unsafe_allow_html=True)

gov_col1, gov_col2, gov_col3 = st.columns(3)

with gov_col1:
    with st.container(border=True):
        st.markdown("<h4 style='color:#25233A; margin:0 0 4px 0;'>📊 DATA PROVENANCE</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B5BD6; font-weight:700; font-size:0.85rem; margin-bottom:6px;'>Controlled synthetic candidate corpus</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-size:0.85rem; margin-bottom:4px;'><strong>8 multi-format profiles</strong> (PDF • DOCX • TXT) across 3 synthetic job descriptions.</p>", unsafe_allow_html=True)
        st.caption("These profiles are synthetic benchmark data created for prototype evaluation and are not representative of a production hiring population.")

with gov_col2:
    with st.container(border=True):
        st.markdown("<h4 style='color:#25233A; margin:0 0 4px 0;'>🛡️ PRIVACY & RESPONSIBLE USE</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B5BD6; font-weight:700; font-size:0.85rem; margin-bottom:6px;'>PII reduction & anonymization</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-size:0.85rem; margin-bottom:4px;'>Candidate names, email addresses, phone numbers and profile URLs are masked before feature extraction and scoring.</p>", unsafe_allow_html=True)
        st.caption("Protected demographic attributes are excluded from ranking. Fairness performance cannot be validated from this small controlled synthetic corpus; real-world deployment would require independent fairness evaluation, monitoring and governance.")

with gov_col3:
    with st.container(border=True):
        st.markdown("<h4 style='color:#25233A; margin:0 0 4px 0;'>⚖️ DECISION SUPPORT</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B5BD6; font-weight:700; font-size:0.85rem; margin-bottom:6px;'>Human-in-the-loop requirement</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#25233A; font-size:0.85rem; margin-bottom:4px;'>TalentMatch ML is a recruiter decision-support prototype, not an autonomous hiring engine.</p>", unsafe_allow_html=True)
        st.caption("Scores represent prototype heuristic alignment and must always be validated by recruiters and hiring managers.")
