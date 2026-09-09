"""
TalentMatch ML - Recruiter Intelligence & Candidate Screening Workspace
Interactive Streamlit decision-support platform for objective, skill-based candidate screening,
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
    page_title="TalentMatch ML — Recruiter Intelligence Workspace",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Light Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1F2933;
    }
    
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.8rem !important;
        max-width: 1440px !important;
    }

    .stApp {
        background-color: #F8F7F2 !important;
        color: #1F2933 !important;
    }
    
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

    /* Native Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D8E0D8 !important;
        border-radius: 14px !important;
        box-shadow: 0 3px 12px rgba(31, 41, 51, 0.04), 0 1px 3px rgba(31, 41, 51, 0.02) !important;
        padding: 16px !important;
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #075E5B !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #53636A !important;
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
""", unsafe_allow_html=True)

# Data & Model Setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resumes_dir = os.path.join(base_dir, "data", "resumes")
jds_dir = os.path.join(base_dir, "data", "job_descriptions")

@st.cache_resource
def load_matcher():
    return TalentMatcher(use_dense_embeddings=True)

matcher = load_matcher()

# 1. Hero & Header Area
hero_col1, hero_col2 = st.columns([3, 1])
with hero_col1:
    st.markdown("## 🎯 **TALENTMATCH ML** — Recruiter Intelligence Workspace")
    st.markdown("##### *Resume Intelligence & Candidate Decision-Support Platform*")
    st.caption("`SCREENING ENGINE • READY` | `PII ANONYMIZED` | `4-TIER HYBRID SCORING` | `TRACK: FIT/AUG26/ML10465`")

with hero_col2:
    st.info("💡 **Decision-Support Prototype**\n\nObjective technical candidate screening and skill-gap diagnostics.")

# Processing Pipeline Visual Banner
with st.container(border=True):
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    p_col1.markdown("**1. Ingestion**\n\n`.pdf` `.docx` `.txt`")
    p_col2.markdown("**2. Privacy**\n\n`PII Filter`")
    p_col3.markdown("**3. Skills**\n\n`Taxonomy (200+)`")
    p_col4.markdown("**4. Matching**\n\n`4-Tier Hybrid`")
    p_col5.markdown("**5. Ranking**\n\n`Composite Score`")
    p_col6.markdown("**6. Decision**\n\n`Skill-Gap Analysis`")

# Sidebar Configuration
st.sidebar.markdown("### 🎯 Target Role Configuration")

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
st.sidebar.markdown(" ".join([f"`{s}`" for s in active_jd['required_skills']]))

st.sidebar.markdown(f"**Preferred Competencies ({len(active_jd['preferred_skills'])}):**")
st.sidebar.markdown(" ".join([f"`{s}`" for s in active_jd['preferred_skills']]))

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
st.markdown(f"### 01. Screening Command Center — *{active_jd['title']}*")

total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 75.0])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 40.0) & (rankings_df["composite_score"] < 75.0)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 40.0])

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.metric("Total Resumes Screened", f"{total_cands} Profiles")
        st.caption("Formats: **PDF, DOCX, TXT**")

with col2:
    with st.container(border=True):
        st.metric("Strong Technical Fits", f"{strong_matches} Candidates")
        st.caption("Action: **Immediate Technical Screen**")

with col3:
    with st.container(border=True):
        st.metric("Review / Moderate Fits", f"{moderate_matches} Candidates")
        st.caption("Action: **Assess Specific Competency Gaps**")

with col4:
    with st.container(border=True):
        st.metric("High Technical Gaps", f"{high_gaps} Candidates")
        st.caption("Action: **Core Competency Mismatch**")

# Section 02: Candidate Leaderboard & Rankings
st.markdown("### 02. Candidate Ranking Leaderboard")

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
    display_df = rankings_df[[
        "Rank", "candidate_id", "composite_score", "gap_severity", 
        "skill_score_pct", "semantic_similarity_pct", "experience_years", "education_level"
    ]].copy()
    display_df.columns = [
        "Rank", "Candidate Profile", "Composite Match (%)", "Gap Severity", 
        "Skill Overlap (%)", "Semantic Fit (%)", "Experience (Yrs)", "Education"
    ]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
            "Composite Match (%)": st.column_config.ProgressColumn("Composite Match", format="%.1f%%", min_value=0, max_value=100),
            "Skill Overlap (%)": st.column_config.NumberColumn("Skill Overlap", format="%.1f%%"),
            "Semantic Fit (%)": st.column_config.NumberColumn("Semantic Fit", format="%.1f%%"),
            "Experience (Yrs)": st.column_config.NumberColumn("Experience", format="%.1f yrs")
        }
    )

# Why This Ranking Insight Box
top_row = rankings_df.iloc[0]
with st.container(border=True):
    st.markdown(f"""
    **💡 Recruiter Decision Context: Why This Ranking?**
    
    Candidates are evaluated objectively via the 4-tier hybrid scoring architecture (**40% Hard Skills**, **30% Dense Semantic Similarity**, **20% TF-IDF Keyword Match**, **10% Experience & Education**).
    
    Top-ranked profile `{top_row['candidate_id']}` achieved the highest composite score (**{top_row['composite_score']:.1f}%**) by satisfying **{len(top_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory technical skills** and demonstrating strong contextual domain alignment (**{top_row['semantic_similarity_pct']:.1f}%** semantic fit).
    """)

# Section 03: Candidate Deep-Dive & Skill Gap Diagnostics
st.markdown("### 03. Candidate Deep-Dive & Profile Analysis Workspace")

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

c_col1, c_col2 = st.columns([1, 1])

with c_col1:
    with st.container(border=True):
        st.markdown(f"#### Profile: `{cand_row['candidate_id']}`")
        st.caption(f"Parsed Format: **.{cand_file_ext}** | PII Status: **Masked & Anonymized** | Rank: **#{cand_row['Rank']} of {total_cands}**")
        
        m_row1, m_row2 = st.columns(2)
        with m_row1:
            st.metric("Composite Match", f"{cand_row['composite_score']:.1f}%", f"{cand_row['gap_severity']} Gap")
            st.metric("Highest Education", cand_row['education_level'])
        with m_row2:
            st.metric("Stated Experience", f"{cand_row['experience_years']} Years", f"Req: {active_jd['min_experience_years']}y")
            st.metric("Gap Severity", cand_row['gap_severity'])
        
        st.markdown("---")
        st.markdown("##### 4-Tier Evaluation Pillar Breakdown:")
        
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"**Tier 1 • Hard Skills (40%):** `{cand_row['skill_score_pct']:.1f}%`")
            st.markdown(f"**Tier 3 • TF-IDF Match (20%):** `{cand_row['lexical_similarity_pct']:.1f}%`")
        with p2:
            st.markdown(f"**Tier 2 • Dense Semantic (30%):** `{cand_row['semantic_similarity_pct']:.1f}%`")
            st.markdown(f"**Tier 4 • Exp/Edu Fit (10%):** `{cand_row['exp_edu_fit_pct']:.1f}%`")
        
        st.markdown("---")
        if cand_row['composite_score'] >= 75.0:
            st.success(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")
        elif cand_row['composite_score'] >= 40.0:
            st.warning(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")
        else:
            st.error(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")

with c_col2:
    with st.container(border=True):
        st.markdown("#### Granular Competency Matrix & Gap Breakdown")
        
        # Matched Mandatory
        st.markdown(f"**✅ Matched Mandatory Skills ({len(cand_row['matched_required'])} / {len(active_jd['required_skills'])}):**")
        if cand_row['matched_required']:
            st.markdown(" ".join([f"`✓ {s}`" for s in cand_row['matched_required']]))
        else:
            st.caption("None identified")
            
        st.markdown("")
        
        # Missing Mandatory
        st.markdown(f"**❌ Missing Mandatory Skills ({len(cand_row['missing_required'])}):**")
        if cand_row['missing_required']:
            st.markdown(" ".join([f"`✗ {s}`" for s in cand_row['missing_required']]))
        else:
            st.success("🎉 All mandatory competencies satisfied!")
            
        st.markdown("")
        
        # Matched Preferred
        st.markdown(f"**⭐ Matched Preferred Bonus Skills ({len(cand_row['matched_preferred'])}):**")
        if cand_row['matched_preferred']:
            st.markdown(" ".join([f"`★ {s}`" for s in cand_row['matched_preferred']]))
        else:
            st.caption("None identified")
            
        st.markdown("---")
        
        # Recruiter Insight
        if cand_row['composite_score'] >= 75.0 and len(cand_row['missing_required']) == 0:
            insight_text = f"Candidate satisfies **all {len(active_jd['required_skills'])} mandatory technical skills** with strong domain vector alignment ({cand_row['semantic_similarity_pct']:.1f}%). Recommended to advance immediately to technical screen."
            st.success(f"**🧠 Recruiter Intelligence Insight:**\n\n{insight_text}")
        elif cand_row['composite_score'] >= 40.0:
            missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "none"
            insight_text = f"Candidate displays partial technical alignment ({len(cand_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory skills). Primary gaps to assess: **{missing_str}**. Suitable for hiring manager review or secondary role routing."
            st.warning(f"**🧠 Recruiter Intelligence Insight:**\n\n{insight_text}")
        else:
            missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "core skills"
            insight_text = f"Candidate demonstrates low alignment for this specific role (missing: **{missing_str}**). Profile indicates primary background in adjacent software domains."
            st.error(f"**🧠 Recruiter Intelligence Insight:**\n\n{insight_text}")

# Section 04: Model Governance & Data Provenance
st.markdown("### 04. Model Governance & Data Provenance")

gov_col1, gov_col2, gov_col3 = st.columns(3)

with gov_col1:
    with st.container(border=True):
        st.markdown("##### 📊 Controlled Benchmark Scope")
        st.caption(
            "Controlled, synthetic candidate resume dataset (8 profiles across PDF, DOCX, and TXT formats) "
            "curated to validate multi-format parsing, PII anonymization, 4-tier hybrid scoring, and "
            "skill-gap extraction in an objective prototype environment."
        )

with gov_col2:
    with st.container(border=True):
        st.markdown("##### 🛡️ Zero-PII Fairness Safeguards")
        st.caption(
            "Candidate names, email addresses, phone numbers, and profile URLs are automatically scrubbed "
            "from resume text prior to feature extraction to mitigate demographic, gender, and unconscious "
            "recruiter bias."
        )

with gov_col3:
    with st.container(border=True):
        st.markdown("##### ⚖️ Decision-Support Notice")
        st.caption(
            "TalentMatch ML is explicitly engineered as a recruiter decision-support tool, not an autonomous "
            "hiring engine. Real-world deployment requires production validation, and human evaluation remains "
            "mandatory for all hiring decisions."
        )
