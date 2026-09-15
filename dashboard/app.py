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

# Sophisticated Lavender / Purple Enterprise SaaS Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #252334;
    }
    
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.8rem !important;
        max-width: 1440px !important;
    }

    /* Primary Background: Very Light Lavender / Cool Off-White */
    .stApp {
        background-color: #F7F5FC !important;
        color: #252334 !important;
    }
    
    /* Sidebar: Deeper Lavender */
    section[data-testid="stSidebar"] {
        background-color: #F0ECF8 !important;
        border-right: 1px solid #E5E0EF !important;
        min-width: 340px !important;
        max-width: 390px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #252334 !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
        margin-bottom: 8px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: #6F6A7D !important;
    }

    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #252334 !important;
    }

    section[data-testid="stSidebar"] label p {
        color: #252334 !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 14px 0 !important;
        border-color: #E5E0EF !important;
    }

    /* Modern Professional Tag Styling for Code Elements */
    code {
        background-color: #F0ECF8 !important;
        color: #6D5BD0 !important;
        border: 1px solid #E5E0EF !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    }

    section[data-testid="stSidebar"] code {
        background-color: #FFFFFF !important;
        color: #6D5BD0 !important;
        border: 1px solid #E5E0EF !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    }

    /* Native Card Containers: White with Lavender-tinted shadow */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E0EF !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(109, 91, 208, 0.05), 0 1px 3px rgba(37, 35, 52, 0.03) !important;
        padding: 18px !important;
    }

    /* Metric Labels - Strong Charcoal Contrast */
    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #252334 !important;
    }

    /* Metric Values - Large, Clear, Primary Purple */
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #6D5BD0 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
    }

    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        font-size: 1.35rem !important;
    }

    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E0EF !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #252334 !important;
        background-color: #FFFFFF !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
        background-color: #FFFFFF !important;
        color: #252334 !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #F0ECF8 !important;
        color: #6D5BD0 !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #252334 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: #6F6A7D !important;
        padding: 8px 16px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6D5BD0 !important;
        border-bottom-color: #6D5BD0 !important;
    }

    /* Primary Purple Export Button */
    div.stDownloadButton > button {
        background-color: #6D5BD0 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid #5B48B8 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 6px rgba(109, 91, 208, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #5B48B8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(109, 91, 208, 0.35) !important;
    }

    div.stDownloadButton > button p {
        color: #FFFFFF !important;
    }

    /* Alert / Callout overrides */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 1px !important;
    }

    /* Progress bar color */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #6D5BD0 !important;
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
total_cands = len(rankings_df)
strong_matches = len(rankings_df[rankings_df["composite_score"] >= 75.0])
moderate_matches = len(rankings_df[(rankings_df["composite_score"] >= 40.0) & (rankings_df["composite_score"] < 75.0)])
high_gaps = len(rankings_df[rankings_df["composite_score"] < 40.0])

st.sidebar.markdown("---")
st.sidebar.markdown("#### ROLE REQUIREMENTS")
st.sidebar.markdown(f"**Role ID:** `{active_jd.get('job_id', 'N/A')}`")
st.sidebar.markdown(f"**Experience:** `{active_jd['min_experience_years']} Years`")
st.sidebar.markdown(f"**Mandatory:** `{len(active_jd['required_skills'])} Skills`")
st.sidebar.markdown(" ".join([f"`✓ {s}`" for s in active_jd['required_skills']]))

st.sidebar.markdown(f"**Preferred:** `{len(active_jd['preferred_skills'])} Skills`")
st.sidebar.markdown(" ".join([f"`★ {s}`" for s in active_jd['preferred_skills']]))

st.sidebar.markdown("---")
st.sidebar.markdown("#### SCORING MODEL")
st.sidebar.markdown("""
- **40%** Hard Skill Overlap
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

# 1. TOP HEADER
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("## 🎯 **TalentMatch ML** — Recruiter Intelligence Platform")
    st.markdown("##### *Resume intelligence, candidate ranking & skill-gap diagnostics*")
    st.caption("● Screening Engine Ready | ✓ PII Anonymized | 4-Tier Hybrid Scoring | Decision Support Only")

with head_col2:
    st.info("💡 **Decision Support Only**\n\nObjective candidate screening & skill-gap diagnostics.")

st.markdown("")

# ==================================================
# SECTION 1: TARGET ROLE OVERVIEW
# ==================================================
with st.container(border=True):
    r_col1, r_col2 = st.columns([2, 1])
    with r_col1:
        st.markdown(f"### {active_jd['title']}")
        st.markdown(f"**Role ID:** `{active_jd.get('job_id', 'N/A')}` &nbsp;|&nbsp; **Required Experience:** `{active_jd['min_experience_years']}+ Years`")
        st.markdown("**Mandatory Technical Competencies:**")
        st.markdown(" ".join([f"`✓ {s}`" for s in active_jd['required_skills']]))
    with r_col2:
        st.markdown("**Preferred Competencies:**")
        st.markdown(" ".join([f"`★ {s}`" for s in active_jd['preferred_skills']]))
        st.caption(f"Domain Focus: **{active_jd.get('domain', 'Engineering & Technology')}**")

st.markdown("")

# ==================================================
# SECTION 2: SCREENING OVERVIEW (4 KPI CARDS)
# ==================================================
st.markdown("### 01. Screening Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    with st.container(border=True):
        st.metric("Total Resumes", f"{total_cands}")
        st.caption("Formats: **PDF, DOCX, TXT**")

with kpi2:
    with st.container(border=True):
        st.metric("Strong Technical Fit", f"{strong_matches}")
        st.caption("Score ≥75% • Ready for Screen")

with kpi3:
    with st.container(border=True):
        st.metric("Review / Moderate Fit", f"{moderate_matches}")
        st.caption("Score 40-74% • Assess Gaps")

with kpi4:
    with st.container(border=True):
        st.metric("High Technical Gap", f"{high_gaps}")
        st.caption("Score <40% • Core Mismatch")

st.markdown("")

# ==================================================
# SECTION 3: CANDIDATE RANKING (Chart + Top Candidate)
# ==================================================
st.markdown("### 02. Candidate Ranking")
st.caption("Technical compatibility across mandatory skills, dense semantic relevance, keyword overlap, and experience.")

rank_col1, rank_col2 = st.columns([3, 2])

top_row = rankings_df.iloc[0]

with rank_col1:
    sorted_chart_df = rankings_df.sort_values(by="composite_score", ascending=True)
    
    bar_colors = []
    for s in sorted_chart_df["composite_score"]:
        if s >= 75.0:
            bar_colors.append("#4F8A70")  # Success Green
        elif s >= 40.0:
            bar_colors.append("#C58A28")  # Warm Amber
        else:
            bar_colors.append("#C85A54")  # Coral
            
    fig = go.Figure(go.Bar(
        x=sorted_chart_df["composite_score"],
        y=[f"#{r} {c}" for r, c in zip(sorted_chart_df["Rank"], sorted_chart_df["candidate_id"])],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="#E5E0EF", width=1)
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
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#6F6A7D"),
        xaxis=dict(
            range=[0, 108],
            ticksuffix="%",
            gridcolor="#F0ECF8",
            zerolinecolor="#E5E0EF",
            tickfont=dict(size=11, color="#6F6A7D")
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#252334", weight=600)
        ),
        height=340
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with rank_col2:
    with st.container(border=True):
        st.markdown("#### 🥇 Top Ranked Candidate")
        st.markdown(f"### `{top_row['candidate_id']}`")
        
        tc_m1, tc_m2 = st.columns(2)
        with tc_m1:
            st.metric("Composite Match", f"{top_row['composite_score']:.1f}%")
            st.metric("Stated Experience", f"{top_row['experience_years']} Years")
        with tc_m2:
            st.metric("Mandatory Skills", f"{len(top_row['matched_required'])} / {len(active_jd['required_skills'])}")
            st.metric("Gap Severity", f"{top_row['gap_severity']} Gap")
            
        st.divider()
        st.success(f"**Recommendation:** {top_row['recommendation']}")

st.markdown("")

# ==================================================
# SECTION 4: RECRUITER INSIGHT
# ==================================================
with st.container(border=True):
    st.markdown("### 💡 Recruiter Insight: Why This Candidate Ranks First")
    
    if len(top_row['missing_required']) == 0:
        insight_summary = f"Candidate **#{top_row['Rank']} (`{top_row['candidate_id']}`)** satisfies **all {len(active_jd['required_skills'])} mandatory technical competencies** and demonstrates the strongest contextual semantic alignment (**{top_row['semantic_similarity_pct']:.1f}%**) for the selected role."
    else:
        insight_summary = f"Candidate **#{top_row['Rank']} (`{top_row['candidate_id']}`)** satisfies **{len(top_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory technical competencies** with an overall composite score of **{top_row['composite_score']:.1f}%**."
        
    st.markdown(insight_summary)
    st.markdown("")
    
    ins_c1, ins_c2, ins_c3, ins_c4 = st.columns(4)
    with ins_c1:
        st.metric("Hard Skills (40%)", f"{top_row['skill_score_pct']:.1f}%")
        st.progress(min(1.0, top_row['skill_score_pct'] / 100.0))
    with ins_c2:
        st.metric("Semantic Fit (30%)", f"{top_row['semantic_similarity_pct']:.1f}%")
        st.progress(min(1.0, top_row['semantic_similarity_pct'] / 100.0))
    with ins_c3:
        st.metric("TF-IDF Match (20%)", f"{top_row['lexical_similarity_pct']:.1f}%")
        st.progress(min(1.0, top_row['lexical_similarity_pct'] / 100.0))
    with ins_c4:
        st.metric("Experience/Edu (10%)", f"{top_row['exp_edu_fit_pct']:.1f}%")
        st.progress(min(1.0, top_row['exp_edu_fit_pct'] / 100.0))

st.markdown("")

# ==================================================
# SECTION 5: CANDIDATE PROFILE DEEP-DIVE
# ==================================================
st.markdown("### 03. Candidate Profile Deep-Dive")

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
        st.caption("CANDIDATE PROFILE")
        st.subheader(cand_row['candidate_id'])
        st.markdown(f"**Rank:** `#{cand_row['Rank']} of {total_cands}` &nbsp;|&nbsp; **Gap:** `{cand_row['gap_severity']}` &nbsp;|&nbsp; **PII:** `Masked` &nbsp;|&nbsp; **Format:** `.{cand_file_ext}`")
        st.divider()
        
        m_row1, m_row2 = st.columns(2)
        with m_row1:
            st.metric("Composite Match", f"{cand_row['composite_score']:.1f}%", f"{cand_row['gap_severity']} Gap")
            st.metric("Highest Education", cand_row['education_level'])
        with m_row2:
            st.metric("Stated Experience", f"{cand_row['experience_years']} Years", f"Req: {active_jd['min_experience_years']}y")
            st.metric("Gap Severity", cand_row['gap_severity'])
        
        st.divider()
        if cand_row['composite_score'] >= 75.0:
            st.success(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")
        elif cand_row['composite_score'] >= 40.0:
            st.warning(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")
        else:
            st.error(f"**Operational Recommendation:**\n\n{cand_row['recommendation']}")

with c_col2:
    with st.container(border=True):
        st.markdown("#### Competency Matrix")
        
        # Matched Mandatory
        st.markdown(f"**✓ Matched Mandatory Skills ({len(cand_row['matched_required'])} / {len(active_jd['required_skills'])}):**")
        if cand_row['matched_required']:
            st.markdown(" ".join([f"`✓ {s}`" for s in cand_row['matched_required']]))
        else:
            st.caption("None identified")
            
        st.markdown("")
        
        # Missing Mandatory
        st.markdown(f"**✕ Missing Mandatory Skills ({len(cand_row['missing_required'])}):**")
        if cand_row['missing_required']:
            st.markdown(" ".join([f"`✕ {s}`" for s in cand_row['missing_required']]))
        else:
            st.success("✓ All mandatory competencies satisfied!")
            
        st.markdown("")
        
        # Matched Preferred
        st.markdown(f"**★ Matched Preferred Skills ({len(cand_row['matched_preferred'])}):**")
        if cand_row['matched_preferred']:
            st.markdown(" ".join([f"`★ {s}`" for s in cand_row['matched_preferred']]))
        else:
            st.caption("None identified")
            
        st.divider()
        
        # Recruiter Insight
        if cand_row['composite_score'] >= 75.0 and len(cand_row['missing_required']) == 0:
            insight_text = f"Candidate satisfies **all {len(active_jd['required_skills'])} mandatory technical skills** with strong domain vector alignment ({cand_row['semantic_similarity_pct']:.1f}%). Recommended to advance immediately to technical screen."
            st.success(f"**Recruiter Intelligence Insight:**\n\n{insight_text}")
        elif cand_row['composite_score'] >= 40.0:
            missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "none"
            insight_text = f"Candidate displays partial technical alignment ({len(cand_row['matched_required'])}/{len(active_jd['required_skills'])} mandatory skills). Primary gaps to assess: **{missing_str}**. Suitable for hiring manager review or secondary role routing."
            st.warning(f"**Recruiter Intelligence Insight:**\n\n{insight_text}")
        else:
            missing_str = ", ".join(cand_row['missing_required']) if cand_row['missing_required'] else "core skills"
            insight_text = f"Candidate demonstrates low alignment for this specific role (missing: **{missing_str}**). Profile indicates primary background in adjacent software domains."
            st.error(f"**Recruiter Intelligence Insight:**\n\n{insight_text}")

st.markdown("")

# ==================================================
# SECTION 6: SKILL GAP ANALYSIS & FULL TABLE
# ==================================================
st.markdown("### 04. Skill Gap Analysis & Evaluation Breakdown")

tab_pillars, tab_table = st.tabs(["📊 Evaluation Pillar Breakdown", "📋 Full Candidate Leaderboard Data"])

with tab_pillars:
    with st.container(border=True):
        st.markdown(f"##### Scoring Breakdown for `{cand_row['candidate_id']}`")
        
        pil_c1, pil_c2, pil_c3, pil_c4 = st.columns(4)
        with pil_c1:
            st.metric("Technical Fit (40%)", f"{cand_row['skill_score_pct']:.1f}%")
            st.progress(min(1.0, cand_row['skill_score_pct'] / 100.0))
            st.caption(f"Matched: **{len(cand_row['matched_required'])}/{len(active_jd['required_skills'])}** req skills")
        with pil_c2:
            st.metric("Semantic Fit (30%)", f"{cand_row['semantic_similarity_pct']:.1f}%")
            st.progress(min(1.0, cand_row['semantic_similarity_pct'] / 100.0))
            st.caption("all-MiniLM-L6-v2 vector match")
        with pil_c3:
            st.metric("Keyword Fit (20%)", f"{cand_row['lexical_similarity_pct']:.1f}%")
            st.progress(min(1.0, cand_row['lexical_similarity_pct'] / 100.0))
            st.caption("Sublinear TF-IDF similarity")
        with pil_c4:
            st.metric("Experience/Edu (10%)", f"{cand_row['exp_edu_fit_pct']:.1f}%")
            st.progress(min(1.0, cand_row['exp_edu_fit_pct'] / 100.0))
            st.caption(f"{cand_row['experience_years']}y exp • {cand_row['education_level']}")

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

st.markdown("")

# ==================================================
# SECTION 7: SCORING ARCHITECTURE
# ==================================================
st.markdown("### 05. Scoring Architecture")
st.caption("Transparent, multi-tier scoring methodology designed for recruiter interpretability.")

arch_c1, arch_c2, arch_c3, arch_c4 = st.columns(4)

with arch_c1:
    with st.container(border=True):
        st.markdown("#### 40%")
        st.markdown("**Hard Skill Overlap**")
        st.caption("Direct matching against mandatory (80%) and preferred (20%) technical competencies using a 200+ skill taxonomy with alias resolution.")

with arch_c2:
    with st.container(border=True):
        st.markdown("#### 30%")
        st.markdown("**Dense Semantic Fit**")
        st.caption("Contextual vector cosine similarity via `SentenceTransformer('all-MiniLM-L6-v2')` to evaluate domain experience depth.")

with arch_c3:
    with st.container(border=True):
        st.markdown("#### 20%")
        st.markdown("**TF-IDF Keyword Match**")
        st.caption("Sublinear term-frequency inverse-document frequency cosine similarity for exact technical keyword alignment.")

with arch_c4:
    with st.container(border=True):
        st.markdown("#### 10%")
        st.markdown("**Experience & Education**")
        st.caption("Stated professional experience ratio against job requirements combined with educational degree attainment.")

st.markdown("")

# ==================================================
# SECTION 8: GOVERNANCE & DATA PROVENANCE
# ==================================================
st.markdown("### 06. Governance & Data Provenance")

gov_col1, gov_col2, gov_col3 = st.columns(3)

with gov_col1:
    with st.container(border=True):
        st.markdown("##### 📊 Data Provenance")
        st.caption(
            "Controlled, synthetic candidate resume dataset (8 profiles across PDF, DOCX, and TXT formats) "
            "curated to validate multi-format parsing, PII anonymization, 4-tier hybrid scoring, and "
            "skill-gap extraction in an objective prototype environment."
        )

with gov_col2:
    with st.container(border=True):
        st.markdown("##### 🛡️ Privacy & Fairness")
        st.caption(
            "Candidate names, email addresses, phone numbers, and profile URLs are automatically scrubbed "
            "from resume text prior to feature extraction to mitigate demographic, gender, and unconscious "
            "recruiter bias."
        )

with gov_col3:
    with st.container(border=True):
        st.markdown("##### ⚖️ Decision Support")
        st.caption(
            "TalentMatch ML is explicitly engineered as a recruiter decision-support tool, not an autonomous "
            "hiring engine. Real-world deployment requires production validation, and human evaluation remains "
            "mandatory for all hiring decisions."
        )
