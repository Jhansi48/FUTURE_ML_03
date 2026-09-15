"""
TalentMatch ML - Visualization & Diagnostic Reporting Module
Generates visual leaderboards, skill gap heatmaps, and score component breakdowns.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def plot_candidate_leaderboard(rankings_df: pd.DataFrame, jd_title: str, output_path: str):
    """Generates candidate ranking leaderboard with color-coded qualification status."""
    plt.figure(figsize=(12, 6))
    
    df_plot = rankings_df.sort_values(by="composite_score", ascending=True)
    colors = ["#4C9073" if s >= 75.0 else ("#D39A2B" if s >= 40.0 else "#D45B61") for s in df_plot["composite_score"]]
    
    bars = plt.barh(df_plot["candidate_id"], df_plot["composite_score"], color=colors, edgecolor="#DDD7EA", height=0.6)
    plt.axvline(75.0, color="#4C9073", linestyle="--", alpha=0.8, label="Strong Technical Fit (≥75%)")
    plt.axvline(40.0, color="#D39A2B", linestyle="--", alpha=0.8, label="Moderate Fit Threshold (40%)")
    
    for bar, score in zip(bars, df_plot["composite_score"]):
        plt.text(score + 1.0, bar.get_y() + bar.get_height()/2, f"{score:.1f}%", va="center", fontweight="bold", fontsize=10)
        
    plt.title(f"TalentMatch ML Candidate Ranking Leaderboard\nTarget Role: {jd_title}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Composite Match Score (%)", fontsize=11)
    plt.xlim(0, 105)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_score_component_breakdown(rankings_df: pd.DataFrame, output_path: str):
    """Stacked horizontal bar chart showing individual scoring factor contributions."""
    plt.figure(figsize=(12, 6))
    
    df_plot = rankings_df.sort_values(by="composite_score", ascending=True)
    
    # Calculate weighted contributions
    w_skill = df_plot["skill_score_pct"] * 0.40
    w_sem = df_plot["semantic_similarity_pct"] * 0.30
    w_lex = df_plot["lexical_similarity_pct"] * 0.20
    w_exp = (df_plot["composite_score"] - w_skill - w_sem - w_lex).clip(lower=0)
    
    y = np.arange(len(df_plot))
    height = 0.55
    
    plt.barh(y, w_skill, height, label="Skill Overlap (40%)", color="#1f77b4")
    plt.barh(y, w_sem, height, left=w_skill, label="Dense Semantic Sim (30%)", color="#2ca02c")
    plt.barh(y, w_lex, height, left=w_skill + w_sem, label="TF-IDF Lexical Sim (20%)", color="#ff7f0e")
    plt.barh(y, w_exp, height, left=w_skill + w_sem + w_lex, label="Experience Fit (10%)", color="#9467bd")
    
    plt.yticks(y, df_plot["candidate_id"])
    plt.title("Candidate Match Score Decomposition by Evaluation Pillar", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Weighted Score Contribution (Points)", fontsize=11)
    plt.xlim(0, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_skill_gap_matrix(candidate_profiles: list, required_skills: list, output_path: str):
    """Binary presence matrix for required skills across candidates."""
    from src.taxonomy import normalize_skill
    
    req_norm = [normalize_skill(s) for s in required_skills]
    cand_ids = [c["candidate_id"] for c in candidate_profiles]
    
    matrix = np.zeros((len(cand_ids), len(req_norm)))
    
    for i, c in enumerate(candidate_profiles):
        cand_skills = {normalize_skill(s) for s in c.get("matched_skills", [])}
        for j, s in enumerate(req_norm):
            if s in cand_skills:
                matrix[i, j] = 1.0
                
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=["#ffcccc", "#c7e9c0"],
        cbar=False,
        xticklabels=req_norm,
        yticklabels=cand_ids,
        linewidths=1.0,
        linecolor="#dddddd"
    )
    plt.title("Mandatory Technical Competency Coverage Matrix (1 = Matched, 0 = Gap)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mandatory Role Skill Requirements", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")
