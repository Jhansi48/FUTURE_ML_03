"""
TalentMatch ML - Multi-Tier Candidate Matching & Skill Gap Analysis Engine
Combines structured taxonomy extraction, TF-IDF lexical cosine similarity,
dense semantic vector embeddings, and experience/education alignment for candidate ranking.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.taxonomy import get_all_canonical_skills, normalize_skill, SKILL_ALIASES

class TalentMatcher:
    def __init__(self, use_dense_embeddings: bool = True):
        self.canonical_skills = get_all_canonical_skills()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.use_dense_embeddings = use_dense_embeddings
        self.embedding_model = None
        
        # Load SentenceTransformer for dense embeddings
        if use_dense_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[TalentMatcher] Loaded SentenceTransformer (all-MiniLM-L6-v2) for dense semantic scoring.")
            except Exception as e:
                print(f"[TalentMatcher] SentenceTransformer fallback ({e}). Using TF-IDF model.")
                self.use_dense_embeddings = False

    def extract_skills_from_text(self, text: str) -> Set[str]:
        """
        Extracts recognized technical skills using regex token boundaries and alias mappings.
        Handles special characters in skills like c++, scikit-learn, node.js, ci/cd.
        """
        if not isinstance(text, str):
            return set()
            
        text_lower = text.lower()
        extracted = set()
        
        # Check canonical skills
        for skill in self.canonical_skills:
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
            if re.search(pattern, text_lower):
                extracted.add(skill)
                
        # Check aliases
        for alias, canonical in SKILL_ALIASES.items():
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, text_lower):
                extracted.add(canonical)
                
        return extracted

    def compute_skill_overlap_score(
        self,
        candidate_skills: Set[str],
        required_skills: Set[str],
        preferred_skills: Set[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """Computes granular skill overlap and missing skill gap analysis."""
        req_norm = {normalize_skill(s) for s in required_skills}
        pref_norm = {normalize_skill(s) for s in preferred_skills}
        cand_norm = {normalize_skill(s) for s in candidate_skills}
        
        matched_required = cand_norm.intersection(req_norm)
        missing_required = req_norm - cand_norm
        
        matched_preferred = cand_norm.intersection(pref_norm)
        missing_preferred = pref_norm - cand_norm
        
        # Required match ratio (0.0 to 1.0)
        req_ratio = len(matched_required) / len(req_norm) if req_norm else 1.0
        # Preferred match bonus (0.0 to 1.0)
        pref_ratio = len(matched_preferred) / len(pref_norm) if pref_norm else 1.0
        
        # Weighted Hard Skill Score (80% Required + 20% Preferred if both present)
        if req_norm and pref_norm:
            skill_score = 0.80 * req_ratio + 0.20 * pref_ratio
        elif req_norm:
            skill_score = req_ratio
        elif pref_norm:
            skill_score = pref_ratio
        else:
            skill_score = 1.0
        
        # Gap severity classification
        if req_ratio >= 0.80:
            severity = "Low"
        elif req_ratio >= 0.50:
            severity = "Moderate"
        else:
            severity = "High"
            
        gap_details = {
            "matched_required": sorted(list(matched_required)),
            "missing_required": sorted(list(missing_required)),
            "matched_preferred": sorted(list(matched_preferred)),
            "missing_preferred": sorted(list(missing_preferred)),
            "required_match_pct": round(req_ratio * 100, 1),
            "preferred_match_pct": round(pref_ratio * 100, 1) if pref_norm else 100.0,
            "gap_severity": severity
        }
        
        return skill_score, gap_details

    def compute_lexical_similarity(self, resume_text: str, jd_text: str) -> float:
        """Computes TF-IDF lexical cosine similarity between resume and job description."""
        try:
            tfidf_mat = self.vectorizer.fit_transform([resume_text, jd_text])
            sim = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.5

    def compute_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Computes dense vector cosine similarity via SentenceTransformer."""
        if self.embedding_model is not None:
            emb_cand = self.embedding_model.encode([resume_text])
            emb_jd = self.embedding_model.encode([jd_text])
            sim = cosine_similarity(emb_cand, emb_jd)[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        else:
            return self.compute_lexical_similarity(resume_text, jd_text)

    def evaluate_candidate(
        self,
        candidate_profile: Dict[str, Any],
        job_description: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates an individual candidate against a structured job description.
        Weights:
        - Hard Skill Overlap: 40%
        - Dense Semantic Similarity: 30%
        - TF-IDF Lexical Similarity: 20%
        - Experience & Education Alignment: 10%
        Total: 100%
        """
        resume_text = candidate_profile["anonymized_text"]
        jd_text = job_description["description_text"]
        
        # Tier 1: Skill Extraction & Overlap (40%)
        candidate_skills = self.extract_skills_from_text(resume_text)
        required_skills = set(job_description.get("required_skills", []))
        preferred_skills = set(job_description.get("preferred_skills", []))
        skill_score, gap_info = self.compute_skill_overlap_score(candidate_skills, required_skills, preferred_skills)
        
        # Tier 2: Dense Semantic Similarity (30%)
        semantic_sim = self.compute_semantic_similarity(resume_text, jd_text)
        
        # Tier 3: Lexical TF-IDF Similarity (20%)
        lexical_sim = self.compute_lexical_similarity(resume_text, jd_text)
        
        # Tier 4: Experience & Education Alignment (10%)
        cand_exp = float(candidate_profile.get("experience_years", 0.0))
        req_exp = float(job_description.get("min_experience_years", 0.0))
        if req_exp > 0.0:
            exp_ratio = min(1.0, cand_exp / req_exp)
        else:
            exp_ratio = 1.0
        
        edu_level = candidate_profile.get("education_level", "Bachelor's Degree")
        edu_scores = {
            "PhD": 1.0,
            "Master's Degree": 0.95,
            "Bachelor's Degree": 0.85,
            "Associate / Certificate": 0.70
        }
        edu_factor = edu_scores.get(edu_level, 0.80)
        exp_edu_fit = 0.70 * exp_ratio + 0.30 * edu_factor
        
        # Composite Match Score (0 - 100 scale)
        composite_score = (
            0.40 * skill_score +
            0.30 * semantic_sim +
            0.20 * lexical_sim +
            0.10 * exp_edu_fit
        ) * 100.0
        composite_score = round(float(np.clip(composite_score, 0.0, 100.0)), 2)
        
        # Decision-support recommendation
        if composite_score >= 80.0 and gap_info["gap_severity"] == "Low":
            recommendation = "Strong Candidate: High technical alignment with core requirements. Advance to Technical Screen."
        elif composite_score >= 65.0:
            recommendation = "Potentially Qualified: Moderate match. Review missing competencies with hiring manager."
        else:
            recommendation = "Skill Gap Identified: Candidate lacks several core technical competencies for this position."
            
        return {
            "candidate_id": candidate_profile["candidate_id"],
            "composite_score": composite_score,
            "skill_score_pct": round(skill_score * 100, 1),
            "semantic_similarity_pct": round(semantic_sim * 100, 1),
            "lexical_similarity_pct": round(lexical_sim * 100, 1),
            "experience_years": cand_exp,
            "education_level": edu_level,
            "exp_edu_fit_pct": round(exp_edu_fit * 100, 1),
            "gap_severity": gap_info["gap_severity"],
            "matched_required": gap_info["matched_required"],
            "missing_required": gap_info["missing_required"],
            "matched_preferred": gap_info["matched_preferred"],
            "missing_preferred": gap_info["missing_preferred"],
            "recommendation": recommendation
        }

    def rank_candidates(
        self,
        candidate_profiles: List[Dict[str, Any]],
        job_description: Dict[str, Any]
    ) -> pd.DataFrame:
        """Ranks a list of candidate resumes against a Job Description."""
        evaluations = [self.evaluate_candidate(c, job_description) for c in candidate_profiles]
        df = pd.DataFrame(evaluations)
        df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"
        return df.reset_index()
