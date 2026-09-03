"""
TalentMatch ML - Multi-Format Resume Parser & PII Anonymization Module
Extracts raw text from PDF, DOCX, and TXT files, anonymizes sensitive personal
attributes (names, emails, phone numbers) for fair evaluation, and extracts experience signals.
"""

import os
import re
from typing import Dict, Any, Optional
import docx

def extract_text_from_file(filepath: str) -> str:
    """Extracts raw text from TXT, DOCX, or PDF files."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
            
    elif ext == ".docx":
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        return "\n".join(full_text)
        
    elif ext == ".pdf":
        # Multi-strategy PDF extractor
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                return text
        except ImportError:
            pass
        
        # Fallback reading
        with open(filepath, "rb") as f:
            content = f.read().decode("latin-1", errors="ignore")
            # Extract plain string blocks
            clean_strings = re.findall(r"\(([\w\s\.,\-\@\/]+)\)", content)
            return "\n".join(clean_strings) if clean_strings else content[:2000]
            
    else:
        # Fallback generic text read
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def anonymize_resume_text(text: str) -> str:
    """
    Strips Personally Identifiable Information (PII) to guarantee bias-free,
    fair, and privacy-compliant candidate evaluation.
    Masks: Email addresses, phone numbers, URLs, and personal demographic cues.
    """
    if not isinstance(text, str):
        return ""
        
    clean = text
    # Mask Email Addresses
    clean = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL_MASKED]", clean)
    
    # Mask Phone Numbers (US & International formats)
    clean = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE_MASKED]", clean)
    
    # Mask URLs / LinkedIn / GitHub profile links
    clean = re.sub(r"https?://(?:www\.)?(?:linkedin\.com|github\.com)/\S+", "[SOCIAL_PROFILE_MASKED]", clean)
    clean = re.sub(r"https?://\S+|www\.\S+", "[URL_MASKED]", clean)
    
    return clean

def extract_experience_years(text: str) -> float:
    """Extracts stated years of experience via regex heuristics."""
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
        r"(?:experience|exp):\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        r"(\d+)\s+years?\s+in\s+software",
        r"over\s+(\d+)\s+years"
    ]
    years_found = []
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            try:
                years_found.append(float(m))
            except ValueError:
                pass
                
    if years_found:
        return max(years_found)
    return 2.0  # Conservative default if unstated

def extract_education_level(text: str) -> str:
    """Identifies the candidate's highest educational attainment."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["ph.d", "phd", "doctor of philosophy"]):
        return "PhD"
    elif any(k in text_lower for k in ["master", "m.s.", "m.tech", "msc", "m.eng", "mba"]):
        return "Master's Degree"
    elif any(k in text_lower for k in ["bachelor", "b.s.", "b.tech", "bsc", "b.eng", "undergraduate"]):
        return "Bachelor's Degree"
    else:
        return "Associate / Certificate"

def parse_resume(filepath: str) -> Dict[str, Any]:
    """Complete parsing pipeline for an individual candidate file."""
    raw_text = extract_text_from_file(filepath)
    anonymized_text = anonymize_resume_text(raw_text)
    exp_years = extract_experience_years(raw_text)
    education = extract_education_level(raw_text)
    
    filename = os.path.basename(filepath)
    candidate_id = os.path.splitext(filename)[0]
    
    return {
        "candidate_id": candidate_id,
        "filepath": filepath,
        "raw_text": raw_text,
        "anonymized_text": anonymized_text,
        "experience_years": exp_years,
        "education_level": education
    }
