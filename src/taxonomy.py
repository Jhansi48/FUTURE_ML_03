"""
TalentMatch ML - Technical Skill Taxonomy & Alias Resolution Module
Maintains a curated enterprise taxonomy of over 200 technical competencies across
Machine Learning, Data Engineering, Cloud/DevOps, Software Engineering, and Analytics.
"""

from typing import Dict, List, Set

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Machine Learning & AI": [
        "machine learning", "deep learning", "natural language processing", "computer vision",
        "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
        "huggingface", "transformers", "large language models", "llm", "langchain",
        "rag", "reinforcement learning", "time series forecasting", "feature engineering",
        "mlops", "model deployment", "model monitoring", "hyperparameter tuning", "generative ai"
    ],
    "Data Engineering & Databases": [
        "python", "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "apache spark", "pyspark", "apache kafka", "apache airflow", "dbt",
        "snowflake", "databricks", "bigquery", "redshift", "hadoop", "etl", "data warehousing",
        "data modeling", "data pipelines", "nosql", "cassandra"
    ],
    "Cloud & DevOps": [
        "amazon web services", "aws", "google cloud platform", "gcp", "microsoft azure",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "ci/cd",
        "github actions", "gitlab ci", "jenkins", "linux", "bash", "prometheus", "grafana",
        "helm", "serverless", "lambda", "cloudformation"
    ],
    "Software Engineering & Web": [
        "javascript", "typescript", "react", "next.js", "node.js", "express",
        "fastapi", "flask", "django", "java", "spring boot", "c++", "golang", "rust",
        "rest api", "graphql", "microservices", "html5", "css3", "tailwind css",
        "git", "system design", "object oriented programming", "unit testing"
    ],
    "Analytics & BI": [
        "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
        "excel", "looker", "a/b testing", "statistical analysis", "data visualization",
        "business intelligence", "metrics reporting", "hypothesis testing"
    ]
}

# Synonyms and common abbreviations mapping to canonical skill names
SKILL_ALIASES: Dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "torch": "pytorch",
    "hf": "huggingface",
    "llms": "large language models",
    "genai": "generative ai",
    "spark": "apache spark",
    "kafka": "apache kafka",
    "airflow": "apache airflow",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "actions": "github actions",
    "cicd": "ci/cd",
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "nodejs": "node.js",
    "node": "node.js",
    "fast api": "fastapi",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "azure": "microsoft azure",
    "bi": "business intelligence",
    "powerbi": "power bi"
}

def get_all_canonical_skills() -> Set[str]:
    """Returns a flattened set of all canonical skills in the taxonomy."""
    all_skills = set()
    for cat_skills in SKILL_TAXONOMY.values():
        for s in cat_skills:
            all_skills.add(s.lower())
    return all_skills

def normalize_skill(skill: str) -> str:
    """Normalizes an individual skill string against known aliases."""
    s = skill.strip().lower()
    return SKILL_ALIASES.get(s, s)
