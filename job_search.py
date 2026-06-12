"""
Job Search Module for Resume Classifier
Integrates with Adzuna API to fetch job listings in India
and constructs LinkedIn apply URLs.
"""

import requests
import streamlit as st
from urllib.parse import quote


# ── Category → Search Keyword Mapping ──────────────────────────────────────────
# Maps ML-predicted resume categories to optimized job search keywords.
CATEGORY_KEYWORDS = {
    "Java Developer": "Java Developer",
    "Python Developer": "Python Developer",
    "Data Science": "Data Scientist",
    "Web Designing": "Web Designer Frontend Developer",
    "HR": "Human Resources HR Manager",
    "Mechanical Engineer": "Mechanical Engineer",
    "Sales": "Sales Executive Manager",
    "Health and fitness": "Healthcare Fitness Trainer",
    "Civil Engineer": "Civil Engineer",
    "Business Analyst": "Business Analyst",
    "SAP Developer": "SAP Developer Consultant",
    "Automation Testing": "QA Automation Tester",
    "Electrical Engineering": "Electrical Engineer",
    "Operations Manager": "Operations Manager",
    "DevOps Engineer": "DevOps Engineer",
    "Network Security Engineer": "Network Security Engineer",
    "PMO": "Project Manager PMO",
    "Database": "Database Administrator DBA",
    "Hadoop": "Hadoop Big Data Engineer",
    "ETL Developer": "ETL Developer Data Engineer",
    "DotNet Developer": ".NET Developer",
    "Blockchain": "Blockchain Developer",
    "Testing": "Software Tester QA Engineer",
    "Arts": "Graphic Designer Creative Arts",
    "Advocate": "Legal Advocate Lawyer",
}


def map_category_to_keywords(category: str) -> str:
    """
    Convert a model-predicted category into optimized search keywords.
    Falls back to the raw category name if no mapping exists.
    """
    return CATEGORY_KEYWORDS.get(category, category)


# ── Adzuna API Integration ─────────────────────────────────────────────────────

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"


def _get_adzuna_credentials():
    """
    Retrieve Adzuna API credentials from Streamlit secrets.
    Returns (app_id, app_key) or (None, None) if not configured.
    """
    try:
        app_id = st.secrets["adzuna"]["app_id"]
        app_key = st.secrets["adzuna"]["app_key"]
        # Check if they are still placeholder values
        if app_id == "YOUR_APP_ID" or app_key == "YOUR_APP_KEY":
            return None, None
        return app_id, app_key
    except (KeyError, FileNotFoundError):
        return None, None


def search_jobs_adzuna(category: str, num_results: int = 10) -> list[dict]:
    """
    Search for jobs on Adzuna for the given category in India.

    Args:
        category: The ML-predicted resume category.
        num_results: Number of job results to return (max 50).

    Returns:
        A list of dicts, each containing:
            - title (str)
            - company (str)
            - location (str)
            - salary_min (float | None)
            - salary_max (float | None)
            - description (str)
            - redirect_url (str)
            - created (str)
        Returns an empty list on failure.
    """
    app_id, app_key = _get_adzuna_credentials()
    if not app_id:
        return []

    keywords = map_category_to_keywords(category)

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": keywords,
        "results_per_page": min(num_results, 50),
        "content-type": "application/json",
        "sort_by": "relevance",
    }

    try:
        response = requests.get(ADZUNA_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        jobs = []
        for result in data.get("results", []):
            job = {
                "title": result.get("title", "Untitled Position"),
                "company": (result.get("company", {}) or {}).get("display_name", "Company Not Listed"),
                "location": (result.get("location", {}) or {}).get("display_name", "India"),
                "salary_min": result.get("salary_min"),
                "salary_max": result.get("salary_max"),
                "description": result.get("description", ""),
                "redirect_url": result.get("redirect_url", ""),
                "created": result.get("created", ""),
            }
            jobs.append(job)

        return jobs

    except requests.exceptions.RequestException:
        return []
    except (ValueError, KeyError):
        return []


# ── LinkedIn URL Builders ──────────────────────────────────────────────────────

LINKEDIN_JOBS_BASE = "https://www.linkedin.com/jobs/search/"


def build_linkedin_apply_url(job_title: str, company: str | None = None) -> str:
    """
    Construct a LinkedIn Jobs search URL for a specific job + company.
    This serves as the 'Apply on LinkedIn' destination.
    """
    query = job_title
    if company and company != "Company Not Listed":
        query = f"{job_title} {company}"
    return f"{LINKEDIN_JOBS_BASE}?keywords={quote(query)}&location=India"


def get_linkedin_search_url(category: str) -> str:
    """
    Construct a general LinkedIn Jobs search URL for a resume category.
    Used for the 'See All Jobs on LinkedIn' button.
    """
    keywords = map_category_to_keywords(category)
    return f"{LINKEDIN_JOBS_BASE}?keywords={quote(keywords)}&location=India"


def is_api_configured() -> bool:
    """Check if Adzuna API credentials are properly configured."""
    app_id, app_key = _get_adzuna_credentials()
    return app_id is not None
