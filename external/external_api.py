# File: external/external_api.py

import csv
import requests
import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from dateutil.parser import parse as parse_date
from tenacity import retry, wait_exponential, stop_after_attempt
import json

from utils.logging_setup import logger
from hubspot_integration.hubspot_api import (
    get_contact_by_email,
    get_contact_properties,
    get_all_notes_for_contact,
    get_associated_company_id,
    get_company_data
)
from utils.logging_setup import logger
from utils.xai_integration import xai_news_search  # Updated import

################################################################################
# CSV-based Season Data
################################################################################

PROJECT_ROOT = Path(__file__).parent.parent
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}

def load_season_data() -> None:
    """Load golf season data from CSV files into CITY_ST_DATA, ST_DATA dictionaries."""
    global CITY_ST_DATA, ST_DATA
    try:
        with CITY_ST_CSV.open('r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                city = row['City'].strip().lower()
                st = row['State'].strip().lower()
                CITY_ST_DATA[(city, st)] = row

        with ST_CSV.open('r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = row['State'].strip().lower()
                ST_DATA[st] = row

        logger.info("Successfully loaded season data")
    except Exception as e:
        logger.error(f"Error loading season data: {str(e)}")
        raise

# Load data at module import
load_season_data()

################################################################################
# Market Research Method Using xAI News Search
################################################################################

def market_research(company_name: str) -> Dict:
    """
    Example 'market_research' using xai_news_search for a quick summary of recent news.
    This completely replaces any old snippet with xAI-based content.
    """
    if not company_name:
        return {
            "company_overview": "",
            "recent_news": [],
            "status": "error"
        }

    query = f"Has {company_name} been in the news lately? Provide a short summary."
    news_response = xai_news_search(query)
    
    if not news_response:
        # If xAI call failed or returned empty
        return {
            "company_overview": f"Could not fetch recent events for {company_name}",
            "recent_news": [],
            "status": "error"
        }

    # If you want to parse the 'news_response' into structured data, do it here.
    # For now, we create a minimal 'recent_news' list with the entire response as a snippet.
    return {
        "company_overview": news_response,
        "recent_news": [
            {
                "title": "Recent News",
                "snippet": news_response,
                "link": "",       # If xAI returns links, place them here
                "date": ""        # If xAI returns a date, parse it here
            }
        ],
        "status": "success"
    }

################################################################################
# Interaction & Season Methods
################################################################################

def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int safely, defaulting if conversion fails.
    """
    if value is None:
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default

def review_previous_interactions(contact_id: str) -> Dict[str, Union[int, str]]:
    """
    Review previous interactions using HubSpot data.
    """
    try:
        lead_data = get_contact_properties(contact_id)
        if not lead_data:
            logger.warning(f"No lead data found for contact_id {contact_id}")
            return {
                "emails_opened": 0,
                "emails_sent": 0,
                "meetings_held": 0,
                "last_response": "No data available",
                "status": "no_data"
            }

        emails_opened = safe_int(lead_data.get("total_opens_weekly"))
        emails_sent = safe_int(lead_data.get("num_contacted_notes"))
        notes = get_all_notes_for_contact(contact_id)

        meeting_keywords = {"meeting", "meet", "call", "zoom", "teams"}
        meetings_held = sum(
            1 for note in notes
            if note.get("body") and any(keyword in note["body"].lower() for keyword in meeting_keywords)
        )

        last_reply = lead_data.get("hs_sales_email_last_replied")
        if last_reply:
            try:
                reply_date = parse_date(last_reply.replace('Z', '+00:00'))
                if reply_date.tzinfo is None:
                    reply_date = reply_date.replace(tzinfo=datetime.timezone.utc)
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                days_ago = (now_utc - reply_date).days
                last_response = f"Responded {days_ago} days ago"
            except ValueError:
                last_response = "Responded recently"
        else:
            if emails_opened > 0:
                last_response = "Opened emails but no direct reply"
            else:
                last_response = "No recent response"

        return {
            "emails_opened": emails_opened,
            "emails_sent": emails_sent,
            "meetings_held": meetings_held,
            "last_response": last_response,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error reviewing interactions for {contact_id}: {str(e)}")
        return {
            "emails_opened": 0,
            "emails_sent": 0,
            "meetings_held": 0,
            "last_response": "Error retrieving data",
            "status": "error",
            "error": str(e)
        }

def analyze_competitors() -> dict:
    """
    Basic placeholder logic to analyze competitors.
    """
    return {
        "industry_trends": "On-course mobile F&B ordering is growing rapidly.",
        "competitor_moves": ["Competitor A launched a pilot at several clubs."]
    }

def determine_club_season(city: str, state: str) -> dict:
    """
    Return the peak season data for the given city/state based on CSV lookups.
    """
    city_key = (city.lower(), state.lower())
    row = CITY_ST_DATA.get(city_key)

    if not row:
        row = ST_DATA.get(state.lower())

    if not row:
        # Default if not found
        return {
            "year_round": "Unknown",
            "start_month": "N/A",
            "end_month": "N/A",
            "peak_season_start": "05-01",
            "peak_season_end": "08-31"
        }

    year_round = row["Year-Round?"].strip()
    start_month_str = row["Start Month"].strip()
    end_month_str = row["End Month"].strip()
    peak_season_start_str = row["Peak Season Start"].strip()
    peak_season_end_str = row["Peak Season End"].strip()

    if not peak_season_start_str or peak_season_start_str == "N/A":
        peak_season_start_str = "May"
    if not peak_season_end_str or peak_season_end_str == "N/A":
        peak_season_end_str = "August"

    return {
        "year_round": year_round,
        "start_month": start_month_str,
        "end_month": end_month_str,
        "peak_season_start": month_to_first_day(peak_season_start_str),
        "peak_season_end": month_to_last_day(peak_season_end_str)
    }

def month_to_first_day(month_name: str) -> str:
    month_map = {
        "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
        "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
        "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
        "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
    }
    if month_name in month_map:
        return f"{month_map[month_name][0]}-01"
    return "05-01"

def month_to_last_day(month_name: str) -> str:
    month_map = {
        "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
        "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
        "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
        "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
    }
    if month_name in month_map:
        return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
    return "08-31"
