# services/data_gatherer_service.py

import json
import csv
import datetime
from typing import Dict, Any, Union, List
from pathlib import Path
from dateutil.parser import parse as parse_date

from services.hubspot_service import HubspotService
from utils.exceptions import HubSpotError
from utils.xai_integration import xai_news_search, xai_club_info_search
from utils.web_fetch import fetch_website_html
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT
from utils.formatting_utils import clean_html

# CSV-based Season Data
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}

TIMEZONE_CSV_PATH = PROJECT_ROOT / 'docs' / 'data' / 'state_timezones.csv'
STATE_TIMEZONES: Dict[str, Dict[str, int]] = {}


class DataGathererService:
    """
    Centralized service to gather all relevant data about a lead in one pass.
    Fetches HubSpot contact & company info, emails, competitor checks,
    interactions, market research, and season data.
    """

    def __init__(self):
        """Initialize the DataGathererService with HubSpot client and season data."""
        self.hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        # Load season data at initialization
        self.load_season_data()
        self.load_state_timezones()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data (now mostly delegated to HubspotService)."""
        return self.hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data from HubSpot, notes, emails, then merges competitor & season data.
        """
        if correlation_id is None:
            correlation_id = f"gather_{lead_email}"

        # 1) Lookup contact_id via email
        contact_data = self.hubspot.get_contact_by_email(lead_email)
        if not contact_data:
            logger.error("Failed to find contact ID", extra={
                "email": lead_email,
                "operation": "gather_lead_data",
                "correlation_id": correlation_id,
                "status": "error"
            })
            return {}
        contact_id = contact_data["id"]

        # 2) Get the contact properties
        contact_props = self.hubspot.get_contact_properties(contact_id)

        # 3) Get the associated company_id
        company_id = self.hubspot.get_associated_company_id(contact_id)

        # 4) Get the company data (including city/state, plus new fields)
        company_props = self.hubspot.get_company_data(company_id)

        # 5) Add calls to fetch emails and notes from HubSpot
        emails = self.hubspot.get_all_emails_for_contact(contact_id)
        notes = self.hubspot.get_all_notes_for_contact(contact_id)

        # Example: competitor check
        competitor_analysis = self.check_competitor_on_website(company_props.get("website", ""))

        # Example: gather news just once
        club_name = company_props.get("name", "")
        news_result = self.gather_club_news(club_name)

        # Build partial lead_sheet
        lead_sheet = {
            "metadata": {
                "contact_id": contact_id,
                "company_id": company_id,
                "lead_email": contact_props.get("email", ""),
                "status": "success"
            },
            "lead_data": {
                "id": contact_id,
                "properties": contact_props,
                # This is where all 15 fields will appear (no filtering):
                "company_data": company_props,
                "emails": emails,
                "notes": notes
            },
            "analysis": {
                "competitor_analysis": competitor_analysis,
                "research_data": {
                    "company_overview": news_result,
                    "recent_news": [{
                        "title": "Recent News",
                        "snippet": news_result,
                        "link": "",
                        "date": ""
                    }] if news_result else [],
                    "status": "success",
                    "error": ""
                },
                "previous_interactions": self.review_previous_interactions(contact_id),
                "season_data": self.determine_club_season(
                    company_props.get("city", ""), 
                    company_props.get("state", "")
                ),
                "facilities": self.check_facilities(
                    club_name,
                    company_props.get("city", ""),
                    company_props.get("state", "")
                )
            }
        }

        # Optionally save or log the final lead_sheet
        self._save_lead_context(lead_sheet, lead_email)

        logger.info(
            "Data gathering completed successfully",
            extra={
                "email": lead_email,
                "contact_id": contact_id,
                "company_id": company_id,
                "correlation_id": correlation_id,
                "operation": "gather_lead_data"
            }
        )
        return lead_sheet

    # -------------------------------------------------------------------------
    # Example competitor-check logic
    # -------------------------------------------------------------------------
    def check_competitor_on_website(self, domain: str, correlation_id: str = None) -> Dict[str, str]:
        if correlation_id is None:
            correlation_id = f"competitor_check_{domain}"
        try:
            if not domain:
                return {
                    "competitor": "",
                    "status": "no_data",
                    "error": "No domain provided"
                }
            url = domain.strip().lower()
            if not url.startswith("http"):
                url = f"https://{url}"
            html = fetch_website_html(url)
            if not html:
                return {
                    "competitor": "",
                    "status": "error",
                    "error": "Could not fetch website content"
                }
            # sample competitor mention
            competitor_mentions = ["jonas club software", "jonas software", "jonasclub"]
            for mention in competitor_mentions:
                if mention in html.lower():
                    return {
                        "competitor": "Jonas",
                        "status": "success",
                        "error": ""
                    }
            return {
                "competitor": "",
                "status": "success",
                "error": ""
            }
        except Exception as e:
            logger.error("Error checking competitor on website", extra={
                "domain": domain,
                "error_type": type(e).__name__,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return {
                "competitor": "",
                "status": "error",
                "error": f"Error checking competitor: {str(e)}"
            }

    def gather_club_info(self, club_name: str, city: str, state: str) -> str:
        correlation_id = f"club_info_{club_name}"
        logger.debug("Starting club info search", extra={
            "club_name": club_name,
            "city": city,
            "state": state,
            "correlation_id": correlation_id
        })
        location_str = f"{city}, {state}".strip(", ")
        try:
            info = xai_club_info_search(club_name, location_str, amenities=None)
            logger.info("Club info search completed", extra={
                "club_name": club_name,
                "has_info": bool(info),
                "correlation_id": correlation_id
            })
            return info
        except Exception as e:
            logger.error("Error searching club info", extra={
                "club_name": club_name,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    def gather_club_news(self, club_name: str) -> str:
        correlation_id = f"club_news_{club_name}"
        logger.debug("Starting club news search", extra={
            "club_name": club_name,
            "correlation_id": correlation_id
        })
        try:
            news = xai_news_search(club_name)
            if isinstance(news, tuple):
                # If xai_news_search returns (news, icebreaker)
                news = news[0]
            logger.info("Club news search completed", extra={
                "club_name": club_name,
                "has_news": bool(news),
                "correlation_id": correlation_id
            })
            return news
        except Exception as e:
            logger.error("Error searching club news", extra={
                "club_name": club_name,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    def market_research(self, company_name: str) -> Dict[str, Any]:
        """Just a wrapper around gather_club_news for example."""
        news_response = self.gather_club_news(company_name)
        return {
            "company_overview": news_response,
            "recent_news": [{
                "title": "Recent News",
                "snippet": news_response,
                "link": "",
                "date": ""
            }] if news_response else [],
            "status": "success",
            "error": ""
        }

    def check_facilities(self, company_name: str, city: str, state: str) -> Dict[str, str]:
        correlation_id = f"facilities_{company_name}"
        if not company_name or not city or not state:
            return {
                "response": "",
                "status": "no_data"
            }
        location_str = f"{city}, {state}".strip(", ")
        try:
            response = xai_club_info_search(company_name, location_str, amenities=["Golf Course", "Pool", "Tennis Courts"])
            return {
                "response": response,
                "status": "success"
            }
        except Exception as e:
            logger.error("Error checking facilities", extra={
                "company": company_name,
                "city": city,
                "state": state,
                "error_type": type(e).__name__,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return {
                "response": "",
                "status": "error"
            }

    def review_previous_interactions(self, contact_id: str) -> Dict[str, Union[int, str]]:
        """
        Review previous interactions for a contact using HubSpot data.
        """
        try:
            lead_data = self.hubspot.get_contact_properties(contact_id)
            if not lead_data:
                return {
                    "emails_opened": 0,
                    "emails_sent": 0,
                    "meetings_held": 0,
                    "last_response": "No data available",
                    "status": "no_data",
                    "error": "Contact not found in HubSpot"
                }

            emails_opened = self._safe_int(lead_data.get("total_opens_weekly"))
            emails_sent = self._safe_int(lead_data.get("num_contacted_notes"))
            notes = self.hubspot.get_all_notes_for_contact(contact_id)

            meeting_keywords = {"meeting", "meet", "call", "zoom", "teams"}
            meetings_held = 0
            for note in notes:
                if note.get("body") and any(keyword in note["body"].lower() for keyword in meeting_keywords):
                    meetings_held += 1

            last_reply = lead_data.get("hs_sales_email_last_replied")
            if last_reply:
                try:
                    dt = parse_date(last_reply.replace("Z", "+00:00"))
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    if not dt.tzinfo:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    days_ago = (now_utc - dt).days
                    last_response = f"Responded {days_ago} days ago"
                except ValueError:
                    last_response = "Responded recently"
            else:
                last_response = "No recent response" if emails_opened == 0 else "Opened emails but no direct reply"

            return {
                "emails_opened": emails_opened,
                "emails_sent": emails_sent,
                "meetings_held": meetings_held,
                "last_response": last_response,
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error("Failed to review contact interactions", extra={
                "contact_id": contact_id,
                "error_type": type(e).__name__,
                "error": str(e)
            })
            return {
                "emails_opened": 0,
                "emails_sent": 0,
                "meetings_held": 0,
                "last_response": "Error retrieving data",
                "status": "error",
                "error": f"Error retrieving interaction data: {str(e)}"
            }

    def _safe_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def load_season_data(self) -> None:
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

            logger.info("Successfully loaded golf season data", extra={
                "city_state_count": len(CITY_ST_DATA),
                "state_count": len(ST_DATA)
            })
        except Exception as e:
            logger.error("Failed to load golf season data", extra={
                "error": str(e),
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            raise

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        """
        if not city and not state:
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "no_data",
                "error": "No location data provided"
            }

        city_key = (city.lower(), state.lower())
        row = CITY_ST_DATA.get(city_key)
        if not row:
            row = ST_DATA.get(state.lower())

        if not row:
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "no_data",
                "error": "Location not found"
            }

        year_round = row["Year-Round?"].strip()
        start_month_str = row["Start Month"].strip()
        end_month_str = row["End Month"].strip()
        peak_season_start_str = row["Peak Season Start"].strip() or "May"
        peak_season_end_str = row["Peak Season End"].strip() or "August"

        return {
            "year_round": year_round,
            "start_month": start_month_str,
            "end_month": end_month_str,
            "peak_season_start": self._month_to_first_day(peak_season_start_str),
            "peak_season_end": self._month_to_last_day(peak_season_end_str),
            "status": "success",
            "error": ""
        }

    def _month_to_first_day(self, month_name: str) -> str:
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-01"
        return "05-01"

    def _month_to_last_day(self, month_name: str) -> str:
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
        return "08-31"

    def _save_lead_context(self, lead_sheet: Dict[str, Any], lead_email: str) -> None:
        """
        Optionally save the lead_sheet for debugging or offline reference.
        """
        # (Implementation detail)
        pass

    def load_state_timezones(self) -> None:
        """Load state timezone offsets from CSV file."""
        global STATE_TIMEZONES
        try:
            with open(TIMEZONE_CSV_PATH, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    state_code = row['state_code'].strip()
                    STATE_TIMEZONES[state_code] = {
                        'dst': int(row['daylight_savings']),
                        'std': int(row['standard_time'])
                    }
            logger.debug(f"Loaded timezone data for {len(STATE_TIMEZONES)} states")
        except Exception as e:
            logger.error(f"Error loading state timezones: {str(e)}")
            # Default to Eastern Time if loading fails
            STATE_TIMEZONES = {}
    
    def get_club_timezone(self, state: str) -> dict:
        """
        Get timezone offset data for a given state.
        
        Args:
            state (str): Two-letter state code
            
        Returns:
            dict: Dictionary containing DST and standard time offsets
                  {'dst': int, 'std': int}
        """
        state_code = state.upper() if state else ''
        timezone_data = STATE_TIMEZONES.get(state_code, {
            'dst': -4,  # Default to Eastern Time
            'std': -5
        })
        
        logger.debug("Retrieved timezone data for state", extra={
            "state": state_code,
            "dst_offset": timezone_data['dst'],
            "std_offset": timezone_data['std']
        })
        
        return timezone_data

    def get_club_geography_and_type(self, club_name: str, city: str, state: str) -> tuple:
        """
        Get club geography and type based on location and HubSpot data.
        
        Args:
            club_name (str): Name of the club
            city (str): City where the club is located
            state (str): State where the club is located
            
        Returns:
            tuple: (geography: str, club_type: str)
        """
        try:
            # Try to get company data from HubSpot
            company_data = self.hubspot.get_company_data(club_name)
            geography = self.determine_geography(city, state)
            
            # Determine club type from HubSpot data
            club_type = company_data.get("type", "Public Clubs")
            if not club_type or club_type.lower() == "unknown":
                club_type = "Public Clubs"
            
            return geography, club_type
            
        except HubSpotError:
            # If company not found in HubSpot, use defaults
            geography = self.determine_geography(city, state)
            return geography, "Public Clubs"

    def determine_geography(self, city: str, state: str) -> str:
        """
        Determine geography string from city and state.
        
        Args:
            city (str): City name
            state (str): State code
            
        Returns:
            str: Geography string in format "City, State" or "Unknown"
        """
        if not city or not state:
            return "Unknown"
        return f"{city}, {state}"
