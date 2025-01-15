# services/data_gatherer_service.py

import json
import csv
import datetime
from typing import Dict, Any, Union, List
from pathlib import Path
from dateutil.parser import parse as parse_date

from services.hubspot_service import HubspotService
from utils.exceptions import HubSpotError
from utils.xai_integration import xai_news_search, xai_club_segmentation_search
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
        logger.debug("Initialized DataGathererService")
        
        # Load season data
        self.load_season_data()
        self.load_state_timezones()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data (now mostly delegated to HubspotService)."""
        return self.hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data from HubSpot, then merges competitor & season data.
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

        # Example: competitor check
        competitor_analysis = self.check_competitor_on_website(company_props.get("website", ""))

        # Example: gather news just once
        club_name = company_props.get("name", "")
        news_result = self.gather_club_news(club_name)

        # Build partial lead_sheet (now without emails and notes)
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
                "company_data": company_props
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
    # Competitor-check logic
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
            # Sample competitor mention
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
        """Get club information using segmentation."""
        if not club_name or not city or not state:
            return ""
        
        location = f"{city}, {state}"
        try:
            # Replace club_info_search with segmentation
            segmentation_data = xai_club_segmentation_search(club_name, location)
            return segmentation_data.get("club_info", "")
        except Exception as e:
            logger.error("Error gathering club info", extra={
                "club_name": club_name,
                "error": str(e)
            })
            return ""

    def gather_club_news(self, club_name: str) -> str:
        correlation_id = f"club_news_{club_name}"
        logger.debug("Starting club news search", extra={
            "club_name": club_name,
            "correlation_id": correlation_id
        })
        try:
            news = xai_news_search(club_name)
            # xai_news_search can return (news, icebreaker), so handle accordingly
            if isinstance(news, tuple):
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
            segmentation_info = xai_club_segmentation_search(company_name, location_str)
            return {
                "response": segmentation_info,
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

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Convert a value to int safely, returning default if it fails."""
        if value is None:
            return default
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def load_season_data(self):
        """Load golf season data from CSV files."""
        try:
            # Load city/state data
            if CITY_ST_CSV.exists():
                with open(CITY_ST_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Log the headers to debug
                    headers = reader.fieldnames
                    logger.debug(f"CSV headers: {headers}")
                    
                    for row in reader:
                        try:
                            # Adjust these keys based on your actual CSV headers
                            city = row.get('city', row.get('City', '')).lower()
                            state = row.get('state', row.get('State', '')).lower()
                            if city and state:
                                city_key = (city, state)
                                CITY_ST_DATA[city_key] = row
                        except Exception as row_error:
                            logger.warning(f"Skipping malformed row in city/state data: {row_error}")
                            continue

            # Load state-only data
            if ST_CSV.exists():
                with open(ST_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            state = row.get('state', row.get('State', '')).lower()
                            if state:
                                ST_DATA[state] = row
                        except Exception as row_error:
                            logger.warning(f"Skipping malformed row in state data: {row_error}")
                            continue

            logger.info("Loaded golf season data", extra={
                "city_st_count": len(CITY_ST_DATA),
                "state_count": len(ST_DATA)
            })

        except FileNotFoundError:
            logger.warning("Season data files not found, using defaults", extra={
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            # Continue with empty data, will use defaults
            pass
        except Exception as e:
            logger.error("Failed to load golf season data", extra={
                "error": str(e),
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            # Continue with empty data, will use defaults
            pass

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        """
        if not city and not state:
            return self._get_default_season_data()

        city_key = (city.lower(), state.lower())
        row = CITY_ST_DATA.get(city_key)
        if not row:
            row = ST_DATA.get(state.lower())

        if not row:
            # For Arizona, override with specific data
            if state.upper() == 'AZ':
                return {
                    "year_round": "Yes",  # Arizona is typically year-round golf
                    "start_month": "1",
                    "end_month": "12",
                    "peak_season_start": "01-01",
                    "peak_season_end": "12-31",
                    "status": "success",
                    "error": ""
                }
            # For Florida, override with specific data
            elif state.upper() == 'FL':
                return {
                    "year_round": "Yes",  # Florida is year-round golf
                    "start_month": "1",
                    "end_month": "12",
                    "peak_season_start": "01-01",
                    "peak_season_end": "12-31",
                    "status": "success",
                    "error": ""
                }
            return self._get_default_season_data()

        return {
            "year_round": "Yes" if row.get("Year-Round?", "").lower() == "yes" else "No",
            "start_month": row.get("Start Month", "1"),
            "end_month": row.get("End Month", "12"),
            "peak_season_start": self._month_to_first_day(row.get("Peak Season Start", "January")),
            "peak_season_end": self._month_to_last_day(row.get("Peak Season End", "December")),
            "status": "success",
            "error": ""
        }

    def _get_default_season_data(self) -> Dict[str, str]:
        """Return default season data."""
        return {
            "year_round": "No",
            "start_month": "3",
            "end_month": "11",
            "peak_season_start": "05-01",
            "peak_season_end": "08-31",
            "status": "default",
            "error": "Location not found, using defaults"
        }

    def _month_to_first_day(self, month_name: str) -> str:
        """
        Convert a month name (January, February, etc.) to a string "MM-01".
        Defaults to "05-01" (May 1) if unknown.
        """
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
        """
        Convert a month name (January, February, etc.) to a string "MM-DD"
        for the last day of that month. Defaults to "08-31" if unknown.
        """
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
        This can be adapted to store lead context in a local JSON file, for example.
        """
        # Implementation detail: e.g.,
        # with open(f"lead_contexts/{lead_email}.json", "w", encoding="utf-8") as f:
        #     json.dump(lead_sheet, f, indent=2)
        pass

    def load_state_timezones(self) -> None:
        """
        Load state timezone offsets from CSV file into STATE_TIMEZONES.
        The file must have columns: state_code, daylight_savings, standard_time
        """
        global STATE_TIMEZONES
        try:
            with open(TIMEZONE_CSV_PATH, 'r', encoding='utf-8') as file:
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
        Return timezone offset data for a given state code.
        Returns a dict: {'dst': int, 'std': int}
        """
        state_code = state.upper() if state else ''
        timezone_data = STATE_TIMEZONES.get(state_code, {
            'dst': -4,  # Default to Eastern (DST)
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
        Get club geography and type based on location + HubSpot data.
        
        Returns:
            (geography: str, club_type: str)
        """
        from utils.exceptions import HubSpotError
        try:
            # Attempt to get company data from HubSpot
            company_data = self.hubspot.get_company_data(club_name)
            geography = self.determine_geography(city, state)
            
            # Determine club type from the HubSpot data
            club_type = company_data.get("type", "Public Clubs")
            if not club_type or club_type.lower() == "unknown":
                club_type = "Public Clubs"
            
            return geography, club_type
            
        except HubSpotError:
            # If we fail to get company from HubSpot, default
            geography = self.determine_geography(city, state)
            return geography, "Public Clubs"

    def determine_geography(self, city: str, state: str) -> str:
        """Return 'City, State' or 'Unknown' if missing."""
        if not city or not state:
            return "Unknown"
        return f"{city}, {state}"
