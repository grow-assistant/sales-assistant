# services/data_gatherer_service.py

import json
import csv
import datetime
from typing import Dict, Any, Union, List
from pathlib import Path
from dateutil.parser import parse as parse_date

from services.hubspot_service import HubspotService
from utils.xai_integration import xai_news_search, xai_club_info_search
from utils.web_fetch import fetch_website_html
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT
# CSV-based Season Data
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}


class DataGathererService:
    """
    Centralized service to gather all relevant data about a lead in one pass.
    Fetches HubSpot contact & company info, emails, competitor checks,
    interactions, market research, and season data.

    This version also saves the final lead context JSON to 'test_data/lead_contexts'
    for debugging or reference.
    """

    def __init__(self):
        """Initialize the DataGathererService with HubSpot client and season data."""
        self._hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        # Load season data at initialization
        self.load_season_data()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data."""
        return self._hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data sequentially using synchronous calls.
        """
        # 1) Lookup contact_id via email
        contact_data = self._hubspot.get_contact_by_email(lead_email)
        if not contact_data:
            logger.error(f"Could not find contact ID for {lead_email}")
            return {}
        contact_id = contact_data["id"]  # ID is directly on the contact object

        # 2) Get the contact properties
        contact_props = self._hubspot.get_contact_properties(contact_id)

        # 3) Get the associated company_id
        company_id = self._hubspot.get_associated_company_id(contact_id)

        # 4) Get the company data (including city/state)
        company_props = self._hubspot.get_company_data(company_id)

        # 5) Combine them into a single JSON
        lead_sheet = {
            "metadata": {
                "contact_id": contact_id,  # Keep for backwards compatibility
                "company_id": company_id,  # Keep for backwards compatibility
                "hs_object_id": contact_id,
                "company_hs_id": company_id,
                "lead_email": contact_props.get("email", ""),
                "status": "success"
            },
            "lead_data": {
                "id": contact_id,
                "properties": contact_props,
                "company_data": {
                    "hs_object_id": company_props.get("hs_object_id", ""),
                    "name": company_props.get("name", ""),
                    "city": company_props.get("city", ""),
                    "state": company_props.get("state", ""),
                    "domain": company_props.get("domain", ""),
                    "website": company_props.get("website", "")
                }
            },
            "analysis": {
                "competitor_analysis": self.check_competitor_on_website(company_props.get("website", "")),
                "research_data": self.market_research(company_props.get("name", "")),
                "previous_interactions": self.review_previous_interactions(contact_id),
                "season_data": self.determine_club_season(company_props.get("city", ""), company_props.get("state", "")),
                "facilities": self.check_facilities(
                    company_props.get("name", ""),
                    company_props.get("city", ""),
                    company_props.get("state", "")
                )
            }
        }

        # 7) Save the lead_sheet to disk so we can review the final context
        self._save_lead_context(lead_sheet, lead_email)

        # Log data gathering success
        logger.info(
            f"Data gathered successfully for email={lead_email}",
            extra={
                "contact_found": bool(contact_id),
                "company_found": bool(company_id),
                "has_research": bool(lead_sheet["analysis"]["research_data"]),
                "has_season_info": bool(lead_sheet["analysis"]["season_data"])
            }
        )
        return lead_sheet

    # -------------------------------------------------------------------------
    # New xAI helpers
    # -------------------------------------------------------------------------
    def gather_club_info(self, club_name: str, city: str, state: str) -> str:
        """
        Calls xai_club_info_search to get a short overview snippet about the club.
        """
        location_str = f"{city}, {state}".strip(", ")
        return xai_club_info_search(club_name, location_str, amenities=None)

    def gather_club_news(self, club_name: str) -> str:
        """
        Calls xai_news_search to get recent news about the club.
        """
        return xai_news_search(club_name)

    # ------------------------------------------------------------------------
    # PRIVATE METHODS FOR SAVING THE LEAD CONTEXT LOCALLY
    # ------------------------------------------------------------------------
    def check_competitor_on_website(self, domain: str) -> Dict[str, str]:
        """
        Check if Jonas Club Software is mentioned on the website.
        
        Args:
            domain (str): The domain to check (without http/https)
            
        Returns:
            Dict containing:
                - competitor: str ("Jonas" if found, empty string otherwise)
                - status: str ("success", "error", or "no_data")
                - error: str (error message if any)
        """
        try:
            if not domain:
                logger.warning("No domain provided for competitor check")
                return {
                    "competitor": "",
                    "status": "no_data",
                    "error": "No domain provided"
                }

            # Build URL carefully
            url = domain.strip().lower()
            if not url.startswith("http"):
                url = f"https://{url}"

            html = fetch_website_html(url)
            if not html:
                logger.warning(
                    "Could not fetch HTML for domain",
                    extra={
                        "domain": domain,
                        "error": "Possible Cloudflare block",
                        "status": "error"
                    }
                )
                return {
                    "competitor": "",
                    "status": "error",
                    "error": "Could not fetch website content"
                }

            # If we have HTML, proceed with competitor checks
            competitor_mentions = [
                "jonas club software",
                "jonas software",
                "jonasclub",
                "jonas club"
            ]

            for mention in competitor_mentions:
                if mention in html.lower():
                    logger.info(
                        "Found competitor mention on website",
                        extra={
                            "domain": domain,
                            "mention": mention,
                            "status": "success"
                        }
                    )
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
            logger.error(
                "Error checking competitor on website",
                extra={
                    "domain": domain,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "competitor": "",
                "status": "error",
                "error": f"Error checking competitor: {str(e)}"
            }

    def check_facilities(self, company_name: str, city: str, state: str) -> Dict[str, str]:
        """
        Query xAI about company facilities (golf course, pool, tennis courts) and club type.
        
        Args:
            company_name: Name of the company to check
            city: City where the company is located
            state: State where the company is located
            
        Returns:
            Dictionary containing:
                - response: Full xAI response about facilities
                - status: Status of the query ("success", "error", or "no_data")
        """
        try:
            if not company_name or not city or not state:
                logger.warning(
                    "Missing location data for facilities check",
                    extra={
                        "company": company_name,
                        "city": city,
                        "state": state,
                        "status": "no_data"
                    }
                )
                return {
                    "response": "",
                    "status": "no_data"
                }

            query = f"Does {company_name} in {city}, {state} have a Golf Course, pool, or tennis courts? Is it a private club?"
            response = xai_news_search(query)

            if not response:
                logger.warning(
                    "Failed to get facilities information",
                    extra={
                        "company": company_name,
                        "city": city,
                        "state": state,
                        "status": "error"
                    }
                )
                return {
                    "response": "",
                    "status": "error"
                }

            logger.info(
                "Facilities check completed",
                extra={
                    "company": company_name,
                    "city": city,
                    "state": state,
                    "status": "success"
                }
            )

            return {
                "response": response,
                "status": "success"
            }

        except Exception as e:
            logger.error(
                "Error checking facilities",
                extra={
                    "company": company_name,
                    "city": city,
                    "state": state,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "response": "",
                "status": "error"
            }

    def market_research(self, company_name: str) -> Dict[str, Any]:
        """
        Perform market research for a company using xAI news search.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            Dictionary containing:
                - company_overview: str (summary of company news)
                - recent_news: List[Dict] (list of news articles)
                - status: str ("success", "error", or "no_data")
                - error: str (error message if any)
        """
        try:
            if not company_name:
                logger.warning(
                    "No company name provided for market research",
                    extra={"status": "no_data"}
                )
                return {
                    "company_overview": "",
                    "recent_news": [],
                    "status": "no_data",
                    "error": "No company name provided"
                }

            query = f"Has {company_name} been in the news lately? Provide a short summary."
            news_response = xai_news_search(query)

            if not news_response:
                logger.warning(
                    "Failed to fetch news for company",
                    extra={
                        "company": company_name,
                        "status": "error"
                    }
                )
                return {
                    "company_overview": f"Could not fetch recent events for {company_name}",
                    "recent_news": [],
                    "status": "error",
                    "error": "No news data available"
                }

            logger.info(
                "Market research completed successfully",
                extra={
                    "company": company_name,
                    "has_news": bool(news_response),
                    "status": "success"
                }
            )
            return {
                "company_overview": news_response,
                "recent_news": [
                    {
                        "title": "Recent News",
                        "snippet": news_response,
                        "link": "",
                        "date": ""
                    }
                ],
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Error performing market research",
                extra={
                    "company": company_name,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "company_overview": "",
                "recent_news": [],
                "status": "error",
                "error": f"Error performing market research: {str(e)}"
            }

    def _save_lead_context(self, lead_sheet: Dict[str, Any], lead_email: str) -> None:
        """
        Save the lead_sheet dictionary to 'test_data/lead_contexts' as a JSON file.
        Masks sensitive data before saving.
        """
        try:
            context_dir = self._create_context_directory()
            filename = self._generate_context_filename(lead_email)
            file_path = context_dir / filename

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(lead_sheet, f, indent=2, ensure_ascii=False)

            logger.info(f"Lead context saved at: {file_path.resolve()}")
        except Exception as e:
            logger.warning(
                "Failed to save lead context (non-critical)",
                extra={
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "context_dir": str(context_dir)
                }
            )

    def _create_context_directory(self) -> Path:
        """
        Ensure test_data/lead_contexts directory exists and return it.
        """
        context_dir = PROJECT_ROOT / "test_data" / "lead_contexts"
        context_dir.mkdir(parents=True, exist_ok=True)
        return context_dir

    def _generate_context_filename(self, lead_email: str) -> str:
        """
        Generate a unique filename for storing the lead context,
        e.g., 'lead_context_smoran_shorthillsclub_org_20241225_001200.json'.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_email = lead_email.replace("@", "_").replace(".", "_")
        return f"lead_context_{safe_email}_{timestamp}.json"

    def load_season_data(self) -> None:
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
        
        Args:
            city (str): City name
            state (str): State name or abbreviation
            
        Returns:
            Dict containing:
                - year_round (str): "Yes", "No", or "Unknown"
                - start_month (str): Season start month or "N/A"
                - end_month (str): Season end month or "N/A"
                - peak_season_start (str): Peak season start date (MM-DD)
                - peak_season_end (str): Peak season end date (MM-DD)
                - status (str): "success", "error", or "no_data"
                - error (str): Error message if any
        """
        try:
            if not city and not state:
                logger.warning(
                    "No city or state provided for season lookup",
                    extra={"status": "no_data"}
                )
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
                logger.info(
                    "No season data found for location, using defaults",
                    extra={
                        "city": city,
                        "state": state,
                        "status": "no_data"
                    }
                )
                return {
                    "year_round": "Unknown",
                    "start_month": "N/A",
                    "end_month": "N/A",
                    "peak_season_start": "05-01",
                    "peak_season_end": "08-31",
                    "status": "no_data",
                    "error": "Location not found in season data"
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

            logger.info(
                "Successfully determined club season",
                extra={
                    "city": city,
                    "state": state,
                    "year_round": year_round,
                    "status": "success"
                }
            )
            return {
                "year_round": year_round,
                "start_month": start_month_str,
                "end_month": end_month_str,
                "peak_season_start": self._month_to_first_day(peak_season_start_str),
                "peak_season_end": self._month_to_last_day(peak_season_end_str),
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Error determining club season",
                extra={
                    "city": city,
                    "state": state,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "error",
                "error": f"Error determining season data: {str(e)}"
            }

    def _month_to_first_day(self, month_name: str) -> str:
        """Convert month name to first day of month in MM-DD format."""
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
        """Convert month name to last day of month in MM-DD format."""
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
        return "08-31"

    def review_previous_interactions(self, hs_object_id: str) -> Dict[str, Union[int, str]]:
        """
        Review previous interactions for a contact using HubSpot data.
        
        Args:
            hs_object_id (str): HubSpot contact ID
            
        Returns:
            Dict containing:
                - emails_opened (int): Number of emails opened
                - emails_sent (int): Number of emails sent
                - meetings_held (int): Number of meetings detected
                - last_response (str): Description of last response
                - status (str): "success", "error", or "no_data"
                - error (str): Error message if any
        """
        try:
            # Get contact properties from HubSpot
            lead_data = self._hubspot.get_contact_properties(hs_object_id)
            if not lead_data:
                logger.warning(
                    "No lead data found for contact",
                    extra={
                        "hs_object_id": hs_object_id,
                        "status": "no_data"
                    }
                )
                return {
                    "emails_opened": 0,
                    "emails_sent": 0,
                    "meetings_held": 0,
                    "last_response": "No data available",
                    "status": "no_data",
                    "error": "Contact not found in HubSpot"
                }

            # Extract email metrics
            emails_opened = self._safe_int(lead_data.get("total_opens_weekly"))
            emails_sent = self._safe_int(lead_data.get("num_contacted_notes"))

            # Get all notes for contact
            notes = self._hubspot.get_all_notes_for_contact(hs_object_id)

            # Count meetings from notes
            meeting_keywords = {"meeting", "meet", "call", "zoom", "teams"}
            meetings_held = sum(
                1 for note in notes
                if note.get("body") and any(keyword in note["body"].lower() for keyword in meeting_keywords)
            )

            # Determine last response status
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

            logger.info(
                "Successfully retrieved interaction history",
                extra={
                    "hs_object_id": hs_object_id,
                    "emails_opened": emails_opened,
                    "emails_sent": emails_sent,
                    "meetings_held": meetings_held,
                    "status": "success"
                }
            )
            return {
                "emails_opened": emails_opened,
                "emails_sent": emails_sent,
                "meetings_held": meetings_held,
                "last_response": last_response,
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Failed to review contact interactions",
                extra={
                    "hs_object_id": hs_object_id,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "emails_opened": 0,
                "emails_sent": 0,
                "meetings_held": 0,
                "last_response": "Error retrieving data",
                "status": "error",
                "error": f"Error retrieving interaction data: {str(e)}"
            }

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """
        Convert a value to int safely, defaulting if conversion fails.
        
        Args:
            value: Value to convert to integer
            default: Default value if conversion fails
            
        Returns:
            int: Converted value or default
        """
        if value is None:
            return default
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default
