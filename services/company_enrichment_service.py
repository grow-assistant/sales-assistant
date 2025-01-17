from typing import Dict, Any, Optional, Tuple
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window
from utils.web_fetch import fetch_website_html


class CompanyEnrichmentService:
    def __init__(self, api_key: str = HUBSPOT_API_KEY):
        logger.debug("Initializing CompanyEnrichmentService")
        self.hubspot = HubspotService(api_key=api_key)

    def enrich_company(self, company_id: str, additional_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Enriches company data with facility type and competitor information."""
        try:
            # Get current company info
            current_info = self._get_facility_info(company_id)
            if not current_info:
                return {"success": False, "error": "Failed to get company info"}

            # Get company name and location
            company_name = current_info.get('name', '')
            location = f"{current_info.get('city', '')}, {current_info.get('state', '')}"
            
            # IMPORTANT: Save the competitor value found in _get_facility_info
            competitor = current_info.get('competitor', 'Unknown')
            logger.debug(f"Competitor found in _get_facility_info: {competitor}")

            # Determine facility type
            facility_info = self._determine_facility_type(company_name, location)
            
            # Get seasonality info
            seasonality_info = self._determine_seasonality(current_info.get('state', ''))
            
            # Combine all new info
            new_info = {
                **facility_info,
                **seasonality_info,
                'competitor': competitor  # Explicitly set competitor here
            }
            
            # Add any additional data
            if additional_data:
                new_info.update(additional_data)
                
            # Log the competitor value before preparing updates
            logger.debug(f"Competitor value before _prepare_updates: {new_info.get('competitor', 'Unknown')}")
            
            # Prepare the final updates
            updates = self._prepare_updates(current_info, new_info)
            
            # Log the final competitor value
            logger.debug(f"Final competitor value in updates: {updates.get('competitor', 'Unknown')}")
            
            # Update the company in HubSpot
            if updates:
                success = self._update_company_properties(company_id, updates)
                if success:
                    return {"success": True, "data": updates}
            
            return {"success": False, "error": "Failed to update company"}
            
        except Exception as e:
            logger.error(f"Error enriching company {company_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_facility_info(self, company_id: str) -> Dict[str, Any]:
        """Fetches the company's current properties from HubSpot."""
        try:
            properties = [
                "name", "city", "state", "annualrevenue",
                "createdate", "hs_lastmodifieddate", "hs_object_id",
                "club_type", "facility_complexity", "has_pool",
                "has_tennis_courts", "number_of_holes",
                "geographic_seasonality", "public_private_flag",
                "club_info", "peak_season_start_month",
                "peak_season_end_month", "start_month", "end_month",
                "competitor", "domain"
            ]
            
            company_data = self.hubspot.get_company_by_id(company_id, properties)
            
            if not company_data:
                return {}
            
            # Check domain for competitor software
            domain = company_data.get('properties', {}).get('domain', '')
            competitor = company_data.get('properties', {}).get('competitor', 'Unknown')
            
            if domain:
                # Try different URL variations
                urls_to_try = []
                base_url = domain.strip().lower()
                if not base_url.startswith('http'):
                    urls_to_try.extend([f"https://{base_url}", f"http://{base_url}"])
                else:
                    urls_to_try.append(base_url)
                
                # Add www. version if not present
                urls_to_try.extend([url.replace('://', '://www.') for url in urls_to_try])
                
                for url in urls_to_try:
                    try:
                        html_content = fetch_website_html(url)
                        if html_content:
                            html_lower = html_content.lower()
                            # Check for Club Essentials mentions
                            clubessential_mentions = [
                                "copyright clubessential",
                                "clubessential, llc",
                                "www.clubessential.com",
                                "http://www.clubessential.com",
                                "clubessential"
                            ]
                            for mention in clubessential_mentions:
                                if mention in html_lower:
                                    competitor = "Club Essentials"
                                    logger.debug(f"Found Club Essentials on {url}")
                                    break
                                    
                            # Check for Jonas mentions if not Club Essentials
                            if competitor == "Unknown":
                                jonas_mentions = ["jonas club software", "jonas software", "jonasclub"]
                                for mention in jonas_mentions:
                                    if mention in html_lower:
                                        competitor = "Jonas"
                                        logger.debug(f"Found Jonas on {url}")
                                        break
                            
                            if competitor != "Unknown":
                                break
                    except Exception as e:
                        logger.debug(f"Failed to fetch {url}: {str(e)}")
                        continue
            
            return {
                'name': company_data.get('properties', {}).get('name', ''),
                'city': company_data.get('properties', {}).get('city', ''),
                'state': company_data.get('properties', {}).get('state', ''),
                'annual_revenue': company_data.get('properties', {}).get('annualrevenue', ''),
                'create_date': company_data.get('properties', {}).get('createdate', ''),
                'last_modified': company_data.get('properties', {}).get('hs_lastmodifieddate', ''),
                'object_id': company_data.get('properties', {}).get('hs_object_id', ''),
                'club_type': company_data.get('properties', {}).get('club_type', 'Unknown'),
                'facility_complexity': company_data.get('properties', {}).get('facility_complexity', 'Unknown'),
                'has_pool': company_data.get('properties', {}).get('has_pool', 'No'),
                'has_tennis_courts': company_data.get('properties', {}).get('has_tennis_courts', 'No'),
                'number_of_holes': company_data.get('properties', {}).get('number_of_holes', 0),
                'geographic_seasonality': company_data.get('properties', {}).get('geographic_seasonality', 'Unknown'),
                'public_private_flag': company_data.get('properties', {}).get('public_private_flag', 'Unknown'),
                'club_info': company_data.get('properties', {}).get('club_info', ''),
                'competitor': competitor
            }
        except Exception as e:
            logger.error(f"Error fetching company data: {e}")
            return {}

    def _determine_facility_type(self, company_name: str, location: str) -> Dict[str, Any]:
        """Uses xAI to determine facility type and details."""
        if not company_name or not location:
            return {}

        segmentation_info = xai_club_segmentation_search(company_name, location)
        club_summary = get_club_summary(company_name, location)

        official_name = (
            segmentation_info.get("name") or 
            company_name
        )

        club_type = "Country Club" if "country club" in official_name.lower() else segmentation_info.get("club_type", "Unknown")

        return {
            "name": official_name,
            "club_type": club_type,
            "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
            "geographic_seasonality": segmentation_info.get("geographic_seasonality", "Unknown"),
            "has_pool": segmentation_info.get("has_pool", "Unknown"),
            "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
            "number_of_holes": segmentation_info.get("number_of_holes", 0),
            "club_info": club_summary,
            "competitor": segmentation_info.get("competitor", "Unknown")
        }

    def _determine_seasonality(self, state: str) -> Dict[str, Any]:
        """Determines golf seasonality based on state."""
        season_data = {
            # Year-round states
            "AZ": {"start": 1, "end": 12, "peak_start": 10, "peak_end": 5, "type": "Year-Round Golf"},
            "FL": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "HI": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "CA": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "TX": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "GA": {"start": 1, "end": 12, "peak_start": 4, "peak_end": 10, "type": "Year-Round Golf"},
            "NV": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "AL": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "MS": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "LA": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            
            # Standard season states (Apr-Oct)
            "NC": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "SC": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "VA": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "TN": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "KY": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "MO": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "KS": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "OK": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "AR": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "NM": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            
            # Short season states (May-Sept/Oct)
            "MI": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "WI": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "MN": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "ND": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "SD": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "MT": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "ID": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "WY": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "ME": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "VT": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "NH": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            
            # Default season (Apr-Oct)
            "default": {"start": 4, "end": 10, "peak_start": 5, "peak_end": 9, "type": "Standard Season"}
        }
        
        state_data = season_data.get(state, season_data["default"])
        
        return {
            "geographic_seasonality": state_data["type"],
            "start_month": state_data["start"],
            "end_month": state_data["end"],
            "peak_season_start_month": state_data["peak_start"],
            "peak_season_end_month": state_data["peak_end"]
        }

    def _prepare_updates(self, current_info: Dict[str, Any], new_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the final update payload with validated values."""
        try:
            # Get competitor value with explicit logging
            competitor = new_info.get('competitor', current_info.get('competitor', 'Unknown'))
            logger.debug(f"Processing competitor value in _prepare_updates: {competitor}")

            # Initialize updates with default values for required fields
            updates = {
                "name": new_info.get("name", current_info.get("name", "")),
                "club_type": new_info.get("club_type", current_info.get("club_type", "Unknown")),
                "facility_complexity": new_info.get("facility_complexity", current_info.get("facility_complexity", "Unknown")),
                "geographic_seasonality": new_info.get("geographic_seasonality", current_info.get("geographic_seasonality", "Unknown")),
                "has_pool": "No",  # Default value
                "has_tennis_courts": new_info.get("has_tennis_courts", current_info.get("has_tennis_courts", "No")),
                "number_of_holes": new_info.get("number_of_holes", current_info.get("number_of_holes", 0)),
                "public_private_flag": new_info.get("public_private_flag", current_info.get("public_private_flag", "Unknown")),
                "start_month": new_info.get("start_month", current_info.get("start_month", "")),
                "end_month": new_info.get("end_month", current_info.get("end_month", "")),
                "peak_season_start_month": new_info.get("peak_season_start_month", current_info.get("peak_season_start_month", "")),
                "peak_season_end_month": new_info.get("peak_season_end_month", current_info.get("peak_season_end_month", "")),
                "competitor": competitor,
                "club_info": new_info.get("club_info", current_info.get("club_info", ""))
            }

            # Handle pool information
            club_info = new_info.get("club_info", "").lower()
            if "pool" in club_info:
                updates["has_pool"] = "Yes"
            else:
                updates["has_pool"] = current_info.get("has_pool", "No")

            # Convert numeric fields to integers
            for key in ["number_of_holes", "start_month", "end_month", "peak_season_start_month", "peak_season_end_month"]:
                if updates.get(key):
                    try:
                        updates[key] = int(updates[key])
                    except (ValueError, TypeError):
                        updates[key] = 0

            # Convert boolean fields to Yes/No
            for key in ["has_tennis_courts", "has_pool"]:
                updates[key] = "Yes" if str(updates.get(key, "")).lower() in ["yes", "true", "1"] else "No"

            # Validate competitor value
            valid_competitors = ["Club Essentials", "Jonas", "Unknown"]
            if competitor in valid_competitors:
                updates["competitor"] = competitor
                logger.debug(f"Set competitor to valid value: {competitor}")
            else:
                logger.debug(f"Invalid competitor value ({competitor}), defaulting to Unknown")
                updates["competitor"] = "Unknown"

            # Map values to HubSpot-accepted values
            property_value_mapping = {
                "club_type": {
                    "Private": "Private",
                    "Public": "Public",
                    "Public - Low Daily Fee": "Public - Low Daily Fee",
                    "Public - High Daily Fee": "Public - High Daily Fee",
                    "Municipal": "Municipal",
                    "Semi-Private": "Semi-Private",
                    "Resort": "Resort",
                    "Country Club": "Country Club",
                    "Private Country Club": "Country Club",
                    "Management Company": "Management Company",
                    "Unknown": "Unknown"
                },
                "facility_complexity": {
                    "Single-Course": "Standard",
                    "Multi-Course": "Multi-Course",
                    "Resort": "Resort",
                    "Unknown": "Unknown"
                },
                "geographic_seasonality": {
                    "Year-Round Golf": "Year-Round",
                    "Peak Summer Season": "Peak Summer Season",
                    "Short Summer Season": "Short Summer Season",
                    "Unknown": "Unknown"
                }
            }

            # Apply mappings
            for key, mapping in property_value_mapping.items():
                if key in updates:
                    updates[key] = mapping.get(str(updates[key]), updates[key])

            logger.debug(f"Final prepared updates: {updates}")
            return updates

        except Exception as e:
            logger.error(f"Error preparing updates: {str(e)}")
            logger.debug(f"Current info: {current_info}")
            logger.debug(f"New info: {new_info}")
            return {}

    def _update_company_properties(self, company_id: str, updates: Dict[str, Any]) -> bool:
        """Updates the company properties in HubSpot."""
        try:
            property_value_mapping = {
                "club_type": {
                    "Private": "Private",
                    "Public": "Public",
                    "Public - Low Daily Fee": "Public - Low Daily Fee",
                    "Public - High Daily Fee": "Public - High Daily Fee",
                    "Municipal": "Municipal",
                    "Semi-Private": "Semi-Private",
                    "Resort": "Resort",
                    "Country Club": "Country Club",
                    "Private Country Club": "Country Club",
                    "Management Company": "Management Company",
                    "Unknown": "Unknown"
                },
                "facility_complexity": {
                    "Single-Course": "Standard",
                    "Multi-Course": "Multi-Course",
                    "Resort": "Resort",
                    "Unknown": "Unknown"
                },
                "geographic_seasonality": {
                    "Year-Round Golf": "Year-Round",
                    "Peak Summer Season": "Peak Summer Season",
                    "Short Summer Season": "Short Summer Season",
                    "Unknown": "Unknown"
                },
                "competitor": {
                    "Jonas": "Jonas",
                    "Club Essentials": "Club Essentials",
                    "Unknown": "Unknown"
                }
            }

            mapped_updates = {}
            for key, value in updates.items():
                if key in property_value_mapping:
                    value = property_value_mapping[key].get(str(value), value)
                    
                if key in ["number_of_holes", "start_month", "end_month", 
                          "peak_season_start_month", "peak_season_end_month"]:
                    value = int(value) if str(value).isdigit() else 0
                elif key in ["has_pool", "has_tennis_courts"]:
                    value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                elif key == "club_info":
                    value = str(value)[:5000]

                mapped_updates[key] = value

            url = f"{self.hubspot.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            try:
                response = self.hubspot._make_hubspot_patch(url, payload)
                if response:
                    return True
                return False
            except HubSpotError as api_error:
                logger.error(f"HubSpot API Error: {str(api_error)}")
                raise
            
        except Exception as e:
            logger.error(f"Error updating company properties: {str(e)}")
            return False 