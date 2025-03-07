from typing import Dict, Any, Optional, Tuple
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger, setup_logging
from scripts.golf_outreach_strategy import get_best_outreach_window
from utils.web_fetch import fetch_website_html

# Create dedicated logger for company enrichment
enrichment_logger = setup_logging(
    log_name='company_enrichment',
    console_level='WARNING',
    file_level='DEBUG',
    max_bytes=5242880  # 5MB
)

class CompanyEnrichmentService:
    def __init__(self, api_key: str = HUBSPOT_API_KEY):
        enrichment_logger.debug("Initializing CompanyEnrichmentService")
        self.hubspot = HubspotService(api_key=api_key)

    def enrich_company(self, company_id: str, additional_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Enriches company data with facility type and competitor information."""
        try:
            # Get current company info
            current_info = self._get_facility_info(company_id)
            if not current_info:
                enrichment_logger.error(f"Failed to get company info for ID: {company_id}")
                return {"success": False, "error": "Failed to get company info"}

            # Get company name and location
            company_name = current_info.get('name', '')
            location = f"{current_info.get('city', '')}, {current_info.get('state', '')}"
            
            enrichment_logger.debug(f"Processing company: {company_name} in {location}")
            
            # IMPORTANT: Save the competitor value found in _get_facility_info
            competitor = current_info.get('competitor', 'Unknown')
            enrichment_logger.debug(f"Competitor found in _get_facility_info: {competitor}")

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
            enrichment_logger.debug(f"Competitor value before _prepare_updates: {new_info.get('competitor', 'Unknown')}")
            
            # Prepare the final updates
            updates = self._prepare_updates(current_info, new_info)
            
            # Log the final competitor value
            enrichment_logger.debug(f"Final competitor value in updates: {updates.get('competitor', 'Unknown')}")
            
            # Update the company in HubSpot
            if updates:
                success = self._update_company_properties(company_id, updates)
                if success:
                    return {"success": True, "data": updates}
            
            return {"success": False, "error": "Failed to update company"}
            
        except Exception as e:
            enrichment_logger.error(f"Error enriching company {company_id}: {str(e)}")
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
                "competitor", "domain", "company_short_name"
            ]
            
            company_data = self.hubspot.get_company_by_id(company_id, properties)
            
            if not company_data:
                return {}
            
            # Check domain for competitor software with shorter timeouts
            domain = company_data.get('properties', {}).get('domain', '')
            competitor = company_data.get('properties', {}).get('competitor', 'Unknown')
            
            if domain:
                # Try different URL variations with shorter timeout
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
                        # Reduced timeout from 10 seconds to 5 seconds
                        html_content = fetch_website_html(url, timeout=5)
                        if html_content:
                            html_lower = html_content.lower()
                            # Check for Club Essentials mentions first
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
                                    enrichment_logger.debug(f"Found Club Essentials on {url}")
                                    break
                                    
                            # Check for Jonas mentions if not Club Essentials
                            if competitor == "Unknown":
                                jonas_mentions = ["jonas club software", "jonas software", "jonasclub"]
                                for mention in jonas_mentions:
                                    if mention in html_lower:
                                        competitor = "Jonas"
                                        enrichment_logger.debug(f"Found Jonas on {url}")
                                        break
                            
                            # If we found a competitor, no need to check other URLs
                            if competitor != "Unknown":
                                break
                            
                    except Exception as e:
                        enrichment_logger.warning(f"Error checking {url}: {str(e)}")
                        continue
                
                # If all URLs failed, log it but continue processing
                if not any(url for url in urls_to_try):
                    enrichment_logger.warning(f"Could not access any URLs for domain: {domain}")
            
            # Return the company data with competitor info
            properties = company_data.get("properties", {})
            return {
                "name": properties.get("name", ""),
                "company_short_name": properties.get("company_short_name", ""),
                "city": properties.get("city", ""),
                "state": properties.get("state", ""),
                "annual_revenue": properties.get("annualrevenue", ""),
                "create_date": properties.get("createdate", ""),
                "last_modified": properties.get("hs_lastmodifieddate", ""),
                "object_id": properties.get("hs_object_id", ""),
                "club_type": properties.get("club_type", "Unknown"),
                "facility_complexity": properties.get("facility_complexity", "Unknown"),
                "has_pool": properties.get("has_pool", "Unknown"),
                "has_tennis_courts": properties.get("has_tennis_courts", "Unknown"),
                "number_of_holes": properties.get("number_of_holes", 0),
                "geographic_seasonality": properties.get("geographic_seasonality", "Unknown"),
                "public_private_flag": properties.get("public_private_flag", "Unknown"),
                "club_info": properties.get("club_info", ""),
                "competitor": competitor
            }

        except Exception as e:
            enrichment_logger.error(f"Error getting facility info: {str(e)}")
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
            "company_short_name": segmentation_info.get("company_short_name", ""),
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
            # Convert None values to "Unknown"
            has_pool = new_info.get("has_pool", current_info.get("has_pool", "Unknown"))
            if has_pool is None:
                has_pool = "Unknown"
            
            has_tennis = new_info.get("has_tennis_courts", current_info.get("has_tennis_courts", "Unknown"))
            if has_tennis is None:
                has_tennis = "Unknown"
            
            competitor = new_info.get("competitor", current_info.get("competitor", "Unknown"))
            if competitor is None:
                competitor = "Unknown"

            # Initialize updates with default values for required fields
            updates = {
                "name": new_info.get("name", current_info.get("name", "")),
                "company_short_name": new_info.get("company_short_name", ""),
                "club_type": new_info.get("club_type", current_info.get("club_type", "Unknown")),
                "facility_complexity": new_info.get("facility_complexity", current_info.get("facility_complexity", "Unknown")),
                "geographic_seasonality": new_info.get("geographic_seasonality", current_info.get("geographic_seasonality", "Unknown")),
                "has_pool": has_pool,
                "has_tennis_courts": has_tennis,
                "number_of_holes": new_info.get("number_of_holes", current_info.get("number_of_holes", 0)),
                "public_private_flag": new_info.get("public_private_flag", current_info.get("public_private_flag", "Unknown")),
                "start_month": new_info.get("start_month", current_info.get("start_month", "")),
                "end_month": new_info.get("end_month", current_info.get("end_month", "")),
                "peak_season_start_month": new_info.get("peak_season_start_month", current_info.get("peak_season_start_month", "")),
                "peak_season_end_month": new_info.get("peak_season_end_month", current_info.get("peak_season_end_month", "")),
                "competitor": competitor,
                "club_info": new_info.get("club_info", current_info.get("club_info", ""))
            }

            # Map values to HubSpot-accepted values
            property_value_mapping = {
                "club_type": {
                    "Private": "Private",
                    "Private Course": "Private",
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
                    "Seasonal": "Standard Season",
                    "Peak Summer Season": "Peak Summer Season",
                    "Short Summer Season": "Short Summer Season",
                    "Unknown": "Unknown"
                }
            }

            # Apply mappings
            for key, mapping in property_value_mapping.items():
                if key in updates:
                    updates[key] = mapping.get(str(updates[key]), updates[key])

            # Set public/private flag based on club type
            club_type = updates["club_type"]
            if "Country Club" in club_type or "Private" in club_type:
                updates["public_private_flag"] = "Private"
            elif any(t in club_type for t in ["Public", "Municipal"]):
                updates["public_private_flag"] = "Public"
            elif updates["public_private_flag"] is None:
                updates["public_private_flag"] = "Unknown"

            enrichment_logger.debug(f"Final prepared updates: {updates}")
            return updates

        except Exception as e:
            enrichment_logger.error(f"Error preparing updates: {str(e)}")
            enrichment_logger.debug(f"Current info: {current_info}")
            enrichment_logger.debug(f"New info: {new_info}")
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
                elif key == "company_short_name":
                    value = str(value)[:100]

                mapped_updates[key] = value

            url = f"{self.hubspot.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            try:
                response = self.hubspot._make_hubspot_patch(url, payload)
                if response:
                    enrichment_logger.debug(f"Successfully updated HubSpot with company_short_name: {updates.get('company_short_name')}")
                    return True
                return False
            except HubSpotError as api_error:
                enrichment_logger.error(f"HubSpot API Error: {str(api_error)}")
                raise
            
        except Exception as e:
            enrichment_logger.error(f"Error updating company properties: {str(e)}")
            return False 