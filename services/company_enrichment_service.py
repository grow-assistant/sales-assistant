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


class CompanyEnrichmentService:
    def __init__(self, api_key: str = HUBSPOT_API_KEY):
        logger.debug("Initializing CompanyEnrichmentService")
        self.hubspot = HubspotService(api_key=api_key)

    def enrich_company(self, company_id: str) -> Dict[str, Any]:
        """Main method to enrich company data."""
        logger.info(f"Starting enrichment for company {company_id}")
        try:
            # Get current company info
            logger.debug("Fetching current company info")
            company_info = self._get_facility_info(company_id)
            logger.debug(f"Retrieved company info: {company_info}")
            
            if not company_info.get('name') or not company_info.get('state'):
                logger.warning(f"Missing required info for company {company_id}")
                return {
                    'success': False,
                    'message': 'Missing required company information (name or state)',
                    'data': company_info
                }

            # Determine new values
            logger.debug(f"Determining facility type for {company_info['name']}")
            facility_data = self._determine_facility_type(
                company_info['name'],
                company_info['state']
            )
            logger.debug(f"Facility data: {facility_data}")
            
            # Add seasonality data
            logger.debug(f"Determining seasonality for state: {company_info['state']}")
            season_data = self._determine_seasonality(company_info['state'])
            logger.debug(f"Season data: {season_data}")
            
            # Combine all data
            enriched_data = {**company_info, **facility_data, **season_data}
            logger.debug(f"Combined enriched data: {enriched_data}")
            
            # Prepare updates
            logger.debug("Preparing updates for HubSpot")
            updates = self._prepare_updates(company_info, enriched_data)
            logger.debug(f"Prepared updates: {updates}")
            
            # Update HubSpot
            logger.info(f"Updating HubSpot for company {company_id}")
            success = self.hubspot.update_company_properties(company_id, updates)
            
            result = {
                'success': success,
                'message': 'Successfully updated company information' if success else 'Failed to update company information',
                'data': enriched_data
            }
            logger.info(f"Enrichment complete for company {company_id}: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error enriching company {company_id}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'data': {}
            }

    def _get_facility_info(self, company_id: str) -> Dict[str, Any]:
        """Fetches the company's current properties from HubSpot."""
        try:
            # Define properties to fetch
            properties = [
                "name", "city", "state", "annualrevenue",
                "createdate", "hs_lastmodifieddate", "hs_object_id",
                "club_type", "facility_complexity", "has_pool",
                "has_tennis_courts", "number_of_holes",
                "geographic_seasonality", "public_private_flag",
                "club_info", "peak_season_start_month",
                "peak_season_end_month", "start_month", "end_month"
            ]
            
            # Get company data using proper URL construction
            url = f"{self.hubspot.companies_endpoint}/{company_id}"
            params = {"properties": properties}
            
            # Make the request
            company_data = self.hubspot.get_company_by_id(company_id, properties)
            
            if not company_data:
                logger.warning(f"No data found for company {company_id}")
                return {}
            
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
                'club_info': company_data.get('properties', {}).get('club_info', '')
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

        # Check if "Country Club" is in the name and set club_type accordingly
        club_type = "Country Club" if "country club" in official_name.lower() else segmentation_info.get("club_type", "Unknown")

        return {
            "name": official_name,
            "club_type": club_type,
            "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
            "geographic_seasonality": segmentation_info.get("geographic_seasonality", "Unknown"),
            "has_pool": segmentation_info.get("has_pool", "Unknown"),
            "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
            "number_of_holes": segmentation_info.get("number_of_holes", 0),
            "club_info": club_summary
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
            # Start with current values as defaults
            updates = {
                "name": new_info.get("name", current_info.get("name", "")),
                "club_type": new_info.get("club_type", current_info.get("club_type", "Unknown")),
                "facility_complexity": new_info.get("facility_complexity", current_info.get("facility_complexity", "Unknown")),
                "geographic_seasonality": new_info.get("geographic_seasonality", current_info.get("geographic_seasonality", "Unknown")),
                "has_tennis_courts": new_info.get("has_tennis_courts", current_info.get("has_tennis_courts", "No")),
                "number_of_holes": new_info.get("number_of_holes", current_info.get("number_of_holes", 0)),
                "public_private_flag": new_info.get("public_private_flag", current_info.get("public_private_flag", "Unknown")),
                "start_month": new_info.get("start_month", current_info.get("start_month", "")),
                "end_month": new_info.get("end_month", current_info.get("end_month", "")),
                "peak_season_start_month": new_info.get("peak_season_start_month", current_info.get("peak_season_start_month", "")),
                "peak_season_end_month": new_info.get("peak_season_end_month", current_info.get("peak_season_end_month", ""))
            }

            # Handle club_info and check for pool
            club_info = new_info.get("club_info", "")
            if club_info:
                updates["club_info"] = club_info
                # Set has_pool based on presence of "pool" in club_info
                updates["has_pool"] = "Yes" if "pool" in club_info.lower() else "No"
            else:
                updates["has_pool"] = current_info.get("has_pool", "No")

            # Clean up values
            if updates["club_type"] == "Private Course":
                updates["club_type"] = "Private"

            # Ensure numeric values are integers
            for key in ["number_of_holes", "start_month", "end_month", "peak_season_start_month", "peak_season_end_month"]:
                if updates[key]:
                    try:
                        updates[key] = int(updates[key])
                    except (ValueError, TypeError):
                        updates[key] = 0

            # Ensure boolean values are Yes/No (except has_pool which is already handled)
            for key in ["has_tennis_courts"]:
                updates[key] = "Yes" if str(updates[key]).lower() in ["yes", "true", "1"] else "No"

            logger.debug(f"Prepared updates: {updates}")
            return updates

        except Exception as e:
            logger.error(f"Error preparing updates: {str(e)}")
            return {}

    def _update_company_properties(self, company_id: str, updates: Dict[str, Any]) -> bool:
        """Updates the company properties in HubSpot."""
        try:
            # Value transformations for HubSpot - EXACT matches for HubSpot enum values
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

            # Clean and map the updates
            mapped_updates = {}
            for key, value in updates.items():
                # Apply enum value transformations
                if key in property_value_mapping:
                    value = property_value_mapping[key].get(str(value), value)
                    
                # Type-specific handling
                if key in ["number_of_holes", "start_month", "end_month", 
                          "peak_season_start_month", "peak_season_end_month"]:
                    value = int(value) if str(value).isdigit() else 0
                elif key in ["has_pool", "has_tennis_courts"]:
                    value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                elif key == "club_info":
                    value = str(value)[:5000]  # Truncate to 5000 chars

                mapped_updates[key] = value

            # Debug logging
            logger.debug("Final HubSpot payload:")
            logger.debug(f"Company ID: {company_id}")
            logger.debug("Properties:")
            for key, value in mapped_updates.items():
                logger.debug(f"  {key}: {value} (Type: {type(value)})")

            # Send update to HubSpot
            url = f"{self.hubspot.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            try:
                response = self.hubspot._make_hubspot_patch(url, payload)
                if response:
                    logger.info(f"Successfully updated company {company_id}")
                    return True
                return False
            except HubSpotError as api_error:
                logger.error(f"HubSpot API Error Details:")
                logger.error(f"Status Code: {getattr(api_error, 'status_code', 'Unknown')}")
                logger.error(f"Response Body: {getattr(api_error, 'response_body', 'Unknown')}")
                logger.error(f"Request Body: {payload}")
                raise
            
        except Exception as e:
            logger.error(f"Error updating company properties: {str(e)}")
            return False 