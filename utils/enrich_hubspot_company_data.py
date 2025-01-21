 # tests/test_hubspot_company_type.py

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window

# Add these constants after imports
###########################
# CONFIG / CONSTANTS
###########################
TEST_MODE = False  # Set to False for production
TEST_LIMIT = 3    # Number of companies to process in test mode
BATCH_SIZE = 25   # Companies per API request
TEST_COMPANY_ID = "15537469970"  # Set this to a specific company ID to test just that company


def get_facility_info(company_id: str) -> tuple[
    str, str, str, str, str, str, str, str, str, str, str, int, str, str, str, str
]:
    """
    Fetches the company's properties from HubSpot.
    Returns a tuple of company properties including club_info and company_short_name.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        company_data = hubspot.get_company_data(company_id)

        name = company_data.get("name", "")
        company_short_name = company_data.get("company_short_name", "")
        city = company_data.get("city", "")
        state = company_data.get("state", "")
        annual_revenue = company_data.get("annualrevenue", "")
        create_date = company_data.get("createdate", "")
        last_modified = company_data.get("hs_lastmodifieddate", "")
        object_id = company_data.get("hs_object_id", "")
        club_type = company_data.get("club_type", "Unknown")
        facility_complexity = company_data.get("facility_complexity", "Unknown")
        has_pool = company_data.get("has_pool", "No")
        has_tennis_courts = company_data.get("has_tennis_courts", "No")
        number_of_holes = company_data.get("number_of_holes", 0)
        geographic_seasonality = company_data.get("geographic_seasonality", "Unknown")
        public_private_flag = company_data.get("public_private_flag", "Unknown")
        club_info = company_data.get("club_info", "")

        return (
            name,
            company_short_name,
            city,
            state,
            annual_revenue,
            create_date,
            last_modified,
            object_id,
            club_type,
            facility_complexity,
            has_pool,
            has_tennis_courts,
            number_of_holes,
            geographic_seasonality,
            public_private_flag,
            club_info,
        )

    except HubSpotError as e:
        print(f"Error fetching company data: {e}")
        return ("", "", "", "", "", "", "", "", "", "No", "No", 0, "", "", "", "")


def determine_facility_type(company_name: str, location: str) -> dict:
    """
    Uses xAI to determine the facility type and official name based on company info.
    """
    if not company_name or not location:
        return {}

    # Get segmentation data
    segmentation_info = xai_club_segmentation_search(company_name, location)
    
    # Get summary for additional context
    club_summary = get_club_summary(company_name, location)

    # Extract name and generate short name from segmentation info
    official_name = segmentation_info.get("name") or company_name
    # Generate short name by removing common words and limiting length
    short_name = official_name.replace("Country Club", "").replace("Golf Club", "").strip()
    short_name = short_name[:100]  # Ensure it fits HubSpot field limit

    full_info = {
        "name": official_name,
        "company_short_name": short_name,
        "club_type": segmentation_info.get("club_type", "Unknown"),
        "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
        "geographic_seasonality": segmentation_info.get("geographic_seasonality", "Unknown"),
        "has_pool": segmentation_info.get("has_pool", "Unknown"),
        "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
        "number_of_holes": segmentation_info.get("number_of_holes", 0),
        "club_info": club_summary
    }

    return full_info


def update_company_properties(company_id: str, club_info: dict, confirmed_updates: dict) -> bool:
    """
    Updates the company's properties in HubSpot based on club segmentation info.
    """
    try:
        # Check club_info for pool mentions before processing
        club_info_text = str(confirmed_updates.get('club_info', '')).lower()
        if 'pool' in club_info_text and confirmed_updates.get('has_pool') in ['Unknown', 'No']:
            logger.debug(f"Found pool mention in club_info, updating has_pool to Yes")
            confirmed_updates['has_pool'] = 'Yes'

        # Debug input values
        logger.debug("Input values for update:")
        for key, value in confirmed_updates.items():
            logger.debug(f"Field: {key}, Value: {value}, Type: {type(value)}")

        # Map our internal property names to HubSpot property names
        hubspot_property_mapping = {
            "name": "name",
            "club_type": "club_type",
            "facility_complexity": "facility_complexity",
            "geographic_seasonality": "geographic_seasonality",
            "public_private_flag": "public_private_flag",
            "has_pool": "has_pool",
            "has_tennis_courts": "has_tennis_courts",
            "number_of_holes": "number_of_holes",
            "club_info": "club_info",
            "season_start": "start_month",
            "season_end": "end_month",
            "peak_season_start_month": "peak_season_start_month",
            "peak_season_end_month": "peak_season_end_month",
            "notes_last_contacted": "notes_last_contacted",
            "num_contacted_notes": "num_contacted_notes",
            "num_associated_contacts": "num_associated_contacts"
        }

        # Value transformations for HubSpot - EXACT matches for HubSpot enum values
        property_value_mapping = {
            "club_type": {
                "Private": "Private",
                "Public": "Public",
                "Public - Low Daily Fee": "Public - Low Daily Fee",
                "Municipal": "Municipal",
                "Semi-Private": "Semi-Private",
                "Resort": "Resort",
                "Country Club": "Country Club",
                "Private Country Club": "Country Club",
                "Management Company": "Management Company",
                "Unknown": "Unknown"
            },
            "facility_complexity": {
                "Single-Course": "Standard",  # Changed from Basic to Standard
                "Multi-Course": "Multi-Course",
                "Resort": "Resort",
                "Unknown": "Unknown"
            },
            "geographic_seasonality": {
                "Year-Round Golf": "Year-Round",
                "Peak Summer Season": "Peak Summer Season",
                "Short Summer Season": "Short Summer Season",
                "Unknown": "Unknown"  # Default value
            }
        }

        # Clean and map the updates
        mapped_updates = {}
        for internal_key, value in confirmed_updates.items():
            hubspot_key = hubspot_property_mapping.get(internal_key)
            if not hubspot_key:
                logger.warning(f"No HubSpot mapping for property: {internal_key}")
                continue

            # Debug pre-transformation
            logger.debug(f"Pre-transform - Key: {internal_key}, Value: {value}, Type: {type(value)}")

            try:
                # Apply enum value transformations first
                if internal_key in property_value_mapping:
                    original_value = value
                    value = property_value_mapping[internal_key].get(str(value), value)
                    logger.debug(f"Enum transformation for {internal_key}: {original_value} -> {value}")

                # Type-specific handling
                if internal_key in ["number_of_holes", "season_start", "season_end", "peak_season_start_month", "peak_season_end_month",
                                   "notes_last_contacted", "num_contacted_notes", "num_associated_contacts"]:
                    original_value = value
                    value = int(value) if str(value).isdigit() else 0
                    logger.debug(f"Number conversion for {internal_key}: {original_value} -> {value}")
                
                elif internal_key in ["has_pool", "has_tennis_courts"]:
                    original_value = value
                    value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                    logger.debug(f"Boolean conversion for {internal_key}: {original_value} -> {value}")
                
                elif internal_key == "club_info":
                    original_length = len(str(value))
                    value = str(value)[:5000]
                    logger.debug(f"Text truncation for {internal_key}: {original_length} chars -> {len(value)} chars")

            except Exception as e:
                logger.error(f"Error transforming {internal_key}: {str(e)}")
                continue

            # Debug post-transformation
            logger.debug(f"Post-transform - Key: {hubspot_key}, Value: {value}, Type: {type(value)}")
            
            mapped_updates[hubspot_key] = value

        # Debug final payload
        logger.debug("Final HubSpot payload:")
        logger.debug(f"Company ID: {company_id}")
        logger.debug("Properties:")
        for key, value in mapped_updates.items():
            logger.debug(f"  {key}: {value} (Type: {type(value)})")

        # Send update to HubSpot with detailed error response
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        url = f"{hubspot.companies_endpoint}/{company_id}"
        payload = {"properties": mapped_updates}

        logger.info(f"Sending update to HubSpot: {payload}")
        try:
            response = hubspot._make_hubspot_patch(url, payload)
            if response:
                logger.info(f"Successfully updated company {company_id}")
                return True
            return False
        except HubSpotError as api_error:
            # Log the detailed error response from HubSpot
            logger.error(f"HubSpot API Error Details:")
            logger.error(f"Status Code: {getattr(api_error, 'status_code', 'Unknown')}")
            logger.error(f"Response Body: {getattr(api_error, 'response_body', 'Unknown')}")
            logger.error(f"Request Body: {payload}")
            raise

    except HubSpotError as e:
        logger.error(f"Error updating company properties: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating company properties: {str(e)}")
        logger.exception("Full traceback:")
        return False


def determine_seasonality(state: str) -> dict:
    """Determine golf seasonality based on state."""
    seasonality_map = {
        # Year-Round Golf States
        "FL": "Year-Round Golf",
        "AZ": "Year-Round Golf",
        "HI": "Year-Round Golf",
        "CA": "Year-Round Golf",
        
        # Short Summer Season States
        "MN": "Short Summer Season",
        "WI": "Short Summer Season",
        "MI": "Short Summer Season",
        "ME": "Short Summer Season",
        "VT": "Short Summer Season",
        "NH": "Short Summer Season",
        "MT": "Short Summer Season",
        "ND": "Short Summer Season",
        "SD": "Short Summer Season",
        
        # Peak Summer Season States (default)
        "default": "Peak Summer Season"
    }
    
    geography = seasonality_map.get(state, seasonality_map["default"])
    
    # Calculate season months
    outreach_window = get_best_outreach_window(
        persona="General Manager",
        geography=geography,
        club_type="Country Club"
    )
    
    best_months = outreach_window["Best Month"]
    return {
        "geographic_seasonality": geography,
        "start_month": min(best_months) if best_months else "",
        "end_month": max(best_months) if best_months else "",
        "peak_season_start_month": min(best_months) if best_months else "",
        "peak_season_end_month": max(best_months) if best_months else ""
    }


def process_company(company_id: str):
    print(f"\n=== Processing Company ID: {company_id} ===")

    (
        name,
        company_short_name,
        city,
        state,
        annual_revenue,
        create_date,
        last_modified,
        object_id,
        club_type,
        facility_complexity,
        has_pool,
        has_tennis_courts,
        number_of_holes,
        geographic_seasonality,
        public_private_flag,
        club_info,
    ) = get_facility_info(company_id)

    print(f"\nProcessing {name} in {city}, {state}")

    if name and state:
        club_info = determine_facility_type(name, state)
        confirmed_updates = {}

        # Update company name if different
        new_name = club_info.get("name")
        if new_name and new_name != name:
            print(f"Updating company name from '{name}' to '{new_name}'")  # Debug print
            confirmed_updates["name"] = new_name

        # Update with explicit string comparisons for boolean fields
        confirmed_updates.update({
            "name": club_info.get("name", name),
            "company_short_name": club_info.get("company_short_name", company_short_name),
            "club_type": club_info.get("club_type", "Unknown"),
            "facility_complexity": club_info.get("facility_complexity", "Unknown"),
            "geographic_seasonality": club_info.get("geographic_seasonality", "Unknown"),
            "has_pool": "Yes" if club_info.get("has_pool") == "Yes" else "No",
            "has_tennis_courts": "Yes" if club_info.get("has_tennis_courts") == "Yes" else "No",
            "number_of_holes": club_info.get("number_of_holes", 0),
            "public_private_flag": public_private_flag
        })

        new_club_info = club_info.get("club_info")
        if new_club_info:
            confirmed_updates["club_info"] = new_club_info

        # Get seasonality data
        season_data = determine_seasonality(state)  # Pass state code
        
        # Add seasonality to confirmed updates
        confirmed_updates.update({
            "geographic_seasonality": season_data["geographic_seasonality"],
            "season_start": season_data["start_month"],
            "season_end": season_data["end_month"],
            "peak_season_start_month": season_data["peak_season_start_month"],
            "peak_season_end_month": season_data["peak_season_end_month"]
        })

        success = update_company_properties(company_id, club_info, confirmed_updates)
        if success:
            print("✓ Successfully updated HubSpot properties")
        else:
            print("✗ Failed to update HubSpot properties")
    else:
        print("Unable to determine facility info - missing company name or location")


def _search_companies_with_filters(hubspot: HubspotService, batch_size=25) -> List[Dict[str, Any]]:
    """
    Search for companies in HubSpot that need club type enrichment.
    Processes one state at a time to avoid filter conflicts.
    """
    states = [
         # "AZ",  # Year-Round Golf
         # "GA",  # Year-Round Golf
        "FL",  # Year-Round Golf
        "MN",  # Short Summer Season
        "WI",  # Short Summer Season
        "MI",  # Short Summer Season
        "ME",  # Short Summer Season
        "VT",  # Short Summer Season
        # "NH",  # Short Summer Season
        # "MT",  # Short Summer Season
        # "ND",  # Short Summer Season
        # "SD"   # Short Summer Season
    ]
    all_results = []
    
    for state in states:
        logger.info(f"Searching for companies in {state}")
        url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
        after = None
        
        while True and (not TEST_MODE or len(all_results) < TEST_LIMIT):
            # Build request payload with single state filter
            payload = {
                "limit": min(batch_size, TEST_LIMIT) if TEST_MODE else batch_size,
                "properties": [
                    "name",
                    "company_short_name",
                    "city",
                    "state",
                    "club_type",
                    "annualrevenue",
                    "facility_complexity",
                    "geographic_seasonality",
                    "has_pool",
                    "has_tennis_courts",
                    "number_of_holes",
                    "public_private_flag",
                    "start_month",
                    "end_month",
                    "peak_season_start_month",
                    "peak_season_end_month",
                    "notes_last_contacted",
                    "num_contacted_notes",
                    "num_associated_contacts"
                ],
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "state",
                                "operator": "EQ",
                                "value": state
                            },
                            {
                                "propertyName": "club_type",
                                "operator": "NOT_HAS_PROPERTY",
                                "value": None
                            },
                            {
                                "propertyName": "annualrevenue",
                                "operator": "GTE", 
                                "value": "10000000"
                            }
                        ]
                    }
                ]
            }
            
            if after:
                payload["after"] = after

            try:
                logger.info(f"Fetching companies in {state} (Test Mode: {TEST_MODE})")
                response = hubspot._make_hubspot_post(url, payload)
                if not response:
                    break

                results = response.get("results", [])
                
                # Double-check state filter
                results = [
                    r for r in results 
                    if r.get("properties", {}).get("state") == state
                ]
                
                all_results.extend(results)
                
                logger.info(f"Retrieved {len(all_results)} total companies so far ({len(results)} from {state})")

                # Handle pagination
                paging = response.get("paging", {})
                next_link = paging.get("next", {}).get("after")
                if not next_link:
                    break
                after = next_link

                # Check if we've hit the test limit
                if TEST_MODE and len(all_results) >= TEST_LIMIT:
                    logger.info(f"Test mode: Reached limit of {TEST_LIMIT} companies")
                    break

            except Exception as e:
                logger.error(f"Error fetching companies from HubSpot for {state}: {str(e)}")
                break

        logger.info(f"Completed search for {state} - Found {len(all_results)} total companies")

    # Ensure we don't exceed test limit
    if TEST_MODE:
        all_results = all_results[:TEST_LIMIT]
        logger.info(f"Test mode: Returning {len(all_results)} companies total")

    return all_results


def main():
    """Main function to process companies needing enrichment."""
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        
        # Check if we're processing a single test company
        if TEST_COMPANY_ID:
            print(f"\n=== Processing Single Test Company: {TEST_COMPANY_ID} ===\n")
            process_company(TEST_COMPANY_ID)
            print("\n=== Completed processing test company ===")
            return
            
        # Regular batch processing
        companies = _search_companies_with_filters(hubspot)
        
        if not companies:
            print("No companies found needing enrichment")
            return
            
        print(f"\n=== Processing {len(companies)} companies ===\n")
        
        for i, company in enumerate(companies, 1):
            company_id = company.get("id")
            if not company_id:
                continue
                
            print(f"\nProcessing company {i} of {len(companies)}")
            process_company(company_id)
        
        print("\n=== Completed processing all companies ===")

    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()