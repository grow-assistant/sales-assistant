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
    xai_club_info_search,
    get_club_summary
)
from utils.logging_setup import logger

# Add these constants after imports
###########################
# CONFIG / CONSTANTS
###########################
TEST_MODE = False  # Set to False for production
TEST_LIMIT = 3    # Number of companies to process in test mode
BATCH_SIZE = 25   # Companies per API request


def get_facility_info(company_id: str) -> tuple[
    str, str, str, str, str, str, str, str, str, str, str, int, str, str, str
]:
    """
    Fetches the company's properties from HubSpot.
    Returns a tuple of company properties including club_info.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        company_data = hubspot.get_company_data(company_id)

        name = company_data.get("name", "")
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
        return ("", "", "", "", "", "", "", "", "", "No", "No", 0, "", "", "")


def determine_facility_type(company_name: str, location: str) -> dict:
    """
    Uses xAI to determine the facility type and official name based on company info.
    """
    if not company_name or not location:
        return {}

    club_info = xai_club_info_search(company_name, location)
    segmentation_info = xai_club_segmentation_search(company_name, location)
    club_summary = get_club_summary(company_name, location)

    full_info = {
        "name": club_info.get("official_name", company_name),
        "club_type": segmentation_info.get("club_type", "Unknown"),
        "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
        "geographic_seasonality": segmentation_info.get(
            "geographic_seasonality", "Unknown"
        ),
        "has_pool": segmentation_info.get("has_pool", "Unknown"),
        "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
        "number_of_holes": segmentation_info.get("number_of_holes", 0),
        "club_info": club_summary,
    }

    return full_info


def update_company_properties(company_id: str, club_info: dict, confirmed_updates: dict) -> bool:
    """
    Updates the company's properties in HubSpot based on club segmentation info.
    """
    try:
        property_value_mapping = {
            "name": str,  # No mapping needed for name
            "club_type": {
                "Private": "Private Course",
                "Public": "Public Course",
                "Municipal": "Municipal Course",
                "Semi-Private": "Semi-Private Course",
                "Resort": "Resort Course",
                "Country Club": "Country Club",
                "Private Country Club": "Country Club",
                "Management Company": "Management Company",
                "Unknown": "Unknown",
            },
            "public_private_flag": {
                "Private": "Private",
                "Public": "Public",
                "Unknown": "Unknown",
            },
            "has_pool": {True: "Yes", False: "No", "Unknown": "Unknown"},
            "has_tennis_courts": {True: "Yes", False: "No", "Unknown": "Unknown"},
        }

        mapped_updates = {}
        for key, value in confirmed_updates.items():
            if key in property_value_mapping:
                # If there's a dictionary mapping for this property
                if isinstance(property_value_mapping[key], dict):
                    mapped_updates[key] = property_value_mapping[key].get(value, value)
                else:
                    mapped_updates[key] = value
            else:
                mapped_updates[key] = value

        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        url = f"{hubspot.companies_endpoint}/{company_id}"
        payload = {"properties": mapped_updates}

        print(f"Sending update to HubSpot: {payload}")  # Debug print
        hubspot._make_hubspot_patch(url, payload)
        return True

    except HubSpotError as e:
        print(f"Error updating company properties: {e}")
        return False


def process_company(company_id: str):
    print(f"\n=== Processing Company ID: {company_id} ===")

    (
        name,
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
            confirmed_updates["name"] = new_name

        # Update with explicit string comparisons for boolean fields
        confirmed_updates.update(
            {
                "club_type": club_info.get("club_type", "Unknown"),
                "facility_complexity": club_info.get("facility_complexity", "Unknown"),
                "geographic_seasonality": club_info.get(
                    "geographic_seasonality", "Unknown"
                ),
                # Compare strings directly
                "has_pool": "Yes" if club_info.get("has_pool") == "Yes" else "No",
                "has_tennis_courts": "Yes" if club_info.get("has_tennis_courts") == "Yes" else "No",
                "number_of_holes": club_info.get("number_of_holes", 0),
            }
        )

        new_club_info = club_info.get("club_info")
        if new_club_info:
            confirmed_updates["club_info"] = new_club_info

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
                    "city", 
                    "state", 
                    "club_type",
                    "annualrevenue",
                    "facility_complexity",
                    "geographic_seasonality",
                    "has_pool",
                    "has_tennis_courts",
                    "number_of_holes",
                    "public_private_flag"
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
        
        # Get companies that need enrichment
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
            
            # # Don't sleep after the last company
            # if i < len(companies):
            #     print("\nWaiting 5 seconds before next company...")
            #     time.sleep(5)
        
        print("\n=== Completed processing all companies ===")

    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()