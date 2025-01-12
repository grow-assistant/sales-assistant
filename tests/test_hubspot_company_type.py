# tests/test_hubspot_company_type.py

import sys
import time
from pathlib import Path

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


def main():
    company_ids = [
        "15537398095", "15537380786", "15537395569", "15537388563", "15537370228", "15537390600", "15537388565", "15537368244",
        "15537401414", "15537395568", "15537370227", "15537398094", "15537350070", "15537380785", "15537406736", "15537470002",
        "15537350072", "15537458854", "15537370224", "15537350068", "15537386157", "15537368245", "15537375516", "15537388564"
    ]

    print(f"\n=== Processing {len(company_ids)} companies ===\n")
    for i, company_id in enumerate(company_ids, 1):
        print(f"\nProcessing company {i} of {len(company_ids)}")
        process_company(company_id)

        # Don't sleep after the last company
        if i < len(company_ids):
            print("\nWaiting 5 seconds before next company...")
            time.sleep(5)

    print("\n=== Completed processing all companies ===")


if __name__ == "__main__":
    main()