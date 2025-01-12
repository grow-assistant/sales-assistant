#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import xai_club_info_search

def get_facility_info(company_id: str) -> tuple[str, str, str]:
    """
    Fetches the company's location from HubSpot and uses xAI to determine facility type.
    Returns tuple of (current_revenue, current_type, location)
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        company_data = hubspot.get_company_data(company_id)
        
        # Get current values and location
        current_revenue = company_data.get("annualrevenue", "")
        current_type = company_data.get("company_type", "")
        location = company_data.get("state", "")
        company_name = company_data.get("name", "")
        
        return (current_revenue, current_type, location, company_name)
    except HubSpotError as e:
        print(f"Error fetching company data: {e}")
        return ("", "", "", "")

def determine_facility_type(company_name: str, location: str) -> str:
    """
    Uses xAI to determine the facility type based on company info
    """
    if not company_name or not location:
        return ""
        
    # Use xAI integration to get club info
    club_info = xai_club_info_search(company_name, location)
    
    # Map xAI facility types to HubSpot accepted values
    facility_type_mapping = {
        'Private': 'Private Course',
        'Public': 'Public Course',
        'Municipal': 'Municipal Course',
        'Semi-Private': 'Semi-Private Course',
        'Resort': 'Resort Course',
        'Country Club': 'Private Course',  # Most country clubs are private courses
        'Management Company': 'Management Company'
    }
    
    # Extract facility type from response and map to HubSpot value
    raw_type = club_info.get('facility_type', 'Other')
    facility_type = facility_type_mapping.get(raw_type, 'Other')
    
    return facility_type

def update_company_properties(company_id: str, annual_revenue: str, company_type: str) -> bool:
    """
    Updates the company's annual revenue and company type properties in HubSpot.
    Returns True if update was successful, False otherwise.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        url = f"{hubspot.companies_endpoint}/{company_id}"
        payload = {
            "properties": {
                "annualrevenue": annual_revenue,
                "company_type": company_type
            }
        }
        hubspot._make_hubspot_patch(url, payload)
        return True
    except HubSpotError as e:
        print(f"Error updating company properties: {e}")
        return False

def main():
    print("=== HubSpot Company Properties Updater ===")
    company_id = input("Enter the HubSpot Company ID: ").strip()
    if not company_id:
        print("Company ID cannot be empty. Exiting.")
        return

    # 1. Fetch current properties and location
    current_revenue, current_type, location, company_name = get_facility_info(company_id)
    
    print(f"Current annual revenue for Company {company_id}: {current_revenue}")
    print(f"Current company type for Company {company_id}: {current_type}")
    print(f"Location: {location}")

    # 2. Use xAI to determine facility type
    if company_name and location:
        new_type = determine_facility_type(company_name, location)
        print(f"\nAnalyzed facility type: {new_type}")
    else:
        print("Unable to determine facility type - missing company name or location")
        return

    # 3. Prompt for the new revenue
    new_revenue = input("Enter NEW annual revenue: ").strip()
    if not new_revenue:
        print("No annual revenue entered. Using current value.")
        new_revenue = current_revenue

    # 4. Confirm facility type update
    print(f"\nProposed facility type: {new_type}")
    confirm = input("Update facility type? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Using current facility type instead.")
        new_type = current_type

    # 5. Attempt to update the properties
    success = update_company_properties(company_id, new_revenue, new_type)
    if success:
        print(f"\nSuccessfully updated Company {company_id} properties:")
        print(f"Annual Revenue: {new_revenue}")
        print(f"Company Type: {new_type}")
    else:
        print("Failed to update company properties.")

if __name__ == "__main__":
    main()
