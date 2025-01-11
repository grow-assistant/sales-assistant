#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError

def get_company_annual_revenue(company_id: str) -> str:
    """
    Fetches the company's current annual revenue from HubSpot.
    Returns the 'annualrevenue' property as a string (or empty if not set).
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        company_data = hubspot.get_company_data(company_id)
        return company_data.get("annualrevenue", "")
    except HubSpotError as e:
        print(f"Error fetching company data: {e}")
        return ""

def update_company_annual_revenue(company_id: str, new_value: str) -> bool:
    """
    Updates the company's annual revenue property in HubSpot.
    Returns True if update was successful, False otherwise.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        url = f"{hubspot.companies_endpoint}/{company_id}"
        payload = {
            "properties": {
                "annualrevenue": new_value
            }
        }
        hubspot._make_hubspot_patch(url, payload)  # We'll need to add this method
        return True
    except HubSpotError as e:
        print(f"Error updating annual revenue: {e}")
        return False

def main():
    print("=== HubSpot Annual Revenue Updater ===")
    company_id = input("Enter the HubSpot Company ID: ").strip()
    if not company_id:
        print("Company ID cannot be empty. Exiting.")
        return

    # 1. Fetch and display current annual revenue
    current_revenue = get_company_annual_revenue(company_id)
    if current_revenue == "":
        print(f"No annual revenue found or error accessing company {company_id}.")
    else:
        print(f"Current annual revenue for Company {company_id}: {current_revenue}")

    # 2. Prompt for the new annual revenue
    new_revenue = input("Enter NEW annual revenue: ").strip()
    if not new_revenue:
        print("No annual revenue entered. Exiting.")
        return

    # 3. Attempt to update the annual revenue
    success = update_company_annual_revenue(company_id, new_revenue)
    if success:
        print(f"Successfully updated Company {company_id}'s annual revenue to: {new_revenue}")
    else:
        print("Failed to update annual revenue.")

if __name__ == "__main__":
    main()
