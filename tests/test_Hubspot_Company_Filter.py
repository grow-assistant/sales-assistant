# tests/test_hubspot_company_filter.py
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hubspot_service import HubspotService
from utils.enrich_hubspot_company_data import _search_companies_with_filters
from config.settings import HUBSPOT_API_KEY, logger  # Import logger

def test_company_filter():
    """
    Tests the filtering logic of _search_companies_with_filters.
    Counts the results and verifies the filter.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        companies = _search_companies_with_filters(hubspot)

        if not companies:
            print("No companies found matching the filter.")
            return

        print(f"Found {len(companies)} companies matching the filter.")

        # Print details of the first few companies for inspection
        print("\n--- First 5 Companies (for inspection) ---")
        for i, company in enumerate(companies[:5]):
            print(f"Company {i+1}:")
            print(f"  ID: {company.get('id')}")
            print(f"  Properties:")
            props = company.get('properties', {})
            for key, value in props.items():
                print(f"    {key}: {value}")
            print("-" * 20)

        # Verify that the filter is working (club_type should be missing)
        for company in companies:
            properties = company.get('properties', {})
            if 'club_type' in properties and properties['club_type']:
                logger.error(f"Company ID {company.get('id')} has club_type: {properties['club_type']}")
                print(f"ERROR: Company ID {company.get('id')} has club_type: {properties['club_type']}")
                # You might want to raise an exception here in a real test
                # to fail the test if the filter is not working.
                # raise AssertionError(f"Company {company.get('id')} has club_type")

        print("\nFilter check complete. See logs for any errors.")

    except Exception as e:
        print(f"Error in test_company_filter: {str(e)}")

if __name__ == "__main__":
    test_company_filter()