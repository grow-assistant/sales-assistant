import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection


def test_company_data():
    # Get API key from environment variable
    api_key = os.getenv('HUBSPOT_API_KEY')
    if not api_key:
        print("Please set HUBSPOT_API_KEY environment variable")
        return

    # Initialize HubSpot service
    hubspot = HubspotService(api_key)

    # Test company ID - replace with your actual company ID
    company_id = "6627440825"  # Replace with actual company ID

    try:
        # Fetch company data
        logger.info(f"Fetching data for company ID: {company_id}")
        company_data = hubspot.get_company_data(company_id)

        # Print results
        print("\nCompany Data Results:")
        print("-" * 50)
        
        # Print specific fields we're interested in
        key_fields = [
            'name',
            'company_short_name',
            'city',
            'state',
            'address_state',
            'club_type',
            'facility_complexity'
        ]

        for field in key_fields:
            value = company_data.get(field, 'Not found')
            print(f"{field}: {value}")

        # Print all properties for debugging
        print("\nAll Properties:")
        print("-" * 50)
        for key, value in company_data.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_company_data() 