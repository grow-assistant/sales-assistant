import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection


def test_lead_data():
    # Get API key from environment variable
    api_key = os.getenv('HUBSPOT_API_KEY')
    if not api_key:
        print("Please set HUBSPOT_API_KEY environment variable")
        return

    # Initialize HubSpot service
    hubspot = HubspotService(api_key)

    # Test email - replace with your actual email
    test_email = "wmchenry@flyinghorseclub.com"  # Replace with actual email

    try:
        # First get the contact by email
        logger.info(f"Fetching contact data for email: {test_email}")
        contact = hubspot.get_contact_by_email(test_email)
        
        if not contact:
            print(f"No contact found for email: {test_email}")
            return

        contact_id = contact.get('id')
        
        # Get associated company ID
        print("\nAssociated Company Data:")
        print("-" * 50)
        company_id = hubspot.get_associated_company_id(contact_id)
        
        if company_id:
            print(f"Associated Company ID: {company_id}")
            # If you want to fetch company details
            company_data = hubspot.get_company_data(company_id)
            print("\nCompany Details:")
            company_fields = [
                'name',
                'company_short_name',
                'city',
                'state',
                'club_type',
                'facility_complexity',
                'has_pool',
                'has_tennis_courts',
                'number_of_holes',
                'geographic_seasonality',
                'public_private_flag',
                'club_info'
            ]
            for field in company_fields:
                value = company_data.get(field, 'Not found')
                print(f"{field}: {value}")
        else:
            print("No associated company found")

        # Fetch contact properties
        contact_data = hubspot.get_contact_properties(contact_id)

        # Print results
        print("\nLead/Contact Data Results:")
        print("-" * 50)
        
        # Print specific fields we're interested in
        key_fields = [
            "email", 
            "jobtitle", 
            "lifecyclestage", 
            "phone",
            "hs_sales_email_last_replied", 
            "firstname", 
            "lastname"
        ]

        for field in key_fields:
            value = contact_data.get(field, 'Not found')
            print(f"{field}: {value}")


    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_lead_data() 