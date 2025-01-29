# test_hubspot_leads_service.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def get_random_lead_id():
    """Get a random lead_id from the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Modified query to ensure we get a valid HubSpot contact ID
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY NEWID()
            """)
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database: {lead_id}")
                return lead_id
            logger.warning("No valid lead_id found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting random lead_id: {str(e)}")
        return None

def get_lead_id_for_email(email):
    """Get lead_id for a specific email address."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.email_address = ?
                  AND e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
            """, (email,))
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database for {email}: {lead_id}")
                return lead_id
            return None
    except Exception as e:
        logger.error(f"Error getting lead_id for email {email}: {str(e)}")
        return None

def test_lead_info():
    """Test function to pull HubSpot data for a random lead ID."""
    try:
        print("\n=== Starting test_lead_info() ===")
        
        # Initialize services in correct order
        print("\nInitializing services...")
        data_gatherer = DataGathererService()  # Initialize without parameters
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service  # Set the service after initialization
        leads_service = LeadsService(data_gatherer)
        print("Services initialized successfully")
        
        # Get random contact ID from database
        print("\nFetching random contact ID...")
        contact_id = get_random_lead_id()
        print(f"Retrieved contact_id: {contact_id}")
        
        if not contact_id:
            print("No lead IDs found in database")
            return
            
        print(f"\nFetching info for contact ID: {contact_id}")
        
        # Verify contact exists in HubSpot before proceeding
        try:
            print("\nFetching contact properties from HubSpot...")
            contact_props = hubspot_service.get_contact_properties(contact_id)
            print(f"Contact properties retrieved: {bool(contact_props)}")
            
            if not contact_props:
                print(f"Contact ID {contact_id} not found in HubSpot")
                return
            
            print("\nContact Properties:")
            print("-" * 50)
            for key, value in contact_props.items():
                print(f"{key}: {value}")
                
            # Get company ID and state
            company_id = contact_props.get('associatedcompanyid')
            print(f"\nDEBUG: Found company_id: {company_id}")
            company_props = None
            
            if company_id:
                # Add debug logging for company lookup
                print(f"\nDEBUG: About to look up company data for ID: {company_id}")
                logger.debug(f"Looking up company properties for ID: {company_id}")
                company_props = hubspot_service.get_company_data(company_id)
                print(f"\nDEBUG: Retrieved company_props: {bool(company_props)}")
                
                print("\nCompany Data Results:")
                print("-" * 50)
                
                # Match exactly the key_fields from the working example
                key_fields = [
                    'name',
                    'company_short_name',
                    'city',
                    'state',
                    'address_state',
                    'club_type',
                    'facility_complexity'
                ]
                
                # Print all key fields with their values
                for field in key_fields:
                    value = company_props.get(field, 'Not found')
                    print(f"{field}: {value}")
                
                # Print all properties for debugging
                print("\nAll Company Properties:")
                print("-" * 50)
                for key, value in company_props.items():
                    print(f"{key}: {value}")
                    
                # Get state after seeing all properties
                company_state = company_props.get('state')
                print(f"\nDEBUG: Final company_state value: {company_state}")
            else:
                print("\nDEBUG: No company_id found in contact properties")
                logger.warning(f"No associated company ID found in contact properties")
                company_state = None
                
        except Exception as e:
            print(f"Error verifying contact in HubSpot: {str(e)}")
            logger.error(f"HubSpot verification error: {str(e)}", exc_info=True)
            return
            
        # Get lead summary using LeadsService
        print("\nFetching lead summary...")
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error in lead summary: {lead_info['error']}")
            return
            
        # Print results
        print("\nFinal Lead Information:")
        print("=" * 50)
        print(f"Last Reply Date: {lead_info['last_reply_date']}")
        print(f"Lifecycle Stage: {lead_info['lifecycle_stage']}")
        print(f"Company Name: {lead_info['company_name']}")
        print(f"Company Short Name: {lead_info['company_short_name']}")
        print(f"Company State: {company_state}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_lead_info: {str(e)}", exc_info=True)

def test_specific_company():
    """Test function to pull HubSpot data for Flying Horse Club."""
    try:
        # Initialize HubSpot service
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        
        # Use the known working company ID
        company_id = "6627440825"  # Flying Horse Club ID
        
        print(f"\nFetching company data for ID: {company_id}")
        company_data = hubspot_service.get_company_data(company_id)
        
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
        print("\nAll Company Properties:")
        print("-" * 50)
        for key, value in company_data.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(str(e), exc_info=True)

def test_lead_data():
    # Initialize HubSpot service
    hubspot = HubspotService(HUBSPOT_API_KEY)

    # Test email address
    email = "wmchenry@flyinghorseclub.com"

    try:
        # Fetch lead data
        logger.info(f"Fetching data for email: {email}")
        lead_data = hubspot.gather_lead_data(email)

        # Print results
        print("\nLead Data Results:")
        print("-" * 50)
        
        # Print contact properties
        print("\nContact Properties:")
        print("-" * 30)
        for key, value in lead_data['properties'].items():
            print(f"{key}: {value}")

        # Print company data
        print("\nCompany Data:")
        print("-" * 30)
        for key, value in lead_data['company_data'].items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    print("\n=== Starting Test Script ===")
    test_specific_company()
    test_lead_data()


