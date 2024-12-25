from services.hubspot_service import HubspotService
import logging
from config.settings import HUBSPOT_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hubspot_endpoints():
    if not HUBSPOT_API_KEY:
        logger.error("HUBSPOT_API_KEY environment variable not found")
        return

    hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
    
    try:
        # Test get_hubspot_leads
        logger.info("Testing get_hubspot_leads...")
        leads = hubspot.get_hubspot_leads()
        logger.info(f"Successfully retrieved {len(leads)} leads")
        
        if leads:
            test_email = leads[0].get('email')
            if test_email:
                # Test get_lead_data_from_hubspot
                logger.info(f"Testing get_lead_data_from_hubspot for {test_email}...")
                lead_data = hubspot.get_lead_data_from_hubspot(test_email)
                logger.info("Successfully retrieved lead data")
                
                # Test get_associated_company_id
                logger.info("Testing get_associated_company_id...")
                company_id = hubspot.get_associated_company_id(lead_data['contact_id'])
                logger.info(f"Successfully retrieved company ID: {company_id}")
                
                # Test get_company_data
                if company_id:
                    logger.info("Testing get_company_data...")
                    company_data = hubspot.get_company_data(company_id)
                    logger.info("Successfully retrieved company data")
    
    except Exception as e:
        logger.error(f"Error testing HubSpot endpoints: {str(e)}")
        raise

if __name__ == "__main__":
    test_hubspot_endpoints()
