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
        # Test gather_lead_data
        test_email = "smoran@shorthillsclub.org"
        logger.info(f"Testing gather_lead_data for {test_email}...")
        lead_data = hubspot.gather_lead_data(test_email)
        
        # Verify lead data structure
        assert "id" in lead_data, "Lead data missing contact ID"
        assert "properties" in lead_data, "Lead data missing properties"
        assert "emails" in lead_data, "Lead data missing emails"
        logger.info("Successfully retrieved and validated lead data")
    
    except Exception as e:
        logger.error(f"Error testing HubSpot endpoints: {str(e)}")
        raise

if __name__ == "__main__":
    test_hubspot_endpoints()
