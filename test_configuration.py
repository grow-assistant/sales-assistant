from services.leads_service import LeadsService
from services.orchestrator_service import OrchestratorService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY

def test_configuration():
    try:
        # Initialize services with proper dependencies
        data_gatherer = DataGathererService()
        leads_service = LeadsService(data_gatherer)
        hubspot_service = HubspotService(api_key=HUBSPOT_API_KEY)
        orchestrator = OrchestratorService(leads_service, hubspot_service)
        
        # Test with our test email
        email = 'smoran@shorthillsclub.org'
        logger.info(f'Testing with email: {email}')
        
        # Test lead context preparation
        lead_context = leads_service.prepare_lead_context(email)
        assert lead_context is not None, "Lead context should not be None"
        logger.info('Lead context prepared successfully')
        
        # Verify lead context structure
        assert "metadata" in lead_context, "Lead context missing metadata"
        assert "lead_data" in lead_context, "Lead context missing lead_data"
        assert "analysis" in lead_context, "Lead context missing analysis"
        
        # Test personalization
        message = orchestrator.personalize_message(lead_context)
        assert message is not None, "Personalized message should not be None"
        assert len(message) > 0, "Personalized message should not be empty"
        
        logger.info('Message personalization successful')
        print('All tests passed successfully!')
        return True
    except Exception as e:
        logger.error(f'Test failed: {str(e)}')
        raise

if __name__ == '__main__':
    test_configuration()
