from services.leads_service import LeadsService
from services.orchestrator_service import OrchestratorService
from utils.logging_setup import logger

def test_configuration():
    try:
        leads_service = LeadsService()
        orchestrator = OrchestratorService()
        
        # Test with our test email
        email = 'smoran@shorthillsclub.org'
        logger.info(f'Testing with email: {email}')
        
        # Test lead context preparation
        lead_context = leads_service.prepare_lead_context(email)
        logger.info('Lead context prepared successfully')
        
        # Test personalization
        if lead_context:
            message = orchestrator.personalize_message(lead_context)
            logger.info('Message personalization successful')
            print('All tests passed successfully!')
            return True
        else:
            logger.error('Failed to prepare lead context')
            return False
    except Exception as e:
        logger.error(f'Test failed: {str(e)}')
        raise

if __name__ == '__main__':
    test_configuration()
