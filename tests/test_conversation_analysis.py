import asyncio
from datetime import datetime
from services.conversation_analysis_service import ConversationAnalysisService
from utils.logging_setup import logger


async def test_conversation_analysis():
    """Test the ConversationAnalysisService with a sample email."""
    
    # Test email address
    test_email = "bryanj@standardclub.org"
    
    try:
        # Initialize the service
        logger.info(f"Testing conversation analysis for {test_email}")
        analysis_service = ConversationAnalysisService()
        
        # Get the summary
        start_time = datetime.now()
        summary = analysis_service.analyze_conversation(test_email)
        duration = datetime.now() - start_time
        
        # Log results
        logger.info(f"Analysis completed in {duration.total_seconds():.2f} seconds")
        
        # Print the summary
        print("\nConversation Analysis Results:")
        print("=" * 50)
        print(summary)
        print("\n" + "=" * 50)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error testing conversation analysis: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_conversation_analysis()) 