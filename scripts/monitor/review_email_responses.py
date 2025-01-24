import sys
import os
# Add this at the top of the file, before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY

def process_invalid_email(email: str, analyzer_result: Dict) -> None:
    """
    Process an invalid email by removing the contact from HubSpot.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get the contact
        contact = hubspot.get_contact_by_email(email)
        if not contact:
            logger.info(f"Contact not found in HubSpot for email: {email}")
            return

        contact_id = contact.get('id')
        if not contact_id:
            logger.error(f"Contact found but missing ID for email: {email}")
            return

        # Archive the contact in HubSpot
        try:
            hubspot.archive_contact(contact_id)
            logger.info(f"Successfully archived contact {email} (ID: {contact_id}) due to invalid email: {analyzer_result['message']}")
        except Exception as e:
            logger.error(f"Failed to archive contact {email}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing invalid email {email}: {str(e)}")

def is_out_of_office(message: str, subject: str) -> bool:
    """
    Check if a message is an out-of-office response.
    """
    ooo_phrases = [
        "out of office",
        "automatic reply",
        "away from",
        "will be out",
        "on vacation",
        "annual leave",
        "business trip",
        "return to the office",
        "be back",
        "currently away"
    ]
    
    # Check both subject and message body for OOO indicators
    message_lower = message.lower()
    subject_lower = subject.lower()
    
    return any(phrase in message_lower or phrase in subject_lower for phrase in ooo_phrases)

def main():
    """
    Main function to analyze email responses and handle invalid emails.
    """
    analyzer = ResponseAnalyzerService()
    
    while True:
        email = input("\nEnter email address to analyze (or 'quit' to exit): ")
        
        if email.lower() == 'quit':
            break
            
        result = analyzer.analyze_response_status(email)
        
        print("\nAnalysis Results:")
        print("-" * 50)
        
        if result["status"] == "BOUNCED":
            print(f"Status: {result['status']}")
            print(f"Type: {result['response_type']}")
            print(f"Message: {result['message'][:200]}...")
            
            confirm = input("\nThis appears to be an invalid email. Would you like to remove this contact from HubSpot? (y/n): ")
            if confirm.lower() == 'y':
                process_invalid_email(email, result)
                print(f"Contact with email {email} has been archived in HubSpot.")
            else:
                print("Contact removal cancelled.")
        
        elif result["status"] == "RESPONSE_FOUND":
            # Check for out-of-office responses first
            if is_out_of_office(result["message"], result.get("subject", "")):
                print("Response Type: OUT_OF_OFFICE")
                print(f"Timestamp: {result['timestamp']}")
                print(f"Subject: {result['subject']}")
                print("\nMessage Preview:")
                print(result["message"][:200] + "..." if len(result["message"]) > 200 else result["message"])
                print("\nNote: This is an out-of-office auto-response.")
            else:
                print(f"Response Type: {result['response_type']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Timestamp: {result['timestamp']}")
                print(f"Subject: {result['subject']}")
                print("\nMessage Preview:")
                print(result["message"][:200] + "..." if len(result["message"]) > 200 else result["message"])
                
                # Handle left company responses
                if result["response_type"] == "LEFT_COMPANY":
                    confirm = input("\nThis contact appears to have left the company. Would you like to remove them from HubSpot? (y/n): ")
                    if confirm.lower() == 'y':
                        process_invalid_email(email, result)
                        print(f"Contact with email {email} has been archived in HubSpot.")
                    else:
                        print("Contact removal cancelled.")
        else:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
