import sys
import os
# Add this at the top of the file, before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from services.data_gatherer_service import DataGathererService
from scheduling.database import get_db_connection

# Define bounce query constant
BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'

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

def process_bounced_email(email: str, gmail_id: str, analyzer_result: Dict) -> None:
    """
    Process a bounced email by deleting from SQL, HubSpot, and archiving the email.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        gmail = GmailService()
        
        # First delete from SQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Delete from emails table - note the % placeholder for SQL Server
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email,))
            # Delete from leads table
            cursor.execute("DELETE FROM leads WHERE email = %s", (email,))
            conn.commit()
            logger.info(f"Deleted records for {email} from SQL database")
        except Exception as e:
            logger.error(f"Error deleting from SQL: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        # Rest of the existing HubSpot and Gmail processing...
        contact = hubspot.get_contact_by_email(email)
        if contact:
            contact_id = contact.get('id')
            if contact_id and hubspot.delete_contact(contact_id):
                logger.info(f"Contact {email} deleted from HubSpot")
        
        # Archive bounce notification
        if gmail.archive_email(gmail_id):
            logger.info(f"Bounce notification archived in Gmail")

    except Exception as e:
        logger.error(f"Error processing bounced email {email}: {str(e)}")

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

def delete_email_from_database(email_address):
    """Helper function to delete email records from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        delete_query = "DELETE FROM dbo.emails WHERE email_address = ?"
        cursor.execute(delete_query, (email_address,))
        conn.commit()
        logger.info(f"Deleted all records for {email_address} from emails table")
    except Exception as e:
        logger.error(f"Error deleting from SQL: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def process_bounce_notification(notification, gmail_service):
    """Process a single bounce notification"""
    bounced_email = notification['bounced_email']
    message_id = notification['message_id']
    logger.info(f"Processing bounce notification - Email: {bounced_email}, Message ID: {message_id}")
    
    # Delete from database
    delete_email_from_database(bounced_email)
    
    try:
        # Archive the Gmail message with explicit error handling
        try:
            logger.debug(f"Attempting to archive message {message_id}")
            gmail_service.archive_email(message_id)
            logger.info(f"Successfully archived bounce notification for {bounced_email} (Message ID: {message_id})")
        except Exception as e:
            logger.error(f"Failed to archive Gmail message {message_id}: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error processing bounce notification for {bounced_email}: {str(e)}")

def process_bounce_notifications():
    """Main function to process all bounce notifications"""
    logger.info("Starting bounce notification processing...")
    logger.info("Searching for bounce notifications in inbox...")
    
    # Create Gmail service instance directly
    gmail_service = GmailService()
    
    bounce_notifications = gmail_service.get_all_bounce_notifications(inbox_only=True)
    logger.debug(f"Gmail API query used: {BOUNCE_QUERY}")
    logger.debug(f"Raw API response: {bounce_notifications}")
    
    if not bounce_notifications:
        logger.info("No bounce notifications found in inbox.")
        return
        
    logger.info(f"Found {len(bounce_notifications)} bounce notifications to process in inbox.")
    
    for notification in bounce_notifications:
        process_bounce_notification(notification, gmail_service)

if __name__ == "__main__":
    process_bounce_notifications()
