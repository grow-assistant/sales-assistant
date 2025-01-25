import sys
import os
import re
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

# Define queries for different types of notifications
BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'
AUTO_REPLY_QUERY = '(subject:"No longer employed" OR subject:"out of office" OR subject:"automatic reply") in:inbox'

# Add at the top of the file with other constants
TESTING = False  # Set to False for production

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
        # if gmail.archive_email(gmail_id):
        #     logger.info(f"Bounce notification archived in Gmail")
        print(f"Would have archived bounce notification with ID: {gmail_id}")

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

def is_no_longer_employed(message: str, subject: str) -> bool:
    """Check if message indicates person is no longer employed."""
    employment_phrases = [
        "no longer employed",
        "no longer with",
        "is not employed",
        "has left",
        "no longer works",
        "no longer at",
        "no longer associated",
        "please remove",
        "has departed"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    
    return any(phrase in message_lower or phrase in subject_lower for phrase in employment_phrases)

def extract_new_contact_email(message: str) -> str:
    """Extract new contact email from message."""
    # Common patterns for replacement emails
    patterns = [
        r'please\s+(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:new\s+email|new\s+contact|instead\s+email)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'email\s+([\w\.-]+@[\w\.-]+\.\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1)
    return None

def process_employment_change(email: str, new_email: str, message_id: str) -> None:
    """Process employment change specifically for jim@chickasawcc.com"""
    # Only process for this specific email
    if email.lower() != "jim@chickasawcc.com":
        return
        
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        gmail = GmailService()
        
        logger.info(f"Processing employment change for {email}")
        
        # Get original contact details
        old_contact = hubspot.get_contact_by_email(email)
        if not old_contact:
            logger.warning(f"Original contact not found in HubSpot: {email}")
            return

        # Create new contact with same properties
        new_email = "news@chickasawcc.com"  # Hardcode new email
        try:
            # Get properties from old contact
            properties = old_contact.get('properties', {})
            new_properties = {
                'email': new_email,
                'firstname': properties.get('firstname', ''),
                'lastname': properties.get('lastname', ''),
                'company': 'Chickasaw Country Club',
                'jobtitle': properties.get('jobtitle', ''),
                'phone': properties.get('phone', ''),
                'website': properties.get('website', '')
            }
            
            if TESTING:
                print(f"Would have created new contact in HubSpot: {new_email}")
            else:
                # Create new contact
                new_contact = hubspot.create_contact(new_properties)
                if new_contact:
                    logger.info(f"Created new contact in HubSpot: {new_email}")
                    
                    # Copy associations
                    old_associations = hubspot.get_contact_associations(old_contact['id'])
                    for association in old_associations:
                        hubspot.create_association(new_contact['id'], association['id'], association['type'])
                        logger.info(f"Copied association to new contact")
                
        except Exception as e:
            logger.error(f"Error creating new contact {new_email}: {str(e)}")
            return
        
        if TESTING:
            print(f"Would have deleted contact {email} with ID {old_contact.get('id')}")
            print(f"Would have deleted {email} from SQL database")
            print(f"Would have archived employment change notification with ID: {message_id}")
        else:
            # Delete old contact
            if old_contact.get('id'):
                if hubspot.delete_contact(old_contact['id']):
                    logger.info(f"Deleted old contact from HubSpot: {email}")
            
            # Delete from SQL database
            delete_email_from_database(email)
            
            # Archive the notification
            if gmail.archive_email(message_id):
                logger.info(f"Archived employment change notification")
            
    except Exception as e:
        logger.error(f"Error processing employment change: {str(e)}")

def is_inactive_email(message: str, subject: str) -> bool:
    """Check if message indicates email is inactive."""
    inactive_phrases = [
        "email is now inactive",
        "email is inactive",
        "no longer active",
        "this address is inactive",
        "email account is inactive",
        "account is inactive",
        "inactive email",
        "email has been deactivated"
    ]
    
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in inactive_phrases)

def process_email_response(message_id: str, email: str, subject: str, body: str, gmail_service: GmailService) -> None:
    """Process an email response."""
    try:
        # Extract full name from email headers
        message_data = gmail_service.get_message(message_id)
        from_header = gmail_service._get_header(message_data, 'from')
        full_name = from_header.split('<')[0].strip()
        
        hubspot = HubspotService(HUBSPOT_API_KEY)
        logger.info(f"Processing response for email: {email}")
        
        # Get contact directly
        contact = hubspot.get_contact_by_email(email)
        
        if contact:
            contact_id = contact.get('id')
            contact_email = contact.get('properties', {}).get('email', '')
            logger.info(f"Found contact in HubSpot: {contact_email} (ID: {contact_id})")
            
            # Check if it's a "no longer employed" message
            if "no longer employed" in subject.lower():
                process_employment_change(contact_email, None, message_id)
            # Check if it's an auto-reply or inactive email
            elif any(phrase in subject.lower() for phrase in ["automatic reply", "out of office"]) or is_inactive_email(body, subject):
                # Process bounce notification to mark as bounced and delete
                notification = {
                    "bounced_email": contact_email,
                    "message_id": message_id
                }
                process_bounce_notification(notification, gmail_service)
                
            logger.info(f"Processed response for contact: {contact_email}")
        else:
            logger.warning(f"No contact found in HubSpot for email: {email}")
            
    except Exception as e:
        logger.error(f"Error processing email response: {str(e)}")

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
    if TESTING:
        print(f"Would have deleted {bounced_email} from SQL database")
    else:
        delete_email_from_database(bounced_email)
    
    try:
        # Delete from HubSpot with API key
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        contact = hubspot.get_contact_by_email(bounced_email)
        if contact:
            contact_id = contact.get('id')
            if contact_id:
                logger.info(f"Attempting to delete contact {bounced_email} from HubSpot")
                if TESTING:
                    print(f"Would have deleted contact {bounced_email} with ID {contact_id}")
                else:
                    if hubspot.delete_contact(contact_id):
                        logger.info(f"Successfully deleted contact {bounced_email} from HubSpot")
                    else:
                        logger.error(f"Failed to delete contact {bounced_email} from HubSpot")
            else:
                logger.warning(f"Contact found but no ID for {bounced_email}")
        else:
            logger.warning(f"No contact found in HubSpot for {bounced_email}")
        
        # Archive the Gmail message
        logger.debug(f"Attempting to archive message {message_id}")
        if TESTING:
            print(f"Would have archived bounce notification with ID: {message_id}")
        else:
            if gmail_service.archive_email(message_id):
                logger.info(f"Successfully archived bounce notification for {bounced_email}")
            else:
                logger.error(f"Failed to archive Gmail message {message_id}")
            
    except Exception as e:
        logger.error(f"Error processing bounce notification for {bounced_email}: {str(e)}")
        raise

def verify_processing(email: str, gmail_service: GmailService, hubspot_service: HubspotService) -> bool:
    """Verify that the contact was properly deleted and emails archived."""
    if TESTING:
        print(f"TESTING: Skipping verification for {email}")
        return True
        
    success = True
    
    # Check if contact still exists in HubSpot
    contact = hubspot_service.get_contact_by_email(email)
    if contact:
        logger.error(f"❌ Verification failed: Contact {email} still exists in HubSpot")
        success = False
    else:
        logger.info(f"✅ Verification passed: Contact {email} successfully deleted from HubSpot")
    
    # Check for any remaining emails in inbox
    query = f"from:{email} in:inbox"
    remaining_emails = gmail_service.search_messages(query)
    if remaining_emails:
        logger.error(f"❌ Verification failed: Found {len(remaining_emails)} unarchived emails from {email}")
        success = False
    else:
        logger.info(f"✅ Verification passed: No remaining emails from {email} in inbox")
    
    return success

def process_bounce_notifications(target_email: str = None):
    """
    Main function to process all bounce notifications and auto-replies.
    
    Args:
        target_email (str, optional): If provided, only process this specific email.
    """
    logger.info(f"Starting bounce notification and auto-reply processing...{' for ' + target_email if target_email else ''}")
    
    gmail_service = GmailService()
    hubspot_service = HubspotService(HUBSPOT_API_KEY)
    processed_emails = set()  # Track processed emails for verification
    
    # Process bounce notifications
    logger.info("Searching for bounce notifications in inbox...")
    bounce_notifications = gmail_service.get_all_bounce_notifications(inbox_only=True)
    
    if bounce_notifications:
        logger.info(f"Found {len(bounce_notifications)} bounce notifications to process.")
        for notification in bounce_notifications:
            email = notification.get('bounced_email')
            if email and (not target_email or email == target_email):
                process_bounce_notification(notification, gmail_service)
                processed_emails.add(email)
    else:
        logger.info("No bounce notifications found in inbox.")
    
    # Process auto-replies
    logger.info("Searching for auto-reply notifications...")
    auto_replies = gmail_service.search_messages(AUTO_REPLY_QUERY)
    
    if auto_replies:
        logger.info(f"Found {len(auto_replies)} auto-reply notifications.")
        for message in auto_replies:
            message_data = gmail_service.get_message(message['id'])
            if message_data:
                from_header = gmail_service._get_header(message_data, 'from')
                subject = gmail_service._get_header(message_data, 'subject')
                body = gmail_service._get_full_body(message_data)
                
                email_match = re.search(r'<(.+?)>', from_header)
                email = email_match.group(1) if email_match else from_header.split()[-1]
                
                # Skip if target_email is specified and doesn't match
                if target_email and email != target_email:
                    continue
                
                logger.info(f"Processing auto-reply from: {from_header}, Subject: {subject}")
                
                # Check for inactive email in the body
                if is_inactive_email(body, subject):
                    logger.info(f"Detected inactive email notification for: {email}")
                
                process_email_response(message['id'], email, subject, body, gmail_service)
                processed_emails.add(email)
    else:
        logger.info("No auto-reply notifications found.")
    
    # Verify processing for all processed emails
    if processed_emails:
        logger.info("Verifying processing results...")
        all_verified = True
        for email in processed_emails:
            if not verify_processing(email, gmail_service, hubspot_service):
                all_verified = False
                logger.error(f"❌ Processing verification failed for {email}")
        
        if all_verified:
            logger.info("✅ All processing verified successfully")
        else:
            logger.error("❌ Some processing verifications failed - check logs for details")
    else:
        logger.info("No emails were processed")

if __name__ == "__main__":
    # For testing specific email
    TARGET_EMAIL = "bhagen@premiergc.com"
    if TESTING:
        print(f"\nRunning in TEST mode - no actual changes will be made\n")
    process_bounce_notifications(TARGET_EMAIL)
    
    # For processing all emails, comment out TARGET_EMAIL and use:
    # process_bounce_notifications()
