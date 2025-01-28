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
from utils.xai_integration import analyze_auto_reply, analyze_employment_change

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

def process_employment_change(email: str, message_id: str, gmail_service: GmailService) -> None:
    """Process an employment change notification."""
    try:
        # Get message details
        message_data = gmail_service.get_message(message_id)
        subject = gmail_service._get_header(message_data, 'subject')
        body = gmail_service._get_full_body(message_data)
        
        # Analyze the message
        contact_info = analyze_employment_change(body, subject)
        
        if contact_info and contact_info.get('new_email'):
            hubspot = HubspotService(HUBSPOT_API_KEY)
            
            # Create new contact properties
            new_properties = {
                'email': contact_info['new_email'],
                'firstname': contact_info['new_contact'].split()[0] if contact_info['new_contact'] != 'Unknown' else '',
                'lastname': ' '.join(contact_info['new_contact'].split()[1:]) if contact_info['new_contact'] != 'Unknown' else '',
                'company': contact_info['company'] if contact_info['company'] != 'Unknown' else '',
                'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else '',
                'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else ''
            }
            
            # Create the new contact
            new_contact = hubspot.create_contact(new_properties)
            if new_contact:
                logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
            
            # Try to archive the message
            if gmail_service.archive_email(message_id):
                logger.info(f"Archived employment change notification")
            
    except Exception as e:
        logger.error(f"Error processing employment change: {str(e)}", exc_info=True)

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

def is_employment_change(body: str, subject: str) -> bool:
    """Check if the message indicates an employment change."""
    employment_phrases = [
        "no longer with",
        "no longer employed",
        "is no longer",
        "has left",
        "no longer works",
        "no longer at",
        "has departed",
        "is not with"
    ]
    
    message_lower = (body + " " + subject).lower()
    return any(phrase in message_lower for phrase in employment_phrases)

def is_do_not_contact_request(message: str, subject: str) -> bool:
    """Check if message is a request to not be contacted."""
    dnc_phrases = [
        "don't contact",
        "do not contact",
        "stop contacting",
        "please remove",
        "unsubscribe",
        "take me off",
        "remove me from",
        "no thanks",
        "not interested"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    
    return any(phrase in message_lower or phrase in subject_lower for phrase in dnc_phrases)

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
        
        # Check if it's a do not contact request first
        if is_do_not_contact_request(body, subject):
            logger.info(f"Detected do-not-contact request from {email}")
            if contact:
                # Get company name from contact if available
                company_name = contact.get('properties', {}).get('company', '')
                
                # Mark as do not contact in HubSpot
                if hubspot.mark_do_not_contact(email, company_name):
                    logger.info(f"Successfully marked {email} as do-not-contact")
                    
                    # Delete from SQL database
                    delete_email_from_database(email)
                    logger.info(f"Removed {email} from SQL database")
                    
                    # Archive the message
                    if gmail_service.archive_email(message_id):
                        logger.info(f"Archived do-not-contact request email")
                return
        
        # First check if it's an employment change notification
        if is_employment_change(body, subject):
            logger.info(f"Detected employment change notification for {email}")
            # Process employment change even if original contact doesn't exist
            process_employment_change(email, message_id, gmail_service)
            return
            
        if contact:
            contact_id = contact.get('id')
            contact_email = contact.get('properties', {}).get('email', '')
            logger.info(f"Found contact in HubSpot: {contact_email} (ID: {contact_id})")
            
            # Then check if it's an auto-reply
            if "automatic reply" in subject.lower():
                logger.info("Detected auto-reply, sending to xAI for analysis...")
                contact_info = analyze_auto_reply(body, subject)
                
                if contact_info and contact_info.get('new_email'):
                    # Get existing contact properties
                    properties = contact.get('properties', {})
                    
                    # Prepare new contact properties
                    new_properties = {
                        'email': contact_info['new_email'],
                        'firstname': contact_info['new_contact'].split()[0] if contact_info['new_contact'] != 'Unknown' else properties.get('firstname', ''),
                        'lastname': ' '.join(contact_info['new_contact'].split()[1:]) if contact_info['new_contact'] != 'Unknown' else properties.get('lastname', ''),
                        'company': contact_info['company'] if contact_info['company'] != 'Unknown' else properties.get('company', ''),
                        'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else properties.get('jobtitle', ''),
                        'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else properties.get('phone', '')
                    }

                    if TESTING:
                        print("\nAuto-reply Analysis Results:")
                        print(f"Original Contact: {email}")
                        print(f"Analysis Results: {contact_info}")
                        print(f"\nWould have performed these actions:")
                        print(f"1. Create new contact in HubSpot: {new_properties}")
                        print(f"2. Copy all associations from old contact: {contact_id}")
                        print(f"3. Delete old contact: {email}")
                        print(f"4. Archive message: {message_id}")
                        return
                    
                    # Create new contact and handle transition
                    new_contact = hubspot.create_contact(new_properties)
                    if new_contact:
                        logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
                        
                        # Copy associations
                        old_associations = hubspot.get_contact_associations(contact_id)
                        for association in old_associations:
                            hubspot.create_association(new_contact['id'], association['id'], association['type'])
                        
                        # Delete old contact
                        if hubspot.delete_contact(contact_id):
                            logger.info(f"Deleted old contact: {email}")
                        
                        # Archive the notification
                        if gmail_service.archive_email(message_id):
                            logger.info(f"Archived employment change notification")
                else:
                    logger.info("No new contact information found, processing as standard auto-reply")
                    # Archive the message since it's just a standard auto-reply
                    if gmail_service.archive_email(message_id):
                        logger.info(f"Archived standard auto-reply message")
            elif is_inactive_email(body, subject):
                notification = {
                    "bounced_email": contact_email,
                    "message_id": message_id
                }
                process_bounce_notification(notification, gmail_service)
            
            logger.info(f"Processed response for contact: {contact_email}")
        else:
            logger.warning(f"No contact found in HubSpot for email: {email}")
            # Check if it's an employment change before giving up
            if is_employment_change(body, subject):
                logger.info(f"Processing employment change for non-existent contact")
                process_employment_change(email, message_id, gmail_service)
            else:
                # Still try to archive the message even if contact not found
                if gmail_service.archive_email(message_id):
                    logger.info(f"Archived message for non-existent contact")
            
    except Exception as e:
        logger.error(f"Error processing email response: {str(e)}", exc_info=True)

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
        # Delete from HubSpot
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
    
    # Add new rejection check
    logger.info("Searching for explicit rejection responses...")
    rejection_query = gmail_service.get_rejection_search_query()
    rejection_messages = gmail_service.search_messages(rejection_query)
    
    if rejection_messages:
        logger.info(f"Found {len(rejection_messages)} rejection messages.")
        for message in rejection_messages:
            message_data = gmail_service.get_message(message['id'])
            if message_data:
                # Properly extract the from header
                from_header = gmail_service._get_header(message_data, 'from')
                email_match = re.search(r'<(.+?)>', from_header)
                email = email_match.group(1) if email_match else from_header.split()[-1]
                
                if target_email and email.lower() == target_email.lower():
                    logger.info(f"Processing rejection from {email}")
                    # Get the body and subject for context
                    body = gmail_service._get_full_body(message_data)
                    subject = gmail_service._get_header(message_data, 'subject')
                    
                    # Mark as DQ in HubSpot
                    hubspot_service = HubspotService(HUBSPOT_API_KEY)
                    contact = hubspot_service.get_contact_by_email(email)
                    if contact:
                        hubspot_service.mark_contact_as_dq(email, "Explicit rejection")
                        logger.info(f"Marked {email} as DQ in HubSpot")
                    
                    # Archive the message
                    if gmail_service.archive_email(message['id']):
                        logger.info(f"Archived rejection message from {email}")
                    
                    processed_emails.add(email)

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

def mark_lead_as_dq(email_address, reason):
    """Mark a lead as disqualified in the database"""
    logger.info(f"Marking {email_address} as DQ. Reason: {reason}")
    # Add your database update logic here
    # Example:
    # db.update_lead_status(email_address, status='DQ', reason=reason)

if __name__ == "__main__":
    # For testing specific email
    TARGET_EMAIL = "bhiggins@foxchapelgolfclub.com"
    if TESTING:
        print(f"\nRunning in TEST mode - no actual changes will be made\n")
    process_bounce_notifications(TARGET_EMAIL)
    
    # For processing all emails, comment out TARGET_EMAIL and use:
    # process_bounce_notifications()
