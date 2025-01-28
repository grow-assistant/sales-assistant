import sys
import os
import re
from datetime import datetime
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
AUTO_REPLY_QUERY = f"""
    (subject:"No longer employed" OR subject:"out of office" OR subject:"automatic reply")
    in:inbox
""".replace('\n', ' ').strip()

TESTING = True  # Set to False for production
REGULAR_RESPONSE_QUERY = '-in:trash -in:spam'  # Search everywhere except trash and spam


def get_first_50_words(text: str) -> str:
    """Get first 50 words of a text string."""
    if not text:
        return "No content"
    words = [w for w in text.split() if w.strip()]
    snippet = ' '.join(words[:50])
    return snippet + ('...' if len(words) > 50 else '')


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
            logger.info(
                f"Successfully archived contact {email} (ID: {contact_id}) due to invalid email: {analyzer_result['message']}"
            )
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
        
        # 1) Delete from SQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email,))
            cursor.execute("DELETE FROM leads WHERE email = %s", (email,))
            conn.commit()
            logger.info(f"Deleted records for {email} from SQL database")
        except Exception as e:
            logger.error(f"Error deleting from SQL: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

        # 2) Delete from HubSpot
        contact = hubspot.get_contact_by_email(email)
        if contact:
            contact_id = contact.get('id')
            if contact_id and hubspot.delete_contact(contact_id):
                logger.info(f"Contact {email} deleted from HubSpot")
        
        # 3) Archive bounce notification (uncomment if you'd like to truly archive)
        # if gmail.archive_email(gmail_id):
        #     logger.info(f"Bounce notification archived in Gmail")
        print(f"Would have archived bounce notification with ID: {gmail_id}")

    except Exception as e:
        logger.error(f"Error processing bounced email {email}: {str(e)}")


def is_out_of_office(message: str, subject: str) -> bool:
    """Check if a message is an out-of-office response."""
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
    import re
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
                'firstname': (
                    contact_info['new_contact'].split()[0]
                    if contact_info['new_contact'] != 'Unknown' else ''
                ),
                'lastname': (
                    ' '.join(contact_info['new_contact'].split()[1:])
                    if contact_info['new_contact'] != 'Unknown' else ''
                ),
                'company': contact_info['company'] if contact_info['company'] != 'Unknown' else '',
                'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else '',
                'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else ''
            }
            
            new_contact = hubspot.create_contact(new_properties)
            if new_contact:
                logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
            
            # Try to archive the message
            if gmail_service.archive_email(message_id):
                logger.info("Archived employment change notification")
            
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


def delete_email_from_database(email_address):
    """Helper function to delete email records from database."""
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
    """Process a single bounce notification."""
    bounced_email = notification['bounced_email']
    message_id = notification['message_id']
    logger.info(f"Processing bounce notification - Email: {bounced_email}, Message ID: {message_id}")
    
    if TESTING:
        print(f"Would have deleted {bounced_email} from SQL database")
    else:
        delete_email_from_database(bounced_email)
    
    try:
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


def mark_lead_as_dq(email_address, reason):
    """Mark a lead as disqualified in the database."""
    logger.info(f"Marking {email_address} as DQ. Reason: {reason}")
    # Add your database update logic here


def check_company_responses(email: str, sent_date: str) -> bool:
    """
    Check if there are any responses from the same company domain after the sent date.
    
    - Now logs each inbound email from any contact with the same domain.
    - Returns True if at least one such email is found after the 'sent_date'.
    """
    try:
        domain = email.split('@')[-1]
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get all contacts with the same domain
        domain_contacts = hubspot.get_contacts_by_company_domain(domain)
        if not domain_contacts:
            logger.info(f"No domain contacts found for {domain}")
            return False
            
        logger.info(f"Found {len(domain_contacts)} contact(s) for domain '{domain}'. Checking inbound emails...")
        
        # Convert sent_date string to datetime for comparison
        sent_datetime = datetime.strptime(sent_date, '%Y-%m-%d %H:%M:%S')
        
        any_responses = False
        
        for c in domain_contacts:
            contact_id = c.get('id')
            if not contact_id:
                continue
            
            contact_email = c.get('properties', {}).get('email', 'Unknown')
            emails = hubspot.get_all_emails_for_contact(contact_id)
            # Filter for inbound/incoming emails after sent_date
            incoming_emails = [
                e for e in emails
                if e.get('direction') in ['INBOUND', 'INCOMING', 'INCOMING_EMAIL']
                and int(e.get('timestamp', 0)) / 1000 > sent_datetime.timestamp()
            ]
            
            for e in incoming_emails:
                # Log each inbound email from the same company domain
                msg_time = datetime.fromtimestamp(int(e.get('timestamp', 0))/1000).strftime('%Y-%m-%d %H:%M:%S')
                snippet = get_first_50_words(e.get('body_text', '') or e.get('text', '') or '')
                logger.info(
                    f"FLAGGED: Response from domain '{domain}' => "
                    f"Contact: {contact_email}, Timestamp: {msg_time}, Snippet: '{snippet}'"
                )
                any_responses = True
        
        return any_responses

    except Exception as e:
        logger.error(f"Error checking company responses for {email}: {str(e)}", exc_info=True)
        return False


def process_email_response(message_id: str, email: str, subject: str, body: str, gmail_service: GmailService) -> None:
    """Process an email response."""
    try:
        logger.info(f"Starting to process email response from {email}")
        message_data = gmail_service.get_message(message_id)
        sent_date = datetime.fromtimestamp(
            int(message_data.get('internalDate', 0)) / 1000
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Message date: {sent_date}")
        
        # ** Updated call: check and flag domain responses
        if check_company_responses(email, sent_date):
            logger.info(f"Found response(s) from the same company domain for {email} after {sent_date}")
            # Example action: update HubSpot contact with a 'company_responded' flag
            hubspot = HubspotService(HUBSPOT_API_KEY)
            contact = hubspot.get_contact_by_email(email)
            if contact:
                properties = {
                    'company_responded': 'true',
                    'last_company_response_date': sent_date
                }
                hubspot.update_contact(contact.get('id'), properties)
                logger.info(f"Updated contact {email} to mark company response")

        # Continue with existing logic...
        from_header = gmail_service._get_header(message_data, 'from')
        full_name = from_header.split('<')[0].strip()
        
        hubspot = HubspotService(HUBSPOT_API_KEY)
        logger.info(f"Processing response for email: {email}")
        
        # Get contact directly
        contact = hubspot.get_contact_by_email(email)
        
        # Check if it's a do-not-contact request
        if is_do_not_contact_request(body, subject):
            logger.info(f"Detected do-not-contact request from {email}")
            if contact:
                company_name = contact.get('properties', {}).get('company', '')
                if hubspot.mark_do_not_contact(email, company_name):
                    logger.info(f"Successfully marked {email} as do-not-contact")
                    delete_email_from_database(email)
                    logger.info(f"Removed {email} from SQL database")
                    if gmail_service.archive_email(message_id):
                        logger.info(f"Archived do-not-contact request email")
            return
        
        # Check if it's an employment change notification
        if is_employment_change(body, subject):
            logger.info(f"Detected employment change notification for {email}")
            process_employment_change(email, message_id, gmail_service)
            return
            
        if contact:
            contact_id = contact.get('id')
            contact_email = contact.get('properties', {}).get('email', '')
            logger.info(f"Found contact in HubSpot: {contact_email} (ID: {contact_id})")
            
            # Check if it's an auto-reply
            if "automatic reply" in subject.lower():
                logger.info("Detected auto-reply, sending to xAI for analysis...")
                contact_info = analyze_auto_reply(body, subject)
                
                if contact_info and contact_info.get('new_email'):
                    if TESTING:
                        print("\nAuto-reply Analysis Results:")
                        print(f"Original Contact: {email}")
                        print(f"Analysis Results: {contact_info}")
                        return
                    
                    new_contact = hubspot.create_contact({
                        'email': contact_info['new_email'],
                        'firstname': (
                            contact_info['new_contact'].split()[0]
                            if contact_info['new_contact'] != 'Unknown'
                            else contact.get('properties', {}).get('firstname', '')
                        ),
                        'lastname': (
                            ' '.join(contact_info['new_contact'].split()[1:])
                            if contact_info['new_contact'] != 'Unknown'
                            else contact.get('properties', {}).get('lastname', '')
                        ),
                        'company': (
                            contact_info['company']
                            if contact_info['company'] != 'Unknown'
                            else contact.get('properties', {}).get('company', '')
                        ),
                        'jobtitle': (
                            contact_info['new_title']
                            if contact_info['new_title'] != 'Unknown'
                            else contact.get('properties', {}).get('jobtitle', '')
                        ),
                        'phone': (
                            contact_info['phone']
                            if contact_info['phone'] != 'Unknown'
                            else contact.get('properties', {}).get('phone', '')
                        )
                    })
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
                            logger.info("Archived employment change notification")
                else:
                    logger.info("No new contact info found, processing as standard auto-reply.")
                    if gmail_service.archive_email(message_id):
                        logger.info("Archived standard auto-reply message")
            
            elif is_inactive_email(body, subject):
                notification = {
                    "bounced_email": contact_email,
                    "message_id": message_id
                }
                process_bounce_notification(notification, gmail_service)
            
            logger.info(f"Processed response for contact: {contact_email}")
        else:
            logger.warning(f"No contact found in HubSpot for email: {email}")
            # Check for employment change before giving up
            if is_employment_change(body, subject):
                logger.info("Processing employment change for non-existent contact")
                process_employment_change(email, message_id, gmail_service)
            else:
                # Still try to archive the message even if no contact found
                if gmail_service.archive_email(message_id):
                    logger.info("Archived message for non-existent contact")

    except Exception as e:
        logger.error(f"Error processing email response: {str(e)}", exc_info=True)


def process_bounce_notifications(target_email: str = None):
    """
    Main function to process all bounce notifications and auto-replies.
    """
    logger.info("Starting bounce notification and auto-reply processing...")
    
    gmail_service = GmailService()
    hubspot_service = HubspotService(HUBSPOT_API_KEY)
    processed_emails = set()
    target_domain = "rainmakersusa.com"  # Set target domain
    
    logger.info(f"Processing emails for domain: {target_domain}")
    
    # 1) Process bounce notifications
    logger.info("=" * 80)
    logger.info("Searching for bounce notifications in inbox...")
    bounce_notifications = gmail_service.get_all_bounce_notifications(inbox_only=True)
    
    if bounce_notifications:
        logger.info(f"Found {len(bounce_notifications)} bounce notifications to process.")
        for notification in bounce_notifications:
            email = notification.get('bounced_email')
            if email:
                domain = email.split('@')[-1].lower()
                if domain == target_domain:
                    logger.info(f"Processing bounce notification for domain email: {email}")
                    process_bounce_notification(notification, gmail_service)
                    processed_emails.add(email)
    
    # 2) Process auto-replies with domain filter
    logger.info("=" * 80)
    logger.info("Searching for auto-reply notifications...")
    logger.info(f"Using auto-reply query: {AUTO_REPLY_QUERY}")
    
    auto_replies = gmail_service.search_messages(AUTO_REPLY_QUERY)
    
    if auto_replies:
        logger.info(f"Found {len(auto_replies)} auto-reply notifications for {target_domain} domain.")
        for message in auto_replies:
            try:
                message_data = gmail_service.get_message(message['id'])
                if message_data:
                    from_header = gmail_service._get_header(message_data, 'from')
                    to_header = gmail_service._get_header(message_data, 'to')
                    subject = gmail_service._get_header(message_data, 'subject')
                    
                    # Extract email addresses from headers
                    email_addresses = []
                    for header in [from_header, to_header]:
                        if header:
                            matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', header)
                            email_addresses.extend(matches)
                    
                    # Find domain-specific emails
                    domain_emails = [
                        email for email in email_addresses 
                        if email.split('@')[-1].lower() == target_domain
                    ]
                    
                    if domain_emails:
                        body = gmail_service._get_full_body(message_data)
                        logger.info(f"Found domain emails in auto-reply: {domain_emails}")
                        
                        for email in domain_emails:
                            logger.info(f"Processing auto-reply for: {email}")
                            logger.info(f"Subject: {subject}")
                            logger.info(f"Preview: {get_first_50_words(body)}")
                            
                            if TESTING:
                                logger.info(f"TESTING MODE: Would process auto-reply for {email}")
                            else:
                                process_email_response(message['id'], email, subject, body, gmail_service)
                                processed_emails.add(email)
                    else:
                        logger.debug(f"No {target_domain} emails found in auto-reply message")
            except Exception as e:
                logger.error(f"Error processing auto-reply: {str(e)}", exc_info=True)
    
    # 3) Process regular responses with broader query
    logger.info("=" * 80)
    logger.info("Searching for regular responses with domain-wide query...")
    
    # Broader query to catch all domain emails
    regular_query = f"""
        (from:@{target_domain} OR to:@{target_domain})
        -in:trash -in:spam -label:sent
        newer_than:30d
    """.replace('\n', ' ').strip()
    
    logger.info(f"Using domain-wide search query: {regular_query}")
    
    
    try:
        regular_responses = gmail_service.search_messages(regular_query)
        
        if regular_responses:
            logger.info(f"Found {len(regular_responses)} potential domain messages.")
            for idx, message in enumerate(regular_responses, 1):
                logger.info("-" * 40)
                logger.info(f"Processing message {idx} of {len(regular_responses)}")
                
                try:
                    message_data = gmail_service.get_message(message['id'])
                    if message_data:
                        # Extract and log all headers
                        from_header = gmail_service._get_header(message_data, 'from')
                        to_header = gmail_service._get_header(message_data, 'to')
                        cc_header = gmail_service._get_header(message_data, 'cc')
                        subject = gmail_service._get_header(message_data, 'subject')
                        date = gmail_service._get_header(message_data, 'date')
                        
                        logger.info(f"Message details:")
                        logger.info(f"  Date: {date}")
                        logger.info(f"  From: {from_header}")
                        logger.info(f"  To: {to_header}")
                        logger.info(f"  CC: {cc_header}")
                        logger.info(f"  Subject: {subject}")
                        
                        # Extract all email addresses from all headers
                        email_addresses = []
                        for header in [from_header, to_header, cc_header]:
                            if header:
                                matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', header)
                                email_addresses.extend(matches)
                        
                        # Find domain-specific emails
                        domain_emails = [
                            email for email in email_addresses 
                            if email.split('@')[-1].lower() == target_domain
                        ]
                        
                        if domain_emails:
                            logger.info(f"Found {len(domain_emails)} domain email(s) in message: {domain_emails}")
                            body = gmail_service._get_full_body(message_data)
                            
                            for email in domain_emails:
                                logger.info(f"Processing response for domain email: {email}")
                                logger.info(f"Message preview: {get_first_50_words(body)}")
                                
                                if TESTING:
                                    logger.info(f"TESTING MODE: Would process message for {email}")
                                else:
                                    process_email_response(message['id'], email, subject, body, gmail_service)
                                    processed_emails.add(email)
                                    logger.info(f"✓ Processed message for {email}")
                        else:
                            logger.info("No domain emails found in message headers")
                
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}", exc_info=True)
                    continue
        else:
            logger.info("No messages found matching domain-wide search criteria.")
    
    except Exception as e:
        logger.error(f"Error during Gmail search: {str(e)}", exc_info=True)

    # 4) Verification
    if processed_emails:
        logger.info("=" * 80)
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
        logger.info("No emails were processed.")


if __name__ == "__main__":
    TARGET_EMAIL = "lvelasquez@rainmakersusa.com"
    if TESTING:
        print("\nRunning in TEST mode - no actual changes will be made\n")
    process_bounce_notifications(TARGET_EMAIL)
    # Or process_bounce_notifications() for all.
