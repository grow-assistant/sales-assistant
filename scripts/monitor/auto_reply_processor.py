import sys
import os
import re
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Optional
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from scheduling.database import get_db_connection

class AutoReplyProcessor:
    def __init__(self, testing: bool = False):
        """Initialize the auto-reply processor."""
        self.gmail_service = GmailService()
        self.hubspot_service = HubspotService(HUBSPOT_API_KEY)
        self.analyzer = ResponseAnalyzerService()
        self.TESTING = testing
        self.processed_count = 0
        self.AUTO_REPLY_QUERY = """
            (subject:"No longer employed" OR subject:"out of office" OR 
            subject:"automatic reply" OR subject:"auto-reply" OR 
            subject:"automated response" OR subject:"inactive")
            in:inbox newer_than:30d
        """.replace('\n', ' ').strip()

    def extract_new_contact_email(self, message: str) -> Optional[str]:
        """Extract new contact email from message."""
        patterns = [
            r'(?:please\s+)?(?:contact|email|send\s+to|forward\s+to)\s+[\w\s]+:\s*([\w\.-]+@[\w\.-]+\.\w+)',
            r'please\s+(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'(?:new\s+email|new\s+contact|instead\s+email)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'email\s+([\w\.-]+@[\w\.-]+\.\w+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                return match.group(1)
        return None

    def process_single_reply(self, message_id: str, email: str, subject: str, body: str) -> bool:
        """
        Process a single auto-reply message.
        Returns True if processing was successful, False otherwise.
        """
        try:
            logger.info(f"Processing auto-reply for: {email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body preview: {body[:200]}...")
            success = True

            # Check for out of office first
            if self.analyzer.is_out_of_office(body, subject):
                logger.info("Detected out of office reply - archiving")
                if not self.TESTING:
                    if self.gmail_service.archive_email(message_id):
                        logger.info("✅ Archived out of office message")
                    else:
                        success = False
                        logger.error("❌ Failed to archive out of office message")
                return success

            # Check for employment change
            if self.analyzer.is_employment_change(body, subject):
                logger.info("Detected employment change notification")
                if self.TESTING:
                    logger.info(f"TESTING: Would process employment change for {email}")
                    return True
                # Try to extract new contact email
                new_email = self.extract_new_contact_email(body)
                if new_email:
                    logger.info(f"Found new contact email: {new_email}")
                    # Get contact info for transfer
                    contact = self.hubspot_service.get_contact_by_email(email)
                    if contact:
                        # Create new contact with existing info
                        new_properties = contact.copy()
                        new_properties['email'] = new_email
                        if self.hubspot_service.create_contact(new_properties):
                            logger.info(f"✅ Created new contact: {new_email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to create new contact: {new_email}")

                # Delete old contact
                contact = self.hubspot_service.get_contact_by_email(email)
                if contact:
                    contact_id = contact.get('id')
                    if contact_id:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"✅ Deleted old contact: {email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to delete contact {email}")
                else:
                    logger.info(f"ℹ️ No contact found for {email}")

            # Check for do not contact request
            elif self.analyzer.is_do_not_contact_request(body, subject):
                logger.info("Detected do not contact request")
                if self.TESTING:
                    logger.info(f"TESTING: Would process do not contact request for {email}")
                    return True
                contact = self.hubspot_service.get_contact_by_email(email)
                if contact:
                    contact_id = contact.get('id')
                    if contact_id:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"✅ Deleted contact per request: {email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to delete contact {email}")
                else:
                    logger.info(f"ℹ️ No contact found for {email}")

            # Check for inactive/bounce
            elif self.analyzer.is_inactive_email(body, subject):
                logger.info("Detected inactive email notification")
                if self.TESTING:
                    logger.info(f"TESTING: Would process inactive email for {email}")
                    return True
                else:
                    notification = {
                        "bounced_email": email,
                        "message_id": message_id
                    }
                    from bounce_processor import BounceProcessor
                    bounce_processor = BounceProcessor(testing=self.TESTING)
                    if bounce_processor.process_single_bounce(notification):
                        logger.info(f"✅ Successfully processed bounce for {email}")
                    else:
                        success = False
                        logger.error(f"❌ Failed to process bounce for {email}")
                    return success

            # If we get here, log why we didn't process it
            logger.info("ℹ️ Message did not match any processing criteria")
            return False

        except Exception as e:
            logger.error(f"❌ Error processing auto-reply for {email}: {str(e)}", exc_info=True)
            return False

    def process_auto_replies(self, target_email: str = None) -> int:
        """
        Process all auto-reply messages, optionally filtered by specific email.
        Returns the number of successfully processed replies.
        """
        logger.info(f"Starting auto-reply processing{' for email: ' + target_email if target_email else ''}")
        self.processed_count = 0

        auto_replies = self.gmail_service.search_messages(self.AUTO_REPLY_QUERY)

        if auto_replies:
            logger.info(f"Found {len(auto_replies)} auto-reply message(s)")

            for message in auto_replies:
                try:
                    message_data = self.gmail_service.get_message(message['id'])
                    if not message_data:
                        continue

                    from_header = self.gmail_service._get_header(message_data, 'from')
                    subject = self.gmail_service._get_header(message_data, 'subject')
                    
                    if not from_header:
                        continue

                    # Extract email from header
                    matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', from_header)
                    if not matches:
                        continue

                    email = matches[0]
                    
                    if target_email and email.lower() != target_email.lower():
                        logger.debug(f"Skipping {email} - not target email {target_email}")
                        continue

                    body = self.gmail_service._get_full_body(message_data)
                    self.process_single_reply(message['id'], email, subject, body)

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue

            if self.processed_count > 0:
                logger.info(f"Successfully processed {self.processed_count} auto-reply message(s)")
            else:
                logger.info("No auto-reply messages were processed")
        else:
            logger.info("No auto-reply messages found")

        return self.processed_count


def main():
    """Main entry point for auto-reply processing."""
    TESTING = False  # Set to False for production
    TARGET_EMAIL = "psanders@rccgolf.com"
    
    processor = AutoReplyProcessor(testing=TESTING)
    if TESTING:
        logger.info("Running in TEST mode - no actual changes will be made")
        logger.info(f"Target Email: {TARGET_EMAIL}")
    
    processed_count = processor.process_auto_replies(target_email=TARGET_EMAIL)
    
    if TESTING:
        logger.info(f"TEST RUN COMPLETE - Would have processed {processed_count} auto-replies")
    else:
        logger.info(f"Processing complete - Successfully processed {processed_count} auto-replies")


if __name__ == "__main__":
    main() 