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

class BounceProcessor:
    def __init__(self, testing: bool = False):
        """Initialize the bounce processor."""
        self.gmail_service = GmailService()
        self.hubspot_service = HubspotService(HUBSPOT_API_KEY)
        self.TESTING = testing
        self.processed_count = 0
        self.BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'

    def delete_from_database(self, email_address: str) -> bool:
        """
        Delete email records from database.
        Returns True if successful, False otherwise.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Delete from emails table
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email_address,))
            emails_deleted = cursor.rowcount
            
            # Delete from leads table
            cursor.execute("DELETE FROM leads WHERE email = %s", (email_address,))
            leads_deleted = cursor.rowcount
            
            conn.commit()
            logger.info(f"Deleted {emails_deleted} email(s) and {leads_deleted} lead(s) for {email_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from database: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def verify_processing(self, email: str) -> bool:
        """Verify that the contact was properly deleted and emails archived."""
        success = True
        
        # Check if contact still exists in HubSpot
        contact = self.hubspot_service.get_contact_by_email(email)
        if contact:
            logger.error(f"❌ Verification failed: Contact {email} still exists in HubSpot")
            success = False
        else:
            logger.info(f"✅ Verification passed: Contact {email} successfully deleted from HubSpot")
        
        # Check for any remaining emails in inbox
        query = f"from:{email} in:inbox"
        remaining_emails = self.gmail_service.search_messages(query)
        if remaining_emails:
            logger.error(f"❌ Verification failed: Found {len(remaining_emails)} unarchived emails from {email}")
            success = False
        else:
            logger.info(f"✅ Verification passed: No remaining emails from {email} in inbox")
        
        return success

    def process_single_bounce(self, notification: Dict) -> bool:
        """
        Process a single bounce notification.
        Returns True if processing was successful, False otherwise.
        """
        bounced_email = notification.get('bounced_email')
        message_id = notification.get('message_id')
        
        if not bounced_email or not message_id:
            logger.error("Invalid bounce notification - missing email or message ID")
            return False
        
        logger.info(f"Processing bounce notification:")
        logger.info(f"  Email: {bounced_email}")
        logger.info(f"  Message ID: {message_id}")
        
        try:
            success = True
            
            # 1. Delete from database
            if self.TESTING:
                logger.info(f"TESTING: Would delete {bounced_email} from database")
            else:
                if not self.delete_from_database(bounced_email):
                    success = False
                    logger.error(f"Failed to delete {bounced_email} from database")
            
            # 2. Delete from HubSpot
            contact = self.hubspot_service.get_contact_by_email(bounced_email)
            if contact:
                contact_id = contact.get('id')
                if contact_id:
                    if self.TESTING:
                        logger.info(f"TESTING: Would delete contact {bounced_email} from HubSpot")
                    else:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"Successfully deleted contact {bounced_email} from HubSpot")
                        else:
                            success = False
                            logger.error(f"Failed to delete contact {bounced_email} from HubSpot")
            else:
                logger.info(f"No HubSpot contact found for {bounced_email}")
            
            # 3. Archive the bounce notification
            if self.TESTING:
                logger.info(f"TESTING: Would archive bounce notification {message_id}")
            else:
                if self.gmail_service.archive_email(message_id):
                    logger.info(f"Successfully archived bounce notification")
                else:
                    success = False
                    logger.error(f"Failed to archive Gmail message {message_id}")
            
            # 4. Verify processing if not in testing mode
            if not self.TESTING and success:
                success = self.verify_processing(bounced_email)
            
            if success:
                self.processed_count += 1
                logger.info(f"✅ Successfully processed bounce for {bounced_email}")
            else:
                logger.error(f"❌ Failed to fully process bounce for {bounced_email}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error processing bounce for {bounced_email}: {str(e)}", exc_info=True)
            return False

    def process_bounces(self, target_email: str = None) -> int:
        """
        Process all bounce notifications, optionally filtered by specific email.
        Returns the number of successfully processed bounces.
        """
        logger.info(f"Starting bounce notification processing{' for email: ' + target_email if target_email else ''}")
        self.processed_count = 0
        
        bounce_notifications = self.gmail_service.get_all_bounce_notifications(inbox_only=True)
        
        if bounce_notifications:
            logger.info(f"Found {len(bounce_notifications)} bounce notification(s)")
            valid_notifications = []
            
            for notification in bounce_notifications:
                email = notification.get('bounced_email')
                if not email:
                    logger.debug("Skipping notification - no email address found")
                    continue
                
                if target_email and email.lower() != target_email.lower():
                    logger.debug(f"Skipping {email} - not target email {target_email}")
                    continue
                
                # Process the bounce notification
                try:
                    logger.info(f"Processing bounce for: {email}")
                    
                    # 1) Delete from SQL database
                    if self.TESTING:
                        logger.info(f"TEST MODE: Would delete {email} from SQL database")
                    else:
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
                    try:
                        hubspot = HubspotService(HUBSPOT_API_KEY)
                        contact = hubspot.get_contact_by_email(email)
                        if contact:
                            contact_id = contact.get('id')
                            if contact_id:
                                if self.TESTING:
                                    logger.info(f"TEST MODE: Would delete contact {email} from HubSpot")
                                else:
                                    if hubspot.delete_contact(contact_id):
                                        logger.info(f"Successfully deleted contact {email} from HubSpot")
                                    else:
                                        logger.error(f"Failed to delete contact {email} from HubSpot")
                            else:
                                logger.warning(f"Contact found but no ID for {email}")
                        else:
                            logger.warning(f"No contact found in HubSpot for {email}")
                    except Exception as e:
                        logger.error(f"Error processing HubSpot deletion: {str(e)}")

                    # 3) Archive the bounce notification
                    if self.TESTING:
                        logger.info(f"TEST MODE: Would archive bounce notification for {email}")
                    else:
                        if self.gmail_service.archive_email(notification['message_id']):
                            logger.info(f"Successfully archived bounce notification for {email}")
                        else:
                            logger.error(f"Failed to archive Gmail message for {email}")

                    self.processed_count += 1
                    logger.info(f"Successfully processed bounce for {email}")
                    
                except Exception as e:
                    logger.error(f"Error processing bounce notification for {email}: {str(e)}")
                    continue

            if self.processed_count > 0:
                logger.info(f"Successfully processed {self.processed_count} bounce notification(s)")
            else:
                logger.info("No bounce notifications were processed")
        else:
            logger.info("No bounce notifications found")
        
        return self.processed_count


def main():
    """Main entry point for bounce processing."""
    TESTING = False  # Set to False for production
    TARGET_EMAIL = "psanders@rccgolf.com"
    
    processor = BounceProcessor(testing=TESTING)
    if TESTING:
        logger.info("Running in TEST mode - no actual changes will be made")
        logger.info(f"Target Email: {TARGET_EMAIL}")
    
    processed_count = processor.process_bounces(target_email=TARGET_EMAIL)
    
    if TESTING:
        logger.info(f"TEST RUN COMPLETE - Would have processed {processed_count} valid bounce(s)")
    else:
        logger.info(f"Processing complete - Successfully processed {processed_count} bounce(s)")


if __name__ == "__main__":
    main() 