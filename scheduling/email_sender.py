from datetime import datetime
from utils.gmail_integration import get_gmail_service
from scheduling.database import get_db_connection
import logging
import time

logger = logging.getLogger(__name__)

def send_scheduled_emails():
    """
    Sends emails that are scheduled for now or in the past.
    Updates their status in the database.
    """
    now = datetime.now()
    logger.info(f"Checking for emails to send at {now}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Find emails that should be sent (both draft and reviewed status)
        cursor.execute("""
            SELECT email_id, draft_id, email_address, scheduled_send_date, status
            FROM emails 
            WHERE status IN ('draft', 'reviewed')
            AND scheduled_send_date <= GETDATE()
            AND actual_send_date IS NULL
            ORDER BY scheduled_send_date ASC
        """)
        
        to_send = cursor.fetchall()
        logger.info(f"Found {len(to_send)} emails to send")

        if not to_send:
            return

        service = get_gmail_service()
        
        for email_id, draft_id, recipient, scheduled_date, status in to_send:
            try:
                logger.debug(f"Attempting to send email_id={email_id} to {recipient} (scheduled for {scheduled_date}, status={status})")
                
                # Send the draft
                message = service.users().drafts().send(
                    userId='me',
                    body={'id': draft_id}
                ).execute()

                # Update database
                cursor.execute("""
                    UPDATE emails 
                    SET status = 'sent',
                        actual_send_date = GETDATE(),
                        gmail_id = ?
                    WHERE email_id = ?
                """, (message['id'], email_id))
                
                conn.commit()
                logger.info(f"Successfully sent email_id={email_id} to {recipient}")

                # Small delay between sends to respect rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to send email_id={email_id} to {recipient}: {str(e)}", exc_info=True)
                conn.rollback()
                continue 