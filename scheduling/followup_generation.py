# followup_generation.py

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft, get_gmail_service
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import (
    get_best_outreach_window,
    adjust_send_time,
    calculate_send_date
)
from datetime import datetime, timedelta
import random
import base64


def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None,
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI and original Gmail message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    email_address,
                    name,
                    company_short_name,
                    body,
                    gmail_id,
                    scheduled_send_date,
                    draft_id
                FROM emails
                WHERE lead_id = ?
                AND sequence_num = 1
                AND gmail_id IS NOT NULL
                ORDER BY created_at DESC
            """, (lead_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"No original email found for lead_id={lead_id}")
                return None

            email_address, name, company_short_name, body, gmail_id, scheduled_date, draft_id = row
            original_email = {
                'email': email_address,
                'name': name,
                'company_short_name': company_short_name,
                'body': body,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'draft_id': draft_id
            }

        # Get the Gmail service
        gmail_service = get_gmail_service()
        
        # Get the original message from Gmail
        try:
            message = gmail_service.users().messages().get(
                userId='me',
                id=original_email['gmail_id'],
                format='full'
            ).execute()
            
            # Get the original HTML content
            original_html = None
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/html':
                        original_html = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            elif message['payload']['mimeType'] == 'text/html':
                original_html = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
            # Extract subject and original message content
            headers = message.get('payload', {}).get('headers', [])
            orig_subject = next(
                (header['value'] for header in headers if header['name'].lower() == 'subject'),
                'No Subject'
            )
            
            # Get the original sender (me) and timestamp
            from_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'from'),
                'me'
            )
            date_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'date'),
                ''
            )
            
            # Get original message body
            if 'parts' in message['payload']:
                orig_body = ''
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        orig_body = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            else:
                orig_body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error fetching Gmail message: {str(e)}", exc_info=True)
            return None

        # Format the follow-up email with the original included
        venue_name = original_email.get('company_short_name', 'your club')
        subject = f"Re: {orig_subject}"
        
        body = (
            f"Following up about improving operations at {venue_name}. "
            f"Would you have 10 minutes this week for a brief call?\n\n"
            f"Thanks,\n"
            f"Ty\n\n\n\n"
            f"On {date_header}, {from_header} wrote:\n"
            f"{orig_body}"
        )

        # Calculate send date using golf_outreach_strategy logic
        send_date = calculate_send_date(
            geography="Year-Round Golf",
            persona="General Manager",
            state="AZ",
            season_data=None
        )

        # Ensure minimum 3-day gap from original send date
        orig_scheduled_date = original_email.get('scheduled_send_date', datetime.now())
        while send_date < (orig_scheduled_date + timedelta(days=3)):
            send_date += timedelta(days=1)

        return {
            'email': original_email.get('email'),
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': str(lead_id),
            'company_short_name': original_email.get('company_short_name', ''),
            'in_reply_to': original_email['gmail_id'],
            'original_html': original_html,
            'thread_id': original_email['gmail_id']
        }
        

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
