# scheduling/followup_generation.py
"""
Functions for generating follow-up emails.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft, get_gmail_service, get_gmail_template
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import (
    get_best_outreach_window,
    adjust_send_time,
    calculate_send_date
)
from datetime import datetime, timedelta
import random
import base64
from services.gmail_service import GmailService

def get_calendly_template() -> str:
    """Load the Calendly HTML template from file."""
    try:
        template_path = Path(project_root) / 'docs' / 'templates' / 'calendly.html'
        logger.debug(f"Loading Calendly template from: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_html = f.read()
            
        logger.debug(f"Loaded template, length: {len(template_html)}")
        return template_html
    except Exception as e:
        logger.error(f"Error loading Calendly template: {str(e)}")
        return ""

def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None,
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI and original Gmail message"""
    try:
        logger.debug(f"Starting follow-up generation for lead_id={lead_id}, sequence_num={sequence_num}")
        
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

        # Get the Gmail service and raw service
        gmail_service = GmailService()
        service = get_gmail_service()
        
        # Get the original message from Gmail
        try:
            message = service.users().messages().get(
                userId='me',
                id=original_email['gmail_id'],
                format='full'
            ).execute()
            
            # Get headers
            headers = message['payload'].get('headers', [])
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
            
            # Get original HTML content
            original_html = None
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/html':
                        original_html = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            elif message['payload'].get('mimeType') == 'text/html':
                original_html = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error fetching Gmail message: {str(e)}", exc_info=True)
            return None

        # Build the new content with proper div wrapper
        venue_name = original_email.get('company_short_name', 'your club')
        subject = f"Re: {orig_subject}"
        
        followup_content = (
            f"<div dir='ltr'>"
            f"<p>Following up about improving operations at {venue_name}. "
            f"Would you have 10 minutes this week for a brief call?</p>"
            f"<p>Thanks,<br>Ty</p>"
            f"</div>"
        )

        # Get Calendly template
        template_html = get_calendly_template()
        
        # Create blockquote for original email
        blockquote_html = (
            f'<blockquote class="gmail_quote" '
            f'style="margin:0 0 0 .8ex;border-left:1px #ccc solid;padding-left:1ex">\n'
            f'{original_html or ""}'
            f'</blockquote>'
        )
        
        # Combine in correct order with proper HTML structure
        full_html = (
            f"{followup_content}\n"     # New follow-up content first
            f"{template_html}\n"        # Calendly template second
            f"{blockquote_html}"        # Original email last (with Gmail blockquote formatting)
        )

        # Now log the combined HTML after it's created
        logger.debug("\n=== Final Combined HTML ===")
        logger.debug("First 1000 chars:")
        logger.debug(full_html[:1000])
        logger.debug("\nLast 1000 chars:")
        logger.debug(full_html[-1000:] if len(full_html) > 1000 else "N/A")
        logger.debug("=" * 80)

        # Calculate send date using golf_outreach_strategy logic
        send_date = calculate_send_date(
            geography="Year-Round Golf",
            persona="General Manager",
            state="AZ",
            sequence_num=sequence_num or 2,  # Pass sequence_num to determine time window
            season_data=None
        )
        logger.debug(f"Calculated send date: {send_date}")

        # Ensure minimum 3-day gap from original send date
        orig_scheduled_date = original_email.get('scheduled_send_date', datetime.now())
        logger.debug(f"Original scheduled date: {orig_scheduled_date}")
        while send_date < (orig_scheduled_date + timedelta(days=3)):
            send_date += timedelta(days=1)
            logger.debug(f"Adjusted send date to ensure 3-day gap: {send_date}")

        return {
            'email': original_email.get('email'),
            'subject': subject,
            'body': full_html,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': str(lead_id),
            'company_short_name': original_email.get('company_short_name', ''),
            'in_reply_to': original_email['gmail_id'],
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
