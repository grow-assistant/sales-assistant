# followup_generation.py

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window, adjust_send_time
from datetime import datetime, timedelta
import random


def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None, 
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    l.email,
                    l.first_name,
                    c.name,
                    e.subject,
                    e.body,
                    e.created_at,
                    c.state
                FROM emails e
                JOIN leads l ON l.lead_id = e.lead_id
                LEFT JOIN companies c ON l.company_id = c.company_id
                WHERE e.lead_id = ?
                AND e.sequence_num = 1
                ORDER BY e.created_at DESC
            """, (lead_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"No original email found for lead_id={lead_id}")
                return None

            email, first_name, company_name, subject, body, created_at, state = row
            original_email = {
                'email': email,
                'first_name': first_name,
                'name': company_name,
                'subject': subject,
                'body': body,
                'created_at': created_at,
                'state': state
            }

        # If original_email is provided, use that instead of querying
        if original_email:
            email = original_email.get('email')
            first_name = original_email.get('first_name')
            company_name = original_email.get('name', 'your club')
            state = original_email.get('state')
            orig_subject = original_email.get('subject')
            orig_body = original_email.get('body')
            orig_date = original_email.get('created_at', datetime.now())
            
            # Get original scheduled send date
            cursor.execute("""
                SELECT TOP 1 scheduled_send_date 
                FROM emails 
                WHERE lead_id = ? AND sequence_num = 1
                ORDER BY created_at DESC
            """, (lead_id,))
            result = cursor.fetchone()
            orig_scheduled_date = result[0] if result else orig_date
        else:
            # Query for required fields
            query = """
                SELECT 
                    l.email,
                    l.first_name,
                    c.state,
                    c.name,
                    e.subject,
                    e.body,
                    e.created_at
                FROM leads l
                LEFT JOIN companies c ON l.company_id = c.company_id
                LEFT JOIN emails e ON l.lead_id = e.lead_id
                WHERE l.lead_id = ? AND e.email_id = ?
            """
            cursor.execute(query, (lead_id, email_id))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"No lead found for lead_id={lead_id}")
                return None

            email, first_name, state, company_name, orig_subject, orig_body, orig_date = result
            company_name = company_name or 'your club'
            
            # Get original scheduled send date
            cursor.execute("""
                SELECT scheduled_send_date 
                FROM emails 
                WHERE email_id = ?
            """, (email_id,))
            result = cursor.fetchone()
            orig_scheduled_date = result[0] if result and result[0] else orig_date

        # If orig_scheduled_date is still None, default to orig_date
        if orig_scheduled_date is None:
            logger.warning("orig_scheduled_date is None, defaulting to orig_date")
            orig_scheduled_date = orig_date

        # Validate required fields
        if not email:
            logger.error("Missing required field: email")
            return None

        # Use RE: with original subject
        subject = f"RE: {orig_subject}"

        # Format the follow-up email body
        body = (
            f"Following up about improving operations at {company_name}. "
            f"Would you have 10 minutes this week for a brief call?\n\n"
            f"Best regards,\n"
            f"Ty\n\n"
            f"Swoop Golf\n"
            f"480-225-9702\n"
            f"swoopgolf.com\n\n"
            f"-------- Original Message --------\n"
            f"Date: {orig_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Subject: {orig_subject}\n"
            f"To: {email}\n\n"
            f"{orig_body}"
        )

        # Calculate base date (3 days after original scheduled date)
        base_date = orig_scheduled_date + timedelta(days=3)
        
        # Get optimal send window
        outreach_window = get_best_outreach_window(
            persona="general",
            geography="US",
        )
        
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        # Adjust to next valid day while preserving the 3-day minimum gap
        while base_date.weekday() not in best_days or base_date < (orig_scheduled_date + timedelta(days=3)):
            base_date += timedelta(days=1)
        
        # Set time within the best window
        send_hour = best_time["start"]
        if random.random() < 0.5:  # 50% chance to use later hour
            send_hour += 1
            
        send_date = base_date.replace(
            hour=send_hour,
            minute=random.randint(0, 59),
            second=0,
            microsecond=0
        )
        
        # Adjust for timezone
        send_date = adjust_send_time(send_date, state) if state else send_date

        logger.debug(f"[followup_generation] Potential scheduled_send_date for lead_id={lead_id} (1st email) is: {send_date}")

        # Calculate base date (3 days after original scheduled date)
        base_date = orig_scheduled_date + timedelta(days=3)
        
        # Get optimal send window
        outreach_window = get_best_outreach_window(
            persona="general",
            geography="US",
        )
        
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        # Adjust to next valid day while preserving the 3-day minimum gap
        while base_date.weekday() not in best_days or base_date < (orig_scheduled_date + timedelta(days=3)):
            base_date += timedelta(days=1)
        
        # Set time within the best window
        send_hour = best_time["start"]
        if random.random() < 0.5:  # 50% chance to use later hour
            send_hour += 1
            
        send_date = base_date.replace(
            hour=send_hour,
            minute=random.randint(0, 59),
            second=0,
            microsecond=0
        )
        
        # Adjust for timezone
        send_date = adjust_send_time(send_date, state) if state else send_date

        logger.debug(f"[followup_generation] Potential scheduled_send_date for lead_id={lead_id} (follow-up) is: {send_date}")

        return {
            'email': email,
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': lead_id,
            'first_name': first_name,
            'state': state
        }

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
