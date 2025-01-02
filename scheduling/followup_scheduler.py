# scheduling/followup_scheduler.py

import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from scheduling.database import get_db_connection, store_email_draft
from scheduling.followup_generation import generate_followup_email_xai
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from config.settings import SEND_EMAILS, ENABLE_FOLLOWUPS
import logging

def check_and_send_followups():
    """Check for and send any pending follow-up emails"""
    if not ENABLE_FOLLOWUPS:
        logger.info("Follow-up emails are disabled via ENABLE_FOLLOWUPS setting")
        return

    logger.debug("Running check_and_send_followups")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get leads needing follow-up with all required fields
        cursor.execute("""
            SELECT 
                e.lead_id,
                e.email_id,
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
            WHERE e.sequence_num = 1
            AND e.status = 'sent'
            AND l.email IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM emails e2 
                WHERE e2.lead_id = e.lead_id 
                AND e2.sequence_num = 2
            )
        """)
        
        for row in cursor.fetchall():
            lead_id, email_id, email, first_name, company_name, subject, body, created_at, state = row
            
            # Package original email data
            original_email = {
                'email': email,
                'first_name': first_name,
                'name': company_name,
                'subject': subject,
                'body': body,
                'created_at': created_at,
                'state': state
            }
            
            # Generate follow-up content
            followup_data = generate_followup_email_xai(
                lead_id=lead_id,
                email_id=email_id,
                sequence_num=2,
                original_email=original_email
            )
            
            if followup_data and followup_data.get('scheduled_send_date'):
                # Create Gmail draft
                draft_result = create_draft(
                    sender="me",
                    to=followup_data['email'],
                    subject=followup_data['subject'],
                    message_text=followup_data['body']
                )

                if draft_result and draft_result.get("status") == "ok":
                    # Store in database with scheduled_send_date
                    store_email_draft(
                        cursor,
                        lead_id=lead_id,
                        subject=followup_data['subject'],
                        body=followup_data['body'],
                        scheduled_send_date=followup_data['scheduled_send_date'],
                        sequence_num=followup_data['sequence_num'],
                        draft_id=draft_result["draft_id"],
                        status='draft'
                    )
                    conn.commit()
                    logger.info(f"Follow-up scheduled for lead_id={lead_id} at {followup_data['scheduled_send_date']}")
            else:
                logger.error(f"Missing scheduled_send_date for lead_id={lead_id}")

    except Exception as e:
        logger.exception("Error in followup scheduler")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def start_scheduler():
    """Initialize and start the follow-up scheduler"""
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            check_and_send_followups,
            'interval',
            minutes=15,
            id='check_and_send_followups',
            next_run_time=datetime.datetime.now()
        )
        
        # Suppress initial scheduler messages
        logging.getLogger('apscheduler').setLevel(logging.WARNING)
        
        scheduler.start()
        logger.info("Follow-up scheduler initialized")
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")

if __name__ == "__main__":
    start_scheduler()
