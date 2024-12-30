# scheduling/followup_scheduler.py

import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from scheduling.database import get_db_connection
from scheduling.followup_generation import generate_followup_email_xai
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from config.settings import SEND_EMAILS
import logging

def print_scheduling_status(scheduled_emails):
    """Prints a human-readable summary of scheduled emails and their status"""
    print("\n=== Email Scheduling Status ===")
    print(f"Found {len(scheduled_emails)} scheduled emails")
    print("----------------------------")
    
    for email in scheduled_emails:
        email_id, scheduled_date, status, recipient = email
        print(f"\nEmail ID: {email_id}")
        print(f"Recipient: {recipient}")
        print(f"Scheduled for: {scheduled_date}")
        print(f"Current status: {status}")
    
    if not scheduled_emails:
        print("No emails currently scheduled")
    print("============================\n")

def check_and_send_followups():
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n🔍 Checking for followups at {now_str}")
    print(f"Email sending is {'ENABLED' if SEND_EMAILS else 'DISABLED'}\n")

    try:
        # Debug query to see ALL emails
        cursor.execute("""
            SELECT 
                e.email_id,
                e.scheduled_send_date,
                e.status,
                l.email,
                l.status as lead_status,
                e.created_at
            FROM emails e
            LEFT JOIN leads l ON l.lead_id = e.lead_id
            ORDER BY e.scheduled_send_date
        """)
        all_emails = cursor.fetchall()
        
        # print("\n=== Current Email Schedule ===")
        # for email in all_emails:
        #     print(f"Email ID: {email[0]}")
        #     print(f"To: {email[3]}")
        #     print(f"Scheduled: {email[1]}")
        #     print(f"Status: {email[2]}")
        #     print(f"Created: {email[5]}")
        #     print("---")
        # print("============================\n")

        # Debug query to see all scheduled emails
        cursor.execute("""
            SELECT e.email_id, e.scheduled_send_date, e.status, l.email 
            FROM emails e
            JOIN leads l ON l.lead_id = e.lead_id
            WHERE e.scheduled_send_date <= ? 
            ORDER BY e.scheduled_send_date
        """, (now_str,))
        scheduled_emails = cursor.fetchall()
        
        # Print scheduling status
        #print_scheduling_status(scheduled_emails)

        if scheduled_emails:
            logger.info("Found scheduled emails:", extra={
                "emails": [{
                    "email_id": e[0],
                    "scheduled_date": e[1],
                    "status": e[2],
                    "recipient": e[3]
                } for e in scheduled_emails]
            })
        else:
            logger.info("No scheduled emails found for current time window")

        # Debug query to see all emails
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_count = cursor.fetchone()[0]
        logger.debug(f"Total emails in database: {total_count}")

        # Debug query to see email statuses
        cursor.execute("SELECT status, COUNT(*) FROM emails GROUP BY status")
        status_counts = cursor.fetchall()
        logger.debug(f"Email status counts: {dict(status_counts)}")

        # Original query with added logging
        query = """
            SELECT COUNT(*) 
            FROM emails e
            JOIN leads l ON l.lead_id = e.lead_id
            WHERE e.scheduled_send_date <= ?
              AND e.status = 'pending'
              AND l.status = 'active'
              AND e.actual_send_date IS NULL
        """
        cursor.execute(query, (now_str,))
        pending_count = cursor.fetchone()[0]
        
        logger.info(f"Found {pending_count} pending emails to process")

        # Then get the actual rows
        rows = cursor.execute("""
            SELECT e.email_id,
                   e.lead_id,
                   e.subject,
                   e.body,
                   e.status,
                   e.sequence_num,
                   l.email,
                   e.scheduled_send_date
            FROM emails e
            JOIN leads l ON l.lead_id = e.lead_id
            WHERE e.scheduled_send_date <= ?
              AND e.status = 'pending'
              AND l.status = 'active'
              AND e.actual_send_date IS NULL
        """, (now_str,)).fetchall()

        for row in rows:
            email_id, lead_id, subject, body, status, seq_num, recipient, send_date = row
            
            logger.info("Processing scheduled email", extra={
                "email_id": email_id,
                "recipient": recipient,
                "scheduled_date": str(send_date),
                "send_emails_enabled": SEND_EMAILS
            })
            
            if not SEND_EMAILS:
                logger.info("Email sending disabled - would have sent email", extra={
                    "email_id": email_id,
                    "recipient": recipient,
                    "subject": subject
                })
                continue
            
            # Create Gmail draft
            draft_res = create_draft(
                sender="me",
                to=recipient,
                subject=subject,
                message_text=body
            )
            
            if draft_res["status"] == "ok":
                cursor.execute("""
                    UPDATE emails
                    SET status = 'draft_created',
                        draft_id = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE email_id = ?
                """, (draft_res.get("draft_id"), email_id))
                conn.commit()
                logger.info(f"Created draft for email_id={email_id} to {recipient}")
            else:
                logger.error(f"Failed to create draft", extra={
                    "email_id": email_id,
                    "recipient": recipient,
                    "error": draft_res.get("error", "Unknown error")
                })

    except Exception as e:
        logger.error("Error in followup scheduler", extra={
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        conn.rollback()
    finally:
        cursor.close()
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

def store_email_draft(cursor, lead_id, subject, body, scheduled_send_date):
    """Store email draft in database with proper date handling"""
    try:
        # Ensure datetime has no timezone info
        if hasattr(scheduled_send_date, 'tzinfo'):
            scheduled_send_date = scheduled_send_date.replace(tzinfo=None)
            
        cursor.execute("""
            INSERT INTO emails (
                lead_id, 
                subject, 
                body, 
                status,
                scheduled_send_date,
                created_at,
                last_updated
            ) VALUES (?, ?, ?, 'pending', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (lead_id, subject, body, scheduled_send_date.strftime("%Y-%m-%d %H:%M:%S")))
        
        email_id = cursor.lastrowid
        print(f"\n=== Email Scheduling Confirmation ===")
        print(f"Email ID: {email_id}")
        print(f"Scheduled for: {scheduled_send_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Status: pending")
        print("============================\n")
        
        return email_id
    except Exception as e:
        logger.error("Failed to store email draft", extra={
            "error": str(e),
            "lead_id": lead_id,
            "scheduled_date": scheduled_send_date
        })
        raise

if __name__ == "__main__":
    start_scheduler()
