# scheduling/followup_scheduler.py

import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from scheduling.database import get_db_connection
from followup_generation import generate_followup_email_xai
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

def check_and_send_followups():
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Adjusted T-SQL style and placeholders
    rows = cursor.execute("""
        SELECT f.followup_id,
               f.lead_id,
               f.sequence_num,
               f.subject,
               f.body,
               f.status,
               l.email,
               l.status AS lead_status
        FROM followups f
        JOIN leads l ON l.lead_id = f.lead_id
        WHERE f.scheduled_send_date <= ?
          AND f.status = 'pending'
          AND l.status = 'active'
    """, (now_str,)).fetchall()

    for row in rows:
        (followup_id, lead_id, seq_num, subject, body, fup_status, email, lead_status) = row

        # If subject/body not populated, call xAI
        if not subject or not body:
            logger.info(f"Generating xAI content for followup_id={followup_id}")
            generate_followup_email_xai(lead_id, seq_num)
            # Re-fetch the updated record
            cursor.execute("""
                SELECT subject, body
                FROM followups
                WHERE followup_id = ?
            """, (followup_id,))
            updated = cursor.fetchone()
            subject, body = updated if updated else (subject, body)

        # Final check
        if not subject or not body:
            logger.warning(f"Skipping followup_id={followup_id} - no subject/body.")
            continue

        # Create Gmail draft
        draft_res = create_draft(
            sender="me",
            to=email,
            subject=subject,
            message_text=body
        )
        if draft_res["status"] == "ok":
            logger.info(f"Draft created for followup_id={followup_id}")
            # Update status
            cursor.execute("""
                UPDATE followups
                SET status = 'draft_created'
                WHERE followup_id = ?
            """, (followup_id,))
            conn.commit()
        else:
            logger.error(f"Failed to create draft for followup_id={followup_id}")

    conn.close()

def start_scheduler():
    """Starts a background scheduler to check for due followups every 15 minutes."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_send_followups, 'interval', minutes=15)
    scheduler.start()

    logger.info("Follow-up scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Follow-up scheduler stopped.")

if __name__ == "__main__":
    start_scheduler()
