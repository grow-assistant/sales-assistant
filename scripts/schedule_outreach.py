"""
scripts/schedule_outreach.py

Schedules multiple outreach steps (emails) via APScheduler.
Now also integrates best outreach-window logic from golf_outreach_strategy.
"""

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from scheduling.database import get_db_connection

# NEW IMPORT
from scripts.golf_outreach_strategy import get_best_outreach_window

# Example outreach schedule steps
OUTREACH_SCHEDULE = [
    {
        "name": "Intro Email (Day 1)",
        "days_from_now": 1,
        "subject": "Enhancing Member Experience with Swoop Golf",
        "body": (
            "Hi [Name],\n\n"
            "I hope this finds you well. I wanted to reach out about Swoop Golf's on-demand F&B platform, "
            "which is helping golf clubs like yours modernize their on-course service and enhance member satisfaction.\n\n"
            "Would you be open to a brief conversation about how we could help streamline your club's F&B "
            "operations while improving the member experience?"
        )
    },
    {
        "name": "Quick Follow-Up (Day 3)",
        "days_from_now": 3,
        "subject": "Quick follow-up: Swoop Golf",
        "body": (
            "Hello [Name],\n\n"
            "I wanted to quickly follow up on my previous email about Swoop Golf's F&B platform. "
            "Have you had a chance to consider how our solution might benefit your club's operations?\n\n"
            "I'd be happy to schedule a brief call to discuss your specific needs."
        )
    },
    # Add additional follow-up steps as needed...
]

def schedule_draft(step_details, sender, recipient, lead_id, sequence_num=None):
    """
    Create a Gmail draft for scheduled sending and store in emails table.
    
    Args:
        step_details (dict): Email step configuration
        sender (str): Email sender
        recipient (str): Email recipient
        lead_id (int): Lead ID for database storage
        sequence_num (int, optional): Follow-up sequence number (0=initial, 1=day3, etc.)
    """
    # Use step index as sequence_num if not provided
    if sequence_num is None:
        sequence_num = OUTREACH_SCHEDULE.index(step_details)
        
    draft_result = create_draft(
        sender=sender,
        to=recipient,
        subject=step_details["subject"],
        message_text=step_details["body"]
    )

    if draft_result["status"] != "ok":
        logger.error(f"Failed to create draft for step '{step_details['name']}'", extra={
            "lead_id": lead_id,
            "sequence_num": sequence_num,
            "step_name": step_details["name"]
        })
        return

    draft_id = draft_result.get("draft_id")
    ideal_send_time = datetime.datetime.now() + datetime.timedelta(days=step_details["days_from_now"])
    
    # Store in database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emails (lead_id, subject, body, scheduled_send_date, sequence_num, draft_id, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (lead_id, step_details["subject"], step_details["body"], 
              ideal_send_time, sequence_num, draft_id))
        conn.commit()
        
        logger.info(f"Created and stored draft for step '{step_details['name']}'", extra={
            "lead_id": lead_id,
            "sequence_num": sequence_num,
            "draft_id": draft_id,
            "scheduled_time": ideal_send_time.isoformat(),
            "step_name": step_details["name"]
        })
    except Exception as e:
        logger.error(f"Failed to store email in database: {str(e)}", extra={
            "lead_id": lead_id,
            "sequence_num": sequence_num,
            "draft_id": draft_id
        })
    finally:
        conn.close()


def main():
    """
    Main scheduling workflow:
     1) Determine the best outreach window for an example persona/season/club type
     2) Start APScheduler
     3) Load lead data from database
     4) Schedule each step of the outreach with proper lead context
    
    Note: This example uses a hardcoded lead_id. In production, this should
    be passed as an argument or loaded from a configuration source.
    """

    # Example of fetching recommended outreach window
    persona = "General Manager"
    geography = "Peak Summer Season"
    club_type = "Private Clubs"

    recommendation = get_best_outreach_window(persona, geography, club_type)
    logger.info(
        f"Recommended outreach for {persona}, {geography}, {club_type}: {recommendation}"
    )

    # If you want to incorporate recommended times/days into your scheduling logic,
    # you can parse or handle them here (for example, adjust 'days_from_now' or
    # specific times of day, etc.).

    # Start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()

    now = datetime.datetime.now()
    sender = "me"                   # 'me' means the authenticated Gmail user
    recipient = "someone@example.com"
    lead_id = 255401               # Example lead ID from database

    # Get lead data from database
    conn = get_db_connection()
    cursor = conn.cursor()
    lead_data = cursor.execute("""
        SELECT lead_id, email, first_name, last_name, role
        FROM leads 
        WHERE lead_id = ?
    """, (lead_id,)).fetchone()
    conn.close()

    if not lead_data:
        logger.error(f"Lead ID {lead_id} not found in database")
        return

    # Use actual lead email if available
    recipient = lead_data["email"] if lead_data else recipient

    # Schedule each step for future sending
    for step in OUTREACH_SCHEDULE:
        run_time = now + datetime.timedelta(days=step["days_from_now"])
        job_id = f"job_{step['name'].replace(' ', '_')}"

        scheduler.add_job(
            schedule_draft,
            'date',
            run_date=run_time,
            id=job_id,
            args=[step, sender, recipient, lead_id]
        )
        logger.info(f"Scheduled job '{job_id}' for {run_time}", extra={
            "lead_id": lead_id,
            "step_name": step["name"],
            "scheduled_time": run_time.isoformat()
        })

    try:
        logger.info("Scheduler running. Press Ctrl+C to exit.")
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
