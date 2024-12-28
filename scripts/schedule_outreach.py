"""
scripts/schedule_outreach.py

Schedules multiple outreach steps via APScheduler.
Integrates best outreach-window logic from golf_outreach_strategy.
"""

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from utils.timezone_utils import adjust_for_timezone
from scripts.golf_outreach_strategy import get_best_outreach_window
from utils.db_connection import get_db_connection

# Example outreach schedule steps
OUTREACH_SCHEDULE = [
    {
        "name": "Email #1 (Day 0)",
        "days_from_now": 0,
        "sequence_num": 0,
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
        "name": "Follow-Up #1 (Day 3–4)",
        "days_from_now": 3,
        "sequence_num": 1,
        "subject": "Quick follow-up: Swoop Golf",
        "body": (
            "Hello [Name],\n\n"
            "I wanted to quickly follow up on my previous email about Swoop Golf's F&B platform. "
            "Have you had a chance to consider how our solution might benefit your club's operations?\n\n"
            "I'd be happy to schedule a brief call to discuss your specific needs."
        )
    },
    {
        "name": "Follow-Up #2 (Day 7)",
        "days_from_now": 7,
        "sequence_num": 2,
        "subject": "Member Experience Improvements with Swoop Golf",
        "body": (
            "Hi [Name],\n\n"
            "It's been about a week since we reached out about Swoop Golf's F&B platform. "
            "I wanted to share that clubs similar to yours have seen significant improvements in both member satisfaction "
            "and operational efficiency after implementing our solution.\n\n"
            "Would you be interested in learning more about these member experiences and how they might apply to your club?"
        )
    },
    {
        "name": "Follow-Up #3 (Day 14)",
        "days_from_now": 14,
        "sequence_num": 3,
        "subject": "Final note: Swoop Golf platform",
        "body": (
            "Hi [Name],\n\n"
            "I wanted to send one final note regarding Swoop Golf's F&B platform. Many clubs have seen improvements in "
            "both member satisfaction and revenue after implementing our solution.\n\n"
            "If you'd like to explore how we could help enhance your club's F&B operations, please don't hesitate to reach out.\n\n"
            "Thank you for your time."
        )
    }
]


def schedule_draft(step_details, sender, recipient, hubspot_contact_id, lead_data):
    try:
        # Get lead_id from email
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT lead_id FROM leads WHERE email = ?", (recipient,))
        result = cursor.fetchone()
        lead_id = result[0] if result else None

        if not lead_id:
            logger.error(f"No lead_id found for email: {recipient}")
            return

        # Calculate send time with timezone adjustment
        base_send_time = datetime.datetime.now() + datetime.timedelta(days=step_details["days_from_now"])
        state = lead_data.get("company_data", {}).get("state", "")
        scheduled_send_time = adjust_for_timezone(base_send_time, state) if state else base_send_time

        logger.info("Scheduling draft", extra={
            "step": step_details["name"],
            "sequence_num": step_details["sequence_num"],
            "scheduled_time": str(scheduled_send_time),
            "recipient": recipient
        })

        # Create the draft
        draft_result = create_draft(sender=sender, to=recipient, 
                                  subject=step_details["subject"], 
                                  message_text=step_details["body"])

        # Insert into database
        cursor.execute("""
            INSERT INTO dbo.emails (
                lead_id, subject, body, scheduled_send_date, draft_id, sequence_num, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'draft_created')
        """, (
            lead_id,
            step_details["subject"],
            step_details["body"],
            scheduled_send_time,
            draft_result.get("draft_id"),
            step_details["sequence_num"]
        ))
        
        conn.commit()
        logger.info("Draft scheduled successfully", extra={
            "step": step_details["name"],
            "sequence_num": step_details["sequence_num"],
            "scheduled_time": str(scheduled_send_time),
            "recipient": recipient
        })

    except Exception as e:
        logger.error("Failed to schedule draft", extra={
            "error": str(e),
            "step": step_details.get("name"),
            "recipient": recipient
        })
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def main(lead_data=None):
    """Main scheduling workflow"""
    if lead_data is None:
        lead_data = {
            "properties": {
                "jobtitle": "General Manager",
                "email": "jhiggins@valleylo.org",
                "hs_object_id": "15537350171"
            },
            "company_data": {
                "state": "IL"
            }
        }
    
    logger.info("Starting outreach scheduling", extra={
        "recipient": lead_data["properties"].get("email"),
        "steps": len(OUTREACH_SCHEDULE)
    })
    
    # Initialize and start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()
    
    # Schedule each step for a future run
    sender = "me"
    recipient = lead_data["properties"].get("email")
    for step in OUTREACH_SCHEDULE:
        run_date = datetime.datetime.now() + datetime.timedelta(days=step["days_from_now"])
        job_id = f"{step['name'].replace(' ', '_')}_{recipient}"
        
        scheduler.add_job(
            schedule_draft,
            'date',
            run_date=run_date,
            id=job_id,
            args=[step, sender, recipient, lead_data["properties"].get("hs_object_id"), lead_data]
        )
        
        logger.info(f"Scheduled job '{job_id}' for {run_date}", extra={"step": step["name"]})
    
    logger.info("All steps have been scheduled. Press Ctrl+C to exit.")

    # Keep the script alive so scheduled jobs run
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
