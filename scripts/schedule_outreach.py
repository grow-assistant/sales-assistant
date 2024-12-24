# schedule_outreach.py

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

# Example schedule: day offsets and email contents
OUTREACH_SCHEDULE = [
    {
        "name": "Email #1 (Day 0)",
        "days_from_now": 0,
        "subject": "Initial Outreach – Let's Chat about Swoop Golf",
        "body": "Hi [Name],\n\nWe discussed Swoop Golf previously..."
    },
    {
        "name": "Follow-Up #1 (Day 3–4)", 
        "days_from_now": 3,
        "subject": "Checking In on Swoop Golf",
        "body": "Hello [Name],\n\nJust wanted to follow up..."
    }
]

def schedule_draft(step_details, sender, recipient, hubspot_contact_id):
    """
    Create a Gmail draft for scheduled sending.
    """
    # Create the Gmail draft
    draft_result = create_draft(
        sender=sender,
        to=recipient,
        subject=step_details["subject"],
        message_text=step_details["body"]
    )
    
    if draft_result["status"] != "ok":
        logger.error(f"Failed to create draft for step '{step_details['name']}'.")
        return

    draft_id = draft_result.get("draft_id")
    ideal_send_time = datetime.datetime.now() + datetime.timedelta(days=step_details["days_from_now"])
    
    logger.info(f"Created draft ID: {draft_id} for step '{step_details['name']}' to be sent at {ideal_send_time}.")

def main():
    # Modify these as needed for your environment
    sender = "me"  # 'me' means the authenticated Gmail user
    recipient = "kowen@capitalcityclub.org"
    hubspot_contact_id = "255401"  # Example contact ID in HubSpot

    scheduler = BackgroundScheduler()
    scheduler.start()

    now = datetime.datetime.now()
    for step in OUTREACH_SCHEDULE:
        run_time = now + datetime.timedelta(days=step["days_from_now"])
        job_id = f"job_{step['name'].replace(' ', '_')}"
        
        scheduler.add_job(
            schedule_draft,
            'date',
            run_date=run_time,
            id=job_id,
            args=[step, sender, recipient, hubspot_contact_id]
        )
        logger.info(f"Scheduled job '{job_id}' for {run_time}")

    try:
        logger.info("Scheduler running. Press Ctrl+C to exit.")
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

if __name__ == "__main__":
    main()
