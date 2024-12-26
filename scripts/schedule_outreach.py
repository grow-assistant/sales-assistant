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
    },
    {
        "name": "Follow-Up #2 (Day 7)",
        "days_from_now": 7,
        "subject": "Just Checking In: Swoop Golf",
        "body": "Hi [Name],\n\nI hope this email finds you well. It's been a week since we reached out about Swoop Golf's on-demand F&B solution for your club. Given your role in enhancing member experiences, I wanted to ensure you had a chance to explore how we're helping clubs like yours modernize their on-course service.\n\nWould you be open to a brief conversation this week to discuss how Swoop Golf could benefit your members?"
    },
    {
        "name": "Follow-Up #3 (Day 14)",
        "days_from_now": 14,
        "subject": "Final Note: Swoop Golf's Club Enhancement Opportunity",
        "body": "Hello [Name],\n\nI wanted to send one final note regarding Swoop Golf's on-demand F&B platform. Many clubs we work with have seen significant improvements in both member satisfaction and F&B revenue after implementing our solution.\n\nIf you're interested in learning how we could create similar results for your club, I'm happy to schedule a quick call at your convenience. Otherwise, I'll assume the timing isn't right and won't continue to follow up."
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
