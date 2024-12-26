# schedule_outreach.py

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

# Follow-up email schedule with specific focus for each stage
OUTREACH_SCHEDULE = [
    {
        "name": "Intro Email (Day 1)",
        "days_from_now": 1,
        "subject": "Enhancing Member Experience with Swoop Golf",
        "body": "Hi [Name],\n\nI hope this finds you well. I wanted to reach out about Swoop Golf's on-demand F&B platform, which is helping golf clubs like yours modernize their on-course service and enhance member satisfaction.\n\nWould you be open to a brief conversation about how we could help streamline your club's F&B operations while improving the member experience?"
    },
    {
        "name": "Quick Follow-Up (Day 3)",
        "days_from_now": 3,
        "subject": "Quick follow-up: Swoop Golf",
        "body": "Hello [Name],\n\nI wanted to quickly follow up on my previous email about Swoop Golf's F&B platform. Have you had a chance to consider how our solution might benefit your club's operations?\n\nI'd be happy to schedule a brief call to discuss your specific needs."
    },
    {
        "name": "Value-Add Follow-Up (Day 7)",
        "days_from_now": 7,
        "subject": "Success Story: How Clubs Are Transforming F&B with Swoop Golf",
        "body": "Hi [Name],\n\nI wanted to share a quick success story: One of our partner clubs saw a 40% increase in on-course F&B revenue and significantly improved member satisfaction scores within just three months of implementing Swoop Golf.\n\nI'd love to discuss how we could achieve similar results for your club. Would you be open to a brief conversation this week?"
    },
    {
        "name": "Final Check-In (Day 14)",
        "days_from_now": 14,
        "subject": "Final Note: Swoop Golf Opportunity",
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
