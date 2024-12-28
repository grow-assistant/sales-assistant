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

def schedule_draft(step_details, sender, recipient, hubspot_contact_id):
    """
    Create a Gmail draft for scheduled sending.
    """
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
    """
    Main scheduling workflow:
     1) Determine the best outreach window for an example persona/season/club type
     2) Start APScheduler
     3) Schedule each step of the outreach
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
    hubspot_contact_id = "255401"   # Example contact ID in HubSpot

    # Schedule each step for future sending
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
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
