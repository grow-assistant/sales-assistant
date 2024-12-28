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
from utils.timezone_utils import adjust_for_timezone
from scripts.golf_outreach_strategy import get_best_outreach_window

# Example outreach schedule steps
OUTREACH_SCHEDULE = [
    {
        "name": "Email #1 (Day 0)",
        "days_from_now": 0,
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
    """
    Create a Gmail draft for scheduled sending based on lead data and outreach strategy.
    
    Args:
        step_details (dict): Details about the outreach step
        sender (str): Email sender
        recipient (str): Email recipient
        hubspot_contact_id (str): HubSpot contact ID
        lead_data (dict): Lead context data containing properties and analysis
    """
    # Get role and club type from lead data
    role = lead_data.get("properties", {}).get("jobtitle", "General Manager")
    club_type = lead_data.get("analysis", {}).get("facilities", {}).get("club_type", "Public Courses")
    geography = lead_data.get("analysis", {}).get("season_data", {}).get("season_label", "Year-Round Golf")

    # Get recommended outreach window
    recommendation = get_best_outreach_window(role, geography, club_type)
    logger.info(
        "Determined outreach window",
        extra={
            "role": role,
            "club_type": club_type,
            "geography": geography,
            "best_month": recommendation["Best Month"],
            "best_time": recommendation["Best Time"],
            "best_day": recommendation["Best Day"]
        }
    )

    draft_result = create_draft(
        sender=sender,
        to=recipient,
        subject=step_details["subject"],
        message_text=step_details["body"]
    )

    if draft_result["status"] != "ok":
        logger.error(
            f"Failed to create draft for step '{step_details['name']}'.",
            extra={"hubspot_contact_id": hubspot_contact_id}
        )
        return

    draft_id = draft_result.get("draft_id")
    
    # Calculate base send time
    base_send_time = datetime.datetime.now() + datetime.timedelta(days=step_details["days_from_now"])
    
    # Get state from lead data and adjust for timezone
    state = lead_data.get("company_data", {}).get("state", "")
    ideal_send_time = adjust_for_timezone(base_send_time, state) if state else base_send_time
    
    logger.info(
        f"Created draft for outreach step",
        extra={
            "draft_id": draft_id,
            "step_name": step_details["name"],
            "send_time": str(ideal_send_time),
            "hubspot_contact_id": hubspot_contact_id
        }
    )


def main(lead_data=None):
    """
    Main scheduling workflow:
     1) Determine the best outreach window based on lead's role and club type
     2) Start APScheduler
     3) Schedule each step of the outreach
     
    Args:
        lead_data (dict): Optional lead context data. If not provided, uses example data.
    """
    if lead_data is None:
        # Example lead data for testing
        lead_data = {
            "properties": {
                "jobtitle": "General Manager",
                "email": "someone@example.com",
                "hs_object_id": "255401"
            },
            "analysis": {
                "facilities": {"club_type": "Private Clubs"},
                "season_data": {"season_label": "Year-Round Golf"}
            }
        }
        logger.warning("Using example lead data - no lead_data provided")

    # Start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()

    now = datetime.datetime.now()
    sender = "me"  # 'me' means the authenticated Gmail user
    recipient = lead_data["properties"].get("email", "someone@example.com")
    hubspot_contact_id = lead_data["properties"].get("hs_object_id", "255401")

    # Schedule each step for future sending
    for step in OUTREACH_SCHEDULE:
        run_time = now + datetime.timedelta(days=step["days_from_now"])
        job_id = f"job_{step['name'].replace(' ', '_')}"

        scheduler.add_job(
            schedule_draft,
            'date',
            run_date=run_time,
            id=job_id,
            args=[step, sender, recipient, hubspot_contact_id, lead_data]
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
