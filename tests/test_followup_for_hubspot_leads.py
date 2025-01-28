# tests/test_followup_for_hubspot_leads.py

import sys
import random
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from test_hubspot_leads_service import get_random_lead_id
from scheduling.database import get_db_connection
from scheduling.followup_generation import generate_followup_email_xai
from services.gmail_service import GmailService
from utils.gmail_integration import create_followup_draft
from scheduling.extended_lead_storage import store_lead_email_info
from utils.logging_setup import logger


def generate_followup_email_with_injection(lead_id: int, original_email: dict) -> dict:
    """
    Wraps generate_followup_email_xai but specifically places the new follow-up
    text *underneath the top reply* and *above the original email* at the bottom.
    """
    # Call your existing function to fetch the default follow-up structure
    followup = generate_followup_email_xai(
        lead_id=lead_id,
        original_email=original_email
    )
    if not followup:
        return {}

    # The followup already contains everything we need, just return it
    return followup


def create_followup_for_unreplied_leads():
    """
    1. Pull leads from the DB (sequence_num=1).
    2. Check if they replied.
    3. If not replied, generate and store follow-up.
    """
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch leads that have a first-sequence email with valid Gmail info
        cursor.execute("""
            SELECT
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date,
                sequence_num,
                company_short_name,
                body
            FROM emails
            WHERE sequence_num = 1
              AND gmail_id IS NOT NULL
              AND company_short_name IS NOT NULL
            ORDER BY created_at ASC
        """)

        rows = cursor.fetchall()
        if not rows:
            logger.error("No sequence=1 emails found in the database.")
            return

        logger.info(f"Found {len(rows)} leads. Checking replies & generating follow-ups if needed...")

        gmail_service = GmailService()

        for idx, row in enumerate(rows, start=1):
            (lead_id, email, name, gmail_id, scheduled_date,
             seq_num, company_short_name, body) = row

            logger.info(f"[{idx}] Checking Lead ID: {lead_id} | Email: {email}")

            # 1) Check if there's a reply in the thread
            replies = gmail_service.search_replies(gmail_id)
            if replies:
                logger.info(f"[{idx}] Lead ID: {lead_id} has replied. Skipping follow-up.")
                continue

            # 2) If no reply, build a dictionary for the original email info
            original_email = {
                'email': email,
                'name': name,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'company_short_name': company_short_name,
                'body': body
            }

            # 3) Generate the follow-up with your injection in the middle
            followup = generate_followup_email_with_injection(
                lead_id=lead_id,
                original_email=original_email
            )
            if not followup:
                logger.error(f"[{idx}] Failed to generate follow-up for Lead ID: {lead_id}")
                continue

            # 4) Create the Gmail draft
            draft_result = create_followup_draft(
                sender="me",
                to=followup['email'],
                subject=followup['subject'],
                message_text=followup['body'],
                lead_id=str(lead_id),
                sequence_num=followup.get('sequence_num', 2),
                original_html=followup.get('original_html'),
                in_reply_to=followup['in_reply_to']
            )

            # 5) Update DB with the new draft info
            if draft_result.get('draft_id'):
                store_lead_email_info(
                    lead_sheet={
                        'lead_data': {
                            'properties': {'hs_object_id': lead_id},
                            'email': email
                        },
                        'company_data': {
                            'company_short_name': company_short_name
                        }
                    },
                    draft_id=draft_result['draft_id'],
                    scheduled_date=followup['scheduled_send_date'],
                    body=followup['body'],
                    sequence_num=followup.get('sequence_num', 2)
                )
                logger.info(f"[{idx}] Successfully created/stored follow-up draft for Lead ID: {lead_id}")
            else:
                logger.error(f"[{idx}] Failed creating Gmail draft for Lead ID: {lead_id}: "
                             f"{draft_result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error during follow-up creation: {str(e)}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    create_followup_for_unreplied_leads()
