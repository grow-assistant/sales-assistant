# File: scripts/check_reviewed_drafts.py
"""
Purpose: Monitors Gmail drafts for review status and generates follow-up emails when appropriate.

Logical steps:
1. Check all draft emails in SQL with sequence_num = 1
2. Verify Gmail labels for each draft
3. If labeled as 'reviewed', generate follow-up draft (sequence_num = 2)
4. Store new drafts with 'to_review' label
5. Skip drafts already labeled 'to_review'
6. Update database records accordingly
"""
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now import project modules
from scheduling.database import get_db_connection
from utils.gmail_integration import get_gmail_service, create_draft
from utils.logging_setup import logger
from services.data_gatherer_service import DataGathererService
from scheduling.followup_generation import generate_followup_email_xai
from utils.gmail_integration import store_draft_info
from scripts.golf_outreach_strategy import get_best_outreach_window
from datetime import datetime
from typing import Optional
import random

###########################
# CONFIG / CONSTANTS
###########################
TO_REVIEW_LABEL = "to_review"
REVIEWED_LABEL = "reviewed"
SQL_DRAFT_STATUS = "draft"

def main():
    """
    Checks all 'draft' emails in the SQL table with sequence_num = 1.
    If an email's Gmail label is 'reviewed', we generate a follow-up draft
    (sequence_num = 2) and store it with label='to_review'.
    If the label is 'to_review', we skip it.
    """
    try:
        # 1) Connect to DB
        conn = get_db_connection()
        cursor = conn.cursor()

        logger.info("Starting 'check_reviewed_drafts' workflow.")

        # 2) Fetch all "draft" emails with sequence_num=1
        #    that presumably need to be reviewed or followed up.
        cursor.execute("""
            SELECT email_id, lead_id, draft_id, subject, body
            FROM emails
            WHERE status = ?
              AND sequence_num = 1
        """, (SQL_DRAFT_STATUS,))
        results = cursor.fetchall()

        if not results:
            logger.info("No draft emails (sequence_num=1) found.")
            return

        logger.info(f"Found {len(results)} draft emails with sequence_num=1.")

        # 3) Get Gmail service
        gmail_service = get_gmail_service()
        if not gmail_service:
            logger.error("Could not initialize Gmail service.")
            return

        # 4) For each email draft, check the label in Gmail
        for (email_id, lead_id, draft_id, subject, body) in results:
            logger.info(f"Processing email_id={email_id}, draft_id={draft_id}")

            if not draft_id:
                logger.warning("No draft_id found in DB. Skipping.")
                continue

            # 4a) Retrieve the draft message from Gmail
            message = _get_gmail_draft_by_id(gmail_service, draft_id)
            if not message:
                logger.error(f"Failed to retrieve Gmail draft for draft_id={draft_id}")
                continue

            # 4b) Extract current labels
            current_labels = message.get("labelIds", [])
            if not current_labels:
                logger.info(f"No labels found for draft_id={draft_id}. Skipping.")
                continue

            # Normalize label IDs to strings
            label_names = _translate_label_ids_to_names(gmail_service, current_labels)
            logger.debug(f"Draft {draft_id} has labels: {label_names}")

            # 5) If label == "to_review", skip
            #    If label == "reviewed", create a follow-up draft
            if REVIEWED_LABEL.lower() in [ln.lower() for ln in label_names]:
                logger.info(f"Draft {draft_id} has label '{REVIEWED_LABEL}'. Creating follow-up.")
                
                # 5a) Generate follow-up (sequence_num=2)
                followup_data = _generate_followup(gmail_service, lead_id, email_id, subject, body)
                if followup_data:
                    logger.info("Follow-up created successfully.")
                else:
                    logger.error("Failed to create follow-up.")
            else:
                # If we only see "to_review", skip
                if TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
                    logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. Skipping.")
                else:
                    logger.info(f"Draft {draft_id} has no matching logic for labels={label_names}")

        logger.info("Completed checking reviewed drafts.")

    except Exception as e:
        logger.exception(f"Error in check_reviewed_drafts workflow: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

###########################
# HELPER FUNCTIONS
###########################

def _get_gmail_draft_by_id(service, draft_id: str) -> Optional[dict]:
    """
    Retrieve a specific Gmail draft by its draftId.
    Returns None if not found or error.
    """
    try:
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        return draft.get("message", {})
    except Exception as e:
        logger.error(f"Error fetching draft {draft_id}: {str(e)}")
        return None


def _translate_label_ids_to_names(service, label_ids: list) -> list:
    """
    Given a list of labelIds, returns the corresponding label names.
    For example: ["Label_12345"] -> ["to_review"].
    """
    # Retrieve all labels
    try:
        labels_response = service.users().labels().list(userId='me').execute()
        all_labels = labels_response.get('labels', [])
        id_to_name = {lbl["id"]: lbl["name"] for lbl in all_labels}

        label_names = []
        for lid in label_ids:
            label_names.append(id_to_name.get(lid, lid))  # fallback to ID if not found
        return label_names
    except Exception as e:
        logger.error(f"Error translating label IDs: {str(e)}")
        return label_ids  # fallback

def _generate_followup(gmail_service, lead_id: int, original_email_id: int, orig_subject: str, orig_body: str) -> bool:
    """
    Generates a follow-up draft (sequence_num=2) and stores it as a new
    Gmail draft with label 'to_review'.
    """
    try:
        # 1) Get lead data from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT l.email, l.first_name, c.name, c.state
            FROM leads l
            LEFT JOIN companies c ON l.company_id = c.company_id
            WHERE l.lead_id = ?
        """, (lead_id,))
        
        lead_data = cursor.fetchone()
        if not lead_data:
            logger.error(f"No lead found for lead_id={lead_id}")
            return False
            
        lead_email, first_name, company_name, state = lead_data
        
        # 2) Build the original email data structure with proper datetime object
        original_email = {
            "email": lead_email,
            "first_name": first_name,
            "name": company_name,
            "state": state,
            "subject": orig_subject,
            "body": orig_body,
            "created_at": datetime.now()  # Use datetime object instead of string
        }

        # 3) Generate follow-up content
        followup_data = generate_followup_email_xai(
            lead_id=lead_id,
            email_id=original_email_id,
            sequence_num=2,
            original_email=original_email
        )
        
        if not followup_data or not followup_data.get("scheduled_send_date"):
            logger.error("No valid followup_data generated.")
            return False

        # 4) Create new Gmail draft
        draft_result = create_draft(
            sender="me",
            to=lead_email,
            subject=followup_data['subject'],
            message_text=followup_data['body']
        )
        
        if draft_result.get("status") != "ok":
            logger.error("Failed to create Gmail draft for follow-up.")
            return False

        new_draft_id = draft_result["draft_id"]
        logger.info(f"Follow-up draft created. draft_id={new_draft_id}")

        # 5) Store the new draft in DB
        store_draft_info(
            lead_id=lead_id,
            draft_id=new_draft_id,
            scheduled_date=followup_data.get('scheduled_send_date'),
            subject=followup_data['subject'],
            body=followup_data['body'],
            sequence_num=2
        )

        # 6) Add label 'to_review' to the new draft
        _add_label_to_message(
            gmail_service,
            new_draft_id,
            label_name=TO_REVIEW_LABEL
        )

        return True

    except Exception as e:
        logger.exception(f"Error generating follow-up for lead_id={lead_id}: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def _add_label_to_message(service, draft_id: str, label_name: str):
    """
    Adds a label to a Gmail draft message. We'll need to fetch the message
    ID from the draft, then modify the labels.
    """
    try:
        # 1) Get the draft
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        message_id = draft["message"]["id"]

        # 2) Get or create the label
        label_id = _get_or_create_label(service, label_name)
        if not label_id:
            logger.warning(f"Could not find/create label '{label_name}'. Skipping label add.")
            return

        # 3) Apply label to the message
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id]}
        ).execute()
        logger.debug(f"Label '{label_name}' added to new follow-up draft_id={draft_id}.")

    except Exception as e:
        logger.error(f"Error adding label '{label_name}' to draft '{draft_id}': {str(e)}")

def _get_or_create_label(service, label_name: str) -> Optional[str]:
    """
    Retrieves or creates the specified Gmail label and returns its labelId.
    """
    try:
        user_id = 'me'
        labels_response = service.users().labels().list(userId=user_id).execute()
        existing_labels = labels_response.get('labels', [])

        # Try finding existing label by name
        for lbl in existing_labels:
            if lbl['name'].lower() == label_name.lower():
                return lbl['id']

        # If not found, create it
        create_body = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show',
        }
        new_label = service.users().labels().create(
            userId=user_id, body=create_body
        ).execute()
        logger.info(f"Created new Gmail label: {label_name}")
        return new_label['id']

    except Exception as e:
        logger.error(f"Error in _get_or_create_label({label_name}): {str(e)}")
        return None


if __name__ == "__main__":
    main()
