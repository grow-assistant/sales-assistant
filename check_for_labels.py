import os
import sys
from datetime import datetime
import pytz
from utils.gmail_integration import get_gmail_service
from utils.logging_setup import logger
from scheduling.database import get_db_connection

###############################################################################
#                           CONSTANTS
###############################################################################

TO_REVIEW_LABEL = "to_review"
REVIEWED_LABEL = "reviewed"
SQL_DRAFT_STATUS = "draft"
SQL_REVIEWED_STATUS = "reviewed"

###############################################################################
#                           GMAIL INTEGRATION
###############################################################################

def get_draft_emails() -> list:
    """
    Retrieve all emails with status='draft'.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email_id,
                       lead_id,
                       name,
                       email_address,
                       sequence_num,
                       body,
                       scheduled_send_date,
                       actual_send_date,
                       created_at,
                       status,
                       draft_id,
                       gmail_id
                  FROM emails
                 WHERE status = ?
            """, (SQL_DRAFT_STATUS,))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                record = {
                    'email_id': row[0],
                    'lead_id': row[1],
                    'name': row[2],
                    'email_address': row[3],
                    'sequence_num': row[4],
                    'body': row[5],
                    'scheduled_send_date': str(row[6]) if row[6] else None,
                    'actual_send_date': str(row[7]) if row[7] else None,
                    'created_at': str(row[8]) if row[8] else None,
                    'status': row[9],
                    'draft_id': row[10],
                    'gmail_id': row[11]
                }
                results.append(record)
            return results

    except Exception as e:
        logger.error(f"Error retrieving draft emails: {str(e)}", exc_info=True)
        return []

def get_gmail_draft_by_id(service, draft_id: str) -> dict:
    """Retrieve a specific Gmail draft by its draftId."""
    try:
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        return draft.get("message", {})
    except Exception as e:
        logger.error(f"Error fetching draft {draft_id}: {str(e)}")
        return {}

def translate_label_ids_to_names(service, label_ids: list) -> list:
    """Convert Gmail label IDs to their corresponding names."""
    try:
        labels_response = service.users().labels().list(userId='me').execute()
        all_labels = labels_response.get('labels', [])
        id_to_name = {lbl["id"]: lbl["name"] for lbl in all_labels}

        label_names = []
        for lid in label_ids:
            label_names.append(id_to_name.get(lid, lid))
        return label_names
    except Exception as e:
        logger.error(f"Error translating label IDs: {str(e)}")
        return label_ids

def update_email_status(email_id: int, new_status: str):
    """Update the status field in the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE emails
                   SET status = ?
                 WHERE email_id = ?
            """, (new_status, email_id))
            conn.commit()
            logger.info(f"Updated email_id={email_id} status to '{new_status}'")
    except Exception as e:
        logger.error(f"Error updating email status for ID {email_id}: {str(e)}", exc_info=True)

###############################################################################
#                               MAIN PROCESS
###############################################################################

def main():
    """Check Gmail drafts for labels and update SQL status accordingly."""
    logger.info("=== Starting Gmail label check process ===")

    # Get all draft emails from SQL
    pending = get_draft_emails()
    if not pending:
        logger.info("No draft emails found. Exiting.")
        return

    logger.info(f"Found {len(pending)} draft emails to process")

    # Get Gmail service
    gmail_service = get_gmail_service()
    if not gmail_service:
        logger.error("Could not initialize Gmail service.")
        return

    # Process each record
    for record in pending:
        email_id = record['email_id']
        draft_id = record['draft_id']

        logger.info(f"\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Draft ID: {draft_id}")

        if not draft_id:
            logger.warning(f"No draft_id found for email_id={email_id}. Skipping.")
            continue

        # Get the Gmail draft and its labels
        message = get_gmail_draft_by_id(gmail_service, draft_id)
        if not message:
            logger.error(f"Failed to retrieve Gmail draft for draft_id={draft_id}")
            continue

        # Get current labels
        current_labels = message.get("labelIds", [])
        if not current_labels:
            logger.info(f"No labels found for draft_id={draft_id}. Skipping.")
            continue

        # Convert label IDs to names
        label_names = translate_label_ids_to_names(gmail_service, current_labels)
        logger.debug(f"Draft {draft_id} has labels: {label_names}")

        # Check for reviewed label and update SQL status
        if REVIEWED_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} marked as '{REVIEWED_LABEL}'. Updating SQL status.")
            update_email_status(email_id, SQL_REVIEWED_STATUS)
        elif TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. No action needed.")
        else:
            logger.info(f"Draft {draft_id} has no relevant labels: {label_names}")

    logger.info("\n=== Completed label check process ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()
