import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from datetime import datetime
import pytz
from utils.gmail_integration import get_gmail_service
from utils.logging_setup import logger
from scheduling.database import get_db_connection
import base64

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

def get_draft_body(message: dict) -> str:
    """Extract the body text from a Gmail draft message."""
    try:
        body = ""
        if 'payload' in message and 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        elif 'payload' in message and 'body' in message['payload']:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
        
        if not body:
            logger.error("No body content found in message")
            return ""
            
        # Log the first few characters to help with debugging
        preview = body[:100] + "..." if len(body) > 100 else body
        logger.debug(f"Retrieved draft body preview: {preview}")
        
        # Check for template placeholders
        if "[firstname]" in body or "{{" in body or "}}" in body:
            logger.warning("Found template placeholders in draft body - draft may not be properly personalized")
            
        return body
        
    except Exception as e:
        logger.error(f"Error extracting draft body: {str(e)}")
        return ""

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

def update_email_status(email_id: int, new_status: str, body: str = None):
    """
    Update the emails table with the new status and optionally update the body.
    Similar to update_email_record in check_for_sent_emails.py
    """
    try:
        if body:
            sql = """
                UPDATE emails
                   SET status = ?,
                       body = ?
                 WHERE email_id = ?
            """
            params = (new_status, body, email_id)
        else:
            sql = """
                UPDATE emails
                   SET status = ?
                 WHERE email_id = ?
            """
            params = (new_status, email_id)
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Email ID {email_id} updated: status='{new_status}'" + 
                       (", body updated" if body else ""))
    except Exception as e:
        logger.error(f"Error updating email ID {email_id}: {str(e)}", exc_info=True)

def delete_orphaned_draft_records():
    """
    Delete records from the emails table that have status='draft' but their
    corresponding Gmail drafts no longer exist.
    """
    try:
        # Get Gmail service
        gmail_service = get_gmail_service()
        if not gmail_service:
            logger.error("Could not initialize Gmail service.")
            return

        # Get all draft records
        drafts = get_draft_emails()
        if not drafts:
            logger.info("No draft emails found to check.")
            return

        logger.info(f"Checking {len(drafts)} draft records for orphaned entries")

        # Get list of all Gmail drafts
        try:
            gmail_drafts = gmail_service.users().drafts().list(userId='me').execute()
            existing_draft_ids = {d['id'] for d in gmail_drafts.get('drafts', [])}
        except Exception as e:
            logger.error(f"Error fetching Gmail drafts: {str(e)}")
            return

        # Track which records to delete
        orphaned_records = []
        for record in drafts:
            if not record['draft_id'] or record['draft_id'] not in existing_draft_ids:
                orphaned_records.append(record['email_id'])
                logger.debug(f"Found orphaned record: email_id={record['email_id']}, draft_id={record['draft_id']}")

        if not orphaned_records:
            logger.info("No orphaned draft records found.")
            return

        # Instead of deleting, print what would be deleted
        print(f"\nWould delete the following orphaned records:")
        print(f"Email IDs: {orphaned_records}")
        print(f"Total records that would be deleted: {len(orphaned_records)}")
        
        # Comment out the deletion code
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(orphaned_records))
                cursor.execute(f
                    DELETE FROM emails
                     WHERE email_id IN ({placeholders})
                       AND status = ?
                , (*orphaned_records, SQL_DRAFT_STATUS))
                conn.commit()
                logger.info(f"Deleted {cursor.rowcount} orphaned draft records")
        except Exception as e:
            logger.error(f"Error deleting orphaned records: {str(e)}", exc_info=True)
        """

    except Exception as e:
        logger.error(f"Error in delete_orphaned_draft_records: {str(e)}", exc_info=True)

###############################################################################
#                               MAIN PROCESS
###############################################################################

def main():
    """Check Gmail drafts for labels and update SQL status accordingly."""
    logger.info("=== Starting Gmail label check process ===")

    # First, clean up any orphaned draft records
    delete_orphaned_draft_records()

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
            logger.info(f"Draft {draft_id} marked as '{REVIEWED_LABEL}'. Updating SQL status and body.")
            # Get the draft body text
            body = get_draft_body(message)
            if body:
                logger.debug(f"Retrieved body text ({len(body)} chars) for draft_id={draft_id}")
                # Update both status and body
                update_email_status(email_id, SQL_REVIEWED_STATUS, body)
            else:
                logger.warning(f"Could not retrieve body text for draft_id={draft_id}")
                # Update only status if no body could be retrieved
                update_email_status(email_id, SQL_REVIEWED_STATUS)
        elif TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. No action needed.")
        else:
            logger.info(f"Draft {draft_id} has no relevant labels: {label_names}")

    logger.info("\n=== Completed label check process ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()
