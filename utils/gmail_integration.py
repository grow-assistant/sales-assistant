import os
import base64
import os.path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from utils.logging_setup import logger
from datetime import datetime
from scheduling.database import get_db_connection
from typing import Dict, Any
from config import settings

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Authenticate and return a Gmail API service instance."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e:
            logger.error(f"Error reading token.json: {e}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                logger.error("Error refreshing token. Delete token.json and re-run authentication.")
                raise
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds, cache_discovery=False)

def create_message(to: str, subject: str, body: str) -> Dict[str, str]:
    """Create an HTML-formatted email message."""
    try:
        # Validate inputs
        if not all([to, subject, body]):
            logger.error(
                "Missing required email fields",
                extra={"has_to": bool(to), "has_subject": bool(subject), "has_body": bool(body)},
            )
            return {}

        # Ensure all inputs are strings
        to = str(to).strip()
        subject = str(subject).strip()
        body = str(body).strip()

        logger.debug(
            "Creating HTML email message",
            extra={"to": to, "subject": subject, "body_length": len(body)},
        )

        # Create the MIME Multipart message
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject

        # Format the HTML body with inline CSS
        formatted_body = body.replace('\n\n', '</p><p>').replace('\n', '<br>')
        html_body = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333333; }}
                    p {{ margin: 1em 0; }}
                    .signature {{ margin-top: 20px; color: #666666; }}
                    .company-info {{ margin-top: 10px; }}
                </style>
            </head>
            <body>
                <p>{formatted_body}</p>
            </body>
        </html>
        """

        # Create both plain text and HTML versions
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(html_body, 'html')

        # Add both parts to the message
        message.attach(text_part)  # Fallback plain text version
        message.attach(html_part)  # Primary HTML version

        # Encode as base64url
        raw_message = message.as_string()
        encoded_message = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("utf-8")

        return {"raw": encoded_message}

    except Exception as e:
        logger.exception(f"Error creating email message: {str(e)}")
        return {}

def get_or_create_label(service, label_name: str = "to_review") -> str:
    """
    Retrieve or create a Gmail label and return its labelId.
    """
    try:
        user_id = 'me'
        # 1) List existing labels and log them
        labels_response = service.users().labels().list(userId=user_id).execute()
        existing_labels = labels_response.get('labels', [])
        
        logger.debug(f"Found {len(existing_labels)} existing labels:")
        for lbl in existing_labels:
            logger.debug(f"  - '{lbl['name']}' (id: {lbl['id']})")

        # 2) Case-insensitive search for existing label
        label_name_lower = label_name.lower()
        for lbl in existing_labels:
            if lbl['name'].lower() == label_name_lower:
                logger.debug(f"Found existing label '{lbl['name']}' with id={lbl['id']}")
                return lbl['id']

        # 3) Create new label if not found
        logger.debug(f"No existing label found for '{label_name}', creating new one...")
        label_body = {
            'name': label_name,  # Use original case for creation
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show',
        }
        created_label = service.users().labels().create(
            userId=user_id, body=label_body
        ).execute()

        logger.debug(f"Created new label '{label_name}' with id={created_label['id']}")
        return created_label["id"]

    except Exception as e:
        logger.error(f"Error in get_or_create_label: {str(e)}", exc_info=True)
        return ""

def create_draft(
    sender: str,
    to: str,
    subject: str,
    message_text: str,
    lead_id: str = None,
    sequence_num: int = None,
) -> Dict[str, Any]:
    """
    Create a Gmail draft email, add the 'to_review' label, and optionally store in DB.
    """
    try:
        logger.debug(
            f"Creating draft with subject='{subject}', body length={len(message_text)}"
        )

        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return {"status": "error", "error": "No Gmail service"}

        # 1) Create the MIME email message
        message = create_message(to=to, subject=subject, body=message_text)
        if not message:
            logger.error("Failed to create message")
            return {"status": "error", "error": "Failed to create message"}

        # 2) Create the actual draft
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": message})
            .execute()
        )

        if "id" not in draft:
            logger.error("Draft creation returned no ID")
            return {"status": "error", "error": "No draft ID returned"}

        draft_id = draft["id"]
        draft_message_id = draft["message"]["id"]
        logger.debug(f"Created draft with id={draft_id}, message_id={draft_message_id}")

        # 3) Store draft info in the DB if lead_id is provided
        if settings.CREATE_FOLLOWUP_DRAFT and lead_id:
            store_draft_info(
                lead_id=lead_id,
                draft_id=draft_id,
                scheduled_date=None,
                subject=subject,
                body=message_text,
                sequence_num=sequence_num,
            )
        else:
            logger.info("Follow-up draft creation is disabled via CREATE_FOLLOWUP_DRAFT setting")

        # 4) Add the "to_review" label to the underlying draft message
        label_id = get_or_create_label(service, "to_review")
        if label_id:
            try:
                service.users().messages().modify(
                    userId="me",
                    id=draft_message_id,
                    body={"addLabelIds": [label_id]},
                ).execute()
                logger.debug(f"Added '{label_id}' label to draft message {draft_message_id}")
            except Exception as e:
                logger.error(f"Failed to add label to draft: {str(e)}")
        else:
            logger.warning("Could not get/create 'to_review' label - draft remains unlabeled")

        return {
            "status": "ok",
            "draft_id": draft_id,
            "sequence_num": sequence_num,
        }

    except Exception as e:
        logger.error(f"Error in create_draft: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

def get_lead_email(lead_id: str) -> str:
    """Get a lead's email from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Updated query to use correct column names
        cursor.execute("""
            SELECT email 
            FROM leads 
            WHERE lead_id = ?
        """, (lead_id,))
        
        result = cursor.fetchone()

        if not result:
            logger.error(f"No email found for lead_id={lead_id}")
            return ""

        return result[0]

    except Exception as e:
        logger.error(f"Error getting lead email: {str(e)}")
        return ""
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()

def store_draft_info(
    lead_id: str,
    draft_id: str,
    scheduled_date: datetime,
    subject: str,
    body: str,
    sequence_num: int = None,
):
    """Store draft information in the database (emails table)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Inserting into existing 'emails' table
        cursor.execute(
            """
            INSERT INTO emails (
                lead_id,
                draft_id,
                scheduled_send_date,
                subject,
                body,
                status,
                sequence_num
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                draft_id,
                scheduled_date,
                subject,
                body,
                "draft",  # set initial status as draft
                sequence_num,
            ),
        )

        conn.commit()
        logger.debug(f"Successfully stored draft info for lead_id={lead_id}")

    except Exception as e:
        logger.error(f"Error storing draft info: {str(e)}")
        conn.rollback()
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()

def send_message(sender, to, subject, message_text) -> Dict[str, Any]:
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(to=to, subject=subject, body=message_text)

    try:
        sent_msg = (
            service.users()
            .messages()
            .send(userId="me", body=message_body)
            .execute()
        )
        if sent_msg.get("id"):
            logger.info(f"Message sent successfully to '{to}' – ID={sent_msg['id']}")
            return {"status": "ok", "message_id": sent_msg["id"]}
        else:
            logger.error(f"Message sent to '{to}' but no ID returned – possibly an API error.")
            return {"status": "error", "error": "No message ID returned"}
    except Exception as e:
        logger.error(
            f"Failed to send message to '{to}' with subject='{subject}'. Error: {e}"
        )
        return {"status": "error", "error": str(e)}

def search_messages(query="") -> list:
    """
    Search for messages in the Gmail inbox using the specified `query`.
    For example:
      - 'from:someone@example.com'
      - 'subject:Testing'
      - 'to:me newer_than:7d'
    Returns a list of message dicts.
    """
    service = get_gmail_service()
    try:
        response = service.users().messages().list(userId="me", q=query).execute()
        return response.get("messages", [])
    except Exception as e:
        logger.error(f"Error searching messages with query='{query}': {e}")
        return []

def check_thread_for_reply(thread_id: str) -> bool:
    """
    Checks if there's more than one message in a given thread, indicating a reply.
    More precise than searching by 'from:' or date alone.
    """
    service = get_gmail_service()
    try:
        thread_data = service.users().threads().get(userId="me", id=thread_id).execute()
        msgs = thread_data.get("messages", [])
        return len(msgs) > 1
    except Exception as e:
        logger.error(f"Error retrieving thread {thread_id}: {e}")
        return False

def search_inbound_messages_for_email(email_address: str, max_results: int = 1) -> list:
    """
    Search for inbound messages sent from `email_address`.
    Returns a list of short snippets from the most recent matching messages.
    """
    query = f"from:{email_address}"
    message_ids = search_messages(query=query)
    if not message_ids:
        return []

    service = get_gmail_service()
    snippets = []
    for m in message_ids[:max_results]:
        try:
            full_msg = service.users().messages().get(
                userId="me",
                id=m["id"],
                format="full",
            ).execute()
            snippet = full_msg.get("snippet", "")
            snippets.append(snippet)
        except Exception as e:
            logger.error(f"Error fetching message {m['id']} from {email_address}: {e}")

    return snippets

def get_signature() -> str:
    """Return HTML-formatted signature block."""
    return """
        <div class="signature">
            Best regards,<br>
            Ty<br>
            <div class="company-info">
                Swoop Golf<br>
                480-225-9702<br>
                <a href="https://swoopgolf.com">swoopgolf.com</a>
            </div>
        </div>
    """
