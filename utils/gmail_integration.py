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
from googleapiclient.errors import HttpError

from utils.logging_setup import logger
from datetime import datetime
from scheduling.database import get_db_connection
from typing import Dict, Any
from config import settings
from pathlib import Path
from config.settings import PROJECT_ROOT
from scheduling.extended_lead_storage import store_lead_email_info

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Get Gmail API service."""
    creds = None
    
    # Use absolute paths from PROJECT_ROOT
    credentials_path = Path(PROJECT_ROOT) / 'credentials' / 'credentials.json'
    token_path = Path(PROJECT_ROOT) / 'credentials' / 'token.json'
    
    # Ensure credentials directory exists
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if credentials exist
        if not credentials_path.exists():
            logger.error(f"Missing credentials file at {credentials_path}")
            raise FileNotFoundError(
                f"credentials.json is missing. Please place it in: {credentials_path}"
            )
            
        # The file token.json stores the user's access and refresh tokens
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), 
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
                
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                
        return build('gmail', 'v1', credentials=creds)
        
    except Exception as e:
        logger.error(f"Error setting up Gmail service: {str(e)}")
        raise

def get_gmail_template(service, template_name: str = "sales") -> str:
    """Fetch HTML email template from Gmail drafts."""
    try:
        # First try to list all drafts
        drafts = service.users().drafts().list(userId='me').execute()
        if not drafts.get('drafts'):
            logger.error("No drafts found in Gmail account")
            return ""

        # Log all available drafts for debugging
        draft_subjects = []
        template_html = ""
        
        for draft in drafts.get('drafts', []):
            msg = service.users().messages().get(
                userId='me',
                id=draft['message']['id'],
                format='full'
            ).execute()
            
            # Get subject from headers
            headers = msg['payload']['headers']
            subject = next(
                (h['value'] for h in headers if h['name'].lower() == 'subject'),
                ''
            ).lower()
            draft_subjects.append(subject)
            
            # If this is our template
            if template_name.lower() in subject:
                logger.debug(f"Found template draft with subject: {subject}")
                
                # Extract HTML content
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/html':
                            template_html = base64.urlsafe_b64decode(
                                part['body']['data']
                            ).decode('utf-8')
                            break
                elif msg['payload']['mimeType'] == 'text/html':
                    template_html = base64.urlsafe_b64decode(
                        msg['payload']['body']['data']
                    ).decode('utf-8')
                
                if template_html:
                    return template_html

        if not template_html:
            logger.error(
                f"No template found with name: {template_name}. "
                f"Available draft subjects: {draft_subjects}"
            )
        return template_html

    except Exception as e:
        logger.error(f"Error fetching Gmail template: {str(e)}", exc_info=True)
        return ""

def create_message(to: str, subject: str, body: str) -> Dict[str, str]:
    """Create an HTML-formatted email message using Gmail template."""
    try:
        # Validate inputs
        if not all([to, subject, body]):
            logger.error("Missing required email fields")
            return {}

        # Ensure all inputs are strings
        to = str(to).strip()
        subject = str(subject).strip()
        body = str(body).strip()

        logger.debug("Creating HTML email message")

        # Create the MIME Multipart message
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject
        message["bcc"] = "20057893@bcc.hubspot.com"

        # Format the body text with paragraphs
        div_start = "<div style='margin-bottom: 20px;'>"
        div_end = "</div>"
        formatted_body = div_start + body.replace('\n\n', div_end + div_start) + div_end

        # Get Gmail service and template
        service = get_gmail_service()
        template = get_gmail_template(service, "salesv2")
        
        if not template:
            logger.error("Failed to get Gmail template, using plain text")
            html_body = formatted_body
        else:
            # Look for common content placeholders in the template
            placeholders = ['{{content}}', '{content}', '[content]', '{{body}}', '{body}', '[body]']
            template_with_content = template
            
            # Try each placeholder until one works
            for placeholder in placeholders:
                if placeholder in template:
                    template_with_content = template.replace(placeholder, formatted_body)
                    logger.debug(f"Found and replaced placeholder: {placeholder}")
                    break
            
            if template_with_content == template:
                # No placeholder found, try to insert before the first signature or calendar section
                signature_markers = ['</signature>', 'calendar-section', 'signature-section']
                inserted = False
                
                for marker in signature_markers:
                    if marker in template.lower():
                        parts = template.lower().split(marker, 1)
                        template_with_content = parts[0] + formatted_body + marker + parts[1]
                        inserted = True
                        logger.debug(f"Inserted content before marker: {marker}")
                        break
                
                if not inserted:
                    # If no markers found, prepend content to template
                    template_with_content = formatted_body + template
                    logger.debug("No markers found, prepended content to template")
            
            html_body = template_with_content

        # Create both plain text and HTML versions
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(html_body, 'html')

        # Add both parts to the message
        message.attach(text_part)  # Fallback plain text version
        message.attach(html_part)  # Primary HTML version

        # Encode the message
        raw_message = message.as_string()
        encoded_message = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("utf-8")
        
        logger.debug(f"Created message with HTML length: {len(html_body)}")
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
    """Store draft information using the consolidated storage function."""
    lead_sheet = {"lead_data": {"properties": {"hs_object_id": lead_id}}}
    store_lead_email_info(
        lead_sheet=lead_sheet,
        draft_id=draft_id,
        scheduled_date=scheduled_date,
        subject=subject,
        body=body,
        sequence_num=sequence_num
    )

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
        logger.debug(f"Attempting to retrieve thread with ID: {thread_id}")
        thread_data = service.users().threads().get(userId="me", id=thread_id).execute()
        msgs = thread_data.get("messages", [])
        return len(msgs) > 1
    except HttpError as e:
        if e.resp.status == 404:
            logger.error(f"Thread with ID {thread_id} not found. It may have been deleted or moved.")
        else:
            logger.error(f"HttpError occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error retrieving thread {thread_id}: {e}", exc_info=True)
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

def create_followup_draft(
    sender: str,
    to: str,
    subject: str,
    message_text: str,
    lead_id: str = None,
    sequence_num: int = None,
    original_html: str = None,
    in_reply_to: str = None
) -> Dict[str, Any]:
    try:
        logger.debug(f"Creating follow-up draft for lead_id={lead_id}, sequence_num={sequence_num}")
        
        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return {"status": "error", "error": "No Gmail service"}

        # Create message container
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject
        message["bcc"] = "20057893@bcc.hubspot.com"

        # Add threading headers
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to

        # Split the message text to remove the original content
        new_content = message_text.split("On ", 1)[0].strip()
        
        # Create the HTML parts separately to avoid f-string backslash issues
        html_start = '<div dir="ltr" style="font-family:Arial, sans-serif;">'
        html_content = new_content.replace("\n", "<br>")
        html_quote_start = '<br><br><div class="gmail_quote"><blockquote class="gmail_quote" style="margin:0 0 0 .8ex;border-left:1px solid #ccc;padding-left:1ex">'
        html_quote_content = original_html if original_html else ""
        html_end = '</blockquote></div></div>'

        # Combine HTML parts
        html = html_start + html_content + html_quote_start + html_quote_content + html_end

        # Create both plain text and HTML versions
        text_part = MIMEText(new_content, 'plain')  # Only include new content in plain text
        html_part = MIMEText(html, 'html')

        # Add both parts to the message
        message.attach(text_part)
        message.attach(html_part)

        # Encode and create the draft
        raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8")).decode("utf-8")
        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw_message}}
        ).execute()

        if "id" not in draft:
            return {"status": "error", "error": "No draft ID returned"}

        draft_id = draft["id"]
        
        # Add 'to_review' label
        label_id = get_or_create_label(service, "to_review")
        if label_id:
            service.users().messages().modify(
                userId="me",
                id=draft["message"]["id"],
                body={"addLabelIds": [label_id]},
            ).execute()

        logger.debug(f"Created follow-up draft with id={draft_id}")

        return {
            "status": "ok",
            "draft_id": draft_id,
            "sequence_num": sequence_num,
        }

    except Exception as e:
        logger.error(f"Error in create_followup_draft: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}
