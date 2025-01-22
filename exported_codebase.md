
## test_followup_generation.py

from scheduling.followup_generation import generate_followup_email_xai
from scheduling.database import get_db_connection
from utils.logging_setup import logger
from utils.gmail_integration import create_draft, create_followup_draft
from scheduling.extended_lead_storage import store_lead_email_info

def test_followup_generation():
    """Test generating a follow-up email from an existing email in the database"""
    try:
        # Get a sample email from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent sent email
        cursor.execute("""
            SELECT TOP 1
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date
            FROM emails
            WHERE gmail_id IS NOT NULL
            AND sequence_num = 1
            ORDER BY created_at DESC
        """)
        
        row = cursor.fetchone()
        if not row:
            logger.error("No sent emails found in database")
            return
            
        lead_id, email, name, gmail_id, scheduled_date = row
        logger.info(f"Found original email to {email} (Lead ID: {lead_id})")
        
        # Generate follow-up email
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            original_email={
                'email': email,
                'name': name,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date
            }
        )
        
        if not followup:
            logger.error("Failed to generate follow-up email")
            return
            
        # Create draft in Gmail using the new follow-up function
        draft_result = create_followup_draft(
            sender="me",
            to=followup['email'],
            subject=followup['subject'],
            message_text=followup['body'],
            lead_id=followup['lead_id'],
            sequence_num=followup['sequence_num']
        )
        
        if draft_result["status"] == "ok":
            logger.info(f"Created follow-up draft for {email}")
            logger.info(f"Draft ID: {draft_result['draft_id']}")
            logger.info(f"Scheduled send date: {followup['scheduled_send_date']}")
            
            # Store the follow-up in database
            store_lead_email_info(
                lead_sheet={
                    "lead_data": {
                        "properties": {
                            "hs_object_id": followup['lead_id'],
                            "firstname": name.split()[0] if name else "",
                            "lastname": name.split()[1] if name and len(name.split()) > 1 else ""
                        }
                    },
                    "company_data": {
                        "name": name
                    }
                },
                draft_id=draft_result['draft_id'],
                scheduled_date=followup['scheduled_send_date'],
                body=followup['body'],
                sequence_num=followup['sequence_num']
            )
            logger.info("Stored follow-up in database")
        else:
            logger.error(f"Failed to create Gmail draft: {draft_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in test_followup_generation: {str(e)}", exc_info=True)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    print("\nStarting follow-up generation test...")
    test_followup_generation() 



## scheduling\database.py

# scheduling/database.py

import sys
from pathlib import Path
import pyodbc
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_setup import logger
from config.settings import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

SERVER = DB_SERVER
DATABASE = DB_NAME
UID = DB_USER
PWD = DB_PASSWORD

def get_db_connection():
    """Get database connection."""
    logger.debug("Connecting to SQL Server", extra={
        "database": DATABASE,
        "server": SERVER,
        "masked_credentials": True
    })
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={UID};"
        f"PWD={PWD}"
    )
    try:
        conn = pyodbc.connect(conn_str)
        logger.debug("SQL connection established successfully.")
        return conn
    except pyodbc.Error as ex:
        logger.error("Error connecting to SQL Server", extra={
            "error": str(ex),
            "error_type": type(ex).__name__,
            "database": DATABASE,
            "server": SERVER
        }, exc_info=True)
        raise

def init_db():
    """Initialize database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        logger.info("Starting init_db...")

        # Create emails table if it doesn't exist
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                         WHERE object_id = OBJECT_ID(N'[dbo].[emails]') 
                         AND type in (N'U'))
            BEGIN
                CREATE TABLE dbo.emails (
                    email_id            INT IDENTITY(1,1) PRIMARY KEY,
                    lead_id            INT NOT NULL,
                    name               VARCHAR(100),
                    email_address      VARCHAR(255),
                    sequence_num       INT NULL,
                    body               VARCHAR(MAX),
                    scheduled_send_date DATETIME NULL,
                    actual_send_date   DATETIME NULL,
                    created_at         DATETIME DEFAULT GETDATE(),
                    status             VARCHAR(50) DEFAULT 'pending',
                    draft_id           VARCHAR(100) NULL,
                    gmail_id           VARCHAR(100)
                )
            END
        """)
        conn.commit()
        logger.info("init_db completed successfully. Emails table created if it didn't exist.")
        
    except Exception as e:
        logger.error("Error in init_db", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "database": DATABASE
        }, exc_info=True)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def clear_tables():
    """Clear the emails table in the database."""
    try:
        with get_db_connection() as conn:
            logger.debug("Clearing emails table")
            
            query = "DELETE FROM dbo.emails"
            logger.debug(f"Executing: {query}")
            conn.execute(query)
                
            logger.info("Successfully cleared emails table")

    except Exception as e:
        logger.exception(f"Failed to clear emails table: {str(e)}")
        raise e

def store_email_draft(cursor, lead_id: int, name: str = None,
                     email_address: str = None,
                     sequence_num: int = None,
                     body: str = None,
                     scheduled_send_date: datetime = None,
                     draft_id: str = None,
                     status: str = 'pending') -> int:
    """
    Store email draft in database. Returns email_id.
    
    Table schema:
    - email_id (auto-generated)
    - lead_id
    - name
    - email_address
    - sequence_num
    - body
    - scheduled_send_date
    - actual_send_date (auto-managed)
    - created_at (auto-managed)
    - status
    - draft_id
    - gmail_id (managed elsewhere)
    """
    # First check if this draft_id already exists
    cursor.execute("""
        SELECT email_id FROM emails 
        WHERE draft_id = ? AND lead_id = ?
    """, (draft_id, lead_id))
    
    existing = cursor.fetchone()
    if existing:
        # Update existing record instead of creating new one
        cursor.execute("""
            UPDATE emails 
            SET name = ?,
                email_address = ?,
                sequence_num = ?,
                body = ?,
                scheduled_send_date = ?,
                status = ?
            WHERE draft_id = ? AND lead_id = ?
        """, (
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id,
            lead_id
        ))
        return existing[0]
    else:
        # Insert new record
        cursor.execute("""
            INSERT INTO emails (
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                status,
                draft_id
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
        """, (
            lead_id,
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id
        ))
        cursor.execute("SELECT SCOPE_IDENTITY()")
        return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")




## scheduling\followup_generation.py

# followup_generation.py

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft, get_gmail_service
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import (
    get_best_outreach_window,
    adjust_send_time,
    calculate_send_date
)
from datetime import datetime, timedelta
import random
import base64


def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None,
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI and original Gmail message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    email_address,
                    name,
                    body,
                    gmail_id,
                    scheduled_send_date,
                    draft_id
                FROM emails
                WHERE lead_id = ?
                AND sequence_num = 1
                AND gmail_id IS NOT NULL
                ORDER BY created_at DESC
            """, (lead_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"No original email found for lead_id={lead_id}")
                return None

            email_address, name, body, gmail_id, scheduled_date, draft_id = row
            original_email = {
                'email': email_address,
                'name': name,
                'body': body,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'draft_id': draft_id
            }

        # Get the Gmail service
        gmail_service = get_gmail_service()
        
        # Get the original message from Gmail
        try:
            message = gmail_service.users().messages().get(
                userId='me',
                id=original_email['gmail_id'],
                format='full'
            ).execute()
            
            # Get the original HTML content
            original_html = None
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/html':
                        original_html = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            elif message['payload']['mimeType'] == 'text/html':
                original_html = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
            # Extract subject and original message content
            headers = message.get('payload', {}).get('headers', [])
            orig_subject = next(
                (header['value'] for header in headers if header['name'].lower() == 'subject'),
                'No Subject'
            )
            
            # Get the original sender (me) and timestamp
            from_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'from'),
                'me'
            )
            date_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'date'),
                ''
            )
            
            # Get original message body
            if 'parts' in message['payload']:
                orig_body = ''
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        orig_body = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            else:
                orig_body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error fetching Gmail message: {str(e)}", exc_info=True)
            return None

        # Format the follow-up email with the original included
        name = original_email.get('name', 'your club')
        subject = f"Re: {orig_subject}"
        
        body = (
            f"Following up about improving operations at {name}. "
            f"Would you have 10 minutes this week for a brief call?\n\n"
            f"Best regards,\n"
            f"Ty\n\n"
            f"On {date_header}, {from_header} wrote:\n"
            f"{orig_body}"
        )

        # Calculate send date using golf_outreach_strategy logic
        send_date = calculate_send_date(
            geography="Year-Round Golf",
            persona="General Manager",
            state="AZ",
            season_data=None
        )

        # Ensure minimum 3-day gap from original send date
        orig_scheduled_date = original_email.get('scheduled_send_date', datetime.now())
        while send_date < (orig_scheduled_date + timedelta(days=3)):
            send_date += timedelta(days=1)

        return {
            'email': original_email.get('email'),
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': lead_id,
            'name': name,
            'in_reply_to': original_email['gmail_id'],
            'original_html': original_html
        }

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()




## utils\gmail_integration.py

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

def create_followup_draft(
    sender: str,
    to: str,
    subject: str,
    message_text: str,
    lead_id: str = None,
    sequence_num: int = None,
    original_html: str = None  # Add parameter for original HTML
) -> Dict[str, Any]:
    """Create a follow-up email draft with proper Gmail threading format."""
    try:
        logger.debug(
            f"Creating follow-up draft with subject='{subject}', body length={len(message_text)}"
        )

        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return {"status": "error", "error": "No Gmail service"}

        # Create message container
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject
        message["bcc"] = "20057893@bcc.hubspot.com"

        # Split the message text into new content and quoted content
        new_content, _, quoted_content = message_text.partition("On ")
        
        # Create the HTML with proper Gmail quote styling
        html = f"""
        <div dir="ltr">
            {new_content.strip().replace("\n", "<br>")}
            <div class="gmail_quote">
                <blockquote class="gmail_quote" 
                    style="margin:0 0 0 .8ex;border-left:1px solid #ccc;padding-left:1ex">
                    {original_html if original_html else quoted_content.replace("\n", "<br>")}
                </blockquote>
            </div>
        </div>
        """

        # Create both plain text and HTML versions
        text_part = MIMEText(message_text, 'plain')
        html_part = MIMEText(html, 'html')

        # Add both parts to the message
        message.attach(text_part)
        message.attach(html_part)

        # Encode the message
        raw_message = message.as_string()
        encoded_message = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("utf-8")
        
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": encoded_message}})
            .execute()
        )

        if "id" not in draft:
            logger.error("Draft creation returned no ID")
            return {"status": "error", "error": "No draft ID returned"}

        draft_id = draft["id"]
        logger.debug(f"Created follow-up draft with id={draft_id}")

        # Add the "to_review" label
        label_id = get_or_create_label(service, "to_review")
        if label_id:
            try:
                service.users().messages().modify(
                    userId="me",
                    id=draft["message"]["id"],
                    body={"addLabelIds": [label_id]},
                ).execute()
                logger.debug(f"Added 'to_review' label to draft message")
            except Exception as e:
                logger.error(f"Failed to add label to draft: {str(e)}")

        return {
            "status": "ok",
            "draft_id": draft_id,
            "sequence_num": sequence_num,
        }

    except Exception as e:
        logger.error(f"Error in create_followup_draft: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

