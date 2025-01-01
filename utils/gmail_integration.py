import os
import base64
import os.path
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from utils.logging_setup import logger
import requests
import json
from datetime import datetime
from scheduling.database import get_db_connection
from typing import Dict, Any

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


def create_message(to: str, subject: str, body: str) -> dict:
    """
    Create a message for an email.
    Args:
        to: Email address of the receiver.
        subject: The subject of the email.
        body: The body of the email message.
    Returns:
        An object containing a base64url encoded email object.
    """
    try:
        # Validate inputs
        if not all([to, subject, body]):
            logger.error("Missing required email fields", extra={
                "has_to": bool(to),
                "has_subject": bool(subject),
                "has_body": bool(body)
            })
            return None

        # Ensure all inputs are strings
        to = str(to).strip()
        subject = str(subject).strip()
        body = str(body).strip()

        logger.debug("Creating email message", extra={
            "to": to,
            "subject": subject,
            "body_length": len(body)
        })

        # Create the message
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        # Create the raw email message
        raw_message = message.as_string()
        
        # Encode in base64url format
        encoded_message = base64.urlsafe_b64encode(raw_message.encode('utf-8')).decode('utf-8')
        
        return {'raw': encoded_message}

    except Exception as e:
        logger.exception(f"Error creating email message: {str(e)}")
        return None


def create_draft(sender: str, to: str, subject: str, message_text: str, lead_id: str = None, sequence_num: int = None) -> Dict[str, Any]:
    """
    Create a Gmail draft email
    
    Args:
        sender: Sender's email or lead_id
        to: Recipient's email
        subject: Email subject
        message_text: Email body text
        lead_id: Optional lead ID for database tracking
        sequence_num: Optional sequence number for follow-ups
    """
    try:
        logger.debug(f"Creating draft with subject='{subject}', body length={len(message_text)}")

        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return None

        # Create the email message
        message = create_message(to=to, subject=subject, body=message_text)
        if not message:
            logger.error("Failed to create email message", extra={
                "lead_id": lead_id,
                "subject_length": len(subject) if subject else 0,
                "body_length": len(message_text) if message_text else 0
            })
            return None

        try:
            # Create the draft
            draft = service.users().drafts().create(userId='me', body={'message': message}).execute()
            draft_id = draft['id']
            
            # Store draft info in database if lead_id provided
            if lead_id:
                store_draft_info(
                    lead_id=lead_id,
                    draft_id=draft_id,
                    scheduled_date=None,  # Will be set by scheduler
                    subject=subject,
                    body=message_text,
                    sequence_num=sequence_num
                )
            
            return {
                "status": "ok",
                "draft_id": draft_id,
                "sequence_num": sequence_num
            }
            
        except Exception as e:
            logger.error(f"Error creating Gmail draft: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    except Exception as e:
        logger.error(f"Error in create_draft: {str(e)}", exc_info=True)
        return None


def get_lead_email(lead_id: str) -> str:
    """Get lead's email from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT email FROM leads WHERE lead_id = ?", (lead_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"No email found for lead_id={lead_id}")
            return None
            
        return result[0]
        
    except Exception as e:
        logger.error(f"Error getting lead email: {str(e)}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def store_draft_info(lead_id: str, draft_id: str, scheduled_date: datetime, subject: str, body: str, sequence_num: int = None):
    """Store draft information in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Using the existing 'emails' table instead of 'email_drafts'
        cursor.execute("""
            INSERT INTO emails (
                lead_id, 
                draft_id, 
                scheduled_send_date, 
                subject, 
                body,
                status,
                sequence_num
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            lead_id, 
            draft_id, 
            scheduled_date, 
            subject, 
            body,
            'draft',  # Set initial status as draft
            sequence_num  # Add sequence number to insert
        ))
        
        conn.commit()
        logger.debug(f"Successfully stored draft info for lead_id={lead_id}")
        
    except Exception as e:
        logger.error(f"Error storing draft info: {str(e)}")
        conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def send_message(sender, to, subject, message_text):
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(sender, to, message_text)
    try:
        sent_msg = service.users().messages().send(
            userId='me',
            body=message_body
        ).execute()
        if sent_msg.get('id'):
            logger.info(f"Message sent successfully to '{to}' – ID={sent_msg['id']}")
            return {"status": "ok", "message_id": sent_msg.get('id')}
        else:
            logger.error(f"Message sent to '{to}' but no ID returned – possibly an API error.")
            return {"status": "error", "error": "No message ID returned"}
    except Exception as e:
        logger.error(
            f"Failed to send message to recipient='{to}' with subject='{subject}'. "
            f"Error: {e}"
        )
        return {"status": "error", "error": str(e)}


def search_messages(query=""):
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
        response = service.users().messages().list(userId='me', q=query).execute()
        messages = response.get('messages', [])
        return messages
    except Exception as e:
        logger.error(f"Error searching messages with query '{query}': {e}")
        return []


def check_thread_for_reply(thread_id):
    """
    Checks if there's more than one message in a given thread, indicating a reply.
    More precise than searching by 'from:' or date alone.
    """
    service = get_gmail_service()
    try:
        thread_data = service.users().threads().get(userId='me', id=thread_id).execute()
        msgs = thread_data.get('messages', [])
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
                userId='me',
                id=m['id'],
                format='full'
            ).execute()
            snippet = full_msg.get('snippet', '')
            snippets.append(snippet)
        except Exception as e:
            logger.error(f"Error fetching message {m['id']} from {email_address}: {e}")

    return snippets
