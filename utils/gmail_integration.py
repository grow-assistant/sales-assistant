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


def create_message(sender, to, subject, message_text):
    """Create a MIMEText email message."""
    logger.debug("=== START CREATE MESSAGE DEBUG ===")
    logger.debug(f"Original message text:\n{message_text}")
    logger.debug(f"Original message length: {len(message_text)}")
    
    # Normalize line endings and clean up any extra spaces
    message_text = message_text.replace('\r\n', '\n')
    message_text = message_text.replace('\n\n\n', '\n\n')
    
    # Create message with explicit content type and encoding
    message = MIMEText(message_text, _subtype='plain', _charset='utf-8')
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    
    # Set content type with format=flowed to preserve line breaks
    message.replace_header('Content-Type', 'text/plain; charset=utf-8; format=flowed; delsp=yes')
    
    # Log the raw message before encoding
    raw_message = message.as_string()
    logger.debug(f"Raw MIME message:\n{raw_message}")
    
    # Convert to bytes, encode, and create raw message
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    logger.debug("=== END CREATE MESSAGE DEBUG ===")
    return {'raw': raw}


def create_draft(sender: str, to: str, subject: str, message_text: str) -> dict:
    """Creates an email draft in Gmail."""
    try:
        # Get Gmail service
        service = get_gmail_service()
        
        # Create the message
        message = create_message(sender, to, subject, message_text)
        
        # Single consolidated log for draft creation
        logger.debug("Creating Gmail draft", extra={
            "to": to,
            "subject": subject
        })

        # Create the draft using service instead of gmail_service
        draft = service.users().drafts().create(
            userId="me",
            body={"message": message}
        ).execute()

        return {
            "status": "ok",
            "draft_id": draft.get("id", "")
        }

    except Exception as e:
        logger.error(f"Error creating draft: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def send_message(sender, to, subject, message_text):
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(sender, to, subject, message_text)
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
