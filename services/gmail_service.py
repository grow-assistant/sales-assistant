from typing import List, Dict, Any, Optional
from utils.gmail_integration import (
    get_gmail_service,
    create_message,
    create_draft,
    send_message,
    search_messages,
    check_thread_for_reply,
    search_inbound_messages_for_email,
    get_or_create_label
)
from utils.logging_setup import logger
from datetime import datetime
import pytz
import re

class GmailService:
    def get_latest_emails_for_contact(self, email_address: str) -> Dict[str, Any]:
        """Get the latest emails from and to the contact from Gmail."""
        try:
            # Search for latest inbound message
            inbound_query = f"from:{email_address}"
            inbound_messages = search_messages(query=inbound_query)
            
            # Search for latest outbound message
            outbound_query = f"to:{email_address}"
            outbound_messages = search_messages(query=outbound_query)
            
            service = get_gmail_service()
            latest_emails = {
                "inbound": None,
                "outbound": None
            }
            
            # Get latest inbound email
            if inbound_messages:
                latest_inbound = service.users().messages().get(
                    userId="me",
                    id=inbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_inbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                latest_emails["inbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_inbound, "subject"),
                    "body_text": latest_inbound.get("snippet", ""),
                    "direction": "INCOMING_EMAIL",
                    "gmail_id": latest_inbound["id"]
                }
            
            # Get latest outbound email
            if outbound_messages:
                latest_outbound = service.users().messages().get(
                    userId="me",
                    id=outbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_outbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                latest_emails["outbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_outbound, "subject"),
                    "body_text": latest_outbound.get("snippet", ""),
                    "direction": "EMAIL",
                    "gmail_id": latest_outbound["id"]
                }
            
            return latest_emails
            
        except Exception as e:
            logger.error(f"Error getting Gmail messages: {str(e)}")
            return {"inbound": None, "outbound": None}

    def create_draft_email(self, to: str, subject: str, body: str, lead_id: str = None, sequence_num: int = None) -> Dict[str, Any]:
        """Create a draft email with the given parameters."""
        return create_draft(
            sender="me",
            to=to,
            subject=subject,
            message_text=body,
            lead_id=lead_id,
            sequence_num=sequence_num
        )

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send an email immediately."""
        return send_message(
            sender="me",
            to=to,
            subject=subject,
            message_text=body
        )

    def check_for_reply(self, thread_id: str) -> bool:
        """Check if there has been a reply in the thread."""
        return check_thread_for_reply(thread_id)

    def search_replies(self, thread_id):
        """
        Search for replies in a Gmail thread.
        """
        try:
            service = get_gmail_service()
            query = f'threadId:{thread_id} is:reply'
            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])
            return messages if messages else None
        except Exception as e:
            logger.error(f"Error searching for replies in thread {thread_id}: {str(e)}", exc_info=True)
            return None

    def _get_header(self, message: Dict[str, Any], header_name: str) -> str:
        """Extract header value from Gmail message."""
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header["name"].lower() == header_name.lower():
                return header["value"]
        return ""

    def get_latest_emails_with_bounces(self, email_address: str) -> Dict[str, Any]:
        """Get the latest emails including bounce notifications for a contact from Gmail."""
        try:
            # Search for latest inbound message
            inbound_query = f"from:{email_address}"
            inbound_messages = search_messages(query=inbound_query)
            
            # Search for bounce notifications and outbound messages
            outbound_query = f"(to:{email_address} OR (subject:\"Delivery Status Notification\" from:mailer-daemon@googlemail.com {email_address}))"
            outbound_messages = search_messages(query=outbound_query)
            
            service = get_gmail_service()
            latest_emails = {
                "inbound": None,
                "outbound": None
            }
            
            # Get latest inbound email
            if inbound_messages:
                latest_inbound = service.users().messages().get(
                    userId="me",
                    id=inbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_inbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                latest_emails["inbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_inbound, "subject"),
                    "body_text": self._get_full_body(latest_inbound),
                    "direction": "INCOMING_EMAIL",
                    "gmail_id": latest_inbound["id"]
                }
            
            # Get latest outbound email or bounce notification
            if outbound_messages:
                latest_outbound = service.users().messages().get(
                    userId="me",
                    id=outbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_outbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                # Check if it's a bounce notification
                from_header = self._get_header(latest_outbound, "from")
                is_bounce = "mailer-daemon@googlemail.com" in from_header.lower()
                
                latest_emails["outbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_outbound, "subject"),
                    "body_text": self._get_full_body(latest_outbound),
                    "direction": "BOUNCE" if is_bounce else "EMAIL",
                    "gmail_id": latest_outbound["id"],
                    "is_bounce": is_bounce
                }
            
            return latest_emails
            
        except Exception as e:
            logger.error(f"Error getting Gmail messages: {str(e)}")
            return {"inbound": None, "outbound": None}

    def _get_full_body(self, message: Dict[str, Any]) -> str:
        """Extract full message body from Gmail message."""
        try:
            if 'payload' not in message:
                return message.get('snippet', '')

            def get_text_from_part(part):
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        import base64
                        try:
                            return base64.urlsafe_b64decode(data).decode('utf-8')
                        except:
                            return ''
                return ''

            # Check main payload
            text = get_text_from_part(message['payload'])
            if text:
                return text

            # Check parts recursively
            parts = message['payload'].get('parts', [])
            for part in parts:
                text = get_text_from_part(part)
                if text:
                    return text

            return message.get('snippet', '')
        except Exception as e:
            logger.error(f"Error getting message body: {str(e)}")
            return message.get('snippet', '')

    def archive_email(self, message_id: str) -> bool:
        """Archive an email in Gmail by removing the INBOX label."""
        try:
            service = get_gmail_service()
            # Remove INBOX label to archive the message
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['INBOX']}
            ).execute()
            logger.info(f"Successfully archived email {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error archiving email {message_id}: {str(e)}")
            return False

    def search_bounce_notifications(self, email_address: str) -> List[str]:
        """Search for bounce notification emails for a specific email address."""
        try:
            # Search for bounce notifications containing the email address
            query = f'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" "{email_address}"'
            messages = search_messages(query=query)
            
            if messages:
                return [msg['id'] for msg in messages]
            return []
            
        except Exception as e:
            logger.error(f"Error searching bounce notifications: {str(e)}")
            return []

    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Search for messages in Gmail using the given query."""
        try:
            logger.debug(f"Executing Gmail search with query: {query}")
            # Use the imported search_messages function
            messages = search_messages(query=query)
            logger.debug(f"Search results: {messages if messages else 'No messages found'}")
            return messages if messages else []
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}", exc_info=True)
            return []

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific message by ID."""
        try:
            service = get_gmail_service()
            message = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()
            return message
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {str(e)}")
            return None

    def get_bounce_message_details(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details from a bounce notification message including the bounced email address.
        
        Args:
            message_id: The Gmail message ID
            
        Returns:
            Dict with bounced email and other details, or None if not found
        """
        try:
            logger.debug(f"Getting bounce message details for ID: {message_id}")
            message = self.get_message(message_id)
            if not message:
                logger.debug(f"No message found for ID: {message_id}")
                return None
            
            # Get the full message body
            body = self._get_full_body(message)
            if not body:
                logger.debug(f"No body found in message: {message_id}")
                return None
            
            logger.debug(f"Message body length: {len(body)}")
            logger.debug(f"First 200 characters of body: {body[:200]}")
            
            # Extract bounced email using various patterns
            patterns = [
                r'Original-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'Final-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'To: <([\w\.-]+@[\w\.-]+\.\w+)>',
                r'The email account that you tried to reach[^\n]*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,})',
                r'failed permanently.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r"message wasn(?:&#39;|\')t delivered to ([\w\.-]+@[\w\.-]+\.\w+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
                if match:
                    bounced_email = match.group(1)
                    if bounced_email and '@' in bounced_email:
                        logger.debug(f"Found bounced email {bounced_email} using pattern: {pattern}")
                        subject = self._get_header(message, 'subject')
                        return {
                            'bounced_email': bounced_email,
                            'subject': subject,
                            'body': body,
                            'message_id': message_id
                        }
                else:
                    logger.debug(f"Pattern did not match: {pattern}")
            
            logger.debug(f"No email found in message {message_id}. Full body: {body[:500]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error getting bounce message details for {message_id}: {str(e)}", exc_info=True)
            return None

    def process_bounce_notification(self, message_id: str) -> Optional[str]:
        """
        Process a bounce notification and extract the bounced email address.
        
        Args:
            message_id: The Gmail message ID
            
        Returns:
            The bounced email address if found, None otherwise
        """
        try:
            details = self.get_bounce_message_details(message_id)
            if details and 'bounced_email' in details:
                return details['bounced_email']
            return None
        except Exception as e:
            logger.error(f"Error processing bounce notification {message_id}: {str(e)}")
            return None

    def get_all_bounce_notifications(self, inbox_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all bounce notifications from Gmail.
        
        Args:
            inbox_only: If True, only search for bounce notifications in the inbox
        
        Returns:
            List of bounce notification message IDs and details
        """
        try:
            # Search for all bounce notifications
            base_query = "from:mailer-daemon@googlemail.com subject:\"Delivery Status Notification\""
            if inbox_only:
                query = f"{base_query} in:inbox"
            else:
                query = base_query
            
            logger.debug(f"Searching with query: {query}")
            messages = self.search_messages(query)
            logger.debug(f"Found {len(messages) if messages else 0} messages matching query")
            
            bounce_notifications = []
            for message in messages:
                message_id = message['id']
                logger.debug(f"Processing message ID: {message_id}")
                details = self.get_bounce_message_details(message_id)
                logger.debug(f"Details extracted for message {message_id}: {details}")
                if details:
                    bounce_notifications.append(details)
                
            logger.debug(f"Total bounce notifications processed: {len(bounce_notifications)}")
            return bounce_notifications
            
        except Exception as e:
            logger.error(f"Error getting bounce notifications: {str(e)}", exc_info=True)
            return []