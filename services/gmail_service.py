## services/gmail_service.py
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
from services.hubspot_service import HubspotService
import base64

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
        """Extract full message body including all parts."""
        try:
            parts = []
            def extract_parts(payload):
                if 'parts' in payload:
                    for part in payload['parts']:
                        extract_parts(part)
                elif 'body' in payload:
                    if 'data' in payload['body']:
                        parts.append(payload['body']['data'])
                    elif 'attachmentId' in payload['body']:
                        # Handle attachments if needed
                        pass
            
            if 'payload' in message:
                extract_parts(message['payload'])
            
            # Join all parts and decode
            full_body = ''
            for part in parts:
                try:
                    decoded = base64.urlsafe_b64decode(part).decode('utf-8')
                    full_body += decoded + '\n'
                except Exception as e:
                    logger.warning(f"Error decoding message part: {str(e)}")
            
            return full_body
        except Exception as e:
            logger.error(f"Error getting full body: {str(e)}")
            return ""

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
            service = get_gmail_service()
            
            # Execute the search
            result = service.users().messages().list(
                userId='me',
                q=query
            ).execute()
            
            messages = result.get('messages', [])
            logger.debug(f"Found {len(messages)} messages matching query")
            
            return messages
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
        """Get details from a bounce notification message including the bounced email address."""
        try:
            logger.debug(f"Getting bounce message details for ID: {message_id}")
            message = self.get_message(message_id)
            if not message:
                return None
            
            body = self._get_full_body(message)
            if not body:
                return None
            
            logger.debug(f"Message body length: {len(body)}")
            logger.debug(f"First 200 characters of body: {body[:200]}")
            
            # Add Office 365 patterns to existing patterns
            patterns = [
                # Existing Gmail patterns
                r'Original-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'Final-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'To: <([\w\.-]+@[\w\.-]+\.\w+)>',
                r'The email account that you tried to reach[^\n]*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,})',
                r'failed permanently.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r"message wasn(?:&#39;|\')t delivered to ([\w\.-]+@[\w\.-]+\.\w+)",
                
                # New Office 365 patterns
                r"Your message to ([\w\.-]+@[\w\.-]+\.\w+) couldn't be delivered",
                r"Recipient Address:\s*([\w\.-]+@[\w\.-]+\.\w+)",
                r"550 5\.1\.10.*?Recipient ([\w\.-]+@[\w\.-]+\.\w+) not found",
                r"RESOLVER\.ADR\.RecipientNotFound; Recipient ([\w\.-]+@[\w\.-]+\.\w+) not found",
            ]
            
            # Update bounce query to include Office 365 postmaster
            self.BOUNCE_QUERY = """
                (from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" 
                 OR from:postmaster@*.outbound.protection.outlook.com subject:"Undeliverable")
                in:inbox
            """
            
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
        """
        try:
            details = self.get_bounce_message_details(message_id)
            if details and 'bounced_email' in details:
                # Add HubSpot logging
                logger.info(f"Attempting to update HubSpot for bounced email: {details['bounced_email']}")
                
                # Ensure HubSpot is being called
                hubspot_service = HubspotService()
                success = hubspot_service.mark_contact_as_bounced(details['bounced_email'])
                
                if success:
                    logger.info(f"Successfully updated HubSpot for {details['bounced_email']}")
                else:
                    logger.error(f"Failed to update HubSpot for {details['bounced_email']}")
                
                return details['bounced_email']
            return None
        except Exception as e:
            logger.error(f"Error processing bounce notification {message_id}: {str(e)}")
            return None

    def get_all_bounce_notifications(self, inbox_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all bounce notifications from Gmail.
        """
        try:
            # Create a clean, precise query that matches both direct and "via" senders
            base_query = (
                '(from:"postmaster@*" OR from:"mailer-daemon@*") '
                'subject:("Undeliverable" OR "Delivery Status Notification")'
            )
            
            if inbox_only:
                query = f"{base_query} in:inbox"
            else:
                query = base_query
            
            logger.debug(f"Using Gmail search query: {query}")
            messages = self.search_messages(query)
            
            # Log the raw results for debugging
            logger.debug(f"Raw search results: {messages}")
            
            bounce_notifications = []
            for message in messages:
                message_id = message['id']
                full_message = self.get_message(message_id)
                if full_message:
                    headers = full_message.get('payload', {}).get('headers', [])
                    from_header = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'Unknown')
                    logger.info(f"Processing message - From: {from_header}, Subject: {subject}")
                
                details = self.get_bounce_message_details(message_id)
                if details:
                    bounce_notifications.append(details)
                    logger.info(f"Found bounce notification for: {details.get('bounced_email')}")
                else:
                    logger.warning(f"Could not extract bounce details from message ID: {message_id}")
            
            return bounce_notifications
            
        except Exception as e:
            logger.error(f"Error getting bounce notifications: {str(e)}", exc_info=True)
            return []

    def get_rejection_search_query(self):
        """Returns search query for explicit rejection emails"""
        rejection_phrases = [
            '"no thanks"',
            '"don\'t contact"',
            '"do not contact"',
            '"please remove"',
            '"not interested"',
            '"we use"',
            '"we already use"',
            '"we have"',
            '"we already have"',
            '"please don\'t contact"',
            '"stop contacting"',
            '"remove me"',
            '"unsubscribe"'
        ]
        
        # Combine phrases with OR and add inbox filter
        query = f"({' OR '.join(rejection_phrases)}) in:inbox"
        return query

    def get_gmail_template(self, template_name: str = "salesv2") -> str:
        """
        Fetch HTML email template from Gmail drafts whose subject
        contains `template_name`.

        Args:
            template_name (str): Name to search for in draft subjects. Defaults to "sales".

        Returns:
            str: The template as an HTML string, or "" if no template found.
        """
        try:
            service = get_gmail_service()
            drafts_list = service.users().drafts().list(userId='me').execute()
            if not drafts_list.get('drafts'):
                logger.error("No drafts found in Gmail account.")
                return ""

            # Optional: gather subjects for debugging
            draft_subjects = []
            template_html = ""

            for draft in drafts_list['drafts']:
                msg_id = draft['message']['id']
                msg = service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()

                # Grab the subject
                headers = msg['payload'].get('headers', [])
                subject = next(
                    (h['value'] for h in headers if h['name'].lower() == 'subject'),
                    ''
                ).lower()
                draft_subjects.append(subject)

                # If the draft subject contains template_name, treat that as the template
                if template_name.lower() in subject:
                    logger.debug(f"Found template draft with subject: {subject}")

                    # Extract HTML parts
                    if 'parts' in msg['payload']:
                        for part in msg['payload']['parts']:
                            if part['mimeType'] == 'text/html':
                                template_html = base64.urlsafe_b64decode(
                                    part['body']['data']
                                ).decode('utf-8', errors='ignore')
                                break
                    elif msg['payload'].get('mimeType') == 'text/html':
                        template_html = base64.urlsafe_b64decode(
                            msg['payload']['body']['data']
                        ).decode('utf-8', errors='ignore')

                    if template_html:
                        return template_html

            # If we got here, we never found a match
            logger.error(
                f"No template found with name: {template_name}. "
                f"Available draft subjects: {draft_subjects}"
            )
            return ""

        except Exception as e:
            logger.error(f"Error fetching Gmail template: {str(e)}", exc_info=True)
            return ""