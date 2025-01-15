from typing import List, Dict, Any, Optional
from utils.gmail_integration import (
    search_inbound_messages_for_email,
    search_messages,
    get_gmail_service,
    create_message,
    create_draft,
    send_message,
    get_or_create_label,
    check_thread_for_reply,
    get_signature
)
from utils.logging_setup import logger
from datetime import datetime
import pytz

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

    def get_email_signature(self) -> str:
        """Get the HTML formatted signature."""
        return get_signature()
    
    def _get_header(self, message: Dict[str, Any], header_name: str) -> str:
        """Extract header value from Gmail message."""
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header["name"].lower() == header_name.lower():
                return header["value"]
        return ""