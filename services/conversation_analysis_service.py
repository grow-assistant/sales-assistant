import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import pytz
from dateutil.parser import parse as parse_date
import openai

from services.data_gatherer_service import DataGathererService
from services.gmail_service import GmailService
from config.settings import OPENAI_API_KEY
from utils.logging_setup import logger


class ConversationAnalysisService:
    def __init__(self):
        self.data_gatherer = DataGathererService()
        self.gmail_service = GmailService()
        openai.api_key = OPENAI_API_KEY

    def analyze_conversation(self, email_address: str) -> str:
        """Main entry point - analyze conversation for an email address."""
        try:
            print(f"Analyzing conversation for email: {email_address}")
            # Get contact data
            contact_data = self.data_gatherer.hubspot.get_contact_by_email(email_address)
            print(f"Contact data: {contact_data}")
            if not contact_data:
                logger.error(f"No contact found for email: {email_address}")
                return "No contact found."

            contact_id = contact_data["id"]
            print(f"Contact ID: {contact_id}")

            # Gather all messages
            all_messages = self._gather_all_messages(contact_id, email_address)
            print(f"All messages: {all_messages}")
            
            # Generate summary
            summary = self._generate_ai_summary(all_messages)
            print(f"Generated summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Error in analyze_conversation: {str(e)}", exc_info=True)
            print(f"Error analyzing conversation: {str(e)}")
            return f"Error analyzing conversation: {str(e)}"

    def _gather_all_messages(self, contact_id: str, email_address: str) -> List[Dict[str, Any]]:
        """Gather and combine all messages from different sources."""
        print(f"Gathering all messages for contact ID: {contact_id} and email: {email_address}")
        # Get HubSpot emails and notes
        hubspot_emails = self.data_gatherer.hubspot.get_all_emails_for_contact(contact_id)
        hubspot_notes = self.data_gatherer.hubspot.get_all_notes_for_contact(contact_id)
        print(f"HubSpot emails: {hubspot_emails}")
        print(f"HubSpot notes: {hubspot_notes}")
        
        # Get Gmail messages
        gmail_emails = self.gmail_service.get_latest_emails_for_contact(email_address)
        print(f"Gmail emails: {gmail_emails}")

        # Process and combine all messages
        all_messages = []
        
        # Add HubSpot emails
        for email in hubspot_emails:
            if email.get("timestamp"):
                timestamp = self._ensure_timezone(parse_date(email["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": email.get("body_text"),
                    "direction": email.get("direction"),
                    "subject": email.get("subject", "No subject"),
                    "source": "HubSpot"
                })

        # Add relevant notes
        for note in hubspot_notes:
            if note.get("timestamp") and "email" in note.get("body", "").lower():
                timestamp = self._ensure_timezone(parse_date(note["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": note.get("body"),
                    "direction": "NOTE",
                    "subject": "Email Note",
                    "source": "HubSpot"
                })

        # Add Gmail messages
        for direction, msg in gmail_emails.items():
            if msg and msg.get("timestamp"):
                timestamp = self._ensure_timezone(parse_date(msg["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": msg["body_text"],
                    "direction": msg["direction"],
                    "subject": msg.get("subject", "No subject"),
                    "source": "Gmail",
                    "gmail_id": msg.get("gmail_id")
                })

        # Sort by timestamp
        sorted_messages = sorted(all_messages, key=lambda x: parse_date(x["timestamp"]))
        print(f"Sorted messages: {sorted_messages}")
        return sorted_messages

    def _generate_ai_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate AI summary of the conversation."""
        if not messages:
            return "No conversation found."

        conversation_text = self._prepare_conversation_text(messages)
        print(f"Prepared conversation text for AI: {conversation_text}")
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.2,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a sales assistant analyzing email conversations. "
                            "Please provide: "
                            "1) A brief summary of the conversation thread, "
                            "2) The latest INCOMING response only (ignore outbound messages from Ryan Donovan or Ty Hayes), including the date and who it was from, "
                            f"3) Whether we responded to the latest incoming message, and if so, what was our response (include the full email text) and how many days ago was it sent relative to {datetime.now().date()}."
                        )
                    },
                    {"role": "user", "content": conversation_text}
                ]
            )
            summary_content = response.choices[0].message.content
            print(f"AI summary content: {summary_content}")
            return summary_content
        except Exception as e:
            logger.error(f"Error in OpenAI summarization: {str(e)}")
            print(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _clean_email_body(self, body_text: Optional[str]) -> Optional[str]:
        """Clean up email body text."""
        if body_text is None:
            return None

        # Split on common email markers
        splits = re.split(
            r'(?:\r\n|\r|\n)*(?:From:|On .* wrote:|_{3,}|Get Outlook|Sent from|<http)',
            body_text
        )

        message = splits[0].strip()
        message = re.sub(r'\s+', ' ', message)
        message = re.sub(r'\[cid:[^\]]+\]', '', message)
        message = re.sub(r'<[^>]+>', '', message)

        cleaned_message = message.strip()
        print(f"Cleaned email body: {cleaned_message}")
        return cleaned_message

    def _prepare_conversation_text(self, messages: List[Dict[str, Any]]) -> str:
        """Prepare conversation text for AI analysis."""
        conversation_text = "Full conversation:\n\n"
        
        for message in messages:
            date = parse_date(message['timestamp']).strftime('%Y-%m-%d %H:%M')
            direction = "OUTBOUND" if message.get('direction') in ['EMAIL', 'NOTE'] else "INBOUND"
            body = self._clean_email_body(message.get('body_text')) or f"[Email with subject: {message.get('subject', 'No subject')}]"
            conversation_text += f"{date} ({direction}): {body}\n\n"

        print(f"Prepared conversation text: {conversation_text}")
        return conversation_text

    @staticmethod
    def _ensure_timezone(dt):
        """Ensure datetime has timezone information."""
        if dt.tzinfo is None:
            localized_dt = pytz.UTC.localize(dt)
            print(f"Localized datetime: {localized_dt}")
            return localized_dt
        print(f"Datetime already has timezone: {dt}")
        return dt