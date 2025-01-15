import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

import openai
from dateutil.parser import parse as parse_date
import pytz

from config.settings import OPENAI_API_KEY
from utils.logging_setup import logger


class ConversationAnalysisService:
    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY

    def clean_email_body(self, body_text: Optional[str]) -> Optional[str]:
        """Clean up email body text by removing signatures, footers, and formatting."""
        if body_text is None:
            return None

        splits = re.split(
            r'(?:\r\n|\r|\n)*(?:From:|On .* wrote:|_{3,}|Get Outlook|Sent from|<http)',
            body_text
        )

        message = splits[0].strip()
        message = re.sub(r'\s+', ' ', message)
        message = re.sub(r'\[cid:[^\]]+\]', '', message)
        message = re.sub(r'<[^>]+>', '', message)

        return message.strip()

    def analyze_conversation(self, 
                           hubspot_emails: List[Dict[str, Any]], 
                           hubspot_notes: List[Dict[str, Any]], 
                           gmail_emails: Dict[str, Any]) -> str:
        """Analyze and summarize the conversation thread using AI."""
        try:
            # Convert notes to email format
            email_notes = []
            for note in hubspot_notes:
                if note.get("timestamp") and "email" in note.get("body", "").lower():
                    timestamp = parse_date(note["timestamp"])
                    if timestamp.tzinfo is None:
                        timestamp = pytz.UTC.localize(timestamp)
                        
                    email_notes.append({
                        "timestamp": timestamp.isoformat(),
                        "body_text": note.get("body"),
                        "direction": "NOTE",
                        "subject": "Email Note"
                    })

            # Add Gmail messages
            gmail_messages = []
            for direction, msg in gmail_emails.items():
                if msg and msg.get("timestamp"):
                    timestamp = parse_date(msg["timestamp"])
                    if timestamp.tzinfo is None:
                        timestamp = pytz.UTC.localize(timestamp)
                        
                    gmail_messages.append({
                        "timestamp": timestamp.isoformat(),
                        "body_text": msg["body_text"],
                        "direction": msg["direction"],
                        "subject": msg.get("subject", "No subject")
                    })

            # Process HubSpot emails
            hubspot_messages = []
            for email in hubspot_emails:
                if email.get("timestamp"):
                    timestamp = parse_date(email["timestamp"])
                    if timestamp.tzinfo is None:
                        timestamp = pytz.UTC.localize(timestamp)
                        
                    hubspot_messages.append({
                        "timestamp": timestamp.isoformat(),
                        "body_text": email.get("body_text"),
                        "direction": email.get("direction"),
                        "subject": email.get("subject", "No subject")
                    })

            # Combine and sort all messages
            all_messages = hubspot_messages + email_notes + gmail_messages
            sorted_messages = sorted(
                all_messages,
                key=lambda x: parse_date(x["timestamp"]),
                reverse=True
            )

            return self._summarize_conversation(sorted_messages)

        except Exception as e:
            logger.error(f"Error in analyze_conversation: {str(e)}", exc_info=True)
            return f"Error analyzing conversation: {str(e)}"

    def _summarize_conversation(self, emails: List[Dict[str, Any]]) -> str:
        """Use OpenAI to summarize the conversation thread."""
        if not emails:
            return "No conversation found."

        # Sort emails by timestamp in ascending order
        sorted_emails = sorted(
            (e for e in emails if e.get('timestamp')),
            key=lambda x: parse_date(x['timestamp'])
        )

        # Find the latest incoming message
        latest_incoming = None
        for email in reversed(sorted_emails):
            if email.get('direction') in ['INCOMING_EMAIL', '←']:
                latest_incoming = email
                break

        # Prepare conversation text for OpenAI
        conversation_text = "Full conversation:\n\n"
        for email in sorted_emails:
            date = parse_date(email['timestamp']).strftime('%Y-%m-%d %H:%M')
            direction = "OUTBOUND" if email.get('direction') in ['EMAIL', '→'] else "INBOUND"
            message = self.clean_email_body(email.get('body_text')) or f"[Email with subject: {email.get('subject', 'No subject')}]"
            conversation_text += f"{date} ({direction}): {message}\n\n"

        if latest_incoming:
            latest_date = parse_date(latest_incoming['timestamp']).strftime('%Y-%m-%d %H:%M')
            conversation_text += f"\nLatest incoming message was at {latest_date}\n"

        try:
            today = datetime.now().date()
            openai.api_key = self.openai_api_key
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
                            f"3) Whether we responded to the latest incoming message, and if so, what was our response (include the full email text) and how many days ago was it sent relative to {today}."
                        )
                    },
                    {"role": "user", "content": conversation_text}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in OpenAI summarization: {str(e)}")
            return f"Error generating summary: {str(e)}" 