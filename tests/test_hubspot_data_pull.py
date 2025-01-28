import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import openai
from dateutil.parser import parse as parse_date
import pytz

from services.data_gatherer_service import DataGathererService
from config.settings import HUBSPOT_API_KEY, OPENAI_API_KEY
from utils.logging_setup import logger
from services.gmail_service import GmailService
from utils.conversation_summary import get_latest_email_date, summarize_lead_interactions


def clean_email_body(body_text: Optional[str]) -> Optional[str]:
    """
    Clean up email body text by removing signatures, footers, and formatting.

    :param body_text: The raw email body text.
    :return: A cleaned version of the email body or None if the input is None.
    """
    if body_text is None:
        return None

    # Split on common email reply markers (signatures, footers, etc.)
    splits = re.split(
        r'(?:\r\n|\r|\n)*(?:From:|On .* wrote:|_{3,}|Get Outlook|Sent from|<http)',
        body_text
    )

    # Take the first part (assumed to be the main message)
    message = splits[0].strip()

    # Remove extra whitespace
    message = re.sub(r'\s+', ' ', message)
    # Remove [cid:...] references
    message = re.sub(r'\[cid:[^\]]+\]', '', message)
    # Remove HTML tags
    message = re.sub(r'<[^>]+>', '', message)

    return message.strip()


def summarize_conversation(emails: List[Dict[str, Any]]) -> str:
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
        if email.get('direction') == 'INCOMING_EMAIL':
            latest_incoming = email
            break

    # Prepare conversation text for OpenAI
    conversation_text = "Full conversation:\n\n"
    for email in sorted_emails:
        date = parse_date(email['timestamp']).strftime('%Y-%m-%d %H:%M')
        direction = "OUTBOUND" if email.get('direction') in ['EMAIL', 'NOTE'] else "INBOUND"
        message = clean_email_body(email.get('body_text')) or f"[Email with subject: {email.get('subject', 'No subject')}]"
        conversation_text += f"{date} ({direction}): {message}\n\n"

    # Add specific information about latest incoming message
    if latest_incoming:
        latest_date = parse_date(latest_incoming['timestamp']).strftime('%Y-%m-%d %H:%M')
        conversation_text += f"\nLatest incoming message was at {latest_date}\n"

    try:
        today = datetime.now().date()
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.2,  # Add moderate temperature for consistent but natural responses
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


def print_conversation_thread(emails: List[Dict[str, Any]], notes: List[Dict[str, Any]], gmail_emails: Dict[str, Any]) -> None:
    """
    Print emails as a conversation thread and provide an AI summary.
    """
    if not emails and not notes and not gmail_emails["inbound"] and not gmail_emails["outbound"]:
        print("\nNo conversation found!")
        return

    # print("\nConversation Thread:")
    # print("=" * 50)

    # Convert notes to email format
    email_notes = []
    for note in notes:
        if note.get("timestamp") and "email" in note.get("body", "").lower():
            # Ensure timestamp is timezone-aware
            timestamp = parse_date(note["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
                
            email_notes.append({
                "timestamp": timestamp.isoformat(),
                "body_text": note.get("body"),
                "direction": "NOTE",
                "subject": "Email Note"
            })

    # Add Gmail messages if they exist
    gmail_messages = []
    for direction, msg in gmail_emails.items():
        if msg and msg.get("timestamp"):
            # Ensure timestamp is timezone-aware
            timestamp = parse_date(msg["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
                
            gmail_messages.append({
                "timestamp": timestamp.isoformat(),
                "body_text": msg["body_text"],
                "direction": msg["direction"],
                "subject": msg.get("subject", "No subject"),
                "source": "Gmail",
                "gmail_id": msg.get("gmail_id")
            })

    # Process HubSpot emails
    hubspot_messages = []
    for email in emails:
        if email.get("timestamp"):
            # Ensure timestamp is timezone-aware
            timestamp = parse_date(email["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
                
            hubspot_messages.append({
                "timestamp": timestamp.isoformat(),
                "body_text": email.get("body_text"),
                "direction": email.get("direction"),
                "subject": email.get("subject", "No subject"),
                "source": "HubSpot"
            })

    # Combine all messages
    all_messages = hubspot_messages + email_notes + gmail_messages

    # Sort all messages by timestamp in descending order
    sorted_messages = sorted(
        all_messages,
        key=lambda x: parse_date(x["timestamp"]),
        reverse=True
    )

    # for message in sorted_messages:
    #     try:
    #         date_obj = parse_date(message["timestamp"])
            
    #         # Determine message source and direction
    #         source = message.get("source", "HubSpot")
    #         if message.get("direction") == "NOTE":
    #             direction_arrow = "📝"
    #         else:
    #             direction_arrow = "→" if message.get("direction") == "EMAIL" else "←"
            
    #         print(f"\n{date_obj.strftime('%Y-%m-%d %H:%M')} {direction_arrow} [{source}] ", end="")

    #         body_text = clean_email_body(message.get("body_text"))
    #         if body_text:
    #             print(body_text)
    #         else:
    #             subject = message.get("subject", "No subject")
    #             print(f"[Email with subject: {subject}]")

    #     except Exception as e:
    #         logger.error(f"Error processing message: {str(e)}")
    #         continue

    # Print AI-generated summary
    print("\nAI Analysis:")
    print("=" * 50)
    summary = summarize_conversation(sorted_messages)
    print(summary)


def get_contacts_from_list(list_id: str) -> List[Dict[str, Any]]:
    """Get all contacts from a specified HubSpot list."""
    try:
        data_gatherer = DataGathererService()
        logger.debug(f"Initializing contact pull from list {list_id}")
        
        # First get list memberships
        url = f"https://api.hubapi.com/crm/v3/lists/{list_id}/memberships"
        logger.debug(f"Fetching list memberships from: {url}")
        
        memberships = data_gatherer.hubspot._make_hubspot_get(url)
        if not memberships:
            logger.warning(f"No memberships found for list {list_id}")
            return []
            
        # Extract record IDs
        record_ids = []
        if isinstance(memberships, dict) and "results" in memberships:
            record_ids = [result["recordId"] for result in memberships.get("results", [])]
            logger.info(f"Found {len(record_ids)} records in list {list_id}")
        else:
            logger.warning(f"Unexpected membership response format for list {list_id}")
            return []
        
        # Now fetch full contact details for each ID (limit to first 25)
        contacts = []
        for record_id in record_ids[:25]:  # Limit to first 25
            try:
                # Get full contact details using v3 API
                contact_url = f"https://api.hubapi.com/crm/v3/objects/contacts/{record_id}"
                params = {
                    "properties": ["email", "firstname", "lastname", "company"]
                }
                contact_data = data_gatherer.hubspot._make_hubspot_get(contact_url, params)
                
                if contact_data:
                    logger.debug(f"Successfully fetched contact {record_id}")
                    contacts.append(contact_data)
                else:
                    logger.warning(f"No data returned for contact {record_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch details for contact {record_id}: {str(e)}")
                continue
        
        if not contacts:
            logger.warning(f"No contact details retrieved for list {list_id}")
        else:
            logger.info(f"Successfully retrieved {len(contacts)} contacts with details from list {list_id}")
        
        return contacts
        
    except Exception as e:
        logger.error(f"Error getting contacts from list {list_id}: {str(e)}", exc_info=True)
        return []


def process_list_contacts(list_id: str) -> List[Dict[str, Any]]:
    """Process all contacts from a specified HubSpot list."""
    try:
        logger.debug(f"Starting to process list ID: {list_id}")
        
        # Get contacts from list
        contacts = get_contacts_from_list(list_id)
        
        if not contacts:
            print(f"No contacts found in list {list_id}")
            return []  # Return empty list instead of None

        print("\nFirst 25 email addresses from list:")
        print("=" * 50)
        
        # Process each contact
        for i, contact in enumerate(contacts, 1):
            # Get email from properties
            properties = contact.get("properties", {})
            email = properties.get("email")
            
            if email:
                print(f"{i}. {email}")
            else:
                print(f"{i}. [No email found]")

        return contacts  # Return the contacts list

    except Exception as e:
        print(f"Error processing list contacts: {str(e)}")
        logger.error(f"Error in process_list_contacts: {str(e)}", exc_info=True)
        return []  # Return empty list on error


def test_email_pull(email_address: str) -> None:
    """Test function to pull and display HubSpot data for a specific email address."""
    try:
        data_gatherer = DataGathererService()
        gmail_service = GmailService()

        # Get contact data
        contact_data = data_gatherer.hubspot.get_contact_by_email(email_address)
        if not contact_data:
            print(f"No contact found for email: {email_address}")
            return

        contact_id = contact_data["id"]

        # Get all emails and notes from HubSpot
        try:
            all_emails = data_gatherer.hubspot.get_all_emails_for_contact(contact_id)
            all_notes = data_gatherer.hubspot.get_all_notes_for_contact(contact_id)
            
            # Get latest Gmail messages
            gmail_emails = gmail_service.get_latest_emails_for_contact(email_address)
            
            # Get latest email date
            latest_date = get_latest_email_date(all_emails)
            if latest_date:
                print(f"\nLatest Email Date: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("\nNo emails found")
            
            print_conversation_thread(all_emails, all_notes, gmail_emails)
            
        except Exception as e:
            print(f"Error fetching messages: {str(e)}")
            all_emails = []
            all_notes = []
            gmail_emails = {"inbound": None, "outbound": None}

        # Save raw data to file
        output_data = {
            "contact_id": contact_id,
            "emails": all_emails,
            "notes": all_notes,
            "gmail_emails": gmail_emails
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hubspot_data_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull response saved to: {filename}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_email_pull: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Enable debug logging
    logger.setLevel("DEBUG")
    
    # Replace this with your actual HubSpot list ID
    TEST_LIST_ID = "221"  # <-- Put your actual HubSpot list ID here
    
    print(f"\nStarting contact pull from HubSpot list: {TEST_LIST_ID}")
    
    try:
        # Get and process contacts from list
        contacts = process_list_contacts(TEST_LIST_ID)
        
        if contacts:  # Check if we have any contacts
            for contact in contacts:
                properties = contact.get("properties", {})
                email = properties.get("email")
                if email:
                    print(f"\nProcessing email: {email}")
                    test_email_pull(email)
        else:
            print("No contacts found to process")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logger.error(f"Main execution error: {str(e)}", exc_info=True)
