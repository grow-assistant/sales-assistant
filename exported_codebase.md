
## scheduling\database.py

# scheduling/database.py
"""
Database operations for the scheduling service.
"""
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
                    gmail_id           VARCHAR(100),
                    company_short_name VARCHAR(100) NULL
                )
            END
        """)
        
        
        conn.commit()
        
        
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
                     status: str = 'pending',
                     company_short_name: str = None) -> int:
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
    - company_short_name
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
                status = ?,
                company_short_name = ?
            WHERE draft_id = ? AND lead_id = ?
        """, (
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            company_short_name,
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
                draft_id,
                company_short_name
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            lead_id,
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id,
            company_short_name
        ))
        cursor.execute("SELECT SCOPE_IDENTITY()")
        return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")





## scheduling\followup_generation.py

# scheduling/followup_generation.py
"""
Functions for generating follow-up emails.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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
        logger.debug(f"Starting follow-up generation for lead_id={lead_id}, sequence_num={sequence_num}")
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    email_address,
                    name,
                    company_short_name,
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

            email_address, name, company_short_name, body, gmail_id, scheduled_date, draft_id = row
            original_email = {
                'email': email_address,
                'name': name,
                'company_short_name': company_short_name,
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
        venue_name = original_email.get('company_short_name', 'your club')
        subject = f"Re: {orig_subject}"
        
        body = (
            f"Following up about improving operations at {venue_name}. "
            f"Would you have 10 minutes this week for a brief call?\n\n"
            f"Thanks,\n"
            f"Ty\n\n\n\n"
            f"On {date_header}, {from_header} wrote:\n"
            f"{orig_body}"
        )

        # Calculate send date using golf_outreach_strategy logic
        send_date = calculate_send_date(
            geography="Year-Round Golf",
            persona="General Manager",
            state="AZ",
            sequence_num=sequence_num or 2,  # Pass sequence_num to determine time window
            season_data=None
        )
        logger.debug(f"Calculated send date: {send_date}")

        # Ensure minimum 3-day gap from original send date
        orig_scheduled_date = original_email.get('scheduled_send_date', datetime.now())
        logger.debug(f"Original scheduled date: {orig_scheduled_date}")
        while send_date < (orig_scheduled_date + timedelta(days=3)):
            send_date += timedelta(days=1)
            logger.debug(f"Adjusted send date to ensure 3-day gap: {send_date}")

        return {
            'email': original_email.get('email'),
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': str(lead_id),
            'company_short_name': original_email.get('company_short_name', ''),
            'in_reply_to': original_email['gmail_id'],
            'original_html': original_html,
            'thread_id': original_email['gmail_id']
        }
        

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()




## scripts\golf_outreach_strategy.py

# scripts/golf_outreach_strategy.py
# """
# Scripts for determining optimal outreach timing based on club and contact attributes.
# """
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os
import random

logger = logging.getLogger(__name__)

def load_state_offsets():
    """Load state hour offsets from CSV file."""
    offsets = {}
    csv_path = os.path.join('docs', 'data', 'state_timezones.csv')
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = row['state_code']
            offsets[state] = {
                'dst': int(row['daylight_savings']),
                'std': int(row['standard_time'])
            }
    logger.debug(f"Loaded timezone offsets for {len(offsets)} states")
    return offsets

STATE_OFFSETS = load_state_offsets()

def adjust_send_time(send_time: datetime, state_code: str) -> datetime:
    """Adjust send time based on state's hour offset."""
    if not state_code:
        logger.warning("No state code provided, using original time")
        return send_time
        
    offsets = STATE_OFFSETS.get(state_code.upper())
    if not offsets:
        logger.warning(f"No offset data for state {state_code}, using original time")
        return send_time
    
    # Determine if we're in DST
    # In US, DST is from second Sunday in March to first Sunday in November
    dt = datetime.now()
    is_dst = 3 <= dt.month <= 11  # True if between March and November
    
    # Get the offset relative to Arizona time
    offset_hours = offsets['dst'] if is_dst else offsets['std']
    
    # Apply offset from Arizona time
    adjusted_time = send_time + timedelta(hours=offset_hours)
    logger.debug(f"Adjusted time from {send_time} to {adjusted_time} for state {state_code} (offset: {offset_hours}h, DST: {is_dst})")
    return adjusted_time

def get_best_month(geography: str, club_type: str = None, season_data: dict = None) -> list:
    """
    Determine best outreach months based on geography/season and club type.
    """
    current_month = datetime.now().month
    logger.debug(f"Determining best month for geography: {geography}, club_type: {club_type}, current month: {current_month}")
    
    # If we have season data, use it as primary decision factor
    if season_data:
        peak_start = season_data.get('peak_season_start', '')
        peak_end = season_data.get('peak_season_end', '')
        logger.debug(f"Using season data - peak start: {peak_start}, peak end: {peak_end}")
        
        if peak_start and peak_end:
            peak_start_month = int(peak_start.split('-')[0])
            peak_end_month = int(peak_end.split('-')[0])
            
            logger.debug(f"Peak season: {peak_start_month} to {peak_end_month}")
            
            # For winter peak season (crossing year boundary)
            if peak_start_month > peak_end_month:
                if current_month >= peak_start_month or current_month <= peak_end_month:
                    logger.debug("In winter peak season, targeting September shoulder season")
                    return [9]  # September (before peak starts)
                else:
                    logger.debug("In winter shoulder season, targeting January")
                    return [1]  # January
            # For summer peak season
            else:
                if peak_start_month <= current_month <= peak_end_month:
                    target = [peak_start_month - 1] if peak_start_month > 1 else [12]
                    logger.debug(f"In summer peak season, targeting month {target}")
                    return target
                else:
                    logger.debug("In summer shoulder season, targeting January")
                    return [1]  # January
    
    # Fallback to geography-based matrix
    month_matrix = {
        "Year-Round Golf": [1, 9],      # January or September
        "Peak Winter Season": [9],       # September
        "Peak Summer Season": [2],       # February
        "Short Summer Season": [1],      # January
        "Shoulder Season Focus": [2, 10]  # February or October
    }
    
    result = month_matrix.get(geography, [1, 9])
    logger.debug(f"Using geography matrix fallback for {geography}, selected months: {result}")
    return result

def get_best_time(persona: str, sequence_num: int) -> dict:
    """
    Determine best time of day based on persona and email sequence number.
    Returns a dict with start and end hours/minutes in 24-hour format.
    Times are aligned to 30-minute windows.
    """
    logger.debug(f"Getting best time for persona: {persona}, sequence_num: {sequence_num}")
    
    time_windows = {
        "General Manager": {
            1: [  # Sequence 1: Morning hours
                {
                    "start_hour": 8, "start_minute": 30,
                    "end_hour": 10, "end_minute": 30
                }
            ],
            2: [  # Sequence 2: Afternoon hours
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ]
        },
        "Food & Beverage Director": {
            1: [  # Sequence 1: Morning hours
                {
                    "start_hour": 9, "start_minute": 30,
                    "end_hour": 11, "end_minute": 30
                }
            ],
            2: [  # Sequence 2: Afternoon hours
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ]
        }
        # "Golf Professional": [
        #     {
        #         "start_hour": 8, "start_minute": 0,
        #         "end_hour": 10, "end_minute": 0
        #     }   # 8:00-10:00 AM
        # ]
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    logger.debug(f"Normalized persona: {persona}")
    
    # Get time windows for the persona and sequence number, defaulting to GM times if not found
    windows = time_windows.get(persona, time_windows["General Manager"]).get(sequence_num, time_windows["General Manager"][1])
    if persona not in time_windows or sequence_num not in time_windows[persona]:
        logger.debug(f"No specific time window for {persona} with sequence {sequence_num}, using General Manager defaults")
    
    # Select the time window
    selected_window = windows[0]  # Since we have only one window per sequence
    logger.debug(f"Selected time window: {selected_window['start_hour']}:{selected_window['start_minute']} - {selected_window['end_hour']}:{selected_window['end_minute']}")
    
    # Update calculate_send_date function expects start/end format
    return {
        "start": selected_window["start_hour"] + selected_window["start_minute"] / 60,
        "end": selected_window["end_hour"] + selected_window["end_minute"] / 60
    }

def get_best_outreach_window(persona: str, geography: str, club_type: str = None, season_data: dict = None) -> Dict[str, Any]:
    """Get the optimal outreach window based on persona and geography."""
    logger.debug(f"Getting outreach window for persona: {persona}, geography: {geography}, club_type: {club_type}")
    
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona, 1)
    best_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday (0 = Monday, 6 = Sunday)
    
    logger.debug(f"Calculated base outreach window", extra={
        "persona": persona,
        "geography": geography,
        "best_months": best_months,
        "best_time": best_time,
        "best_days": best_days
    })
    
    return {
        "Best Month": best_months,
        "Best Time": best_time,
        "Best Day": best_days
    }

def calculate_send_date(geography: str, persona: str, state: str, sequence_num: int, season_data: dict = None) -> datetime:
    """Calculate the next appropriate send date based on outreach window."""
    logger.debug(f"Calculating send date for: geography={geography}, persona={persona}, state={state}, sequence_num={sequence_num}")
    
    outreach_window = get_best_outreach_window(geography, persona, season_data=season_data)
    best_months = outreach_window["Best Month"]
    preferred_time = get_best_time(persona, sequence_num)
    preferred_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    # Get current time and adjust it for target state's timezone first
    now = datetime.now()
    state_now = adjust_send_time(now, state)
    today_weekday = state_now.weekday()
    
    # Check if we can use today (must be preferred day AND before end time in STATE's timezone)
    end_hour = int(preferred_time["end"])
    if (today_weekday in preferred_days and 
        state_now.hour < end_hour):  # Compare state's local time to end hour
        target_date = now
        logger.debug(f"Using today ({target_date}) as it's a preferred day (weekday: {today_weekday}) and before end time ({end_hour})")
    else:
        days_ahead = [(day - today_weekday) % 7 for day in preferred_days]
        next_preferred_day = min(days_ahead)
        target_date = now + timedelta(days=next_preferred_day)
        logger.debug(f"Using future date ({target_date}) as today isn't valid (weekday: {today_weekday} or after {end_hour})")
    
    # Apply preferred time
    start_hour = int(preferred_time["start"])
    start_minutes = int((preferred_time["start"] % 1) * 60)
    target_date = target_date.replace(hour=start_hour, minute=start_minutes)
    logger.debug(f"Applied preferred time: {target_date}")
    
    # Final timezone adjustment
    final_date = adjust_send_time(target_date, state)
    logger.debug(f"Final scheduled date after timezone adjustment: {final_date}")
    
    # Log the final scheduled send date and time
    logger.info(f"Scheduled send date and time: {final_date}")
    
    return final_date




## services\gmail_service.py

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
            import base64
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



## services\leads_service.py

## services/leads_service.py
"""
services/leads_service.py

Handles lead-related business logic, including generating lead summaries
(not the full data gathering, which now lives in DataGathererService).
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT, DEBUG_MODE
from services.data_gatherer_service import DataGathererService
from utils.logging_setup import logger
from utils.doc_reader import read_doc
from utils.exceptions import HubSpotError


class LeadContextError(Exception):
    """Custom exception for lead context preparation errors."""
    pass


class LeadsService:
    """
    Responsible for higher-level lead operations such as generating
    specialized summaries or reading certain docs for personalization.
    NOTE: The actual data gathering is now centralized in DataGathererService.
    """
    
    def __init__(self, data_gatherer_service: DataGathererService):
        """
        Initialize LeadsService.
        
        Args:
            data_gatherer_service: Service for gathering lead data
        """
        self.data_gatherer = data_gatherer_service

    def prepare_lead_context(self, lead_email: str, lead_sheet: Dict = None, correlation_id: str = None) -> Dict[str, Any]:
        """
        Prepare lead context for personalization (subject/body).
        
        Args:
            lead_email: Email address of the lead
            lead_sheet: Optional pre-gathered lead data
            correlation_id: Optional correlation ID for tracing operations
        """
        if correlation_id is None:
            correlation_id = f"prepare_context_{lead_email}"
            
        logger.debug("Starting lead context preparation", extra={
            "email": lead_email,
            "correlation_id": correlation_id
        })
        
        # Use provided lead_sheet or gather new data if none provided
        if not lead_sheet:
            lead_sheet = self.data_gatherer.gather_lead_data(lead_email, correlation_id=correlation_id)
        
        if not lead_sheet:
            logger.warning("No lead data found", extra={"email": lead_email})
            return {}
        
        # Extract relevant data
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        
        # Get job title for template selection
        props = lead_data.get("properties", {})
        job_title = (props.get("jobtitle", "") or "").strip()
        filename_job_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )

        # Get template content
        template_path = f"templates/{filename_job_title}_initial_outreach.md"
        try:
            template_content = read_doc(template_path)
            if isinstance(template_content, str):
                template_content = {
                    "subject": "Default Subject",
                    "body": template_content
                }
            subject = template_content.get("subject", "Default Subject")
            body = template_content.get("body", template_content.get("content", ""))
        except Exception as e:
            logger.warning(
                "Template read failed, using fallback content",
                extra={
                    "template": template_path,
                    "error": str(e),
                    "correlation_id": correlation_id
                }
            )
            subject = "Fallback Subject"
            body = "Fallback Body..."

        # Rest of your existing code...
        return {
            "metadata": {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "email": lead_email,
                "job_title": job_title
            },
            "lead_data": lead_data,
            "subject": subject,
            "body": body
        }

    def get_lead_summary(self, lead_id: str) -> Dict[str, Optional[str]]:
        """
        Get key summary information about a lead.
        
        Args:
            lead_id: HubSpot contact ID
            
        Returns:
            Dict containing:
                - last_reply_date: Date of latest email reply
                - lifecycle_stage: Current lifecycle stage
                - company_short_name: Short name of associated company
                - company_name: Full name of associated company
                - error: Error message if any
        """
        try:
            logger.info(f"Fetching HubSpot properties for lead {lead_id}...")
            contact_props = self.data_gatherer.hubspot.get_contact_properties(lead_id)
            result = {
                'last_reply_date': None,
                'lifecycle_stage': None,
                'company_short_name': None,
                'company_name': None,
                'error': None
            }
            
            # Get basic contact info
            result['last_reply_date'] = contact_props.get('hs_sales_email_last_replied')
            result['lifecycle_stage'] = contact_props.get('lifecyclestage')
            
            # Get company information
            company_id = self.data_gatherer.hubspot.get_associated_company_id(lead_id)
            if company_id:
                company_props = self.data_gatherer.hubspot.get_company_data(company_id)
                if company_props:
                    result['company_name'] = company_props.get('name')
                    result['company_short_name'] = company_props.get('company_short_name')
            
            return result
            
        except HubSpotError as e:
            if '404' in str(e):
                return {'error': '404 - Lead not found'}
            return {'error': str(e)}




## tests\test_followup_generation.py

## test_followup_generation.py
"""
Test script for generating follow-up emails.
""" 

import sys
from pathlib import Path
import random

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.followup_generation import generate_followup_email_xai
from scheduling.database import get_db_connection
from utils.logging_setup import logger
from utils.gmail_integration import create_draft, create_followup_draft
from scheduling.extended_lead_storage import store_lead_email_info
from services.gmail_service import GmailService

def test_followup_generation_for_60():
    """Generate follow-up emails for 5 random sequence=1 emails in the database."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch all sequence 1 emails
        cursor.execute("""
            SELECT
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date,
                sequence_num,
                company_short_name,
                body
            FROM emails
            WHERE sequence_num = 1
              AND gmail_id IS NOT NULL
              AND company_short_name IS NOT NULL
            ORDER BY created_at ASC
        """)
        
        rows = cursor.fetchall()
        if not rows:
            logger.error("No sequence=1 emails found in the database.")
            return
        
        # Select 5 random emails from the list
        random_rows = random.sample(rows, min(5, len(rows)))
        
        logger.info(f"Selected {len(random_rows)} random emails. Generating follow-ups...")
        
        gmail_service = GmailService()
        
        for idx, row in enumerate(random_rows, start=1):
            (lead_id, email, name, gmail_id, scheduled_date,
             seq_num, company_short_name, body) = row
            
            logger.info(f"[{idx}] Checking for reply for Lead ID: {lead_id}, Email: {email}")
            
            # Check if there is a reply in the thread
            try:
                logger.debug(f"Searching for replies in thread with ID: {gmail_id}")
                replies = gmail_service.search_replies(gmail_id)
                if replies:
                    logger.info(f"[{idx}] Lead ID: {lead_id} has replied. Skipping follow-up.")
                    continue
            except Exception as e:
                logger.error(f"Error searching for replies in thread {gmail_id}: {str(e)}", exc_info=True)
                continue
            
            logger.info(f"[{idx}] Generating follow-up for Lead ID: {lead_id}, Email: {email}")
            
            followup = generate_followup_email_xai(
                lead_id=lead_id,
                original_email={
                    'email': email,
                    'name': name,
                    'gmail_id': gmail_id,
                    'scheduled_send_date': scheduled_date,
                    'company_short_name': company_short_name,
                    'body': body  # Pass original body to provide context
                }
            )
            
            if followup:
                draft_result = create_followup_draft(
                    sender="me",
                    to=email,
                    subject=followup['subject'],
                    message_text=followup['body'],
                    lead_id=str(lead_id),
                    sequence_num=followup.get('sequence_num', 2),
                    original_html=followup.get('original_html'),
                    in_reply_to=followup['in_reply_to']
                )
                
                if draft_result.get('draft_id'):
                    store_lead_email_info(
                        lead_sheet={
                            'lead_data': {
                                'properties': {'hs_object_id': lead_id},
                                'email': email
                            },
                            'company_data': {
                                'company_short_name': company_short_name
                            }
                        },
                        draft_id=draft_result['draft_id'],
                        scheduled_date=followup['scheduled_send_date'],
                        body=followup['body'],
                        sequence_num=followup.get('sequence_num', 2)
                    )
                    logger.info(f"[{idx}] Successfully stored follow-up for Lead ID: {lead_id}")
                else:
                    logger.error(f"[{idx}] Failed to create Gmail draft for Lead ID: {lead_id} "
                                 f"({draft_result.get('error', 'Unknown error')})")
            else:
                logger.error(f"[{idx}] Failed to generate follow-up for Lead ID: {lead_id}")
        
    except Exception as e:
        logger.error(f"Error while generating follow-ups: {str(e)}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def test_followup_generation_for_specific_lead(lead_id):
    """Generate follow-up email for a specific lead."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch the specific lead email
        cursor.execute("""
            SELECT
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date,
                sequence_num,
                company_short_name,
                body
            FROM emails
            WHERE lead_id = ?
              AND sequence_num = 1
              AND gmail_id IS NOT NULL
              AND company_short_name IS NOT NULL
        """, (lead_id,))
        
        row = cursor.fetchone()
        if not row:
            logger.error(f"No email found for lead_id={lead_id}.")
            return
        
        logger.info(f"Generating follow-up for Lead ID: {lead_id}...")
        
        gmail_service = GmailService()
        
        (lead_id, email, name, gmail_id, scheduled_date,
         seq_num, company_short_name, body) = row
        
        logger.info(f"Checking for reply for Lead ID: {lead_id}, Email: {email}")
        
        # Check if there is a reply in the thread
        try:
            logger.debug(f"Searching for replies in thread with ID: {gmail_id}")
            replies = gmail_service.search_replies(gmail_id)
            if replies:
                logger.info(f"Lead ID: {lead_id} has replied. Skipping follow-up.")
                return
        except Exception as e:
            logger.error(f"Error searching for replies in thread {gmail_id}: {str(e)}", exc_info=True)
            return
        
        logger.info(f"Generating follow-up for Lead ID: {lead_id}, Email: {email}")
        
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            original_email={
                'email': email,
                'name': name,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'company_short_name': company_short_name,
                'body': body  # Pass original body to provide context
            }
        )
        
        if followup:
            draft_result = create_followup_draft(
                sender="me",
                to=email,
                subject=followup['subject'],
                message_text=followup['body'],
                lead_id=str(lead_id),
                sequence_num=followup.get('sequence_num', 2),
                original_html=followup.get('original_html'),
                in_reply_to=followup['in_reply_to']
            )
            
            if draft_result.get('draft_id'):
                store_lead_email_info(
                    lead_sheet={
                        'lead_data': {
                            'properties': {'hs_object_id': lead_id},
                            'email': email
                        },
                        'company_data': {
                            'company_short_name': company_short_name
                        }
                    },
                    draft_id=draft_result['draft_id'],
                    scheduled_date=followup['scheduled_send_date'],
                    body=followup['body'],
                    sequence_num=followup.get('sequence_num', 2)
                )
                logger.info(f"Successfully stored follow-up for Lead ID: {lead_id}")
            else:
                logger.error(f"Failed to create Gmail draft for Lead ID: {lead_id} "
                             f"({draft_result.get('error', 'Unknown error')})")
        else:
            logger.error(f"Failed to generate follow-up for Lead ID: {lead_id}")
        
    except Exception as e:
        logger.error(f"Error while generating follow-up: {str(e)}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    print("\nStarting follow-up generation for a specific lead...")
    test_followup_generation_for_specific_lead(61301)  # Replace with the specific lead_id you want to test




## tests\test_hubspot_leads_service.py

# test_hubspot_leads_service.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def get_random_lead_id():
    """Get a random lead_id from the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Modified query to ensure we get a valid HubSpot contact ID
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY NEWID()
            """)
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database: {lead_id}")
                return lead_id
            logger.warning("No valid lead_id found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting random lead_id: {str(e)}")
        return None

def test_lead_info():
    """Test function to pull HubSpot data for a random lead ID."""
    try:
        # Initialize services in correct order
        data_gatherer = DataGathererService()  # Initialize without parameters
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service  # Set the service after initialization
        leads_service = LeadsService(data_gatherer)
        
        # Get random contact ID from database
        contact_id = get_random_lead_id()
        if not contact_id:
            print("No lead IDs found in database")
            return
            
        print(f"\nFetching info for contact ID: {contact_id}")
        
        # Verify contact exists in HubSpot before proceeding
        try:
            # Test if we can get contact properties
            contact_props = hubspot_service.get_contact_properties(contact_id)
            if not contact_props:
                print(f"Contact ID {contact_id} not found in HubSpot")
                return
        except Exception as e:
            print(f"Error verifying contact in HubSpot: {str(e)}")
            return
            
        # Get lead summary using LeadsService
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error: {lead_info['error']}")
            return
            
        # Print results
        print("\nLead Information:")
        print("=" * 50)
        print(f"Last Reply Date: {lead_info['last_reply_date']}")
        print(f"Lifecycle Stage: {lead_info['lifecycle_stage']}")
        print(f"Company Name: {lead_info['company_name']}")
        print(f"Company Short Name: {lead_info['company_short_name']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_lead_info: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Enable debug logging
    logger.setLevel("DEBUG")
    test_lead_info()



