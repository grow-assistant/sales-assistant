import os
import sys
import base64
from datetime import datetime
import pytz
from email.utils import parsedate_to_datetime
from thefuzz import fuzz

# Your existing helper imports
from utils.gmail_integration import get_gmail_service, search_messages
from utils.logging_setup import logger
from scheduling.database import get_db_connection
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def parse_sql_datetime(date_str: str) -> datetime:
    """
    Parse a datetime from your SQL table based on typical SQL formats:
      - 'YYYY-MM-DD HH:MM:SS'
      - 'YYYY-MM-DD HH:MM:SS.fff'
    Returns a UTC datetime object or None if parsing fails.
    """
    if not date_str:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Assuming DB times are UTC-naive or local; attach UTC as needed:
            dt_utc = dt.replace(tzinfo=pytz.UTC)
            return dt_utc
        except ValueError:
            continue

    logger.error(f"Unable to parse SQL datetime '{date_str}' with known formats.")
    return None


def parse_gmail_datetime(gmail_date_header: str) -> datetime:
    """
    Parse the Gmail 'Date' header (e.g. 'Wed, 15 Jan 2025 13:01:22 -0500').
    Convert it to a datetime in UTC.
    """
    if not gmail_date_header:
        return None
    try:
        dt = parsedate_to_datetime(gmail_date_header)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception as e:
        logger.error(f"Failed to parse Gmail date '{gmail_date_header}': {str(e)}", exc_info=True)
        return None


def is_within_24_hours(dt1: datetime, dt2: datetime) -> bool:
    """
    Return True if dt1 and dt2 are within 24 hours of each other.
    """
    if not dt1 or not dt2:
        return False
    diff = abs((dt1 - dt2).total_seconds())
    return diff <= 86400  # 86400 seconds = 24 hours


def is_within_days(dt1: datetime, dt2: datetime, days: int = 7) -> bool:
    """Return True if dt1 and dt2 are within X days of each other."""
    if not dt1 or not dt2:
        return False
    diff = abs((dt1 - dt2).total_seconds())
    return diff <= (days * 24 * 60 * 60)  # Convert days to seconds


###############################################################################
#                        DATABASE-RELATED FUNCTIONS
###############################################################################

def get_pending_emails(lead_id: int) -> list:
    """
    Retrieve emails where:
      - lead_id = ?
      - status IN ('pending','draft','sent')
      
    The table columns are:
      email_id, lead_id, name, email_address, sequence_num,
      body, scheduled_send_date, actual_send_date, created_at,
      status, draft_id, gmail_id
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email_id,
                       lead_id,
                       name,
                       email_address,
                       sequence_num,
                       body,
                       scheduled_send_date,
                       actual_send_date,
                       created_at,
                       status,
                       draft_id,
                       gmail_id
                  FROM emails
                 WHERE lead_id = ?
                   AND status IN ('pending','draft','sent') 
            """, (lead_id,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            record = {
                'email_id': row[0],
                'lead_id': row[1],
                'name': row[2],
                'email_address': row[3],
                'sequence_num': row[4],
                'body': row[5],
                'scheduled_send_date': str(row[6]) if row[6] else None,
                'actual_send_date': str(row[7]) if row[7] else None,
                'created_at': str(row[8]) if row[8] else None,
                'status': row[9],
                'draft_id': row[10],
                'gmail_id': row[11]
            }
            results.append(record)
        return results

    except Exception as e:
        logger.error(f"Error retrieving pending emails: {str(e)}", exc_info=True)
        return []


def update_email_record(email_id: int, gmail_id: str, actual_send_date_utc: datetime, body: str = None):
    """
    Update the emails table with the matched Gmail ID, set actual_send_date,
    mark status='sent', and optionally update the body.
    """
    try:
        actual_send_date_str = actual_send_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        
        if body:
            sql = """
                UPDATE emails
                   SET gmail_id = ?,
                       actual_send_date = ?,
                       status = 'sent',
                       body = ?
                 WHERE email_id = ?
            """
            params = (gmail_id, actual_send_date_str, body, email_id)
        else:
            sql = """
                UPDATE emails
                   SET gmail_id = ?,
                       actual_send_date = ?,
                       status = 'sent'
                 WHERE email_id = ?
            """
            params = (gmail_id, actual_send_date_str, email_id)
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Email ID {email_id} updated: gmail_id={gmail_id}, status='sent'.")
    except Exception as e:
        logger.error(f"Error updating email ID {email_id}: {str(e)}", exc_info=True)


def update_email_address(email_id: int, email_address: str):
    """Update the email_address field in the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE emails
                   SET email_address = ?
                 WHERE email_id = ?
            """, (email_address, email_id))
            conn.commit()
            logger.info(f"Updated email_id={email_id} with email_address={email_address}")
    except Exception as e:
        logger.error(f"Error updating email address for ID {email_id}: {str(e)}", exc_info=True)


###############################################################################
#                           GMAIL INTEGRATION
###############################################################################

def search_gmail_for_messages(to_address: str, subject: str, scheduled_dt: datetime = None) -> list:
    """
    Search Gmail for messages in 'sent' folder that match the criteria.
    If scheduled_dt is provided, only return messages within 7 days.
    """
    service = get_gmail_service()
    
    logger.info(f"Searching Gmail with to_address={to_address}")
    logger.info(f"Searching Gmail with subject={subject}")
    logger.info(f"Scheduled date: {scheduled_dt}")
    
    # Modify query to be more lenient
    subject_quoted = subject.replace('"', '\\"')  # Escape any quotes in subject
    query = f'in:sent to:{to_address}'
    
    logger.info(f"Gmail search query: {query}")
    messages = search_messages(query)
    
    if messages:
        logger.info(f"Found {len(messages)} Gmail messages matching query: {query}")
        
        # Filter messages by date if scheduled_dt is provided
        if scheduled_dt:
            filtered_messages = []
            for msg in messages:
                details = get_email_details(msg)
                sent_date = details.get('date_parsed')
                if sent_date and is_within_days(sent_date, scheduled_dt):
                    filtered_messages.append(msg)
                    logger.info(f"Found matching message: Subject='{details.get('subject')}', "
                              f"To={details.get('to')}, Date={details.get('date_raw')}")
            
            logger.info(f"Found {len(filtered_messages)} messages within 7 days of scheduled date")
            return filtered_messages
        
        return messages
    else:
        logger.info("No messages found with this query")
        return []


def get_email_details(gmail_message: dict) -> dict:
    """Extract details including body from a Gmail message."""
    try:
        service = get_gmail_service()
        full_message = service.users().messages().get(
            userId='me', 
            id=gmail_message['id'], 
            format='full'
        ).execute()
        
        payload = full_message.get('payload', {})
        headers = payload.get('headers', [])
        
        def find_header(name: str):
            return next((h['value'] for h in headers if h['name'].lower() == name.lower()), None)
        
        # Extract body from multipart message
        def get_body_from_parts(parts):
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    if part.get('body', {}).get('data'):
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part.get('parts'):  # Handle nested parts
                    body = get_body_from_parts(part['parts'])
                    if body:
                        return body
            return None

        # Get body content
        body = None
        if payload.get('mimeType') == 'text/plain':
            if payload.get('body', {}).get('data'):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif payload.get('mimeType', '').startswith('multipart/'):
            body = get_body_from_parts(payload.get('parts', []))

        # Clean up the body - remove signature
        if body:
            body = body.split('Time zone: Eastern Time')[0].strip()
            logger.info("=== Extracted Email Body ===")
            logger.info(body)
            logger.info("=== End of Email Body ===")
        
        date_str = find_header('Date')
        parsed_dt = parse_gmail_datetime(date_str)

        details = {
            'gmail_id':  full_message['id'],
            'thread_id': full_message['threadId'],
            'subject':   find_header('Subject'),
            'from':      find_header('From'),
            'to':        find_header('To'),
            'cc':        find_header('Cc'),
            'bcc':       find_header('Bcc'),
            'date_raw':  date_str,
            'date_parsed': parsed_dt,
            'body':      body
        }
        
        return details
        
    except Exception as e:
        logger.error(f"Error getting email details: {str(e)}", exc_info=True)
        return {}


###############################################################################
#                                MAIN LOGIC
###############################################################################

def get_draft_emails(email_id: int = None) -> list:
    """Retrieve specific draft email or all draft emails that need processing."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if email_id:
                cursor.execute("""
                    SELECT email_id,
                           lead_id,
                           subject,
                           name,
                           company_name,
                           body,
                           scheduled_send_date,
                           created_at,
                           draft_id,
                           email_address,
                           status,
                           company_city,
                           company_st,
                           company_type,
                           sequence_num,
                           gmail_id
                      FROM emails 
                     WHERE email_id = ?
                       AND status IN ('draft','sent')
                """, (email_id,))
            else:
                cursor.execute("""
                    SELECT email_id,
                           lead_id,
                           subject,
                           name,
                           company_name,
                           body,
                           scheduled_send_date,
                           created_at,
                           draft_id,
                           email_address,
                           status,
                           company_city,
                           company_st,
                           company_type,
                           sequence_num,
                           gmail_id
                      FROM emails 
                     WHERE status = 'draft'
                  ORDER BY email_id DESC
                """)
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'email_id': row[0],
                    'lead_id': row[1],
                    'subject': row[2],
                    'name': row[3],
                    'company_name': row[4],
                    'body': row[5],
                    'scheduled_send_date': str(row[6]) if row[6] else None,
                    'created_at': str(row[7]) if row[7] else None,
                    'draft_id': row[8],
                    'email_address': row[9],
                    'status': row[10],
                    'company_city': row[11],
                    'company_st': row[12],
                    'company_type': row[13],
                    'sequence_num': row[14],
                    'gmail_id': row[15]
                })
            return results
    except Exception as e:
        logger.error(f"Error retrieving draft emails: {str(e)}", exc_info=True)
        return []

def main():
    """Process all draft emails."""
    logger.info("=== Starting Gmail match process ===")

    # Get all draft emails (no specific email_id)
    pending = get_draft_emails()
    if not pending:
        logger.info("No draft emails found. Exiting.")
        return

    logger.info(f"Found {len(pending)} draft emails to process")

    # Initialize HubSpot service
    hubspot = HubspotService(HUBSPOT_API_KEY)

    # Process each record
    for record in pending:
        email_id = record['email_id']
        lead_id = record['lead_id']
        to_address = record['email_address']
        subject = record['subject'] or ""
        scheduled_dt = parse_sql_datetime(record['scheduled_send_date'])
        created_dt = parse_sql_datetime(record['created_at'])

        logger.info("\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Lead ID: {lead_id}")
        logger.info(f"Current email address: {to_address}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Scheduled: {scheduled_dt}")
        logger.info(f"Created: {created_dt}")

        # If no email address, try to get it from HubSpot
        if not to_address:
            try:
                logger.info(f"Fetching contact data from HubSpot for lead_id: {lead_id}")
                contact_props = hubspot.get_contact_properties(str(lead_id))
                
                if contact_props and 'email' in contact_props:
                    email_address = contact_props['email']
                    if email_address:
                        logger.info(f"Found email address in HubSpot: {email_address}")
                        update_email_address(email_id, email_address)
                        to_address = email_address
                    else:
                        logger.warning(f"No email found in HubSpot properties for lead_id={lead_id}")
                        continue
                else:
                    logger.warning(f"No contact data found in HubSpot for lead_id={lead_id}")
                    continue
            except Exception as e:
                logger.error(f"Error fetching HubSpot data: {str(e)}", exc_info=True)
                continue

        # Now proceed with Gmail search
        messages = search_gmail_for_messages(to_address, subject, scheduled_dt)
        if not messages:
            logger.info(f"No matching Gmail messages found for email_id={email_id}.")
            continue

        # Try to find a valid match
        matched_gmail = None
        for msg in messages:
            details = get_email_details(msg)
            dt_sent = details.get('date_parsed')
            if not dt_sent:
                continue

            logger.info(f"\nChecking message sent at {dt_sent}:")
            logger.info(f"Within 7 days of scheduled ({scheduled_dt}): {is_within_days(dt_sent, scheduled_dt)}")
            
            if is_within_days(dt_sent, scheduled_dt):
                matched_gmail = details
                break

        if matched_gmail:
            logger.info(f"\nMatched Gmail ID={matched_gmail['gmail_id']} for email_id={email_id}")
            logger.info(f"Sent date: {matched_gmail['date_parsed']}")
            logger.info(f"To: {matched_gmail['to']}")
            logger.info("Updating database record...")
            
            update_email_record(
                email_id=email_id,
                gmail_id=matched_gmail['gmail_id'],
                actual_send_date_utc=matched_gmail['date_parsed'],
                body=matched_gmail.get('body', '')
            )
        else:
            logger.info(f"\nNo valid match found for email_id={email_id}.")

    logger.info("\n=== Completed email matching process. ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()
