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
