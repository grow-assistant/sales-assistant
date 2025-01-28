
## main.py

#!/usr/bin/env python3
# main.py
# -----------------------------------------------------------------------------
# UPDATED FILE: REMOVED FILTERS / USING LIST-BASED PULL
# -----------------------------------------------------------------------------

import logging
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import openai
import os
import shutil
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from utils.xai_integration import (
    personalize_email_with_xai,
    build_context_block
)
from utils.gmail_integration import search_messages
import json
import time

# -----------------------------------------------------------------------------
# PROJECT IMPORTS (Adjust paths/names as needed)
# -----------------------------------------------------------------------------
from services.hubspot_service import HubspotService
from services.company_enrichment_service import CompanyEnrichmentService
from services.data_gatherer_service import DataGathererService
from scripts.golf_outreach_strategy import (
    get_best_outreach_window, 
    get_best_month, 
    adjust_send_time
)
from scripts.build_template import build_outreach_email
from scheduling.database import get_db_connection
from scheduling.extended_lead_storage import store_lead_email_info
from scheduling.followup_generation import generate_followup_email_xai
from scheduling.followup_scheduler import start_scheduler
from config.settings import (
    HUBSPOT_API_KEY, 
    OPENAI_API_KEY, 
    MODEL_FOR_GENERAL, 
    PROJECT_ROOT, 
    CLEAR_LOGS_ON_START
)
from utils.exceptions import LeadContextError
from utils.gmail_integration import create_draft, store_draft_info
from utils.logging_setup import logger, setup_logging
from services.conversation_analysis_service import ConversationAnalysisService

# -----------------------------------------------------------------------------
# LIMIT FOR HOW MANY CONTACTS/LEADS TO PROCESS
# -----------------------------------------------------------------------------
LEADS_TO_PROCESS = 200  # This is just a processing limit, not a filter.

# -----------------------------------------------------------------------------
# INIT SERVICES & LOGGING
# -----------------------------------------------------------------------------
setup_logging()
data_gatherer = DataGathererService()
conversation_analyzer = ConversationAnalysisService()

# -----------------------------------------------------------------------------
# CONTEXT MANAGER FOR WORKFLOW STEPS
# -----------------------------------------------------------------------------
@contextmanager
def workflow_step(step_name: str, step_description: str, logger_context: dict = None):
    """Context manager for logging workflow steps."""
    logger_context = logger_context or {}
    logger_context.update({
        'step_number': step_name,
        'step_description': step_description
    })
    logger.info(f"Starting {step_description}", extra=logger_context)
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {step_description}: {str(e)}", extra=logger_context)
        raise
    else:
        logger.info(f"Completed {step_description}", extra=logger_context)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS (still used in the pipeline)
# -----------------------------------------------------------------------------
def clear_files_on_start():
    """Clear log files and temp directories on startup."""
    try:
        logs_dir = Path(PROJECT_ROOT) / "logs"
        if logs_dir.exists():
            for file in logs_dir.glob("*"):
                if file.is_file():
                    try:
                        file.unlink()
                    except PermissionError:
                        logger.debug(f"Skipping locked file: {file}")
                        continue
            logger.debug("Cleared logs directory")

        temp_dir = Path(PROJECT_ROOT) / "temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()
                logger.debug("Cleared temp directory")
            except PermissionError:
                logger.debug("Skipping locked temp directory")

    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}")

def extract_lead_data(company_props: Dict, lead_props: Dict) -> Dict:
    """Extract and organize lead and company data."""
    return {
        "company_data": {
            "name": company_props.get("name", ""),
            "city": company_props.get("city", ""),
            "state": company_props.get("state", ""),
            "club_type": company_props.get("club_type", ""),
            "facility_complexity": company_props.get("facility_complexity", ""),
            "has_pool": company_props.get("has_pool", ""),
            "has_tennis_courts": company_props.get("has_tennis_courts", ""),
            "number_of_holes": company_props.get("number_of_holes", ""),
            "public_private_flag": company_props.get("public_private_flag", ""),
            "geographic_seasonality": company_props.get("geographic_seasonality", ""),
            "club_info": company_props.get("club_info", ""),
            "peak_season_start_month": company_props.get("peak_season_start_month", ""),
            "peak_season_end_month": company_props.get("peak_season_end_month", ""),
            "start_month": company_props.get("start_month", ""),
            "end_month": company_props.get("end_month", "")
        },
        "lead_data": {
            "firstname": lead_props.get("firstname", ""),
            "lastname": lead_props.get("lastname", ""),
            "email": lead_props.get("email", ""),
            "jobtitle": lead_props.get("jobtitle", "")
        }
    }

def gather_personalization_data(company_name: str, city: str, state: str) -> Dict:
    """Gather additional personalization data for the company."""
    try:
        news_result = data_gatherer.gather_club_news(company_name)
        
        has_news = False
        news_text = ""
        if news_result:
            if isinstance(news_result, tuple):
                news_text = news_result[0]
            else:
                news_text = str(news_result)
            has_news = "has not been" not in news_text.lower()
        
        return {
            "has_news": has_news,
            "news_text": news_text if has_news else None
        }
        
    except Exception as e:
        logger.error(f"Error gathering personalization data: {str(e)}")
        return {
            "has_news": False,
            "news_text": None
        }

def clean_company_name(text: str) -> str:
    """Replace old company name references with current name."""
    if not text:
        return text
    replacements = {
        "Byrdi": "Swoop",
        "byrdi": "swoop",
        "BYRDI": "SWOOP"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def summarize_lead_interactions(lead_sheet: dict) -> str:
    """
    Collects all prior emails and notes from the lead_sheet, then uses OpenAI
    to summarize them. This remains here in case you still process extended data.
    """
    try:
        lead_data = lead_sheet.get("lead_data", {})
        emails = lead_data.get("emails", [])
        notes = lead_data.get("notes", [])
        
        logger.debug(f"Found {len(emails)} emails and {len(notes)} notes to summarize")
        
        interactions = []
        
        # Collect emails
        for email in sorted(emails, key=lambda x: x.get('timestamp', ''), reverse=True):
            if isinstance(email, dict):
                date = email.get('timestamp', '').split('T')[0]
                subject = clean_company_name(email.get('subject', '').encode('utf-8', errors='ignore').decode('utf-8'))
                body = clean_company_name(email.get('body_text', '').encode('utf-8', errors='ignore').decode('utf-8'))
                direction = email.get('direction', '')
                body = body.split('On ')[0].strip()
                email_type = "from the lead" if direction == "INCOMING_EMAIL" else "to the lead"
                
                logger.debug(f"Processing email from {date}: {subject[:50]}...")
                
                interaction = {
                    'date': date,
                    'type': f'email {email_type}',
                    'direction': direction,
                    'subject': subject,
                    'notes': body[:1000]
                }
                interactions.append(interaction)
        
        # Collect notes
        for note in sorted(notes, key=lambda x: x.get('timestamp', ''), reverse=True):
            if isinstance(note, dict):
                date = note.get('timestamp', '').split('T')[0]
                content = clean_company_name(note.get('body', '').encode('utf-8', errors='ignore').decode('utf-8'))
                
                interaction = {
                    'date': date,
                    'type': 'note',
                    'direction': 'internal',
                    'subject': 'Internal Note',
                    'notes': content[:1000]
                }
                interactions.append(interaction)
        
        if not interactions:
            return "No prior interactions found."

        interactions.sort(key=lambda x: x['date'], reverse=True)
        recent_interactions = interactions[:10]
        
        prompt = (
            "Please summarize these interactions, focusing on:\n"
            "1. Most recent email FROM THE LEAD if there is one\n"
            "2. Key points of interest or next steps discussed\n"
            "3. Overall progression of the conversation\n\n"
            "Recent Interactions:\n"
        )
        
        for interaction in recent_interactions:
            prompt += f"\nDate: {interaction['date']}\n"
            prompt += f"Type: {interaction['type']}\n"
            prompt += f"Direction: {interaction['direction']}\n"
            prompt += f"Subject: {interaction['subject']}\n"
            prompt += f"Content: {interaction['notes']}\n"
            prompt += "-" * 50 + "\n"
        
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=MODEL_FOR_GENERAL,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful assistant that summarizes business interactions. "
                            "Anything from Ty or Ryan is from Swoop."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error getting summary from OpenAI: {str(e)}")
            return "Error summarizing interactions."
            
    except Exception as e:
        logger.error(f"Error in summarize_lead_interactions: {str(e)}")
        return "Error processing interactions."

def schedule_followup(lead_id: int, email_id: int):
    """Schedule a follow-up email."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                e.subject, e.body, e.created_at,
                l.email, l.first_name,
                c.name, c.state
            FROM emails e
            JOIN leads l ON e.lead_id = l.lead_id
            LEFT JOIN companies c ON l.company_id = c.company_id
            WHERE e.email_id = ? AND e.lead_id = ?
        """, (email_id, lead_id))
        
        result = cursor.fetchone()
        if not result:
            logger.error(f"No email found for email_id={email_id}")
            return
            
        orig_subject, orig_body, created_at, email, first_name, company_name, state = result
        
        original_email = {
            'email': email,
            'first_name': first_name,
            'name': company_name,
            'state': state,
            'subject': orig_subject,
            'body': orig_body,
            'created_at': created_at
        }
        
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            email_id=email_id,
            sequence_num=2,
            original_email=original_email
        )
        
        if followup:
            draft_result = create_draft(
                sender="me",
                to=followup.get('email'),
                subject=followup.get('subject'),
                message_text=followup.get('body')
            )
            
            if draft_result["status"] == "ok":
                store_draft_info(
                    lead_id=lead_id,
                    draft_id=draft_result["draft_id"],
                    scheduled_date=followup.get('scheduled_send_date'),
                    subject=followup.get('subject'),
                    body=followup.get('body'),
                    sequence_num=followup.get('sequence_num', 2)
                )
                logger.info(f"Follow-up email scheduled with draft_id {draft_result.get('draft_id')}")
            else:
                logger.error("Failed to create Gmail draft for follow-up")
        else:
            logger.error("Failed to generate follow-up email content")
            
    except Exception as e:
        logger.error(f"Error scheduling follow-up: {str(e)}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def store_draft_info(lead_id, name, email_address, sequence_num, body, scheduled_date, draft_id):
    """
    Persists the draft info to the 'emails' table.
    """
    try:
        logger.debug(f"[store_draft_info] Attempting to store draft info for lead_id={lead_id}, scheduled_date={scheduled_date}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE emails
            SET name = ?, 
                email_address = ?, 
                sequence_num = ?,
                body = ?,
                scheduled_send_date = ?,
                draft_id = ?
            WHERE lead_id = ? AND sequence_num = ?
            
            IF @@ROWCOUNT = 0
            BEGIN
                INSERT INTO emails (
                    lead_id,
                    name,
                    email_address,
                    sequence_num,
                    body,
                    scheduled_send_date,
                    status,
                    draft_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, 'draft', ?
                )
            END
            """,
            (
                # UPDATE parameters
                name,
                email_address,
                sequence_num,
                body,
                scheduled_date,
                draft_id,
                lead_id,
                sequence_num,
                
                # INSERT parameters
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_date,
                draft_id
            )
        )
        
        conn.commit()
        logger.debug(f"[store_draft_info] Successfully wrote scheduled_date={scheduled_date} for lead_id={lead_id}, draft_id={draft_id}")
        
    except Exception as e:
        logger.error(f"[store_draft_info] Failed to store draft info: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def calculate_send_date(geography, persona, state_code, season_data=None):
    """Calculate optimal send date based on geography and persona."""
    try:
        if season_data:
            season_data = {
                "peak_season_start": str(season_data.get("peak_season_start", "")),
                "peak_season_end": str(season_data.get("peak_season_end", ""))
            }
        
        outreach_window = get_best_outreach_window(
            persona=persona,
            geography=geography,
            season_data=season_data
        )
        
        best_months = outreach_window.get("Best Month", [])
        best_time = outreach_window.get("Best Time", {"start": 9, "end": 11})
        best_days = outreach_window.get("Best Day", [1,2,3])  # Mon-Fri default
        
        now = datetime.now()
        target_date = now  # Start with today instead of tomorrow
        
        # If current time is past the best window end time, move to next day
        if now.hour >= int(best_time["end"]):
            target_date = now + timedelta(days=1)
        
        # Shift to a "best month"
        while target_date.month not in best_months:
            if target_date.month == 12:
                target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
            else:
                target_date = target_date.replace(month=target_date.month + 1, day=1)
        
        # Shift to a "best day"
        while target_date.weekday() not in best_days:
            target_date += timedelta(days=1)
        
        # Ensure we're using integers for hour and minute
        send_hour = int(best_time["start"])
        send_minute = random.randint(0, 59)
        
        # If it's today and current time is before best window start, use best window start
        if target_date.date() == now.date() and now.hour < send_hour:
            send_hour = int(best_time["start"])
        # If it's today and current time is within window, use current hour + small delay
        elif target_date.date() == now.date() and now.hour >= send_hour and now.hour < int(best_time["end"]):
            send_hour = now.hour
            send_minute = now.minute + random.randint(2, 15)
            if send_minute >= 60:
                send_hour += 1
                send_minute -= 60
                if send_hour >= int(best_time["end"]):
                    target_date += timedelta(days=1)
                    send_hour = int(best_time["start"])
        
        send_date = target_date.replace(
            hour=send_hour,
            minute=send_minute,
            second=0,
            microsecond=0
        )
        
        final_send_date = adjust_send_time(send_date, state_code)
        return final_send_date
        
    except Exception as e:
        logger.error(f"Error calculating send date: {str(e)}", exc_info=True)
        # Fallback to next business day at 10am
        return datetime.now() + timedelta(days=1, hours=10)

def get_next_month_first_day(current_date):
    if current_date.month == 12:
        return current_date.replace(year=current_date.year + 1, month=1, day=1)
    return current_date.replace(month=current_date.month + 1, day=1)

def get_template_path(club_type: str, role: str) -> str:
    """
    Get the appropriate template path based on club type and role.
    """
    logger.info(f"[TARGET] Selecting template for club_type={club_type}, role={role}")
    
    club_type_map = {
        "Country Club": "country_club",
        "Private Course": "private_course",
        "Private Club": "private_course",
        "Resort Course": "resort_course",
        "Resort": "resort_course",
        "Public Course": "public_high_daily_fee",
        "Public - High Daily Fee": "public_high_daily_fee",
        "Public - Low Daily Fee": "public_low_daily_fee",
        "Public": "public_high_daily_fee",
        "Semi-Private": "public_high_daily_fee",
        "Semi Private": "public_high_daily_fee",
        "Municipal": "public_low_daily_fee",
        "Municipal Course": "public_low_daily_fee",
        "Management Company": "management_companies",
        "Unknown": "country_club",
    }
    
    role_map = {
        "General Manager": "general_manager",
        "GM": "general_manager",
        "Club Manager": "general_manager",
        "Golf Professional": "golf_ops",
        "Head Golf Professional": "golf_ops",
        "Director of Golf": "golf_ops",
        "Golf Operations": "golf_ops",
        "Golf Operations Manager": "golf_ops",
        "F&B Manager": "fb_manager",
        "Food & Beverage Manager": "fb_manager",
        "Food and Beverage Manager": "fb_manager",
        "F&B Director": "fb_manager",
        "Food & Beverage Director": "fb_manager",
        "Food and Beverage Director": "fb_manager",
        "Restaurant Manager": "fb_manager",
    }
    
    normalized_club_type = club_type_map.get(club_type, "country_club")
    normalized_role = role_map.get(role, "general_manager")
    
    # Randomly select template variation
    sequence_num = random.randint(1, 2)
    logger.info(f"[VARIANT] Selected template variation {sequence_num}")
    
    template_path = Path(PROJECT_ROOT) / "docs" / "templates" / normalized_club_type / f"{normalized_role}_initial_outreach_{sequence_num}.md"
    logger.info(f"[PATH] Using template path: {template_path}")
    
    if not template_path.exists():
        fallback_path = Path(PROJECT_ROOT) / "docs" / "templates" / normalized_club_type / f"fallback_{sequence_num}.md"
        logger.warning(f"❌ Specific template not found at {template_path}, falling back to {fallback_path}")
        template_path = fallback_path
    
    return str(template_path)

def clear_logs():
    try:
        if os.path.exists('logs/app.log'):
            with open('logs/app.log', 'w') as f:
                f.truncate(0)
    except PermissionError:
        logger.warning("Could not clear log file - file is in use")
    except Exception as e:
        logger.error(f"Error clearing log file: {e}")

def has_recent_email(email_address: str, months: int = 2) -> bool:
    """Check if we've sent an email to this address in the last X months."""
    try:
        cutoff_date = (datetime.now() - timedelta(days=30 * months)).strftime('%Y/%m/%d')
        query = f"to:{email_address} after:{cutoff_date}"
        messages = search_messages(query=query)
        
        if messages:
            logger.info(f"Found previous email to {email_address} within last {months} months")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error checking recent emails for {email_address}: {str(e)}")
        return False

def replace_placeholders(text: str, lead_data: dict) -> str:
    """Replace placeholders in text with actual values."""
    text = clean_company_name(text)  # Clean any old references
    replacements = {
        "[firstname]": lead_data["lead_data"].get("firstname", ""),
        "[LastName]": lead_data["lead_data"].get("lastname", ""),
        "[clubname]": lead_data["company_data"].get("name", ""),
        "[JobTitle]": lead_data["lead_data"].get("jobtitle", ""),
        "[companyname]": lead_data["company_data"].get("name", ""),
        "[City]": lead_data["company_data"].get("city", ""),
        "[State]": lead_data["company_data"].get("state", "")
    }
    result = text
    for placeholder, value in replacements.items():
        if value:
            result = result.replace(placeholder, value)
    return result

# -----------------------------------------------------------------------------
# NEW: LIST-BASED CONTACT PULL
# -----------------------------------------------------------------------------
def get_contacts_from_list(list_id: str, hubspot: HubspotService) -> List[Dict[str, Any]]:
    """
    Pull contacts from a specified HubSpot list. This replaces all prior
    lead-filter or company-filter logic in main. We simply fetch the list
    membership and then retrieve each contact individually.
    """
    try:
        # 1) Get list memberships
        url = f"https://api.hubapi.com/crm/v3/lists/{list_id}/memberships"
        logger.debug(f"Fetching list memberships from: {url}")
        memberships = hubspot._make_hubspot_get(url)

        # 2) Extract record IDs
        record_ids = []
        if isinstance(memberships, dict) and "results" in memberships:
            record_ids = [result["recordId"] for result in memberships.get("results", [])]

        logger.debug(f"Found {len(record_ids)} records in list {list_id}")

        # 3) Fetch contact details for each ID
        contacts = []
        for record_id in record_ids:
            try:
                contact_url = f"https://api.hubapi.com/crm/v3/objects/contacts/{record_id}"
                params = {
                    "properties": [
                        "email",
                        "firstname",
                        "lastname",
                        "jobtitle",
                        "associatedcompanyid"
                    ]
                }
                contact_data = hubspot._make_hubspot_get(contact_url, params)
                if contact_data:
                    contacts.append(contact_data)
            except Exception as e:
                logger.warning(f"Failed to fetch details for contact {record_id}: {str(e)}")
                continue

        logger.info(f"Successfully retrieved {len(contacts)} contacts from list {list_id}")
        return contacts

    except Exception as e:
        logger.error(f"Error getting contacts from list {list_id}: {str(e)}", exc_info=True)
        return []

def get_company_by_id(hubspot: HubspotService, company_id: str) -> Dict:
    """Get company details by ID."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/companies/{company_id}"
        params = {
            "properties": [
                "name", 
                "city", 
                "state", 
                "club_type",
                "facility_complexity",
                "geographic_seasonality",
                "has_pool",
                "has_tennis_courts",
                "number_of_holes",
                "public_private_flag",
                "club_info",
                "peak_season_start_month",
                "peak_season_end_month",
                "start_month",
                "end_month"
            ]
        }
        response = hubspot._make_hubspot_get(url, params)
        return response
    except Exception as e:
        logger.error(f"Error getting company {company_id}: {e}")
        return {}

# -----------------------------------------------------------------------------
# REPLACEMENT WORKFLOW USING THE LIST
# -----------------------------------------------------------------------------
def main_list_workflow():
    """
    1) Pull contacts from a specified HubSpot list.
    2) For each contact, retrieve the associated company (if any).
    3) Enrich the company data, gather personalization, build & store an outreach email draft.
    """
    try:
        # Limit how many contacts (leads) we process
        leads_processed = 0
        
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
        conversation_analyzer = ConversationAnalysisService()

        # Replace this with your actual HubSpot list ID:
        LIST_ID = "221"
        
        with workflow_step("1", "Get contacts from HubSpot list", workflow_context):
            contacts = get_contacts_from_list(LIST_ID, hubspot)
            logger.info(f"Found {len(contacts)} contacts to process from list {LIST_ID}")
            
            # Shuffle to randomize processing order if desired
            random.shuffle(contacts)

        with workflow_step("2", "Process each contact & its associated company", workflow_context):
            for contact in contacts:
                contact_id = contact.get("id")
                props = contact.get("properties", {})
                email = props.get("email")

                logger.info(f"Processing contact: {email} (ID: {contact_id})")

                if not email:
                    logger.info(f"Skipping contact {contact_id} - no email found.")
                    continue

                # Check if we've recently sent them an email
                if has_recent_email(email):
                    logger.info(f"Skipping {email} - an email was sent in the last 2 months")
                    continue

                # 1) Grab company if associated
                associated_company_id = props.get("associatedcompanyid")
                company_props = {}
                
                if associated_company_id:
                    company_data = get_company_by_id(hubspot, associated_company_id)
                    if company_data and company_data.get("id"):
                        company_props = company_data.get("properties", {})
                        
                        # Optionally check competitor from website, etc.
                        website = company_props.get("website")
                        if website:
                            competitor_info = data_gatherer.check_competitor_on_website(website)
                            if competitor_info["status"] == "success" and competitor_info["competitor"]:
                                competitor = competitor_info["competitor"]
                                logger.debug(f"Found competitor {competitor} for company {associated_company_id}")
                                try:
                                    # Update HubSpot
                                    hubspot._update_company_properties(
                                        company_data["id"], 
                                        {"competitor": competitor}
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update competitor for company {associated_company_id}: {e}")
                            
                        # Enrich the company data
                        enrichment_result = company_enricher.enrich_company(company_data["id"])
                        if enrichment_result.get("success", False):
                            company_props.update(enrichment_result.get("data", {}))
                
                # 2) Extract lead + company data into our unified structure
                lead_data_full = extract_lead_data(company_props, {
                    "firstname": props.get("firstname", ""),
                    "lastname": props.get("lastname", ""),
                    "email": email,
                    "jobtitle": props.get("jobtitle", "")
                })

                # 3) Gather personalization data
                personalization = gather_personalization_data(
                    company_name=lead_data_full["company_data"]["name"],
                    city=lead_data_full["company_data"]["city"],
                    state=lead_data_full["company_data"]["state"]
                )

                # 4) Pick a template path
                template_path = get_template_path(
                    club_type=lead_data_full["company_data"]["club_type"],
                    role=lead_data_full["lead_data"]["jobtitle"]
                )

                # 5) Calculate best send date
                send_date = calculate_send_date(
                    geography=lead_data_full["company_data"]["geographic_seasonality"],
                    persona=lead_data_full["lead_data"]["jobtitle"],
                    state_code=lead_data_full["company_data"]["state"],
                    season_data={
                        "peak_season_start": lead_data_full["company_data"].get("peak_season_start_month"),
                        "peak_season_end": lead_data_full["company_data"].get("peak_season_end_month")
                    }
                )

                # 6) Build outreach email
                email_content = build_outreach_email(
                    template_path=template_path,
                    profile_type=lead_data_full["lead_data"]["jobtitle"],
                    placeholders={
                        "firstname": lead_data_full["lead_data"]["firstname"],
                        "LastName": lead_data_full["lead_data"]["lastname"],
                        "companyname": lead_data_full["company_data"]["name"],
                        "company_short_name": lead_data_full["company_data"].get("company_short_name", ""),
                        "JobTitle": lead_data_full["lead_data"]["jobtitle"],
                        "company_info": lead_data_full["company_data"].get("club_info", ""),
                        "has_news": personalization.get("has_news", False),
                        "news_text": personalization.get("news_text", ""),
                        "clubname": lead_data_full["company_data"]["name"]
                    },
                    current_month=datetime.now().month,
                    start_peak_month=lead_data_full["company_data"].get("peak_season_start_month"),
                    end_peak_month=lead_data_full["company_data"].get("peak_season_end_month")
                )

                if not email_content:
                    logger.error(f"Failed to generate email content for contact {contact_id}")
                    continue

                # 7) Replace placeholders in subject/body
                subject = replace_placeholders(email_content[0], lead_data_full)
                body = replace_placeholders(email_content[1], lead_data_full)

                # 8) Get conversation summary
                try:
                    logger.debug(f"Starting conversation analysis for email: {email}")
                    
                    # Log the input state
                    logger.debug(f"Conversation analyzer state before analysis:")
                    logger.debug(f"- Email: {email}")
                    logger.debug(f"- Analyzer type: {type(conversation_analyzer).__name__}")
                    
                    # Perform the analysis with timing
                    start_time = time.time()
                    conversation_summary = conversation_analyzer.analyze_conversation(email)
                    analysis_time = time.time() - start_time
                    
                    # Log the results
                    logger.info(f"Conversation analysis completed in {analysis_time:.2f} seconds")
                    logger.debug(f"Analysis results:")
                    logger.debug(f"- Summary length: {len(conversation_summary) if conversation_summary else 0}")
                    logger.debug(f"- Summary content: {conversation_summary}")
                    
                    if not conversation_summary:
                        logger.warning("Conversation analysis returned empty summary")
                
                except Exception as e:
                    logger.error(f"Error during conversation analysis for {email}: {str(e)}", exc_info=True)
                    conversation_summary = None
                    raise

                # 9) Build context block
                context = build_context_block(
                    interaction_history=conversation_summary,
                    original_email={"subject": subject, "body": body},
                    company_data={
                        "name": lead_data_full["company_data"]["name"],
                        "company_short_name": (
                            lead_data_full["company_data"].get("company_short_name") or
                            lead_data_full.get("company_short_name") or
                            lead_data_full["company_data"]["name"].split(" ")[0]
                        ),
                        "city": lead_data_full["company_data"].get("city", ""),
                        "state": lead_data_full["company_data"].get("state", ""),
                        "club_type": lead_data_full["company_data"]["club_type"],
                        "club_info": lead_data_full["company_data"].get("club_info", "")
                    }
                )

                # 10) Personalize with xAI
                personalized_content = personalize_email_with_xai(
                    lead_sheet=lead_data_full,
                    subject=subject,
                    body=body,
                    summary=conversation_summary,
                    context=context
                )

                # 11) Create Gmail draft
                draft_result = create_draft(
                    sender="me",
                    to=email,
                    subject=personalized_content["subject"],
                    message_text=personalized_content["body"]
                )

                if draft_result["status"] == "ok":
                    # 12) Store info in local DB
                    store_lead_email_info(
                        lead_sheet={
                            "lead_data": {
                                "email": email,
                                "properties": {
                                    "hs_object_id": contact_id,
                                    "firstname": lead_data_full["lead_data"]["firstname"],
                                    "lastname": lead_data_full["lead_data"]["lastname"]
                                }
                            },
                            "company_data": {
                                "name": lead_data_full["company_data"]["name"],
                                "city": lead_data_full["company_data"]["city"],
                                "state": lead_data_full["company_data"]["state"],
                                "company_type": lead_data_full["company_data"]["club_type"]
                            }
                        },
                        draft_id=draft_result["draft_id"],
                        scheduled_date=send_date,
                        body=email_content[1],
                        sequence_num=1
                    )
                    logger.info(f"Created draft email for {email}")
                else:
                    logger.error(f"Failed to create Gmail draft for contact {contact_id}")

                leads_processed += 1
                logger.info(f"Completed processing contact {contact_id} ({leads_processed}/{LEADS_TO_PROCESS})")

                if leads_processed >= LEADS_TO_PROCESS:
                    logger.info(f"Reached processing limit of {LEADS_TO_PROCESS} contacts")
                    return

    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in main_list_workflow: {str(e)}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# SINGLE ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Start scheduler in background
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # We now only use the list-based approach
    main_list_workflow()

# -----------------------------------------------------------------------------
# RUN SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()




## test_followup_generation.py

## test_followup_generation.py

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




## scripts\golf_outreach_strategy.py

"""
Scripts for determining optimal outreach timing based on club and contact attributes.
"""
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




## scripts\monitor\monitor_email_review_status.py

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from datetime import datetime
import pytz
from utils.gmail_integration import get_gmail_service
from utils.logging_setup import logger
from scheduling.database import get_db_connection
import base64

###############################################################################
#                           CONSTANTS
###############################################################################

TO_REVIEW_LABEL = "to_review"
REVIEWED_LABEL = "reviewed"
SQL_DRAFT_STATUS = "draft"
SQL_REVIEWED_STATUS = "reviewed"

###############################################################################
#                           GMAIL INTEGRATION
###############################################################################

def get_draft_emails() -> list:
    """
    Retrieve all emails with status='draft'.
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
                 WHERE status = ?
            """, (SQL_DRAFT_STATUS,))
            
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
        logger.error(f"Error retrieving draft emails: {str(e)}", exc_info=True)
        return []

def get_gmail_draft_by_id(service, draft_id: str) -> dict:
    """Retrieve a specific Gmail draft by its draftId."""
    try:
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        return draft.get("message", {})
    except Exception as e:
        logger.error(f"Error fetching draft {draft_id}: {str(e)}")
        return {}

def get_draft_body(message: dict) -> str:
    """Extract the body text from a Gmail draft message."""
    try:
        body = ""
        if 'payload' in message and 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        elif 'payload' in message and 'body' in message['payload']:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
        
        if not body:
            logger.error("No body content found in message")
            return ""
            
        # Log the first few characters to help with debugging
        preview = body[:100] + "..." if len(body) > 100 else body
        logger.debug(f"Retrieved draft body preview: {preview}")
        
        # Check for template placeholders
        if "[firstname]" in body or "{{" in body or "}}" in body:
            logger.warning("Found template placeholders in draft body - draft may not be properly personalized")
            
        return body
        
    except Exception as e:
        logger.error(f"Error extracting draft body: {str(e)}")
        return ""

def translate_label_ids_to_names(service, label_ids: list) -> list:
    """Convert Gmail label IDs to their corresponding names."""
    try:
        labels_response = service.users().labels().list(userId='me').execute()
        all_labels = labels_response.get('labels', [])
        id_to_name = {lbl["id"]: lbl["name"] for lbl in all_labels}

        label_names = []
        for lid in label_ids:
            label_names.append(id_to_name.get(lid, lid))
        return label_names
    except Exception as e:
        logger.error(f"Error translating label IDs: {str(e)}")
        return label_ids

def update_email_status(email_id: int, new_status: str, body: str = None):
    """
    Update the emails table with the new status and optionally update the body.
    Similar to update_email_record in check_for_sent_emails.py
    """
    try:
        if body:
            sql = """
                UPDATE emails
                   SET status = ?,
                       body = ?
                 WHERE email_id = ?
            """
            params = (new_status, body, email_id)
        else:
            sql = """
                UPDATE emails
                   SET status = ?
                 WHERE email_id = ?
            """
            params = (new_status, email_id)
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Email ID {email_id} updated: status='{new_status}'" + 
                       (", body updated" if body else ""))
    except Exception as e:
        logger.error(f"Error updating email ID {email_id}: {str(e)}", exc_info=True)

def delete_orphaned_draft_records():
    """
    Delete records from the emails table that have status='draft' but their
    corresponding Gmail drafts no longer exist.
    """
    try:
        # Get Gmail service
        gmail_service = get_gmail_service()
        if not gmail_service:
            logger.error("Could not initialize Gmail service.")
            return

        # Get all draft records
        drafts = get_draft_emails()
        if not drafts:
            logger.info("No draft emails found to check.")
            return

        logger.info(f"Checking {len(drafts)} draft records for orphaned entries")

        # Get list of all Gmail drafts
        try:
            gmail_drafts = gmail_service.users().drafts().list(userId='me').execute()
            existing_draft_ids = {d['id'] for d in gmail_drafts.get('drafts', [])}
        except Exception as e:
            logger.error(f"Error fetching Gmail drafts: {str(e)}")
            return

        # Track which records to delete
        orphaned_records = []
        for record in drafts:
            if not record['draft_id'] or record['draft_id'] not in existing_draft_ids:
                orphaned_records.append(record['email_id'])
                logger.debug(f"Found orphaned record: email_id={record['email_id']}, draft_id={record['draft_id']}")

        if not orphaned_records:
            logger.info("No orphaned draft records found.")
            return

        # Delete orphaned records
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(orphaned_records))
                cursor.execute(f"""
                    DELETE FROM emails
                     WHERE email_id IN ({placeholders})
                       AND status = ?
                """, (*orphaned_records, SQL_DRAFT_STATUS))
                conn.commit()
                logger.info(f"Deleted {cursor.rowcount} orphaned draft records")
        except Exception as e:
            logger.error(f"Error deleting orphaned records: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in delete_orphaned_draft_records: {str(e)}", exc_info=True)

###############################################################################
#                               MAIN PROCESS
###############################################################################

def main():
    """Check Gmail drafts for labels and update SQL status accordingly."""
    logger.info("=== Starting Gmail label check process ===")

    # First, clean up any orphaned draft records
    delete_orphaned_draft_records()

    # Get all draft emails from SQL
    pending = get_draft_emails()
    if not pending:
        logger.info("No draft emails found. Exiting.")
        return

    logger.info(f"Found {len(pending)} draft emails to process")

    # Get Gmail service
    gmail_service = get_gmail_service()
    if not gmail_service:
        logger.error("Could not initialize Gmail service.")
        return

    # Process each record
    for record in pending:
        email_id = record['email_id']
        draft_id = record['draft_id']

        logger.info(f"\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Draft ID: {draft_id}")

        if not draft_id:
            logger.warning(f"No draft_id found for email_id={email_id}. Skipping.")
            continue

        # Get the Gmail draft and its labels
        message = get_gmail_draft_by_id(gmail_service, draft_id)
        if not message:
            logger.error(f"Failed to retrieve Gmail draft for draft_id={draft_id}")
            continue

        # Get current labels
        current_labels = message.get("labelIds", [])
        if not current_labels:
            logger.info(f"No labels found for draft_id={draft_id}. Skipping.")
            continue

        # Convert label IDs to names
        label_names = translate_label_ids_to_names(gmail_service, current_labels)
        logger.debug(f"Draft {draft_id} has labels: {label_names}")

        # Check for reviewed label and update SQL status
        if REVIEWED_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} marked as '{REVIEWED_LABEL}'. Updating SQL status and body.")
            # Get the draft body text
            body = get_draft_body(message)
            if body:
                logger.debug(f"Retrieved body text ({len(body)} chars) for draft_id={draft_id}")
                # Update both status and body
                update_email_status(email_id, SQL_REVIEWED_STATUS, body)
            else:
                logger.warning(f"Could not retrieve body text for draft_id={draft_id}")
                # Update only status if no body could be retrieved
                update_email_status(email_id, SQL_REVIEWED_STATUS)
        elif TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. No action needed.")
        else:
            logger.info(f"Draft {draft_id} has no relevant labels: {label_names}")

    logger.info("\n=== Completed label check process ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()




## scripts\monitor\monitor_email_sent_status.py

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

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
                   AND status IN ('reviewed','draft','sent') 
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
                     WHERE email_id = ?
                       AND status IN ('draft','reviewed')
                """, (email_id,))
            else:
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
                     WHERE status IN ('draft','reviewed')
                  ORDER BY email_id DESC
                """)
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
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
        scheduled_dt = parse_sql_datetime(record['scheduled_send_date'])
        created_dt = parse_sql_datetime(record['created_at'])

        logger.info("\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Lead ID: {lead_id}")
        logger.info(f"Current email address: {to_address}")
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

        # Now proceed with Gmail search - without subject
        messages = search_gmail_for_messages(to_address, "", scheduled_dt)
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




## scripts\monitor\review_email_responses.py

import sys
import os
import re
# Add this at the top of the file, before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from services.data_gatherer_service import DataGathererService
from scheduling.database import get_db_connection
from utils.xai_integration import analyze_auto_reply

# Define queries for different types of notifications
BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'
AUTO_REPLY_QUERY = '(subject:"No longer employed" OR subject:"out of office" OR subject:"automatic reply") in:inbox'

# Add at the top of the file with other constants
TESTING = False  # Set to False for production

def process_invalid_email(email: str, analyzer_result: Dict) -> None:
    """
    Process an invalid email by removing the contact from HubSpot.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get the contact
        contact = hubspot.get_contact_by_email(email)
        if not contact:
            logger.info(f"Contact not found in HubSpot for email: {email}")
            return

        contact_id = contact.get('id')
        if not contact_id:
            logger.error(f"Contact found but missing ID for email: {email}")
            return

        # Archive the contact in HubSpot
        try:
            hubspot.archive_contact(contact_id)
            logger.info(f"Successfully archived contact {email} (ID: {contact_id}) due to invalid email: {analyzer_result['message']}")
        except Exception as e:
            logger.error(f"Failed to archive contact {email}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing invalid email {email}: {str(e)}")

def process_bounced_email(email: str, gmail_id: str, analyzer_result: Dict) -> None:
    """
    Process a bounced email by deleting from SQL, HubSpot, and archiving the email.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        gmail = GmailService()
        
        # First delete from SQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Delete from emails table - note the % placeholder for SQL Server
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email,))
            # Delete from leads table
            cursor.execute("DELETE FROM leads WHERE email = %s", (email,))
            conn.commit()
            logger.info(f"Deleted records for {email} from SQL database")
        except Exception as e:
            logger.error(f"Error deleting from SQL: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        # Rest of the existing HubSpot and Gmail processing...
        contact = hubspot.get_contact_by_email(email)
        if contact:
            contact_id = contact.get('id')
            if contact_id and hubspot.delete_contact(contact_id):
                logger.info(f"Contact {email} deleted from HubSpot")
        
        # Archive bounce notification
        # if gmail.archive_email(gmail_id):
        #     logger.info(f"Bounce notification archived in Gmail")
        print(f"Would have archived bounce notification with ID: {gmail_id}")

    except Exception as e:
        logger.error(f"Error processing bounced email {email}: {str(e)}")

def is_out_of_office(message: str, subject: str) -> bool:
    """
    Check if a message is an out-of-office response.
    """
    ooo_phrases = [
        "out of office",
        "automatic reply",
        "away from",
        "will be out",
        "on vacation",
        "annual leave",
        "business trip",
        "return to the office",
        "be back",
        "currently away"
    ]
    
    # Check both subject and message body for OOO indicators
    message_lower = message.lower()
    subject_lower = subject.lower()
    
    return any(phrase in message_lower or phrase in subject_lower for phrase in ooo_phrases)

def is_no_longer_employed(message: str, subject: str) -> bool:
    """Check if message indicates person is no longer employed."""
    employment_phrases = [
        "no longer employed",
        "no longer with",
        "is not employed",
        "has left",
        "no longer works",
        "no longer at",
        "no longer associated",
        "please remove",
        "has departed"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    
    return any(phrase in message_lower or phrase in subject_lower for phrase in employment_phrases)

def extract_new_contact_email(message: str) -> str:
    """Extract new contact email from message."""
    # Common patterns for replacement emails
    patterns = [
        r'please\s+(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:new\s+email|new\s+contact|instead\s+email)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'email\s+([\w\.-]+@[\w\.-]+\.\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1)
    return None

def process_employment_change(email: str, message_id: str, gmail_service: GmailService) -> None:
    """Process employment change notification."""
    try:
        message_data = gmail_service.get_message(message_id)
        subject = gmail_service._get_header(message_data, 'subject')
        body = gmail_service._get_full_body(message_data)
        
        # Use xAI to analyze the auto-reply
        contact_info = analyze_auto_reply(body, subject)
        logger.debug(f"Auto-reply analysis results: {contact_info}")
        
        if not contact_info or not contact_info['new_email']:
            logger.warning(f"No valid new contact information found for {email}")
            return

        hubspot = HubspotService(HUBSPOT_API_KEY)
        old_contact = hubspot.get_contact_by_email(email)
        
        if not old_contact:
            logger.warning(f"Original contact not found in HubSpot: {email}")
            return

        # Create new contact with combined properties
        old_properties = old_contact.get('properties', {})
        logger.debug(f"Old contact properties: {old_properties}")
        
        new_properties = {
            'email': contact_info['new_email'],
            'firstname': contact_info['new_contact'].split()[0] if contact_info['new_contact'] != 'Unknown' else old_properties.get('firstname', ''),
            'lastname': ' '.join(contact_info['new_contact'].split()[1:]) if contact_info['new_contact'] != 'Unknown' else old_properties.get('lastname', ''),
            'company': contact_info['company'] if contact_info['company'] != 'Unknown' else old_properties.get('company', ''),
            'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else old_properties.get('jobtitle', ''),
            'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else old_properties.get('phone', '')
        }
        
        logger.debug(f"Attempting to create new contact with properties: {new_properties}")

        if TESTING:
            print(f"\nAuto-reply Analysis Results:")
            print(f"Original Contact: {email}")
            print(f"Analysis Results: {contact_info}")
            print(f"\nWould have performed these actions:")
            print(f"1. Create new contact in HubSpot: {new_properties}")
            print(f"2. Delete old contact: {email}")
            print(f"3. Archive message: {message_id}")
            return

        # Create new contact
        new_contact = hubspot.create_contact(new_properties)
        if new_contact:
            logger.info(f"Created/Updated contact in HubSpot: {contact_info['new_email']}")
            
            # Copy associations before deleting old contact
            old_associations = hubspot.get_contact_associations(old_contact['id'])
            logger.debug(f"Found {len(old_associations)} associations to copy")
            
            for assoc in old_associations:
                if hubspot.create_association(new_contact['id'], assoc['id'], assoc['type']):
                    logger.debug(f"Copied association {assoc['type']} to new contact")
                else:
                    logger.warning(f"Failed to copy association {assoc['type']}")
            
            # Delete old contact
            if hubspot.delete_contact(old_contact['id']):
                logger.info(f"Deleted old contact: {email}")
            else:
                logger.error(f"Failed to delete old contact: {email}")
            
            # Archive the notification
            if gmail_service.archive_email(message_id):
                logger.info(f"Archived employment change notification")
            else:
                logger.error(f"Failed to archive notification")
        else:
            logger.error(f"Failed to create/update new contact in HubSpot")

    except Exception as e:
        logger.error(f"Error processing employment change: {str(e)}", exc_info=True)

def is_inactive_email(message: str, subject: str) -> bool:
    """Check if message indicates email is inactive."""
    inactive_phrases = [
        "email is now inactive",
        "email is inactive",
        "no longer active",
        "this address is inactive",
        "email account is inactive",
        "account is inactive",
        "inactive email",
        "email has been deactivated"
    ]
    
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in inactive_phrases)

def process_email_response(message_id: str, email: str, subject: str, body: str, gmail_service: GmailService) -> None:
    """Process an email response."""
    try:
        # Extract full name from email headers
        message_data = gmail_service.get_message(message_id)
        from_header = gmail_service._get_header(message_data, 'from')
        full_name = from_header.split('<')[0].strip()
        
        hubspot = HubspotService(HUBSPOT_API_KEY)
        logger.info(f"Processing response for email: {email}")
        
        # Get contact directly
        contact = hubspot.get_contact_by_email(email)
        
        if contact:
            contact_id = contact.get('id')
            contact_email = contact.get('properties', {}).get('email', '')
            logger.info(f"Found contact in HubSpot: {contact_email} (ID: {contact_id})")
            
            # First check if it's an auto-reply and analyze with xAI
            if "automatic reply" in subject.lower():
                logger.info("Detected auto-reply, sending to xAI for analysis...")
                contact_info = analyze_auto_reply(body, subject)
                
                if contact_info and contact_info.get('new_email'):
                    # Get existing contact properties
                    properties = contact.get('properties', {})
                    
                    # Prepare new contact properties
                    new_properties = {
                        'email': contact_info['new_email'],
                        'firstname': contact_info['new_contact'].split()[0] if contact_info['new_contact'] != 'Unknown' else properties.get('firstname', ''),
                        'lastname': ' '.join(contact_info['new_contact'].split()[1:]) if contact_info['new_contact'] != 'Unknown' else properties.get('lastname', ''),
                        'company': contact_info['company'] if contact_info['company'] != 'Unknown' else properties.get('company', ''),
                        'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else properties.get('jobtitle', ''),
                        'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else properties.get('phone', '')
                    }

                    if TESTING:
                        print("\nAuto-reply Analysis Results:")
                        print(f"Original Contact: {email}")
                        print(f"Analysis Results: {contact_info}")
                        print(f"\nWould have performed these actions:")
                        print(f"1. Create new contact in HubSpot: {new_properties}")
                        print(f"2. Copy all associations from old contact: {contact_id}")
                        print(f"3. Delete old contact: {email}")
                        print(f"4. Archive message: {message_id}")
                        return
                    
                    # Create new contact and handle transition
                    new_contact = hubspot.create_contact(new_properties)
                    if new_contact:
                        logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
                        
                        # Copy associations
                        old_associations = hubspot.get_contact_associations(contact_id)
                        for association in old_associations:
                            hubspot.create_association(new_contact['id'], association['id'], association['type'])
                        
                        # Delete old contact
                        if hubspot.delete_contact(contact_id):
                            logger.info(f"Deleted old contact: {email}")
                        
                        # Archive the notification
                        if gmail_service.archive_email(message_id):
                            logger.info(f"Archived employment change notification")
                else:
                    logger.info("No new contact information found, processing as standard auto-reply")
                    notification = {
                        "bounced_email": contact_email,
                        "message_id": message_id
                    }
                    process_bounce_notification(notification, gmail_service)
            
            # Other conditions remain unchanged
            elif "no longer employed" in subject.lower():
                process_employment_change(contact_email, message_id, gmail_service)
            elif is_inactive_email(body, subject):
                notification = {
                    "bounced_email": contact_email,
                    "message_id": message_id
                }
                process_bounce_notification(notification, gmail_service)
                
            logger.info(f"Processed response for contact: {contact_email}")
        else:
            logger.warning(f"No contact found in HubSpot for email: {email}")
            
    except Exception as e:
        logger.error(f"Error processing email response: {str(e)}")

def delete_email_from_database(email_address):
    """Helper function to delete email records from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        delete_query = "DELETE FROM dbo.emails WHERE email_address = ?"
        cursor.execute(delete_query, (email_address,))
        conn.commit()
        logger.info(f"Deleted all records for {email_address} from emails table")
    except Exception as e:
        logger.error(f"Error deleting from SQL: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def process_bounce_notification(notification, gmail_service):
    """Process a single bounce notification"""
    bounced_email = notification['bounced_email']
    message_id = notification['message_id']
    logger.info(f"Processing bounce notification - Email: {bounced_email}, Message ID: {message_id}")
    
    # Delete from database
    if TESTING:
        print(f"Would have deleted {bounced_email} from SQL database")
    else:
        delete_email_from_database(bounced_email)
    
    try:
        # Delete from HubSpot
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        contact = hubspot.get_contact_by_email(bounced_email)
        
        if contact:
            contact_id = contact.get('id')
            if contact_id:
                logger.info(f"Attempting to delete contact {bounced_email} from HubSpot")
                if TESTING:
                    print(f"Would have deleted contact {bounced_email} with ID {contact_id}")
                else:
                    if hubspot.delete_contact(contact_id):
                        logger.info(f"Successfully deleted contact {bounced_email} from HubSpot")
                    else:
                        logger.error(f"Failed to delete contact {bounced_email} from HubSpot")
            else:
                logger.warning(f"Contact found but no ID for {bounced_email}")
        else:
            logger.warning(f"No contact found in HubSpot for {bounced_email}")
        
        # Archive the Gmail message
        if TESTING:
            print(f"Would have archived bounce notification with ID: {message_id}")
        else:
            if gmail_service.archive_email(message_id):
                logger.info(f"Successfully archived bounce notification for {bounced_email}")
            else:
                logger.error(f"Failed to archive Gmail message {message_id}")
            
    except Exception as e:
        logger.error(f"Error processing bounce notification for {bounced_email}: {str(e)}")
        raise

def verify_processing(email: str, gmail_service: GmailService, hubspot_service: HubspotService) -> bool:
    """Verify that the contact was properly deleted and emails archived."""
    if TESTING:
        print(f"TESTING: Skipping verification for {email}")
        return True
        
    success = True
    
    # Check if contact still exists in HubSpot
    contact = hubspot_service.get_contact_by_email(email)
    if contact:
        logger.error(f"❌ Verification failed: Contact {email} still exists in HubSpot")
        success = False
    else:
        logger.info(f"✅ Verification passed: Contact {email} successfully deleted from HubSpot")
    
    # Check for any remaining emails in inbox
    query = f"from:{email} in:inbox"
    remaining_emails = gmail_service.search_messages(query)
    if remaining_emails:
        logger.error(f"❌ Verification failed: Found {len(remaining_emails)} unarchived emails from {email}")
        success = False
    else:
        logger.info(f"✅ Verification passed: No remaining emails from {email} in inbox")
    
    return success

def process_bounce_notifications(target_email: str = None):
    """
    Main function to process all bounce notifications and auto-replies.
    
    Args:
        target_email (str, optional): If provided, only process this specific email.
    """
    logger.info(f"Starting bounce notification and auto-reply processing...{' for ' + target_email if target_email else ''}")
    
    gmail_service = GmailService()
    hubspot_service = HubspotService(HUBSPOT_API_KEY)
    processed_emails = set()  # Track processed emails for verification
    
    # Process bounce notifications
    logger.info("Searching for bounce notifications in inbox...")
    bounce_notifications = gmail_service.get_all_bounce_notifications(inbox_only=True)
    
    if bounce_notifications:
        logger.info(f"Found {len(bounce_notifications)} bounce notifications to process.")
        for notification in bounce_notifications:
            email = notification.get('bounced_email')
            if email and (not target_email or email == target_email):
                process_bounce_notification(notification, gmail_service)
                processed_emails.add(email)
    else:
        logger.info("No bounce notifications found in inbox.")
    
    # Process auto-replies
    logger.info("Searching for auto-reply notifications...")
    auto_replies = gmail_service.search_messages(AUTO_REPLY_QUERY)
    
    if auto_replies:
        logger.info(f"Found {len(auto_replies)} auto-reply notifications.")
        for message in auto_replies:
            message_data = gmail_service.get_message(message['id'])
            if message_data:
                from_header = gmail_service._get_header(message_data, 'from')
                subject = gmail_service._get_header(message_data, 'subject')
                body = gmail_service._get_full_body(message_data)
                
                email_match = re.search(r'<(.+?)>', from_header)
                email = email_match.group(1) if email_match else from_header.split()[-1]
                
                # Skip if target_email is specified and doesn't match
                if target_email and email != target_email:
                    continue
                
                logger.info(f"Processing auto-reply from: {from_header}, Subject: {subject}")
                
                # Check for inactive email in the body
                if is_inactive_email(body, subject):
                    logger.info(f"Detected inactive email notification for: {email}")
                
                process_email_response(message['id'], email, subject, body, gmail_service)
                processed_emails.add(email)
    else:
        logger.info("No auto-reply notifications found.")
    
    # Verify processing for all processed emails
    if processed_emails:
        logger.info("Verifying processing results...")
        all_verified = True
        for email in processed_emails:
            if not verify_processing(email, gmail_service, hubspot_service):
                all_verified = False
                logger.error(f"❌ Processing verification failed for {email}")
        
        if all_verified:
            logger.info("✅ All processing verified successfully")
        else:
            logger.error("❌ Some processing verifications failed - check logs for details")
    else:
        logger.info("No emails were processed")

if __name__ == "__main__":
    # For testing specific email
    TARGET_EMAIL = "cmccarthy@mountainbranch.com"
    if TESTING:
        print(f"\nRunning in TEST mode - no actual changes will be made\n")
    process_bounce_notifications(TARGET_EMAIL)
    
    # For processing all emails, comment out TARGET_EMAIL and use:
    # process_bounce_notifications()




## scripts\monitor\run_email_monitoring.py

#!/usr/bin/env python3

"""
Email Monitoring System
======================

This script orchestrates the email monitoring process by coordinating two main components:
1. Review Status Monitor (monitor_email_review_status.py)
   - Tracks Gmail drafts for review labels
   - Updates database when drafts are approved

2. Sent Status Monitor (monitor_email_sent_status.py)
   - Verifies when emails are actually sent
   - Updates database with send confirmation and details

Email Status Flow
----------------
1. draft → reviewed → sent
   - draft: Initial state when email is created
   - reviewed: Email has been approved via Gmail labels
   - sent: Email has been sent and confirmed via Gmail API

Configuration
------------
The script supports the following environment variables:
- EMAIL_MONITOR_INTERVAL: Seconds between monitoring runs (default: 300)
- EMAIL_MONITOR_MAX_RETRIES: Max retry attempts per run (default: 3)
- EMAIL_MONITOR_RETRY_DELAY: Seconds between retries (default: 60)
"""

import os
import sys
import time
from datetime import datetime
import pytz
from typing import Optional, Tuple, Dict
import traceback

# Local imports
from utils.logging_setup import logger
import monitor_email_review_status
import monitor_email_sent_status

###############################################################################
#                           CONFIGURATION
###############################################################################

class MonitorConfig:
    """Configuration settings for the monitoring process."""
    
    def __init__(self):
        # Process interval settings
        self.monitor_interval = int(os.getenv('EMAIL_MONITOR_INTERVAL', 300))  # 5 minutes
        self.max_retries = int(os.getenv('EMAIL_MONITOR_MAX_RETRIES', 3))
        self.retry_delay = int(os.getenv('EMAIL_MONITOR_RETRY_DELAY', 60))
        
        # Monitoring flags
        self.check_reviews = True
        self.check_sent = True
        
    def __str__(self) -> str:
        """Return string representation of config."""
        return (
            f"MonitorConfig("
            f"interval={self.monitor_interval}s, "
            f"max_retries={self.max_retries}, "
            f"retry_delay={self.retry_delay}s, "
            f"check_reviews={self.check_reviews}, "
            f"check_sent={self.check_sent})"
        )

###############################################################################
#                           MONITORING FUNCTIONS
###############################################################################

def run_review_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the review status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_review_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_sent_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the sent status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_sent_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_monitoring_cycle(config: MonitorConfig) -> Dict[str, bool]:
    """
    Run a complete monitoring cycle with both checks.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        Dict[str, bool]: Status of each check
    """
    results = {
        'review_check': False,
        'sent_check': False
    }
    
    start_time = datetime.now(pytz.UTC)
    logger.info(f"=== Starting Monitoring Cycle at {start_time} ===")
    
    # Review Status Check
    if config.check_reviews:
        for attempt in range(config.max_retries):
            success, error = run_review_check()
            if success:
                results['review_check'] = True
                break
            else:
                logger.error(f"Review check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    # Brief pause between checks
    time.sleep(1)
    
    # Sent Status Check
    if config.check_sent:
        for attempt in range(config.max_retries):
            success, error = run_sent_check()
            if success:
                results['sent_check'] = True
                break
            else:
                logger.error(f"Sent check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    end_time = datetime.now(pytz.UTC)
    duration = end_time - start_time
    
    logger.info("\n=== Monitoring Cycle Complete ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Results: {results}")
    
    return results

###############################################################################
#                               MAIN PROCESS
###############################################################################

def run_continuous_monitoring(config: MonitorConfig):
    """
    Run the monitoring process continuously with the specified interval.
    
    Args:
        config: MonitorConfig instance with settings
    """
    logger.info(f"Starting continuous monitoring with config: {config}")
    
    while True:
        try:
            run_monitoring_cycle(config)
            logger.info(f"Waiting {config.monitor_interval} seconds until next cycle...")
            time.sleep(config.monitor_interval)
            
        except KeyboardInterrupt:
            logger.info("\nMonitoring interrupted by user")
            break
            
        except Exception as e:
            logger.error("Unexpected error in monitoring cycle:")
            logger.error(traceback.format_exc())
            logger.info(f"Retrying in {config.monitor_interval} seconds...")
            time.sleep(config.monitor_interval)

def run_single_cycle(config: MonitorConfig) -> bool:
    """
    Run a single monitoring cycle.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        bool: True if all enabled checks succeeded
    """
    results = run_monitoring_cycle(config)
    return all(results.values())

def main():
    """
    Main entry point for the email monitoring system.
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = MonitorConfig()
        
        # Check for command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run single cycle
            success = run_single_cycle(config)
            return 0 if success else 1
        else:
            # Run continuous monitoring
            run_continuous_monitoring(config)
            return 0
            
    except Exception as e:
        logger.error(f"Fatal error in monitoring system: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 



## services\gmail_service.py

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



## services\hubspot_service.py

"""
HubSpot service for API operations.
"""
from typing import Dict, List, Optional, Any
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from config.settings import logger
from utils.exceptions import HubSpotError
from utils.formatting_utils import clean_html
from datetime import datetime


class HubspotService:
    """Service class for HubSpot API operations."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.hubapi.com"):
        """Initialize HubSpot service with API credentials."""
        logger.debug("Initializing HubspotService")
        if not api_key:
            logger.error("No API key provided to HubspotService")
            raise ValueError("HubSpot API key is required")
            
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.debug(f"HubspotService initialized with base_url: {self.base_url}")
        
        # Endpoints
        self.contacts_endpoint = f"{self.base_url}/crm/v3/objects/contacts"
        self.companies_endpoint = f"{self.base_url}/crm/v3/objects/companies"
        self.notes_search_url = f"{self.base_url}/crm/v3/objects/notes/search"
        self.tasks_endpoint = f"{self.base_url}/crm/v3/objects/tasks"
        self.emails_search_url = f"{self.base_url}/crm/v3/objects/emails/search"

        # Property mappings
        self.hubspot_property_mapping = {
            "name": "name",
            "company_short_name": "company_short_name",
            "club_type": "club_type",
            "facility_complexity": "facility_complexity",
            "geographic_seasonality": "geographic_seasonality",
            "has_pool": "has_pool",
            "has_tennis_courts": "has_tennis_courts",
            "number_of_holes": "number_of_holes",
            "public_private_flag": "public_private_flag",
            "club_info": "club_info",
            "start_month": "start_month",
            "end_month": "end_month",
            "peak_season_start_month": "peak_season_start_month",
            "peak_season_end_month": "peak_season_end_month",
            "competitor": "competitor",
            "domain": "domain",
            "notes_last_contacted": "notes_last_contacted",
            "num_contacted_notes": "num_contacted_notes",
            "num_associated_contacts": "num_associated_contacts"
        }

        self.property_value_mapping = {
            "club_type": {
                "Private": "Private",
                "Private Course": "Private",
                "Country Club": "Country Club",
                "Public": "Public",
                "Public - Low Daily Fee": "Public - Low Daily Fee",
                "Municipal": "Municipal",
                "Semi-Private": "Semi-Private",
                "Resort": "Resort",
                "Management Company": "Management Company",
                "Unknown": "Unknown"
            },
            "facility_complexity": {
                "Single-Course": "Standard",
                "Multi-Course": "Multi-Course",
                "Resort": "Resort",
                "Unknown": "Unknown"
            },
            "geographic_seasonality": {
                "Year-Round": "Year-Round Golf",
                "Peak Summer Season": "Peak Summer Season",
                "Short Summer Season": "Short Summer Season",
                "Unknown": "Unknown"
            },
            "competitor": {
                "Club Essentials": "Club Essentials",
                "Jonas": "Jonas",
                "Northstar": "Northstar",
                "Unknown": "Unknown"
            }
        }

    def search_country_clubs(self, batch_size: int = 25) -> List[Dict[str, Any]]:
        """Search for Country Club type companies in HubSpot."""
        url = f"{self.companies_endpoint}/search"
        all_results = []
        after = None
        
        while True:
            payload = {
                "limit": batch_size,
                "properties": [
                    "name", "company_short_name", "city", "state",
                    "club_type", "facility_complexity", "geographic_seasonality",
                    "has_pool", "has_tennis_courts", "number_of_holes",
                    "public_private_flag", "club_info",
                    "peak_season_start_month", "peak_season_end_month",
                    "start_month", "end_month", "domain",
                    "notes_last_contacted", "num_contacted_notes",
                    "num_associated_contacts"
                ],
                "filterGroups": [{
                    "filters": [{
                        "propertyName": "club_type",
                        "operator": "EQ",
                        "value": "Country Club"
                    }]
                }]
            }
            
            if after:
                payload["after"] = after
                
            try:
                response = self._make_hubspot_post(url, payload)
                results = response.get("results", [])
                all_results.extend(results)
                
                paging = response.get("paging", {})
                next_link = paging.get("next", {}).get("after")
                if not next_link:
                    break
                after = next_link
                
            except Exception as e:
                logger.error(f"Error fetching Country Clubs: {str(e)}")
                break
                
        return all_results

    def update_company_properties(self, company_id: str, properties: Dict[str, Any]) -> bool:
        """Update company properties in HubSpot."""
        logger.debug(f"Starting update_company_properties for company_id: {company_id}")
        logger.debug(f"Input properties: {properties}")
        
        try:
            mapped_updates = {}
            
            # Map and transform properties
            for internal_key, value in properties.items():
                logger.debug(f"Processing property - Key: {internal_key}, Value: {value}")
                
                if value is None or value == "":
                    logger.debug(f"Skipping empty value for key: {internal_key}")
                    continue

                hubspot_key = self.hubspot_property_mapping.get(internal_key)
                if not hubspot_key:
                    logger.warning(f"No HubSpot mapping for property: {internal_key}")
                    continue

                logger.debug(f"Pre-transform - Key: {internal_key}, Value: {value}, Type: {type(value)}")

                try:
                    # Apply enum value transformations
                    if internal_key in self.property_value_mapping:
                        original_value = value
                        value = self.property_value_mapping[internal_key].get(str(value), value)
                        logger.debug(f"Enum transformation for {internal_key}: {original_value} -> {value}")

                    # Type-specific handling
                    if internal_key in ["number_of_holes", "start_month", "end_month", 
                                      "peak_season_start_month", "peak_season_end_month",
                                      "notes_last_contacted", "num_contacted_notes",
                                      "num_associated_contacts"]:
                        logger.debug(f"Converting numeric value for {internal_key}: {value}")
                        value = int(value) if str(value).isdigit() else 0
                    elif internal_key in ["has_pool", "has_tennis_courts"]:
                        logger.debug(f"Converting boolean value for {internal_key}: {value}")
                        value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                    elif internal_key == "club_info":
                        logger.debug(f"Truncating club_info from length {len(str(value))}")
                        value = str(value)[:5000]
                    elif internal_key == "company_short_name":
                        logger.debug(f"Processing company_short_name: {value}")
                        value = str(value)[:100]

                except Exception as e:
                    logger.error(f"Error transforming {internal_key}: {str(e)}", exc_info=True)
                    continue

                mapped_updates[hubspot_key] = value

            # Debug logging
            logger.debug("Final HubSpot payload:")
            logger.debug(f"Company ID: {company_id}")
            logger.debug("Properties:")
            for key, value in mapped_updates.items():
                logger.debug(f"  {key}: {value} (Type: {type(value)})")

            url = f"{self.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            logger.info(f"Making PATCH request to HubSpot - URL: {url}")
            logger.debug(f"Request payload: {payload}")
            
            response = self._make_hubspot_patch(url, payload)
            success = bool(response)
            logger.info(f"HubSpot update {'successful' if success else 'failed'} for company {company_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating company {company_id}: {str(e)}", exc_info=True)
            return False

    def get_contact_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find a contact by email address."""
        url = f"{self.contacts_endpoint}/search"
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": email
                        }
                    ]
                }
            ],
            "properties": ["email", "firstname", "lastname", "company", "jobtitle"]
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            return results[0] if results else None
        except Exception as e:
            raise HubSpotError(f"Error searching for contact by email {email}: {str(e)}")

    def get_contact_properties(self, contact_id: str) -> dict:
        """Get properties for a contact."""
        props = [
            "email", "jobtitle", "lifecyclestage", "phone",
            "hs_sales_email_last_replied", "firstname", "lastname"
        ]
        query_params = "&".join([f"properties={p}" for p in props])
        url = f"{self.contacts_endpoint}/{contact_id}?{query_params}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {})
        except Exception as e:
            raise HubSpotError(f"Error fetching contact properties: {str(e)}")

    def get_all_emails_for_contact(self, contact_id: str) -> list:
        """Fetch all Email objects for a contact."""
        all_emails = []
        after = None
        has_more = True
        
        while has_more:
            payload = {
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "associations.contact",
                                "operator": "EQ",
                                "value": contact_id
                            }
                        ]
                    }
                ],
                "properties": ["hs_email_subject", "hs_email_text", "hs_email_direction",
                               "hs_email_status", "hs_timestamp"]
            }
            if after:
                payload["after"] = after

            try:
                response = requests.post(self.emails_search_url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                for item in results:
                    all_emails.append({
                        "id": item.get("id"),
                        "subject": item["properties"].get("hs_email_subject", ""),
                        "body_text": item["properties"].get("hs_email_text", ""),
                        "direction": item["properties"].get("hs_email_direction", ""),
                        "status": item["properties"].get("hs_email_status", ""),
                        "timestamp": item["properties"].get("hs_timestamp", "")
                    })
                
                paging = data.get("paging", {}).get("next")
                if paging and paging.get("after"):
                    after = paging["after"]
                else:
                    has_more = False
            
            except Exception as e:
                raise HubSpotError(f"Error fetching emails for contact {contact_id}: {e}")

        return all_emails

    def get_all_notes_for_contact(self, contact_id: str) -> list:
        """Get all notes for a contact."""
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "associations.contact",
                            "operator": "EQ",
                            "value": contact_id
                        }
                    ]
                }
            ],
            "properties": ["hs_note_body", "hs_timestamp", "hs_lastmodifieddate"]
        }

        try:
            response = requests.post(self.notes_search_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            return [{
                "id": note.get("id"),
                "body": note["properties"].get("hs_note_body", ""),
                "timestamp": note["properties"].get("hs_timestamp", ""),
                "last_modified": note["properties"].get("hs_lastmodifieddate", "")
            } for note in results]
        except Exception as e:
            raise HubSpotError(f"Error fetching notes for contact {contact_id}: {str(e)}")

    def get_associated_company_id(self, contact_id: str) -> Optional[str]:
        """Get the associated company ID for a contact."""
        url = f"{self.contacts_endpoint}/{contact_id}/associations/company"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            return results[0].get("id") if results else None
        except Exception as e: 
            raise HubSpotError(f"Error fetching associated company ID: {str(e)}")

    def get_company_data(self, company_id: str) -> dict:
        """
        Get company data, including the 15 fields required:
        name, city, state, annualrevenue, createdate, hs_lastmodifieddate,
        hs_object_id, club_type, facility_complexity, has_pool,
        has_tennis_courts, number_of_holes, geographic_seasonality,
        public_private_flag, club_info, peak_season_start_month,
        peak_season_end_month, start_month, end_month, notes_last_contacted,
        num_contacted_notes, num_associated_contacts.
        """
        if not company_id:
            return {}
            
        url = (
            f"{self.companies_endpoint}/{company_id}?"
            "properties=name"
            "&properties=company_short_name"
            "&properties=city"
            "&properties=state"
            "&properties=annualrevenue"
            "&properties=createdate"
            "&properties=hs_lastmodifieddate"
            "&properties=hs_object_id"
            "&properties=club_type"
            "&properties=facility_complexity"
            "&properties=has_pool"
            "&properties=has_tennis_courts"
            "&properties=number_of_holes"
            "&properties=geographic_seasonality"
            "&properties=public_private_flag"
            "&properties=club_info"
            "&properties=peak_season_start_month"
            "&properties=peak_season_end_month"
            "&properties=start_month"
            "&properties=end_month"
            "&properties=notes_last_contacted"
            "&properties=num_contacted_notes"
            "&properties=num_associated_contacts"
        )
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {})
        except Exception as e:
            raise HubSpotError(f"Error fetching company data: {str(e)}")

    def gather_lead_data(self, email: str) -> Dict[str, Any]:
        """
        Gather all lead data sequentially.
        """
        # 1. Get contact ID
        contact = self.get_contact_by_email(email)
        if not contact:
            raise HubSpotError(f"No contact found for email: {email}")
        
        contact_id = contact.get('id')
        if not contact_id:
            raise HubSpotError(f"Contact found but missing ID for email: {email}")

        # 2. Fetch data points sequentially
        contact_props = self.get_contact_properties(contact_id)
        emails = self.get_all_emails_for_contact(contact_id)
        notes = self.get_all_notes_for_contact(contact_id)
        company_id = self.get_associated_company_id(contact_id)

        # 3. Get company data if available
        company_data = self.get_company_data(company_id) if company_id else {}

        # 4. Combine all data
        return {
            "id": contact_id,
            "properties": contact_props,
            "emails": emails,
            "notes": notes,
            "company_data": company_data
        }

    def get_random_contacts(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get a random sample of contact email addresses from HubSpot.
        
        Args:
            count: Number of random contacts to retrieve (default: 3)
            
        Returns:
            List of dicts containing contact info (email, name, etc.)
        """
        try:
            # First, get total count of contacts
            url = f"{self.contacts_endpoint}/search"
            payload = {
                "filterGroups": [],  # No filters to get all contacts
                "properties": ["email", "firstname", "lastname", "company"],
                "limit": 1,  # Just need count
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            total = response.json().get("total", 0)
            
            if total == 0:
                logger.warning("No contacts found in HubSpot")
                return []
            
            # Generate random offset to get different contacts each time
            import random
            random_offset = random.randint(0, max(0, total - count * 2))
            
            # Get a batch starting from random offset
            batch_size = min(count * 2, total)  # Get 2x needed to ensure enough valid contacts
            payload.update({
                "limit": batch_size,
                "after": str(random_offset)  # Add random offset
            })
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            contacts = response.json().get("results", [])
            
            # Randomly sample from the batch
            selected = random.sample(contacts, min(count, len(contacts)))
            
            # Format the results
            results = []
            for contact in selected:
                props = contact.get("properties", {})
                results.append({
                    "id": contact.get("id"),
                    "email": props.get("email"),
                    "first_name": props.get("firstname"),
                    "last_name": props.get("lastname"),
                    "company": props.get("company")
                })
            
            logger.debug(f"Retrieved {len(results)} random contacts from HubSpot (offset: {random_offset})")
            return results
            
        except Exception as e:
            logger.error(f"Error getting random contacts: {str(e)}")
            return []

    def _make_hubspot_post(self, url: str, payload: dict) -> dict:
        """
        Make a POST request to HubSpot API with retries.
        
        Args:
            url: The endpoint URL
            payload: The request payload
            
        Returns:
            dict: The JSON response from HubSpot
        """
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HubSpot API error: {str(e)}")
            raise HubSpotError(f"Failed to make HubSpot POST request: {str(e)}")
            
    def _make_hubspot_get(self, url: str, params: Dict = None) -> Dict[str, Any]:
        """Make a GET request to HubSpot API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _make_hubspot_patch(self, url: str, payload: Dict) -> Any:
        """Make a PATCH request to HubSpot API."""
        try:
            logger.debug(f"Making PATCH request to: {url}")
            logger.debug(f"Headers: {self.headers}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.patch(url, headers=self.headers, json=payload)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response body: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            if hasattr(e.response, 'text'):
                logger.error(f"Response body: {e.response.text}")
            raise HubSpotError(f"PATCH request failed: {str(e)}")

    def get_company_by_id(self, company_id: str, properties: List[str]) -> Dict[str, Any]:
        """Get company by ID with specified properties."""
        try:
            url = f"{self.companies_endpoint}/{company_id}"
            params = {
                "properties": properties
            }
            response = self._make_hubspot_get(url, params=params)
            return response
        except Exception as e:
            logger.error(f"Error getting company {company_id}: {e}")
            return {}

    def get_contacts_from_list(self, list_id: str) -> List[Dict[str, Any]]:
        """Get all contacts from a specified HubSpot list."""
        url = f"{self.base_url}/contacts/v1/lists/{list_id}/contacts/all"
        all_contacts = []
        vidOffset = 0
        has_more = True
        
        while has_more:
            try:
                params = {
                    "count": 100,
                    "vidOffset": vidOffset
                }
                response = self._make_hubspot_get(url, params)
                
                contacts = response.get("contacts", [])
                all_contacts.extend(contacts)
                
                has_more = response.get("has-more", False)
                vidOffset = response.get("vid-offset", 0)
                
            except Exception as e:
                logger.error(f"Error fetching contacts from list: {str(e)}")
                break
        
        return all_contacts

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully deleted contact {contact_id} from HubSpot")
            return True
        except Exception as e:
            logger.error(f"Error deleting contact {contact_id}: {str(e)}")
            return False

    def mark_contact_as_bounced(self, email: str) -> bool:
        """Mark a contact as bounced in HubSpot."""
        try:
            # Add detailed logging
            logger.info(f"Marking contact as bounced in HubSpot: {email}")
            
            # Get the contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"Contact not found in HubSpot: {email}")
                return False
            
            # Update the contact properties
            properties = {
                "email_bounced": "true",
                "email_bounced_date": datetime.now().strftime("%Y-%m-%d"),
                "email_bounced_reason": "Hard bounce - Invalid recipient"
            }
            
            # Make the API call
            success = self.update_contact(contact['id'], properties)
            
            if success:
                logger.info(f"Successfully marked contact as bounced in HubSpot: {email}")
            else:
                logger.error(f"Failed to mark contact as bounced in HubSpot: {email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking contact as bounced in HubSpot: {str(e)}")
            return False

    def create_contact(self, properties: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Create a new contact in HubSpot.
        
        Args:
            properties (Dict[str, str]): Dictionary of contact properties
                Required keys: email
                Optional keys: firstname, lastname, company, jobtitle, phone
                
        Returns:
            Optional[Dict[str, Any]]: The created contact data or None if creation fails
        """
        try:
            email = properties.get('email')
            logger.info(f"Creating new contact in HubSpot with email: {email}")
            logger.debug(f"Contact properties: {properties}")
            
            # Validate required email property
            if not email:
                logger.error("Cannot create contact: email is required")
                return None
            
            # Check if contact already exists
            existing_contact = self.get_contact_by_email(email)
            if existing_contact:
                logger.warning(f"Contact already exists with email {email}. Updating instead.")
                # Update existing contact with new properties
                if self.update_contact(existing_contact['id'], properties):
                    logger.info(f"Successfully updated existing contact: {existing_contact['id']}")
                    return existing_contact
                return None
            
            payload = {
                "properties": {
                    key: str(value) for key, value in properties.items() if value and value != 'Unknown'
                }
            }
            
            logger.debug(f"Making create contact request with payload: {payload}")
            response = requests.post(
                self.contacts_endpoint,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 409:
                logger.warning(f"Conflict creating contact {email} - contact may already exist")
                # Try to get the existing contact again (in case it was just created)
                existing_contact = self.get_contact_by_email(email)
                if existing_contact:
                    logger.info(f"Found existing contact after conflict: {existing_contact['id']}")
                    if self.update_contact(existing_contact['id'], properties):
                        return existing_contact
                return None
            
            response.raise_for_status()
            contact_data = response.json()
            logger.info(f"Successfully created new contact: {contact_data.get('id')}")
            logger.debug(f"New contact data: {contact_data}")
            return contact_data
            
        except Exception as e:
            logger.error(f"Error creating contact in HubSpot: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    def get_contact_associations(self, contact_id: str) -> List[Dict[str, Any]]:
        """Get all associations for a contact."""
        url = f"{self.contacts_endpoint}/{contact_id}/associations"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching contact associations: {str(e)}")
            return []

    def create_association(self, from_id: str, to_id: str, association_type: str) -> bool:
        """Create an association between two objects."""
        url = f"{self.contacts_endpoint}/{from_id}/associations/{association_type}/{to_id}"
        
        try:
            response = requests.put(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully created association between {from_id} and {to_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating association: {str(e)}")
            return False

    def update_contact(self, contact_id: str, properties: Dict[str, Any]) -> bool:
        """Update contact properties in HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            payload = {"properties": properties}
            
            response = requests.patch(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating contact {contact_id}: {str(e)}")
            return False




## utils\xai_integration.py

# utils/xai_integration.py

import os
import re
import json
import time
import random
import requests

from typing import Tuple, Dict, Any, List
from datetime import datetime, date
from dotenv import load_dotenv

from utils.logging_setup import logger
from config.settings import DEBUG_MODE

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.2"))

# Simple caches to avoid repeated calls
_cache = {
    "news": {},
    "club_segmentation": {},
    "icebreakers": {},
}





def get_email_rules() -> List[str]:
    """
    Returns the standardized list of rules for email personalization.
    """
    return [
        "# IMPORTANT: FOLLOW THESE RULES:\n",
        f"**Time Context:** Use relative date terms compared to Today's date of {date.today().strftime('%B %d, %Y')}.",
        "**Tone:** Professional but conversational, focusing on starting a dialogue.",
        "**Closing:** End emails directly after your call-to-action.",
        "**Previous Contact:** If no prior replies, do not reference previous emails or special offers.",
        "**Signature:** DO NOT include a signature block - this will be added later.",
    ]


def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends a request to the xAI API with retry logic.
    Logs request/response details for debugging.
    """
    TIMEOUT = 30
    logger.debug(
        "Full xAI Request Payload:",
        extra={
            "extra_data": {
                "request_details": {
                    "model": payload.get("model", MODEL_NAME),
                    "temperature": payload.get("temperature", EMAIL_TEMPERATURE),
                    "max_tokens": payload.get("max_tokens", 1000),
                    "messages": [
                        {"role": msg.get("role"), "content": msg.get("content")}
                        for msg in payload.get("messages", [])
                    ],
                }
            }
        },
    )

    for attempt in range(max_retries):
        try:
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN,
                },
                json=payload,
                timeout=TIMEOUT,
            )

            logger.debug(
                "Full xAI Response:",
                extra={
                    "extra_data": {
                        "response_details": {
                            "status_code": response.status_code,
                            "response_body": json.loads(response.text)
                            if response.text
                            else None,
                            "attempt": attempt + 1,
                            "headers": dict(response.headers),
                        }
                    }
                },
            )

            if response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time}s before retry"
                )
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(
                    f"xAI API error ({response.status_code}): {response.text}"
                )
                return ""

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                logger.debug(
                    "Received xAI response:\n%s",
                    content[:200] + "..." if len(content) > 200 else content,
                )
                return content

            except (KeyError, json.JSONDecodeError) as e:
                logger.error(
                    "Error parsing xAI response",
                    extra={
                        "error": str(e),
                        "response_text": response.text[:500],
                    },
                )
                return ""

        except Exception as e:
            logger.error(
                "xAI request failed",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "payload": payload,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return ""


##############################################################################
# News Search + Icebreaker
##############################################################################
def xai_news_search(club_name: str) -> tuple[str, str]:
    """
    Checks if a club is in the news and returns both news and icebreaker.
    Returns: Tuple of (news_summary, icebreaker)
    """
    if not club_name.strip():
        return "", ""

    if club_name in _cache["news"]:
        if DEBUG_MODE:
            logger.debug(f"Using cached news result for {club_name}")
        news = _cache["news"][club_name]
        icebreaker = _build_icebreaker_from_news(club_name, news)
        return news, icebreaker

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise summaries of recent club news.",
            },
            {
                "role": "user",
                "content": (
                    f"Tell me about any recent news for {club_name}. "
                    "If none exists, respond with 'has not been in the news.'"
                ),
            },
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0,
    }

    logger.info(f"Searching news for club: {club_name}")
    news = _send_xai_request(payload)
    logger.debug(f"News search result for {club_name}:")

    _cache["news"][club_name] = news

    if news:
        if news.startswith("Has ") and " has not been in the news" in news:
            news = news.replace("Has ", "")
        news = news.replace(" has has ", " has ")

    icebreaker = _build_icebreaker_from_news(club_name, news)
    return news, icebreaker


def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker if news is available.
    Returns an empty string if no relevant news found.
    """
    if (
        not club_name.strip()
        or not news_summary.strip()
        or "has not been in the news" in news_summary.lower()
    ):
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing from Swoop Golf's perspective, reaching out to golf clubs. "
                    "Create brief, natural-sounding icebreakers based on recent club news. "
                    "Keep the tone professional and focused on business value."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, natural-sounding icebreaker about {club_name} "
                    f"based on this news: {news_summary}\n\n"
                    "Requirements:\n"
                    "1. Single sentence only\n"
                    "2. Focus on business impact\n"
                    "3. No generic statements\n"
                    "4. Must relate to the news provided"
                ),
            },
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1,
    }

    logger.info(f"Building icebreaker for club: {club_name}")
    icebreaker = _send_xai_request(payload)
    logger.debug(f"Generated icebreaker for {club_name}:")

    return icebreaker


##############################################################################
# Personalize Email
##############################################################################
def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = "",
    context: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Personalizes email content using xAI.
    Returns a dictionary with 'subject' and 'body' keys.
    """
    try:
        # Ensure lead_sheet is a dictionary
        if not isinstance(lead_sheet, dict):
            logger.warning(f"Invalid lead_sheet type: {type(lead_sheet)}. Using empty dict.")
            lead_sheet = {}

        # Create a filtered company_data with only specific fields
        company_data = lead_sheet.get("company_data", {})
        allowed_fields = ['name', 'city', 'state', 'has_pool']
        filtered_company_data = {
            k: v for k, v in company_data.items() 
            if k in allowed_fields
        }
        
        # Update lead_sheet with filtered company data
        filtered_lead_sheet = {
            "lead_data": lead_sheet.get("lead_data", {}),
            "company_data": filtered_company_data,
            "analysis": lead_sheet.get("analysis", {})
        }
        
        # Use filtered_lead_sheet in the rest of the function
        previous_interactions = filtered_lead_sheet.get("analysis", {}).get("previous_interactions", {})
        has_prior_emails = bool(lead_sheet.get("lead_data", {}).get("emails", []))
        logger.debug(f"Has the lead previously emailed us? {has_prior_emails}")

        objection_handling = ""
        if has_prior_emails:
            with open("docs/templates/objection_handling.txt", "r") as f:
                objection_handling = f.read()
            logger.debug("Objection handling content loaded")
        else:
            logger.debug("Objection handling content not loaded (lead has not emailed us)")

        system_message = (
            "You are a helpful assistant that personalizes outreach emails for golf clubs, focusing on business value and relevant solutions. "
            "IMPORTANT: Do not include any signature block - this will be added later."
        )

        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_sheet.get("company_data", {})
        
        # Use provided context if available, otherwise build it
        if context is None:
            context_block = build_context_block(
                interaction_history=summary if summary else "No previous interactions",
                objection_handling=objection_handling if has_prior_emails else "",
                original_email={"subject": subject, "body": body},
                company_data=filtered_company_data
            )
        else:
            # Add filtered company data to existing context
            context.update({"company_data": filtered_company_data})
            context_block = context
            
        logger.debug(f"Context block: {json.dumps(context_block, indent=2)}")

        rules_text = "\n".join(get_email_rules())
        user_message = (
            "You are an expert at personalizing sales emails for golf industry outreach. "
            "CRITICAL RULES:\n"
            "1. DO NOT modify the subject line\n"
            "2. DO NOT reference weather or seasonal conditions unless specifically provided\n" 
            "3. DO NOT reference any promotions from previous emails\n"
            "4. Focus on business value and problem-solving aspects\n"
            "5. Avoid presumptive descriptions of club features\n"
            "6. Keep club references brief and relevant to the service\n"
            "7. Keep tone professional and direct\n"
            "8. ONLY modify the first paragraph of the email - leave the rest unchanged\n"
            "Format response as:\n"
            "Subject: [keep original subject]\n\n"
            "Body:\n[personalized body]\n\n"
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{rules_text}\n\n"
            "TASK:\n"
            "1. Focus on one key benefit relevant to the club\n"
            "2. Maintain professional tone\n"
            "3. Return ONLY the subject and body\n"
            "4. Only modify the first paragraph after the greeting - keep all other paragraphs exactly as provided"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "model": MODEL_NAME,
            "temperature": 0.3,
        }

        logger.info("Personalizing email for:")
        response = _send_xai_request(payload)
        logger.debug(f"Received xAI response:\n{response}")

        personalized_subject, personalized_body = _parse_xai_response(response)
        
        # Ensure we're returning strings, not dictionaries
        final_subject = personalized_subject if personalized_subject else subject
        final_body = personalized_body if personalized_body else body
        
        if isinstance(final_body, dict):
            final_body = final_body.get('body', body)
            
        # Replace Byrdi with Swoop in response
        if isinstance(final_body, str):
            final_body = final_body.replace("Byrdi", "Swoop")
        if isinstance(final_subject, str):
            final_subject = final_subject.replace("Byrdi", "Swoop")

        # Check for any remaining placeholders (for debugging)
        remaining_placeholders = check_for_placeholders(final_subject) + check_for_placeholders(final_body)
        if remaining_placeholders:
            logger.warning(f"Unreplaced placeholders found: {remaining_placeholders}")

        return {
            "subject": final_subject,
            "body": final_body
        }

    except Exception as e:
        logger.error(f"Error in email personalization: {str(e)}")
        return {
            "subject": subject,
            "body": body
        }


def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body.
    Handles various response formats consistently.
    """
    try:
        if not response:
            raise ValueError("Empty response received")

        lines = [line.strip() for line in response.split("\n") if line.strip()]

        subject = ""
        body_lines = []
        in_body = False

        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith("subject:"):
                subject = line.replace("Subject:", "", 1).strip()
            elif lower_line.startswith("body:"):
                in_body = True
            elif in_body:
                # Simple grouping into paragraphs/signature
                if line.startswith(("Hey", "Hi", "Dear")):
                    body_lines.append(f"{line}\n\n")
                else:
                    body_lines.append(f"{line}\n\n")

        body = "".join(body_lines)

        while "\n\n\n" in body:
            body = body.replace("\n\n\n", "\n\n")
        body = body.rstrip() + "\n"

        if not subject:
            subject = "Follow-up"

        logger.debug(
            f"Parsed result - Subject: {subject}, Body length: {len(body)}"
        )
        return subject, body

    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}")
        raise


def get_xai_icebreaker(club_name: str, recipient_name: str, timeout: int = 10) -> str:
    """Get a personalized icebreaker from the xAI service."""
    try:
        if not club_name.strip():
            logger.debug("Empty club name provided")
            return ""

        cache_key = f"icebreaker:{club_name}:{recipient_name}"
        if cache_key in _cache["icebreakers"]:
            logger.debug(f"Using cached icebreaker for {club_name}")
            return _cache["icebreakers"][cache_key]

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at creating icebreakers for golf club outreach.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a brief, natural-sounding icebreaker for {club_name}. "
                        "Keep it concise and professional."
                    ),
                },
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": ANALYSIS_TEMPERATURE,
        }

        logger.info(f"Generating icebreaker for club: {club_name}")
        response = _send_xai_request(payload, max_retries=3, retry_delay=1)
        
        if not response:
            logger.debug(f"Empty response from xAI for {club_name}")
            return ""
            
        # Clean and validate response
        icebreaker = response.strip()
        if not icebreaker:
            logger.debug(f"Empty icebreaker after cleaning for {club_name}")
            return ""

        logger.debug(f"Generated icebreaker for {club_name}: {icebreaker[:100]}")
        _cache["icebreakers"][cache_key] = icebreaker
        return icebreaker

    except Exception as e:
        logger.debug(f"Error generating icebreaker: {str(e)}")
        return ""

def xai_club_segmentation_search(club_name: str, location: str) -> Dict[str, Any]:
    """
    Returns a dictionary with the club's likely segmentation profile:
      - club_type
      - facility_complexity
      - geographic_seasonality
      - has_pool
      - has_tennis_courts
      - number_of_holes
      - analysis_text
      - company_short_name
    """
    if "club_segmentation" not in _cache:
        _cache["club_segmentation"] = {}

    cache_key = f"{club_name}_{location}"
    if cache_key in _cache["club_segmentation"]:
        logger.debug(f"Using cached segmentation result for {club_name} in {location}")
        return _cache["club_segmentation"][cache_key]

    logger.info(f"Searching for club segmentation info: {club_name} in {location}")

    prompt = f"""
Classify {club_name} in {location} with precision:

0. **OFFICIAL NAME**: What is the correct, official name of this facility?
1. **SHORT NAME**: Create a brief, memorable name by removing common terms like "Country Club", "Golf Club", "Golf Course", etc. Keep it under 100 characters.
2. **CLUB TYPE**: Is it Private, Public - High Daily Fee, Public - Low Daily Fee, Municipal, Resort, Country Club, or Unknown?
3. **FACILITY COMPLEXITY**: Single-Course, Multi-Course, or Unknown?
4. **GEOGRAPHIC SEASONALITY**: Year-Round or Seasonal?
5. **POOL**: ONLY answer 'Yes' if you find clear, direct evidence of a pool.
6. **TENNIS COURTS**: ONLY answer 'Yes' if there's explicit evidence.
7. **GOLF HOLES**: Verify from official sources or consistent user mentions.

CRITICAL RULES:
- **Do not assume amenities based on the type or perceived status of the club.**
- **Confirm amenities only with solid evidence; otherwise, use 'Unknown'.**
- **Use specific references for each answer where possible.**
- **For SHORT NAME: Keep it professional and recognizable while being concise.**

Format your response with these exact headings:
OFFICIAL NAME:
[Answer]

SHORT NAME:
[Answer]

CLUB TYPE:
[Answer]

FACILITY COMPLEXITY:
[Answer]

GEOGRAPHIC SEASONALITY:
[Answer]

POOL:
[Answer]

TENNIS COURTS:
[Answer]

GOLF HOLES:
[Answer]
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at segmenting golf clubs for marketing outreach. "
                    "CRITICAL: Only state amenities as present if verified with certainty. "
                    "Use 'Unknown' if not certain."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "model": MODEL_NAME,
        "temperature": 0.0,
    }

    response = _send_xai_request(payload)
    logger.debug(f"Club segmentation result for {club_name}:")

    parsed_segmentation = _parse_segmentation_response(response)
    _cache["club_segmentation"][cache_key] = parsed_segmentation
    return parsed_segmentation


def _parse_segmentation_response(response: str) -> Dict[str, Any]:
    """Parse the structured response from xAI segmentation search."""
    def clean_value(text: str) -> str:
        if "**Evidence**:" in text:
            text = text.split("**Evidence**:")[0]
        elif "- **Evidence**:" in text:
            text = text.split("- **Evidence**:")[0]
        return text.strip().split('\n')[0].strip()

    result = {
        'name': '',
        'company_short_name': '',
        'club_type': 'Unknown',
        'facility_complexity': 'Unknown',
        'geographic_seasonality': 'Unknown',
        'has_pool': 'Unknown',
        'has_tennis_courts': 'Unknown',
        'number_of_holes': 0,
        'analysis_text': ''
    }
    
    # Add name and short name detection patterns
    name_match = re.search(r'(?:OFFICIAL NAME|NAME):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    short_name_match = re.search(r'SHORT NAME:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    
    if name_match:
        result['name'] = clean_value(name_match.group(1))
    if short_name_match:
        result['company_short_name'] = clean_value(short_name_match.group(1))
    
    logger.debug(f"Raw segmentation response:\n{response}")
    
    sections = {
        'club_type': re.search(r'CLUB TYPE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'facility_complexity': re.search(r'FACILITY COMPLEXITY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'geographic_seasonality': re.search(r'GEOGRAPHIC SEASONALITY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'pool': re.search(r'(?:POOL|HAS POOL):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'tennis': re.search(r'(?:TENNIS COURTS|HAS TENNIS):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'holes': re.search(r'GOLF HOLES:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    }

    # Process club type with better matching
    if sections['club_type']:
        club_type = clean_value(sections['club_type'].group(1))
        logger.debug(f"Processing club type: '{club_type}'")
        
        club_type_lower = club_type.lower()
        
        # Handle combined types first
        if any(x in club_type_lower for x in ['private', 'semi-private']) and 'country club' in club_type_lower:
            result['club_type'] = 'Country Club'
        # Handle public course variations
        elif 'public' in club_type_lower:
            if 'high' in club_type_lower and 'daily fee' in club_type_lower:
                result['club_type'] = 'Public - High Daily Fee'
            elif 'low' in club_type_lower and 'daily fee' in club_type_lower:
                result['club_type'] = 'Public - Low Daily Fee'
            else:
                result['club_type'] = 'Public Course'
        # Then handle other types
        elif 'country club' in club_type_lower:
            result['club_type'] = 'Country Club'
        elif 'private' in club_type_lower:
            result['club_type'] = 'Private Course'
        elif 'resort' in club_type_lower:
            result['club_type'] = 'Resort Course'
        elif 'municipal' in club_type_lower:
            result['club_type'] = 'Municipal Course'
        elif 'semi-private' in club_type_lower:
            result['club_type'] = 'Semi-Private Course'
        elif 'management company' in club_type_lower:
            result['club_type'] = 'Management Company'

    # Keep existing pool detection
    if sections['pool']:
        pool_text = clean_value(sections['pool'].group(1)).lower()
        logger.debug(f"Found pool text in section: {pool_text}")
        if 'yes' in pool_text:
            result['has_pool'] = 'Yes'
            logger.debug("Pool found in standard section")

    # Add additional pool detection patterns
    if result['has_pool'] != 'Yes':  # Only check if we haven't found a pool yet
        pool_patterns = [
            r'AMENITIES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)',
            r'FACILITIES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)',
            r'(?:^|\n)-\s*(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$)',
            r'FEATURES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)'
        ]
        
        for pattern in pool_patterns:
            pool_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if pool_match:
                logger.debug(f"Found pool in additional pattern: {pool_match.group(0)}")
                result['has_pool'] = 'Yes'
                break

    # Process geographic seasonality
    if sections['geographic_seasonality']:
        seasonality = clean_value(sections['geographic_seasonality'].group(1)).lower()
        if 'year' in seasonality or 'round' in seasonality:
            result['geographic_seasonality'] = 'Year-Round'
        elif 'seasonal' in seasonality:
            result['geographic_seasonality'] = 'Seasonal'

    # Process number of holes with better validation
    if sections['holes']:
        holes_text = clean_value(sections['holes'].group(1)).lower()
        logger.debug(f"Processing holes text: '{holes_text}'")
        
        # First check for explicit mentions of multiple courses
        if 'three' in holes_text and '9' in holes_text:
            result['number_of_holes'] = 27
        elif 'two' in holes_text and '9' in holes_text:
            result['number_of_holes'] = 18
        elif '27' in holes_text:
            result['number_of_holes'] = 27
        elif '18' in holes_text:
            result['number_of_holes'] = 18
        elif '9' in holes_text:
            result['number_of_holes'] = 9
        else:
            # Try to extract any other number
            number_match = re.search(r'(\d+)', holes_text)
            if number_match:
                try:
                    result['number_of_holes'] = int(number_match.group(1))
                    logger.debug(f"Found {result['number_of_holes']} holes")
                except ValueError:
                    logger.warning(f"Could not convert {number_match.group(1)} to integer")

    # Process facility complexity
    if sections['facility_complexity']:
        complexity = clean_value(sections['facility_complexity'].group(1)).lower()
        logger.debug(f"Processing facility complexity: '{complexity}'")
        
        if 'single' in complexity or 'single-course' in complexity:
            result['facility_complexity'] = 'Single-Course'
        elif 'multi' in complexity or 'multi-course' in complexity:
            result['facility_complexity'] = 'Multi-Course'
        elif complexity and complexity != 'unknown':
            # Log unexpected values for debugging
            logger.warning(f"Unexpected facility complexity value: {complexity}")
            
    logger.debug(f"Parsed segmentation result: {result}")

    # Enhanced tennis detection
    tennis_found = False
    # First check standard TENNIS section
    if sections['tennis']:
        tennis_text = clean_value(sections['tennis'].group(1)).lower()
        logger.debug(f"Found tennis text: {tennis_text}")
        if 'yes' in tennis_text:
            result['has_tennis_courts'] = 'Yes'
            tennis_found = True
            logger.debug("Tennis courts found in standard section")
    
    # If no tennis found in standard section, check additional patterns
    if not tennis_found:
        tennis_patterns = [
            r'TENNIS COURTS:\s*(.+?)(?=\n[A-Z ]+?:|$)',
            r'HAS TENNIS:\s*(.+?)(?=\n[A-Z ]+?:|$)',
            r'AMENITIES:.*?(?:tennis|tennis courts?).*?(?=\n[A-Z ]+?:|$)'
        ]
        for pattern in tennis_patterns:
            tennis_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if tennis_match:
                tennis_text = clean_value(tennis_match.group(1) if tennis_match.groups() else tennis_match.group(0)).lower()
                logger.debug(f"Found additional tennis text: {tennis_text}")
                if any(word in tennis_text for word in ['yes', 'tennis']):
                    result['has_tennis_courts'] = 'Yes'
                    logger.debug("Tennis courts found in additional patterns")
                    break

    return result


def get_club_summary(club_name: str, location: str) -> str:
    """
    Get a one-paragraph summary of the club using xAI.
    """
    if not club_name or not location:
        return ""

    # Only get segmentation info
    segmentation_info = xai_club_segmentation_search(club_name, location)

    # Create system prompt based on verified info
    verified_info = {
        'type': segmentation_info.get('club_type', 'Unknown'),
        'holes': segmentation_info.get('number_of_holes', 0),
        'has_pool': segmentation_info.get('has_pool', 'No'),
        'has_tennis': segmentation_info.get('has_tennis_courts', 'No')
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a club analyst. Provide a factual one paragraph summary "
                    "based on verified information. Do not make assumptions."
                )
            },
            {
                "role": "user", 
                "content": f"Give me a one paragraph summary about {club_name} in {location}, "
                          f"using these verified facts: {verified_info}"
            }
        ],
        "model": MODEL_NAME,
        "temperature": 0.0,
    }

    response = _send_xai_request(payload)
    return response.strip()


def build_context_block(interaction_history=None, objection_handling=None, original_email=None, company_data=None):
    """Build context block for email personalization."""
    context = {}
    
    if interaction_history:
        context["interaction_history"] = interaction_history
        
    if objection_handling:
        context["objection_handling"] = objection_handling
        
    if original_email:
        context["original_email"] = original_email if isinstance(original_email, dict) else {
            "subject": original_email[0],
            "body": original_email[1]
        }
    
    if company_data:
        # Use short name from segmentation if it exists, otherwise use full name
        short_name = company_data.get("company_short_name") or company_data.get("name", "")
        logger.debug(f"Using company_short_name: {short_name}")
        
        context["company_data"] = {
            "name": company_data.get("name", ""),
            "company_short_name": short_name,
            "city": company_data.get("city", ""),
            "state": company_data.get("state", ""),
            "has_pool": company_data.get("has_pool", "No"),
            "club_type": company_data.get("club_type", ""),
            "club_info": company_data.get("club_info", "")
        }
        
        # Debug logging for company data
        logger.debug(f"Building context block with company data:")
        logger.debug(f"Input company_data: {json.dumps(company_data, indent=2)}")
        logger.debug(f"Processed context: {json.dumps(context['company_data'], indent=2)}")
    
    return context


def check_for_placeholders(text: str) -> List[str]:
    """Check for any remaining placeholders in the text."""
    import re
    pattern = r'\[([^\]]+)\]'
    return re.findall(pattern, text)

def analyze_auto_reply(body: str, subject: str) -> Dict[str, str]:
    """
    Analyze auto-reply email to extract contact transition information.
    
    Args:
        body (str): The email body text
        subject (str): The email subject line
        
    Returns:
        Dict with structured contact transition information:
        {
            'original_person': str,
            'new_contact': str,
            'new_email': str,
            'new_title': str,
            'phone': str,
            'company': str,
            'reason': str,
            'permanent': str
        }
    """
    prompt = f"""
Analyze this auto-reply email and extract the following information with precision:

1. **ORIGINAL PERSON**: Who sent the auto-reply?
2. **NEW CONTACT**: Who is the new person to contact?
3. **NEW EMAIL**: What is their new email address?
4. **NEW TITLE**: What is their job title/role?
5. **PHONE**: Any phone number provided?
6. **COMPANY**: Company name if mentioned
7. **REASON**: Why is the original person no longer available? (Retired, Left Company, etc.)
8. **PERMANENT**: Is this a permanent change? (Yes/No/Unknown)

CRITICAL RULES:
- Only extract information explicitly stated in the message
- Use 'Unknown' if information is not clearly provided
- Do not make assumptions
- For emails, only include if properly formatted (user@domain.com)

Email Subject: {subject}
Email Body:
{body}

Format your response with these exact headings:
ORIGINAL PERSON:
[Answer]

NEW CONTACT:
[Answer]

NEW EMAIL:
[Answer]

NEW TITLE:
[Answer]

PHONE:
[Answer]

COMPANY:
[Answer]

REASON:
[Answer]

PERMANENT:
[Answer]
"""

    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing auto-reply emails and extracting "
                        "contact information changes. Only return verified information, "
                        "use 'Unknown' if not certain."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_NAME,
            "temperature": 0.0
        }

        response = _send_xai_request(payload)
        logger.debug(f"Raw xAI response for contact extraction:\n{response}")

        # Parse the response
        result = {
            'original_person': '',
            'new_contact': '',
            'new_email': '',
            'new_title': 'Unknown',
            'phone': 'Unknown',
            'company': 'Unknown',
            'reason': 'Unknown',
            'permanent': 'Unknown'
        }

        sections = {
            'original_person': re.search(r'ORIGINAL PERSON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_contact': re.search(r'NEW CONTACT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_email': re.search(r'NEW EMAIL:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_title': re.search(r'NEW TITLE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'phone': re.search(r'PHONE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'company': re.search(r'COMPANY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'reason': re.search(r'REASON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'permanent': re.search(r'PERMANENT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
        }

        for key, match in sections.items():
            if match:
                result[key] = match.group(1).strip().split('\n')[0].strip()

        # Validate email format
        if result['new_email'] and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', result['new_email']):
            result['new_email'] = ''

        logger.debug(f"Parsed contact info: {result}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing auto-reply: {str(e)}")
        return None
