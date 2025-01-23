
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
                conversation_summary = conversation_analyzer.analyze_conversation(email)

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

