#!/usr/bin/env python3
# main.py
# -----------------------------------------------------------------------------
# EXAMPLE FULL WORKING FILE
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
# GLOBAL WORKFLOW MODE VARIABLE
# -----------------------------------------------------------------------------
# Choose "companies" if you want to filter for companies first.
# Choose "leads" if you want to filter for leads first.
WORKFLOW_MODE = "leads"

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
COMPANY_FILTERS = [
    {
        "propertyName": "club_type",
        "operator": "EQ",
        "value": ""
    },
    {
        "propertyName": "annualrevenue",
        "operator": "GT",
        "value": ""
    },
    {
        "propertyName": "state",
        "operator": "EQ",
        "value": ""
    },
    {
        "propertyName": "name",
        "operator": "EQ",
        "value": ""
    },
    {
        "propertyName": "facility_complexity",
        "operator": "EQ",
        "value": ""
    }
]

LEAD_FILTERS = [
    {
        "propertyName": "associatedcompanyid",
        "operator": "EQ",
        "value": ""  # Will be set per company
    },
    {
        "propertyName": "lead_score",
        "operator": "GT",
        "value": "0"
    },
    {
        "propertyName": "hs_sales_email_last_replied",
        "operator": "GTE",
        "value": ""
    },
    {
        "propertyName": "email",
        "operator": "EQ",
        "value": ""
    },
    {
        "propertyName": "email_domain",
        "operator": "CONTAINS",
        "value": ""
    }
]

LEADS_TO_PROCESS = 3

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
# UTILITY FUNCTIONS
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

def get_country_club_companies(hubspot: HubspotService) -> List[Dict[str, Any]]:
    """Get all country club companies using HubspotService and our filter groups."""
    try:
        # Only include filters that have a valid value
        active_filters = [f for f in COMPANY_FILTERS if f.get("value") not in [None, "", []]]
        
        url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
        payload = {
            "filterGroups": [{"filters": active_filters}],
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
            ],
            "limit": 100
        }
        
        logger.debug(f"Searching companies with filters: {active_filters}")
        response = hubspot._make_hubspot_post(url, payload)
        results = response.get("results", [])
        logger.info(f"Found {len(results)} companies matching filters")
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting country club companies: {str(e)}", exc_info=True)
        return []

def get_leads_for_company(hubspot: HubspotService, company_id: str) -> List[Dict]:
    """Get all leads/contacts associated with a company with score > 0."""
    try:
        # Filter out any LEAD_FILTERS that are empty
        active_filters = [f for f in LEAD_FILTERS if f.get("value") not in [None, "", []]]
        # Force company ID for this search
        company_filter = {
            "propertyName": "associatedcompanyid",
            "operator": "EQ",
            "value": company_id
        }
        # Make sure we have a unique entry for the company
        active_filters.append(company_filter)
        
        url = f"{hubspot.base_url}/crm/v3/objects/contacts/search"
        payload = {
            "filterGroups": [{"filters": active_filters}],
            "properties": [
                "email", 
                "firstname", 
                "lastname", 
                "jobtitle", 
                "lead_score", 
                "hs_object_id",
                "recent_interaction",
                "last_email_date"
            ],
            "sorts": [
                {
                    "propertyName": "lead_score",
                    "direction": "DESCENDING"
                }
            ],
            "limit": 100
        }
        
        logger.debug(f"Searching leads with filters: {active_filters}")
        response = hubspot._make_hubspot_post(url, payload)
        results = response.get("results", [])
        # # Sort by lead_score descending, just in case
        # sorted_results = sorted(
        #     results,
        #     key=lambda x: float(x.get("properties", {}).get("lead_score", "0") or "0"),
        #     reverse=True
        # )
        sorted_results = results
        
        logger.info(f"Found {len(sorted_results)} leads for company {company_id}")
        return sorted_results
        
    except Exception as e:
        logger.error(f"Error getting leads for company {company_id}: {str(e)}", exc_info=True)
        return []

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
    """Collects all prior emails and notes from the lead_sheet."""
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

def store_draft_info(lead_id, subject, body, scheduled_date, sequence_num, draft_id):
    """
    Persists the draft info (subject, body, scheduled send date, etc.) to the 'emails' table.
    """
    try:
        logger.debug(f"[store_draft_info] Attempting to store draft info for lead_id={lead_id}, scheduled_date={scheduled_date}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE emails
            SET subject = ?, body = ?, scheduled_send_date = ?, sequence_num = ?, draft_id = ?
            WHERE lead_id = ? AND sequence_num = ?
            
            IF @@ROWCOUNT = 0
            BEGIN
                INSERT INTO emails
                    (lead_id, subject, body, scheduled_send_date, sequence_num, draft_id, status)
                VALUES
                    (?, ?, ?, ?, ?, ?, 'draft')
            END
            """,
            (
                subject, 
                body,
                scheduled_date,
                sequence_num,
                draft_id,
                lead_id, 
                sequence_num,
                
                lead_id,
                subject,
                body,
                scheduled_date,
                sequence_num,
                draft_id
            )
        )
        
        conn.commit()
        logger.debug(f"[store_draft_info] Successfully wrote scheduled_date={scheduled_date} for lead_id={lead_id}, draft_id={draft_id}")
        
    except Exception as e:
        logger.error(f"[store_draft_info] Failed to store draft info: {str(e)}")
        conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def get_signature() -> str:
    """Return standardized signature block."""
    return "\n\nCheers,\nTy\n\nSwoop Golf\n480-225-9702\nswoopgolf.com"

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
        best_days = outreach_window.get("Best Day", [0,1,2,3,4])  # Mon-Fri default
        
        now = datetime.now()
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
        
        send_hour = best_time["start"]
        send_minute = random.randint(0, 59)
        if random.random() < 0.5:
            send_hour = min(send_hour + 1, best_time["end"])
            
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
        return datetime.now() + timedelta(days=1, hours=10)

def get_next_month_first_day(current_date):
    if current_date.month == 12:
        return current_date.replace(year=current_date.year + 1, month=1, day=1)
    return current_date.replace(month=current_date.month + 1, day=1)

def get_template_path(club_type: str, role: str, sequence_num: int = 1) -> str:
    """
    Get the appropriate template path based on club type and role.
    """
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
    
    template_path = Path(PROJECT_ROOT) / "docs" / "templates" / normalized_club_type / f"{normalized_role}_initial_outreach_{sequence_num}.md"
    
    if not template_path.exists():
        fallback_path = Path(PROJECT_ROOT) / "docs" / "templates" / normalized_club_type / f"fallback_{sequence_num}.md"
        logger.warning(f"Specific template not found at {template_path}, falling back to {fallback_path}")
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

def get_leads(hubspot: HubspotService, min_score: float = 0) -> List[Dict]:
    """Get all leads with scores above minimum threshold."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/contacts/search"
        
        # Use LEAD_FILTERS for the leads-first approach
        active_filters = []
        for filter_def in LEAD_FILTERS:
            if filter_def["propertyName"] == "lead_score":
                filter_def["value"] = str(min_score)
            if filter_def.get("value"):
                active_filters.append(filter_def)
        
        payload = {
            "filterGroups": [{"filters": active_filters}],
            "properties": [
                "email", 
                "firstname", 
                "lastname", 
                "jobtitle", 
                "lead_score",
                "hs_sales_email_last_replied",
                "recent_interaction",
                "associatedcompanyid"
            ],
            "limit": 100,
            "sorts": [
                {
                    "propertyName": "lead_score",
                    "direction": "ASCENDING"
                }
            ]
        }
        response = hubspot._make_hubspot_post(url, payload)
        
        leads = response.get("results", [])
        logger.info(f"Found {len(leads)} leads after applying filters")
        for lead in leads:
            logger.debug(
                f"Lead {lead.get('id')}: "
                f"Score = {lead.get('properties', {}).get('lead_score')}, "
                f"Last reply = {lead.get('properties', {}).get('hs_sales_email_last_replied')}"
            )
        
        return leads
    except Exception as e:
        logger.error(f"Error getting high scoring leads: {e}")
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

def is_company_in_best_state(company_props: Dict) -> bool:
    """
    Determine if the current month is within the 'best months' 
    for the company's geographic seasonality.
    """
    try:
        geography = company_props.get("geographic_seasonality", "") or "Year-Round Golf"
        club_type = company_props.get("club_type", "")
        
        season_data = {}
        peak_start = company_props.get("peak_season_start_month")
        peak_end = company_props.get("peak_season_end_month")
        
        if peak_start and peak_end:
            season_data = {
                "peak_season_start": f"{peak_start}-01",
                "peak_season_end": f"{peak_end}-01"
            }
        
        best_months = get_best_month(geography, club_type, season_data)
        current_month = datetime.now().month
        
        is_best = current_month in best_months
        logger.debug(
            f"Company seasonality check: {company_props.get('name', 'Unknown')} "
            f"({geography}) - Current month {current_month} "
            f"{'is' if is_best else 'is not'} in best months {best_months}"
        )
        
        return is_best
        
    except Exception as e:
        logger.error(
            f"Error checking company best state for {company_props.get('name', 'Unknown')}: {str(e)}", 
            exc_info=True
        )
        return False

def replace_placeholders(text: str, lead_data: dict) -> str:
    """Replace placeholders in text with actual values."""
    text = clean_company_name(text)  # Clean any old company references
    replacements = {
        "[FirstName]": lead_data["lead_data"].get("firstname", ""),
        "[LastName]": lead_data["lead_data"].get("lastname", ""),
        "[ClubName]": lead_data["company_data"].get("name", ""),
        "[JobTitle]": lead_data["lead_data"].get("jobtitle", ""),
        "[CompanyName]": lead_data["company_data"].get("name", ""),
        "[City]": lead_data["company_data"].get("city", ""),
        "[State]": lead_data["company_data"].get("state", "")
    }
    
    result = text
    for placeholder, value in replacements.items():
        if value:
            result = result.replace(placeholder, value)
    return result

def check_lead_filters(lead_data: dict) -> bool:
    """(Optional) Add custom logic to check if lead meets your internal filter criteria."""
    # You can add whatever logic you want here. For example:
    # if lead_data["email"].endswith("@spamdomain.com"): return False
    # Just a placeholder for demonstration.
    return True

# -----------------------------------------------------------------------------
# COMPANIES-FIRST WORKFLOW
# -----------------------------------------------------------------------------
def main_companies_first():
    """
    1) Filter for companies first via COMPANY_FILTERS
    2) For each matching company, get leads (LEAD_FILTERS)
    3) Build and store the outreach email for each lead
    """
    try:
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
        conversation_analyzer = ConversationAnalysisService()
        leads_processed = 0
        
        with workflow_step("1", "Get Country Club companies", workflow_context):
            companies = get_country_club_companies(hubspot)
            logger.info(f"Found {len(companies)} companies to process")
        
        with workflow_step("2", "Process each company & its leads", workflow_context):
            for company in companies:
                company_id = company.get("id")
                company_props = company.get("properties", {})
                
                # Enrich the company
                enrichment_result = company_enricher.enrich_company(company_id)
                if not enrichment_result.get("success", False):
                    logger.warning(f"Enrichment failed for company {company_id}, skipping.")
                    continue
                company_props.update(enrichment_result.get("data", {}))
                
                # Optional: check if the company meets your filter/time-of-year logic
                if not is_company_in_best_state(company_props):
                    logger.info(f"Company {company_id} not in best outreach window, skipping.")
                    continue
                
                # Get the leads for this company
                leads = get_leads_for_company(hubspot, company_id)
                if not leads:
                    logger.info(f"No leads found for company {company_id}")
                    continue
                
                for lead in leads:
                    lead_id = lead.get("id")
                    lead_props = lead.get("properties", {})
                    logger.info(f"Processing lead: {lead_props.get('email')} (ID: {lead_id})")
                    
                    # Build your email (same as in main_leads_first)
                    try:
                        lead_data_full = extract_lead_data(company_props, lead_props)
                        if not check_lead_filters(lead_data_full["lead_data"]):
                            logger.info(f"Lead {lead_id} did not pass custom checks, skipping.")
                            continue
                        
                        # Gather personalization
                        personalization = gather_personalization_data(
                            company_name=lead_data_full["company_data"]["name"],
                            city=lead_data_full["company_data"]["city"],
                            state=lead_data_full["company_data"]["state"]
                        )
                        
                        # Decide which template
                        template_path = get_template_path(
                            club_type=lead_data_full["company_data"]["club_type"],
                            role=lead_data_full["lead_data"]["jobtitle"],
                            sequence_num=1
                        )
                        
                        # Calculate a good send date
                        send_date = calculate_send_date(
                            geography=lead_data_full["company_data"]["geographic_seasonality"],
                            persona=lead_data_full["lead_data"]["jobtitle"],
                            state_code=lead_data_full["company_data"]["state"],
                            season_data={
                                "peak_season_start": lead_data_full["company_data"].get("peak_season_start_month"),
                                "peak_season_end": lead_data_full["company_data"].get("peak_season_end_month")
                            }
                        )
                        
                        # Build the outreach email
                        email_content = build_outreach_email(
                            template_path=template_path,
                            profile_type=lead_data_full["lead_data"]["jobtitle"],
                            placeholders={
                                "company_name": lead_data_full["company_data"]["name"],
                                "first_name": lead_data_full["lead_data"]["firstname"],
                                "last_name": lead_data_full["lead_data"]["lastname"],
                                "job_title": lead_data_full["lead_data"]["jobtitle"],
                                "company_info": lead_data_full["company_data"].get("club_info", ""),
                                "has_news": personalization.get("has_news", False),
                                "news_text": personalization.get("news_text", ""),
                                "ClubName": lead_data_full["company_data"]["name"]
                            },
                            current_month=datetime.now().month,
                            start_peak_month=lead_data_full["company_data"].get("peak_season_start_month"),
                            end_peak_month=lead_data_full["company_data"].get("peak_season_end_month")
                        )

                        if email_content:
                            # Get conversation analysis for personalization
                            email_address = lead_data_full["lead_data"]["email"]
                            conversation_summary = conversation_analyzer.analyze_conversation(email_address)
                            
                            # Create context block with the summary
                            context = build_context_block(
                                interaction_history=conversation_summary,
                                original_email={"subject": email_content[0], "body": email_content[1]}
                            )
                            
                            # Use the context in personalization
                            personalized_content = personalize_email_with_xai(
                                lead_sheet=lead_data_full,
                                subject=email_content[0],
                                body=email_content[1],
                                summary=conversation_summary,
                                context=context
                            )
                            
                            # Add signature to the personalized body
                            personalized_content["body"] = personalized_content["body"].rstrip() + get_signature()
                            
                            # Create the Gmail draft with personalized content
                            draft_result = create_draft(
                                sender="me",
                                to=email_address,
                                subject=personalized_content["subject"],
                                message_text=personalized_content["body"]
                            )
                            
                            if draft_result["status"] == "ok":
                                store_lead_email_info(
                                    lead_sheet={
                                        "lead_data": {
                                            "email": lead_data_full["lead_data"]["email"],
                                            "properties": {
                                                "hs_object_id": lead_id,
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
                                    subject=email_content[0],
                                    body=email_content[1],
                                    sequence_num=1
                                )
                                logger.info(f"Created draft email for {lead_data_full['lead_data']['email']}")
                            else:
                                logger.error(f"Failed to create Gmail draft for lead {lead_id}")
                        else:
                            logger.error(f"Failed to generate email content for lead {lead_id}")
                    
                    except Exception as e:
                        logger.error(f"Error processing lead {lead_id} for company {company_id}: {e}", exc_info=True)
                    
                    leads_processed += 1
                    if leads_processed >= LEADS_TO_PROCESS:
                        logger.info(f"Reached processing limit of {LEADS_TO_PROCESS} leads")
                        return
    
    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in main_companies_first workflow: {str(e)}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# LEADS-FIRST WORKFLOW
# -----------------------------------------------------------------------------
def main_leads_first():
    """
    1) Get leads first
    2) Retrieve each lead’s associated company
    3) Check if that company passes the workflow filters
    4) Build and store the outreach email
    """
    try:
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
        conversation_analyzer = ConversationAnalysisService()
        leads_processed = 0
        
        with workflow_step("1", "Initialize and get leads", workflow_context):
            # Pull high-scoring leads first
            leads = get_leads(hubspot, min_score=0)
            logger.info(f"Found {len(leads)} total leads (score > 0)")
        
        with workflow_step("2", "Process each lead's associated company", workflow_context):
            for lead in leads:
                lead_id = lead.get("id")
                lead_props = lead.get("properties", {})
                
                logger.info(f"Processing lead: {lead_props.get('email')} (ID: {lead_id})")
                
                # 1) Retrieve associated company
                associated_company_id = lead_props.get("associatedcompanyid")
                if not associated_company_id:
                    logger.warning(f"No associated company for lead {lead_id}, skipping.")
                    continue
                
                company = get_company_by_id(hubspot, associated_company_id)
                company_id = company.get("id")
                company_props = company.get("properties", {}) if company else {}
                
                if not company_id:
                    logger.warning(f"Could not retrieve valid company for lead {lead_id}, skipping.")
                    continue
                
                # 2) Enrich the company data
                enrichment_result = company_enricher.enrich_company(company_id)
                if not enrichment_result.get("success", False):
                    logger.warning(f"Enrichment failed for company {company_id}, skipping.")
                    continue
                company_props.update(enrichment_result.get("data", {}))
                
                # 3) Check if this company meets the workflow's COMPANY_FILTERS 
                active_company_filters = [f for f in COMPANY_FILTERS if f.get("value")]
                logger.debug(f"\nChecking filters for {company_props.get('name', 'Unknown Company')} (ID: {company_id})")
                logger.debug("----------------------------------------")
                
                meets_filters = True
                for f in active_company_filters:
                    prop_name = f["propertyName"]
                    operator = f["operator"]
                    filter_value = f["value"]
                    company_value = company_props.get(prop_name, "")
                    
                    logger.info(
                        f"\nChecking filter for {company_props.get('name', 'Unknown Company')}:\n"
                        f"Property: {prop_name}\n"
                        f"Operator: {operator}\n"
                        f"Expected: {filter_value}\n"
                        f"Actual: {company_value}"
                    )
                    
                    if operator == "EQ":
                        if str(company_value) != str(filter_value):
                            logger.info(f"FAILED: Value mismatch")
                            meets_filters = False
                            break
                    elif operator == "GT":
                        try:
                            # Convert revenue string to numeric value by removing non-numeric characters
                            company_value = ''.join(filter(str.isdigit, str(company_value))) if company_value else '0'
                            filter_value = ''.join(filter(str.isdigit, str(filter_value)))
                            
                            if not company_value or float(company_value) <= float(filter_value):
                                logger.info(f"FAILED: Value too low or empty")
                                meets_filters = False
                                break
                        except ValueError:
                            logger.info(f"FAILED: Invalid numeric value")
                            meets_filters = False
                            break
                    elif operator == "LTE":
                        # If you had a "less than or equal" comparison, handle it here
                        # (Not used in your code, but left as an example)
                        pass
                    
                    logger.info("PASSED")
                
                if not meets_filters:
                    logger.info(
                        f"{company_props.get('name', 'Unknown Company')} "
                        f"(ID: {company_id}) filtered out - failed filter checks"
                    )
                    continue
                
                # 4) Check if the current month is in the "best" time for outreach
                if not is_company_in_best_state(company_props):
                    logger.info(f"Company {company_id} not in best outreach window, skipping lead {lead_id}.")
                    continue
                
                # 5) Now process the lead
                with workflow_step("3", f"Processing lead {lead_id}", workflow_context):
                    try:
                        # Extract combined lead+company data
                        lead_data_full = extract_lead_data(company_props, lead_props)
                        
                        # Skip if unknown data
                        if "unknown" in lead_data_full["company_data"]["name"].lower():
                            logger.info(f"Skipping lead {lead_id} - Club name contains 'Unknown'")
                            continue
                        if lead_data_full["company_data"]["club_type"] == "Unknown":
                            logger.info(f"Skipping lead {lead_id} - Club type is Unknown")
                            continue
                        
                        if not check_lead_filters(lead_data_full["lead_data"]):
                            logger.info(f"Lead {lead_id} did not pass custom checks, skipping.")
                            continue
                        
                        # Instead of SQL summary, just use the 'recent_interaction' property from HubSpot
                        interaction_summary = lead_props.get("recent_interaction", "")
                        
                        # Gather personalization data
                        personalization = gather_personalization_data(
                            company_name=lead_data_full["company_data"]["name"],
                            city=lead_data_full["company_data"]["city"],
                            state=lead_data_full["company_data"]["state"]
                        )
                        
                        # Get correct template
                        template_path = get_template_path(
                            club_type=lead_data_full["company_data"]["club_type"],
                            role=lead_data_full["lead_data"]["jobtitle"],
                            sequence_num=1
                        )
                        
                        # Calculate send date
                        send_date = calculate_send_date(
                            geography=lead_data_full["company_data"]["geographic_seasonality"],
                            persona=lead_data_full["lead_data"]["jobtitle"],
                            state_code=lead_data_full["company_data"]["state"],
                            season_data={
                                "peak_season_start": lead_data_full["company_data"].get("peak_season_start_month"),
                                "peak_season_end": lead_data_full["company_data"].get("peak_season_end_month")
                            }
                        )
                        
                        # Build outreach email content
                        email_content = build_outreach_email(
                            template_path=template_path,
                            profile_type=lead_data_full["lead_data"]["jobtitle"],
                            placeholders={
                                "company_name": lead_data_full["company_data"]["name"],
                                "first_name": lead_data_full["lead_data"]["firstname"],
                                "last_name": lead_data_full["lead_data"]["lastname"],
                                "job_title": lead_data_full["lead_data"]["jobtitle"],
                                "company_info": lead_data_full["company_data"].get("club_info", ""),
                                "has_news": personalization.get("has_news", False),
                                "news_text": personalization.get("news_text", ""),
                                "ClubName": lead_data_full["company_data"]["name"]
                            },
                            current_month=datetime.now().month,
                            start_peak_month=lead_data_full["company_data"].get("peak_season_start_month"),
                            end_peak_month=lead_data_full["company_data"].get("peak_season_end_month")
                        )

                        if email_content:
                            # Get conversation analysis for personalization
                            email_address = lead_data_full["lead_data"]["email"]
                            conversation_summary = conversation_analyzer.analyze_conversation(email_address)
                            
                            # Create context block with the summary
                            context = build_context_block(
                                interaction_history=conversation_summary,
                                original_email={"subject": email_content[0], "body": email_content[1]}
                            )
                            
                            # Use the context in personalization
                            personalized_content = personalize_email_with_xai(
                                lead_sheet=lead_data_full,
                                subject=email_content[0],
                                body=email_content[1],
                                summary=conversation_summary,
                                context=context
                            )
                            
                            # Add signature to the personalized body
                            personalized_content["body"] = personalized_content["body"].rstrip() + get_signature()
                            
                            # Create the Gmail draft with personalized content
                            draft_result = create_draft(
                                sender="me",
                                to=email_address,
                                subject=personalized_content["subject"],
                                message_text=personalized_content["body"]
                            )
                            
                            if draft_result["status"] == "ok":
                                store_lead_email_info(
                                    lead_sheet={
                                        "lead_data": {
                                            "email": lead_data_full["lead_data"]["email"],
                                            "properties": {
                                                "hs_object_id": lead_id,
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
                                    subject=email_content[0],
                                    body=email_content[1],
                                    sequence_num=1
                                )
                                logger.info(f"Created draft email for {lead_data_full['lead_data']['email']}")
                            else:
                                logger.error(f"Failed to create Gmail draft for lead {lead_id}")
                        else:
                            logger.error(f"Failed to generate email content for lead {lead_id}")
                    
                    except Exception as e:
                        logger.error(f"Error processing lead {lead_id}: {str(e)}", exc_info=True)
                        continue
                    
                    leads_processed += 1
                    logger.info(f"Completed processing lead {lead_id} ({leads_processed}/{LEADS_TO_PROCESS})")
                    
                    if leads_processed >= LEADS_TO_PROCESS:
                        logger.info(f"Reached processing limit of {LEADS_TO_PROCESS} leads")
                        return
    
    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in main_leads_first workflow: {str(e)}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# SINGLE ENTRY POINT THAT CHECKS WORKFLOW_MODE
# -----------------------------------------------------------------------------
def main():
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START} and WORKFLOW_MODE={WORKFLOW_MODE}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Start scheduler in background
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Decide which approach to use
    if WORKFLOW_MODE.lower() == "companies":
        main_companies_first()
    else:
        main_leads_first()

# -----------------------------------------------------------------------------
# RUN SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
