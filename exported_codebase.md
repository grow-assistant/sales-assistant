
## main.py

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
# GLOBAL WORKFLOW MODE VARIABLE
# -----------------------------------------------------------------------------
# Choose "companies" if you want to filter for companies first.
# Choose "leads" if you want to filter for leads first.
WORKFLOW_MODE = "companies"

# -----------------------------------------------------------------------------
# FILTER-BUILDING FUNCTION (allowing both AND and OR groups)
# -----------------------------------------------------------------------------
def get_company_filters_with_conditions(
    contact_conditions: List[Dict] = None,
    club_type_conditions: List[Dict] = None,
    states: List[str] = None,
    has_pool: bool = None,
    company_id: str = None
) -> Dict:
    """
    Generate company filters with complex OR conditions.
    Each condition group uses OR logic within itself and AND logic with other groups.
    """
    # Build base filters (these will be applied to all groups)
    base_filters = []
    
    # # Add contact count filter
    # base_filters.append({
    #     "propertyName": "num_contacted_notes",
    #     "operator": "LT",
    #     "value": "3"  # Only include companies contacted less than 3 times
    # })

    # Add minimum associated contacts filter
    base_filters.append({
        "propertyName": "num_associated_contacts",
        "operator": "GTE",
        "value": "1"  # Only include companies with at least 1 contact
    })

    # Add company ID filter if specified
    if company_id:
        base_filters.append({
            "propertyName": "hs_object_id",
            "operator": "EQ",
            "value": company_id
        })
        logger.debug(f"Added company ID filter: {company_id}")

    # Add state filter if specified
    if states:
        # Ensure states are uppercase and stripped of whitespace
        formatted_states = [state.strip().lower() for state in states]
        state_filter = {
            "propertyName": "state",
            "operator": "IN",
            "values": formatted_states
        }
        base_filters.append(state_filter)
        logger.debug(f"Added states filter: {json.dumps(state_filter, indent=2)}")
    
    # Add pool filter if specified
    if has_pool is not None:
        base_filters.append({
            "propertyName": "has_pool",
            "operator": "EQ",
            "value": has_pool
        })
        logger.debug(f"Added pool filter: {has_pool}")

    # Initialize filter groups with base filters
    filter_groups = []
    
    # If no conditions were specified, just use base filters
    if not filter_groups and base_filters:
        filter_groups.append({"filters": base_filters})
        logger.debug("Created group with only base filters")

    logger.debug(f"Final filter structure: {json.dumps({'filterGroups': filter_groups}, indent=2)}")
    return {"filterGroups": filter_groups}

# -----------------------------------------------------------------------------
# EXAMPLE USAGE OF get_company_filters_with_conditions
# (Uncomment or customize as needed)
# -----------------------------------------------------------------------------
COMPANY_FILTERS = get_company_filters_with_conditions(
    states=["KS"],  # Must be in these states
    #states=["KY", "NC", "SC", "VA", "TN", "KY", "MO", "KS", "OK", "AR", "NM", "FA"],  Early Season States
    has_pool=None,              # Must have a pool
    company_id=None             # Example placeholder, or set a specific ID
)

# -----------------------------------------------------------------------------
# LEAD FILTERS
# -----------------------------------------------------------------------------
LEAD_FILTERS = [
    {
        "propertyName": "associatedcompanyid",
        "operator": "EQ",
        "value": ""  # Will be set per company
    },
    {
        "propertyName": "lead_score",
        "operator": "GT",
        "value": ""
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

LEADS_TO_PROCESS = 100

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

def get_country_club_companies(hubspot: HubspotService, states: List[str] = None) -> List[Dict[str, Any]]:
    """Get all country club companies using HubspotService and our filter groups."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
        
        # Use the globally defined COMPANY_FILTERS created above
        payload = {
            "filterGroups": COMPANY_FILTERS["filterGroups"],
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
                "end_month",
                "notes_last_contacted",
                "domain",
                "competitor"
            ],
            "limit": 100
        }
        
        logger.debug(f"Searching companies with payload: {json.dumps(payload, indent=2)}")
        response = hubspot._make_hubspot_post(url, payload)
        
        if not response.get("results"):
            logger.warning(f"No results found. Response: {json.dumps(response, indent=2)}")
            
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
            "limit": 1  # Changed from 100 to 1 to only get the first lead
        }
        
        logger.debug(f"Searching leads with filters: {active_filters}")
        response = hubspot._make_hubspot_post(url, payload)
        results = response.get("results", [])
        
        logger.info(f"Found {len(results)} leads for company {company_id}")
        return results[:1]  # Extra safety to ensure only one lead is returned
        
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

def store_draft_info(lead_id, name, email_address, sequence_num, body, scheduled_date, draft_id):
    """
    Persists the draft info to the 'emails' table.
    
    Fields ordered as per schema:
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
        send_hour = int(best_time["start"])  # Convert to int
        send_minute = random.randint(0, 59)
        
        # Randomly adjust hour within window
        if random.random() < 0.5:
            send_hour = min(send_hour + 1, int(best_time["end"]))  # Convert to int
            
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

def check_lead_filters(lead_data: dict) -> bool:
    """
    (Optional) Add custom logic to check if lead meets your internal filter criteria.
    Example:
      if lead_data["email"].endswith("@spamdomain.com"): return False
    """
    return True

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

def check_company_conditions(company_props: Dict, contact_conditions: List[Dict] = None, club_type_conditions: List[Dict] = None) -> bool:
    """
    Test if a company meets the contact and club type conditions.
    Returns True if company passes all checks, False otherwise.
    """
    logger.debug(
        f"\nChecking conditions for {company_props.get('name', 'Unknown Company')}:\n"
        f"Contact conditions: {contact_conditions}\n"
        f"Club type conditions: {club_type_conditions}"
    )

    # Check contact conditions (OR logic)
    if contact_conditions:
        contact_passed = False
        last_contacted = company_props.get("notes_last_contacted")
        logger.debug(f"Checking contact conditions - Last contacted: {last_contacted}")
        
        for condition in contact_conditions:
            operator = condition["operator"]
            value = condition.get("value")
            
            logger.debug(f"Testing contact condition: operator={operator}, value={value}")
            
            if operator == "NOT_HAS_PROPERTY" and not last_contacted:
                contact_passed = True
                logger.debug("PASSED: No contact history found")
                break
            elif operator == "LTE" and last_contacted and last_contacted <= value:
                contact_passed = True
                logger.debug(f"PASSED: Last contact ({last_contacted}) is before {value}")
                break
            # Add other operators as needed
        
        if not contact_passed:
            logger.info(
                f"Company failed contact conditions:\n"
                f"Last contacted: {last_contacted}\n"
                f"Required conditions: {contact_conditions}"
            )
            return False

    # Check club type conditions (OR logic)
    if club_type_conditions:
        club_type_passed = False
        club_type = company_props.get("club_type")
        logger.debug(f"Checking club type conditions - Club type: {club_type}")
        
        for condition in club_type_conditions:
            operator = condition["operator"]
            value = condition.get("value")
            
            logger.debug(f"Testing club type condition: operator={operator}, value={value}")
            
            if operator == "NOT_HAS_PROPERTY" and not club_type:
                club_type_passed = True
                logger.debug("PASSED: No club type specified")
                break
            elif operator == "EQ" and club_type == value:
                club_type_passed = True
                logger.debug(f"PASSED: Club type matches {value}")
                break
            # Add other operators as needed
            
        if not club_type_passed:
            logger.info(
                f"Company failed club type conditions:\n"
                f"Current club type: {club_type}\n"
                f"Required conditions: {club_type_conditions}"
            )
            return False

    logger.debug("Company passed all conditions")
    return True

# -----------------------------------------------------------------------------
# COMPANIES-FIRST WORKFLOW
# -----------------------------------------------------------------------------
def main_companies_first():
    """
    1) Get companies first
    2) For each company, get its leads
    3) Process each lead
    """
    try:
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
        conversation_analyzer = ConversationAnalysisService()
        leads_processed = 0
        
        with workflow_step("1", "Get Country Club companies", workflow_context):
            # Get companies
            companies = get_country_club_companies(hubspot)
            logger.info(f"Found {len(companies)} companies to process")
            
            # Randomize the order
            random.shuffle(companies)
            
        with workflow_step("2", "Process each company & its leads", workflow_context):
            # Process each company
            for company in companies:
                company_id = company.get("id")
                company_props = company.get("properties", {})
                
                # Check for competitor info first
                website = company_props.get("website")
                if website:
                    competitor_info = data_gatherer.check_competitor_on_website(website)
                    if competitor_info["status"] == "success" and competitor_info["competitor"]:
                        competitor = competitor_info["competitor"]
                        logger.debug(f"Found competitor {competitor} for company {company_id}")
                        
                        # Update company with competitor info
                        try:
                            hubspot._update_company_properties(
                                company_id, 
                                {"competitor": competitor}
                            )
                            logger.debug(f"Updated company {company_id} with competitor: {competitor}")
                        except Exception as e:
                            logger.error(f"Failed to update competitor for company {company_id}: {e}")
                
                # Continue with regular enrichment
                enrichment_result = company_enricher.enrich_company(company_id)
                
                if not enrichment_result.get("success", False):
                    logger.warning(f"Enrichment failed for company {company_id}, skipping.")
                    continue
                
                # Update company properties all at once
                company_props.update(enrichment_result.get("data", {}))
                
                # Simple direct checks
                club_type = company_props.get("club_type", "")
                last_contacted = company_props.get("notes_last_contacted")
                
                # # Check if club type is valid (must be Country Club or not specified)
                # if club_type and club_type != "Country Club":
                #     logger.info(f"Skipping company {company_id} - Club type is {club_type}")
                #     continue
                    
                # Check if we've contacted them recently
                if last_contacted and last_contacted > "2025-01-01T00:00:00Z":
                    logger.info(f"Skipping company {company_id} - Last contacted on {last_contacted}")
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
                    
                    # Check for recent emails early
                    email_address = lead_props["email"]
                    if has_recent_email(email_address):
                        logger.info(f"Skipping {email_address} - email sent in last 2 months")
                        continue
                    
                    try:
                        # Build your email (similar steps as main_leads_first)
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
                            role=lead_data_full["lead_data"]["jobtitle"]
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

                        if email_content:
                            # First replace placeholders
                            subject = replace_placeholders(email_content[0], lead_data_full)
                            body = replace_placeholders(email_content[1], lead_data_full)
                            
                            # Then get conversation analysis
                            email_address = lead_data_full["lead_data"]["email"]
                            conversation_summary = conversation_analyzer.analyze_conversation(email_address)
                            
                            # Create context block with placeholder-replaced content
                            context = build_context_block(
                                interaction_history=conversation_summary,
                                original_email={"subject": subject, "body": body},
                                company_data={
                                    "name": lead_data_full["company_data"]["name"],
                                    "company_short_name": (
                                        lead_data_full["company_data"].get("company_short_name") or
                                        lead_data_full.get("company_short_name") or  # Try alternate location
                                        lead_data_full["company_data"]["name"].split(" ")[0]  # Fallback
                                    ),
                                    "city": lead_data_full["company_data"].get("city", ""),
                                    "state": lead_data_full["company_data"].get("state", ""),
                                    "club_type": lead_data_full["company_data"]["club_type"],
                                    "club_info": lead_data_full["company_data"].get("club_info", "")
                                }
                            )
                            
                            # Add debug logging
                            logger.debug(f"Building context with company_short_name: {lead_data_full['company_data'].get('company_short_name', '')}")
                            
                            # Finally, personalize with xAI using the placeholder-replaced content
                            personalized_content = personalize_email_with_xai(
                                lead_sheet=lead_data_full,
                                subject=subject,
                                body=body,
                                summary=conversation_summary,
                                context=context
                            )
                                                            
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
    2) Retrieve each lead's associated company
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
                
                # Simple direct checks
                club_type = company_props.get("club_type", "")
                last_contacted = company_props.get("notes_last_contacted")
                
                # Check if club type is valid (must be Country Club or not specified)
                if club_type and club_type != "Country Club":
                    logger.info(f"Skipping company {company_id} - Club type is {club_type}")
                    continue
                    
                # Check if we've contacted them recently
                if last_contacted and last_contacted > "2025-01-01T00:00:00Z":
                    logger.info(f"Skipping company {company_id} - Last contacted on {last_contacted}")
                    continue
                
                # 4) Check if the current month is in the "best" time for outreach
                if not is_company_in_best_state(company_props):
                    logger.info(f"Company {company_id} not in best outreach window, skipping lead {lead_id}.")
                    continue
                
                # 5) Now process the lead
                with workflow_step("3", f"Processing lead {lead_id}", workflow_context):
                    try:
                        lead_data_full = extract_lead_data(company_props, lead_props)
                        
                        if "unknown" in lead_data_full["company_data"]["name"].lower():
                            logger.info(f"Skipping lead {lead_id} - Club name contains 'Unknown'")
                            continue
                        if lead_data_full["company_data"]["club_type"] == "Unknown":
                            logger.info(f"Skipping lead {lead_id} - Club type is Unknown")
                            continue
                        
                        if not check_lead_filters(lead_data_full["lead_data"]):
                            logger.info(f"Lead {lead_id} did not pass custom checks, skipping.")
                            continue
                        
                        email_address = lead_data_full["lead_data"]["email"]
                        if has_recent_email(email_address):
                            logger.info(f"Skipping {email_address} - email sent in last 2 months")
                            continue
                        
                        interaction_summary = lead_props.get("recent_interaction", "")
                        
                        personalization = gather_personalization_data(
                            company_name=lead_data_full["company_data"]["name"],
                            city=lead_data_full["company_data"]["city"],
                            state=lead_data_full["company_data"]["state"]
                        )
                        
                        template_path = get_template_path(
                            club_type=lead_data_full["company_data"]["club_type"],
                            role=lead_data_full["lead_data"]["jobtitle"]
                        )
                        
                        send_date = calculate_send_date(
                            geography=lead_data_full["company_data"]["geographic_seasonality"],
                            persona=lead_data_full["lead_data"]["jobtitle"],
                            state_code=lead_data_full["company_data"]["state"],
                            season_data={
                                "peak_season_start": lead_data_full["company_data"].get("peak_season_start_month"),
                                "peak_season_end": lead_data_full["company_data"].get("peak_season_end_month")
                            }
                        )
                        
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

                        if email_content:
                            subject = replace_placeholders(email_content[0], lead_data_full)
                            body = replace_placeholders(email_content[1], lead_data_full)
                            
                            conversation_summary = conversation_analyzer.analyze_conversation(email_address)
                            
                            context = build_context_block(
                                interaction_history=conversation_summary,
                                original_email={"subject": subject, "body": body},
                                company_data={
                                    "name": lead_data_full["company_data"]["name"],
                                    "company_short_name": (
                                        lead_data_full["company_data"].get("company_short_name") or
                                        lead_data_full.get("company_short_name") or  # Try alternate location
                                        lead_data_full["company_data"]["name"].split(" ")[0]  # Fallback
                                    ),
                                    "city": lead_data_full["company_data"].get("city", ""),
                                    "state": lead_data_full["company_data"].get("state", ""),
                                    "club_type": lead_data_full["company_data"]["club_type"],
                                    "club_info": lead_data_full["company_data"].get("club_info", "")
                                }
                            )
                            
                            # Add debug logging
                            logger.debug(f"Building context with company_short_name: {lead_data_full['company_data'].get('company_short_name', '')}")
                            
                            personalized_content = personalize_email_with_xai(
                                lead_sheet=lead_data_full,
                                subject=subject,
                                body=body,
                                summary=conversation_summary,
                                context=context
                            )
                                                            
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




## docs\templates\country_club\general_manager_initial_outreach_1.md

Hey [firstname],

[ICEBREAKER]

[SEASON_VARIATION], I’d love to introduce Swoop Golf—a platform that enables members to effortlessly order food & beverages from their mobile devices. We’ve just elevated our offering into a full-service full-club concierge platform—managing on-course orders, poolside F&B, racquet-sport deliveries, and sophisticated to-go services.

We’re inviting 2–3 clubs to join us at no cost for 2025, to ensure we perfectly address the needs of top-tier properties. At Pinetree Country Club, this model reduced average order times by 40%, keeping members impressed and pace of play consistent.

Swoop will enhance your members’ experience and preserve the exclusivity they value. Let’s chat about how this might work for [company_short_name]?



## scheduling\extended_lead_storage.py

# scheduling/extended_lead_storage.py

from datetime import datetime, timedelta
from utils.logging_setup import logger
from scheduling.database import get_db_connection, store_email_draft

def find_next_available_timeslot(desired_send_date: datetime) -> datetime:
    """
    Moves 'desired_send_date' forward if needed so that:
      1) It's at least 2 minutes after the last scheduled email
      2) We never exceed 15 emails in any rolling 3-minute window
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        while True:
            # 1) Ensure at least 2 minutes from the last scheduled email
            cursor.execute("""
                SELECT TOP 1 scheduled_send_date
                FROM emails
                WHERE scheduled_send_date IS NOT NULL
                ORDER BY scheduled_send_date DESC
            """)
            row = cursor.fetchone()
            if row:
                last_scheduled = row[0]
                min_allowed = last_scheduled + timedelta(minutes=2)
                if desired_send_date < min_allowed:
                    desired_send_date = min_allowed

            # 2) Check how many are scheduled in the 3-minute window prior to 'desired_send_date'
            window_start = desired_send_date - timedelta(minutes=3)
            cursor.execute("""
                SELECT COUNT(*)
                FROM emails
                WHERE scheduled_send_date BETWEEN ? AND ?
            """, (window_start, desired_send_date))
            count_in_3min = cursor.fetchone()[0]

            # If we already have 15 or more in that window, push out by 2 more minutes and repeat
            if count_in_3min >= 15:
                desired_send_date += timedelta(minutes=2)
            else:
                break

        return desired_send_date


def store_lead_email_info(
    lead_sheet: dict, 
    draft_id: str = None,
    scheduled_date: datetime = None,
    body: str = None,
    sequence_num: int = None,
    correlation_id: str = None
) -> None:
    """
    Store all email-related information for a lead in the 'emails' table.

    New logic enforces:
      - No more than 15 emails in any rolling 3-minute window
      - Each new email at least 2 minutes after the previously scheduled one
    """
    if correlation_id is None:
        correlation_id = f"store_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"

    try:
        # Default to 'now + 10 minutes' if no scheduled_date was provided
        if scheduled_date is None:
            scheduled_date = datetime.now() + timedelta(minutes=10)

        # ---- Enforce our scheduling constraints ----
        scheduled_date = find_next_available_timeslot(scheduled_date)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Extract basic lead info
        lead_data = lead_sheet.get("lead_data", {})
        lead_props = lead_data.get("properties", {})

        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        email_address = lead_data.get("email")

        # Insert into emails table with the adjusted 'scheduled_date'
        email_id = store_email_draft(
            cursor,
            lead_id=lead_id,
            name=name,
            email_address=email_address,
            sequence_num=sequence_num,
            body=body,
            scheduled_send_date=scheduled_date,
            draft_id=draft_id,
            status='draft'
        )

        conn.commit()
        logger.info(
            f"[store_lead_email_info] Scheduled email for lead_id={lead_id}, email={email_address}, "
            f"draft_id={draft_id} at {scheduled_date}",
            extra={"correlation_id": correlation_id}
        )

    except Exception as e:
        logger.error(f"Error storing lead email info: {str(e)}", extra={
            "correlation_id": correlation_id
        })
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()




## scripts\build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path
from config.settings import PROJECT_ROOT
from utils.xai_integration import (
    _build_icebreaker_from_news,
    _send_xai_request,
    xai_news_search,
    get_xai_icebreaker
)
from scripts.job_title_categories import categorize_job_title
from datetime import datetime
import re
from typing import Dict, Any
import json

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
SUBJECT_TEMPLATES = [
    "Question about 2025"
]


###############################################################################
# 2) PICK SUBJECT LINE BASED ON LEAD ROLE & LAST INTERACTION
###############################################################################
def pick_subject_line_based_on_lead(profile_type: str, placeholders: dict) -> str:
    """Choose a subject line from SUBJECT_TEMPLATES."""
    try:
        logger.debug("Selecting subject line from simplified templates")
        chosen_template = random.choice(SUBJECT_TEMPLATES)
        logger.debug(f"Selected template: {chosen_template}")

        # Normalize placeholder keys to match template format exactly
        normalized_placeholders = {
            "firstname": placeholders.get("firstname", ""),  # Match exact case
            "LastName": placeholders.get("lastname", ""),
            "companyname": placeholders.get("company_name", ""),
            "company_short_name": placeholders.get("company_short_name", ""),
            # Add other placeholders as needed
        }
        
        # Replace placeholders in the subject
        for key, val in normalized_placeholders.items():
            if val:  # Only replace if value exists
                placeholder = f"[{key}]"
                if placeholder in chosen_template:
                    chosen_template = chosen_template.replace(placeholder, str(val))
                    logger.debug(f"Replaced placeholder {placeholder} with {val}")
                
        if "[" in chosen_template or "]" in chosen_template:
            logger.warning(f"Unreplaced placeholders in subject: {chosen_template}")
            
        return chosen_template
        
    except Exception as e:
        logger.error(f"Error picking subject line: {str(e)}")
        return "Quick Question"  # Safe fallback


###############################################################################
# 3) SEASON VARIATION LOGIC (OPTIONAL)
###############################################################################
def apply_season_variation(email_text: str, snippet: str) -> str:
    """Replaces {SEASON_VARIATION} in an email text with the chosen snippet."""
    logger.debug("Applying season variation:", extra={
        "original_length": len(email_text),
        "snippet_length": len(snippet),
        "has_placeholder": "{SEASON_VARIATION}" in email_text
    })
    
    result = email_text.replace("{SEASON_VARIATION}", snippet)
    
    logger.debug("Season variation applied:", extra={
        "final_length": len(result),
        "successful": result != email_text
    })
    
    return result


###############################################################################
# 4) OPTION: READING AN .MD TEMPLATE (BODY ONLY)
###############################################################################
def extract_subject_and_body(template_content: str) -> tuple[str, str]:
    """
    Extract body from template content, treating the entire content as body.
    Subject will be handled separately via CONDITION_SUBJECTS.
    """
    try:
        # Clean up the template content
        body = template_content.strip()
        
        logger.debug(f"Template content length: {len(template_content)}")
        logger.debug(f"Cleaned body length: {len(body)}")
        
        if len(body) == 0:
            logger.warning("Warning: Template body is empty")
            
        # Return empty subject since it's handled elsewhere
        return "", body
        
    except Exception as e:
        logger.error(f"Error processing template: {e}")
        return "", ""


###############################################################################
# 5) MAIN FUNCTION FOR BUILDING EMAIL
###############################################################################

def generate_icebreaker(has_news: bool, club_name: str, news_text: str = None) -> str:
    """Generate an icebreaker based on news availability."""
    try:
        # Log parameters for debugging
        logger.debug(f"Icebreaker params - has_news: {has_news}, club_name: {club_name}, news_text: {news_text}")
        
        if not club_name.strip():
            logger.debug("No club name provided for icebreaker")
            return ""
            
        # Try news-based icebreaker first
        if has_news and news_text:
            news, icebreaker = xai_news_search(club_name)
            if icebreaker:
                return icebreaker
        
        # Fallback to general icebreaker if no news
        icebreaker = get_xai_icebreaker(
            club_name=club_name,
            recipient_name=""  # Leave blank as we don't have it at this stage
        )
        
        return icebreaker if icebreaker else ""
            
    except Exception as e:
        logger.debug(f"Error in icebreaker generation: {str(e)}")
        return ""

def build_outreach_email(
    template_path: str = None,
    profile_type: str = None,
    placeholders: dict = None,
    current_month: int = 9,
    start_peak_month: int = 5,
    end_peak_month: int = 8,
    use_markdown_template: bool = True
) -> tuple[str, str]:
    """Build email content from template."""
    try:
        placeholders = placeholders or {}
        
        logger.info(f"Building email for {profile_type}")
        
        if template_path and Path(template_path).exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                body = f.read().strip()
            
            # 1. Replace standard placeholders first
            standard_replacements = {
                "[firstname]": placeholders.get("firstname", ""),
                "[LastName]": placeholders.get("LastName", ""),
                "[companyname]": placeholders.get("companyname", ""),
                "[company_short_name]": placeholders.get("company_short_name", ""),
                "[JobTitle]": placeholders.get("JobTitle", ""),
                "[clubname]": placeholders.get("companyname", "")
            }
            
            for key, value in standard_replacements.items():
                if value:
                    body = body.replace(key, str(value))

            # 1. Handle season variation first
            if "[SEASON_VARIATION]" in body:
                season_key = get_season_variation_key(
                    current_month=current_month,
                    start_peak_month=start_peak_month,
                    end_peak_month=end_peak_month
                )
                season_snippet = pick_season_snippet(season_key)
                body = body.replace("[SEASON_VARIATION]", season_snippet)
            
            # 2. Handle icebreaker
            try:
                has_news = placeholders.get('has_news', False)
                news_result = placeholders.get('news_text', '')
                club_name = placeholders.get('clubname', '')
                
                if has_news and news_result and "has not been in the news" not in news_result.lower():
                    icebreaker = _build_icebreaker_from_news(club_name, news_result)
                    if icebreaker:
                        body = body.replace("[ICEBREAKER]", icebreaker)
                    else:
                        body = body.replace("[ICEBREAKER]\n\n", "")
                        body = body.replace("[ICEBREAKER]\n", "")
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    body = body.replace("[ICEBREAKER]\n\n", "")
                    body = body.replace("[ICEBREAKER]\n", "")
                    body = body.replace("[ICEBREAKER]", "")
            except Exception as e:
                logger.error(f"Icebreaker generation error: {e}")
                body = body.replace("[ICEBREAKER]\n\n", "")
                body = body.replace("[ICEBREAKER]\n", "")
                body = body.replace("[ICEBREAKER]", "")
            
            # 3. Clean up multiple newlines
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")
            
            # 4. Replace remaining placeholders
            for key, value in placeholders.items():
                if value:
                    body = body.replace(f"[{key}]", str(value))
            
            # Add Byrdi to Swoop replacement
            body = body.replace("Byrdi", "Swoop")
                      
            # Clean up any double newlines
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")
            
            # Get subject
            subject = pick_subject_line_based_on_lead(profile_type, placeholders)
            
            # Remove signature as it's in the HTML template
            body = body.split("\n\nCheers,")[0].strip()
            
            if body:
                logger.info("Successfully built email template")
            else:
                logger.error("Failed to build email template")
                
            return subject, body
            
    except Exception as e:
        logger.error(f"Error building email: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return "", ""

def get_fallback_template() -> str:
    """Returns a basic fallback template if all other templates fail."""
    return """Connecting About Club Services
---
Hi [firstname],

I wanted to reach out about how we're helping clubs like [clubname] enhance their member experience through our comprehensive platform.

Would you be open to a brief conversation to explore if our solution might be a good fit for your needs?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com"""

def validate_template(template_content: str) -> bool:
    """Validate template format and structure"""
    if not template_content:
        raise ValueError("Template content cannot be empty")
    
    # Basic validation that template has some content
    lines = template_content.strip().split('\n')
    if len(lines) < 2:  # At least need greeting and body
        raise ValueError("Template must have sufficient content")
    
    return True

def build_template(template_path):
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
            
        # Validate template before processing
        validate_template(template_content)
        
        # Rest of the template building logic...
        # ...
    except Exception as e:
        logger.error(f"Error building template from {template_path}: {str(e)}")
        raise

def get_xai_icebreaker(club_name: str, recipient_name: str) -> str:
    """
    Get personalized icebreaker from xAI with proper error handling and debugging
    """
    try:
        # Log the request parameters
        logger.debug(f"Requesting xAI icebreaker for club: {club_name}, recipient: {recipient_name}")
        
        # Create the payload for xAI request
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are writing from Swoop Golf's perspective, reaching out to golf clubs about our technology platform."
                },
                {
                    "role": "user",
                    "content": f"Create a brief, professional icebreaker for {club_name}. Focus on improving their operations and member experience. Keep it concise."
                }
            ],
            "model": "grok-2-1212",
            "stream": False,
            "temperature": 0.1
        }
        
        # Use _send_xai_request directly with recipient email
        response = _send_xai_request(payload, timeout=10)
        
        if not response:
            raise ValueError("Empty response from xAI service")
            
        cleaned_response = response.strip()
        
        if len(cleaned_response) < 10:
            raise ValueError(f"Response too short ({len(cleaned_response)} chars)")
            
        if '[' in cleaned_response or ']' in cleaned_response:
            logger.warning("Response contains unresolved template variables")
        
        if cleaned_response.lower().startswith(('hi', 'hello', 'dear')):
            logger.warning("Response appears to be a full greeting instead of an icebreaker")
            
        return cleaned_response
        
    except Exception as e:
        logger.warning(
            "Failed to get xAI icebreaker",
            extra={
                'error': str(e),
                'club_name': club_name,
                'recipient_name': recipient_name
            }
        )
        return "I wanted to reach out about enhancing your club's operations"  # Fallback icebreaker

def parse_template(template_content):
    """Parse template content - subject lines are handled separately via CONDITION_SUBJECTS"""
    logger.debug(f"Parsing template content of length: {len(template_content)}")
    lines = template_content.strip().split('\n')
    logger.debug(f"Template contains {len(lines)} lines")
    
    # Just return the body content directly
    result = {
        'subject': None,  # Subject will be set from CONDITION_SUBJECTS
        'body': template_content.strip()
    }
     
    logger.debug(f"Parsed template - Body length: {len(result['body'])}")
    return result

def build_email(template_path, parameters):
    """Build email from template and parameters"""
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
    
        template_data = parse_template(template_content)
        
        # Get the body text
        body = template_data['body']
        
        # Add this line to call replace_placeholders
        body = replace_placeholders(body, parameters)
        logger.debug(f"After placeholder replacement - Preview: {body[:200]}")
        
        return body
        
    except Exception as e:
        logger.error(f"Error building email: {str(e)}")
        return ""

def extract_template_body(template_content: str) -> str:
    """Extract body from template content."""
    try:
        # Simply clean up the template content
        body = template_content.strip()
        
        logger.debug(f"Extracted body length: {len(body)}")
        if len(body) == 0:
            logger.warning("Warning: Extracted body is empty")
        
        return body
        
    except Exception as e:
        logger.error(f"Error extracting body: {e}")
        return ""

def process_template(template_path):
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.debug(f"Raw template content length: {len(content)}")
            return extract_template_body(content)
    except Exception as e:
        logger.error(f"Error reading template file: {e}")
        return ""

def get_template_path(club_type: str, role: str) -> str:
    """Get the appropriate template path based on club type and role."""
    try:
        # Normalize inputs
        club_type = club_type.lower().strip().replace(" ", "_")
        role_category = categorize_job_title(role)
        
        # Map role categories to template names
        template_map = {
            "fb_manager": "fb_manager_initial_outreach",
            "membership_director": "membership_director_initial_outreach",
            "golf_operations": "golf_operations_initial_outreach",
            "general_manager": "general_manager_initial_outreach"
        }
        
        # Get template name or default to general manager
        template_name = template_map.get(role_category, "general_manager_initial_outreach")
        
        # Randomly select sequence number (1 or 2)
        sequence_num = random.randint(1, 2)
        logger.debug(f"Selected template variation {sequence_num} for {template_name} (role: {role})")
        
        # Build template path
        template_path = os.path.join(
            PROJECT_ROOT,
            "docs",
            "templates",
            club_type,
            f"{template_name}_{sequence_num}.md"
        )
        
        logger.debug(f"Using template path: {template_path}")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template not found: {template_path}, falling back to general manager template")
            template_path = os.path.join(
                PROJECT_ROOT,
                "docs",
                "templates",
                club_type,
                f"general_manager_initial_outreach_{sequence_num}.md"
            )
        
        return template_path
        
    except Exception as e:
        logger.error(f"Error getting template path: {str(e)}")
        # Fallback to general manager template
        return os.path.join(
            PROJECT_ROOT,
            "docs",
            "templates",
            "country_club",
            f"general_manager_initial_outreach_{sequence_num}.md"
        )

def replace_placeholders(text: str, data: Dict[str, Any]) -> str:
    """Replace placeholders in template with actual values."""
    try:
        # Log incoming data structure
        logger.debug(f"replace_placeholders received data structure: {json.dumps(data, indent=2)}")
        
        # Normalize data structure to handle both nested and flat dictionaries
        lead_data = data.get("lead_data", {}) if isinstance(data.get("lead_data"), dict) else data
        company_data = data.get("company_data", {}) if isinstance(data.get("company_data"), dict) else data
        
        # Get company short name with explicit logging
        company_short_name = company_data.get("company_short_name", "")
        company_full_name = company_data.get("name", "")
        logger.debug(f"Found company_short_name: '{company_short_name}'")
        logger.debug(f"Found company_full_name: '{company_full_name}'")
        
        # Build replacements dict with flexible data structure handling
        replacements = {
            "[firstname]": (lead_data.get("firstname", "") or 
                          lead_data.get("first_name", "") or 
                          data.get("firstname", "")),
            
            "[LastName]": (lead_data.get("lastname", "") or 
                          lead_data.get("last_name", "") or 
                          data.get("lastname", "")),
            
            "[companyname]": (company_data.get("name", "") or 
                             data.get("company_name", "") or 
                             company_full_name),
            
            "[company_short_name]": (company_short_name or 
                                  data.get("company_short_name", "") or 
                                  company_full_name),
            
            "[City]": company_data.get("city", ""),
            "[State]": company_data.get("state", "")
        }
        
        logger.debug(f"Built replacements dictionary: {json.dumps(replacements, indent=2)}")
        
        # Do the replacements with logging
        result = text
        for placeholder, value in replacements.items():
            if placeholder in result:
                logger.debug(f"Replacing '{placeholder}' with '{value}'")
                result = result.replace(placeholder, value)
            else:
                logger.debug(f"Placeholder '{placeholder}' not found in text")
        
        # Check for any remaining placeholders
        remaining = re.findall(r'\[([^\]]+)\]', result)
        if remaining:
            logger.warning(f"Unreplaced placeholders found: {remaining}")
        
        logger.debug(f"Final text preview: {result[:200]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in replace_placeholders: {str(e)}")
        return text



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
