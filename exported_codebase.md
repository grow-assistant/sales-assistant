
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
    is_dst = datetime.now().astimezone().dst() != timedelta(0)
    offset_hours = offsets['dst'] if is_dst else offsets['std']
    
    # Apply offset
    adjusted_time = send_time + timedelta(hours=offset_hours)
    logger.debug(f"Adjusted time from {send_time} to {adjusted_time} for state {state_code} (offset: {offset_hours}h)")
    return adjusted_time

def get_best_month(geography: str, club_type: str = None, season_data: dict = None) -> list:
    """
    Determine best outreach months based on geography/season and club type.
    """
    current_month = datetime.now().month
    
    # If we have season data, use it as primary decision factor
    if season_data:
        peak_start = season_data.get('peak_season_start', '')
        peak_end = season_data.get('peak_season_end', '')
        
        if peak_start and peak_end:
            peak_start_month = int(peak_start.split('-')[0])
            peak_end_month = int(peak_end.split('-')[0])
            
            logger.debug(f"Peak season: {peak_start_month} to {peak_end_month}")
            
            # For winter peak season (crossing year boundary)
            if peak_start_month > peak_end_month:
                if current_month >= peak_start_month or current_month <= peak_end_month:
                    # We're in peak season, target shoulder season
                    return [9]  # September (before peak starts)
                else:
                    # We're in shoulder season
                    return [1]  # January
            # For summer peak season
            else:
                if peak_start_month <= current_month <= peak_end_month:
                    # We're in peak season, target shoulder season
                    return [peak_start_month - 1] if peak_start_month > 1 else [12]
                else:
                    # We're in shoulder season
                    return [1]  # January
    
    # Fallback to geography-based matrix
    month_matrix = {
        "Year-Round Golf": [1, 9],      # January or September
        "Peak Winter Season": [9],       # September
        "Peak Summer Season": [2],       # February
        "Short Summer Season": [1],      # January
        "Shoulder Season Focus": [2, 10]  # February or October
    }
    
    return month_matrix.get(geography, [1, 9])

def get_best_time(persona: str) -> dict:
    """
    Determine best time of day based on persona.
    Returns a dict with start and end hours/minutes in 24-hour format.
    Randomly selects between morning and afternoon windows.
    """
    time_windows = {
        "General Manager": [
            {
                "start_hour": 8, "start_minute": 30,
                "end_hour": 10, "end_minute": 30
            },  # 8:30-10:30 AM
            {
                "start_hour": 15, "start_minute": 0,
                "end_hour": 16, "end_minute": 30
            }   # 3:00-4:30 PM
        ],
        "Food & Beverage Director": [
            {
                "start_hour": 9, "start_minute": 30,
                "end_hour": 11, "end_minute": 30
            },  # 9:30-11:30 AM
            {
                "start_hour": 15, "start_minute": 0,
                "end_hour": 16, "end_minute": 30
            }   # 3:00-4:30 PM
        ],
        "Golf Professional": [
            {
                "start_hour": 8, "start_minute": 0,
                "end_hour": 10, "end_minute": 0
            }   # 8:00-10:00 AM
        ],
        "Membership Director": [
            {
                "start_hour": 13, "start_minute": 0,
                "end_hour": 15, "end_minute": 0
            }   # 1:00-3:00 PM
        ]
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    
    # Get time windows for the persona, defaulting to GM times if not found
    windows = time_windows.get(persona, time_windows["General Manager"])
    
    # Randomly select between morning and afternoon windows if multiple exist
    selected_window = random.choice(windows)
    
    # Update calculate_send_date function expects start/end format
    return {
        "start": selected_window["start_hour"] + selected_window["start_minute"] / 60,
        "end": selected_window["end_hour"] + selected_window["end_minute"] / 60
    }

def get_best_outreach_window(persona: str, geography: str, club_type: str = None, season_data: dict = None) -> Dict[str, Any]:
    """Get the optimal outreach window based on persona and geography."""
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona)
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

def calculate_send_date(geography: str, persona: str, state: str, season_data: dict = None) -> datetime:
    """Calculate the next appropriate send date based on outreach window."""
    outreach_window = get_best_outreach_window(geography, persona, season_data=season_data)
    best_months = outreach_window["Best Month"]
    preferred_time = outreach_window["Best Time"]
    preferred_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    # Find the next preferred day of week
    today = datetime.now().date()
    days_ahead = [(day - today.weekday()) % 7 for day in preferred_days]
    next_preferred_day = min(days_ahead)
    
    # Adjust to next month if needed
    if today.month not in best_months:
        target_month = min(best_months)
        if today.month > target_month:
            target_year = today.year + 1
        else:
            target_year = today.year
        target_date = datetime(target_year, target_month, 1)
    else:
        target_date = today + timedelta(days=next_preferred_day)
    
    # Apply preferred time
    target_date = target_date.replace(hour=preferred_time["start"])
    
    # Adjust for state timezone
    return adjust_send_time(target_date, state)

