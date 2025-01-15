import logging
import random
from typing import List, Dict, Any
from services.hubspot_service import HubspotService
from services.data_gatherer_service import DataGathererService
from config.settings import HUBSPOT_API_KEY, OPENAI_API_KEY, MODEL_FOR_GENERAL, DEBUG_MODE, PROJECT_ROOT, CLEAR_LOGS_ON_START
from utils.logging_setup import logger, setup_logging
from datetime import datetime, timedelta
import openai
from utils.gmail_integration import create_draft, store_draft_info
from scheduling.database import get_db_connection, store_email_draft
from scheduling.followup_generation import generate_followup_email_xai
from scripts.golf_outreach_strategy import get_best_outreach_window, get_best_month, get_best_time, get_best_day, adjust_send_time
from utils.date_utils import convert_to_club_timezone
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from utils.xai_integration import (
    _build_icebreaker_from_news, 
    get_random_subject_template,
    personalize_email_with_xai
)
from scripts.build_template import build_outreach_email
from pathlib import Path
import os
import shutil
from contextlib import contextmanager
import uuid
from scheduling.extended_lead_storage import store_lead_email_info
from utils.exceptions import LeadContextError
from services.company_enrichment_service import CompanyEnrichmentService
from scripts.job_title_categories import categorize_job_title
from scheduling.sql_lookup import build_lead_sheet_from_sql
from services.leads_service import LeadsService
from scheduling.followup_scheduler import start_scheduler
import threading


COMPANY_FILTERS = [
    {  # 1) Club Type filter - required for this workflow
        "propertyName": "club_type",
        "operator": "EQ",
        "value": ""
    },
    {  # 2) Revenue filter
        "propertyName": "annualrevenue",
        "operator": "GT",
        "value": ""
    },
    {  # 3) State filter - will be populated with best states
        "propertyName": "state",
        "operator": "EQ",
        "value": ""
    },
    {  # 4) Optional filter
        "propertyName": "geographic_seasonality",
        "operator": "EQ",
        "value": ""
    },
    {  # 5) Optional filter
        "propertyName": "facility_complexity",
        "operator": "EQ",
        "value": ""
    }
]

LEAD_FILTERS = [
    {  # 1) Company ID filter - required
        "propertyName": "associatedcompanyid",
        "operator": "EQ",
        "value": ""  # Will be set per company
    },
    {  # 2) Lead Score filter
        "propertyName": "lead_score",
        "operator": "GT",
        "value": "0"
    },
    {  # 3) Email reply date filter - exclude recent replies
        "propertyName": "hs_sales_email_last_replied",
        "operator": "GTE",  # Changed from GTE to LTE
        "value": "2020-01-01"  # Exclude leads who replied after this date
    },
    {  # 4) Optional filter
        "propertyName": "recent_interaction",
        "operator": "HAS_PROPERTY",
        "value": ""
    },
    {  # 5) Optional filter
        "propertyName": "email_domain",
        "operator": "CONTAINS",
        "value": ""
    }
]

# Number of leads to process
LEADS_TO_PROCESS = 25

# Initialize logging and services
setup_logging()
data_gatherer = DataGathererService()

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
        active_filters = [f for f in LEAD_FILTERS if f.get("value") not in [None, "", []]]
        # Always add company ID filter
        company_filter = {
            "propertyName": "associatedcompanyid",
            "operator": "EQ",
            "value": company_id
        }
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
        
        # Double-check sorting by lead_score
        sorted_results = sorted(
            results,
            key=lambda x: float(x.get("properties", {}).get("lead_score", "0") or "0"),
            reverse=True
        )
        
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
        if news_result:
            if isinstance(news_result, tuple):
                news_text = news_result[0]
            else:
                news_text = str(news_result)
            has_news = "has not been" not in news_text.lower()
        
        return {
            "has_news": has_news,
            "news_text": news_result if has_news else None
        }
        
    except Exception as e:
        logger.error(f"Error gathering personalization data: {str(e)}")
        return {
            "has_news": False,
            "news_text": None
        }


def summarize_lead_interactions(lead_sheet: dict) -> str:
    """
    Summarize prior emails and notes from the lead_sheet using OpenAI.
    """
    try:
        lead_data = lead_sheet.get("lead_data", {})
        emails = lead_data.get("emails", [])
        notes = lead_data.get("notes", [])
        
        interactions = []
        
        # Collect emails
        for email in sorted(emails, key=lambda x: x.get('timestamp', ''), reverse=True):
            if isinstance(email, dict):
                date = email.get('timestamp', '').split('T')[0]
                subject = email.get('subject', '').encode('utf-8', errors='ignore').decode('utf-8')
                body = email.get('body_text', '').encode('utf-8', errors='ignore').decode('utf-8')
                direction = email.get('direction', '')
                body = body.split('On ')[0].strip()
                email_type = "from the lead" if direction == "INCOMING_EMAIL" else "to the lead"
                
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
                content = note.get('body', '').encode('utf-8', errors='ignore').decode('utf-8')
                
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
            if DEBUG_MODE:
                logger.info(f"Interaction Summary:\n{summary}")
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
        
        # Use the predefined LEAD_FILTERS
        active_filters = []
        for filter_def in LEAD_FILTERS:
            if filter_def["propertyName"] == "lead_score":
                filter_def["value"] = str(min_score)
            if filter_def.get("value"):  # Only include filters with values
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
                    "direction": "DESCENDING"
                }
            ]
        }
        response = hubspot._make_hubspot_post(url, payload)
        
        # Add debug logging
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
    """Replace all placeholders in the text with actual values."""
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
        if value:  # Only replace if we have a value
            result = result.replace(placeholder, value)
    return result


def check_lead_filters(lead_data: dict) -> bool:
    """Check if lead meets all filter criteria."""
    logger.debug(f"Checking email reply date for {lead_data['email']}")
    last_reply = lead_data.get('hs_sales_email_last_replied')
    if last_reply:
        logger.debug(f"Last reply date: {last_reply}")
    else:
        logger.debug("No previous reply found")
    
    # ... rest of the filter checks ...


def main():
    """
    Restructured main function:
    - 1) Get leads first (rather than companies)
    - 2) For each lead, retrieve its associated company
    - 3) Check if that company passes the workflow filters
    - 4) Then proceed with the same core processing steps
    """
    try:
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
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
                
                # 2) Enrich the company data FIRST
                enrichment_result = company_enricher.enrich_company(company_id)
                if not enrichment_result.get("success", False):
                    logger.warning(f"Enrichment failed for company {company_id}, skipping.")
                    continue
                
                # Update company_props with enriched data
                company_props.update(enrichment_result.get("data", {}))
                
                # 3) Now check if this company meets the workflow's COMPANY_FILTERS 
                active_company_filters = [f for f in COMPANY_FILTERS if f.get("value")]
                logger.debug(f"\nChecking filters for {company_props.get('name', 'Unknown Company')} (ID: {company_id})")
                logger.debug("----------------------------------------")
                
                meets_filters = True
                for f in active_company_filters:
                    prop_name = f["propertyName"]
                    operator = f["operator"]
                    filter_value = f["value"]
                    company_value = company_props.get(prop_name, "")
                    
                    # Add more detailed logging
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
                            if float(company_value) <= float(filter_value):
                                logger.info(f"FAILED: Value too low")
                                meets_filters = False
                                break
                        except ValueError:
                            logger.info(f"FAILED: Invalid numeric value")
                            meets_filters = False
                            break
                    elif operator == "LTE":  # Add specific logging for email reply date
                        if company_value and company_value > filter_value:
                            logger.info(f"FAILED: Reply date too recent")
                            meets_filters = False
                            break
                    
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
                
                # 5) Now process the lead with the standard steps
                with workflow_step("3", f"Processing lead {lead_id}", workflow_context):
                    try:
                        # Extract combined lead+company data
                        lead_data = extract_lead_data(company_props, lead_props)
                        
                        # Skip if club name contains "Unknown"
                        if "unknown" in lead_data["company_data"]["name"].lower():
                            logger.info(f"Skipping lead {lead_id} - Club name contains 'Unknown'")
                            continue
                        
                        # Skip if club type is "Unknown"
                        if lead_data["company_data"]["club_type"] == "Unknown":
                            logger.info(f"Skipping lead {lead_id} - Club type is Unknown")
                            continue
                            
                        # Instead of SQL summary, use HubSpot data for interaction summary
                        interaction_summary = lead_props.get("recent_interaction", "")
                        
                        # Gather personalization data
                        personalization = gather_personalization_data(
                            company_name=lead_data["company_data"]["name"],
                            city=lead_data["company_data"]["city"],
                            state=lead_data["company_data"]["state"]
                        )
                        
                        # Get correct template
                        template_path = get_template_path(
                            club_type=lead_data["company_data"]["club_type"],
                            role=lead_data["lead_data"]["jobtitle"],
                            sequence_num=1
                        )
                        
                        # Calculate send date
                        send_date = calculate_send_date(
                            geography=lead_data["company_data"]["geographic_seasonality"],
                            persona=lead_data["lead_data"]["jobtitle"],
                            state_code=lead_data["company_data"]["state"],
                            season_data={
                                "peak_season_start": lead_data["company_data"].get("peak_season_start_month"),
                                "peak_season_end": lead_data["company_data"].get("peak_season_end_month")
                            }
                        )
                        
                        # Build outreach email content
                        email_content = build_outreach_email(
                            template_path=template_path,
                            profile_type=lead_data["lead_data"]["jobtitle"],
                            placeholders={
                                "company_name": lead_data["company_data"]["name"],
                                "first_name": lead_data["lead_data"]["firstname"],
                                "last_name": lead_data["lead_data"]["lastname"],
                                "job_title": lead_data["lead_data"]["jobtitle"],
                                "company_info": lead_data["company_data"].get("club_info", "")
                            },
                            current_month=datetime.now().month,
                            start_peak_month=lead_data["company_data"].get("peak_season_start_month"),
                            end_peak_month=lead_data["company_data"].get("peak_season_end_month")
                        )
                        
                        if email_content:
                            # Replace placeholders in subject and body
                            email_content["subject"] = replace_placeholders(email_content["subject"], lead_data)
                            email_content["body"] = replace_placeholders(email_content["body"], lead_data)
                            
                            # Further personalize with XAI
                            personalized_content = personalize_email_with_xai(
                                lead_sheet={
                                    "lead_data": lead_data["lead_data"],
                                    "company_data": lead_data["company_data"],
                                    "analysis": {"previous_interactions": interaction_summary}
                                },
                                subject=email_content["subject"],
                                body=email_content["body"],
                                summary=interaction_summary
                            )
                            if personalized_content:
                                email_content.update(personalized_content)
                            
                            # Insert icebreaker if we have recent news
                            if personalization.get("has_news") and personalization.get("news_text"):
                                icebreaker = _build_icebreaker_from_news(
                                    lead_data["company_data"]["name"],
                                    personalization["news_text"]
                                )
                                if icebreaker:
                                    email_content["body"] = email_content["body"].replace(
                                        "[ICEBREAKER]", 
                                        icebreaker
                                    )
                            
                            # Clean up leftover placeholders or extra newlines
                            email_content["body"] = email_content["body"].replace("[ICEBREAKER]\n\n", "")
                            email_content["body"] = email_content["body"].replace("[ICEBREAKER]\n", "")
                            email_content["body"] = email_content["body"].replace("[ICEBREAKER]", "")
                            while "\n\n\n" in email_content["body"]:
                                email_content["body"] = email_content["body"].replace("\n\n\n", "\n\n")
                            
                            # Create the draft
                            draft_result = create_draft(
                                sender="me",
                                to=lead_data["lead_data"]["email"],
                                subject=email_content["subject"],
                                message_text=email_content["body"]
                            )
                            
                            if draft_result["status"] == "ok":
                                # Store the draft data
                                store_lead_email_info(
                                    lead_sheet={
                                        "lead_data": {
                                            "email": lead_data["lead_data"]["email"],
                                            "properties": {
                                                "hs_object_id": lead_id,
                                                "firstname": lead_data["lead_data"]["firstname"],
                                                "lastname": lead_data["lead_data"]["lastname"]
                                            }
                                        },
                                        "company_data": {
                                            "name": lead_data["company_data"]["name"],
                                            "city": lead_data["company_data"]["city"],
                                            "state": lead_data["company_data"]["state"],
                                            "company_type": lead_data["company_data"]["club_type"]
                                        }
                                    },
                                    draft_id=draft_result["draft_id"],
                                    scheduled_date=send_date,
                                    subject=email_content["subject"],
                                    body=email_content["body"],
                                    sequence_num=1
                                )
                                
                                logger.info(f"Created draft email for {lead_data['lead_data']['email']}")
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
        logger.error(f"Error in main workflow: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Start scheduler in background
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Run main
    main()
