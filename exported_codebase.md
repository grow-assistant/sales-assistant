
## main.py

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
from scheduling.database import get_db_connection
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


# Initialize logging and services
setup_logging()
data_gatherer = DataGathererService()

@contextmanager
def workflow_step(step_name: str, step_description: str, logger_context: dict = None):
    """Context manager for logging workflow steps."""
    # Initialize empty dict if None provided
    logger_context = logger_context or {}
    
    # Add step info to context
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
        # Clear logs directory
        logs_dir = Path(PROJECT_ROOT) / "logs"
        if logs_dir.exists():
            for file in logs_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.debug("Cleared logs directory")
            
        # Clear temp directory
        temp_dir = Path(PROJECT_ROOT) / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            logger.debug("Cleared temp directory")
            
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}")

def get_country_club_companies(hubspot: HubspotService, batch_size=25) -> List[Dict[str, Any]]:
    """Search for Country Club type companies in HubSpot."""
    logger.debug("Searching for Country Club type companies")
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    all_results = []
    after = None
    
    while True:
        payload = {
            "limit": batch_size,
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
                "club_info"
            ],
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "club_type",
                            "operator": "EQ",
                            "value": "Country Club"
                        }
                    ]
                }
            ]
        }
        
        if after:
            payload["after"] = after
            
        try:
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break
                
            results = response.get("results", [])
            all_results.extend(results)
            
            logger.info(f"Retrieved {len(all_results)} Country Clubs so far")
            
            # Handle pagination
            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link
            
        except Exception as e:
            logger.error(f"Error fetching Country Clubs from HubSpot: {str(e)}")
            break
    
    return all_results

def get_leads_for_company(hubspot: HubspotService, company_id: str) -> List[Dict]:
    """Get all leads/contacts associated with a company with score > 0."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/contacts/search"
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "associatedcompanyid",
                            "operator": "EQ",
                            "value": company_id
                        },
                        {
                            "propertyName": "lead_score",
                            "operator": "GT",
                            "value": "0"
                        }
                    ]
                }
            ],
            "properties": ["email", "firstname", "lastname", "jobtitle", "lead_score"],
            "limit": 100
        }
        response = hubspot._make_hubspot_post(url, payload)
        results = response.get("results", [])
        
        # Sort results by lead score (highest to lowest)
        sorted_results = sorted(
            results,
            key=lambda x: float(x.get("properties", {}).get("lead_score", "0") or "0"),
            reverse=True
        )
        
        return sorted_results
    except Exception as e:
        logger.error(f"Error getting leads for company {company_id}: {e}")
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
            "club_info": company_props.get("club_info", "")
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
        # Only get news, skip club_info
        news_result = data_gatherer.gather_club_news(company_name)
        
        # Check if we have valid news
        has_news = False
        if news_result:
            if isinstance(news_result, tuple):
                news_text = news_result[0]
            else:
                news_text = str(news_result)
            has_news = "has not been" not in news_text.lower()
        
        return {
            "news": news_result if has_news else None,
            "has_news": has_news
        }
    except Exception as e:
        logger.error(f"Error gathering personalization data: {e}")
        return {
            "news": None,
            "has_news": False
        }

def summarize_lead_interactions(lead_sheet: dict) -> str:
    """
    Collects all prior emails and notes from the lead_sheet,
    then sends them to OpenAI to request a concise summary.
    """
    try:
        # 1) Grab prior emails and notes from the lead_sheet
        lead_data = lead_sheet.get("lead_data", {})
        emails = lead_data.get("emails", [])
        notes = lead_data.get("notes", [])
        
        # 2) Format the interactions for OpenAI
        interactions = []
        
        # Add emails with proper encoding handling and sort by timestamp
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
        
        # Add notes
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
                    {"role": "system", "content": "You are a helpful assistant that summarizes business interactions. Anything from Ty or Ryan is from Swoop."},
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
    """Schedule a follow-up email"""
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
        outreach_window = get_best_outreach_window(
            persona=persona,
            geography=geography,
            season_data=season_data
        )
        
        best_months = outreach_window["Best Month"]
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        now = datetime.now()
        target_date = now + timedelta(days=1)
        
        while target_date.month not in best_months:
            if target_date.month == 12:
                target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
            else:
                target_date = target_date.replace(month=target_date.month + 1, day=1)
        
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
    
    Args:
        club_type: Type of club from HubSpot
        role: Role of recipient
        sequence_num: Template sequence number (1-3)
        
    Returns:
        str: Path to the template file
    """
    # Normalize club type to directory name format
    club_type_map = {
        # Primary Types
        "Country Club": "country_club",
        "Private Course": "private_course",
        "Private Club": "private_course",
        "Resort Course": "resort_course",
        "Resort": "resort_course",
        
        # Public Course Variations
        "Public Course": "public_high_daily_fee",
        "Public - High Daily Fee": "public_high_daily_fee",
        "Public - Low Daily Fee": "public_low_daily_fee",
        "Public": "public_high_daily_fee",
        
        # Semi-Private Variations
        "Semi-Private": "public_high_daily_fee",
        "Semi Private": "public_high_daily_fee",
        
        # Municipal Variations
        "Municipal": "public_low_daily_fee",
        "Municipal Course": "public_low_daily_fee",
        
        # Management Companies
        "Management Company": "management_companies",
        
        # Unknown/Default
        "Unknown": "country_club",  # Default to country club templates
    }
    
    # Normalize role to template name format
    role_map = {
        # General Manager variations
        "General Manager": "general_manager",
        "GM": "general_manager",
        "Club Manager": "general_manager",
        
        # Golf Professional variations
        "Golf Professional": "golf_ops",
        "Head Golf Professional": "golf_ops",
        "Director of Golf": "golf_ops",
        "Golf Operations": "golf_ops",
        "Golf Operations Manager": "golf_ops",
        
        # F&B Manager variations
        "F&B Manager": "fb_manager",
        "Food & Beverage Manager": "fb_manager",
        "Food and Beverage Manager": "fb_manager",
        "F&B Director": "fb_manager",
        "Food & Beverage Director": "fb_manager",
        "Food and Beverage Director": "fb_manager",
        "Restaurant Manager": "fb_manager",
    }
    
    # Get normalized club type and role
    normalized_club_type = club_type_map.get(club_type, "country_club")
    normalized_role = role_map.get(role, "general_manager")
    
    # Build template path
    template_path = Path(PROJECT_ROOT) / "docs" / "templates" / normalized_club_type / f"{normalized_role}_initial_outreach_{sequence_num}.md"
    
    # Fallback to default template if specific one doesn't exist
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

def get_high_scoring_leads(hubspot: HubspotService, min_score: float = 0) -> List[Dict]:
    """Get all leads with scores above minimum threshold."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/contacts/search"
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "lead_score",
                            "operator": "GT",
                            "value": str(min_score)
                        }
                    ]
                }
            ],
            "properties": [
                "email", 
                "firstname", 
                "lastname", 
                "jobtitle", 
                "lead_score",
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
        return response.get("results", [])
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
                "club_info"
            ]
        }
        response = hubspot._make_hubspot_get(url, params)
        return response
    except Exception as e:
        logger.error(f"Error getting company {company_id}: {e}")
        return {}

def main():
    """Main function to get companies and select a random lead."""
    try:
        # Create a context dictionary that will be used across all steps
        workflow_context = {
            'correlation_id': str(uuid.uuid4())
        }
        
        # Number of leads to process
        LEADS_TO_PROCESS = 2
        leads_processed = 0

        # Initialize HubSpot service
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get high scoring leads first
        print("Fetching high scoring leads from HubSpot...")
        leads = get_high_scoring_leads(hubspot, min_score=0)
        
        if not leads:
            print("No qualified leads found!")
            return
            
        # Process leads in order of score (highest first)
        for lead in leads:
            if leads_processed >= LEADS_TO_PROCESS:
                print(f"\nCompleted processing {LEADS_TO_PROCESS} leads")
                break

            lead_props = lead.get("properties", {})
            company_id = lead_props.get("associatedcompanyid")
            
            if not company_id:
                continue
                
            # Get company details
            company = get_company_by_id(hubspot, company_id)
            if not company:
                continue
                
            company_props = company.get("properties", {})
            
            # Check if company meets criteria (e.g., is a Country Club)
            if company_props.get("club_type") != "Country Club":
                continue
                
            # Update context with lead info
            workflow_context.update({
                'lead_id': lead.get('id'),
                'company_name': company_props.get('name', 'Unknown')
            })
            
            print(f"\nFound qualified lead at: {company_props.get('name', 'Unknown')}")
            print(f"Lead Score: {lead_props.get('lead_score', '0')}")
            
            # Extract lead data
            print("\nExtracting lead data...")
            lead_data = {
                "firstname": lead_props.get("firstname", ""),
                "lastname": lead_props.get("lastname", ""),
                "jobtitle": lead_props.get("jobtitle", "General Manager"),
                "email": lead_props.get("email", ""),
                "lead_score": lead_props.get("lead_score", "0")
            }
            
            # Gather personalization data
            print("Gathering personalization data...")
            personalization_data = gather_personalization_data(
                company_props.get("name", ""),
                company_props.get("city", ""),
                company_props.get("state", "")
            )
            
            # Summarize interactions
            print("Summarizing interactions...")
            interaction_summary = summarize_lead_interactions({
                "lead_data": lead_data,
                "company_data": company_props
            })
            print(f"\nInteraction Summary:\n{interaction_summary}")
            
            # Step 8: Build initial outreach email
            with workflow_step(8, "Building email draft", workflow_context):
                # Get template path using enhanced function
                template_path = get_template_path(
                    club_type=company_props.get("club_type", "Country Club"),
                    role=lead_props.get("jobtitle", "General Manager")
                )
                logger.debug(f"Using template path: {template_path}")
                
                # Get subject and initial body
                subject = get_random_subject_template()
                
                # Create placeholders
                placeholders = {
                    "FirstName": lead_data["firstname"],
                    "LastName": lead_data["lastname"],
                    "ClubName": company_props.get("name", "Your Club"),
                    "Role": lead_data["jobtitle"],
                    "SEASON_VARIATION": pick_season_snippet(get_season_variation_key(
                        current_month=datetime.now().month,
                        start_peak_month=5,
                        end_peak_month=8
                    )).rstrip(',')
                }
                
                # Replace placeholders in subject
                for key, val in placeholders.items():
                    subject = subject.replace(f"[{key}]", val)
                
                # Build email body using enhanced template selection
                _, body = build_outreach_email(
                    profile_type="general_manager",
                    last_interaction_days=0,
                    placeholders=placeholders,
                    current_month=datetime.now().month,
                    start_peak_month=5,
                    end_peak_month=8,
                    use_markdown_template=True,
                    template_path=template_path
                )
                
                # Store original content for fallback
                orig_subject = subject
                orig_body = body
                
                # Enhanced personalization with xAI
                try:
                    subject, body = personalize_email_with_xai(
                        lead_sheet={
                            "lead_data": lead_data,
                            "company_data": company_props
                        },
                        subject=subject,
                        body=body,
                        summary=interaction_summary,
                        news_summary=personalization_data.get('news'),
                        club_info=company_props.get('club_info', '')
                    )
                    
                    # Fallback to original if xAI fails
                    if not subject.strip():
                        subject = orig_subject
                    if not body.strip():
                        body = orig_body
                        
                except Exception as e:
                    logger.error(f"xAI personalization error: {e}")
                    subject, body = orig_subject, orig_body
                    
                # Calculate optimal send date
                send_date = calculate_send_date(
                    geography=company_props.get('geographic_seasonality', 'Year-Round Golf'),
                    persona=lead_data["jobtitle"],
                    state_code=company_props.get('state'),
                    season_data=None
                )

            # After successful processing, increment counter
            leads_processed += 1
            print(f"\nProcessed {leads_processed} of {LEADS_TO_PROCESS} leads")
            
        if leads_processed == 0:
            print("\nNo qualified companies found for high-scoring leads!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    

    main()




## main_old.py

import csv
import logging
from pathlib import Path
import os
import shutil
import random
from datetime import timedelta, datetime
import json
from typing import List, Dict, Any

from utils.logging_setup import logger, workflow_step, setup_logging
from utils.exceptions import LeadContextError
from utils.xai_integration import (
    personalize_email_with_xai,
    _build_icebreaker_from_news,
    get_email_critique,
    revise_email_with_critique,
    _send_xai_request,
    MODEL_NAME,
    get_random_subject_template
)
from utils.gmail_integration import create_draft, store_draft_info
from scripts.build_template import build_outreach_email
from scripts.job_title_categories import categorize_job_title
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY, PROJECT_ROOT, CLEAR_LOGS_ON_START, USE_RANDOM_LEAD, TEST_EMAIL, CREATE_FOLLOWUP_DRAFT, USE_LEADS_LIST
from scheduling.extended_lead_storage import upsert_full_lead
from scheduling.followup_generation import generate_followup_email_xai
from scheduling.sql_lookup import build_lead_sheet_from_sql
from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
import openai
from config.settings import OPENAI_API_KEY, DEFAULT_TEMPERATURE, MODEL_FOR_GENERAL
import uuid
from scheduling.database import get_db_connection
from scripts.golf_outreach_strategy import get_best_outreach_window, get_best_month, get_best_time, get_best_day, adjust_send_time
from utils.date_utils import convert_to_club_timezone
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from services.hubspot_service import HubspotService

print(f"USE_RANDOM_LEAD: {os.getenv('USE_RANDOM_LEAD')}")

# Initialize logging before creating DataGathererService instance
setup_logging()

# Initialize services
data_gatherer = DataGathererService()
leads_service = LeadsService(data_gatherer)

TIMEZONE_CSV_PATH = "docs/data/state_timezones.csv"

def load_state_timezones():
    """Load state timezones from CSV file."""
    state_timezones = {}
    with open(TIMEZONE_CSV_PATH, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state_code = row['state_code'].strip()
            state_timezones[state_code] = {
                'dst': int(row['daylight_savings']),
                'std': int(row['standard_time'])
            }
    logger.debug(f"Loaded timezone data for {len(state_timezones)} states")
    return state_timezones

STATE_TIMEZONES = load_state_timezones()

###############################################################################
# Summarize lead interactions
###############################################################################
def summarize_lead_interactions(lead_sheet: dict) -> str:
    """
    Collects all prior emails and notes from the lead_sheet,
    then sends them to OpenAI to request a concise summary.
    """
    try:
        # 1) Grab prior emails and notes from the lead_sheet
        lead_data = lead_sheet.get("lead_data", {})
        emails = lead_data.get("emails", [])
        notes = lead_data.get("notes", [])
        
        # 2) Format the interactions for OpenAI
        interactions = []
        
        # Add emails with proper encoding handling and sort by timestamp
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
        
        # Add notes
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
                    {"role": "system", "content": "You are a helpful assistant that summarizes business interactions. Anything from Ty or Ryan is from Swoop."},
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

###############################################################################
# Main Workflow
###############################################################################
def get_country_club_companies(hubspot: HubspotService, batch_size=25) -> List[Dict[str, Any]]:
    """
    Search for Country Club type companies in HubSpot.
    """
    logger.debug("Searching for Country Club type companies")
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    all_results = []
    after = None
    
    while True:
        payload = {
            "limit": batch_size,
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
                "public_private_flag"
            ],
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "club_type",
                            "operator": "EQ",
                            "value": "Country Club"
                        }
                    ]
                }
            ]
        }
        
        if after:
            payload["after"] = after
            
        try:
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break
                
            results = response.get("results", [])
            all_results.extend(results)
            
            logger.debug(f"Retrieved {len(all_results)} Country Clubs so far")
            
            # Handle pagination
            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link
            
        except Exception as e:
            logger.error(f"Error fetching Country Clubs from HubSpot: {str(e)}")
            break
    
    logger.info(f"Found {len(all_results)} total Country Clubs")
    return all_results

def get_leads_for_company(hubspot: HubspotService, company_id: str) -> List[Dict]:
    """Get all leads/contacts associated with a company."""
    try:
        url = f"{hubspot.base_url}/crm/v3/objects/contacts/search"
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "associatedcompanyid",
                            "operator": "EQ",
                            "value": company_id
                        }
                    ]
                }
            ],
            "properties": ["email", "firstname", "lastname", "jobtitle"],
            "limit": 100
        }
        response = hubspot._make_hubspot_post(url, payload)
        return response.get("results", [])
    except Exception as e:
        logger.error(f"Error getting leads for company {company_id}: {e}")
        return []

def get_random_lead_email() -> str:
    """
    Get a random lead email from "Country Club" companies in HubSpot.
    Removed the additional 'annualrevenue' filter so we only check 
    `club_type = "Country Club"`.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
        
        # Search for Country Clubs specifically (NO annual revenue filter).
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "club_type",
                            "operator": "EQ",
                            "value": "Country Club"
                        }
                    ]
                }
            ],
            "properties": [
                "name",
                "city",
                "state",
                "club_type",
                "annualrevenue",
                "facility_complexity",
                "geographic_seasonality",
                "has_pool",
                "has_tennis_courts",
                "number_of_holes",
                "public_private_flag"
            ],
            "limit": 100
        }
        
        logger.debug("Searching for Country Club companies in HubSpot (no annualrevenue filter)")
        response = hubspot._make_hubspot_post(url, payload)
        
        if not response or not response.get("results"):
            logger.warning("No Country Clubs found, falling back to TEST_EMAIL")
            return TEST_EMAIL
            
        # Filter for clubs with valid contacts/leads
        valid_clubs = []
        for company in response.get("results", []):
            company_id = company.get("id")
            leads = get_leads_for_company(hubspot, company_id)
            if leads:
                valid_clubs.append((company, leads))
        
        if not valid_clubs:
            logger.warning("No Country Clubs with leads found, falling back to TEST_EMAIL")
            return TEST_EMAIL
            
        # Randomly select a club and lead
        company, leads = random.choice(valid_clubs)
        lead = random.choice(leads)
        email = lead.get("email")
        
        company_name = company.get("properties", {}).get("name", "Unknown Club")
        company_state = company.get("properties", {}).get("state", "Unknown State")
        
        if email:
            logger.info(f"Selected lead ({email}) from Country Club: {company_name} in {company_state}")
            return email
        else:
            logger.warning("Selected lead has no email, falling back to TEST_EMAIL")
            return TEST_EMAIL
            
    except Exception as e:
        logger.error(f"Error getting random Country Club lead: {e}")
        return TEST_EMAIL

def get_lead_email() -> str:
    """Get lead email from HubSpot country clubs."""
    logger.debug(f"USE_RANDOM_LEAD setting is: {USE_RANDOM_LEAD}")
    logger.debug(f"TEST_EMAIL setting is: {TEST_EMAIL}")
    
    if TEST_EMAIL:
        logger.debug("Using TEST_EMAIL setting")
        return TEST_EMAIL
        
    logger.debug("Getting random Country Club lead from HubSpot")
    return get_random_lead_email()

def main():
    """Main entry point for the sales assistant application."""
    correlation_id = str(uuid.uuid4())
    logger_context = {"correlation_id": correlation_id}
    
    # Force debug logging at start
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting main workflow", extra=logger_context)
    logger.debug(f"USE_RANDOM_LEAD is set to: {USE_RANDOM_LEAD}")
    
    try:
        # Step 1: Get lead email
        with workflow_step(1, "Getting lead email"):
            email = get_lead_email()
            if not email:
                logger.error("No email provided; exiting.", extra=logger_context)
                return
            logger.info(f"Using email: {email}")
            logger_context["email"] = email

        # Step 2: External data gathering
        with workflow_step(2, "Gathering external data"):
            lead_sheet = data_gatherer.gather_lead_data(email, correlation_id=correlation_id)
            if not lead_sheet.get("metadata", {}).get("status") == "success":
                logger.error("Failed to gather lead data", extra=logger_context)
                return

        # Step 3: Database operations
        with workflow_step(3, "Saving to database"):
            try:
                logger.debug("Starting database upsert")
                email_count = len(lead_sheet.get("lead_data", {}).get("emails", []))
                logger.debug(f"Found {email_count} emails to store")
                
                upsert_full_lead(lead_sheet)
                
                logger.info(f"Successfully stored lead data including {email_count} emails")
            except Exception as e:
                logger.error("Database operation failed", 
                            extra={**logger_context, "error": str(e)}, 
                            exc_info=True)
                raise

        # Verify lead_sheet success
        if lead_sheet.get("metadata", {}).get("status") != "success":
            logger.error("Failed to prepare or retrieve lead context. Exiting.", extra={
                "email": email,
                "correlation_id": correlation_id
            })
            return

        # Step 4: Prepare lead context
        with workflow_step(4, "Preparing lead context"):
            emails = lead_sheet.get("lead_data", {}).get("emails", [])
            logger.debug(f"Found {len(emails)} emails in lead sheet", extra={
                "email_count": len(emails),
                "email_subjects": [e.get('subject', 'No subject') for e in emails]
            })
            
            lead_context = leads_service.prepare_lead_context(
                email, 
                lead_sheet=lead_sheet,
                correlation_id=correlation_id
            )

        # Step 5: Extract relevant data (updated for logging new fields)
        with workflow_step(5, "Extracting lead data"):
            lead_data = lead_sheet.get("lead_data", {})
            company_data = lead_data.get("company_data", {})

            # Safely extract data
            first_name = (lead_data.get("properties", {}).get("firstname") or "").strip()
            last_name = (lead_data.get("properties", {}).get("lastname") or "").strip()
            
            # *** NEW *** Pull in all 15 fields explicitly
            name = company_data.get("name", "")
            city = company_data.get("city", "")
            state = company_data.get("state", "")
            annual_revenue = company_data.get("annualrevenue", "")
            created_date = company_data.get("createdate", "")
            last_modified = company_data.get("hs_lastmodifieddate", "")
            object_id = company_data.get("hs_object_id", "")
            club_type = company_data.get("club_type", "")
            facility_complexity = company_data.get("facility_complexity", "")
            has_pool = company_data.get("has_pool", "")
            has_tennis_courts = company_data.get("has_tennis_courts", "")
            number_of_holes = company_data.get("number_of_holes", "")
            geographic_seasonality = company_data.get("geographic_seasonality", "")
            public_private_flag = company_data.get("public_private_flag", "")
            club_info = company_data.get("club_info", "")

            # Add debug log verifying these fields
            logger.debug("Pulled in HubSpot company fields", extra={
                "company_name": name,
                "city": city,
                "state": state,
                "annual_revenue": annual_revenue,
                "created_date": created_date,
                "last_modified": last_modified,
                "object_id": object_id,
                "club_type": club_type,
                "facility_complexity": facility_complexity,
                "has_pool": has_pool,
                "has_tennis_courts": has_tennis_courts,
                "number_of_holes": number_of_holes,
                "geographic_seasonality": geographic_seasonality,
                "public_private_flag": public_private_flag,
                "club_info": club_info
            })

            # For usage in placeholders or further logic
            club_name = name.strip()  # rename for clarity
            logger.info(f"Successfully extracted {club_name} ({object_id}) located in {city}, {state}")

            # Proceed with existing placeholders logic:
            current_month = datetime.now().month
            if state == "AZ":
                start_peak_month = 0
                end_peak_month = 11
            else:
                start_peak_month = 5
                end_peak_month = 8

            season_key = get_season_variation_key(
                current_month=current_month,
                start_peak_month=start_peak_month,
                end_peak_month=end_peak_month
            )
            season_text = pick_season_snippet(season_key)

            placeholders = {
                "FirstName": first_name,
                "LastName": last_name,
                "ClubName": club_name or "Your Club",
                "DeadlineDate": "Oct 15th",
                "Role": lead_data.get("jobtitle", "General Manager"),
                "Task": "Staff Onboarding",
                "Topic": "On-Course Ordering Platform",
                "YourName": "Ty",
                "SEASON_VARIATION": season_text.rstrip(',')
            }
            logger.debug("Placeholders built", extra={**placeholders, "season_key": season_key})

        # Step 6: Gather additional personalization data
        with workflow_step(6, "Gathering personalization data"):
            club_info_snippet = data_gatherer.gather_club_info(club_name, city, state)
            news_result = data_gatherer.gather_club_news(club_name)
            has_news = False
            if news_result:
                if isinstance(news_result, tuple):
                    news_text = news_result[0]
                else:
                    news_text = str(news_result)
                has_news = "has not been" not in news_text.lower()

            jobtitle_str = lead_data.get("jobtitle", "")
            profile_type = categorize_job_title(jobtitle_str)

        # Step 7: Summarize interactions
        with workflow_step(7, "Summarizing interactions"):
            interaction_summary = summarize_lead_interactions(lead_sheet)
            logger.info("Interaction Summary:\n" + interaction_summary)

            last_interaction = lead_data.get("properties", {}).get("hs_sales_email_last_replied", "")
            last_interaction_days = 0
            if last_interaction:
                try:
                    last_date = datetime.fromtimestamp(int(last_interaction)/1000)
                    last_interaction_days = (datetime.now() - last_date).days
                except (ValueError, TypeError):
                    last_interaction_days = 0

        # Step 8: Build initial outreach email
        with workflow_step(8, "Building email draft"):
            subject = get_random_subject_template()
            for key, val in placeholders.items():
                subject = subject.replace(f"[{key}]", val)
            
            _, body = build_outreach_email(
                profile_type=profile_type,
                last_interaction_days=last_interaction_days,
                placeholders=placeholders,
                current_month=9,
                start_peak_month=5,
                end_peak_month=8,
                use_markdown_template=True
            )
            logger.debug("Loaded email template", extra={
                "subject_template": subject,
                "body_template": body
            })

            try:
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

            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")

            orig_subject, orig_body = subject, body
            for key, val in placeholders.items():
                subject = subject.replace(f"[{key}]", val)
                body = body.replace(f"[{key}]", val)

        # Step 9: Personalize with xAI
        with workflow_step(9, "Personalizing with AI"):
            try:
                lead_email = lead_data.get("email", email)
                
                subject, body = personalize_email_with_xai(
                    lead_sheet=lead_sheet,
                    subject=subject,
                    body=body,
                    summary=interaction_summary,
                    news_summary=news_result,
                    club_info=club_info_snippet
                )
                
                if not subject.strip():
                    subject = orig_subject
                if not body.strip():
                    body = orig_body

                body = body.replace("**", "")  # Cleanup leftover bold

                for key, val in placeholders.items():
                    subject = subject.replace(f"[{key}]", val)
                    body = body.replace(f"[{key}]", val)

                if has_news:
                    try:
                        icebreaker = _build_icebreaker_from_news(club_name, news_result)
                        if icebreaker:
                            body = body.replace("[ICEBREAKER]", icebreaker)
                        else:
                            body = body.replace("[ICEBREAKER]", "")
                    except Exception as e:
                        logger.error(f"Icebreaker generation error: {e}")
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    body = body.replace("[ICEBREAKER]", "")

                logger.debug("Creating email draft", extra={
                    **logger_context,
                    "to": lead_email,
                    "subject": subject
                })
            except Exception as e:
                logger.error(f"xAI personalization error: {e}")
                subject, body = orig_subject, orig_body

        # Step 10: Create Gmail draft and save to database
        with workflow_step(10, "Creating Gmail draft"):
            persona = profile_type
            club_tz = data_gatherer.get_club_timezone(state)
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT company_type FROM companies 
                    WHERE name = ? AND city = ? AND state = ?
                """, (club_name, city, state))
                result = cursor.fetchone()
                stored_company_type = result[0] if result else None

                if stored_company_type:
                    if "private" in stored_company_type.lower():
                        local_club_type = "Private Clubs"
                    elif "semi-private" in stored_company_type.lower():
                        local_club_type = "Semi-Private Clubs"
                    elif "public" in stored_company_type.lower() or "municipal" in stored_company_type.lower():
                        local_club_type = "Public Clubs"
                    else:
                        local_club_type = "Public Clubs"
                else:
                    _, local_club_type = data_gatherer.get_club_geography_and_type(club_name, city, state)

                if state == "AZ":
                    geography = "Year-Round Golf"
                else:
                    geography = data_gatherer.determine_geography(city, state)

            except Exception as e:
                logger.error(f"Error getting company type from database: {str(e)}")
                geography, local_club_type = data_gatherer.get_club_geography_and_type(club_name, city, state)
            finally:
                cursor.close()
                conn.close()

            outreach_window = {
                "Best Month": get_best_month(geography),
                "Best Time": get_best_time(persona),
                "Best Day": get_best_day(persona)
            }
            
            try:
                from random import randint
                best_months = outreach_window["Best Month"]
                best_time = outreach_window["Best Time"]
                best_days = outreach_window["Best Day"]
                
                now = datetime.now()
                target_date = now + timedelta(days=1)
                
                while target_date.month not in best_months:
                    if target_date.month == 12:
                        target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
                    else:
                        target_date = target_date.replace(month=target_date.month + 1, day=1)
                
                while target_date.weekday() not in best_days:
                    target_date += timedelta(days=1)
                
                target_hour = randint(best_time["start"], best_time["end"])
                scheduled_send_date = target_date.replace(
                    hour=target_hour,
                    minute=randint(0, 59),
                    second=0,
                    microsecond=0
                )
                
                state_offsets = STATE_TIMEZONES.get(state.upper())
                scheduled_send_date = convert_to_club_timezone(scheduled_send_date, state_offsets)

            except Exception as e:
                logger.warning(f"Error calculating send date: {str(e)}. Using current time + 1 day", extra={
                    "error": str(e),
                    "fallback_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_optimal_time": False
                })
                scheduled_send_date = datetime.now() + timedelta(days=1)

            if scheduled_send_date is None:
                logger.warning("scheduled_send_date is None, setting default send date")
                scheduled_send_date = datetime.now() + timedelta(days=1)

            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT lead_id FROM leads 
                    WHERE email = ?
                """, (lead_email,))
                result = cursor.fetchone()
                
                if result:
                    lead_id = result[0]
                    
                    draft_result = create_draft(
                        sender="me",
                        to=lead_email,
                        subject=subject,
                        message_text=body,
                        lead_id=lead_id,
                        sequence_num=1
                    )

                    if draft_result["status"] == "ok":
                        store_draft_info(
                            lead_id=lead_id,
                            draft_id=draft_result["draft_id"],
                            scheduled_date=scheduled_send_date,
                            subject=subject,
                            body=body,
                            sequence_num=1
                        )
                        conn.commit()
                        logger.info("Email draft saved to database", extra=logger_context)
                        
                        if CREATE_FOLLOWUP_DRAFT:
                            cursor.execute("""
                                SELECT COALESCE(MAX(sequence_num), 0) + 1
                                FROM emails 
                                WHERE lead_id = ?
                            """, (lead_id,))
                            next_seq = cursor.fetchone()[0]
                            
                            cursor.execute("""
                                SELECT email_id 
                                FROM emails 
                                WHERE draft_id = ? AND lead_id = ?
                            """, (draft_result["draft_id"], lead_id))
                            email_id = cursor.fetchone()[0]
                            
                            followup_data = generate_followup_email_xai(lead_id, email_id, next_seq)
                            if followup_data:
                                schedule_followup(lead_id, email_id)
                            else:
                                logger.warning("No follow-up data generated")
                        else:
                            logger.info("Follow-up draft creation is disabled via CREATE_FOLLOWUP_DRAFT setting")
                    else:
                        logger.error("Failed to create Gmail draft", extra=logger_context)
                else:
                    logger.error("Could not find lead_id for email", extra={
                        **logger_context,
                        "email": lead_email
                    })
                    
            except Exception as e:
                logger.error("Database operation failed", 
                            extra={**logger_context, "error": str(e)},
                            exc_info=True)
                conn.rollback()
                raise
            finally:
                cursor.close()
                conn.close()

    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}", extra=logger_context)
    except Exception as e:
        logger.error("Workflow failed", 
                    extra={**logger_context, "error": str(e)}, 
                    exc_info=True)
        raise

def verify_templates():
    """Verify that required templates exist."""
    template_dir = PROJECT_ROOT / 'docs' / 'templates'
    
    if not template_dir.exists():
        logger.error(f"Template directory not found: {template_dir}")
        return
        
    template_types = {
        'general_manager': 'general_manager_initial_outreach_*.md',
        'fb_manager': 'fb_manager_initial_outreach_*.md',
        'fallback': 'fallback_*.md'
    }
    
    for type_name, pattern in template_types.items():
        templates = list(template_dir.glob(pattern))
        if not templates:
            logger.warning(f"No {type_name} templates found matching pattern: {pattern}")
        else:
            logger.debug(f"Found {len(templates)} {type_name} templates: {[t.name for t in templates]}")

def calculate_send_date(geography, persona, state_code, season_data=None):
    """Calculate optimal send date based on geography and persona."""
    try:
        outreach_window = get_best_outreach_window(
            persona=persona,
            geography=geography,
            season_data=season_data
        )
        
        best_months = outreach_window["Best Month"]
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        now = datetime.now()
        target_date = now + timedelta(days=1)
        
        while target_date.month not in best_months:
            if target_date.month == 12:
                target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
            else:
                target_date = target_date.replace(month=target_date.month + 1, day=1)
        
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

def clear_sql_tables():
    """Clear all records from SQL tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        tables = [
            'emails',
            'lead_properties',
            'company_properties',
            'leads',
            'companies'
        ]
        
        cursor.execute("ALTER TABLE emails NOCHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE lead_properties NOCHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE company_properties NOCHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE leads NOCHECK CONSTRAINT ALL")
        
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM dbo.{table}")
                print(f"Cleared table: {table}")
            except Exception as e:
                print(f"Error clearing table {table}: {e}")
        
        cursor.execute("ALTER TABLE emails CHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE lead_properties CHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE company_properties CHECK CONSTRAINT ALL")
        cursor.execute("ALTER TABLE leads CHECK CONSTRAINT ALL")
        
        conn.commit()
        print("All SQL tables cleared")
        
    except Exception as e:
        print(f"Failed to clear SQL tables: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def clear_files_on_start():
    from config.settings import CLEAR_LOGS_ON_START
    
    if not CLEAR_LOGS_ON_START:
        print("Skipping file cleanup - CLEAR_LOGS_ON_START is False")
        return
        
    log_path = os.path.join(PROJECT_ROOT, 'logs', 'app.log')
    if os.path.exists(log_path):
        try:
            open(log_path, 'w').close()
            print("Log file cleared")
        except Exception as e:
            print(f"Failed to clear log file: {e}")
    
    lead_contexts_path = os.path.join(PROJECT_ROOT, 'lead_contexts')
    if os.path.exists(lead_contexts_path):
        try:
            for filename in os.listdir(lead_contexts_path):
                file_path = os.path.join(lead_contexts_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print("Lead contexts cleared")
        except Exception as e:
            print(f"Failed to clear lead contexts: {e}")
    
    clear_sql_tables()

def schedule_followup(lead_id: int, email_id: int):
    """Schedule a follow-up email"""
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


if __name__ == "__main__":
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    verify_templates()
    
    from scheduling.followup_scheduler import start_scheduler
    import threading
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    for i in range(3):
        logger.info(f"Starting iteration {i+1} of 3")
        main()
        logger.info(f"Completed iteration {i+1} of 3")




## scheduling\database.py

# scheduling/database.py

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
    """
    Recreate all tables, with season data columns in the 'companies' table:
      - year_round, start_month, end_month, peak_season_start, peak_season_end
    Remove them from 'company_properties' so we don't store them twice.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        logger.info("Starting init_db...")

        # 1) Drop all foreign key constraints
        logger.info("Dropping foreign key constraints...")
        cursor.execute("""
            DECLARE @SQL NVARCHAR(MAX) = '';
            SELECT @SQL += 'ALTER TABLE ' + QUOTENAME(OBJECT_SCHEMA_NAME(parent_object_id))
                + '.' + QUOTENAME(OBJECT_NAME(parent_object_id))
                + ' DROP CONSTRAINT ' + QUOTENAME(name) + ';'
            FROM sys.foreign_keys;
            EXEC sp_executesql @SQL;
        """)
        conn.commit()

        # 2) Drop existing tables
        logger.info("Dropping existing tables if they exist...")
        cursor.execute("""
            IF OBJECT_ID('dbo.emails', 'U') IS NOT NULL
                DROP TABLE dbo.emails;

            IF OBJECT_ID('dbo.lead_properties', 'U') IS NOT NULL
                DROP TABLE dbo.lead_properties;

            IF OBJECT_ID('dbo.leads', 'U') IS NOT NULL
                DROP TABLE dbo.leads;

            IF OBJECT_ID('dbo.company_properties', 'U') IS NOT NULL
                DROP TABLE dbo.company_properties;

            IF OBJECT_ID('dbo.companies', 'U') IS NOT NULL
                DROP TABLE dbo.companies;
        """)
        conn.commit()

        ################################################################
        # companies (static) – with new season data columns
        ################################################################
        cursor.execute("""
        CREATE TABLE dbo.companies (
            company_id           INT IDENTITY(1,1) PRIMARY KEY,
            name                 VARCHAR(255) NOT NULL,
            city                 VARCHAR(255),
            state                VARCHAR(255),
            company_type         VARCHAR(50),
            created_at           DATETIME DEFAULT GETDATE(),

            -- HubSpot data:
            hs_object_id         VARCHAR(50) NULL,
            hs_createdate        DATETIME NULL,
            hs_lastmodifieddate  DATETIME NULL,

            -- NEW SEASON DATA COLUMNS:
            year_round           VARCHAR(10),
            start_month          VARCHAR(20),
            end_month            VARCHAR(20),
            peak_season_start    VARCHAR(10),
            peak_season_end      VARCHAR(10),

            -- xAI Facilities Data:
            xai_facilities_info  VARCHAR(MAX)
        );
        """)
        conn.commit()

        ################################################################
        # company_properties (other dynamic fields, without season data)
        ################################################################
        cursor.execute("""
        CREATE TABLE dbo.company_properties (
            property_id          INT IDENTITY(1,1) PRIMARY KEY,
            company_id           INT NOT NULL,
            annualrevenue        VARCHAR(50),
            xai_facilities_news  VARCHAR(MAX),
            last_modified        DATETIME DEFAULT GETDATE(),

            CONSTRAINT FK_company_props
                FOREIGN KEY (company_id) REFERENCES dbo.companies(company_id)
        );
        """)
        conn.commit()

        ################################################################
        # leads (static)
        ################################################################
        cursor.execute("""
        CREATE TABLE dbo.leads (
            lead_id                INT IDENTITY(1,1) PRIMARY KEY,
            email                  VARCHAR(255) NOT NULL,
            first_name             VARCHAR(255),
            last_name              VARCHAR(255),
            role                   VARCHAR(255),
            status                 VARCHAR(50) DEFAULT 'active',
            created_at             DATETIME DEFAULT GETDATE(),
            company_id             INT NULL,

            -- HubSpot data:
            hs_object_id           VARCHAR(50) NULL,
            hs_createdate          DATETIME NULL,
            hs_lastmodifieddate    DATETIME NULL,

            CONSTRAINT UQ_leads_email UNIQUE (email),
            CONSTRAINT FK_leads_companies
                FOREIGN KEY (company_id) REFERENCES dbo.companies(company_id)
        );
        """)
        conn.commit()

        ################################################################
        # lead_properties (dynamic/refreshable)
        ################################################################
        cursor.execute("""
        CREATE TABLE dbo.lead_properties (
            property_id           INT IDENTITY(1,1) PRIMARY KEY,
            lead_id               INT NOT NULL,
            phone                 VARCHAR(50),
            lifecyclestage        VARCHAR(50),
            competitor_analysis   VARCHAR(MAX),
            last_response_date    DATETIME,
            last_modified         DATETIME DEFAULT GETDATE(),

            CONSTRAINT FK_lead_properties
                FOREIGN KEY (lead_id) REFERENCES dbo.leads(lead_id)
        );
        """)
        conn.commit()

        ################################################################
        # emails (tracking)
        ################################################################
        cursor.execute("""
        CREATE TABLE dbo.emails (
            email_id            INT IDENTITY(1,1) PRIMARY KEY,
            lead_id             INT NOT NULL,
            subject             VARCHAR(500),
            body                VARCHAR(MAX),
            status             VARCHAR(50) DEFAULT 'pending',
            scheduled_send_date DATETIME NULL,
            actual_send_date   DATETIME NULL,
            created_at         DATETIME DEFAULT GETDATE(),
            sequence_num       INT NULL,
            draft_id          VARCHAR(100) NULL,

            CONSTRAINT FK_emails_leads
                FOREIGN KEY (lead_id) REFERENCES dbo.leads(lead_id)
        );
        """)
        conn.commit()

        logger.info("init_db completed successfully. All tables dropped and recreated.")
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
    """Clear all tables in the database."""
    try:
        with get_db_connection() as conn:
            logger.debug("Clearing all tables")
            
            tables = [
                "dbo.emails",
                "dbo.leads", 
                "dbo.companies",
                "dbo.lead_properties",
                "dbo.company_properties"
            ]
            
            for table in tables:
                query = f"DELETE FROM {table}"
                logger.debug(f"Executing: {query}")
                conn.execute(query)
                
            logger.info("Successfully cleared all tables")

    except Exception as e:
        logger.exception(f"Failed to clear SQL tables: {str(e)}")
        raise e

def store_email_draft(cursor, lead_id: int, subject: str, body: str, 
                     scheduled_send_date: datetime = None, 
                     sequence_num: int = None,
                     draft_id: str = None,
                     status: str = 'pending') -> int:
    """
    Store email draft in database. Returns email_id.
    """
    cursor.execute("""
        INSERT INTO emails (
            lead_id, subject, body, status,
            scheduled_send_date, created_at,
            sequence_num, draft_id
        ) VALUES (?, ?, ?, ?, ?, GETDATE(), ?, ?)
    """, (
        lead_id, subject, body, status,
        scheduled_send_date, sequence_num, draft_id
    ))
    cursor.execute("SELECT SCOPE_IDENTITY()")
    return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database tables dropped and recreated.")




## scheduling\extended_lead_storage.py

# scheduling/extended_lead_storage.py

import datetime
import json
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger
from scheduling.database import get_db_connection
from utils.formatting_utils import clean_phone_number

def safe_parse_date(date_str):
    """
    Safely parse a date string into a Python datetime.
    Returns None if parsing fails or date_str is None.
    """
    if not date_str:
        return None
    try:
        return parse_date(date_str).replace(tzinfo=None)  # Remove timezone info
    except Exception:
        return None

def upsert_full_lead(lead_sheet: dict, correlation_id: str = None) -> None:
    """
    Upsert the lead and related company data into SQL:
      1) leads (incl. hs_object_id, hs_createdate, hs_lastmodifieddate)
      2) lead_properties (phone, lifecyclestage, competitor_analysis, etc.)
      3) companies (incl. hs_object_id, hs_createdate, hs_lastmodifieddate, plus season data)
      4) company_properties (annualrevenue, xai_facilities_news)
    
    Args:
        lead_sheet: Dictionary containing lead and company data
        correlation_id: Optional correlation ID for tracing operations
    """
    if correlation_id is None:
        correlation_id = f"upsert_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # == Extract relevant fields from the JSON ==
        metadata = lead_sheet.get("metadata", {})
        lead_data = lead_sheet.get("lead_data", {})
        analysis_data = lead_sheet.get("analysis", {})
        company_data = lead_data.get("company_data", {})

        # 1) Basic lead info
        email = lead_data.get("email") or metadata.get("lead_email", "")
        if not email:
            logger.error("No email found in lead_sheet; cannot upsert lead.", extra={
                "correlation_id": correlation_id,
                "status": "error"
            })
            return

        logger.debug("Upserting lead", extra={
            "email": email,
            "correlation_id": correlation_id,
            "operation": "upsert_lead"
        })

        # Access properties from lead_data structure
        lead_properties = lead_data.get("properties", {})
        first_name = lead_properties.get("firstname", "")
        last_name = lead_properties.get("lastname", "")
        role = lead_properties.get("jobtitle", "")

        # 2) HubSpot lead-level data
        lead_hs_id = lead_properties.get("hs_object_id", "")
        lead_created_str = lead_properties.get("createdate", "")
        lead_lastmod_str = lead_properties.get("lastmodifieddate", "")

        lead_hs_createdate = safe_parse_date(lead_created_str)
        lead_hs_lastmodified = safe_parse_date(lead_lastmod_str)

        # 3) Company static data
        static_company_name = company_data.get("name", "")
        static_city = company_data.get("city", "")
        static_state = company_data.get("state", "")

        # 4) Company HubSpot data
        company_hs_id = company_data.get("hs_object_id", "")
        company_created_str = company_data.get("createdate", "")
        company_lastmod_str = company_data.get("lastmodifieddate", "")

        company_hs_createdate = safe_parse_date(company_created_str)
        company_hs_lastmodified = safe_parse_date(company_lastmod_str)

        # 5) lead_properties (dynamic)
        phone = lead_data.get("phone")
        cleaned_phone = clean_phone_number(phone) if phone is not None else None
        lifecyclestage = lead_properties.get("lifecyclestage", "")

        competitor_analysis = analysis_data.get("competitor_analysis", "")
        # Convert competitor_analysis to JSON string if it's a dictionary
        if isinstance(competitor_analysis, dict):
            competitor_analysis = json.dumps(competitor_analysis)

        # Per request: do NOT save competitor_analysis (set blank)
        competitor_analysis = ""

        # Attempt to parse "last_response" as needed; storing None for now
        last_resp_str = analysis_data.get("previous_interactions", {}).get("last_response", "")
        last_response_date = None  # "Responded 1148 days ago" is not ISO, so we keep it as None

        # 6) Season data (stored in companies)
        season_data = analysis_data.get("season_data", {})
        year_round = season_data.get("year_round", "")
        start_month = season_data.get("start_month", "")
        end_month = season_data.get("end_month", "")
        peak_season_start = season_data.get("peak_season_start", "")
        peak_season_end = season_data.get("peak_season_end", "")

        # 7) Other company_properties (dynamic)
        annualrevenue = company_data.get("annualrevenue", "")

        # 8) xAI Facilities Data
        #   - xai_facilities_info goes in dbo.companies
        #   - xai_facilities_news goes in dbo.company_properties
        facilities_info = analysis_data.get("facilities", {}).get("response", "")
        if facilities_info == "No recent news found.":
            facilities_info = ""  # Just set to empty string instead of making assumptions

        # Get news separately
        research_data = analysis_data.get("research_data", {})
        facilities_news = ""
        if research_data.get("recent_news"):
            try:
                news_item = research_data["recent_news"][0]
                if isinstance(news_item, dict):
                    facilities_news = str(news_item.get("snippet", ""))[:500]  # Limit length
            except (IndexError, KeyError, TypeError):
                logger.warning("Could not extract facilities news from research data")

        # Get the company overview separately
        research_data = analysis_data.get("research_data", {})
        company_overview = research_data.get("company_overview", "")

        # ==========================================================
        # 1. Upsert into leads (static fields + HS fields)
        # ==========================================================
        existing_company_id = None  # Initialize before the query
        cursor.execute("SELECT lead_id, company_id, hs_object_id FROM dbo.leads WHERE email = ?", (email,))
        row = cursor.fetchone()

        if row:
            lead_id = row[0]
            existing_company_id = row[1]  # Will update if found
            lead_hs_id = row[2]
            logger.debug(f"Lead with email={email} found (lead_id={lead_id}); updating record.")
            cursor.execute("""
                UPDATE dbo.leads
                SET first_name = ?,
                    last_name = ?,
                    role = ?,
                    hs_object_id = ?,
                    hs_createdate = ?,
                    hs_lastmodifieddate = ?,
                    status = 'active'
                WHERE lead_id = ?
            """, (
                first_name,
                last_name,
                role,
                lead_hs_id,
                lead_hs_createdate,
                lead_hs_lastmodified,
                lead_id
            ))
        else:
            logger.debug(f"Lead with email={email} not found; inserting new record.")
            cursor.execute("""
                INSERT INTO dbo.leads (
                    email,
                    first_name,
                    last_name,
                    role,
                    status,
                    hs_object_id,
                    hs_createdate,
                    hs_lastmodifieddate
                )
                OUTPUT Inserted.lead_id
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """, (
                email,
                first_name,
                last_name,
                role,
                lead_hs_id,
                lead_hs_createdate,
                lead_hs_lastmodified
            ))
            
            # Capture the new lead_id
            result = cursor.fetchone()
            lead_id = result[0] if result else None
            
            if not lead_id:
                raise ValueError("Failed to get lead_id after insertion")
            
            conn.commit()
            logger.info("Successfully inserted new lead record", extra={
                "email": email,
                "lead_id": lead_id,
                "hs_object_id": lead_hs_id
            })

        # ==========================================================
        # 2. Upsert into companies (static fields + HS fields + season data)
        # ==========================================================
        if not static_company_name.strip():
            logger.debug("No company name found, skipping upsert for companies.")
        else:
            cursor.execute("""
                SELECT company_id, hs_object_id
                FROM dbo.companies
                WHERE name = ? AND city = ? AND state = ?
            """, (static_company_name, static_city, static_state))
            existing_co = cursor.fetchone()

            if existing_co:
                company_id = existing_co[0]
                company_hs_id = existing_co[1]
                logger.debug(f"Company found (company_id={company_id}); updating HS fields + season data if needed.")
                
                # Convert any potential dictionaries to strings
                facilities_info_str = json.dumps(facilities_info) if isinstance(facilities_info, dict) else str(facilities_info) if facilities_info else None
                year_round_str = str(year_round) if year_round else None
                start_month_str = str(start_month) if start_month else None
                end_month_str = str(end_month) if end_month else None
                peak_season_start_str = str(peak_season_start) if peak_season_start else None
                peak_season_end_str = str(peak_season_end) if peak_season_end else None
                
                cursor.execute("""
                    UPDATE dbo.companies
                    SET city = ?,
                        state = ?,
                        company_type = ?,
                        hs_createdate = ?,
                        hs_lastmodifieddate = ?,
                        year_round = ?,
                        start_month = ?,
                        end_month = ?,
                        peak_season_start = ?,
                        peak_season_end = ?,
                        xai_facilities_info = ?
                    WHERE company_id = ?
                """, (
                    static_city,
                    static_state,
                    company_data.get("company_type", ""),
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round_str,
                    start_month_str,
                    end_month_str,
                    peak_season_start_str,
                    peak_season_end_str,
                    facilities_info_str,
                    company_id
                ))
            else:
                logger.debug(f"No matching company; inserting new row for name={static_company_name}.")
                
                # Convert any potential dictionaries to strings
                facilities_info_str = json.dumps(facilities_info) if isinstance(facilities_info, dict) else str(facilities_info) if facilities_info else None
                year_round_str = str(year_round) if year_round else None
                start_month_str = str(start_month) if start_month else None
                end_month_str = str(end_month) if end_month else None
                peak_season_start_str = str(peak_season_start) if peak_season_start else None
                peak_season_end_str = str(peak_season_end) if peak_season_end else None
                
                cursor.execute("""
                    INSERT INTO dbo.companies (
                        name, city, state, company_type,
                        hs_object_id, hs_createdate, hs_lastmodifieddate,
                        year_round, start_month, end_month,
                        peak_season_start, peak_season_end,
                        xai_facilities_info
                    )
                    OUTPUT Inserted.company_id
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    static_company_name,
                    static_city,
                    static_state,
                    company_data.get("company_type", ""),
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round_str,
                    start_month_str,
                    end_month_str,
                    peak_season_start_str,
                    peak_season_end_str,
                    facilities_info_str
                ))
                
                # Capture the new company_id
                result = cursor.fetchone()
                company_id = result[0] if result else None
                
                if not company_id:
                    raise ValueError("Failed to get company_id after insertion")
                
                conn.commit()
                logger.info("Successfully inserted new company record", extra={
                    "company_name": static_company_name,
                    "company_id": company_id,
                    "hs_object_id": company_hs_id
                })

            # If we have a company_id, ensure leads.company_id is updated
            logger.debug("Updating lead's company_id", extra={
                "lead_id": lead_id,
                "company_id": company_id,
                "email": email
            })
            if company_id:
                if not existing_company_id or existing_company_id != company_id:
                    logger.debug(f"Updating lead with lead_id={lead_id} to reference company_id={company_id}.")
                    cursor.execute("""
                        UPDATE dbo.leads
                        SET company_id = ?
                        WHERE lead_id = ?
                    """, (company_id, lead_id))
                    conn.commit()

        # ==========================================================
        # 3. Upsert into lead_properties (phone, lifecycle, competitor, etc.)
        # ==========================================================
        cursor.execute("""
            SELECT property_id FROM dbo.lead_properties WHERE lead_id = ?
        """, (lead_id,))
        lp_row = cursor.fetchone()

        if lp_row:
            prop_id = lp_row[0]
            logger.debug("Updating existing lead properties", extra={
                "lead_id": lead_id,
                "property_id": prop_id,
                "phone": phone,
                "lifecyclestage": lifecyclestage,
                "last_response_date": last_response_date
            })
            cursor.execute("""
                UPDATE dbo.lead_properties
                SET phone = ?,
                    lifecyclestage = ?,
                    competitor_analysis = ?,
                    last_response_date = ?,
                    last_modified = GETDATE()
                WHERE property_id = ?
            """, (
                phone,
                lifecyclestage,
                competitor_analysis,
                last_response_date,
                prop_id
            ))
        else:
            logger.debug(f"No lead_properties row found; inserting new one for lead_id={lead_id}.")
            cursor.execute("""
                INSERT INTO dbo.lead_properties (
                    lead_id, phone, lifecyclestage, competitor_analysis,
                    last_response_date, last_modified
                )
                VALUES (?, ?, ?, ?, ?, GETDATE())
            """, (
                lead_id,
                phone,
                lifecyclestage,
                competitor_analysis,
                last_response_date
            ))
        conn.commit()

        # ==========================================================
        # 4. Upsert into company_properties (dynamic fields)
        #    No competitor_analysis is saved here (store blank).
        #    Make sure xai_facilities_news is saved.
        # ==========================================================
        if company_id:
            cursor.execute("""
                SELECT property_id FROM dbo.company_properties WHERE company_id = ?
            """, (company_id,))
            cp_row = cursor.fetchone()

            if cp_row:
                cp_id = cp_row[0]
                logger.debug("Updating existing company properties", extra={
                    "company_id": company_id,
                    "property_id": cp_id,
                    "annualrevenue": annualrevenue,
                    "has_facilities_news": facilities_news is not None
                })
                cursor.execute("""
                    UPDATE dbo.company_properties
                    SET annualrevenue = ?,
                        xai_facilities_news = ?,
                        last_modified = GETDATE()
                    WHERE property_id = ?
                """, (
                    annualrevenue,
                    facilities_news,
                    cp_id
                ))
            else:
                logger.debug(f"No company_properties row found; inserting new one for company_id={company_id}.")
                cursor.execute("""
                    INSERT INTO dbo.company_properties (
                        company_id,
                        annualrevenue,
                        xai_facilities_news,
                        last_modified
                    )
                    VALUES (?, ?, ?, GETDATE())
                """, (
                    company_id,
                    str(annualrevenue) if annualrevenue else None,
                    str(facilities_news)[:500] if facilities_news else None  # Limit length and ensure string
                ))
            conn.commit()

        logger.info("Successfully completed lead and company upsert", extra={
            "email": email,
            "lead_id": lead_id,
            "company_id": company_id if company_id else None,
            "has_lead_properties": bool(lp_row),
            "has_company_properties": bool(cp_row) if company_id else False
        })

        # Add email storage
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get lead_id for the email records
                lead_email = lead_sheet["lead_data"]["properties"]["email"]
                cursor.execute("SELECT lead_id FROM leads WHERE email = ?", (lead_email,))
                lead_id = cursor.fetchone()[0]
                
                # Insert emails - Add check for existing draft
                emails = lead_sheet.get("lead_data", {}).get("emails", [])
                for email in emails:
                    # Skip drafts - these are handled by store_draft_info
                    if email.get('status') == 'draft':
                        logger.debug(f"Skipping draft email for lead_id={lead_id}, will be handled by store_draft_info")
                        continue

                    # Check if email already exists
                    cursor.execute("""
                        SELECT email_id FROM emails 
                        WHERE lead_id = ? AND draft_id = ?
                    """, (lead_id, email.get('id')))
                    
                    if cursor.fetchone():
                        logger.debug(f"Email already exists for lead_id={lead_id}, draft_id={email.get('id')}")
                        continue

                    # Map the email status
                    status = 'sent' if email.get('status') == 'sent' else 'pending'
                    
                    # Convert timestamp to datetime if present
                    send_date = None
                    if email.get('timestamp'):
                        try:
                            send_date = parse_date(email.get('timestamp'))
                        except:
                            logger.warning(f"Could not parse timestamp: {email.get('timestamp')}")

                    cursor.execute("""
                        INSERT INTO emails (
                            lead_id, subject, body, 
                            status, actual_send_date, created_at,
                            draft_id
                        )
                        VALUES (?, ?, ?, ?, ?, GETDATE(), ?)
                    """, (
                        lead_id,
                        email.get('subject'),
                        email.get('body_text'),
                        status,
                        send_date,
                        email.get('id')
                    ))
                
                conn.commit()
                logger.info("Successfully stored email records")
                
        except Exception as e:
            logger.error(f"Failed to store emails: {str(e)}")
            raise

        logger.info(f"Successfully stored {len(emails)} email records")

    except Exception as e:
        logger.error(f"Error in upsert_full_lead: {str(e)}", exc_info=True)
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

system_message = (
    "You are a factual assistant that provides objective, data-focused overviews of clubs. "
    "CRITICAL: Only state amenities that are explicitly verified. If you are not certain "
    "about an amenity, respond with 'Unknown' rather than making assumptions. Never infer "
    "amenities based on club type or location."
)




## scheduling\followup_generation.py

# followup_generation.py

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window, adjust_send_time
from datetime import datetime, timedelta
import random
from utils.xai_integration import get_random_subject_template

def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None, 
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    l.email,
                    l.first_name,
                    c.name,
                    e.subject,
                    e.body,
                    e.created_at,
                    c.state
                FROM emails e
                JOIN leads l ON l.lead_id = e.lead_id
                LEFT JOIN companies c ON l.company_id = c.company_id
                WHERE e.lead_id = ?
                AND e.sequence_num = 1
                ORDER BY e.created_at DESC
            """, (lead_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"No original email found for lead_id={lead_id}")
                return None

            email, first_name, company_name, subject, body, created_at, state = row
            original_email = {
                'email': email,
                'first_name': first_name,
                'name': company_name,
                'subject': subject,
                'body': body,
                'created_at': created_at,
                'state': state
            }

        # If original_email is provided, use that instead of querying
        if original_email:
            email = original_email.get('email')
            first_name = original_email.get('first_name')
            company_name = original_email.get('name', 'your club')
            state = original_email.get('state')
            orig_subject = original_email.get('subject')
            orig_body = original_email.get('body')
            orig_date = original_email.get('created_at', datetime.now())
            
            # Get original scheduled send date
            cursor.execute("""
                SELECT TOP 1 scheduled_send_date 
                FROM emails 
                WHERE lead_id = ? AND sequence_num = 1
                ORDER BY created_at DESC
            """, (lead_id,))
            result = cursor.fetchone()
            orig_scheduled_date = result[0] if result else orig_date
        else:
            # Query for required fields
            query = """
                SELECT 
                    l.email,
                    l.first_name,
                    c.state,
                    c.name,
                    e.subject,
                    e.body,
                    e.created_at
                FROM leads l
                LEFT JOIN companies c ON l.company_id = c.company_id
                LEFT JOIN emails e ON l.lead_id = e.lead_id
                WHERE l.lead_id = ? AND e.email_id = ?
            """
            cursor.execute(query, (lead_id, email_id))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"No lead found for lead_id={lead_id}")
                return None

            email, first_name, state, company_name, orig_subject, orig_body, orig_date = result
            company_name = company_name or 'your club'
            
            # Get original scheduled send date
            cursor.execute("""
                SELECT scheduled_send_date 
                FROM emails 
                WHERE email_id = ?
            """, (email_id,))
            result = cursor.fetchone()
            orig_scheduled_date = result[0] if result and result[0] else orig_date

        # If orig_scheduled_date is still None, default to orig_date
        if orig_scheduled_date is None:
            logger.warning("orig_scheduled_date is None, defaulting to orig_date")
            orig_scheduled_date = orig_date

        # Validate required fields
        if not email:
            logger.error("Missing required field: email")
            return None

        # Use RE: with original subject
        subject = f"RE: {orig_subject}"

        # Format the follow-up email body
        body = (
            f"Following up about improving operations at {company_name}. "
            f"Would you have 10 minutes this week for a brief call?\n\n"
            f"Best regards,\n"
            f"Ty\n\n"
            f"Swoop Golf\n"
            f"480-225-9702\n"
            f"swoopgolf.com\n\n"
            f"-------- Original Message --------\n"
            f"Date: {orig_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Subject: {orig_subject}\n"
            f"To: {email}\n\n"
            f"{orig_body}"
        )

        # Calculate base date (3 days after original scheduled date)
        base_date = orig_scheduled_date + timedelta(days=3)
        
        # Get optimal send window
        outreach_window = get_best_outreach_window(
            persona="general",
            geography="US",
        )
        
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        # Adjust to next valid day while preserving the 3-day minimum gap
        while base_date.weekday() not in best_days or base_date < (orig_scheduled_date + timedelta(days=3)):
            base_date += timedelta(days=1)
        
        # Set time within the best window
        send_hour = best_time["start"]
        if random.random() < 0.5:  # 50% chance to use later hour
            send_hour += 1
            
        send_date = base_date.replace(
            hour=send_hour,
            minute=random.randint(0, 59),
            second=0,
            microsecond=0
        )
        
        # Adjust for timezone
        send_date = adjust_send_time(send_date, state) if state else send_date

        logger.debug(f"[followup_generation] Potential scheduled_send_date for lead_id={lead_id} (1st email) is: {send_date}")

        # Calculate base date (3 days after original scheduled date)
        base_date = orig_scheduled_date + timedelta(days=3)
        
        # Get optimal send window
        outreach_window = get_best_outreach_window(
            persona="general",
            geography="US",
        )
        
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        # Adjust to next valid day while preserving the 3-day minimum gap
        while base_date.weekday() not in best_days or base_date < (orig_scheduled_date + timedelta(days=3)):
            base_date += timedelta(days=1)
        
        # Set time within the best window
        send_hour = best_time["start"]
        if random.random() < 0.5:  # 50% chance to use later hour
            send_hour += 1
            
        send_date = base_date.replace(
            hour=send_hour,
            minute=random.randint(0, 59),
            second=0,
            microsecond=0
        )
        
        # Adjust for timezone
        send_date = adjust_send_time(send_date, state) if state else send_date

        logger.debug(f"[followup_generation] Potential scheduled_send_date for lead_id={lead_id} (follow-up) is: {send_date}")

        return {
            'email': email,
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': lead_id,
            'first_name': first_name,
            'state': state
        }

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()




## scheduling\followup_scheduler.py

# scheduling/followup_scheduler.py

import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from scheduling.database import get_db_connection, store_email_draft
from scheduling.followup_generation import generate_followup_email_xai
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from config.settings import SEND_EMAILS, ENABLE_FOLLOWUPS
import logging

def check_and_send_followups():
    """Check for and send any pending follow-up emails"""
    if not ENABLE_FOLLOWUPS:
        logger.info("Follow-up emails are disabled via ENABLE_FOLLOWUPS setting")
        return

    logger.debug("Running check_and_send_followups")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get leads needing follow-up with all required fields
        cursor.execute("""
            SELECT 
                e.lead_id,
                e.email_id,
                l.email,
                l.first_name,
                c.name,
                e.subject,
                e.body,
                e.created_at,
                c.state
            FROM emails e
            JOIN leads l ON l.lead_id = e.lead_id
            LEFT JOIN companies c ON l.company_id = c.company_id
            WHERE e.sequence_num = 1
            AND e.status = 'sent'
            AND l.email IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM emails e2 
                WHERE e2.lead_id = e.lead_id 
                AND e2.sequence_num = 2
            )
        """)
        
        for row in cursor.fetchall():
            lead_id, email_id, email, first_name, company_name, subject, body, created_at, state = row
            
            # Package original email data
            original_email = {
                'email': email,
                'first_name': first_name,
                'name': company_name,
                'subject': subject,
                'body': body,
                'created_at': created_at,
                'state': state
            }
            
            # Generate follow-up content
            followup_data = generate_followup_email_xai(
                lead_id=lead_id,
                email_id=email_id,
                sequence_num=2,
                original_email=original_email
            )
            
            if followup_data and followup_data.get('scheduled_send_date'):
                # Create Gmail draft
                draft_result = create_draft(
                    sender="me",
                    to=followup_data['email'],
                    subject=followup_data['subject'],
                    message_text=followup_data['body']
                )

                if draft_result and draft_result.get("status") == "ok":
                    # Store in database with scheduled_send_date
                    store_email_draft(
                        cursor,
                        lead_id=lead_id,
                        subject=followup_data['subject'],
                        body=followup_data['body'],
                        scheduled_send_date=followup_data['scheduled_send_date'],
                        sequence_num=followup_data['sequence_num'],
                        draft_id=draft_result["draft_id"],
                        status='draft'
                    )
                    conn.commit()
                    logger.info(f"Follow-up scheduled for lead_id={lead_id} at {followup_data['scheduled_send_date']}")
            else:
                logger.error(f"Missing scheduled_send_date for lead_id={lead_id}")

    except Exception as e:
        logger.exception("Error in followup scheduler")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def start_scheduler():
    """Initialize and start the follow-up scheduler"""
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            check_and_send_followups,
            'interval',
            minutes=15,
            id='check_and_send_followups',
            next_run_time=datetime.datetime.now()
        )
        
        # Suppress initial scheduler messages
        logging.getLogger('apscheduler').setLevel(logging.WARNING)
        
        scheduler.start()
        logger.info("Follow-up scheduler initialized")
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")

if __name__ == "__main__":
    start_scheduler()




## scripts\build_template.py

# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path
from config.settings import PROJECT_ROOT
from utils.xai_integration import _send_xai_request

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
CONDITION_SUBJECTS = {
    "general_manager": [
        "Quick Question for [FirstName]",
        "New Ways to Elevate [ClubName]'s Operations",
        "Boost [ClubName]'s Efficiency with Swoop",
        "Need Assistance with [Task]? – [FirstName]"
    ],
    "fnb_manager": [
        "Ideas for Increasing F&B Revenue at [ClubName]",
        "Quick Note for [FirstName] about On-Demand Service",
        "A Fresh Take on [ClubName]'s F&B Operations"
    ],
    "golf_ops": [
        "Keeping [ClubName] Rounds on Pace: Quick Idea",
        "New Golf Ops Tools for [ClubName]",
        "Quick Question for [FirstName] – On-Course Efficiency"
    ],
    # New line: If job title doesn't match any known category,
    # we map it to this fallback template
    "fallback": [
        "Enhancing Your Club's Efficiency with Swoop",
        "Is [ClubName] Looking to Modernize?"
    ]
}


###############################################################################
# 2) PICK SUBJECT LINE BASED ON LEAD ROLE & LAST INTERACTION
###############################################################################
def pick_subject_line_based_on_lead(
    lead_role: str,
    last_interaction_days: int,
    placeholders: dict
) -> str:
    """
    Choose a subject line from CONDITION_SUBJECTS based on the lead role
    and the days since last interaction. Then replace placeholders.
    """
    # 1) Decide which subject lines to use based on role
    if lead_role in CONDITION_SUBJECTS:
        subject_variations = CONDITION_SUBJECTS[lead_role]
    else:
        subject_variations = CONDITION_SUBJECTS["fallback"]

    # 2) Example condition: if lead is "older" than 60 days, pick the first subject
    #    otherwise pick randomly.
    if last_interaction_days > 60:
        chosen_template = subject_variations[0]
    else:
        chosen_template = random.choice(subject_variations)

    # 3) Replace placeholders in the subject
    for key, val in placeholders.items():
        chosen_template = chosen_template.replace(f"[{key}]", val)

    return chosen_template


###############################################################################
# 3) SEASON VARIATION LOGIC (OPTIONAL)
###############################################################################
def apply_season_variation(email_text: str, snippet: str) -> str:
    """
    Replaces {SEASON_VARIATION} in an email text with the chosen snippet.
    """
    return email_text.replace("{SEASON_VARIATION}", snippet)


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
def build_outreach_email(
    profile_type: str,
    last_interaction_days: int,
    placeholders: dict,
    current_month: int,
    start_peak_month: int,
    end_peak_month: int,
    use_markdown_template: bool = True,
    template_path: str = None
) -> tuple[str, str]:
    """Build email content from template."""
    try:
        # Use provided template if available
        if template_path and Path(template_path).exists():
            logger.debug(f"Using provided template: {template_path}")
            logger.info(f"Template file exists: {Path(template_path).exists()}")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                logger.debug(f"Successfully read template file. Content length: {len(template_content)}")
                logger.debug(f"First 100 chars of template: {template_content[:100]}...")
                
            # Validate template
            logger.debug("Validating template content...")
            validate_template(template_content)
            logger.debug("Template validation successful")
            
            # Extract subject and body from markdown
            logger.debug("Extracting subject and body from template...")
            subject, body = extract_subject_and_body(template_content)
            logger.debug(f"Extracted subject length: {len(subject)}")
            logger.debug(f"Extracted body length: {len(body)}")
            
            # Apply season variation if present
            if "{SEASON_VARIATION}" in body:
                logger.debug("Applying season variation...")
                season_key = get_season_variation_key(
                    current_month=current_month,
                    start_peak_month=start_peak_month,
                    end_peak_month=end_peak_month
                )
                season_snippet = pick_season_snippet(season_key)
                body = apply_season_variation(body, season_snippet)
                logger.debug("Season variation applied successfully")
            
            logger.info("Template processing completed successfully")
            return subject, body
                
        # Fallback to existing template selection logic
        logger.warning(f"Template path not provided or doesn't exist: {template_path}")
        # ... rest of existing fallback logic ...

    except FileNotFoundError as e:
        logger.error(f"Template file not found: {template_path}")
        logger.error(f"Error details: {str(e)}")
        return get_fallback_template().split('---\n', 1)
    except Exception as e:
        logger.error(f"Error building outreach email: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Full traceback:")
        return get_fallback_template().split('---\n', 1)

def get_fallback_template() -> str:
    """Returns a basic fallback template if all other templates fail."""
    return """Connecting About Club Services
---
Hi [FirstName],

I wanted to reach out about how we're helping clubs like [ClubName] enhance their member experience through our comprehensive platform.

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
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    template_data = parse_template(template_content)
    
    # Replace parameters in both subject and body
    subject = template_data['subject']
    body = template_data['body']
    
    for key, value in parameters.items():
        subject = subject.replace(f'[{key}]', str(value))
        body = body.replace(f'[{key}]', str(value))
        # Handle season variation differently since it uses curly braces
        if key == 'SEASON_VARIATION':
            body = body.replace('{SEASON_VARIATION}', str(value))
    
    return {
        'subject': subject,
        'body': body
    }

def extract_template_body(template_content):
    """Extract body from template content, no subject needed"""
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



## scripts\golf_outreach_strategy.py

"""
Scripts for determining optimal outreach timing based on club and contact attributes.
"""
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os

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
    Returns a dict with start and end hours in 24-hour format.
    """
    time_windows = {
        "General Manager": {"start": 9, "end": 11},  # 9am-11am
        "Membership Director": {"start": 13, "end": 15},  # 1pm-3pm
        "Food & Beverage Director": {"start": 10, "end": 12},  # 10am-12pm
        "Golf Professional": {"start": 8, "end": 10},  # 8am-10am
        "Superintendent": {"start": 7, "end": 9}  # 7am-9am
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    return time_windows.get(persona, {"start": 9, "end": 11})

def get_best_day(persona: str) -> list:
    """
    Determine best days of week based on persona.
    Returns a list of day numbers (0 = Monday, 6 = Sunday)
    """
    day_mappings = {
        "General Manager": [1, 3],  # Tuesday, Thursday
        "Membership Director": [2, 3],  # Wednesday, Thursday
        "Food & Beverage Director": [2, 3],  # Wednesday, Thursday
        "Golf Professional": [1, 2],  # Tuesday, Wednesday
        "Superintendent": [1, 2]  # Tuesday, Wednesday
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    return day_mappings.get(persona, [1, 3])

def get_best_outreach_window(persona: str, geography: str, club_type: str = None, season_data: dict = None) -> Dict[str, Any]:
    """Get the optimal outreach window based on persona and geography."""
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona)
    best_days = get_best_day(persona)
    
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

def calculate_send_date(geography: str, profile_type: str, state: str, preferred_days: list, preferred_time: dict) -> datetime:
    """
    Calculate the next appropriate send date based on outreach window and preferred days/time.
    """
    outreach_window = get_best_outreach_window(geography, profile_type)
    best_months = outreach_window["Best Month"]
    
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
    
    return target_date




## services\data_gatherer_service.py

# services/data_gatherer_service.py

import json
import csv
import datetime
from typing import Dict, Any, Union, List
from pathlib import Path
from dateutil.parser import parse as parse_date

from services.hubspot_service import HubspotService
from utils.exceptions import HubSpotError
from utils.xai_integration import xai_news_search, xai_club_info_search
from utils.web_fetch import fetch_website_html
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT
from utils.formatting_utils import clean_html

# CSV-based Season Data
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}

TIMEZONE_CSV_PATH = PROJECT_ROOT / 'docs' / 'data' / 'state_timezones.csv'
STATE_TIMEZONES: Dict[str, Dict[str, int]] = {}


class DataGathererService:
    """
    Centralized service to gather all relevant data about a lead in one pass.
    Fetches HubSpot contact & company info, emails, competitor checks,
    interactions, market research, and season data.
    """

    def __init__(self):
        """Initialize the DataGathererService with HubSpot client and season data."""
        self.hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        # Load season data at initialization
        self.load_season_data()
        self.load_state_timezones()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data (now mostly delegated to HubspotService)."""
        return self.hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data from HubSpot, notes, emails, then merges competitor & season data.
        """
        if correlation_id is None:
            correlation_id = f"gather_{lead_email}"

        # 1) Lookup contact_id via email
        contact_data = self.hubspot.get_contact_by_email(lead_email)
        if not contact_data:
            logger.error("Failed to find contact ID", extra={
                "email": lead_email,
                "operation": "gather_lead_data",
                "correlation_id": correlation_id,
                "status": "error"
            })
            return {}
        contact_id = contact_data["id"]

        # 2) Get the contact properties
        contact_props = self.hubspot.get_contact_properties(contact_id)

        # 3) Get the associated company_id
        company_id = self.hubspot.get_associated_company_id(contact_id)

        # 4) Get the company data (including city/state, plus new fields)
        company_props = self.hubspot.get_company_data(company_id)

        # 5) Add calls to fetch emails and notes from HubSpot
        emails = self.hubspot.get_all_emails_for_contact(contact_id)
        notes = self.hubspot.get_all_notes_for_contact(contact_id)

        # Example: competitor check
        competitor_analysis = self.check_competitor_on_website(company_props.get("website", ""))

        # Example: gather news just once
        club_name = company_props.get("name", "")
        news_result = self.gather_club_news(club_name)

        # Build partial lead_sheet
        lead_sheet = {
            "metadata": {
                "contact_id": contact_id,
                "company_id": company_id,
                "lead_email": contact_props.get("email", ""),
                "status": "success"
            },
            "lead_data": {
                "id": contact_id,
                "properties": contact_props,
                # This is where all 15 fields will appear (no filtering):
                "company_data": company_props,
                "emails": emails,
                "notes": notes
            },
            "analysis": {
                "competitor_analysis": competitor_analysis,
                "research_data": {
                    "company_overview": news_result,
                    "recent_news": [{
                        "title": "Recent News",
                        "snippet": news_result,
                        "link": "",
                        "date": ""
                    }] if news_result else [],
                    "status": "success",
                    "error": ""
                },
                "previous_interactions": self.review_previous_interactions(contact_id),
                "season_data": self.determine_club_season(
                    company_props.get("city", ""), 
                    company_props.get("state", "")
                ),
                "facilities": self.check_facilities(
                    club_name,
                    company_props.get("city", ""),
                    company_props.get("state", "")
                )
            }
        }

        # Optionally save or log the final lead_sheet
        self._save_lead_context(lead_sheet, lead_email)

        logger.info(
            "Data gathering completed successfully",
            extra={
                "email": lead_email,
                "contact_id": contact_id,
                "company_id": company_id,
                "correlation_id": correlation_id,
                "operation": "gather_lead_data"
            }
        )
        return lead_sheet

    # -------------------------------------------------------------------------
    # Example competitor-check logic
    # -------------------------------------------------------------------------
    def check_competitor_on_website(self, domain: str, correlation_id: str = None) -> Dict[str, str]:
        if correlation_id is None:
            correlation_id = f"competitor_check_{domain}"
        try:
            if not domain:
                return {
                    "competitor": "",
                    "status": "no_data",
                    "error": "No domain provided"
                }
            url = domain.strip().lower()
            if not url.startswith("http"):
                url = f"https://{url}"
            html = fetch_website_html(url)
            if not html:
                return {
                    "competitor": "",
                    "status": "error",
                    "error": "Could not fetch website content"
                }
            # sample competitor mention
            competitor_mentions = ["jonas club software", "jonas software", "jonasclub"]
            for mention in competitor_mentions:
                if mention in html.lower():
                    return {
                        "competitor": "Jonas",
                        "status": "success",
                        "error": ""
                    }
            return {
                "competitor": "",
                "status": "success",
                "error": ""
            }
        except Exception as e:
            logger.error("Error checking competitor on website", extra={
                "domain": domain,
                "error_type": type(e).__name__,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return {
                "competitor": "",
                "status": "error",
                "error": f"Error checking competitor: {str(e)}"
            }

    def gather_club_info(self, club_name: str, city: str, state: str) -> str:
        correlation_id = f"club_info_{club_name}"
        logger.debug("Starting club info search", extra={
            "club_name": club_name,
            "city": city,
            "state": state,
            "correlation_id": correlation_id
        })
        location_str = f"{city}, {state}".strip(", ")
        try:
            info = xai_club_info_search(club_name, location_str, amenities=None)
            logger.info("Club info search completed", extra={
                "club_name": club_name,
                "has_info": bool(info),
                "correlation_id": correlation_id
            })
            return info
        except Exception as e:
            logger.error("Error searching club info", extra={
                "club_name": club_name,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    def gather_club_news(self, club_name: str) -> str:
        correlation_id = f"club_news_{club_name}"
        logger.debug("Starting club news search", extra={
            "club_name": club_name,
            "correlation_id": correlation_id
        })
        try:
            news = xai_news_search(club_name)
            if isinstance(news, tuple):
                # If xai_news_search returns (news, icebreaker)
                news = news[0]
            logger.info("Club news search completed", extra={
                "club_name": club_name,
                "has_news": bool(news),
                "correlation_id": correlation_id
            })
            return news
        except Exception as e:
            logger.error("Error searching club news", extra={
                "club_name": club_name,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    def market_research(self, company_name: str) -> Dict[str, Any]:
        """Just a wrapper around gather_club_news for example."""
        news_response = self.gather_club_news(company_name)
        return {
            "company_overview": news_response,
            "recent_news": [{
                "title": "Recent News",
                "snippet": news_response,
                "link": "",
                "date": ""
            }] if news_response else [],
            "status": "success",
            "error": ""
        }

    def check_facilities(self, company_name: str, city: str, state: str) -> Dict[str, str]:
        correlation_id = f"facilities_{company_name}"
        if not company_name or not city or not state:
            return {
                "response": "",
                "status": "no_data"
            }
        location_str = f"{city}, {state}".strip(", ")
        try:
            response = xai_club_info_search(company_name, location_str, amenities=["Golf Course", "Pool", "Tennis Courts"])
            return {
                "response": response,
                "status": "success"
            }
        except Exception as e:
            logger.error("Error checking facilities", extra={
                "company": company_name,
                "city": city,
                "state": state,
                "error_type": type(e).__name__,
                "error": str(e),
                "correlation_id": correlation_id
            }, exc_info=True)
            return {
                "response": "",
                "status": "error"
            }

    def review_previous_interactions(self, contact_id: str) -> Dict[str, Union[int, str]]:
        """
        Review previous interactions for a contact using HubSpot data.
        """
        try:
            lead_data = self.hubspot.get_contact_properties(contact_id)
            if not lead_data:
                return {
                    "emails_opened": 0,
                    "emails_sent": 0,
                    "meetings_held": 0,
                    "last_response": "No data available",
                    "status": "no_data",
                    "error": "Contact not found in HubSpot"
                }

            emails_opened = self._safe_int(lead_data.get("total_opens_weekly"))
            emails_sent = self._safe_int(lead_data.get("num_contacted_notes"))
            notes = self.hubspot.get_all_notes_for_contact(contact_id)

            meeting_keywords = {"meeting", "meet", "call", "zoom", "teams"}
            meetings_held = 0
            for note in notes:
                if note.get("body") and any(keyword in note["body"].lower() for keyword in meeting_keywords):
                    meetings_held += 1

            last_reply = lead_data.get("hs_sales_email_last_replied")
            if last_reply:
                try:
                    dt = parse_date(last_reply.replace("Z", "+00:00"))
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    if not dt.tzinfo:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    days_ago = (now_utc - dt).days
                    last_response = f"Responded {days_ago} days ago"
                except ValueError:
                    last_response = "Responded recently"
            else:
                last_response = "No recent response" if emails_opened == 0 else "Opened emails but no direct reply"

            return {
                "emails_opened": emails_opened,
                "emails_sent": emails_sent,
                "meetings_held": meetings_held,
                "last_response": last_response,
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error("Failed to review contact interactions", extra={
                "contact_id": contact_id,
                "error_type": type(e).__name__,
                "error": str(e)
            })
            return {
                "emails_opened": 0,
                "emails_sent": 0,
                "meetings_held": 0,
                "last_response": "Error retrieving data",
                "status": "error",
                "error": f"Error retrieving interaction data: {str(e)}"
            }

    def _safe_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def load_season_data(self) -> None:
        global CITY_ST_DATA, ST_DATA
        try:
            with CITY_ST_CSV.open('r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    city = row['City'].strip().lower()
                    st = row['State'].strip().lower()
                    CITY_ST_DATA[(city, st)] = row

            with ST_CSV.open('r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    st = row['State'].strip().lower()
                    ST_DATA[st] = row

            logger.info("Successfully loaded golf season data", extra={
                "city_state_count": len(CITY_ST_DATA),
                "state_count": len(ST_DATA)
            })
        except Exception as e:
            logger.error("Failed to load golf season data", extra={
                "error": str(e),
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            raise

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        """
        if not city and not state:
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "no_data",
                "error": "No location data provided"
            }

        city_key = (city.lower(), state.lower())
        row = CITY_ST_DATA.get(city_key)
        if not row:
            row = ST_DATA.get(state.lower())

        if not row:
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "no_data",
                "error": "Location not found"
            }

        year_round = row["Year-Round?"].strip()
        start_month_str = row["Start Month"].strip()
        end_month_str = row["End Month"].strip()
        peak_season_start_str = row["Peak Season Start"].strip() or "May"
        peak_season_end_str = row["Peak Season End"].strip() or "August"

        return {
            "year_round": year_round,
            "start_month": start_month_str,
            "end_month": end_month_str,
            "peak_season_start": self._month_to_first_day(peak_season_start_str),
            "peak_season_end": self._month_to_last_day(peak_season_end_str),
            "status": "success",
            "error": ""
        }

    def _month_to_first_day(self, month_name: str) -> str:
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-01"
        return "05-01"

    def _month_to_last_day(self, month_name: str) -> str:
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
        return "08-31"

    def _save_lead_context(self, lead_sheet: Dict[str, Any], lead_email: str) -> None:
        """
        Optionally save the lead_sheet for debugging or offline reference.
        """
        # (Implementation detail)
        pass

    def load_state_timezones(self) -> None:
        """Load state timezone offsets from CSV file."""
        global STATE_TIMEZONES
        try:
            with open(TIMEZONE_CSV_PATH, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    state_code = row['state_code'].strip()
                    STATE_TIMEZONES[state_code] = {
                        'dst': int(row['daylight_savings']),
                        'std': int(row['standard_time'])
                    }
            logger.debug(f"Loaded timezone data for {len(STATE_TIMEZONES)} states")
        except Exception as e:
            logger.error(f"Error loading state timezones: {str(e)}")
            # Default to Eastern Time if loading fails
            STATE_TIMEZONES = {}
    
    def get_club_timezone(self, state: str) -> dict:
        """
        Get timezone offset data for a given state.
        
        Args:
            state (str): Two-letter state code
            
        Returns:
            dict: Dictionary containing DST and standard time offsets
                  {'dst': int, 'std': int}
        """
        state_code = state.upper() if state else ''
        timezone_data = STATE_TIMEZONES.get(state_code, {
            'dst': -4,  # Default to Eastern Time
            'std': -5
        })
        
        logger.debug("Retrieved timezone data for state", extra={
            "state": state_code,
            "dst_offset": timezone_data['dst'],
            "std_offset": timezone_data['std']
        })
        
        return timezone_data

    def get_club_geography_and_type(self, club_name: str, city: str, state: str) -> tuple:
        """
        Get club geography and type based on location and HubSpot data.
        
        Args:
            club_name (str): Name of the club
            city (str): City where the club is located
            state (str): State where the club is located
            
        Returns:
            tuple: (geography: str, club_type: str)
        """
        try:
            # Try to get company data from HubSpot
            company_data = self.hubspot.get_company_data(club_name)
            geography = self.determine_geography(city, state)
            
            # Determine club type from HubSpot data
            club_type = company_data.get("type", "Public Clubs")
            if not club_type or club_type.lower() == "unknown":
                club_type = "Public Clubs"
            
            return geography, club_type
            
        except HubSpotError:
            # If company not found in HubSpot, use defaults
            geography = self.determine_geography(city, state)
            return geography, "Public Clubs"

    def determine_geography(self, city: str, state: str) -> str:
        """
        Determine geography string from city and state.
        
        Args:
            city (str): City name
            state (str): State code
            
        Returns:
            str: Geography string in format "City, State" or "Unknown"
        """
        if not city or not state:
            return "Unknown"
        return f"{city}, {state}"




## utils\logging_setup.py

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextlib import contextmanager
from config.settings import DEBUG_MODE
import json

class StepLogger(logging.Logger):
    def step_complete(self, step_number: int, message: str):
        self.info(f"✓ Step {step_number}: {message}")
        
    def step_start(self, step_number: int, message: str):
        self.debug(f"Starting Step {step_number}: {message}")

def setup_logging():
    # Register custom logger class
    logging.setLoggerClass(StepLogger)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Configure formatters with more detail
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: [%(pathname)s:%(lineno)d]\n'
        '%(extra_data)s'  # New field for extra data
    )
    
    # Add custom filter to handle extra data
    class DetailedExtraDataFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'extra_data'):
                record.extra_data = ''
            elif record.extra_data:
                if isinstance(record.extra_data, dict):
                    # Pretty print with increased depth and width
                    record.extra_data = '\n' + json.dumps(
                        record.extra_data,
                        indent=2,
                        ensure_ascii=False,  # Properly handle Unicode
                        default=str  # Handle non-serializable objects
                    )
                    # Add separator lines for readability
                    record.extra_data = (
                        '\n' + '='*80 + '\n' +
                        record.extra_data +
                        '\n' + '='*80
                    )
            return True

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with increased size limit
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        delay=True  # Only create file when first record is written
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(DetailedExtraDataFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress external library logs except warnings
    for logger_name in ['urllib3', 'googleapiclient', 'google.auth', 'openai']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

def workflow_step(step_num: int, description: str):
    """Context manager for workflow steps."""
    class WorkflowStep:
        def __enter__(self):
            logger.debug(f"Starting Step {step_num}: {description}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                logger.info(f"✓ Step {step_num}: {description}")
            return False

    return WorkflowStep()

# Make it available for import
__all__ = ['logger', 'workflow_step']




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
from pathlib import Path
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
    "club_info": {},
    "icebreakers": {},
}

SUBJECT_TEMPLATES = [
    "Quick Chat, [FirstName]?",
    "Quick Question, [FirstName]?",
    "Swoop: [ClubName]'s Edge?",
    "Question about 2025",
    "Quick Question",
]


def get_random_subject_template() -> str:
    """Returns a random subject line template from the predefined list."""
    return random.choice(SUBJECT_TEMPLATES)


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
                    "max_tokens": payload.get("max_tokens", 2000),
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

                logger.info(
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
    logger.info(
        "News search result for %s:",
        club_name,
        extra={"news": news},
    )

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
    logger.info(
        "Generated icebreaker for %s:",
        club_name,
        extra={"icebreaker": icebreaker},
    )

    return icebreaker


##############################################################################
# Club Info Search
##############################################################################
def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> Dict[str, Any]:
    """
    Search for club information using xAI.
    Returns a dict with keys:
      'overview', 'facility_type', 'has_pool', 'amenities'
    """
    cache_key = f"{club_name}_{location}"
    if cache_key in _cache["club_info"]:
        logger.debug(f"Using cached club info for {club_name} in {location}")
        return _cache["club_info"][cache_key]

    logger.info(f"Searching for club info: {club_name} in {location}")
    amenity_str = ", ".join(amenities) if amenities else ""

    prompt = f"""
    Please provide a brief overview of {club_name} located in {location}. Include key facts such as:
    - Type of facility (Public, Private, Municipal, Semi-Private, Country Club, Resort, Management Company)
    - Does the club have a pool? (Answer with 'Yes' or 'No')
    - Notable amenities or features (DO NOT include pro shop, fitness center, or dining facilities)
    - Any other relevant information

    Format your response with these exact headings:
    OVERVIEW:
    [Overview text]

    FACILITY TYPE:
    [Type]

    HAS POOL:
    [Yes/No]

    AMENITIES:
    - [amenity 1]
    - [amenity 2]
    """

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a factual assistant that provides objective, data-focused overviews of clubs. "
                    "Focus only on verifiable facts like location, type, amenities, etc. "
                    "CRITICAL: DO NOT list golf course or pro shop as amenities as these are assumed. "
                    "Only list additional amenities explicitly verified. "
                    "Avoid subjective descriptions or flowery language."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "model": MODEL_NAME,
        "temperature": ANALYSIS_TEMPERATURE,
    }

    response = _send_xai_request(payload)
    logger.info(
        "Club info search result for %s:",
        club_name,
        extra={"info": response},
    )

    parsed_info = _parse_club_response(response)
    _cache["club_info"][cache_key] = parsed_info

    return parsed_info


def _parse_club_response(response: str) -> Dict[str, Any]:
    """
    Parse structured club information response.
    """
    result = {
        "overview": "",
        "facility_type": "Other",
        "has_pool": "No",
        "amenities": [],
    }

    sections = {
        "overview": re.search(
            r"OVERVIEW:\s*(.+?)(?=FACILITY TYPE:|$)", response, re.DOTALL
        ),
        "facility_type": re.search(
            r"FACILITY TYPE:\s*(.+?)(?=HAS POOL:|$)", response, re.DOTALL
        ),
        "has_pool": re.search(
            r"HAS POOL:\s*(.+?)(?=AMENITIES:|$)", response, re.DOTALL
        ),
        "amenities": re.search(r"AMENITIES:\s*(.+?)$", response, re.DOTALL),
    }

    if sections["overview"]:
        result["overview"] = sections["overview"].group(1).strip()

    if sections["facility_type"]:
        result["facility_type"] = sections["facility_type"].group(1).strip()

    if sections["has_pool"]:
        pool_value = sections["has_pool"].group(1).strip().lower()
        result["has_pool"] = "Yes" if "yes" in pool_value else "No"

    if sections["amenities"]:
        amenities_text = sections["amenities"].group(1)
        result["amenities"] = [
            a.strip("- ").strip()
            for a in amenities_text.split("\n")
            if a.strip("- ").strip()
        ]

    return result

##############################################################################
# Personalize Email
##############################################################################
def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = "",
    club_info: str = "",
) -> Tuple[str, str]:
    """
    Personalizes email content using xAI.
    Returns a tuple of (subject, body).
    """
    try:
        previous_interactions = lead_sheet.get("analysis", {}).get("previous_interactions", {})
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
            "You are a helpful assistant that personalizes outreach emails for golf clubs, focusing on business value and relevant solutions."
        )

        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_sheet.get("company_data", {})
        
        context_block = build_context_block(
            interaction_history=summary if summary else "No previous interactions",
            club_details=club_info if club_info else "",
            objection_handling=objection_handling if has_prior_emails else "",
            original_email={"subject": subject, "body": body},
        )
        logger.debug(f"Context block: {json.dumps(context_block, indent=2)}")

        rules_text = "\n".join(get_email_rules())
        user_message = (
            "You are an expert at personalizing sales emails for golf industry outreach. "
            "CRITICAL RULES:\n"
            "1. DO NOT modify the subject line\n"
            "2. DO NOT reference weather or seasonal conditions unless specifically provided\n"
            "3. DO NOT reference any promotions from previous emails\n"
            "4. Focus on the business value and problem-solving aspects\n"
            "5. Avoid presumptive descriptions of club features or scenery\n"
            "6. Keep references to club specifics brief and relevant to the service\n"
            "7. Keep the tone professional and direct\n\n"
            "Format response as:\n"
            "Subject: [keep original subject]\n\n"
            "Body:\n[personalized body]\n\n"
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{rules_text}\n\n"
            "TASK:\n"
            "1. Keep the original email structure and flow\n"
            "2. Add relevant context about the club's specific features\n"
            "3. Maintain professional tone\n"
            "4. Return ONLY the subject and body"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "model": MODEL_NAME,
            "temperature": EMAIL_TEMPERATURE,
        }

        logger.info(
            "Personalizing email for:",
            extra={
                "company": lead_sheet.get("company_name"),
                "original_subject": subject,
            },
        )
        response = _send_xai_request(payload)
        logger.info(
            "Email personalization result:",
            extra={"company": lead_sheet.get("company_name"), "response": response},
        )

        return _parse_xai_response(response)

    except Exception as e:
        logger.error(f"Error in email personalization: {str(e)}")
        return subject, body


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
                elif line in ["Best regards,", "Sincerely,", "Regards,"]:
                    body_lines.append(f"\n{line}")
                elif line == "Ty":
                    body_lines.append(f" {line}\n\n")
                elif line == "Swoop Golf":
                    body_lines.append(f"{line}\n")
                elif line == "480-225-9702":
                    body_lines.append(f"{line}\n")
                elif line == "swoopgolf.com":
                    body_lines.append(line)
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
    """
    Get a personalized icebreaker from the xAI service (with caching if desired).
    """
    cache_key = f"icebreaker:{club_name}:{recipient_name}"
    if cache_key in _cache["icebreakers"]:
        if DEBUG_MODE:
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
    logger.info(
        "Generated icebreaker for %s:",
        club_name,
        extra={"icebreaker": response},
    )

    _cache["icebreakers"][cache_key] = response
    return response

# def get_email_critique(email_subject: str, email_body: str, guidance: dict) -> str:
#     """
#     Get expert critique of the email draft.
#     """
#     rules = get_email_rules()
#     rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
# 
#     payload = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an expert at critiquing emails using specific rules. "
#                     "Analyze the email draft and provide specific critiques focusing on:\n"
#                     f"{rules_text}\n"
#                     "Provide actionable recommendations for improvement."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": f"""
#                 Original Email:
#                 Subject: {email_subject}
#                 
#                 Body:
#                 {email_body}
#                 
#                 Original Guidance:
#                 {json.dumps(guidance, indent=2)}
#                 
#                 Please provide specific critiques and recommendations for improvement.
#                 """,
#             },
#         ],
#         "model": MODEL_NAME,
#         "stream": False,
#         "temperature": EMAIL_TEMPERATURE,
#     }
# 
#     logger.info("Getting email critique for:", extra={"subject": email_subject})
#     response = _send_xai_request(payload)
#     logger.info("Email critique result:", extra={"critique": response})
#     return response

# def revise_email_with_critique(email_subject: str, email_body: str, critique: str) -> tuple[str, str]:
#     """
#     Revise the email based on the critique.
#     Returns a tuple of (new_subject, new_body).
#     """
#     rules = get_email_rules()
#     rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
# 
#     payload = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a renowned expert at cold email outreach, similar to Alex Berman. "
#                     "Apply your proven methodology to rewrite this email. "
#                     "Use all of your knowledge just as you teach in Cold Email University."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": f"""
#                 Original Email:
#                 Subject: {email_subject}
#                 
#                 Body:
#                 {email_body}
#                 
#                 Instructions:
#                 {rules_text}
# 
#                 Expert Critique:
#                 {critique}
#                 
#                 Please rewrite the email incorporating these recommendations.
#                 Format the response as:
#                 Subject: [new subject]
#                 
#                 Body:
#                 [new body]
#                 """,
#             },
#         ],
#         "model": MODEL_NAME,
#         "stream": False,
#         "temperature": EMAIL_TEMPERATURE,
#     }
# 
#     logger.info(
#         "Revising email with critique for:",
#         extra={"subject": email_subject},
#     )
#     result = _send_xai_request(payload)
#     logger.info("Email revision result:", extra={"result": result})
#     return _parse_xai_response(result)


def generate_followup_email_content(
    first_name: str,
    company_name: str,
    original_subject: str,
    original_date: str,
    sequence_num: int,
    original_email: dict = None,
) -> Tuple[str, str]:
    """
    Generate follow-up email content using xAI.
    Returns (subject, body).
    """
    logger.debug(
        f"[generate_followup_email_content] Called with first_name='{first_name}', "
        f"company_name='{company_name}', original_subject='{original_subject}', "
        f"original_date='{original_date}', sequence_num={sequence_num}, "
        f"original_email keys={list(original_email.keys()) if original_email else 'None'}"
    )
    try:
        if sequence_num == 2 and original_email:
            logger.debug("[generate_followup_email_content] Handling second follow-up logic.")

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a sales professional writing a brief follow-up email. "
                            "Keep the tone professional but friendly. "
                            "The response should be under 50 words and focus on getting a response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Original Email:
                        Subject: {original_email['subject']}
                        Body: {original_email['body']}

                        Write a brief follow-up that starts with:
                        "I understand my last email might not have made it high enough in your inbox."

                        Requirements:
                        - Keep it under 50 words
                        - Be concise and direct
                        - End with a clear call to action
                        - Don't repeat information from the original email
                        """,
                    },
                ],
                "model": MODEL_NAME,
                "stream": False,
                "temperature": EMAIL_TEMPERATURE,
            }

            logger.info(
                "Generating second follow-up email for:",
                extra={"company": company_name, "sequence_num": sequence_num},
            )
            result = _send_xai_request(payload)
            logger.info("Second follow-up generation result:", extra={"result": result})

            if not result:
                logger.error(
                    "[generate_followup_email_content] Empty response from xAI for follow-up generation."
                )
                return "", ""

            follow_up_body = result.strip()
            subject = f"RE: {original_email['subject']}"
            body = (
                f"{follow_up_body}\n\n"
                f"Best regards,\n"
                f"Ty\n\n"
                f"-------- Original Message --------\n"
                f"Subject: {original_email['subject']}\n"
                f"Sent: {original_email.get('created_at', 'Unknown date')}\n\n"
                f"{original_email['body']}"
            )

            logger.debug(
                f"[generate_followup_email_content] Returning second follow-up subject='{subject}', "
                f"body length={len(body)}"
            )
            return subject, body

        else:
            logger.debug("[generate_followup_email_content] Handling default follow-up logic.")
            prompt = f"""
            Generate a follow-up email for:
            - Name: {first_name}
            - Company: {company_name}
            - Original Email Subject: {original_subject}
            - Original Send Date: {original_date}

            This is follow-up #{sequence_num}. Keep it brief and focused on scheduling a call.

            Rules:
            1. Keep it under 3 short paragraphs
            2. Reference the original email naturally
            3. Add new value proposition or insight
            4. End with clear call to action
            5. Maintain professional but friendly tone

            Format the response with 'Subject:' and 'Body:' labels.
            """

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at writing follow-up emails that are brief, "
                            "professional, and effective."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "model": MODEL_NAME,
                "temperature": EMAIL_TEMPERATURE,
            }

            logger.info(
                "Generating follow-up email for:",
                extra={"company": company_name, "sequence_num": sequence_num},
            )
            result = _send_xai_request(payload)
            logger.info("Follow-up generation result:", extra={"result": result})

            if not result:
                logger.error(
                    "[generate_followup_email_content] Empty response from xAI for default follow-up generation."
                )
                return "", ""

            subject, body = _parse_xai_response(result)
            if not subject or not body:
                logger.error(
                    "[generate_followup_email_content] Failed to parse follow-up email content."
                )
                return "", ""

            logger.debug(
                f"[generate_followup_email_content] Returning subject='{subject}', body length={len(body)}"
            )
            return subject, body

    except Exception as e:
        logger.error(
            f"[generate_followup_email_content] Error generating follow-up email content: {str(e)}"
        )
        return "", ""


def parse_personalization_response(response_text):
    """
    Parse JSON-based email personalization response text from xAI.
    Returns (subject, body).
    Falls back to defaults on parsing error.
    """
    try:
        response_data = json.loads(response_text)
        subject = response_data.get("subject")
        body = response_data.get("body")

        if not subject or not body:
            raise ValueError("Subject or body missing in xAI response")

        return subject, body

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception(f"Error parsing xAI response: {str(e)}")
        return "Follow-up", (
            "Thank you for your interest. Let me know if you have any other questions!"
        )


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
1. **CLUB TYPE**: Is it Private, Public - High Daily Fee, Public - Low Daily Fee, Municipal, Resort, Country Club, or Unknown?
2. **FACILITY COMPLEXITY**: Single-Course, Multi-Course, or Unknown?
3. **GEOGRAPHIC SEASONALITY**: Year-Round or Seasonal?
4. **POOL**: ONLY answer 'Yes' if you find clear, direct evidence of a pool.
5. **TENNIS COURTS**: ONLY answer 'Yes' if there's explicit evidence.
6. **GOLF HOLES**: Verify from official sources or consistent user mentions.

CRITICAL RULES:
- **Do not assume amenities based on the type or perceived status of the club.**
- **Confirm amenities only with solid evidence; otherwise, use 'Unknown'.**
- **Use specific references for each answer where possible.**

Format your response with these exact headings:
OFFICIAL NAME:
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
    logger.info(
        "Club segmentation result for %s:",
        club_name,
        extra={"segmentation": response},
    )

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
        'club_type': 'Unknown',
        'facility_complexity': 'Unknown',
        'geographic_seasonality': 'Unknown',
        'has_pool': 'Unknown',
        'has_tennis_courts': 'Unknown',
        'number_of_holes': 0,
        'analysis_text': ''
    }
    
    # Add name detection pattern
    name_match = re.search(r'(?:OFFICIAL NAME|NAME):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    if name_match:
        result['name'] = clean_value(name_match.group(1))
    
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

    # Process number of holes with better number extraction
    if sections['holes']:
        holes_text = clean_value(sections['holes'].group(1)).lower()
        logger.debug(f"Processing holes text: '{holes_text}'")
        
        # Try to extract number from text
        number_match = re.search(r'(\d+)', holes_text)
        if number_match:
            try:
                result['number_of_holes'] = int(number_match.group(1))
                logger.debug(f"Found {result['number_of_holes']} holes")
            except ValueError:
                logger.warning(f"Could not convert {number_match.group(1)} to integer")
        
        # Handle text numbers for multiple courses
        if 'three' in holes_text and '18' in holes_text:
            result['number_of_holes'] = 54
        elif 'two' in holes_text and '18' in holes_text:
            result['number_of_holes'] = 36

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

    # First get verified info from segmentation and club info searches
    segmentation_info = xai_club_segmentation_search(club_name, location)
    club_info = xai_club_info_search(club_name, location)

    # Get overview from club_info response
    overview_match = re.search(r'OVERVIEW:\s*(.+?)(?=\n[A-Z ]+?:|$)', club_info.get('raw_response', ''), re.DOTALL)
    overview = overview_match.group(1).strip() if overview_match else ""

    # Create strict system prompt based on verified info
    verified_info = {
        'type': segmentation_info.get('club_type', 'Unknown'),
        'holes': segmentation_info.get('number_of_holes', 0),
        'has_pool': segmentation_info.get('has_pool', 'No'),
        'has_tennis': segmentation_info.get('has_tennis_courts', 'No'),
        'overview': overview
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a club analyst. Provide a simple one paragraph summary."
            },
            {
                "role": "user", 
                "content": f"Give me a one paragraph summary about {club_name} in {location}."
            }
        ],
        "model": MODEL_NAME,
        "temperature": 0.0,
    }

    logger.info(f"Generating club summary for: {club_name} in {location}")
    response = _send_xai_request(payload)
    logger.info("Generated club summary:", extra={"summary": response})

    return response.strip()

def build_context_block(interaction_history=None, club_details=None, objection_handling=None, original_email=None):
    context = {}
    
    # Only add non-empty fields
    if interaction_history:
        context["interaction_history"] = interaction_history
    
    if club_details:
        context["club_details"] = club_details
        
    if objection_handling:
        context["objection_handling"] = objection_handling
        
    if original_email:
        context["original_email"] = {
            "subject": original_email.get("subject", ""),
            "body": original_email.get("body", "")
        }
        # Remove original_email if both subject and body are empty
        if not context["original_email"]["subject"] and not context["original_email"]["body"]:
            del context["original_email"]
    
    return context
