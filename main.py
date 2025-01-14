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
from scheduling.extended_lead_storage import upsert_full_lead
from utils.exceptions import LeadContextError


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
                    try:
                        file.unlink()
                    except PermissionError:
                        # Skip files that are currently in use
                        logger.debug(f"Skipping locked file: {file}")
                        continue
            logger.debug("Cleared logs directory")
            
        # Clear temp directory
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

            # Get analysis data
            analysis_data = {
                "competitor_analysis": data_gatherer.check_competitor_on_website(
                    company_props.get("website", "")
                ),
                "season_data": {
                    "year_round": company_props.get("geographic_seasonality") == "Year-Round Golf",
                    "start_month": company_props.get("season_start", ""),
                    "end_month": company_props.get("season_end", ""),
                    "peak_season_start": company_props.get("peak_season_start", ""),
                    "peak_season_end": company_props.get("peak_season_end", "")
                }
            }

            # Gather personalization data BEFORE creating lead_sheet
            print("Gathering personalization data...")
            personalization_data = gather_personalization_data(
                company_props.get("name", ""),
                company_props.get("city", ""),
                company_props.get("state", "")
            )

            # Calculate season data
            outreach_window = get_best_outreach_window(
                persona=lead_data["jobtitle"],
                geography=company_props.get("geographic_seasonality", "Year-Round Golf"),
                club_type=company_props.get("club_type", "Country Club")
            )
            
            # Get season months
            best_months = outreach_window["Best Month"]
            season_data = {
                "year_round": company_props.get("geographic_seasonality") == "Year-Round Golf",
                "start_month": min(best_months) if best_months else "",
                "end_month": max(best_months) if best_months else "",
                "peak_season_start": f"{min(best_months)}-01" if best_months else "",
                "peak_season_end": f"{max(best_months)}-28" if best_months else ""
            }

            # Create lead_sheet with calculated season data
            lead_sheet = {
                "metadata": {
                    "status": "success",
                    "correlation_id": workflow_context['correlation_id'],
                    "lead_email": lead_data["email"]
                },
                "lead_data": {
                    "email": lead_data["email"],
                    "properties": {
                        "firstname": lead_props.get("firstname", ""),
                        "lastname": lead_props.get("lastname", ""),
                        "jobtitle": lead_props.get("jobtitle", "General Manager"),
                        "hs_object_id": lead_props.get("hs_object_id", ""),
                        "createdate": lead_props.get("createdate", ""),
                        "lastmodifieddate": lead_props.get("lastmodifieddate", ""),
                        "lifecyclestage": lead_props.get("lifecyclestage", ""),
                        "phone": lead_props.get("phone", "")
                    },
                    "company_data": {
                        "name": company_props.get("name", ""),
                        "city": company_props.get("city", ""),
                        "state": company_props.get("state", ""),
                        "company_type": company_props.get("club_type", ""),
                        "hs_object_id": company_props.get("hs_object_id", ""),
                        "createdate": company_props.get("createdate", ""),
                        "lastmodifieddate": company_props.get("lastmodifieddate", ""),
                        "annualrevenue": company_props.get("annualrevenue", "")
                    }
                },
                "analysis": {
                    "competitor_analysis": analysis_data.get("competitor_analysis", ""),
                    "season_data": season_data,
                    "facilities": {
                        "response": company_props.get("club_info", "")
                    },
                    "research_data": {
                        "recent_news": [
                            {"snippet": personalization_data.get("news", "")}
                        ] if personalization_data.get("news") else []
                    }
                }
            }

            # Log the company type for debugging
            logger.debug("Company type determined", extra={
                "company_name": company_props.get("name", ""),
                "company_type": company_props.get("club_type", ""),
                "correlation_id": workflow_context['correlation_id']
            })

            # Step 7: Store lead data in database
            with workflow_step("7", "Upserting lead data into DB", workflow_context):
                try:
                    upsert_full_lead(lead_sheet, correlation_id=workflow_context['correlation_id'])
                    logger.info("Successfully upserted lead data", extra=workflow_context)
                except Exception as e:
                    logger.error(f"Database upsert failed: {str(e)}", extra=workflow_context)
                    continue
            
            # Summarize interactions
            print("Summarizing interactions...")
            interaction_summary = summarize_lead_interactions(lead_sheet)
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
        
    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    

    main()
