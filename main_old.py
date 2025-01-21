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
                "firstname": first_name,
                "LastName": last_name,
                "clubname": club_name or "Your Club",
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
