import csv
import logging
from pathlib import Path
import os
import shutil
import random
from datetime import timedelta, datetime
import json

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
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY, PROJECT_ROOT, CLEAR_LOGS_ON_START, USE_RANDOM_LEAD, TEST_EMAIL, CREATE_FOLLOWUP_DRAFT
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
            state_code = row['state_code'].strip()      # matches CSV column name
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
                date = email.get('timestamp', '').split('T')[0]  # Extract just the date
                subject = email.get('subject', '').encode('utf-8', errors='ignore').decode('utf-8')
                body = email.get('body_text', '').encode('utf-8', errors='ignore').decode('utf-8')
                direction = email.get('direction', '')
                
                # Only include relevant parts of the email thread
                body = body.split('On ')[0].strip()  # Take only the most recent part
                
                # Add clear indication of email direction
                email_type = "from the lead" if direction == "INCOMING_EMAIL" else "to the lead"
                
                interaction = {
                    'date': date,
                    'type': f'email {email_type}',
                    'direction': direction,
                    'subject': subject,
                    'notes': body[:1000]  # Limit length to prevent token overflow
                }
                interactions.append(interaction)
        
        # Add notes with proper encoding handling
        for note in sorted(notes, key=lambda x: x.get('timestamp', ''), reverse=True):
            if isinstance(note, dict):
                date = note.get('timestamp', '').split('T')[0]
                content = note.get('body', '').encode('utf-8', errors='ignore').decode('utf-8')
                
                interaction = {
                    'date': date,
                    'type': 'note',
                    'direction': 'internal',
                    'subject': 'Internal Note',
                    'notes': content[:1000]  # Limit length to prevent token overflow
                }
                interactions.append(interaction)
        
        if not interactions:
            return "No prior interactions found."

        # Sort all interactions by date
        interactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Take only the last 10 interactions to keep the context focused
        recent_interactions = interactions[:10]
        
        prompt = (
            "Please summarize these interactions, focusing on:\n"
            "1. Most recent email FROM THE LEAD if there is one (note if no emails from lead exist)\n"
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
        
        # Get summary from OpenAI
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=MODEL_FOR_GENERAL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes business interactions. Anything coming from Ty or Ryan is from Swoop, otherwise it's from the lead."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            summary = response.choices[0].message.content.strip()
            # Remove debug logging, keep only essential info
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
def get_random_lead_email() -> str:
    """Get a random lead email from HubSpot."""
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        logger.debug("Calling hubspot.get_random_contacts()")
        contacts = hubspot.get_random_contacts(count=1)
        logger.debug(f"Received contacts: {contacts}")
        
        if contacts and contacts[0].get('email'):
            email = contacts[0]['email']
            logger.info(f"Selected random lead: {email}")
            return email
        else:
            logger.warning("No random lead found, falling back to TEST_EMAIL")
            return TEST_EMAIL
    except Exception as e:
        logger.error(f"Error getting random lead: {e}")
        return TEST_EMAIL

def get_lead_email() -> str:
    """Get lead email either randomly or via user input based on settings."""
    # Add debug logging
    logger.debug(f"USE_RANDOM_LEAD setting is: {USE_RANDOM_LEAD}")
    logger.debug(f"TEST_EMAIL setting is: {TEST_EMAIL}")
    
    if USE_RANDOM_LEAD:
        logger.debug("Using random lead selection")
        try:
            hubspot = HubspotService(HUBSPOT_API_KEY)
            contacts = hubspot.get_random_contacts(count=1)
            logger.debug(f"Retrieved contacts: {contacts}")
            
            if contacts and contacts[0].get('email'):
                email = contacts[0]['email']
                logger.info(f"Selected random lead: {email}")
                return email
            else:
                logger.warning("No random lead found, falling back to TEST_EMAIL")
                return TEST_EMAIL
        except Exception as e:
            logger.error(f"Error getting random lead: {e}")
            return TEST_EMAIL
    else:
        logger.debug("Using manual email input")
        while True:
            email = input("\nPlease enter a lead's email address: ").strip()
            if email:
                return email
            print("Email address cannot be empty. Please try again.")

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
            # Add logging for email data
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

        # Step 5: Extract relevant data
        with workflow_step(5, "Extracting lead data"):
            lead_data = lead_sheet.get("lead_data", {})
            company_data = lead_data.get("company_data", {})

            first_name = lead_data.get("properties", {}).get("firstname", "")
            last_name = lead_data.get("properties", {}).get("lastname", "")
            club_name = company_data.get("name", "").strip()
            city = company_data.get("city", "").strip()
            state = company_data.get("state", "").strip()

            # Replace the old season logic with new season snippet logic
            current_month = datetime.now().month
            
            # Define peak season months based on state
            if state == "AZ":
                start_peak_month = 0  # January
                end_peak_month = 11   # December (year-round)
            else:
                start_peak_month = 5  # June
                end_peak_month = 8    # September

            # Get the season state and appropriate snippet
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
                "SEASON_VARIATION": season_text.rstrip(',')  # Remove trailing comma if present
            }
            logger.debug("Placeholders built", extra={
                **placeholders,
                "season_key": season_key
            })

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
            # Get random subject template before xAI processing
            subject = get_random_subject_template()
            
            # Replace placeholders in subject
            for key, val in placeholders.items():
                subject = subject.replace(f"[{key}]", val)
            
            # Get body from template as usual
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
                        # If no icebreaker generated, remove placeholder and extra newlines
                        body = body.replace("[ICEBREAKER]\n\n", "")
                        body = body.replace("[ICEBREAKER]\n", "")
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    # No news, remove icebreaker placeholder and extra newlines
                    body = body.replace("[ICEBREAKER]\n\n", "")
                    body = body.replace("[ICEBREAKER]\n", "")
                    body = body.replace("[ICEBREAKER]", "")
            except Exception as e:
                logger.error(f"Icebreaker generation error: {e}")
                # Remove placeholder and extra newlines
                body = body.replace("[ICEBREAKER]\n\n", "")
                body = body.replace("[ICEBREAKER]\n", "")
                body = body.replace("[ICEBREAKER]", "")

            # Clean up any multiple consecutive newlines that might be left
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")

            orig_subject, orig_body = subject, body
            for key, val in placeholders.items():
                subject = subject.replace(f"[{key}]", val)
                body = body.replace(f"[{key}]", val)

        # Step 9: Personalize with xAI
        with workflow_step(9, "Personalizing with AI"):
            try:
                # Get lead email first to avoid reference error
                lead_email = lead_data.get("email", email)
                
                # Get initial draft from xAI
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

                # Re-apply placeholders if xAI reintroduced them
                for key, val in placeholders.items():
                    subject = subject.replace(f"[{key}]", val)
                    body = body.replace(f"[{key}]", val)

                # Insert ICEBREAKER if we have news
                if has_news:
                    try:
                        icebreaker = _build_icebreaker_from_news(club_name, news_result)
                        if icebreaker:
                            body = body.replace("[ICEBREAKER]", icebreaker)
                        else:
                            # If no icebreaker generated, just remove the placeholder
                            body = body.replace("[ICEBREAKER]", "")
                    except Exception as e:
                        logger.error(f"Icebreaker generation error: {e}")
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    # No news, remove icebreaker placeholder
                    body = body.replace("[ICEBREAKER]", "")

                logger.debug("Creating email draft", extra={
                    **logger_context,
                    "to": lead_email,
                    "subject": subject
                })
                # # Get expert critique
                # critique = get_email_critique(subject, body, {
                #     "club_name": club_name,
                #     "lead_name": first_name,
                #     "interaction_history": interaction_summary,
                #     "news": news_result,
                #     "club_info": club_info_snippet
                # })
                
                # if critique:
                #     logger.info("Received email critique", extra={
                #         "critique_length": len(critique)
                #     })
                    
                #     # Revise email based on critique
                #     subject, body = revise_email_with_critique(subject, body, critique)
                #     logger.info("Email revised based on critique")
                # else:
                #     logger.warning("No critique received, using original version")

            except Exception as e:
                logger.error(f"xAI personalization error: {e}")
                subject, body = orig_subject, orig_body

        # Step 10: Create Gmail draft and save to database
        with workflow_step(10, "Creating Gmail draft"):
            # Get the outreach window
            persona = profile_type
            club_tz = data_gatherer.get_club_timezone(state)
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                # Get company_type from database
                cursor.execute("""
                    SELECT company_type FROM companies 
                    WHERE name = ? AND city = ? AND state = ?
                """, (club_name, city, state))
                result = cursor.fetchone()
                stored_company_type = result[0] if result else None

                if stored_company_type:
                    # Use stored company type if available
                    if "private" in stored_company_type.lower():
                        club_type = "Private Clubs"
                    elif "semi-private" in stored_company_type.lower():
                        club_type = "Semi-Private Clubs"
                    elif "public" in stored_company_type.lower() or "municipal" in stored_company_type.lower():
                        club_type = "Public Clubs"
                    else:
                        club_type = "Public Clubs"  # Default
                else:
                    # Fallback to HubSpot lookup if not in database
                    _, club_type = data_gatherer.get_club_geography_and_type(club_name, city, state)

                # Get geography
                if state == "AZ":
                    geography = "Year-Round Golf"
                else:
                    geography = data_gatherer.determine_geography(city, state)

            except Exception as e:
                logger.error(f"Error getting company type from database: {str(e)}")
                geography, club_type = data_gatherer.get_club_geography_and_type(club_name, city, state)
            finally:
                cursor.close()
                conn.close()

            outreach_window = {
                "Best Month": get_best_month(geography),
                "Best Time": get_best_time(persona),
                "Best Day": get_best_day(persona)
            }
            
            try:
                # Get structured timing data
                best_months = get_best_month(geography)
                best_time = get_best_time(persona)
                best_days = get_best_day(persona)
                
                # print("\n=== Email Scheduling Logic ===")
                # print(f"Lead Profile: {persona}")
                # print(f"Geography: {geography}")
                # print(f"State: {state}")
                # print("\nOptimal Send Window:")
                # print(f"- Months: {best_months}")
                # print(f"- Days: {best_days} (0=Monday, 6=Sunday)")
                # print(f"- Time: {best_time['start']}:00 - {best_time['end']}:00")
                
                # Pick random hour within the time window
                from random import randint
                target_hour = randint(best_time["start"], best_time["end"])
                
                # Calculate the next occurrence
                now = datetime.now()
                
                # print("\nScheduling Process:")
                # print(f"1. Starting with tomorrow: {(now + timedelta(days=1)).strftime('%Y-%m-%d')}")
                
                # Start with tomorrow
                target_date = now + timedelta(days=1)
                
                # Find the next valid month if current month isn't ideal
                while target_date.month not in best_months:
                    # print(f"   ❌ Month {target_date.month} not in optimal months {best_months}")
                    if target_date.month == 12:
                        target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
                    else:
                        target_date = target_date.replace(month=target_date.month + 1, day=1)
                    # print(f"   ➡️ Advanced to: {target_date.strftime('%Y-%m-%d')}")
                
                # Find the next valid day of week
                while target_date.weekday() not in best_days:
                    # print(f"   ❌ Day {target_date.weekday()} not in optimal days {best_days}")
                    target_date += timedelta(days=1)
                    # print(f"   ➡️ Advanced to: {target_date.strftime('%Y-%m-%d')}")
                
                # Set the target time
                scheduled_send_date = target_date.replace(
                    hour=target_hour,
                    minute=randint(0, 59),
                    second=0,
                    microsecond=0
                )
                
                # Adjust for state's offset
                state_offsets = STATE_TIMEZONES.get(state.upper())
                scheduled_send_date = convert_to_club_timezone(scheduled_send_date, state_offsets)
                
                # print(f"\nFinal Schedule:")
                # print(f"✅ Selected send time: {scheduled_send_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                # print("============================\n")

            except Exception as e:
                logger.warning(f"Error calculating send date: {str(e)}. Using current time + 1 day", extra={
                    "error": str(e),
                    "fallback_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_optimal_time": False
                })
                scheduled_send_date = datetime.now() + timedelta(days=1)

            if scheduled_send_date is None:
                logger.warning("scheduled_send_date is None, setting default send date")
                scheduled_send_date = datetime.now() + timedelta(days=1)  # Default to 1 day later

            # First get the lead_id from the database
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                # Get lead_id based on email
                cursor.execute("""
                    SELECT lead_id FROM leads 
                    WHERE email = ?
                """, (lead_email,))
                result = cursor.fetchone()
                
                if result:
                    lead_id = result[0]
                    
                    # Create Gmail draft first
                    draft_result = create_draft(
                        sender="me",
                        to=lead_email,
                        subject=subject,
                        message_text=body,
                        lead_id=lead_id,
                        sequence_num=1
                    )

                    if draft_result["status"] == "ok":
                        # Store in database once
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
                            # Get next sequence number
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
    required_templates = [
        'general_manager_initial_outreach.md',
        'fb_manager_initial_outreach.md',
        'fallback.md'
    ]
    
    missing_templates = [
        template for template in required_templates
        if not (template_dir / template).exists()
    ]
    
    if missing_templates:
        logger.warning(f"Missing templates: {missing_templates}")

def calculate_send_date(geography, persona, state_code, season_data=None):
    """Calculate optimal send date based on geography and persona."""
    try:
        # Get base timing data
        outreach_window = get_best_outreach_window(
            persona=persona,
            geography=geography,
            season_data=season_data
        )
        
        best_months = outreach_window["Best Month"]
        best_time = outreach_window["Best Time"]
        best_days = outreach_window["Best Day"]
        
        # Start with tomorrow
        now = datetime.now()
        target_date = now + timedelta(days=1)
        
        #print("\nScheduling Process:")
        #print(f"1. Starting with tomorrow: {target_date.strftime('%Y-%m-%d')}")
        
        # Find the next valid month
        while target_date.month not in best_months:
            #print(f"   ❌ Month {target_date.month} not in optimal months {best_months}")
            if target_date.month == 12:
                target_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
            else:
                target_date = target_date.replace(month=target_date.month + 1, day=1)
            #print(f"   ➡️ Advanced to: {target_date.strftime('%Y-%m-%d')}")
        
        # Find the next valid day of week
        while target_date.weekday() not in best_days:
            #print(f"   ❌ Day {target_date.weekday()} not in optimal days {best_days}")
            target_date += timedelta(days=1)
            #print(f"   ➡️ Advanced to: {target_date.strftime('%Y-%m-%d')}")
        
        # Set time within the best window (9:00-11:00)
        send_hour = best_time["start"]  # This will be 9
        send_minute = random.randint(0, 59)
        
        # If we want to randomize the hour but ensure we stay before 11:00
        if random.random() < 0.5:  # 50% chance to use hour 10 instead of 9
            send_hour = 10
            
        send_date = target_date.replace(
            hour=send_hour,
            minute=send_minute,
            second=0,
            microsecond=0
        )
        
        # Adjust for state's offset
        final_send_date = adjust_send_time(send_date, state_code)
        
        #print(f"\nFinal Schedule:")
        #print(f"✅ Selected send time: {final_send_date.strftime('%Y-%m-%d %H:%M:%S')}")
        #print("============================\n")
        
        return final_send_date
        
    except Exception as e:
        logger.error(f"Error calculating send date: {str(e)}", exc_info=True)
        # Return tomorrow at 10 AM as fallback
        return datetime.now() + timedelta(days=1, hours=10)

def get_next_month_first_day(current_date):
    """Helper function to get the first day of the next month"""
    if current_date.month == 12:
        return current_date.replace(year=current_date.year + 1, month=1, day=1)
    return current_date.replace(month=current_date.month + 1, day=1)

def clear_sql_tables():
    """Clear all records from SQL tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # List of tables to clear (in order to handle foreign key constraints)
        tables = [
            'emails',
            'lead_properties',
            'company_properties',
            'leads',
            'companies'
        ]
        
        # Disable foreign key constraints for SQL Server
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
        
        # Re-enable foreign key constraints
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
    """Clear log file, lead contexts, and SQL data if CLEAR_LOGS_ON_START is True"""
    from config.settings import CLEAR_LOGS_ON_START
    
    if not CLEAR_LOGS_ON_START:
        print("Skipping file cleanup - CLEAR_LOGS_ON_START is False")
        return
        
    # Clear log file
    log_path = os.path.join(PROJECT_ROOT, 'logs', 'app.log')
    if os.path.exists(log_path):
        try:
            open(log_path, 'w').close()
            print("Log file cleared")
        except Exception as e:
            print(f"Failed to clear log file: {e}")
    
    # Clear lead contexts directory
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
    
    # Clear SQL tables
    clear_sql_tables()

def schedule_followup(lead_id: int, email_id: int):
    """Schedule a follow-up email"""
    try:
        # Get lead data first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get original email data
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
        
        # Package original email data
        original_email = {
            'email': email,
            'first_name': first_name,
            'name': company_name,
            'state': state,
            'subject': orig_subject,
            'body': orig_body,
            'created_at': created_at
        }
        
        # Generate the follow-up email content
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            email_id=email_id,
            sequence_num=2,  # Second email in sequence
            original_email=original_email  # Pass the original email data
        )
        
        if followup:
            # Create Gmail draft for follow-up
            draft_result = create_draft(
                sender="me",
                to=followup.get('email'),
                subject=followup.get('subject'),
                message_text=followup.get('body')
            )
            
            if draft_result["status"] == "ok":
                # Save follow-up to database
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
        
        # Insert or update the email record
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
                
                # For the INSERT
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

if __name__ == "__main__":
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        # Only clear console if logs should be cleared
        os.system('cls' if os.name == 'nt' else 'clear')
    
    verify_templates()
    
    # Start the scheduler silently
    from scheduling.followup_scheduler import start_scheduler
    import threading
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Start main workflow
    main()
