import sys
import logging
import datetime

from utils.logging_setup import logger
from utils.exceptions import LeadContextError
from utils.xai_integration import (
    personalize_email_with_xai,
    _build_icebreaker_from_news
)
from utils.gmail_integration import create_draft
from scripts.build_template import build_outreach_email
from scripts.job_title_categories import categorize_job_title
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY
from scheduling.extended_lead_storage import upsert_full_lead
from scheduling.followup_generation import generate_followup_email_xai
from scheduling.sql_lookup import build_lead_sheet_from_sql
from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
import openai
from config.settings import OPENAI_API_KEY, DEFAULT_TEMPERATURE, MODEL_FOR_GENERAL

# Initialize services
data_gatherer = DataGathererService()
leads_service = LeadsService(data_gatherer)

###############################################################################
# Summarize lead interactions
###############################################################################
def summarize_lead_interactions(lead_sheet: dict) -> str:
    """
    Collects all prior emails and notes from the lead_sheet,
    then sends them to OpenAI to request a concise summary.

    The summary should identify:
      - How many times the lead responded
      - Where the conversation left off
      - Overall tone and progress of the interactions

    :param lead_sheet: dict containing 'lead_data', which has 'emails' and 'notes'
    :return: str summary from OpenAI or an error message
    """
    import openai
    from config.settings import OPENAI_API_KEY, DEFAULT_TEMPERATURE, MODEL_FOR_GENERAL

    # 1) Grab prior emails and notes from the lead_sheet
    lead_data = lead_sheet.get("lead_data", {})
    emails = lead_data.get("emails", [])
    notes = lead_data.get("notes", [])

    # If there's truly no data, return early
    if not emails and not notes:
        return "No prior emails or notes found."

    # 2) Compile a single text block of the conversation history
    compiled_history = "=== PRIOR EMAILS ===\n"
    for e in emails:
        subject = e.get("subject", "(No Subject)")
        body_text = e.get("body_text", "(No Body)").strip()
        timestamp = e.get("timestamp", "Unknown time")
        direction = e.get("direction", "")
        compiled_history += (
            f"[{timestamp} | direction={direction.upper()}]\n"
            f"Subject: {subject}\n"
            f"{body_text}\n\n"
        )

    compiled_history += "=== NOTES ===\n"
    for n in notes:
        note_body = n.get("body", "").strip()
        note_time = n.get("timestamp", "Unknown time")
        compiled_history += (
            f"[{note_time}]\n"
            f"{note_body}\n\n"
        )

    # 3) Build the prompt for OpenAI
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that summarizes a lead's interaction history."
    }
    user_prompt = {
        "role": "user",
        "content": (
            "Below is all the historical emails and notes with a lead:\n\n"
            f"{compiled_history}\n\n"
            "Create a summary starting from the most recent interactions, moving backward. "
            "One Paragraph max. Condense older details more than recent ones. "
            "Exclude specific dates, but note the time elapsed since the lead's last response "
            "in months or years if you can infer it."
        )
    }

    # 4) Call OpenAI to get the summary
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_FOR_GENERAL,  # e.g. "gpt-3.5-turbo" or "gpt-4"
            temperature=DEFAULT_TEMPERATURE,
            messages=[system_prompt, user_prompt]
        )
        summary_text = response["choices"][0]["message"]["content"].strip()
        return summary_text
    except Exception as e:
        return f"Error summarizing lead interactions: {str(e)}"

###############################################################################
# Main Workflow
###############################################################################
def main():
    """
    Main entry point for the sales assistant application. Handles the workflow of:
    1. Getting lead email input
    2. Retrieving or creating lead context from SQL/external sources
    3. Building personalized email content with AI assistance
    4. Creating Gmail drafts with the generated content
    
    The function prompts for a lead's email address and orchestrates the entire
    process of generating and saving a personalized email draft.
    """
    import uuid
    correlation_id = str(uuid.uuid4())
    
    # Set up logging level
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled in main workflow", extra={
            "correlation_id": correlation_id
        })
    else:
        logger.info("Starting main workflow...", extra={
            "correlation_id": correlation_id
        })

    try:
        # 1) Prompt for lead email
        email = input("Please enter a lead's email address: ").strip()
        if not email:
            logger.error("No email entered; exiting.", extra={
                "correlation_id": correlation_id
            })
            return

        # 2) Gather data from external sources
        if DEBUG_MODE:
            logger.debug(f"Fetching lead data for '{email}'...", extra={
                "email": email,
                "correlation_id": correlation_id
            })
        lead_sheet = data_gatherer.gather_lead_data(email, correlation_id=correlation_id)
        logger.debug(f"Company data received: {lead_sheet.get('lead_data', {}).get('company_data', {})}")

        # 3) Save lead data to SQL
        try:
            logger.debug("Attempting to save lead data to SQL database...", extra={
                "correlation_id": correlation_id,
                "email": email
            })
            upsert_full_lead(lead_sheet)
            logger.info("Successfully saved lead data to SQL database", extra={
                "correlation_id": correlation_id,
                "email": email
            })
        except Exception as e:
            logger.error(f"Failed to save lead data to SQL: {str(e)}", extra={
                "correlation_id": correlation_id,
                "email": email,
                "error": str(e)
            }, exc_info=True)

        # Verify lead_sheet success
        if lead_sheet.get("metadata", {}).get("status") != "success":
            logger.error("Failed to prepare or retrieve lead context. Exiting.", extra={
                "email": email,
                "correlation_id": correlation_id
            })
            return

        # Prepare lead context with correlation ID (if needed by older code)
        lead_context = leads_service.prepare_lead_context(email, correlation_id=correlation_id)

        # 4) Extract relevant data
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})

        first_name = lead_data.get("properties", {}).get("firstname", "")
        last_name = lead_data.get("properties", {}).get("lastname", "")
        club_name = company_data.get("name", "").strip()
        city = company_data.get("city", "").strip()
        state = company_data.get("state", "").strip()

        if city and state:
            location_str = f"{city}, {state}"
        elif city:
            location_str = city
        elif state:
            location_str = state
        else:
            location_str = "an unknown location"

        placeholders = {
            "FirstName": first_name,
            "LastName": last_name,
            "ClubName": club_name or "Your Club",
            "DeadlineDate": "Oct 15th",
            "Role": lead_data.get("jobtitle", "General Manager"),
            "Task": "Staff Onboarding",
            "Topic": "On-Course Ordering Platform",
            "YourName": "Ty"
        }
        logger.debug("Placeholders built", extra=placeholders)

        # 5) Additional data for personalization
        club_info_snippet = data_gatherer.gather_club_info(club_name, city, state)
        news_result = data_gatherer.gather_club_news(club_name)
        has_news = bool(news_result and "has not been" not in news_result.lower())

        jobtitle_str = lead_data.get("jobtitle", "")
        profile_type = categorize_job_title(jobtitle_str)

        # 6) Summarize interactions
        interaction_summary = summarize_lead_interactions(lead_sheet)
        logger.info("Interaction Summary:\n" + interaction_summary)

        last_interaction = lead_data.get("properties", {}).get("hs_sales_email_last_replied", "")
        last_interaction_days = 0
        if last_interaction:
            try:
                last_date = datetime.datetime.fromtimestamp(int(last_interaction)/1000)
                last_interaction_days = (datetime.datetime.now() - last_date).days
            except (ValueError, TypeError):
                last_interaction_days = 0

        # 7) Build initial outreach email
        subject, body = build_outreach_email(
            profile_type=profile_type,
            last_interaction_days=last_interaction_days,
            placeholders=placeholders,
            current_month=9,      # Example
            start_peak_month=5,
            end_peak_month=8,
            use_markdown_template=True
        )
        logger.debug("Loaded email template", extra={
            "subject_template": subject,
            "body_template": body
        })

        if not has_news:
            body = body.replace("[ICEBREAKER]\n\n", "").replace("[ICEBREAKER]", "")

        orig_subject, orig_body = subject, body
        for key, val in placeholders.items():
            subject = subject.replace(f"[{key}]", val)
            body = body.replace(f"[{key}]", val)

        # 8) Personalize with xAI (passing summary, club info, and news)
        try:
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
                except Exception as e:
                    logger.error(f"Icebreaker generation error: {e}")
                    body = body.replace("[ICEBREAKER]", "")

        except Exception as e:
            logger.error(f"xAI personalization error: {e}")
            subject, body = orig_subject, orig_body

        logger.debug("Final content after xAI", extra={
            "subject": subject,
            "body": body
        })

        # 9) Create Gmail draft
        lead_email = lead_data.get("email", email)
        draft_result = create_draft(
            sender="me",
            to=lead_email,
            subject=subject,
            message_text=body
        )
        if draft_result["status"] == "ok":
            if DEBUG_MODE:
                logger.debug(f"Gmail draft created: {draft_result.get('draft_id')}")
            else:
                logger.info("Gmail draft created successfully")
        else:
            logger.error("Failed to create Gmail draft.")

    except LeadContextError as e:
        logger.error(f"Failed to prepare lead context: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")


if __name__ == "__main__":
    main()
