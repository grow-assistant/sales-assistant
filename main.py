# File: main.py

import sys
import logging
import datetime

from utils.logging_setup import logger
from utils.exceptions import LeadContextError
from utils.xai_integration import (
    personalize_email_with_xai,
    xai_news_search,
    _build_icebreaker_from_news,
    xai_club_info_search
)
from utils.gmail_integration import create_draft, search_inbound_messages_for_email
from scripts.build_template import build_outreach_email
from scripts.job_title_categories import categorize_job_title
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY
from scheduling.extended_lead_storage import upsert_full_lead
from scheduling.followup_generation import generate_followup_email_xai
from scheduling.sql_lookup import build_lead_sheet_from_sql
from services.leads_service import LeadsService

# Initialize services
leads_service = LeadsService()

###############################################################################
# Attempt to gather last inbound snippet from:
#   1) Gmail
#   2) lead_sheet["lead_data"]["emails"]
#   3) lead_sheet["lead_data"]["notes"]   (if you store inbound text in notes)
###############################################################################
def get_any_inbound_snippet(lead_email: str, lead_sheet: dict, max_chars=200) -> str:
    """
    Returns a snippet (up to max_chars) from the most recent inbound message 
    found in Gmail or the lead sheet. Logs where the snippet came from.
    """
    # 1) Check Gmail
    inbound_snippets = search_inbound_messages_for_email(lead_email, max_results=1)
    if inbound_snippets:
        snippet_gmail = inbound_snippets[0].strip()
        if DEBUG_MODE:
            logger.debug(f"Found inbound snippet from Gmail: {snippet_gmail[:50]}...")
        return snippet_gmail[:max_chars] + "..." if len(snippet_gmail) > max_chars else snippet_gmail

    # 2) Check lead_data["emails"]
    lead_data = lead_sheet.get("lead_data", {})
    all_emails = lead_data.get("emails", [])
    inbound_emails = [
        e for e in all_emails
        if e.get("direction", "").lower() in ("incoming_email", "inbound")
    ]
    if inbound_emails:
        # Sort descending by timestamp if available
        inbound_emails.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        snippet_lead_email = inbound_emails[0].get("body_text", "").strip()
        if DEBUG_MODE:
            logger.debug(f"Found inbound snippet from lead_data['emails']: {snippet_lead_email[:50]}...")
        return snippet_lead_email[:max_chars] + "..." if len(snippet_lead_email) > max_chars else snippet_lead_email

    # 3) Check lead_data["notes"] if you store inbound text there
    notes = lead_data.get("notes", [])
    if notes:
        # Sort descending by lastmodifieddate or createdate
        notes.sort(key=lambda n: n.get("lastmodifieddate", ""), reverse=True)
        for note in notes:
            body = note.get("body", "").strip()
            # Heuristic: If the body references lead_email or says "inbound"
            if lead_email.lower() in body.lower() or "inbound" in body.lower():
                snippet_note = body
                if DEBUG_MODE:
                    logger.debug(f"Found inbound snippet in lead_data['notes']: {snippet_note[:50]}...")
                return snippet_note[:max_chars] + "..." if len(snippet_note) > max_chars else snippet_note

    # No inbound snippet found
    if DEBUG_MODE:
        logger.debug("No inbound snippet found in Gmail, lead_data['emails'], or lead_data['notes'].")
    return ""


###############################################################################
# Main Workflow
###############################################################################
def main():
    # Set up logging level
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled in main workflow")
    else:
        logger.info("Starting main workflow...")

    try:
        # 1) Prompt for lead email
        email = input("Please enter a lead's email address: ").strip()
        if not email:
            logger.error("No email entered; exiting.")
            return

        # 2) Attempt to fetch from SQL
        lead_sheet = build_lead_sheet_from_sql(email)
        if lead_sheet:
            if DEBUG_MODE:
                logger.debug(f"Lead '{email}' found in SQL. Using local data.")
        else:
            if DEBUG_MODE:
                logger.debug(f"Lead '{email}' not found in SQL; fetching from external source...")
            lead_sheet = leads_service.prepare_lead_context(email)

        # 3) Verify lead_sheet success
        if lead_sheet.get("metadata", {}).get("status") != "success":
            logger.error("Failed to prepare or retrieve lead context. Exiting.")
            return

        # 4) Upsert into SQL
        upsert_full_lead(lead_sheet)

        # 5) Extract data for building the email
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})

        first_name = lead_data.get("firstname", "")
        last_name = lead_data.get("lastname", "")
        club_name = company_data.get("name", "").strip()
        city = company_data.get("city", "").strip()
        state = company_data.get("state", "").strip()

        # Build location string
        if city and state:
            location_str = f"{city}, {state}"
        elif city:
            location_str = city
        elif state:
            location_str = state
        else:
            location_str = "an unknown location"

        placeholders = {
            "FirstName": first_name or "Friend",
            "LastName": last_name,
            "ClubName": club_name or "Your Club",
            "DeadlineDate": "Oct 15th",
            "Role": lead_data.get("jobtitle", "General Manager"),
            "Task": "Staff Onboarding",
            "Topic": "On-Course Ordering Platform"
        }

        if DEBUG_MODE:
            logger.debug(
                "Extracted lead data fields:",
                extra={
                    "first_name": first_name,
                    "last_name": last_name,
                    "company": club_name or "N/A",
                    "role": lead_data.get("jobtitle", "N/A"),
                    "email": lead_data.get("email", "N/A"),
                    "location": location_str
                }
            )

        last_interaction_days = 32  # Example fallback

        # 6) If we have a club name, fetch optional club info & news
        if club_name:
            if DEBUG_MODE:
                logger.debug(f"Fetching club info/news for '{club_name}'")
            club_info_snippet = xai_club_info_search(club_name, location_str)
            news_result = xai_news_search(club_name)
        else:
            club_info_snippet = ""
            news_result = ""

        has_news = bool(news_result and "has not been" not in news_result.lower())

        # 7) Determine job title category
        jobtitle_str = lead_data.get("jobtitle", "")
        profile_type = categorize_job_title(jobtitle_str)

        # 8) Build initial outreach email
        subject, body = build_outreach_email(
            profile_type=profile_type,
            last_interaction_days=last_interaction_days,
            placeholders=placeholders,
            current_month=9,      # Example current month
            start_peak_month=5,   # Example peak start
            end_peak_month=8,     # Example peak end
            use_markdown_template=True
        )

        if not has_news:
            # Remove the ICEBREAKER placeholder if no news
            body = body.replace("[ICEBREAKER]\n\n", "")

        # 9) Personalize with xAI
        orig_subject, orig_body = subject, body
        try:
            subject, body = personalize_email_with_xai(lead_sheet, subject, body)
            if not subject.strip():
                subject = orig_subject
            if not body.strip():
                body = orig_body

            # Clean up leftover bold
            body = body.replace("**", "")

            # Insert greeting if missing
            if placeholders["FirstName"]:
                if not body.lower().startswith("hey "):
                    body = f"Hey {placeholders['FirstName']},\n\n{body}"
            else:
                if not body.lower().startswith("hey"):
                    body = f"Hey there,\n\n{body}"

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

        # 10) Attempt to find inbound snippet from Gmail or lead sheet
        inbound_snippet = get_any_inbound_snippet(email, lead_sheet)
        if inbound_snippet:
            body += f"\n\n---\nP.S. Here's the last inbound message you sent:\n\"{inbound_snippet}\""

        # 11) Create Gmail draft
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

        # Possibly generate follow-ups
        # generate_followup_email_xai(lead_id=..., sequence_num=2)

    except LeadContextError as e:
        logger.error(f"Failed to prepare lead context: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")


###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    main()
