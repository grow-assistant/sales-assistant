# Sales Assistant Codebase

This document contains the core functionality of the Sales Assistant project.

## Table of Contents
- [agents\__init__.py](#agents\__init__.py)
- [agents\context_preparer.py](#agents\context_preparer.py)
- [agents\functions.py](#agents\functions.py)
- [agents\orchestrator.py](#agents\orchestrator.py)
- [agents\personalization.py](#agents\personalization.py)
- [external\__init__.py](#external\__init__.py)
- [external\external_api.py](#external\external_api.py)
- [hubspot_integration\__init__.py](#hubspot_integration\__init__.py)
- [hubspot_integration\fetch_leads.py](#hubspot_integration\fetch_leads.py)
- [hubspot_integration\hubspot_utils.py](#hubspot_integration\hubspot_utils.py)
- [hubspot_integration\lead_qualification.py](#hubspot_integration\lead_qualification.py)
- [main.py](#main.py)
- [scheduling\__init__.py](#scheduling\__init__.py)
- [scheduling\database.py](#scheduling\database.py)
- [scheduling\extended_lead_storage.py](#scheduling\extended_lead_storage.py)
- [scheduling\followup_generation.py](#scheduling\followup_generation.py)
- [scheduling\followup_scheduler.py](#scheduling\followup_scheduler.py)
- [scheduling\sql_lookup.py](#scheduling\sql_lookup.py)
- [scripts\__init__.py](#scripts\__init__.py)
- [scripts\build_template.py](#scripts\build_template.py)
- [scripts\check_reviewed_drafts.py](#scripts\check_reviewed_drafts.py)
- [scripts\get_random_contacts.py](#scripts\get_random_contacts.py)
- [scripts\golf_outreach_strategy.py](#scripts\golf_outreach_strategy.py)
- [scripts\job_title_categories.py](#scripts\job_title_categories.py)
- [scripts\migrate_emails_table.py](#scripts\migrate_emails_table.py)
- [scripts\ping_hubspot_for_gm.py](#scripts\ping_hubspot_for_gm.py)
- [scripts\schedule_outreach.py](#scripts\schedule_outreach.py)
- [services\__init__.py](#services\__init__.py)
- [services\company_enrichment_service.py](#services\company_enrichment_service.py)
- [services\conversation_analysis_service.py](#services\conversation_analysis_service.py)
- [services\data_gatherer_service.py](#services\data_gatherer_service.py)
- [services\gmail_service.py](#services\gmail_service.py)
- [services\hubspot_service.py](#services\hubspot_service.py)
- [services\leads_service.py](#services\leads_service.py)
- [services\orchestrator_service.py](#services\orchestrator_service.py)
- [utils\__init__.py](#utils\__init__.py)
- [utils\conversation_summary.py](#utils\conversation_summary.py)
- [utils\date_utils.py](#utils\date_utils.py)
- [utils\doc_reader.py](#utils\doc_reader.py)
- [utils\enrich_hubspot_company_data.py](#utils\enrich_hubspot_company_data.py)
- [utils\exceptions.py](#utils\exceptions.py)
- [utils\export_codebase.py](#utils\export_codebase.py)
- [utils\export_codebase_primary_files.py](#utils\export_codebase_primary_files.py)
- [utils\export_templates.py](#utils\export_templates.py)
- [utils\formatting_utils.py](#utils\formatting_utils.py)
- [utils\gmail_integration.py](#utils\gmail_integration.py)
- [utils\hubspot_field_finder.py](#utils\hubspot_field_finder.py)
- [utils\logger_base.py](#utils\logger_base.py)
- [utils\logging_setup.py](#utils\logging_setup.py)
- [utils\model_selector.py](#utils\model_selector.py)
- [utils\season_snippet.py](#utils\season_snippet.py)
- [utils\templates_directory.py](#utils\templates_directory.py)
- [utils\web_fetch.py](#utils\web_fetch.py)
- [utils\xai_integration.py](#utils\xai_integration.py)

## agents\__init__.py
```python
# agents/__init__.py

"""
Agents module for handling AI-powered sales automation.
"""

```

## agents\context_preparer.py
```python
# agents/context_preparer.py

from typing import Dict, Any
from services.leads_service import LeadsService

# Initialize the leads service
_leads_service = LeadsService()

def prepare_lead_context(lead_email: str) -> Dict[str, Any]:
    """
    Prepare comprehensive lead context including HubSpot data,
    company information, and external research.
    
    This function now delegates to the LeadsService while maintaining
    the same interface for backward compatibility.
    """
    return _leads_service.prepare_lead_context(lead_email)

```

## agents\functions.py
```python
# agents/functions.py
from typing import Dict, Any
import logging
import openai

# Import your environment/config
from config.settings import (
    OPENAI_API_KEY, 
    DEFAULT_TEMPERATURE, 
    HUBSPOT_API_KEY
)
from utils.logging_setup import logger

# Replace old hubspot_integration.hubspot_api with HubspotService
from services.hubspot_service import HubspotService

from external.external_api import (
    review_previous_interactions as external_review,
    market_research as external_market_research,
    determine_club_season
)
from utils.gmail_integration import create_draft as gmail_create_draft

openai.api_key = OPENAI_API_KEY

# Initialize HubSpot service
hubspot = HubspotService(api_key=HUBSPOT_API_KEY)

def call_function(name: str, arguments: dict, context: dict) -> dict:
    logger.info(f"Calling function '{name}' with arguments: {arguments}")
    try:
        #######################################################################
        # Replaced references to get_hubspot_leads, get_lead_data_from_hubspot,
        # get_associated_company_id, get_company_data with methods on `hubspot`.
        #######################################################################
        if name == "get_hubspot_leads":
            leads = hubspot.get_hubspot_leads()
            context["lead_list"] = leads
            logger.info(f"Retrieved {len(leads)} leads from HubSpot.")
            return {"content": "Leads retrieved.", "status": "ok"}

        elif name == "get_lead_data_from_hubspot":
            contact_id = arguments["contact_id"]
            logger.debug(f"Fetching lead data for contact_id={contact_id}")
            lead_data = hubspot.get_lead_data_from_hubspot(contact_id)
            if not lead_data:
                logger.error(f"No lead data found for {contact_id}")
                return {"content": f"No lead data found for contact_id {contact_id}", "status": "error"}
            # Truncate any long text fields
            for key, value in lead_data.items():
                if isinstance(value, str) and len(value) > 500:
                    lead_data[key] = value[:500] + "..."
            context["lead_data"] = lead_data
            logger.info(f"Lead data retrieved for contact_id={contact_id}")
            return {"content": "Lead data retrieved.", "status": "ok"}

        elif name == "review_previous_interactions":
            contact_id = arguments["contact_id"]
            logger.debug(f"Reviewing previous interactions for contact_id={contact_id}")
            interactions = external_review(contact_id)
            context["previous_interactions"] = interactions
            logger.info(f"Interactions for contact_id={contact_id}: {interactions}")
            details = (
                f"Emails opened: {interactions.get('emails_opened', 0)}, "
                f"Emails sent: {interactions.get('emails_sent', 0)}, "
                f"Meetings held: {interactions.get('meetings_held', 0)}, "
                f"Last response: {interactions.get('last_response', 'None')}"
            )
            return {"content": f"Interactions reviewed. Details: {details}", "status": "ok"}

        elif name == "market_research":
            company_name = arguments.get("company_name", "")
            if not company_name:
                logger.warning("No company name found; skipping market research.")
                return {"content": "Skipped market research due to missing company name.", "status": "ok"}

            logger.debug(f"Performing market research for {company_name}")
            data = external_market_research(company_name)
            # Truncate research data if too verbose
            if isinstance(data.get("description"), str) and len(data["description"]) > 1000:
                data["description"] = data["description"][:1000] + "..."
            if isinstance(data.get("market_analysis"), str) and len(data["market_analysis"]) > 1000:
                data["market_analysis"] = data["market_analysis"][:1000] + "..."
            context["research_data"] = data
            logger.info(f"Market research result: {data}")
            return {"content": "Market research completed.", "status": "ok"}

        elif name == "analyze_competitors":
            logger.debug("Analyzing competitors...")
            lead_data = context.get("lead_data", {})
            contact_id = lead_data.get("contact_id", "")
            if contact_id:
                company_id = hubspot.get_associated_company_id(contact_id)
            else:
                company_id = None

            if company_id:
                company_data = hubspot.get_company_data(company_id)
            else:
                company_data = {}

            domain = None
            email = lead_data.get("email", "")
            if "@" in email:
                domain = email.split("@")[-1].strip().lower()
            if not domain and "website" in company_data.get("properties", {}):
                # if the company data has a website property
                domain = company_data["properties"]["website"].replace("http://", "").replace("https://", "").strip().lower()

            competitor_info = {
                "competitor_found": False,
                "status": "ok",
                "message": "No competitor software detected"
            }
            context["competitor_data"] = competitor_info
            logger.info(f"Competitor analysis result: {competitor_info}")
            return {"content": "Competitor analysis completed.", "status": "ok"}

        elif name == "personalize_message":
            lead_data = arguments.get("lead_data", {})
            if not lead_data:
                logger.error("Missing lead_data for personalize_message")
                return {"content": "Missing lead_data for personalization", "status": "error"}

            logger.debug("Personalizing message...")
            season_data = determine_club_season(
                lead_data.get("city", ""), 
                lead_data.get("state", "")
            )

            fallback_msg = (
                f"Hi {lead_data.get('firstname', 'there')},\n\n"
                f"With {lead_data.get('company','your club')}'s peak season approaching, "
                "we wanted to share how our on-demand F&B solution is helping clubs increase revenue by 15%. "
                "Members order directly from the course, and your team can focus on great service.\n\n"
                "Could we have a quick chat next week?\n\nBest,\nThe Swoop Team"
            )

            try:
                system_prompt = {
                    "role": "system",
                    "content": "You are a sales copywriter creating a personalized outreach message."
                }
                user_prompt = {
                    "role": "user", 
                    "content": (
                        f"Create a short, personalized sales email for {lead_data.get('firstname','there')} "
                        f"at {lead_data.get('company','their club')} based on peak season."
                    )
                }

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[system_prompt, user_prompt],
                    temperature=DEFAULT_TEMPERATURE
                )
                refined_msg = response.choices[0].message.content.strip()
                context["personalized_message"] = refined_msg
                logger.info("Personalized message created.")
                return {"content": "Message personalized.", "status": "ok"}

            except Exception as e:
                logger.error(f"Error in personalize_message: {str(e)}")
                context["personalized_message"] = fallback_msg
                return {"content": "Used fallback message due to error.", "status": "ok"}

        elif name == "create_gmail_draft":
            sender = arguments["sender"]
            to = arguments["to"]
            subject = arguments.get("subject", "Introductory Email – Swoop Golf")
            message_text = arguments.get("message_text", context.get("personalized_message", ""))

            if not message_text:
                message_text = (
                    f"Hi {context.get('lead_data', {}).get('firstname', 'there')},\n\n"
                    f"With {context.get('lead_data', {}).get('company', 'your club')}'s peak season approaching, "
                    "we wanted to share how our on-demand F&B solution is helping clubs increase revenue by 15%. "
                    "Members order directly from the course, and your team can focus on great service.\n\n"
                    "Could we have a quick chat next week?\n\nBest,\nThe Swoop Team"
                )

            # Truncate if too long
            if len(message_text) > 2000:
                message_text = message_text[:2000] + "..."

            logger.debug(f"Creating Gmail draft email from {sender} to {to}")
            result = gmail_create_draft(sender, to, subject, message_text)
            logger.info(f"Gmail draft creation result: {result}")
            return {"content": f"Gmail draft creation result: {result}", "status": result.get("status", "error")}

        else:
            logger.error(f"Function {name} not implemented.")
            return {"content": f"Function {name} not implemented.", "status": "error"}

    except Exception as e:
        logger.error(f"Error executing function {name}: {str(e)}")
        return {"content": f"Error executing {name}: {str(e)}", "status": "error"}

```

## agents\orchestrator.py
```python
# agents/orchestrator.py

import openai
from dotenv import load_dotenv
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, TypedDict
from config import OPENAI_API_KEY
from config.constants import DEFAULTS
from utils.logging_setup import logger
from utils.exceptions import LeadContextError, HubSpotError
from config.settings import DEBUG_MODE, OPENAI_API_KEY
from services.orchestrator_service import OrchestratorService
from services.leads_service import LeadsService
from services.hubspot_service import HubspotService

# NEW import
from services.data_gatherer_service import DataGathererService

load_dotenv()
openai.api_key = OPENAI_API_KEY


class Context(TypedDict):
    lead_id: str       # We assume this is actually an email address in your system
    domain_docs: Dict[str, str]
    messages: list
    last_action: Any
    metadata: Dict[str, Any]


class OrchestrationResult(TypedDict):
    success: bool
    lead_id: str
    actions_taken: list
    completion_time: datetime
    error: Any


async def run_sales_leader(context: Context) -> OrchestrationResult:
    """
    High-level workflow:
     1) Gather all lead data from DataGathererService in one pass.
     2) (Optional) Personalized messaging or further steps.
     3) Let the Sales Leader Agent decide the next steps.
    """
    result: OrchestrationResult = {
        'success': False,
        'lead_id': context['lead_id'],
        'actions_taken': [],
        'completion_time': datetime.now(),
        'error': None
    }

    try:
        # 1) Gather data with the new DataGathererService
        gatherer = DataGathererService()
        lead_sheet = gatherer.gather_lead_data(context['lead_id'])

        # Save the lead sheet in context for further usage
        context["messages"].append({
            "role": "system",
            "content": "Lead data gathered successfully."
        })
        context["messages"].append({
            "role": "assistant",
            "content": f"Enriched lead data: {lead_sheet}"
        })

        # 2) Summarize context
        summary = create_context_summary(lead_sheet)
        context["messages"].append({"role": "assistant", "content": summary})

        # 3) Initiate decision loop
        context["messages"].append({
            "role": "user",
            "content": "Now that we have the info, what's the next best step?"
        })
        decision_success = await decision_loop(context)
        if decision_success:
            logger.info("Workflow completed successfully.")
            result['success'] = True
            result['actions_taken'] = [
                msg["content"] for msg in context["messages"] if msg["role"] == "assistant"
            ]
        else:
            logger.error("Decision phase failed or timed out.")

    except (LeadContextError, HubSpotError) as e:
        logger.error(f"Business logic error: {e}")
        result['error'] = str(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        result['error'] = f"Internal error: {str(e)}"

    return result


def prune_or_summarize_messages(messages, max_messages=10):
    """
    If the message list exceeds max_messages, summarize older messages
    into one condensed message to reduce token usage.
    """
    if len(messages) > max_messages:
        older_messages = messages[:-max_messages]
        recent_messages = messages[-max_messages:]

        summary_text = "Summary of older messages:\n"
        for msg in older_messages:
            snippet = msg["content"]
            snippet = (snippet[:100] + "...") if len(snippet) > 100 else snippet
            summary_text += f"- ({msg['role']}): {snippet}\n"

        summary_message = {
            "role": "assistant",
            "content": summary_text.strip()
        }
        messages = [summary_message] + recent_messages

    return messages


async def decision_loop(context: Context) -> bool:
    """
    Continuously query the OpenAI model for decisions until it says 'We are done'
    or reaches max iterations. Prune messages each iteration to avoid token issues.
    """
    iteration = 0
    max_iterations = DEFAULTS["MAX_ITERATIONS"]

    logger.info("Entering decision loop...")
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Decision loop iteration {iteration}/{max_iterations}")

        context["messages"] = prune_or_summarize_messages(context["messages"], max_messages=10)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=context["messages"],
                temperature=0
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI API in decision loop: {str(e)}")
            return False

        assistant_message = response.choices[0].message
        content = assistant_message.get("content", "").strip()
        logger.info(f"Assistant response at iteration {iteration}: {content}")

        context["messages"].append({"role": "assistant", "content": content})

        if "we are done" in content.lower():
            logger.info("Sales Leader indicated that we are done.")
            return True

        if iteration >= 2:
            logger.info("We have recommended next steps. Exiting loop.")
            return True

        context["messages"].append({
            "role": "user",
            "content": "What else should we consider?"
        })

    logger.info("Reached max iterations in decision loop without completion.")
    return False


def create_context_summary(lead_sheet: Dict[str, Any]) -> str:
    """
    Create a brief summary of the gathered lead sheet info for the Sales Leader.
    """
    metadata = lead_sheet.get("metadata", {})
    lead_data = lead_sheet.get("lead_data", {})
    analysis_data = lead_sheet.get("analysis", {})

    summary_lines = [
        "Context Summary:",
        f"- Contact ID: {metadata.get('contact_id')}",
        f"- Company ID: {metadata.get('company_id')}",
        f"- Email: {metadata.get('lead_email')}",
        f"Competitor Analysis: {analysis_data.get('competitor_analysis', 'None')}",
        f"Season: {analysis_data.get('season_data', {})}"
    ]
    return "\n".join(summary_lines)

```

## agents\personalization.py
```python
"""
Module for lead personalization functionality.
Note: Core implementation has been moved to LeadsService.
This module maintains the original interface for backward compatibility.
"""

from typing import Dict, Any
from services.leads_service import LeadsService

# Initialize the leads service
_leads_service = LeadsService()


def generate_lead_summary(lead_email: str) -> Dict[str, Any]:
    """
    Gather and summarize lead's info according to the personalization steps.
    
    This function delegates to the LeadsService while maintaining
    the same interface for backward compatibility.
    
    Args:
        lead_email: Email address of the lead
        
    Returns:
        Dict containing lead summary information including:
        - lead_summary: List of bullet points about the lead
        - club_context: List of bullet points about the club
        - subject: Email subject
        - body: Email body
    """
    return _leads_service.generate_lead_summary(lead_email)

```

## external\__init__.py
```python
# external/__init__.py

```

## external\external_api.py
```python
# File: external/external_api.py

import csv
import requests
import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from dateutil.parser import parse as parse_date
from tenacity import retry, wait_exponential, stop_after_attempt
import json

from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from services.hubspot_service import HubspotService
# Initialize hubspot service:
_hubspot = HubspotService(api_key=HUBSPOT_API_KEY)

################################################################################
# CSV-based Season Data
################################################################################

PROJECT_ROOT = Path(__file__).parent.parent
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}

def load_season_data() -> None:
    """Load golf season data from CSV files into CITY_ST_DATA, ST_DATA dictionaries."""
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

# Load data at module import
load_season_data()

################################################################################
# Interaction & Season Methods
################################################################################

# Removed safe_int and review_previous_interactions - moved to DataGathererService

def analyze_competitors() -> dict:
    """
    Basic placeholder logic to analyze competitors.
    """
    return {
        "industry_trends": "On-course mobile F&B ordering is growing rapidly.",
        "competitor_moves": ["Competitor A launched a pilot at several clubs."]
    }

def determine_club_season(city: str, state: str) -> dict:
    """
    Return the peak season data for the given city/state based on CSV lookups.
    """
    city_key = (city.lower(), state.lower())
    row = CITY_ST_DATA.get(city_key)

    if not row:
        row = ST_DATA.get(state.lower())

    if not row:
        # Default if not found
        return {
            "year_round": "Unknown",
            "start_month": "N/A",
            "end_month": "N/A",
            "peak_season_start": "05-01",
            "peak_season_end": "08-31"
        }

    year_round = row["Year-Round?"].strip()
    start_month_str = row["Start Month"].strip()
    end_month_str = row["End Month"].strip()
    peak_season_start_str = row["Peak Season Start"].strip()
    peak_season_end_str = row["Peak Season End"].strip()

    if not peak_season_start_str or peak_season_start_str == "N/A":
        peak_season_start_str = "May"
    if not peak_season_end_str or peak_season_end_str == "N/A":
        peak_season_end_str = "August"

    return {
        "year_round": year_round,
        "start_month": start_month_str,
        "end_month": end_month_str,
        "peak_season_start": month_to_first_day(peak_season_start_str),
        "peak_season_end": month_to_last_day(peak_season_end_str)
    }

def month_to_first_day(month_name: str) -> str:
    month_map = {
        "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
        "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
        "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
        "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
    }
    if month_name in month_map:
        return f"{month_map[month_name][0]}-01"
    return "05-01"

def month_to_last_day(month_name: str) -> str:
    month_map = {
        "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
        "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
        "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
        "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
    }
    if month_name in month_map:
        return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
    return "08-31"

```

## hubspot_integration\__init__.py
```python
# hubspot/__init__.py

# This file can be empty, it just marks the directory as a Python package

```

## hubspot_integration\fetch_leads.py
```python
# File: fetch_hubspot_leads.py

__package__ = 'hubspot_integration'

from .hubspot_api import (
    get_hubspot_leads,
    get_lead_data_from_hubspot,
    get_contact_properties
)

def main():
    # Retrieve up to 10 leads from HubSpot
    lead_ids = get_hubspot_leads()  # returns a list of contact IDs

    # For each lead, fetch the contact properties (including email)
    for lead_id in lead_ids:
        contact_props = get_contact_properties(lead_id)
        email = contact_props.get("email", "No email found")
        print(f"Lead ID: {lead_id}, Email: {email}")

if __name__ == "__main__":
    main()

```

## hubspot_integration\hubspot_utils.py
```python
import re
from datetime import datetime
from html import unescape
from dateutil.parser import parse as parse_datetime
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def clean_note_body(text: str) -> str:
    """
    Cleans and normalizes HubSpot note body text by removing HTML tags, campaign footers,
    and standardizing whitespace and special characters.
    
    :param text: str - Raw note text from HubSpot
    :return: str - Cleaned and normalized text
    """
    text = unescape(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'Email sent from campaign.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Subject:\s*.*?Quick Question.*?)(?=(Text:|$))', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'(Text:\s*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[cid:.*?\]', '', text)
    text = re.sub(r'\[Logo, company name.*?\]', '', text)
    text = text.replace('â€œ', '"').replace('â€', '"').replace('â€™', "'").replace('â€�', '"')
    text = text.strip()
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n\s+\n', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text

def identify_sender(note_body: str) -> str:
    """
    Identifies the sender of a note by searching for known names in the note body.
    
    :param note_body: str - The cleaned body text of the note
    :return: str - Name of the identified sender or "Unknown Sender"
    """
    known_senders = ["Ryan Donovan", "Ty Hayes", "Ryan", "Ty"]
    for line in note_body.splitlines():
        line = line.strip()
        for sender in known_senders:
            if sender.lower() in line.lower():
                return sender
    return "Unknown Sender"

def format_timestamp(timestamp: str) -> str:
    """
    Formats a timestamp string into a standardized datetime format.
    
    :param timestamp: str - Raw timestamp string from HubSpot
    :return: str - Formatted timestamp (YYYY-MM-DD HH:MM AM/PM) or "N/A" if invalid
    """
    if not timestamp:
        return "N/A"
    try:
        dt = parse_datetime(timestamp)
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except:
        return timestamp

def summarize_activities(activities: str) -> str:
    """
    Uses GPT-4 to generate a concise summary of user interaction activities.
    
    :param activities: str - Raw text of user interaction history
    :return: str - Summarized activities or "No recent activity found" if empty
    """
    if not activities.strip():
        return "No recent activity found."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional assistant that summarizes user interaction history."},
            {"role": "user", "content": f"Summarize the following activities in a concise manner:\n\n{activities}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

```

## hubspot_integration\lead_qualification.py
```python
def qualify_lead(lead_data: dict) -> dict:
    score = 0
    reasons = []

    # Example qualification criteria
    if "Manager" in lead_data.get("job_title", ""):
        score += 10
    else:
        reasons.append("Job title does not indicate a decision-making role")

    if lead_data.get("interactions"):
        score += 5
    else:
        reasons.append("No recent interactions or engagement detected")

    qualified = score >= 10
    return {
        "qualified": qualified,
        "score": score,
        "reasons": reasons if not qualified else []
    }

```

## main.py
```python
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

```

## scheduling\__init__.py
```python
# This file can be empty, it just marks the directory as a Python package 
```

## scheduling\database.py
```python
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
    """Get database connection."""
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
    """Initialize database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        logger.info("Starting init_db...")

        # Create emails table if it doesn't exist
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                         WHERE object_id = OBJECT_ID(N'[dbo].[emails]') 
                         AND type in (N'U'))
            BEGIN
                CREATE TABLE dbo.emails (
                    email_id            INT IDENTITY(1,1) PRIMARY KEY,
                    lead_id            INT NOT NULL,
                    name               VARCHAR(100),
                    email_address      VARCHAR(255),
                    sequence_num       INT NULL,
                    body               VARCHAR(MAX),
                    scheduled_send_date DATETIME NULL,
                    actual_send_date   DATETIME NULL,
                    created_at         DATETIME DEFAULT GETDATE(),
                    status             VARCHAR(50) DEFAULT 'pending',
                    draft_id           VARCHAR(100) NULL,
                    gmail_id           VARCHAR(100)
                )
            END
        """)
        conn.commit()
        logger.info("init_db completed successfully. Emails table created if it didn't exist.")
        
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
    """Clear the emails table in the database."""
    try:
        with get_db_connection() as conn:
            logger.debug("Clearing emails table")
            
            query = "DELETE FROM dbo.emails"
            logger.debug(f"Executing: {query}")
            conn.execute(query)
                
            logger.info("Successfully cleared emails table")

    except Exception as e:
        logger.exception(f"Failed to clear emails table: {str(e)}")
        raise e

def store_email_draft(cursor, lead_id: int, name: str = None,
                     email_address: str = None,
                     sequence_num: int = None,
                     body: str = None,
                     scheduled_send_date: datetime = None,
                     draft_id: str = None,
                     status: str = 'pending') -> int:
    """
    Store email draft in database. Returns email_id.
    
    Table schema:
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
    # First check if this draft_id already exists
    cursor.execute("""
        SELECT email_id FROM emails 
        WHERE draft_id = ? AND lead_id = ?
    """, (draft_id, lead_id))
    
    existing = cursor.fetchone()
    if existing:
        # Update existing record instead of creating new one
        cursor.execute("""
            UPDATE emails 
            SET name = ?,
                email_address = ?,
                sequence_num = ?,
                body = ?,
                scheduled_send_date = ?,
                status = ?
            WHERE draft_id = ? AND lead_id = ?
        """, (
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id,
            lead_id
        ))
        return existing[0]
    else:
        # Insert new record
        cursor.execute("""
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
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
        """, (
            lead_id,
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id
        ))
        cursor.execute("SELECT SCOPE_IDENTITY()")
        return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")

```

## scheduling\extended_lead_storage.py
```python
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

```

## scheduling\followup_generation.py
```python
# followup_generation.py

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window, adjust_send_time
from datetime import datetime, timedelta
import random


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

```

## scheduling\followup_scheduler.py
```python
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
                e.name,
                e.body,
                e.created_at,
                e.email_address
            FROM emails e
            JOIN leads l ON l.lead_id = e.lead_id
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
            lead_id, email_id, email, first_name, name, body, created_at, email_address = row
            
            # Package original email data
            original_email = {
                'email': email,
                'first_name': first_name,
                'name': name,
                'body': body,
                'created_at': created_at,
                'email_address': email_address
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
                    message_text=followup_data['body']
                )

                if draft_result and draft_result.get("status") == "ok":
                    # Store in database with scheduled_send_date
                    store_email_draft(
                        cursor,
                        lead_id=lead_id,
                        body=followup_data['body'],
                        email_address=followup_data['email'],
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

```

## scheduling\sql_lookup.py
```python
# scheduling/sql_lookup.py

from typing import Dict
from scheduling.database import get_db_connection
from utils.logging_setup import logger

def build_lead_sheet_from_sql(email: str) -> Dict:
    """
    Attempt to build a lead_sheet dictionary from SQL tables
    (leads, lead_properties, companies, company_properties)
    for the given `email`.
    
    Returns a dict if found, or {} if not found.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Check if lead is present in dbo.leads
        cursor.execute("""
            SELECT 
                l.lead_id,
                l.first_name,
                l.last_name,
                l.role,
                l.hs_object_id,
                l.hs_createdate,
                l.hs_lastmodifieddate,
                l.company_id
            FROM dbo.leads l
            WHERE l.email = ?
        """, (email,))
        row = cursor.fetchone()

        if not row:
            return {}

        lead_id = row.lead_id
        first_name = row.first_name or ""
        last_name = row.last_name or ""
        role = row.role or ""
        hs_object_id = row.hs_object_id or ""
        hs_createdate = row.hs_createdate
        hs_lastmodifieddate = row.hs_lastmodifieddate
        company_id = row.company_id

        # 2) lead_properties
        cursor.execute("""
            SELECT phone, lifecyclestage, competitor_analysis, last_response_date
            FROM dbo.lead_properties
            WHERE lead_id = ?
        """, (lead_id,))
        lp = cursor.fetchone()
        phone = lp.phone if lp else ""
        lifecyclestage = lp.lifecyclestage if lp else ""
        competitor_analysis = lp.competitor_analysis if lp else ""
        last_response_date = lp.last_response_date if lp else None

        # 3) companies
        company_data = {}
        if company_id:
            cursor.execute("""
                SELECT 
                    c.name,
                    c.city,
                    c.state,
                    c.hs_object_id,
                    c.hs_createdate,
                    c.hs_lastmodifieddate,
                    
                    -- If you DO have year_round / start_month / end_month in companies:
                    c.year_round,
                    c.start_month,
                    c.end_month,
                    c.peak_season_start,
                    c.peak_season_end
                
                FROM dbo.companies c
                WHERE c.company_id = ?
            """, (company_id,))
            co = cursor.fetchone()

            if co:
                company_data = {
                    "name": co.name or "",
                    "city": co.city or "",
                    "state": co.state or "",
                    "hs_object_id": co.hs_object_id or "",
                    "createdate": co.hs_createdate.isoformat() if co.hs_createdate else None,
                    "hs_lastmodifieddate": co.hs_lastmodifieddate.isoformat() if co.hs_lastmodifieddate else None,

                    # If your DB truly has these columns:
                    "year_round": co.year_round or "",
                    "start_month": co.start_month or "",
                    "end_month": co.end_month or "",
                    "peak_season_start": co.peak_season_start or "",
                    "peak_season_end": co.peak_season_end or ""
                }

            # 4) company_properties (no peak_season_* columns here anymore)
            cursor.execute("""
                SELECT 
                    annualrevenue,
                    competitor_analysis,
                    company_overview
                FROM dbo.company_properties
                WHERE company_id = ?
            """, (company_id,))
            cp = cursor.fetchone()
            if cp:
                # Merge these fields into same dict
                company_data["annualrevenue"] = cp.annualrevenue or ""
                # competitor_analysis might overlap, decide how to unify:
                if cp.competitor_analysis:
                    competitor_analysis = cp.competitor_analysis  # or keep the lead_properties version
                company_data["company_overview"] = cp.company_overview or ""

        # 5) Build lead_data
        lead_data = {
            "email": email,
            "firstname": first_name,
            "lastname": last_name,
            "jobtitle": role,
            "hs_object_id": hs_object_id,
            "createdate": hs_createdate.isoformat() if hs_createdate else None,
            "lastmodifieddate": hs_lastmodifieddate.isoformat() if hs_lastmodifieddate else None,
            "phone": phone,
            "lifecyclestage": lifecyclestage,
            "company_data": company_data
        }

        # 6) Build analysis
        analysis_data = {
            "competitor_analysis": competitor_analysis,
            "previous_interactions": {
                "last_response_date": last_response_date.isoformat() if last_response_date else None
            }
        }

        lead_sheet = {
            "metadata": {
                "status": "success",
                "lead_email": email,
                "source": "sql"
            },
            "lead_data": lead_data,
            "analysis": analysis_data
        }
        return lead_sheet

    except Exception as e:
        logger.error(f"Error building lead sheet from SQL for email={email}: {e}")
        return {}
    finally:
        conn.close()

```

## scripts\__init__.py
```python

```

## scripts\build_template.py
```python
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
        # Get company data with detailed logging
        company_data = data.get("company_data", {})
        logger.debug(f"Full data context: {json.dumps(data, indent=2)}")
        logger.debug(f"Company data for replacements: {json.dumps(company_data, indent=2)}")
        
        # Get company short name with explicit logging
        company_short_name = company_data.get("company_short_name", "")
        company_full_name = company_data.get("name", "")
        logger.debug(f"Found company_short_name: '{company_short_name}'")
        logger.debug(f"Found company_full_name: '{company_full_name}'")
        
        # Build replacements dict with logging
        replacements = {
            "[firstname]": data.get("lead_data", {}).get("firstname", ""),
            "[LastName]": data.get("lead_data", {}).get("lastname", ""),
            "[companyname]": company_full_name,
            "[company_short_name]": company_short_name or company_full_name,  # Fall back to full name
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
```

## scripts\check_reviewed_drafts.py
```python
# File: scripts/check_reviewed_drafts.py

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now import project modules
from scheduling.database import get_db_connection
from utils.gmail_integration import get_gmail_service, create_draft
from utils.logging_setup import logger
from services.data_gatherer_service import DataGathererService
from scheduling.followup_generation import generate_followup_email_xai
from utils.gmail_integration import store_draft_info
from scripts.golf_outreach_strategy import get_best_outreach_window, adjust_send_time
from datetime import datetime
from typing import Optional
import random

###########################
# CONFIG / CONSTANTS
###########################
TO_REVIEW_LABEL = "to_review"
REVIEWED_LABEL = "reviewed"
SQL_DRAFT_STATUS = "draft"

def main():
    """
    Checks all 'draft' emails in the SQL table with sequence_num = 1.
    If an email's Gmail label is 'reviewed', we generate a follow-up draft
    (sequence_num = 2) and store it with label='to_review'.
    If the label is 'to_review', we skip it.
    """
    try:
        # 1) Connect to DB
        conn = get_db_connection()
        cursor = conn.cursor()

        logger.info("Starting 'check_reviewed_drafts' workflow.")

        # 2) Fetch all "draft" emails with sequence_num=1
        #    that presumably need to be reviewed or followed up.
        cursor.execute("""
            SELECT email_id, lead_id, draft_id, subject, body
            FROM emails
            WHERE status = ?
              AND sequence_num = 1
        """, (SQL_DRAFT_STATUS,))
        results = cursor.fetchall()

        if not results:
            logger.info("No draft emails (sequence_num=1) found.")
            return

        logger.info(f"Found {len(results)} draft emails with sequence_num=1.")

        # 3) Get Gmail service
        gmail_service = get_gmail_service()
        if not gmail_service:
            logger.error("Could not initialize Gmail service.")
            return

        # 4) For each email draft, check the label in Gmail
        for (email_id, lead_id, draft_id, subject, body) in results:
            logger.info(f"Processing email_id={email_id}, draft_id={draft_id}")

            if not draft_id:
                logger.warning("No draft_id found in DB. Skipping.")
                continue

            # 4a) Retrieve the draft message from Gmail
            message = _get_gmail_draft_by_id(gmail_service, draft_id)
            if not message:
                logger.error(f"Failed to retrieve Gmail draft for draft_id={draft_id}")
                continue

            # 4b) Extract current labels
            current_labels = message.get("labelIds", [])
            if not current_labels:
                logger.info(f"No labels found for draft_id={draft_id}. Skipping.")
                continue

            # Normalize label IDs to strings
            label_names = _translate_label_ids_to_names(gmail_service, current_labels)
            logger.debug(f"Draft {draft_id} has labels: {label_names}")

            # 5) If label == "to_review", skip
            #    If label == "reviewed", create a follow-up draft
            if REVIEWED_LABEL.lower() in [ln.lower() for ln in label_names]:
                logger.info(f"Draft {draft_id} has label '{REVIEWED_LABEL}'. Creating follow-up.")
                
                # 5a) Generate follow-up (sequence_num=2)
                followup_data = _generate_followup(gmail_service, lead_id, email_id, subject, body)
                if followup_data:
                    logger.info("Follow-up created successfully.")
                else:
                    logger.error("Failed to create follow-up.")
            else:
                # If we only see "to_review", skip
                if TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
                    logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. Skipping.")
                else:
                    logger.info(f"Draft {draft_id} has no matching logic for labels={label_names}")

        logger.info("Completed checking reviewed drafts.")

    except Exception as e:
        logger.exception(f"Error in check_reviewed_drafts workflow: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

###########################
# HELPER FUNCTIONS
###########################

def _get_gmail_draft_by_id(service, draft_id: str) -> Optional[dict]:
    """
    Retrieve a specific Gmail draft by its draftId.
    Returns None if not found or error.
    """
    try:
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        return draft.get("message", {})
    except Exception as e:
        logger.error(f"Error fetching draft {draft_id}: {str(e)}")
        return None


def _translate_label_ids_to_names(service, label_ids: list) -> list:
    """
    Given a list of labelIds, returns the corresponding label names.
    For example: ["Label_12345"] -> ["to_review"].
    """
    # Retrieve all labels
    try:
        labels_response = service.users().labels().list(userId='me').execute()
        all_labels = labels_response.get('labels', [])
        id_to_name = {lbl["id"]: lbl["name"] for lbl in all_labels}

        label_names = []
        for lid in label_ids:
            label_names.append(id_to_name.get(lid, lid))  # fallback to ID if not found
        return label_names
    except Exception as e:
        logger.error(f"Error translating label IDs: {str(e)}")
        return label_ids  # fallback

def _generate_followup(gmail_service, lead_id: int, original_email_id: int, orig_subject: str, orig_body: str) -> bool:
    """
    Generates a follow-up draft (sequence_num=2) and stores it as a new
    Gmail draft with label 'to_review'.
    """
    try:
        # 1) Get lead data from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT l.email, l.first_name, c.name, c.state
            FROM leads l
            LEFT JOIN companies c ON l.company_id = c.company_id
            WHERE l.lead_id = ?
        """, (lead_id,))
        
        lead_data = cursor.fetchone()
        if not lead_data:
            logger.error(f"No lead found for lead_id={lead_id}")
            return False
            
        lead_email, first_name, company_name, state = lead_data
        
        # 2) Build the original email data structure with proper datetime object
        original_email = {
            "email": lead_email,
            "first_name": first_name,
            "name": company_name,
            "state": state,
            "subject": orig_subject,
            "body": orig_body,
            "created_at": datetime.now()  # Use datetime object instead of string
        }

        # 3) Generate follow-up content
        followup_data = generate_followup_email_xai(
            lead_id=lead_id,
            email_id=original_email_id,
            sequence_num=2,
            original_email=original_email
        )
        
        if not followup_data or not followup_data.get("scheduled_send_date"):
            logger.error("No valid followup_data generated.")
            return False

        # 4) Create new Gmail draft
        draft_result = create_draft(
            sender="me",
            to=lead_email,
            subject=followup_data['subject'],
            message_text=followup_data['body']
        )
        
        if draft_result.get("status") != "ok":
            logger.error("Failed to create Gmail draft for follow-up.")
            return False

        new_draft_id = draft_result["draft_id"]
        logger.info(f"Follow-up draft created. draft_id={new_draft_id}")

        # 5) Store the new draft in DB
        store_draft_info(
            lead_id=lead_id,
            draft_id=new_draft_id,
            scheduled_date=followup_data.get('scheduled_send_date'),
            subject=followup_data['subject'],
            body=followup_data['body'],
            sequence_num=2
        )

        # 6) Add label 'to_review' to the new draft
        _add_label_to_message(
            gmail_service,
            new_draft_id,
            label_name=TO_REVIEW_LABEL
        )

        return True

    except Exception as e:
        logger.exception(f"Error generating follow-up for lead_id={lead_id}: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def _add_label_to_message(service, draft_id: str, label_name: str):
    """
    Adds a label to a Gmail draft message. We'll need to fetch the message
    ID from the draft, then modify the labels.
    """
    try:
        # 1) Get the draft
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        message_id = draft["message"]["id"]

        # 2) Get or create the label
        label_id = _get_or_create_label(service, label_name)
        if not label_id:
            logger.warning(f"Could not find/create label '{label_name}'. Skipping label add.")
            return

        # 3) Apply label to the message
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id]}
        ).execute()
        logger.debug(f"Label '{label_name}' added to new follow-up draft_id={draft_id}.")

    except Exception as e:
        logger.error(f"Error adding label '{label_name}' to draft '{draft_id}': {str(e)}")

def _get_or_create_label(service, label_name: str) -> Optional[str]:
    """
    Retrieves or creates the specified Gmail label and returns its labelId.
    """
    try:
        user_id = 'me'
        labels_response = service.users().labels().list(userId=user_id).execute()
        existing_labels = labels_response.get('labels', [])

        # Try finding existing label by name
        for lbl in existing_labels:
            if lbl['name'].lower() == label_name.lower():
                return lbl['id']

        # If not found, create it
        create_body = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show',
        }
        new_label = service.users().labels().create(
            userId=user_id, body=create_body
        ).execute()
        logger.info(f"Created new Gmail label: {label_name}")
        return new_label['id']

    except Exception as e:
        logger.error(f"Error in _get_or_create_label({label_name}): {str(e)}")
        return None


if __name__ == "__main__":
    main()

```

## scripts\get_random_contacts.py
```python
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import HUBSPOT_API_KEY
from services.hubspot_service import HubspotService

def main():
    # Initialize HubSpot service
    hubspot = HubspotService(HUBSPOT_API_KEY)
    
    # Get 3 random contacts
    contacts = hubspot.get_random_contacts(count=3)
    
    # Print results
    print("\nRandom Contacts from HubSpot:")
    print("=" * 40)
    for contact in contacts:
        print(f"\nEmail: {contact['email']}")
        print(f"Name: {contact['first_name']} {contact['last_name']}")
        print(f"Company: {contact['company']}")
    print("\n")

if __name__ == "__main__":
    main() 
```

## scripts\golf_outreach_strategy.py
```python
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

```

## scripts\job_title_categories.py
```python
# scripts/job_title_categories.py

def categorize_job_title(title: str) -> str:
    """Categorize job title into standardized roles."""
    if not title:
        return "general_manager"  # Default fallback
    
    title = title.lower().strip()
    
    # F&B Related Titles
    fb_titles = [
        "f&b", "food", "beverage", "dining", "restaurant", "culinary",
        "chef", "kitchen", "hospitality", "catering", "banquet"
    ]
    
    # Membership Related Titles
    membership_titles = [
        "member", "membership", "marketing", "sales", "business development"
    ]
    
    # Golf Operations Titles
    golf_ops_titles = [
        "golf pro", "pro shop", "golf operations", "head pro",
        "director of golf", "golf director", "pga", "golf professional"
    ]
    
    # Check categories in order of specificity
    for fb_term in fb_titles:
        if fb_term in title:
            return "fb_manager"
            
    for membership_term in membership_titles:
        if membership_term in title:
            return "membership_director"
            
    for golf_term in golf_ops_titles:
        if golf_term in title:
            return "golf_operations"
    
    # Default to general manager for other titles
    return "general_manager"

```

## scripts\migrate_emails_table.py
```python
#!/usr/bin/env python3
# scripts/migrate_emails_table.py

import sys
from pathlib import Path
import pyodbc
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_setup import logger
from config.settings import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    """Get database connection."""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD}"
    )
    return pyodbc.connect(conn_str)

def migrate_emails_table():
    """
    Migrate the emails table to the new schema.
    Steps:
    1. Create new table with correct schema
    2. Copy data from old table to new table
    3. Drop old table
    4. Rename new table to original name
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        logger.info("Starting emails table migration...")
        
        # Step 1: Create new table with correct schema
        cursor.execute("""
            CREATE TABLE emails_new (
                email_id            INT IDENTITY(1,1) PRIMARY KEY,
                lead_id            INT NOT NULL,
                name               VARCHAR(100),
                email_address      VARCHAR(255),
                sequence_num       INT NULL,
                body              VARCHAR(MAX),
                scheduled_send_date DATETIME NULL,
                actual_send_date   DATETIME NULL,
                created_at         DATETIME DEFAULT GETDATE(),
                status             VARCHAR(50) DEFAULT 'pending',
                draft_id           VARCHAR(100) NULL,
                gmail_id           VARCHAR(100)
            )
        """)
        
        # Step 2: Copy data from old table to new table
        logger.info("Copying data to new table...")
        cursor.execute("""
            INSERT INTO emails_new (
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                actual_send_date,
                created_at,
                status,
                draft_id,
                gmail_id
            )
            SELECT 
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                actual_send_date,
                created_at,
                status,
                draft_id,
                gmail_id
            FROM emails
        """)
        
        # Get row counts for verification
        cursor.execute("SELECT COUNT(*) FROM emails")
        old_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM emails_new")
        new_count = cursor.fetchone()[0]
        
        logger.info(f"Old table row count: {old_count}")
        logger.info(f"New table row count: {new_count}")
        
        if old_count != new_count:
            raise ValueError(f"Row count mismatch: old={old_count}, new={new_count}")
        
        # Step 3: Drop old table
        logger.info("Dropping old table...")
        cursor.execute("DROP TABLE emails")
        
        # Step 4: Rename new table to original name
        logger.info("Renaming new table...")
        cursor.execute("EXEC sp_rename 'emails_new', 'emails'")
        
        conn.commit()
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}", exc_info=True)
        conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def verify_migration():
    """Verify the migration was successful."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'emails'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        logger.info("\nTable structure verification:")
        for col in columns:
            logger.info(f"Column: {col[0]}, Type: {col[1]}, Length: {col[2]}")
        
        # Check for any NULL values in required fields
        cursor.execute("""
            SELECT COUNT(*) 
            FROM emails 
            WHERE lead_id IS NULL
        """)
        null_leads = cursor.fetchone()[0]
        
        if null_leads > 0:
            logger.warning(f"Found {null_leads} rows with NULL lead_id")
        else:
            logger.info("No NULL values found in required fields")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting migration process...")
    
    # Confirm with user
    response = input("This will modify the emails table. Are you sure you want to continue? (y/N): ")
    if response.lower() != 'y':
        logger.info("Migration cancelled by user")
        sys.exit(0)
    
    try:
        migrate_emails_table()
        verify_migration()
        logger.info("Migration and verification completed successfully!")
    except Exception as e:
        logger.error("Migration failed!", exc_info=True)
        sys.exit(1) 
```

## scripts\ping_hubspot_for_gm.py
```python
# File: scripts/ping_hubspot_for_gm.py

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random

# Imports from your codebase
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from scripts.golf_outreach_strategy import (
    get_best_month, 
    get_best_outreach_window,
    adjust_send_time
)
from services.data_gatherer_service import DataGathererService  # optional if you want to reuse
from utils.logging_setup import logger

# Initialize data gatherer service
data_gatherer = DataGathererService()  # Add this line

###########################
# CONFIG / CONSTANTS
###########################
BATCH_SIZE = 25  # Keep original batch size
DEFAULT_GEOGRAPHY = "Year-Round Golf"  # Default geography if none specified
MIN_REVENUE = 1000000  # Minimum annual revenue filter
EXCLUDED_STATES = []    # List of states to exclude from search
TEST_MODE = True  # Add test mode flag
TEST_LIMIT = 10  # Number of companies to process in test mode

###########################
# SCRIPT START
###########################

def calculate_send_date(geography, profile_type, state, preferred_days, preferred_time):
    """
    Calculate optimal send date based on geography and profile.
    
    Args:
        geography: Geographic region of the club
        profile_type: Type of contact profile (e.g. General Manager)
        state: Club's state location
        preferred_days: List of preferred weekdays for sending
        preferred_time: Dict with start/end hours for sending
        
    Returns:
        datetime: Optimal send date and time
    """
    # Start with tomorrow
    base_date = datetime.now() + timedelta(days=1)
    
    # Find next preferred day
    while base_date.weekday() not in preferred_days:
        base_date += timedelta(days=1)
        
    # Set time within preferred window
    send_hour = preferred_time["start"]
    if random.random() < 0.5:  # 50% chance to use later hour
        send_hour += 1
        
    return base_date.replace(
        hour=send_hour,
        minute=random.randint(0, 59),
        second=0,
        microsecond=0
    )

def _calculate_lead_score(
    revenue: float,
    club_type: str,
    geography: str,
    current_month: int,
    best_months: List[int],
    season_data: Dict[str, Any]
) -> float:
    """
    Calculate a score for lead prioritization based on multiple factors.
    
    Args:
        revenue: Annual revenue
        club_type: Type of club
        geography: Geographic region
        current_month: Current month number
        best_months: List of best months for outreach
        season_data: Dictionary containing peak season information
        
    Returns:
        float: Score from 0-100
    """
    score = 0.0
    
    # Revenue scoring (up to 40 points)
    if revenue >= 5000000:
        score += 40
    elif revenue >= 2000000:
        score += 30
    elif revenue >= 1000000:
        score += 20

    # Club type scoring (up to 30 points)
    club_type_scores = {
        "Private": 30,
        "Semi-Private": 25,
        "Resort": 20,
        "Public": 15
    }
    score += club_type_scores.get(club_type, 0)

    # Timing/Season scoring (up to 30 points)
    if current_month in best_months:
        score += 30
    elif abs(current_month - min(best_months)) <= 1:
        score += 15

    # Geography bonus (up to 10 points)
    if geography == "Year-Round Golf":
        score += 10
    elif geography in ["Peak Summer Season", "Peak Winter Season"]:
        score += 5

    return score

def main():
    """
    Enhanced version with seasonal intelligence and lead scoring.
    """
    try:
        logger.info("==== Starting enhanced ping_hubspot_for_gm workflow ====")
        
        # Initialize services
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        data_gatherer = DataGathererService()

        # Get current timing info
        current_month = datetime.now().month
        current_date = datetime.now()
        
        # Fetch companies with geographic data
        all_companies = _search_companies_with_filters(
            hubspot,
            batch_size=BATCH_SIZE,
            min_revenue=MIN_REVENUE,
            excluded_states=EXCLUDED_STATES
        )
        
        logger.info(f"Retrieved {len(all_companies)} companies matching initial criteria")

        # Store scored and prioritized leads
        scored_leads = []

        for company in all_companies:
            props = company.get("properties", {})
            geography_data = company.get("geography_data", {})
            
            # Get seasonal data
            season_data = {
                'peak_season_start': props.get('peak_season_start'),
                'peak_season_end': props.get('peak_season_end')
            }
            
            # Get outreach window
            outreach_window = get_best_outreach_window(
                persona="General Manager",
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                club_type=geography_data.get('club_type'),
                season_data=season_data
            )
            
            # Calculate lead score
            lead_score = _calculate_lead_score(
                revenue=float(props.get('annualrevenue', 0) or 0),
                club_type=geography_data.get('club_type', ''),
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                current_month=current_month,
                best_months=outreach_window["Best Month"],
                season_data=season_data
            )
            
            # Skip low-scoring leads (optional)
            if lead_score < 20:  # Adjust threshold as needed
                logger.debug(f"Skipping {props.get('name')}: Low score ({lead_score})")
                continue
            
            # Calculate optimal send time
            send_date = calculate_send_date(
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                profile_type="General Manager",
                state=props.get('state', ''),
                preferred_days=outreach_window["Best Day"],
                preferred_time=outreach_window["Best Time"]
            )
            
            adjusted_send_time = adjust_send_time(send_date, props.get('state'))
            
            # Process contacts
            company_id = company.get("id")
            if not company_id:
                continue
                
            associated_contacts = _get_contacts_for_company(hubspot, company_id)
            if not associated_contacts:
                continue

            for contact in associated_contacts:
                c_props = contact.get("properties", {})
                jobtitle = c_props.get("jobtitle", "")
                
                if not jobtitle or not is_general_manager_jobtitle(jobtitle):
                    continue
                    
                email = c_props.get("email", "missing@noemail.com")
                first_name = c_props.get("firstname", "")
                last_name = c_props.get("lastname", "")
                
                scored_leads.append({
                    "score": lead_score,
                    "email": email,
                    "name": f"{first_name} {last_name}".strip(),
                    "company": props.get("name", ""),
                    "jobtitle": jobtitle,
                    "geography": geography_data.get('geography', DEFAULT_GEOGRAPHY),
                    "best_months": outreach_window["Best Month"],
                    "optimal_send_time": adjusted_send_time,
                    "club_type": geography_data.get('club_type', ''),
                    "peak_season": season_data,
                    "revenue": props.get('annualrevenue', 'N/A')
                })

        # Sort leads by score (highest first) and then by send time
        scored_leads.sort(key=lambda x: (-x["score"], x["optimal_send_time"]))
        
        logger.info(f"Found {len(scored_leads)} scored and prioritized GM leads")
        
        # Print results with scores
        for lead in scored_leads:
            print(
                f"Score: {lead['score']:.1f} | "
                f"Send Time: {lead['optimal_send_time'].strftime('%Y-%m-%d %H:%M')} | "
                f"{lead['name']} | "
                f"{lead['company']} | "
                f"Revenue: ${float(lead['revenue'] or 0):,.0f} | "
                f"Type: {lead['club_type']} | "
                f"Geography: {lead['geography']}"
            )

    except Exception as e:
        logger.exception(f"Error in enhanced ping_hubspot_for_gm: {str(e)}")


###########################
# HELPER FUNCTIONS
###########################

def is_general_manager_jobtitle(title: str) -> bool:
    """
    Returns True if the jobtitle indicates 'General Manager'.
    
    Args:
        title: Job title string to check
        
    Returns:
        bool: True if title contains 'general manager'
    """
    title_lower = title.lower()
    # simple approach
    if "general manager" in title_lower:
        return True
    return False


def _search_companies_in_batches(hubspot: HubspotService, batch_size=25, max_pages=1) -> List[Dict[str, Any]]:
    """
    Searches for companies in HubSpot using the CRM API with pagination.
    
    Args:
        hubspot: HubspotService instance
        batch_size: Number of records per request
        max_pages: Maximum number of pages to fetch
        
    Returns:
        List of company records
    """
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    after = None
    all_results = []
    pages_fetched = 0

    while pages_fetched < max_pages:
        # Build request payload
        payload = {
            "limit": batch_size,
            "properties": ["name", "city", "state", "annualrevenue", "company_type"],
            # you can add filters if you want to only get certain companies
            # "filterGroups": [...],
        }
        if after:
            payload["after"] = after

        try:
            # Make API request
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break

            # Process results
            results = response.get("results", [])
            all_results.extend(results)

            # Handle pagination
            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            else:
                after = next_link

            pages_fetched += 1
        except Exception as e:
            logger.error(f"Error fetching companies from HubSpot page={pages_fetched}: {str(e)}")
            break

    return all_results


def _get_contacts_for_company(hubspot: HubspotService, company_id: str) -> List[Dict[str, Any]]:
    """
    Find all associated contacts for a company.
    
    Args:
        hubspot: HubspotService instance
        company_id: ID of company to get contacts for
        
    Returns:
        List of contact records
    """
    # HubSpot API: GET /crm/v3/objects/companies/{companyId}/associations/contacts
    url = f"{hubspot.base_url}/crm/v3/objects/companies/{company_id}/associations/contacts"
    
    try:
        # Get contact associations
        response = hubspot._make_hubspot_get(url)
        if not response:
            return []
        contact_associations = response.get("results", [])
        # Each association looks like: {"id": <contactId>, "type": "company_to_contact"}
        
        if not contact_associations:
            return []

        # Collect the contact IDs
        contact_ids = [assoc["id"] for assoc in contact_associations if assoc.get("id")]

        # Bulk fetch each contact's properties
        contact_records = []
        for cid in contact_ids:
            # Reuse hubspot.get_contact_properties (which returns minimal)
            # or do a direct GET / search for that contact object.
            contact_data = _get_contact_by_id(hubspot, cid)
            if contact_data:
                contact_records.append(contact_data)
        
        return contact_records

    except Exception as e:
        logger.error(f"Error fetching contacts for company_id={company_id}: {str(e)}")
        return []


def _get_contact_by_id(hubspot: HubspotService, contact_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single contact by ID with all relevant properties.
    
    Args:
        hubspot: HubspotService instance
        contact_id: ID of contact to retrieve
        
    Returns:
        Contact record dict or None if error
    """
    url = f"{hubspot.base_url}/crm/v3/objects/contacts/{contact_id}"
    query_params = {
        "properties": ["email", "firstname", "lastname", "jobtitle"], 
        "archived": "false"
    }
    try:
        response = hubspot._make_hubspot_get(url, params=query_params)
        return response
    except Exception as e:
        logger.error(f"Error fetching contact_id={contact_id}: {str(e)}")
        return None


def _search_companies_with_filters(
    hubspot: HubspotService,
    batch_size: int = 25,
    min_revenue: float = 1000000,
    excluded_states: List[str] = None
) -> List[Dict[str, Any]]:
    """Enhanced company search with filtering."""
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    after = None
    all_results = []

    while True and (not TEST_MODE or len(all_results) < TEST_LIMIT):
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "annualrevenue",
                            "operator": "GTE",
                            "value": str(min_revenue)
                        }
                    ]
                }
            ],
            "properties": [
                "name", "city", "state", "annualrevenue",
                "company_type", "industry", "website",
                "peak_season_start", "peak_season_end"
            ],
            "limit": min(batch_size, TEST_LIMIT) if TEST_MODE else batch_size
        }
        
        if after:
            payload["after"] = after

        try:
            logger.info(f"Fetching companies (Test Mode: {TEST_MODE})")
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break

            results = response.get("results", [])
            
            # Filter out excluded states
            if excluded_states:
                results = [
                    r for r in results 
                    if r.get("properties", {}).get("state") not in excluded_states
                ]

            all_results.extend(results)
            
            logger.info(f"Retrieved {len(all_results)} companies so far")

            # Break if we've reached test limit
            if TEST_MODE and len(all_results) >= TEST_LIMIT:
                logger.info(f"Test mode: Reached limit of {TEST_LIMIT} companies")
                break

            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link

        except Exception as e:
            logger.error(f"Error fetching companies: {str(e)}")
            break

    # Ensure we don't exceed test limit
    if TEST_MODE:
        all_results = all_results[:TEST_LIMIT]
        logger.info(f"Test mode: Returning {len(all_results)} companies")

    return all_results


if __name__ == "__main__":
    main()

```

## scripts\schedule_outreach.py
```python
"""
scripts/schedule_outreach.py

Schedules multiple outreach steps (emails) via APScheduler.
Now also integrates best outreach-window logic from golf_outreach_strategy.
"""

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

# NEW IMPORT
from scripts.golf_outreach_strategy import get_best_outreach_window

# Example outreach schedule steps
OUTREACH_SCHEDULE = [
    {
        "name": "Intro Email (Day 1)",
        "days_from_now": 1,
        "subject": "Enhancing Member Experience with Swoop Golf",
        "body": (
            "Hi [Name],\n\n"
            "I hope this finds you well. I wanted to reach out about Swoop Golf's on-demand F&B platform, "
            "which is helping golf clubs like yours modernize their on-course service and enhance member satisfaction.\n\n"
            "Would you be open to a brief conversation about how we could help streamline your club's F&B "
            "operations while improving the member experience?"
        )
    },
    {
        "name": "Quick Follow-Up (Day 3)",
        "days_from_now": 3,
        "subject": "Quick follow-up: Swoop Golf",
        "body": (
            "Hello [Name],\n\n"
            "I wanted to quickly follow up on my previous email about Swoop Golf's F&B platform. "
            "Have you had a chance to consider how our solution might benefit your club's operations?\n\n"
            "I'd be happy to schedule a brief call to discuss your specific needs."
        )
    },
    # Add additional follow-up steps as needed...
]

def schedule_draft(step_details, sender, recipient, hubspot_contact_id):
    """
    Create a Gmail draft for scheduled sending.
    """
    draft_result = create_draft(
        sender=sender,
        to=recipient,
        subject=step_details["subject"],
        message_text=step_details["body"]
    )

    if draft_result["status"] != "ok":
        logger.error(f"Failed to create draft for step '{step_details['name']}'.")
        return

    draft_id = draft_result.get("draft_id")
    ideal_send_time = datetime.datetime.now() + datetime.timedelta(days=step_details["days_from_now"])
    logger.info(f"Created draft ID: {draft_id} for step '{step_details['name']}' to be sent at {ideal_send_time}.")


def main():
    """
    Main scheduling workflow:
     1) Determine the best outreach window for an example persona/season/club type
     2) Start APScheduler
     3) Schedule each step of the outreach
    """

    # Example of fetching recommended outreach window
    persona = "General Manager"
    geography = "Peak Summer Season"
    club_type = "Private Clubs"

    recommendation = get_best_outreach_window(persona, geography, club_type)
    logger.info(
        f"Recommended outreach for {persona}, {geography}, {club_type}: {recommendation}"
    )

    # If you want to incorporate recommended times/days into your scheduling logic,
    # you can parse or handle them here (for example, adjust 'days_from_now' or
    # specific times of day, etc.).

    # Start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()

    now = datetime.datetime.now()
    sender = "me"                   # 'me' means the authenticated Gmail user
    recipient = "someone@example.com"
    hubspot_contact_id = "255401"   # Example contact ID in HubSpot

    # Schedule each step for future sending
    for step in OUTREACH_SCHEDULE:
        run_time = now + datetime.timedelta(days=step["days_from_now"])
        job_id = f"job_{step['name'].replace(' ', '_')}"

        scheduler.add_job(
            schedule_draft,
            'date',
            run_date=run_time,
            id=job_id,
            args=[step, sender, recipient, hubspot_contact_id]
        )
        logger.info(f"Scheduled job '{job_id}' for {run_time}")

    try:
        logger.info("Scheduler running. Press Ctrl+C to exit.")
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()

```

## services\__init__.py
```python
"""
services/__init__.py

Package for business logic services that coordinate between
different parts of the application.
"""

```

## services\company_enrichment_service.py
```python
from typing import Dict, Any, Optional, Tuple
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window
from utils.web_fetch import fetch_website_html


class CompanyEnrichmentService:
    def __init__(self, api_key: str = HUBSPOT_API_KEY):
        logger.debug("Initializing CompanyEnrichmentService")
        self.hubspot = HubspotService(api_key=api_key)

    def enrich_company(self, company_id: str, additional_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Enriches company data with facility type and competitor information."""
        try:
            # Get current company info
            current_info = self._get_facility_info(company_id)
            if not current_info:
                return {"success": False, "error": "Failed to get company info"}

            # Get company name and location
            company_name = current_info.get('name', '')
            location = f"{current_info.get('city', '')}, {current_info.get('state', '')}"
            
            # IMPORTANT: Save the competitor value found in _get_facility_info
            competitor = current_info.get('competitor', 'Unknown')
            logger.debug(f"Competitor found in _get_facility_info: {competitor}")

            # Determine facility type
            facility_info = self._determine_facility_type(company_name, location)
            
            # Get seasonality info
            seasonality_info = self._determine_seasonality(current_info.get('state', ''))
            
            # Combine all new info
            new_info = {
                **facility_info,
                **seasonality_info,
                'competitor': competitor  # Explicitly set competitor here
            }
            
            # Add any additional data
            if additional_data:
                new_info.update(additional_data)
                
            # Log the competitor value before preparing updates
            logger.debug(f"Competitor value before _prepare_updates: {new_info.get('competitor', 'Unknown')}")
            
            # Prepare the final updates
            updates = self._prepare_updates(current_info, new_info)
            
            # Log the final competitor value
            logger.debug(f"Final competitor value in updates: {updates.get('competitor', 'Unknown')}")
            
            # Update the company in HubSpot
            if updates:
                success = self._update_company_properties(company_id, updates)
                if success:
                    return {"success": True, "data": updates}
            
            return {"success": False, "error": "Failed to update company"}
            
        except Exception as e:
            logger.error(f"Error enriching company {company_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_facility_info(self, company_id: str) -> Dict[str, Any]:
        """Fetches the company's current properties from HubSpot."""
        try:
            properties = [
                "name", "city", "state", "annualrevenue",
                "createdate", "hs_lastmodifieddate", "hs_object_id",
                "club_type", "facility_complexity", "has_pool",
                "has_tennis_courts", "number_of_holes",
                "geographic_seasonality", "public_private_flag",
                "club_info", "peak_season_start_month",
                "peak_season_end_month", "start_month", "end_month",
                "competitor", "domain", "company_short_name"
            ]
            
            company_data = self.hubspot.get_company_by_id(company_id, properties)
            
            if not company_data:
                return {}
            
            # Check domain for competitor software
            domain = company_data.get('properties', {}).get('domain', '')
            competitor = company_data.get('properties', {}).get('competitor', 'Unknown')
            
            if domain:
                # Try different URL variations
                urls_to_try = []
                base_url = domain.strip().lower()
                if not base_url.startswith('http'):
                    urls_to_try.extend([f"https://{base_url}", f"http://{base_url}"])
                else:
                    urls_to_try.append(base_url)
                
                # Add www. version if not present
                urls_to_try.extend([url.replace('://', '://www.') for url in urls_to_try])
                
                for url in urls_to_try:
                    try:
                        html_content = fetch_website_html(url)
                        if html_content:
                            html_lower = html_content.lower()
                            # Check for Club Essentials mentions first
                            clubessential_mentions = [
                                "copyright clubessential",
                                "clubessential, llc",
                                "www.clubessential.com",
                                "http://www.clubessential.com",
                                "clubessential"
                            ]
                            for mention in clubessential_mentions:
                                if mention in html_lower:
                                    competitor = "Club Essentials"
                                    logger.debug(f"Found Club Essentials on {url}")
                                    break
                                    
                            # Check for Jonas mentions if not Club Essentials
                            if competitor == "Unknown":
                                jonas_mentions = ["jonas club software", "jonas software", "jonasclub"]
                                for mention in jonas_mentions:
                                    if mention in html_lower:
                                        competitor = "Jonas"
                                        logger.debug(f"Found Jonas on {url}")
                                        break
                            
                            # Check for Northstar mentions if still Unknown
                            if competitor == "Unknown":
                                northstar_mentions = [
                                    "northstar technologies",
                                    "globalnorthstar.com",
                                    "northstar club management",
                                    "northstartech"
                                ]
                                for mention in northstar_mentions:
                                    if mention in html_lower:
                                        competitor = "Northstar"
                                        logger.debug(f"Found Northstar on {url}")
                                        break
                            
                            if competitor != "Unknown":
                                break
                    except Exception as e:
                        logger.debug(f"Failed to fetch {url}: {str(e)}")
                        continue
            
            return {
                'name': company_data.get('properties', {}).get('name', ''),
                'company_short_name': company_data.get('properties', {}).get('company_short_name', ''),
                'city': company_data.get('properties', {}).get('city', ''),
                'state': company_data.get('properties', {}).get('state', ''),
                'annual_revenue': company_data.get('properties', {}).get('annualrevenue', ''),
                'create_date': company_data.get('properties', {}).get('createdate', ''),
                'last_modified': company_data.get('properties', {}).get('hs_lastmodifieddate', ''),
                'object_id': company_data.get('properties', {}).get('hs_object_id', ''),
                'club_type': company_data.get('properties', {}).get('club_type', 'Unknown'),
                'facility_complexity': company_data.get('properties', {}).get('facility_complexity', 'Unknown'),
                'has_pool': company_data.get('properties', {}).get('has_pool', 'No'),
                'has_tennis_courts': company_data.get('properties', {}).get('has_tennis_courts', 'No'),
                'number_of_holes': company_data.get('properties', {}).get('number_of_holes', 0),
                'geographic_seasonality': company_data.get('properties', {}).get('geographic_seasonality', 'Unknown'),
                'public_private_flag': company_data.get('properties', {}).get('public_private_flag', 'Unknown'),
                'club_info': company_data.get('properties', {}).get('club_info', ''),
                'competitor': competitor
            }
        except Exception as e:
            logger.error(f"Error fetching company data: {e}")
            return {}

    def _determine_facility_type(self, company_name: str, location: str) -> Dict[str, Any]:
        """Uses xAI to determine facility type and details."""
        if not company_name or not location:
            return {}

        segmentation_info = xai_club_segmentation_search(company_name, location)
        club_summary = get_club_summary(company_name, location)

        official_name = (
            segmentation_info.get("name") or 
            company_name
        )

        club_type = "Country Club" if "country club" in official_name.lower() else segmentation_info.get("club_type", "Unknown")

        return {
            "name": official_name,
            "company_short_name": segmentation_info.get("company_short_name", ""),
            "club_type": club_type,
            "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
            "geographic_seasonality": segmentation_info.get("geographic_seasonality", "Unknown"),
            "has_pool": segmentation_info.get("has_pool", "Unknown"),
            "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
            "number_of_holes": segmentation_info.get("number_of_holes", 0),
            "club_info": club_summary,
            "competitor": segmentation_info.get("competitor", "Unknown")
        }

    def _determine_seasonality(self, state: str) -> Dict[str, Any]:
        """Determines golf seasonality based on state."""
        season_data = {
            # Year-round states
            "AZ": {"start": 1, "end": 12, "peak_start": 10, "peak_end": 5, "type": "Year-Round Golf"},
            "FL": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "HI": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "CA": {"start": 1, "end": 12, "peak_start": 1, "peak_end": 12, "type": "Year-Round Golf"},
            "TX": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "GA": {"start": 1, "end": 12, "peak_start": 4, "peak_end": 10, "type": "Year-Round Golf"},
            "NV": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "AL": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "MS": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            "LA": {"start": 1, "end": 12, "peak_start": 3, "peak_end": 11, "type": "Year-Round Golf"},
            
            # Standard season states (Apr-Oct)
            "NC": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "SC": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "VA": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "TN": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "KY": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "MO": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "KS": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "OK": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "AR": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            "NM": {"start": 3, "end": 11, "peak_start": 4, "peak_end": 10, "type": "Standard Season"},
            
            # Short season states (May-Sept/Oct)
            "MI": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "WI": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "MN": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "ND": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "SD": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "MT": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "ID": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "WY": {"start": 5, "end": 9, "peak_start": 6, "peak_end": 8, "type": "Short Summer Season"},
            "ME": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "VT": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            "NH": {"start": 5, "end": 10, "peak_start": 6, "peak_end": 9, "type": "Short Summer Season"},
            
            # Default season (Apr-Oct)
            "default": {"start": 4, "end": 10, "peak_start": 5, "peak_end": 9, "type": "Standard Season"}
        }
        
        state_data = season_data.get(state, season_data["default"])
        
        return {
            "geographic_seasonality": state_data["type"],
            "start_month": state_data["start"],
            "end_month": state_data["end"],
            "peak_season_start_month": state_data["peak_start"],
            "peak_season_end_month": state_data["peak_end"]
        }

    def _prepare_updates(self, current_info: Dict[str, Any], new_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the final update payload with validated values."""
        try:
            # Get competitor value with explicit logging
            competitor = new_info.get('competitor', current_info.get('competitor', 'Unknown'))
            logger.debug(f"Processing competitor value in _prepare_updates: {competitor}")

            # Initialize updates with default values for required fields
            updates = {
                "name": new_info.get("name", current_info.get("name", "")),
                "company_short_name": new_info.get("company_short_name", ""),  # Don't fall back to current_info yet
                "club_type": new_info.get("club_type", current_info.get("club_type", "Unknown")),
                "facility_complexity": new_info.get("facility_complexity", current_info.get("facility_complexity", "Unknown")),
                "geographic_seasonality": new_info.get("geographic_seasonality", current_info.get("geographic_seasonality", "Unknown")),
                "has_pool": "No",  # Default value
                "has_tennis_courts": new_info.get("has_tennis_courts", current_info.get("has_tennis_courts", "No")),
                "number_of_holes": new_info.get("number_of_holes", current_info.get("number_of_holes", 0)),
                "public_private_flag": new_info.get("public_private_flag", current_info.get("public_private_flag", "Unknown")),
                "start_month": new_info.get("start_month", current_info.get("start_month", "")),
                "end_month": new_info.get("end_month", current_info.get("end_month", "")),
                "peak_season_start_month": new_info.get("peak_season_start_month", current_info.get("peak_season_start_month", "")),
                "peak_season_end_month": new_info.get("peak_season_end_month", current_info.get("peak_season_end_month", "")),
                "competitor": competitor,
                "club_info": new_info.get("club_info", current_info.get("club_info", ""))
            }

            # Handle company_short_name with proper fallback logic
            if not updates["company_short_name"]:
                # Try current_info short name first
                updates["company_short_name"] = current_info.get("company_short_name", "")
                
                # If still empty, use full name
                if not updates["company_short_name"]:
                    updates["company_short_name"] = updates["name"]
                    logger.debug(f"Using full name as company_short_name: {updates['company_short_name']}")

            # Ensure company_short_name is not truncated inappropriately
            if updates["company_short_name"]:
                updates["company_short_name"] = str(updates["company_short_name"])[:100]
                logger.debug(f"Final company_short_name: {updates['company_short_name']}")

            # Handle pool information
            club_info = new_info.get("club_info", "").lower()
            if "pool" in club_info:
                updates["has_pool"] = "Yes"
            else:
                updates["has_pool"] = current_info.get("has_pool", "No")

            # Convert numeric fields to integers
            for key in ["number_of_holes", "start_month", "end_month", "peak_season_start_month", "peak_season_end_month"]:
                if updates.get(key):
                    try:
                        updates[key] = int(updates[key])
                    except (ValueError, TypeError):
                        updates[key] = 0

            # Convert boolean fields to Yes/No
            for key in ["has_tennis_courts", "has_pool"]:
                updates[key] = "Yes" if str(updates.get(key, "")).lower() in ["yes", "true", "1"] else "No"

            # Validate competitor value
            valid_competitors = ["Club Essentials", "Jonas", "Unknown"]
            if competitor in valid_competitors:
                updates["competitor"] = competitor
                logger.debug(f"Set competitor to valid value: {competitor}")
            else:
                logger.debug(f"Invalid competitor value ({competitor}), defaulting to Unknown")
                updates["competitor"] = "Unknown"

            # Map values to HubSpot-accepted values
            property_value_mapping = {
                "club_type": {
                    "Private": "Private",
                    "Public": "Public",
                    "Public - Low Daily Fee": "Public - Low Daily Fee",
                    "Public - High Daily Fee": "Public - High Daily Fee",
                    "Municipal": "Municipal",
                    "Semi-Private": "Semi-Private",
                    "Resort": "Resort",
                    "Country Club": "Country Club",
                    "Private Country Club": "Country Club",
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
                    "Year-Round Golf": "Year-Round",
                    "Peak Summer Season": "Peak Summer Season",
                    "Short Summer Season": "Short Summer Season",
                    "Unknown": "Unknown"
                }
            }

            # Apply mappings
            for key, mapping in property_value_mapping.items():
                if key in updates:
                    updates[key] = mapping.get(str(updates[key]), updates[key])

            logger.debug(f"Final prepared updates: {updates}")
            return updates

        except Exception as e:
            logger.error(f"Error preparing updates: {str(e)}")
            logger.debug(f"Current info: {current_info}")
            logger.debug(f"New info: {new_info}")
            return {}

    def _update_company_properties(self, company_id: str, updates: Dict[str, Any]) -> bool:
        """Updates the company properties in HubSpot."""
        try:
            property_value_mapping = {
                "club_type": {
                    "Private": "Private",
                    "Public": "Public",
                    "Public - Low Daily Fee": "Public - Low Daily Fee",
                    "Public - High Daily Fee": "Public - High Daily Fee",
                    "Municipal": "Municipal",
                    "Semi-Private": "Semi-Private",
                    "Resort": "Resort",
                    "Country Club": "Country Club",
                    "Private Country Club": "Country Club",
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
                    "Year-Round Golf": "Year-Round",
                    "Peak Summer Season": "Peak Summer Season",
                    "Short Summer Season": "Short Summer Season",
                    "Unknown": "Unknown"
                },
                "competitor": {
                    "Jonas": "Jonas",
                    "Club Essentials": "Club Essentials",
                    "Unknown": "Unknown"
                }
            }

            mapped_updates = {}
            for key, value in updates.items():
                if key in property_value_mapping:
                    value = property_value_mapping[key].get(str(value), value)
                    
                if key in ["number_of_holes", "start_month", "end_month", 
                          "peak_season_start_month", "peak_season_end_month"]:
                    value = int(value) if str(value).isdigit() else 0
                elif key in ["has_pool", "has_tennis_courts"]:
                    value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                elif key == "club_info":
                    value = str(value)[:5000]
                elif key == "company_short_name":
                    value = str(value)[:100]

                mapped_updates[key] = value

            url = f"{self.hubspot.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            try:
                response = self.hubspot._make_hubspot_patch(url, payload)
                if response:
                    return True
                return False
            except HubSpotError as api_error:
                logger.error(f"HubSpot API Error: {str(api_error)}")
                raise
            
        except Exception as e:
            logger.error(f"Error updating company properties: {str(e)}")
            return False 
```

## services\conversation_analysis_service.py
```python
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import pytz
from dateutil.parser import parse as parse_date
import openai

from services.data_gatherer_service import DataGathererService
from services.gmail_service import GmailService
from config.settings import OPENAI_API_KEY


class ConversationAnalysisService:
    def __init__(self):
        self.data_gatherer = DataGathererService()
        self.gmail_service = GmailService()
        openai.api_key = OPENAI_API_KEY

    def analyze_conversation(self, email_address: str) -> str:
        """Main entry point - analyze conversation for an email address."""
        try:
            # Get contact data
            contact_data = self.data_gatherer.hubspot.get_contact_by_email(email_address)
            if not contact_data:
                return "No contact found."

            contact_id = contact_data["id"]
            
            # Gather all messages
            all_messages = self._gather_all_messages(contact_id, email_address)
            
            # Generate summary
            summary = self._generate_ai_summary(all_messages)
            return summary

        except Exception as e:
            return f"Error analyzing conversation: {str(e)}"

    def _gather_all_messages(self, contact_id: str, email_address: str) -> List[Dict[str, Any]]:
        """Gather and combine all messages from different sources."""
        # Get HubSpot emails and notes
        hubspot_emails = self.data_gatherer.hubspot.get_all_emails_for_contact(contact_id)
        hubspot_notes = self.data_gatherer.hubspot.get_all_notes_for_contact(contact_id)
        
        # Get Gmail messages
        gmail_emails = self.gmail_service.get_latest_emails_for_contact(email_address)

        # Process and combine all messages
        all_messages = []
        
        # Add HubSpot emails
        for email in hubspot_emails:
            if email.get("timestamp"):
                timestamp = self._ensure_timezone(parse_date(email["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": email.get("body_text"),
                    "direction": email.get("direction"),
                    "subject": email.get("subject", "No subject"),
                    "source": "HubSpot"
                })

        # Add relevant notes
        for note in hubspot_notes:
            if note.get("timestamp") and "email" in note.get("body", "").lower():
                timestamp = self._ensure_timezone(parse_date(note["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": note.get("body"),
                    "direction": "NOTE",
                    "subject": "Email Note",
                    "source": "HubSpot"
                })

        # Add Gmail messages
        for direction, msg in gmail_emails.items():
            if msg and msg.get("timestamp"):
                timestamp = self._ensure_timezone(parse_date(msg["timestamp"]))
                all_messages.append({
                    "timestamp": timestamp.isoformat(),
                    "body_text": msg["body_text"],
                    "direction": msg["direction"],
                    "subject": msg.get("subject", "No subject"),
                    "source": "Gmail",
                    "gmail_id": msg.get("gmail_id")
                })

        # Sort by timestamp
        sorted_messages = sorted(all_messages, key=lambda x: parse_date(x["timestamp"]))
        return sorted_messages

    def _generate_ai_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate AI summary of the conversation."""
        if not messages:
            return "No conversation found."

        conversation_text = self._prepare_conversation_text(messages)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a sales assistant analyzing email conversations. "
                            "Please provide: "
                            "1) A brief summary of the conversation thread (2-3 sentences max), "
                            "2) The latest INCOMING response only (ignore outbound messages from Ryan Donovan or Ty Hayes), including the date and who it was from, "
                            f"3) Whether we responded to the latest incoming message, and if so, what was our response (include the full email text) and how many days ago was it sent relative to {datetime.now().date()}. "
                            "Keep all responses clear and concise."
                        )
                    },
                    {"role": "user", "content": conversation_text}
                ]
            )
            
            summary_content = response.choices[0].message.content
            return summary_content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _clean_email_body(self, body_text: Optional[str]) -> Optional[str]:
        """Clean up email body text."""
        if body_text is None:
            return None

        # Split on common email markers
        splits = re.split(
            r'(?:\r\n|\r|\n)*(?:From:|On .* wrote:|_{3,}|Get Outlook|Sent from|<http)',
            body_text
        )

        message = splits[0].strip()
        message = re.sub(r'\s+', ' ', message)
        message = re.sub(r'\[cid:[^\]]+\]', '', message)
        message = re.sub(r'<[^>]+>', '', message)

        cleaned_message = message.strip()
        return cleaned_message

    def _prepare_conversation_text(self, messages: List[Dict[str, Any]]) -> str:
        """Prepare conversation text for AI analysis."""
        conversation_text = "Full conversation:\n\n"
        
        for message in messages:
            date = parse_date(message['timestamp']).strftime('%Y-%m-%d %H:%M')
            direction = "OUTBOUND" if message.get('direction') in ['EMAIL', 'NOTE'] else "INBOUND"
            body = self._clean_email_body(message.get('body_text')) or f"[Email with subject: {message.get('subject', 'No subject')}]"
            conversation_text += f"{date} ({direction}): {body}\n\n"

        return conversation_text

    @staticmethod
    def _ensure_timezone(dt):
        """Ensure datetime has timezone information."""
        if dt.tzinfo is None:
            return pytz.UTC.localize(dt)
        return dt
```

## services\data_gatherer_service.py
```python
# services/data_gatherer_service.py

import json
import csv
import datetime
from typing import Dict, Any, Union, List
from pathlib import Path
from dateutil.parser import parse as parse_date

from services.hubspot_service import HubspotService
from utils.exceptions import HubSpotError
from utils.xai_integration import xai_news_search, xai_club_segmentation_search
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
        logger.debug("Initialized DataGathererService")
        
        # Load season data
        self.load_season_data()
        self.load_state_timezones()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data (now mostly delegated to HubspotService)."""
        return self.hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data from HubSpot, then merges competitor & season data.
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

        # 4) Get the company data (including domain)
        company_props = self.hubspot.get_company_data(company_id)

        # Example: competitor check using domain
        competitor_analysis = self.check_competitor_on_website(
            company_props.get("domain", "")
        )
        if competitor_analysis["status"] == "success" and competitor_analysis["competitor"]:
            # Update HubSpot with competitor info
            self.hubspot.update_company_properties(company_id, {
                "competitor": competitor_analysis["competitor"]
            })

        # Example: gather news just once
        club_name = company_props.get("name", "")
        news_result = self.gather_club_news(club_name)

        # Build partial lead_sheet (now without emails and notes)
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
                "company_data": company_props
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

        return lead_sheet
    # -------------------------------------------------------------------------
    # Competitor-check logic
    # -------------------------------------------------------------------------
    def check_competitor_on_website(self, domain: str, correlation_id: str = None) -> Dict[str, str]:
        """Check for competitor software mentions on the website."""
        if correlation_id is None:
            correlation_id = f"competitor_check_{domain}"
        try:
            if not domain:
                return {
                    "competitor": "",
                    "status": "no_data", 
                    "error": "No domain provided"
                }

            # Try different URL variations
            urls_to_try = []
            base_url = domain.strip().lower()
            if not base_url.startswith('http'):
                urls_to_try.extend([f"https://{base_url}", f"http://{base_url}"])
            else:
                urls_to_try.append(base_url)
            
            # Add www. version if not present
            urls_to_try.extend([url.replace('://', '://www.') for url in urls_to_try])
            
            for url in urls_to_try:
                try:
                    html = fetch_website_html(url)
                    if html:
                        html_lower = html.lower()
                        
                        # Check for Club Essentials mentions first
                        clubessential_mentions = [
                            "copyright clubessential",
                            "clubessential, llc",
                            "www.clubessential.com",
                            "http://www.clubessential.com",
                            "clubessential"
                        ]
                        for mention in clubessential_mentions:
                            if mention in html_lower:
                                logger.debug(f"Found Club Essentials on {url}")
                                return {
                                    "competitor": "Club Essentials",
                                    "status": "success",
                                    "error": ""
                                }
                        
                        # Check for Jonas mentions
                        jonas_mentions = ["jonas club software", "jonas software", "jonasclub"]
                        for mention in jonas_mentions:
                            if mention in html_lower:
                                logger.debug(f"Found Jonas on {url}")
                                return {
                                    "competitor": "Jonas",
                                    "status": "success",
                                    "error": ""
                                }
                except Exception as e:
                    logger.debug(f"Failed to fetch {url}: {str(e)}")
                    continue
            
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
        """Get club information using segmentation."""
        if not club_name or not city or not state:
            return ""
        
        location = f"{city}, {state}"
        try:
            # Replace club_info_search with segmentation
            segmentation_data = xai_club_segmentation_search(club_name, location)
            return segmentation_data.get("club_info", "")
        except Exception as e:
            logger.error("Error gathering club info", extra={
                "club_name": club_name,
                "error": str(e)
            })
            return ""

    def gather_club_news(self, club_name: str) -> str:
        correlation_id = f"club_news_{club_name}"
        logger.debug("Starting club news search", extra={
            "club_name": club_name,
            "correlation_id": correlation_id
        })
        try:
            news = xai_news_search(club_name)
            # xai_news_search can return (news, icebreaker), so handle accordingly
            if isinstance(news, tuple):
                news = news[0]
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
            segmentation_info = xai_club_segmentation_search(company_name, location_str)
            return {
                "response": segmentation_info,
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

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Convert a value to int safely, returning default if it fails."""
        if value is None:
            return default
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def load_season_data(self):
        """Load golf season data from CSV files."""
        try:
            # Load city/state data
            if CITY_ST_CSV.exists():
                with open(CITY_ST_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Log the headers to debug
                    headers = reader.fieldnames
                    logger.debug(f"CSV headers: {headers}")
                    
                    for row in reader:
                        try:
                            # Adjust these keys based on your actual CSV headers
                            city = row.get('city', row.get('City', '')).lower()
                            state = row.get('state', row.get('State', '')).lower()
                            if city and state:
                                city_key = (city, state)
                                CITY_ST_DATA[city_key] = row
                        except Exception as row_error:
                            logger.warning(f"Skipping malformed row in city/state data: {row_error}")
                            continue

            # Load state-only data
            if ST_CSV.exists():
                with open(ST_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            state = row.get('state', row.get('State', '')).lower()
                            if state:
                                ST_DATA[state] = row
                        except Exception as row_error:
                            logger.warning(f"Skipping malformed row in state data: {row_error}")
                            continue



        except FileNotFoundError:
            logger.warning("Season data files not found, using defaults", extra={
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            # Continue with empty data, will use defaults
            pass
        except Exception as e:
            logger.error("Failed to load golf season data", extra={
                "error": str(e),
                "city_st_path": str(CITY_ST_CSV),
                "st_path": str(ST_CSV)
            })
            # Continue with empty data, will use defaults
            pass

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        """
        if not city and not state:
            return self._get_default_season_data()

        city_key = (city.lower(), state.lower())
        row = CITY_ST_DATA.get(city_key)
        if not row:
            row = ST_DATA.get(state.lower())

        if not row:
            # For Arizona, override with specific data
            if state.upper() == 'AZ':
                return {
                    "year_round": "Yes",  # Arizona is typically year-round golf
                    "start_month": "1",
                    "end_month": "12",
                    "peak_season_start": "01-01",
                    "peak_season_end": "12-31",
                    "status": "success",
                    "error": ""
                }
            # For Florida, override with specific data
            elif state.upper() == 'FL':
                return {
                    "year_round": "Yes",  # Florida is year-round golf
                    "start_month": "1",
                    "end_month": "12",
                    "peak_season_start": "01-01",
                    "peak_season_end": "12-31",
                    "status": "success",
                    "error": ""
                }
            return self._get_default_season_data()

        return {
            "year_round": "Yes" if row.get("Year-Round?", "").lower() == "yes" else "No",
            "start_month": row.get("Start Month", "1"),
            "end_month": row.get("End Month", "12"),
            "peak_season_start": self._month_to_first_day(row.get("Peak Season Start", "January")),
            "peak_season_end": self._month_to_last_day(row.get("Peak Season End", "December")),
            "status": "success",
            "error": ""
        }

    def _get_default_season_data(self) -> Dict[str, str]:
        """Return default season data."""
        return {
            "year_round": "No",
            "start_month": "3",
            "end_month": "11",
            "peak_season_start": "05-01",
            "peak_season_end": "08-31",
            "status": "default",
            "error": "Location not found, using defaults"
        }

    def _month_to_first_day(self, month_name: str) -> str:
        """
        Convert a month name (January, February, etc.) to a string "MM-01".
        Defaults to "05-01" (May 1) if unknown.
        """
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
        """
        Convert a month name (January, February, etc.) to a string "MM-DD"
        for the last day of that month. Defaults to "08-31" if unknown.
        """
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
        This can be adapted to store lead context in a local JSON file, for example.
        """
        # Implementation detail: e.g.,
        # with open(f"lead_contexts/{lead_email}.json", "w", encoding="utf-8") as f:
        #     json.dump(lead_sheet, f, indent=2)
        pass

    def load_state_timezones(self) -> None:
        """
        Load state timezone offsets from CSV file into STATE_TIMEZONES.
        The file must have columns: state_code, daylight_savings, standard_time
        """
        global STATE_TIMEZONES
        try:
            with open(TIMEZONE_CSV_PATH, 'r', encoding='utf-8') as file:
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
        Return timezone offset data for a given state code.
        Returns a dict: {'dst': int, 'std': int}
        """
        state_code = state.upper() if state else ''
        timezone_data = STATE_TIMEZONES.get(state_code, {
            'dst': -4,  # Default to Eastern (DST)
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
        Get club geography and type based on location + HubSpot data.
        
        Returns:
            (geography: str, club_type: str)
        """
        from utils.exceptions import HubSpotError
        try:
            # Attempt to get company data from HubSpot
            company_data = self.hubspot.get_company_data(club_name)
            geography = self.determine_geography(city, state)
            
            # Determine club type from the HubSpot data
            club_type = company_data.get("type", "Public Clubs")
            if not club_type or club_type.lower() == "unknown":
                club_type = "Public Clubs"
            
            return geography, club_type
            
        except HubSpotError:
            # If we fail to get company from HubSpot, default
            geography = self.determine_geography(city, state)
            return geography, "Public Clubs"

    def determine_geography(self, city: str, state: str) -> str:
        """Return 'City, State' or 'Unknown' if missing."""
        if not city or not state:
            return "Unknown"
        return f"{city}, {state}"


```

## services\gmail_service.py
```python
from typing import List, Dict, Any, Optional
from utils.gmail_integration import (
    get_gmail_service,
    create_message,
    create_draft,
    send_message,
    search_messages,
    check_thread_for_reply,
    search_inbound_messages_for_email,
    get_or_create_label
)
from utils.logging_setup import logger
from datetime import datetime
import pytz

class GmailService:
    def get_latest_emails_for_contact(self, email_address: str) -> Dict[str, Any]:
        """Get the latest emails from and to the contact from Gmail."""
        try:
            # Search for latest inbound message
            inbound_query = f"from:{email_address}"
            inbound_messages = search_messages(query=inbound_query)
            
            # Search for latest outbound message
            outbound_query = f"to:{email_address}"
            outbound_messages = search_messages(query=outbound_query)
            
            service = get_gmail_service()
            latest_emails = {
                "inbound": None,
                "outbound": None
            }
            
            # Get latest inbound email
            if inbound_messages:
                latest_inbound = service.users().messages().get(
                    userId="me",
                    id=inbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_inbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                latest_emails["inbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_inbound, "subject"),
                    "body_text": latest_inbound.get("snippet", ""),
                    "direction": "INCOMING_EMAIL",
                    "gmail_id": latest_inbound["id"]
                }
            
            # Get latest outbound email
            if outbound_messages:
                latest_outbound = service.users().messages().get(
                    userId="me",
                    id=outbound_messages[0]["id"],
                    format="full"
                ).execute()
                
                # Convert timestamp to UTC aware datetime
                timestamp = datetime.fromtimestamp(
                    int(latest_outbound["internalDate"]) / 1000,
                    tz=pytz.UTC
                )
                
                latest_emails["outbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_outbound, "subject"),
                    "body_text": latest_outbound.get("snippet", ""),
                    "direction": "EMAIL",
                    "gmail_id": latest_outbound["id"]
                }
            
            return latest_emails
            
        except Exception as e:
            logger.error(f"Error getting Gmail messages: {str(e)}")
            return {"inbound": None, "outbound": None}

    def create_draft_email(self, to: str, subject: str, body: str, lead_id: str = None, sequence_num: int = None) -> Dict[str, Any]:
        """Create a draft email with the given parameters."""
        return create_draft(
            sender="me",
            to=to,
            subject=subject,
            message_text=body,
            lead_id=lead_id,
            sequence_num=sequence_num
        )

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send an email immediately."""
        return send_message(
            sender="me",
            to=to,
            subject=subject,
            message_text=body
        )

    def check_for_reply(self, thread_id: str) -> bool:
        """Check if there has been a reply in the thread."""
        return check_thread_for_reply(thread_id)

    
    def _get_header(self, message: Dict[str, Any], header_name: str) -> str:
        """Extract header value from Gmail message."""
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header["name"].lower() == header_name.lower():
                return header["value"]
        return ""
```

## services\hubspot_service.py
```python
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

```

## services\leads_service.py
```python
"""
services/leads_service.py

Handles lead-related business logic, including generating lead summaries
(not the full data gathering, which now lives in DataGathererService).
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT, DEBUG_MODE
from services.data_gatherer_service import DataGathererService
from utils.logging_setup import logger
from utils.doc_reader import read_doc


class LeadContextError(Exception):
    """Custom exception for lead context preparation errors."""
    pass


class LeadsService:
    """
    Responsible for higher-level lead operations such as generating
    specialized summaries or reading certain docs for personalization.
    NOTE: The actual data gathering is now centralized in DataGathererService.
    """
    
    def __init__(self, data_gatherer_service: DataGathererService):
        """
        Initialize LeadsService.
        
        Args:
            data_gatherer_service: Service for gathering lead data
        """
        self.data_gatherer = data_gatherer_service

    def prepare_lead_context(self, lead_email: str, lead_sheet: Dict = None, correlation_id: str = None) -> Dict[str, Any]:
        """
        Prepare lead context for personalization (subject/body).
        
        Args:
            lead_email: Email address of the lead
            lead_sheet: Optional pre-gathered lead data
            correlation_id: Optional correlation ID for tracing operations
        """
        if correlation_id is None:
            correlation_id = f"prepare_context_{lead_email}"
            
        logger.debug("Starting lead context preparation", extra={
            "email": lead_email,
            "correlation_id": correlation_id
        })
        
        # Use provided lead_sheet or gather new data if none provided
        if not lead_sheet:
            lead_sheet = self.data_gatherer.gather_lead_data(lead_email, correlation_id=correlation_id)
        
        if not lead_sheet:
            logger.warning("No lead data found", extra={"email": lead_email})
            return {}
        
        # Extract relevant data
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        
        # Get job title for template selection
        props = lead_data.get("properties", {})
        job_title = (props.get("jobtitle", "") or "").strip()
        filename_job_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )

        # Get template content
        template_path = f"templates/{filename_job_title}_initial_outreach.md"
        try:
            template_content = read_doc(template_path)
            if isinstance(template_content, str):
                template_content = {
                    "subject": "Default Subject",
                    "body": template_content
                }
            subject = template_content.get("subject", "Default Subject")
            body = template_content.get("body", template_content.get("content", ""))
        except Exception as e:
            logger.warning(
                "Template read failed, using fallback content",
                extra={
                    "template": template_path,
                    "error": str(e),
                    "correlation_id": correlation_id
                }
            )
            subject = "Fallback Subject"
            body = "Fallback Body..."

        # Rest of your existing code...
        return {
            "metadata": {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "email": lead_email,
                "job_title": job_title
            },
            "lead_data": lead_data,
            "subject": subject,
            "body": body
        }

```

## services\orchestrator_service.py
```python
"""
services/orchestrator_service.py

Previously handled some data gathering steps, now simplified to keep 
only final actions like message personalization. 
For full data enrichment, use DataGathererService instead.
"""

from typing import Dict, Any
from services.leads_service import LeadsService
from services.hubspot_service import HubspotService
from utils.logging_setup import logger
from utils.exceptions import LeadContextError, HubSpotError


class OrchestratorService:
    """Service for orchestrating final messaging steps, excluding data-gathering."""
    
    def __init__(self, leads_service: LeadsService, hubspot_service: HubspotService):
        self.leads_service = leads_service
        self.hubspot_service = hubspot_service
        self.logger = logger

    def personalize_message(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Personalize a message for a lead using the existing LeadsService summary.
        This does not fetch new data; that is the job of DataGathererService.
        """
        metadata = lead_data.get("metadata", {})
        lead_email = metadata.get("email")
        if not lead_email:
            raise LeadContextError("No email found in lead data metadata for personalization.")

        return self.leads_service.prepare_lead_context(lead_email)

```

## utils\__init__.py
```python
# utils/__init__.py

# Empty init file to make utils a proper package

```

## utils\conversation_summary.py
```python
from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger

def get_latest_email_date(emails: List[Dict[str, Any]]) -> Optional[datetime]:
    """
    Find the most recent email date in a conversation thread.
    
    Args:
        emails: List of email dictionaries with timestamp and direction
        
    Returns:
        datetime object of latest email or None if no emails found
    """
    try:
        if not emails:
            return None
            
        # Sort emails by timestamp in descending order
        sorted_emails = sorted(
            [e for e in emails if e.get('timestamp')],
            key=lambda x: parse_date(x['timestamp']),
            reverse=True
        )
        
        if sorted_emails:
            return parse_date(sorted_emails[0]['timestamp'])
        return None
        
    except Exception as e:
        logger.error(f"Error getting latest email date: {str(e)}")
        return None

def summarize_lead_interactions(lead_sheet: Dict) -> str:
    """
    Get a simple summary of when we last contacted the lead.
    
    Args:
        lead_sheet: Dictionary containing lead data and interactions
        
    Returns:
        String summary of last contact date
    """
    try:
        latest_date = get_latest_email_date(lead_sheet.get('emails', []))
        if latest_date:
            return f"Last contact: {latest_date.strftime('%Y-%m-%d')}"
        return "No previous contact found"
        
    except Exception as e:
        logger.error(f"Error summarizing interactions: {str(e)}")
        return "Error getting interaction summary" 
```

## utils\date_utils.py
```python
from datetime import datetime, timedelta

def convert_to_club_timezone(dt, state_offsets):
    """
    Adjusts datetime based on state's hour offset.
    
    Args:
        dt: datetime object to adjust
        state_offsets: dict with 'dst' and 'std' hour offsets
    """
    if not state_offsets:
        return dt
        
    # Determine if we're in DST (simple check - could be enhanced)
    is_dst = datetime.now().month in [3,4,5,6,7,8,9,10]
    offset_hours = state_offsets['dst'] if is_dst else state_offsets['std']
    
    # Apply offset
    return dt + timedelta(hours=offset_hours) 
```

## utils\doc_reader.py
```python
# utils/doc_reader.py

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from utils.logging_setup import logger

@dataclass
class DocumentMetadata:
    """Metadata for a document in the system.
    
    Attributes:
        path (Path): Path to the document
        name (str): Name of the document
        category (str): Category/type of the document
        last_modified (float): Timestamp of last modification
    """
    path: Path
    name: str
    category: str
    last_modified: float

class DocReader:
    """Handles reading and managing document content from the docs directory.
    
    This class provides functionality to read, manage, and summarize documents
    from a specified directory structure.
    
    Attributes:
        project_root (Path): Root directory of the project
        docs_dir (Path): Directory containing the documents
        supported_extensions (List[str]): List of supported file extensions
    """
    
    def __init__(self, docs_dir: Optional[str] = None) -> None:
        """
        Initialize the DocReader with a docs directory.
        
        Args:
            docs_dir: Optional path to the docs directory. If not provided,
                      defaults to 'docs' in the project root.
        """
        self.project_root: Path = Path(__file__).parent.parent
        self.docs_dir: Path = Path(docs_dir) if docs_dir else self.project_root / 'docs'
        self.supported_extensions: List[str] = ['.txt', '.md']
    
    def get_doc_path(self, doc_name: str) -> Optional[Path]:
        """
        Find the document path for the given document name.
        
        Args:
            doc_name: Name of the document to find (with or without extension)
            
        Returns:
            Path object if document exists, None otherwise
        """
        # If doc_name already has an extension, check directly
        if any(doc_name.endswith(ext) for ext in self.supported_extensions):
            full_path = self.docs_dir / doc_name
            return full_path if full_path.exists() else None
        
        # Otherwise, try each supported extension in turn
        for ext in self.supported_extensions:
            full_path = self.docs_dir / f"{doc_name}{ext}"
            if full_path.exists():
                return full_path
        
        return None
    
    def read_file(self, file_path: Path) -> Optional[str]:
        """
        Read content from a file with error handling.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            String content of the file if successful, None if error occurs
        """
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def read_doc(self, doc_name: str, fallback_content: str = "") -> str:
        """
        Read document content with fallback strategy.
        """
        logger.debug(f"Attempting to read document: {doc_name}")
        
        # Handle numbered template variations
        if 'initial_outreach' in doc_name:
            base_name = doc_name.replace('.md', '')
            for i in range(1, 4):
                variation_name = f"{base_name}_{i}.md"
                doc_path = self.get_doc_path(variation_name)
                
                if doc_path:
                    logger.debug(f"Found template variation: {variation_name}")
                    content = self.read_file(doc_path)
                    if content is not None:
                        return content
            
            logger.warning(f"No variations found for {doc_name}, trying fallback")
        
        # Try direct path as fallback
        doc_path = self.get_doc_path(doc_name)
        if doc_path:
            content = self.read_file(doc_path)
            if content is not None:
                logger.info(f"Successfully read document: {doc_path}")
                return content
        
        logger.warning(f"Could not read document '{doc_name}'; using fallback content.")
        return fallback_content
    
    def get_all_docs(self, directory: Optional[str] = None) -> Dict[str, str]:
        """
        Get all documents in a directory.
        
        Args:
            directory: Optional subdirectory to search in
            
        Returns:
            Dictionary mapping relative file paths to their content
        """
        docs: Dict[str, str] = {}
        search_dir = self.docs_dir / (directory or "")
        
        if not search_dir.exists():
            logger.warning(f"Directory does not exist: {search_dir}")
            return docs
        
        for file_path in search_dir.rglob("*"):
            if file_path.suffix in self.supported_extensions:
                relative_path = file_path.relative_to(self.docs_dir)
                content = self.read_file(file_path)
                if content is not None:
                    docs[str(relative_path)] = content
        
        return docs
    
    def summarize_domain_documents(self, docs_dict: Dict[str, str]) -> str:
        """
        Create a summary of multiple domain documents.
        
        Args:
            docs_dict: Dictionary mapping document names to their content
            
        Returns:
            String containing summaries of all documents
        """
        summaries: List[str] = []
        for doc_name, content in docs_dict.items():
            summary = f"Document: {doc_name}\n"
            preview = content[:200] + "..." if len(content) > 200 else content
            summary += f"Preview: {preview}\n"
            summaries.append(summary)
        
        return "\n".join(summaries)

#
#  Top-Level Convenience Function:
#  --------------------------------
#  This allows you to do:
#    from utils.doc_reader import read_doc
#  in your code, which calls the DocReader internally.
#
def read_doc(doc_name: str, fallback_content: str = "") -> str:
    """A convenience function so we can do `from utils.doc_reader import read_doc`."""
    return DocReader().read_doc(doc_name, fallback_content)

def summarize_domain_documents(docs_dict: Dict[str, str]) -> str:
    """
    Standalone function to summarize domain documents.
    """
    reader = DocReader()
    return reader.summarize_domain_documents(docs_dict)

def verify_docs_setup():
    """
    Verify the docs setup and print status.
    """
    doc_reader = DocReader()
    
    # Check docs directory
    if not doc_reader.docs_dir.exists():
        print(f"❌ Docs directory not found: {doc_reader.docs_dir}")
        return False
    
    # Check for required documents
    required_docs = {
        'brand/guidelines.txt': 'Brand guidelines document',
        'case_studies/success_story.txt': 'Case study document',
        'templates/email_template.txt': 'Email template document'
    }
    
    issues = []
    for doc_name, description in required_docs.items():
        if doc_reader.get_doc_path(doc_name):
            print(f"✅ Found {description}")
        else:
            issues.append(f"❌ Missing {description}: {doc_name}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
        return False
    
    print("\nAll required documents are present! ✅")
    return True

if __name__ == "__main__":
    verify_docs_setup()

```

## utils\enrich_hubspot_company_data.py
```python
 # tests/test_hubspot_company_type.py

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window

# Add these constants after imports
###########################
# CONFIG / CONSTANTS
###########################
TEST_MODE = False  # Set to False for production
TEST_LIMIT = 3    # Number of companies to process in test mode
BATCH_SIZE = 25   # Companies per API request
TEST_COMPANY_ID = "15537469970"  # Set this to a specific company ID to test just that company


def get_facility_info(company_id: str) -> tuple[
    str, str, str, str, str, str, str, str, str, str, str, int, str, str, str, str
]:
    """
    Fetches the company's properties from HubSpot.
    Returns a tuple of company properties including club_info and company_short_name.
    """
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        company_data = hubspot.get_company_data(company_id)

        name = company_data.get("name", "")
        company_short_name = company_data.get("company_short_name", "")
        city = company_data.get("city", "")
        state = company_data.get("state", "")
        annual_revenue = company_data.get("annualrevenue", "")
        create_date = company_data.get("createdate", "")
        last_modified = company_data.get("hs_lastmodifieddate", "")
        object_id = company_data.get("hs_object_id", "")
        club_type = company_data.get("club_type", "Unknown")
        facility_complexity = company_data.get("facility_complexity", "Unknown")
        has_pool = company_data.get("has_pool", "No")
        has_tennis_courts = company_data.get("has_tennis_courts", "No")
        number_of_holes = company_data.get("number_of_holes", 0)
        geographic_seasonality = company_data.get("geographic_seasonality", "Unknown")
        public_private_flag = company_data.get("public_private_flag", "Unknown")
        club_info = company_data.get("club_info", "")

        return (
            name,
            company_short_name,
            city,
            state,
            annual_revenue,
            create_date,
            last_modified,
            object_id,
            club_type,
            facility_complexity,
            has_pool,
            has_tennis_courts,
            number_of_holes,
            geographic_seasonality,
            public_private_flag,
            club_info,
        )

    except HubSpotError as e:
        print(f"Error fetching company data: {e}")
        return ("", "", "", "", "", "", "", "", "", "No", "No", 0, "", "", "", "")


def determine_facility_type(company_name: str, location: str) -> dict:
    """
    Uses xAI to determine the facility type and official name based on company info.
    """
    if not company_name or not location:
        return {}

    # Get segmentation data
    segmentation_info = xai_club_segmentation_search(company_name, location)
    
    # Get summary for additional context
    club_summary = get_club_summary(company_name, location)

    # Extract name and generate short name from segmentation info
    official_name = segmentation_info.get("name") or company_name
    # Generate short name by removing common words and limiting length
    short_name = official_name.replace("Country Club", "").replace("Golf Club", "").strip()
    short_name = short_name[:100]  # Ensure it fits HubSpot field limit

    full_info = {
        "name": official_name,
        "company_short_name": short_name,
        "club_type": segmentation_info.get("club_type", "Unknown"),
        "facility_complexity": segmentation_info.get("facility_complexity", "Unknown"),
        "geographic_seasonality": segmentation_info.get("geographic_seasonality", "Unknown"),
        "has_pool": segmentation_info.get("has_pool", "Unknown"),
        "has_tennis_courts": segmentation_info.get("has_tennis_courts", "Unknown"),
        "number_of_holes": segmentation_info.get("number_of_holes", 0),
        "club_info": club_summary
    }

    return full_info


def update_company_properties(company_id: str, club_info: dict, confirmed_updates: dict) -> bool:
    """
    Updates the company's properties in HubSpot based on club segmentation info.
    """
    try:
        # Check club_info for pool mentions before processing
        club_info_text = str(confirmed_updates.get('club_info', '')).lower()
        if 'pool' in club_info_text and confirmed_updates.get('has_pool') in ['Unknown', 'No']:
            logger.debug(f"Found pool mention in club_info, updating has_pool to Yes")
            confirmed_updates['has_pool'] = 'Yes'

        # Debug input values
        logger.debug("Input values for update:")
        for key, value in confirmed_updates.items():
            logger.debug(f"Field: {key}, Value: {value}, Type: {type(value)}")

        # Map our internal property names to HubSpot property names
        hubspot_property_mapping = {
            "name": "name",
            "club_type": "club_type",
            "facility_complexity": "facility_complexity",
            "geographic_seasonality": "geographic_seasonality",
            "public_private_flag": "public_private_flag",
            "has_pool": "has_pool",
            "has_tennis_courts": "has_tennis_courts",
            "number_of_holes": "number_of_holes",
            "club_info": "club_info",
            "season_start": "start_month",
            "season_end": "end_month",
            "peak_season_start_month": "peak_season_start_month",
            "peak_season_end_month": "peak_season_end_month",
            "notes_last_contacted": "notes_last_contacted",
            "num_contacted_notes": "num_contacted_notes",
            "num_associated_contacts": "num_associated_contacts"
        }

        # Value transformations for HubSpot - EXACT matches for HubSpot enum values
        property_value_mapping = {
            "club_type": {
                "Private": "Private",
                "Public": "Public",
                "Public - Low Daily Fee": "Public - Low Daily Fee",
                "Municipal": "Municipal",
                "Semi-Private": "Semi-Private",
                "Resort": "Resort",
                "Country Club": "Country Club",
                "Private Country Club": "Country Club",
                "Management Company": "Management Company",
                "Unknown": "Unknown"
            },
            "facility_complexity": {
                "Single-Course": "Standard",  # Changed from Basic to Standard
                "Multi-Course": "Multi-Course",
                "Resort": "Resort",
                "Unknown": "Unknown"
            },
            "geographic_seasonality": {
                "Year-Round Golf": "Year-Round",
                "Peak Summer Season": "Peak Summer Season",
                "Short Summer Season": "Short Summer Season",
                "Unknown": "Unknown"  # Default value
            }
        }

        # Clean and map the updates
        mapped_updates = {}
        for internal_key, value in confirmed_updates.items():
            hubspot_key = hubspot_property_mapping.get(internal_key)
            if not hubspot_key:
                logger.warning(f"No HubSpot mapping for property: {internal_key}")
                continue

            # Debug pre-transformation
            logger.debug(f"Pre-transform - Key: {internal_key}, Value: {value}, Type: {type(value)}")

            try:
                # Apply enum value transformations first
                if internal_key in property_value_mapping:
                    original_value = value
                    value = property_value_mapping[internal_key].get(str(value), value)
                    logger.debug(f"Enum transformation for {internal_key}: {original_value} -> {value}")

                # Type-specific handling
                if internal_key in ["number_of_holes", "season_start", "season_end", "peak_season_start_month", "peak_season_end_month",
                                   "notes_last_contacted", "num_contacted_notes", "num_associated_contacts"]:
                    original_value = value
                    value = int(value) if str(value).isdigit() else 0
                    logger.debug(f"Number conversion for {internal_key}: {original_value} -> {value}")
                
                elif internal_key in ["has_pool", "has_tennis_courts"]:
                    original_value = value
                    value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                    logger.debug(f"Boolean conversion for {internal_key}: {original_value} -> {value}")
                
                elif internal_key == "club_info":
                    original_length = len(str(value))
                    value = str(value)[:5000]
                    logger.debug(f"Text truncation for {internal_key}: {original_length} chars -> {len(value)} chars")

            except Exception as e:
                logger.error(f"Error transforming {internal_key}: {str(e)}")
                continue

            # Debug post-transformation
            logger.debug(f"Post-transform - Key: {hubspot_key}, Value: {value}, Type: {type(value)}")
            
            mapped_updates[hubspot_key] = value

        # Debug final payload
        logger.debug("Final HubSpot payload:")
        logger.debug(f"Company ID: {company_id}")
        logger.debug("Properties:")
        for key, value in mapped_updates.items():
            logger.debug(f"  {key}: {value} (Type: {type(value)})")

        # Send update to HubSpot with detailed error response
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        url = f"{hubspot.companies_endpoint}/{company_id}"
        payload = {"properties": mapped_updates}

        logger.info(f"Sending update to HubSpot: {payload}")
        try:
            response = hubspot._make_hubspot_patch(url, payload)
            if response:
                logger.info(f"Successfully updated company {company_id}")
                return True
            return False
        except HubSpotError as api_error:
            # Log the detailed error response from HubSpot
            logger.error(f"HubSpot API Error Details:")
            logger.error(f"Status Code: {getattr(api_error, 'status_code', 'Unknown')}")
            logger.error(f"Response Body: {getattr(api_error, 'response_body', 'Unknown')}")
            logger.error(f"Request Body: {payload}")
            raise

    except HubSpotError as e:
        logger.error(f"Error updating company properties: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating company properties: {str(e)}")
        logger.exception("Full traceback:")
        return False


def determine_seasonality(state: str) -> dict:
    """Determine golf seasonality based on state."""
    seasonality_map = {
        # Year-Round Golf States
        "FL": "Year-Round Golf",
        "AZ": "Year-Round Golf",
        "HI": "Year-Round Golf",
        "CA": "Year-Round Golf",
        
        # Short Summer Season States
        "MN": "Short Summer Season",
        "WI": "Short Summer Season",
        "MI": "Short Summer Season",
        "ME": "Short Summer Season",
        "VT": "Short Summer Season",
        "NH": "Short Summer Season",
        "MT": "Short Summer Season",
        "ND": "Short Summer Season",
        "SD": "Short Summer Season",
        
        # Peak Summer Season States (default)
        "default": "Peak Summer Season"
    }
    
    geography = seasonality_map.get(state, seasonality_map["default"])
    
    # Calculate season months
    outreach_window = get_best_outreach_window(
        persona="General Manager",
        geography=geography,
        club_type="Country Club"
    )
    
    best_months = outreach_window["Best Month"]
    return {
        "geographic_seasonality": geography,
        "start_month": min(best_months) if best_months else "",
        "end_month": max(best_months) if best_months else "",
        "peak_season_start_month": min(best_months) if best_months else "",
        "peak_season_end_month": max(best_months) if best_months else ""
    }


def process_company(company_id: str):
    print(f"\n=== Processing Company ID: {company_id} ===")

    (
        name,
        company_short_name,
        city,
        state,
        annual_revenue,
        create_date,
        last_modified,
        object_id,
        club_type,
        facility_complexity,
        has_pool,
        has_tennis_courts,
        number_of_holes,
        geographic_seasonality,
        public_private_flag,
        club_info,
    ) = get_facility_info(company_id)

    print(f"\nProcessing {name} in {city}, {state}")

    if name and state:
        club_info = determine_facility_type(name, state)
        confirmed_updates = {}

        # Update company name if different
        new_name = club_info.get("name")
        if new_name and new_name != name:
            print(f"Updating company name from '{name}' to '{new_name}'")  # Debug print
            confirmed_updates["name"] = new_name

        # Update with explicit string comparisons for boolean fields
        confirmed_updates.update({
            "name": club_info.get("name", name),
            "company_short_name": club_info.get("company_short_name", company_short_name),
            "club_type": club_info.get("club_type", "Unknown"),
            "facility_complexity": club_info.get("facility_complexity", "Unknown"),
            "geographic_seasonality": club_info.get("geographic_seasonality", "Unknown"),
            "has_pool": "Yes" if club_info.get("has_pool") == "Yes" else "No",
            "has_tennis_courts": "Yes" if club_info.get("has_tennis_courts") == "Yes" else "No",
            "number_of_holes": club_info.get("number_of_holes", 0),
            "public_private_flag": public_private_flag
        })

        new_club_info = club_info.get("club_info")
        if new_club_info:
            confirmed_updates["club_info"] = new_club_info

        # Get seasonality data
        season_data = determine_seasonality(state)  # Pass state code
        
        # Add seasonality to confirmed updates
        confirmed_updates.update({
            "geographic_seasonality": season_data["geographic_seasonality"],
            "season_start": season_data["start_month"],
            "season_end": season_data["end_month"],
            "peak_season_start_month": season_data["peak_season_start_month"],
            "peak_season_end_month": season_data["peak_season_end_month"]
        })

        success = update_company_properties(company_id, club_info, confirmed_updates)
        if success:
            print("✓ Successfully updated HubSpot properties")
        else:
            print("✗ Failed to update HubSpot properties")
    else:
        print("Unable to determine facility info - missing company name or location")


def _search_companies_with_filters(hubspot: HubspotService, batch_size=25) -> List[Dict[str, Any]]:
    """
    Search for companies in HubSpot that need club type enrichment.
    Processes one state at a time to avoid filter conflicts.
    """
    states = [
         # "AZ",  # Year-Round Golf
         # "GA",  # Year-Round Golf
        "FL",  # Year-Round Golf
        "MN",  # Short Summer Season
        "WI",  # Short Summer Season
        "MI",  # Short Summer Season
        "ME",  # Short Summer Season
        "VT",  # Short Summer Season
        # "NH",  # Short Summer Season
        # "MT",  # Short Summer Season
        # "ND",  # Short Summer Season
        # "SD"   # Short Summer Season
    ]
    all_results = []
    
    for state in states:
        logger.info(f"Searching for companies in {state}")
        url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
        after = None
        
        while True and (not TEST_MODE or len(all_results) < TEST_LIMIT):
            # Build request payload with single state filter
            payload = {
                "limit": min(batch_size, TEST_LIMIT) if TEST_MODE else batch_size,
                "properties": [
                    "name",
                    "company_short_name",
                    "city",
                    "state",
                    "club_type",
                    "annualrevenue",
                    "facility_complexity",
                    "geographic_seasonality",
                    "has_pool",
                    "has_tennis_courts",
                    "number_of_holes",
                    "public_private_flag",
                    "start_month",
                    "end_month",
                    "peak_season_start_month",
                    "peak_season_end_month",
                    "notes_last_contacted",
                    "num_contacted_notes",
                    "num_associated_contacts"
                ],
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "state",
                                "operator": "EQ",
                                "value": state
                            },
                            {
                                "propertyName": "club_type",
                                "operator": "NOT_HAS_PROPERTY",
                                "value": None
                            },
                            {
                                "propertyName": "annualrevenue",
                                "operator": "GTE", 
                                "value": "10000000"
                            }
                        ]
                    }
                ]
            }
            
            if after:
                payload["after"] = after

            try:
                logger.info(f"Fetching companies in {state} (Test Mode: {TEST_MODE})")
                response = hubspot._make_hubspot_post(url, payload)
                if not response:
                    break

                results = response.get("results", [])
                
                # Double-check state filter
                results = [
                    r for r in results 
                    if r.get("properties", {}).get("state") == state
                ]
                
                all_results.extend(results)
                
                logger.info(f"Retrieved {len(all_results)} total companies so far ({len(results)} from {state})")

                # Handle pagination
                paging = response.get("paging", {})
                next_link = paging.get("next", {}).get("after")
                if not next_link:
                    break
                after = next_link

                # Check if we've hit the test limit
                if TEST_MODE and len(all_results) >= TEST_LIMIT:
                    logger.info(f"Test mode: Reached limit of {TEST_LIMIT} companies")
                    break

            except Exception as e:
                logger.error(f"Error fetching companies from HubSpot for {state}: {str(e)}")
                break

        logger.info(f"Completed search for {state} - Found {len(all_results)} total companies")

    # Ensure we don't exceed test limit
    if TEST_MODE:
        all_results = all_results[:TEST_LIMIT]
        logger.info(f"Test mode: Returning {len(all_results)} companies total")

    return all_results


def main():
    """Main function to process companies needing enrichment."""
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        
        # Check if we're processing a single test company
        if TEST_COMPANY_ID:
            print(f"\n=== Processing Single Test Company: {TEST_COMPANY_ID} ===\n")
            process_company(TEST_COMPANY_ID)
            print("\n=== Completed processing test company ===")
            return
            
        # Regular batch processing
        companies = _search_companies_with_filters(hubspot)
        
        if not companies:
            print("No companies found needing enrichment")
            return
            
        print(f"\n=== Processing {len(companies)} companies ===\n")
        
        for i, company in enumerate(companies, 1):
            company_id = company.get("id")
            if not company_id:
                continue
                
            print(f"\nProcessing company {i} of {len(companies)}")
            process_company(company_id)
        
        print("\n=== Completed processing all companies ===")

    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
```

## utils\exceptions.py
```python
"""
Custom exceptions for the Swoop Golf application.
"""

class SwoopError(Exception):
    """Base exception class for Swoop Golf application."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class LeadContextError(SwoopError):
    """Raised when there's an error preparing lead context."""
    pass

class HubSpotError(SwoopError):
    """Raised when there's an error interacting with HubSpot API."""
    pass

class EmailGenerationError(SwoopError):
    """Raised when there's an error generating email content."""
    pass

class ConfigurationError(SwoopError):
    """Raised when there's a configuration-related error."""
    pass

class ExternalAPIError(SwoopError):
    """Raised when there's an error with external API calls."""
    pass

class OpenAIError(SwoopError):
    """Raised when there's an error with OpenAI API calls."""
    pass

```

## utils\export_codebase.py
```python
import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    # Normalize path separators
    filepath = filepath.replace('\\', '/')
    
    # Exclude patterns
    exclude_patterns = [
        '*/__pycache__/*',
        '*/test_*',
        '*/.git/*',
        '*.pyc',
        '*.log',
        '*.env*',
        '*.yml',
        '*.txt',
        '*.json',
        'config/*',
    ]
    
    # Check if file matches any exclude pattern
    for pattern in exclude_patterns:
        if glob.fnmatch.fnmatch(filepath, pattern):
            return False
    
    # Include only Python files from core functionality
    include_dirs = [
        'agents',
        'services',
        'external',
        'utils',
        'scripts',
        'hubspot_integration',
        'scheduling'
    ]
    
    # More flexible directory matching
    for dir_name in include_dirs:
        if f'/{dir_name}/' in filepath or filepath.startswith(f'{dir_name}/'):
            return True
            
    # Include main.py
    if filepath.endswith('main.py'):
        return True
            
    return False

def get_file_content(filepath):
    """Read and return file content with proper markdown formatting."""
    try:
        # Try UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to cp1252 if UTF-8 fails
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If both fail, try with errors='ignore'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(__file__)))
    
    return f"""
## {rel_path}
```python
{content}
```
"""

def export_codebase(root_dir, output_file):
    """Export core codebase to a markdown file."""
    # Get all Python files
    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if should_include_file(filepath):
                    all_files.append(filepath)
    
    # Sort files for consistent output
    all_files.sort()
    
    # Generate markdown content
    content = """# Sales Assistant Codebase

This document contains the core functionality of the Sales Assistant project.

## Table of Contents
"""
    
    # Add TOC
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, start=root_dir)
        content += f"- [{rel_path}](#{rel_path.replace('/', '-')})\n"
    
    # Add file contents
    for filepath in all_files:
        content += get_file_content(filepath)
    
    # Write output file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'codebase_export.md')
    
    export_codebase(project_root, output_path)
    print(f"Codebase exported to: {output_path}")

```

## utils\export_codebase_primary_files.py
```python
import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    # Normalize path separators
    filepath = filepath.replace('\\', '/')
    # List of primary files to include based on most frequently referenced
    primary_files = [
        'main.py',
        'scripts/golf_outreach_strategy.py',
        'scheduling/database.py',
        'scheduling/extended_lead_storage.py', 
        'scheduling/followup_scheduler.py',
        'scheduling/followup_generation.py',
        'utils/gmail_integration.py',
        'utils/xai_integration.py',
        'scripts/build_template.py'
    ]
    # Check if file is in primary files list
    for primary_file in primary_files:
        if filepath.endswith(primary_file):
            return True
            
    return False

def get_file_content(filepath):
    """Read and return file content with proper markdown formatting."""
    try:
        # Try UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # For leads_list.csv, only include top 10 records
            if filepath.endswith('leads_list.csv'):
                lines = content.splitlines()
                content = '\n'.join(lines[:11]) # Header + 10 records
                
    except UnicodeDecodeError:
        try:
            # Fallback to cp1252 if UTF-8 fails
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If both fail, try with errors='ignore'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(__file__)))
    
    return f"""
## {rel_path}

{content}
"""

def export_files(output_path='exported_codebase.md'):
    """Export all primary files to a markdown file."""
    all_content = []
    
    # Get list of all files in project
    for root, _, files in os.walk(os.path.dirname(os.path.dirname(__file__))):
        for file in files:
            filepath = os.path.join(root, file)
            filepath = filepath.replace('\\', '/')
            
            if should_include_file(filepath):
                content = get_file_content(filepath)
                all_content.append(content)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_content))
    
    return output_path

if __name__ == '__main__':
    output_file = export_files()
    print(f'Codebase exported to: {output_file}')

```

## utils\export_templates.py
```python
import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    filepath = filepath.replace('\\', '/')
    return 'docs/templates/country_club' in filepath

def get_file_content(filepath):
    """Read and return file content with proper markdown formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(__file__)))
    extension = os.path.splitext(filename)[1].lower()
    lang = 'markdown' if extension == '.md' else 'text'
    
    return f"## {rel_path}\n```{lang}\n{content}\n```\n"

def export_codebase(root_dir, output_file):
    """Export country club templates to a markdown file."""
    all_files = []
    country_club_path = os.path.join(root_dir, 'docs', 'templates', 'country_club')
    
    for root, _, files in os.walk(country_club_path):
        for file in files:
            filepath = os.path.join(root, file)
            if should_include_file(filepath):
                all_files.append(filepath)
    
    all_files.sort()
    
    content = "# Country Club Email Templates\n\nThis document contains all country club email templates.\n\n## Table of Contents\n"
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, start=root_dir)
        content += f"- [{rel_path}](#{rel_path.replace('/', '-')})\n"
    
    for filepath in all_files:
        content += get_file_content(filepath)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'country_club_templates.md')
    
    export_codebase(project_root, output_path)
    print(f"Country club templates exported to: {output_path}")

```

## utils\formatting_utils.py
```python
"""
Utility functions for text formatting and cleaning.
"""

import re
from bs4 import BeautifulSoup
from typing import Optional

def clean_phone_number(raw_phone):
    """
    Example phone cleaning logic:
    1) Remove non-digit chars
    2) Format as needed (e.g., ###-###-####)
    """
    if raw_phone is None:
        return None
    
    digits = "".join(char for char in raw_phone if char.isdigit())
    if len(digits) == 10:
        # e.g. (123) 456-7890
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    else:
        return digits

def clean_html(text):
    """Clean HTML from text while handling both markup and file paths."""
    if not text:
        return ""
        
    # If text is a file path, read the file first
    if isinstance(text, str) and ('\n' not in text) and ('.' in text):
        try:
            with open(text, 'r', encoding='utf-8') as f:
                text = f.read()
        except (IOError, OSError):
            # If we can't open it as a file, treat it as markup
            pass
            
    # Parse with BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

def extract_text_from_html(html_content: str, preserve_newlines: bool = False) -> str:
    """
    Extract readable text from HTML content, removing scripts and styling.
    Useful for content analysis and summarization.
    
    Args:
        html_content: HTML content to process
        preserve_newlines: If True, uses newlines as separator, else uses space
    
    Returns:
        Extracted text content
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Get text content
    separator = "\n" if preserve_newlines else " "
    text = soup.get_text(separator=separator, strip=True)
    
    # Remove excessive whitespace while preserving single newlines if needed
    if preserve_newlines:
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        # Replace all whitespace (including newlines) with single space
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

```

## utils\gmail_integration.py
```python
import os
import base64
import os.path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from utils.logging_setup import logger
from datetime import datetime
from scheduling.database import get_db_connection
from typing import Dict, Any
from config import settings
from pathlib import Path
from config.settings import PROJECT_ROOT
from scheduling.extended_lead_storage import store_lead_email_info

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Get Gmail API service."""
    creds = None
    
    # Use absolute paths from PROJECT_ROOT
    credentials_path = Path(PROJECT_ROOT) / 'credentials' / 'credentials.json'
    token_path = Path(PROJECT_ROOT) / 'credentials' / 'token.json'
    
    # Ensure credentials directory exists
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if credentials exist
        if not credentials_path.exists():
            logger.error(f"Missing credentials file at {credentials_path}")
            raise FileNotFoundError(
                f"credentials.json is missing. Please place it in: {credentials_path}"
            )
            
        # The file token.json stores the user's access and refresh tokens
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), 
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
                
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                
        return build('gmail', 'v1', credentials=creds)
        
    except Exception as e:
        logger.error(f"Error setting up Gmail service: {str(e)}")
        raise

def get_gmail_template(service, template_name: str = "sales") -> str:
    """Fetch HTML email template from Gmail drafts."""
    try:
        # First try to list all drafts
        drafts = service.users().drafts().list(userId='me').execute()
        if not drafts.get('drafts'):
            logger.error("No drafts found in Gmail account")
            return ""

        # Log all available drafts for debugging
        draft_subjects = []
        template_html = ""
        
        for draft in drafts.get('drafts', []):
            msg = service.users().messages().get(
                userId='me',
                id=draft['message']['id'],
                format='full'
            ).execute()
            
            # Get subject from headers
            headers = msg['payload']['headers']
            subject = next(
                (h['value'] for h in headers if h['name'].lower() == 'subject'),
                ''
            ).lower()
            draft_subjects.append(subject)
            
            # If this is our template
            if template_name.lower() in subject:
                logger.debug(f"Found template draft with subject: {subject}")
                
                # Extract HTML content
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/html':
                            template_html = base64.urlsafe_b64decode(
                                part['body']['data']
                            ).decode('utf-8')
                            break
                elif msg['payload']['mimeType'] == 'text/html':
                    template_html = base64.urlsafe_b64decode(
                        msg['payload']['body']['data']
                    ).decode('utf-8')
                
                if template_html:
                    return template_html

        if not template_html:
            logger.error(
                f"No template found with name: {template_name}. "
                f"Available draft subjects: {draft_subjects}"
            )
        return template_html

    except Exception as e:
        logger.error(f"Error fetching Gmail template: {str(e)}", exc_info=True)
        return ""

def create_message(to: str, subject: str, body: str) -> Dict[str, str]:
    """Create an HTML-formatted email message using Gmail template."""
    try:
        # Validate inputs
        if not all([to, subject, body]):
            logger.error("Missing required email fields")
            return {}

        # Ensure all inputs are strings
        to = str(to).strip()
        subject = str(subject).strip()
        body = str(body).strip()

        logger.debug("Creating HTML email message")

        # Create the MIME Multipart message
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject
        message["bcc"] = "20057893@bcc.hubspot.com"

        # Format the body text with paragraphs
        div_start = "<div style='margin-bottom: 20px;'>"
        div_end = "</div>"
        formatted_body = div_start + body.replace('\n\n', div_end + div_start) + div_end

        # Get Gmail service and template
        service = get_gmail_service()
        template = get_gmail_template(service, "sales")
        
        if not template:
            logger.error("Failed to get Gmail template, using plain text")
            html_body = formatted_body
        else:
            # Look for common content placeholders in the template
            placeholders = ['{{content}}', '{content}', '[content]', '{{body}}', '{body}', '[body]']
            template_with_content = template
            
            # Try each placeholder until one works
            for placeholder in placeholders:
                if placeholder in template:
                    template_with_content = template.replace(placeholder, formatted_body)
                    logger.debug(f"Found and replaced placeholder: {placeholder}")
                    break
            
            if template_with_content == template:
                # No placeholder found, try to insert before the first signature or calendar section
                signature_markers = ['</signature>', 'calendar-section', 'signature-section']
                inserted = False
                
                for marker in signature_markers:
                    if marker in template.lower():
                        parts = template.lower().split(marker, 1)
                        template_with_content = parts[0] + formatted_body + marker + parts[1]
                        inserted = True
                        logger.debug(f"Inserted content before marker: {marker}")
                        break
                
                if not inserted:
                    # If no markers found, prepend content to template
                    template_with_content = formatted_body + template
                    logger.debug("No markers found, prepended content to template")
            
            html_body = template_with_content

        # Create both plain text and HTML versions
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(html_body, 'html')

        # Add both parts to the message
        message.attach(text_part)  # Fallback plain text version
        message.attach(html_part)  # Primary HTML version

        # Encode the message
        raw_message = message.as_string()
        encoded_message = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("utf-8")
        
        logger.debug(f"Created message with HTML length: {len(html_body)}")
        return {"raw": encoded_message}

    except Exception as e:
        logger.exception(f"Error creating email message: {str(e)}")
        return {}

def get_or_create_label(service, label_name: str = "to_review") -> str:
    """
    Retrieve or create a Gmail label and return its labelId.
    """
    try:
        user_id = 'me'
        # 1) List existing labels and log them
        labels_response = service.users().labels().list(userId=user_id).execute()
        existing_labels = labels_response.get('labels', [])
        
        logger.debug(f"Found {len(existing_labels)} existing labels:")
        for lbl in existing_labels:
            logger.debug(f"  - '{lbl['name']}' (id: {lbl['id']})")

        # 2) Case-insensitive search for existing label
        label_name_lower = label_name.lower()
        for lbl in existing_labels:
            if lbl['name'].lower() == label_name_lower:
                logger.debug(f"Found existing label '{lbl['name']}' with id={lbl['id']}")
                return lbl['id']

        # 3) Create new label if not found
        logger.debug(f"No existing label found for '{label_name}', creating new one...")
        label_body = {
            'name': label_name,  # Use original case for creation
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show',
        }
        created_label = service.users().labels().create(
            userId=user_id, body=label_body
        ).execute()

        logger.debug(f"Created new label '{label_name}' with id={created_label['id']}")
        return created_label["id"]

    except Exception as e:
        logger.error(f"Error in get_or_create_label: {str(e)}", exc_info=True)
        return ""

def create_draft(
    sender: str,
    to: str,
    subject: str,
    message_text: str,
    lead_id: str = None,
    sequence_num: int = None,
) -> Dict[str, Any]:
    """
    Create a Gmail draft email, add the 'to_review' label, and optionally store in DB.
    """
    try:
        logger.debug(
            f"Creating draft with subject='{subject}', body length={len(message_text)}"
        )

        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return {"status": "error", "error": "No Gmail service"}

        # 1) Create the MIME email message
        message = create_message(to=to, subject=subject, body=message_text)
        if not message:
            logger.error("Failed to create message")
            return {"status": "error", "error": "Failed to create message"}

        # 2) Create the actual draft
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": message})
            .execute()
        )

        if "id" not in draft:
            logger.error("Draft creation returned no ID")
            return {"status": "error", "error": "No draft ID returned"}

        draft_id = draft["id"]
        draft_message_id = draft["message"]["id"]
        logger.debug(f"Created draft with id={draft_id}, message_id={draft_message_id}")

        # 3) Store draft info in the DB if lead_id is provided
        if settings.CREATE_FOLLOWUP_DRAFT and lead_id:
            store_draft_info(
                lead_id=lead_id,
                draft_id=draft_id,
                scheduled_date=None,
                subject=subject,
                body=message_text,
                sequence_num=sequence_num,
            )
        else:
            logger.info("Follow-up draft creation is disabled via CREATE_FOLLOWUP_DRAFT setting")

        # 4) Add the "to_review" label to the underlying draft message
        label_id = get_or_create_label(service, "to_review")
        if label_id:
            try:
                service.users().messages().modify(
                    userId="me",
                    id=draft_message_id,
                    body={"addLabelIds": [label_id]},
                ).execute()
                logger.debug(f"Added '{label_id}' label to draft message {draft_message_id}")
            except Exception as e:
                logger.error(f"Failed to add label to draft: {str(e)}")
        else:
            logger.warning("Could not get/create 'to_review' label - draft remains unlabeled")

        return {
            "status": "ok",
            "draft_id": draft_id,
            "sequence_num": sequence_num,
        }

    except Exception as e:
        logger.error(f"Error in create_draft: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

def get_lead_email(lead_id: str) -> str:
    """Get a lead's email from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Updated query to use correct column names
        cursor.execute("""
            SELECT email 
            FROM leads 
            WHERE lead_id = ?
        """, (lead_id,))
        
        result = cursor.fetchone()

        if not result:
            logger.error(f"No email found for lead_id={lead_id}")
            return ""

        return result[0]

    except Exception as e:
        logger.error(f"Error getting lead email: {str(e)}")
        return ""
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()

def store_draft_info(
    lead_id: str,
    draft_id: str,
    scheduled_date: datetime,
    subject: str,
    body: str,
    sequence_num: int = None,
):
    """Store draft information using the consolidated storage function."""
    lead_sheet = {"lead_data": {"properties": {"hs_object_id": lead_id}}}
    store_lead_email_info(
        lead_sheet=lead_sheet,
        draft_id=draft_id,
        scheduled_date=scheduled_date,
        subject=subject,
        body=body,
        sequence_num=sequence_num
    )

def send_message(sender, to, subject, message_text) -> Dict[str, Any]:
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(to=to, subject=subject, body=message_text)

    try:
        sent_msg = (
            service.users()
            .messages()
            .send(userId="me", body=message_body)
            .execute()
        )
        if sent_msg.get("id"):
            logger.info(f"Message sent successfully to '{to}' – ID={sent_msg['id']}")
            return {"status": "ok", "message_id": sent_msg["id"]}
        else:
            logger.error(f"Message sent to '{to}' but no ID returned – possibly an API error.")
            return {"status": "error", "error": "No message ID returned"}
    except Exception as e:
        logger.error(
            f"Failed to send message to '{to}' with subject='{subject}'. Error: {e}"
        )
        return {"status": "error", "error": str(e)}

def search_messages(query="") -> list:
    """
    Search for messages in the Gmail inbox using the specified `query`.
    For example:
      - 'from:someone@example.com'
      - 'subject:Testing'
      - 'to:me newer_than:7d'
    Returns a list of message dicts.
    """
    service = get_gmail_service()
    try:
        response = service.users().messages().list(userId="me", q=query).execute()
        return response.get("messages", [])
    except Exception as e:
        logger.error(f"Error searching messages with query='{query}': {e}")
        return []

def check_thread_for_reply(thread_id: str) -> bool:
    """
    Checks if there's more than one message in a given thread, indicating a reply.
    More precise than searching by 'from:' or date alone.
    """
    service = get_gmail_service()
    try:
        thread_data = service.users().threads().get(userId="me", id=thread_id).execute()
        msgs = thread_data.get("messages", [])
        return len(msgs) > 1
    except Exception as e:
        logger.error(f"Error retrieving thread {thread_id}: {e}")
        return False

def search_inbound_messages_for_email(email_address: str, max_results: int = 1) -> list:
    """
    Search for inbound messages sent from `email_address`.
    Returns a list of short snippets from the most recent matching messages.
    """
    query = f"from:{email_address}"
    message_ids = search_messages(query=query)
    if not message_ids:
        return []

    service = get_gmail_service()
    snippets = []
    for m in message_ids[:max_results]:
        try:
            full_msg = service.users().messages().get(
                userId="me",
                id=m["id"],
                format="full",
            ).execute()
            snippet = full_msg.get("snippet", "")
            snippets.append(snippet)
        except Exception as e:
            logger.error(f"Error fetching message {m['id']} from {email_address}: {e}")

    return snippets

```

## utils\hubspot_field_finder.py
```python
import json
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now we can import project modules
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger

class HubspotFieldFinder:
    def __init__(self):
        # Initialize HubspotService with API key from settings
        self.hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        
    def get_all_properties(self, object_type: str) -> List[Dict]:
        """Get all properties for a given object type (company or contact)."""
        try:
            url = f"{self.hubspot.base_url}/crm/v3/properties/{object_type}"
            response = self.hubspot._make_hubspot_get(url)
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Error getting {object_type} properties: {str(e)}")
            return []

    def search_property(self, search_term: str, object_type: str) -> List[Dict]:
        """Search for properties containing the search term."""
        properties = self.get_all_properties(object_type)
        matches = []
        
        search_term = search_term.lower()
        for prop in properties:
            if (search_term in prop.get("label", "").lower() or
                search_term in prop.get("name", "").lower() or
                search_term in prop.get("description", "").lower()):
                
                matches.append({
                    "internal_name": prop.get("name"),
                    "label": prop.get("label"),
                    "type": prop.get("type"),
                    "description": prop.get("description"),
                    "group_name": prop.get("groupName"),
                    "options": prop.get("options", [])
                })
        
        return matches

def print_matches(matches: List[Dict], search_term: str, object_type: str):
    """Pretty print the matching properties."""
    if not matches:
        print(f"\nNo matches found for '{search_term}' in {object_type} properties.")
        return
        
    print(f"\nFound {len(matches)} matches for '{search_term}' in {object_type} properties:")
    print("=" * 80)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. Internal Name: {match['internal_name']}")
        print(f"   Label: {match['label']}")
        print(f"   Type: {match['type']}")
        print(f"   Group: {match['group_name']}")
        if match['description']:
            print(f"   Description: {match['description']}")
        if match['options']:
            print("   Options:")
            for opt in match['options']:
                print(f"     - {opt.get('label')} ({opt.get('value')})")
        print("-" * 80)

def save_results(matches: List[Dict], object_type: str):
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hubspot_fields_{object_type}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(matches, f, indent=2)
    print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Search HubSpot fields by name or description')
    parser.add_argument('search_term', nargs='?', help='Term to search for in field names/descriptions')
    parser.add_argument('--type', '-t', choices=['companies', 'contacts'], default='companies',
                      help='Object type to search (companies or contacts)')
    parser.add_argument('--save', '-s', action='store_true',
                      help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Only show internal names (useful for scripting)')
    
    args = parser.parse_args()
    
    # If no search term provided, enter interactive mode
    if not args.search_term:
        print("\nHubSpot Field Finder")
        print("1. Search Company Fields")
        print("2. Search Contact Fields")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "3":
            return
            
        if choice not in ["1", "2"]:
            print("Invalid choice. Please try again.")
            return
            
        args.type = "companies" if choice == "1" else "contacts"
        args.search_term = input(f"\nEnter search term for {args.type} fields: ")
    
    finder = HubspotFieldFinder()
    matches = finder.search_property(args.search_term, args.type)
    
    if args.quiet:
        # Only print internal names, one per line
        for match in matches:
            print(match['internal_name'])
    else:
        print_matches(matches, args.search_term, args.type)
        
    if args.save:
        save_results(matches, args.type)

if __name__ == "__main__":
    main() 
```

## utils\logger_base.py
```python
"""Base logger configuration without settings dependencies."""
import logging
from pathlib import Path
from typing import Optional

def get_base_logger(
    name: Optional[str] = None,
    log_level: str = 'INFO'
) -> logging.Logger:
    """Create a basic logger without settings dependencies."""
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    # Set basic log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Add console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create base logger
logger = get_base_logger(__name__)

```

## utils\logging_setup.py
```python
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create logger first
    logger = logging.getLogger('utils.logging_setup')
    logger.setLevel(logging.DEBUG)
    
    # Skip if handlers already exist
    if logger.handlers:
        return logger
        
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Force UTF-8 encoding for the log file
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    # Update file handler to use UTF-8
    file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Add filters to reduce noise
    class VerboseFilter(logging.Filter):
        def filter(self, record):
            skip_patterns = [
                "Looking up time zone info",
                "Full xAI Request Payload",
                "Full xAI Response",
                "Found existing label",
                "CSV headers",
                "Loaded timezone data",
                "Replaced placeholder",
                "Raw segmentation response"
            ]
            return not any(pattern in str(record.msg) for pattern in skip_patterns)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: [%(pathname)s:%(lineno)d]\n'
    )
    
    # Set formatters
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add filters
    file_handler.addFilter(VerboseFilter())
    console_handler.addFilter(VerboseFilter())
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set levels for other loggers
    logging.getLogger('utils.gmail_integration').setLevel(logging.INFO)
    logging.getLogger('utils.xai_integration').setLevel(logging.INFO)
    logging.getLogger('services.data_gatherer_service').setLevel(logging.INFO)
    
    return logger

# Create the logger instance
logger = setup_logging()

```

## utils\model_selector.py
```python
from config.settings import (
    MODEL_FOR_EMAILS,
    MODEL_FOR_GENERAL,
    MODEL_FOR_ANALYSIS,
    DEFAULT_TEMPERATURE,
    EMAIL_TEMPERATURE,
    ANALYSIS_TEMPERATURE
)
from utils.logging_setup import logger

def get_openai_model(task_type: str = "general") -> tuple[str, float]:
    """
    Returns the appropriate OpenAI model and temperature based on the task type.
    
    Args:
        task_type (str): The type of task. 
                        "email" for email-related tasks,
                        "analysis" for detailed analysis tasks,
                        "general" for other tasks.
    Returns:
        tuple[str, float]: The model name and temperature setting
    """
    model_config = {
        "email": (MODEL_FOR_EMAILS, EMAIL_TEMPERATURE),
        "analysis": (MODEL_FOR_ANALYSIS, ANALYSIS_TEMPERATURE),
        "general": (MODEL_FOR_GENERAL, DEFAULT_TEMPERATURE)
    }
    
    model, temp = model_config.get(task_type, (MODEL_FOR_GENERAL, DEFAULT_TEMPERATURE))
    logger.debug(f"Selected model {model} with temperature {temp} for task type: {task_type}")
    
    return model, temp 
```

## utils\season_snippet.py
```python
# utils/season_snippet.py

import random

def get_season_variation_key(current_month, start_peak_month, end_peak_month):
    """
    Determine the season state based on the current_month (0-11),
    start_peak_month, and end_peak_month (also 0-11).
    
    0–2 months away from start: "approaching"
    Within start-end: "in_season"
    Less than 1 month left to end: "winding_down"
    Else: "off_season"
    """
    # If within peak
    if start_peak_month <= current_month <= end_peak_month:
        # If we are close to end_peak_month (like 0 or 1 months away)
        if (end_peak_month - current_month) < 1:
            return "winding_down"
        else:
            return "in_season"
    
    # If 0–2 months away from start
    # e.g., if start is 5 (June) and current_month is 3 or 4
    # that means we are 1 or 2 months away from peak
    months_away = start_peak_month - current_month
    # handle wrap-around if start_peak < current_month (peak season crosses year boundary)
    if months_away < 0:
        # e.g. peak is Jan (0) but current_month is Dec (11) -> we might do some logic
        months_away += 12
    
    if 1 <= months_away <= 2:
        return "approaching"
    else:
        return "off_season"


def pick_season_snippet(season_key):
    """
    Return a random snippet from the specified season_key.
    """
    # Each state has two snippet options
    snippet_options = {
        "approaching": [
            "As you prepare for the upcoming season,",
            "With the season just around the corner,",
        ],
        "in_season": [
            "With the season in full swing,"

        ],
        "winding_down": [
            "As the season winds down,",
        ],
        "off_season": [
            "As you prepare for the year ahead,"
        ]
    }

    # fallback if not found
    if season_key not in snippet_options:
        return ""

    return random.choice(snippet_options[season_key])

```

## utils\templates_directory.py
```python
import os
from pathlib import Path
import sys
from typing import List, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change relative import to absolute import
from utils.exceptions import ConfigurationError

def get_template_paths() -> Dict[str, List[str]]:
    """
    Gets all template directory and file paths under the docs/templates directory.
    
    Returns:
        Dict[str, List[str]]: Dictionary with keys 'directories' and 'files' containing
                             lists of all template directory and file paths respectively
    
    Raises:
        ConfigurationError: If templates directory cannot be found
    """
    # Get the project root directory (parent of utils/)
    root_dir = Path(__file__).parent.parent
    templates_dir = root_dir / "docs" / "templates"
    
    if not templates_dir.exists():
        raise ConfigurationError(
            "Templates directory not found",
            {"expected_path": str(templates_dir)}
        )

    template_paths = {
        "directories": [],
        "files": []
    }
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(templates_dir):
        # Convert to relative paths from project root
        rel_root = os.path.relpath(root, root_dir)
        
        # Add directory paths
        template_paths["directories"].append(rel_root)
        
        # Add file paths
        for file in files:
            rel_path = os.path.join(rel_root, file)
            template_paths["files"].append(rel_path)
            
    return template_paths

if __name__ == "__main__":
    try:
        print("\nScanning for templates...")
        paths = get_template_paths()
        
        print("\nTemplate Directories:")
        for directory in paths["directories"]:
            print(f"  - {directory}")
            
        print("\nTemplate Files:")
        for file in paths["files"]:
            print(f"  - {file}")
            
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e.message}")
        if e.details:
            print("Details:", e.details)
    except Exception as e:
        print(f"\nError: {str(e)}")

```

## utils\web_fetch.py
```python
import requests
from typing import Optional
from utils.logging_setup import logger
import urllib3
import random
from urllib.parse import urlparse, urlunparse

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of common user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

def sanitize_url(url: str) -> str:
    """Sanitize and normalize URL format."""
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    if not parsed.netloc.startswith("www."):
        parsed = parsed._replace(netloc=f"www.{parsed.netloc}")
    return urlunparse(parsed)

def fetch_website_html(url: str) -> Optional[str]:
    """Fetch HTML content from a website with proper headers and error handling."""
    if not url:
        logger.error("No URL provided")
        return None
        
    # Clean up the URL
    url = url.strip().lower()
    logger.debug(f"Original URL: {url}")
    
    # Generate URL variations
    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc.replace('www.', '')
    
    urls_to_try = [
        f"https://www.{base_domain}{parsed_url.path}",
        f"https://{base_domain}{parsed_url.path}",
        f"http://www.{base_domain}{parsed_url.path}",
        f"http://{base_domain}{parsed_url.path}"
    ]
    
    logger.debug(f"Will try URLs: {urls_to_try}")
    
    # Setup headers with additional browser-like headers
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',  # Do Not Track
        'Pragma': 'no-cache'
    }
    
    session = requests.Session()
    
    for attempt_url in urls_to_try:
        try:
            logger.debug(f"Trying to fetch: {attempt_url}")
            
            response = session.get(
                attempt_url,
                headers=headers,
                timeout=10,
                verify=False,  # Still keeping verify=False for testing
                allow_redirects=True
            )
            
            logger.debug(f"Response code: {response.status_code} for URL: {attempt_url}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Check if we got a successful response
            if response.status_code == 200:
                content = response.text
                logger.debug(f"Content preview: {content[:200]}")
                return content
            else:
                logger.debug(f"Failed with status code {response.status_code}")
                if response.text:
                    logger.debug(f"Error response preview: {response.text[:200]}")
                
        except requests.RequestException as e:
            logger.error(f"Error fetching {attempt_url}: {str(e)}", exc_info=True)
            continue
            
    logger.error(f"Failed to fetch website content after trying multiple URLs for {url}")
    return None

```

## utils\xai_integration.py
```python
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
```
