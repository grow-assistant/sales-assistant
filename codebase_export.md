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
- [scheduling\email_sender.py](#scheduling\email_sender.py)
- [scheduling\extended_lead_storage.py](#scheduling\extended_lead_storage.py)
- [scheduling\followup_generation.py](#scheduling\followup_generation.py)
- [scheduling\followup_scheduler.py](#scheduling\followup_scheduler.py)
- [scheduling\sql_lookup.py](#scheduling\sql_lookup.py)
- [scripts\__init__.py](#scripts\__init__.py)
- [scripts\build_template.py](#scripts\build_template.py)
- [scripts\check_reviewed_drafts.py](#scripts\check_reviewed_drafts.py)
- [scripts\find_email_ids_with_responses.py](#scripts\find_email_ids_with_responses.py)
- [scripts\get_random_contacts.py](#scripts\get_random_contacts.py)
- [scripts\golf_outreach_strategy.py](#scripts\golf_outreach_strategy.py)
- [scripts\job_title_categories.py](#scripts\job_title_categories.py)
- [scripts\migrate_emails_table.py](#scripts\migrate_emails_table.py)
- [scripts\monitor\auto_reply_processor.py](#scripts\monitor\auto_reply_processor.py)
- [scripts\monitor\bounce_processor.py](#scripts\monitor\bounce_processor.py)
- [scripts\monitor\monitor_email_review_status.py](#scripts\monitor\monitor_email_review_status.py)
- [scripts\monitor\monitor_email_sent_status.py](#scripts\monitor\monitor_email_sent_status.py)
- [scripts\monitor\review_email_responses.py](#scripts\monitor\review_email_responses.py)
- [scripts\monitor\run_email_monitoring.py](#scripts\monitor\run_email_monitoring.py)
- [scripts\ping_hubspot_for_gm.py](#scripts\ping_hubspot_for_gm.py)
- [scripts\run_email_monitoring.py](#scripts\run_email_monitoring.py)
- [scripts\run_scheduler.py](#scripts\run_scheduler.py)
- [scripts\schedule_outreach.py](#scripts\schedule_outreach.py)
- [scripts\update_company_names.py](#scripts\update_company_names.py)
- [services\__init__.py](#services\__init__.py)
- [services\company_enrichment_service.py](#services\company_enrichment_service.py)
- [services\conversation_analysis_service.py](#services\conversation_analysis_service.py)
- [services\data_gatherer_service.py](#services\data_gatherer_service.py)
- [services\gmail_service.py](#services\gmail_service.py)
- [services\hubspot_service.py](#services\hubspot_service.py)
- [services\leads_service.py](#services\leads_service.py)
- [services\orchestrator_service.py](#services\orchestrator_service.py)
- [services\response_analyzer_service.py](#services\response_analyzer_service.py)
- [utils\__init__.py](#utils\__init__.py)
- [utils\conversation_summary.py](#utils\conversation_summary.py)
- [utils\data_extraction.py](#utils\data_extraction.py)
- [utils\date_utils.py](#utils\date_utils.py)
- [utils\doc_reader.py](#utils\doc_reader.py)
- [utils\enrich_hubspot_company_data.py](#utils\enrich_hubspot_company_data.py)
- [utils\exceptions.py](#utils\exceptions.py)
- [utils\export_codebase.py](#utils\export_codebase.py)
- [utils\export_codebase_primary_files.py](#utils\export_codebase_primary_files.py)
- [utils\export_templates.py](#utils\export_templates.py)
- [utils\formatting_utils.py](#utils\formatting_utils.py)
- [utils\gmail_integration.py](#utils\gmail_integration.py)
- [utils\gmail_service.py](#utils\gmail_service.py)
- [utils\hubspot_field_finder.py](#utils\hubspot_field_finder.py)
- [utils\logger_base.py](#utils\logger_base.py)
- [utils\logging_setup.py](#utils\logging_setup.py)
- [utils\main_followups.py](#utils\main_followups.py)
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
# UPDATED FILE: REMOVED FILTERS / USING LIST-BASED PULL
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
import time
from utils.data_extraction import extract_lead_data

# -----------------------------------------------------------------------------
# PROJECT IMPORTS (Adjust paths/names as needed)
# -----------------------------------------------------------------------------
from services.hubspot_service import HubspotService
from services.company_enrichment_service import CompanyEnrichmentService
from services.data_gatherer_service import DataGathererService
from scripts.golf_outreach_strategy import (
    get_best_outreach_window, 
    get_best_month
    
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
# LIMIT FOR HOW MANY CONTACTS/LEADS TO PROCESS
# -----------------------------------------------------------------------------
LEADS_TO_PROCESS = 200  # This is just a processing limit, not a filter.

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
# UTILITY FUNCTIONS (still used in the pipeline)
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
    """
    Collects all prior emails and notes from the lead_sheet, then uses OpenAI
    to summarize them. This remains here in case you still process extended data.
    """
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
        target_date = now  # Start with today instead of tomorrow
        
        # If current time is past the best window end time, move to next day
        if now.hour >= int(best_time["end"]):
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
        send_hour = int(best_time["start"])
        send_minute = random.randint(0, 59)
        
        # If it's today and current time is before best window start, use best window start
        if target_date.date() == now.date() and now.hour < send_hour:
            send_hour = int(best_time["start"])
        # If it's today and current time is within window, use current hour + small delay
        elif target_date.date() == now.date() and now.hour >= send_hour and now.hour < int(best_time["end"]):
            send_hour = now.hour
            send_minute = now.minute + random.randint(2, 15)
            if send_minute >= 60:
                send_hour += 1
                send_minute -= 60
                if send_hour >= int(best_time["end"]):
                    target_date += timedelta(days=1)
                    send_hour = int(best_time["start"])
        
        send_date = target_date.replace(
            hour=send_hour,
            minute=send_minute,
            second=0,
            microsecond=0
        )
        
        final_send_date = send_date
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

# -----------------------------------------------------------------------------
# NEW: LIST-BASED CONTACT PULL
# -----------------------------------------------------------------------------
def get_contacts_from_list(list_id: str, hubspot: HubspotService) -> List[Dict[str, Any]]:
    """
    Pull contacts from a specified HubSpot list. This replaces all prior
    lead-filter or company-filter logic in main. We simply fetch the list
    membership and then retrieve each contact individually.
    """
    try:
        # 1) Get list memberships
        url = f"https://api.hubapi.com/crm/v3/lists/{list_id}/memberships"
        logger.debug(f"Fetching list memberships from: {url}")
        memberships = hubspot._make_hubspot_get(url)

        # 2) Extract record IDs
        record_ids = []
        if isinstance(memberships, dict) and "results" in memberships:
            record_ids = [result["recordId"] for result in memberships.get("results", [])]

        logger.debug(f"Found {len(record_ids)} records in list {list_id}")

        # 3) Fetch contact details for each ID
        contacts = []
        for record_id in record_ids:
            try:
                contact_url = f"https://api.hubapi.com/crm/v3/objects/contacts/{record_id}"
                params = {
                    "properties": [
                        "email",
                        "firstname",
                        "lastname",
                        "jobtitle",
                        "associatedcompanyid"
                    ]
                }
                contact_data = hubspot._make_hubspot_get(contact_url, params)
                if contact_data:
                    contacts.append(contact_data)
            except Exception as e:
                logger.warning(f"Failed to fetch details for contact {record_id}: {str(e)}")
                continue

        logger.info(f"Successfully retrieved {len(contacts)} contacts from list {list_id}")
        return contacts

    except Exception as e:
        logger.error(f"Error getting contacts from list {list_id}: {str(e)}", exc_info=True)
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

# -----------------------------------------------------------------------------
# REPLACEMENT WORKFLOW USING THE LIST
# -----------------------------------------------------------------------------
def main_list_workflow():
    """
    1) Pull contacts from a specified HubSpot list.
    2) For each contact, retrieve the associated company (if any).
    3) Enrich the company data, gather personalization, build & store an outreach email draft.
    """
    try:
        # Limit how many contacts (leads) we process
        leads_processed = 0
        
        workflow_context = {'correlation_id': str(uuid.uuid4())}
        hubspot = HubspotService(HUBSPOT_API_KEY)
        company_enricher = CompanyEnrichmentService()
        data_gatherer = DataGathererService()
        conversation_analyzer = ConversationAnalysisService()

        # Replace this with your actual HubSpot list ID:
        LIST_ID = "221"
        
        with workflow_step("1", "Get contacts from HubSpot list", workflow_context):
            contacts = get_contacts_from_list(LIST_ID, hubspot)
            logger.info(f"Found {len(contacts)} contacts to process from list {LIST_ID}")
            
            # Shuffle to randomize processing order if desired
            random.shuffle(contacts)

        with workflow_step("2", "Process each contact & its associated company", workflow_context):
            for contact in contacts:
                contact_id = contact.get("id")
                props = contact.get("properties", {})
                email = props.get("email")

                logger.info(f"Processing contact: {email} (ID: {contact_id})")

                if not email:
                    logger.info(f"Skipping contact {contact_id} - no email found.")
                    continue

                # Check if we've recently sent them an email
                if has_recent_email(email):
                    logger.info(f"Skipping {email} - an email was sent in the last 2 months")
                    continue

                # 1) Grab company if associated
                associated_company_id = props.get("associatedcompanyid")
                company_props = {}
                
                if associated_company_id:
                    company_data = get_company_by_id(hubspot, associated_company_id)
                    if company_data and company_data.get("id"):
                        company_props = company_data.get("properties", {})
                        
                        # Optionally check competitor from website, etc.
                        website = company_props.get("website")
                        if website:
                            competitor_info = data_gatherer.check_competitor_on_website(website)
                            if competitor_info["status"] == "success" and competitor_info["competitor"]:
                                competitor = competitor_info["competitor"]
                                logger.debug(f"Found competitor {competitor} for company {associated_company_id}")
                                try:
                                    # Update HubSpot
                                    hubspot._update_company_properties(
                                        company_data["id"], 
                                        {"competitor": competitor}
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update competitor for company {associated_company_id}: {e}")
                            
                        # Enrich the company data
                        enrichment_result = company_enricher.enrich_company(company_data["id"])
                        if enrichment_result.get("success", False):
                            company_props.update(enrichment_result.get("data", {}))
                
                # 2) Extract lead + company data into our unified structure
                lead_data_full = extract_lead_data(company_props, {
                    "firstname": props.get("firstname", ""),
                    "lastname": props.get("lastname", ""),
                    "email": email,
                    "jobtitle": props.get("jobtitle", "")
                })

                # 3) Gather personalization data
                personalization = gather_personalization_data(
                    company_name=lead_data_full["company_data"]["name"],
                    city=lead_data_full["company_data"]["city"],
                    state=lead_data_full["company_data"]["state"]
                )

                # 4) Pick a template path
                template_path = get_template_path(
                    club_type=lead_data_full["company_data"]["club_type"],
                    role=lead_data_full["lead_data"]["jobtitle"]
                )

                # 5) Calculate best send date
                send_date = calculate_send_date(
                    geography=lead_data_full["company_data"]["geographic_seasonality"],
                    persona=lead_data_full["lead_data"]["jobtitle"],
                    state_code=lead_data_full["company_data"]["state"],
                    season_data={
                        "peak_season_start": lead_data_full["company_data"].get("peak_season_start_month"),
                        "peak_season_end": lead_data_full["company_data"].get("peak_season_end_month")
                    }
                )

                # 6) Build outreach email
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

                if not email_content:
                    logger.error(f"Failed to generate email content for contact {contact_id}")
                    continue

                # 7) Replace placeholders in subject/body
                subject = replace_placeholders(email_content[0], lead_data_full)
                body = replace_placeholders(email_content[1], lead_data_full)

                # 8) Get conversation summary
                try:
                    logger.debug(f"Starting conversation analysis for email: {email}")
                    
                    # Log the input state
                    logger.debug(f"Conversation analyzer state before analysis:")
                    logger.debug(f"- Email: {email}")
                    logger.debug(f"- Analyzer type: {type(conversation_analyzer).__name__}")
                    
                    # Perform the analysis with timing
                    start_time = time.time()
                    conversation_summary = conversation_analyzer.analyze_conversation(email)
                    analysis_time = time.time() - start_time
                    
                    # Log the results
                    logger.info(f"Conversation analysis completed in {analysis_time:.2f} seconds")
                    logger.debug(f"Analysis results:")
                    logger.debug(f"- Summary length: {len(conversation_summary) if conversation_summary else 0}")
                    logger.debug(f"- Summary content: {conversation_summary}")
                    
                    if not conversation_summary:
                        logger.warning("Conversation analysis returned empty summary")
                
                except Exception as e:
                    logger.error(f"Error during conversation analysis for {email}: {str(e)}", exc_info=True)
                    conversation_summary = None
                    raise

                # 9) Build context block
                context = build_context_block(
                    interaction_history=conversation_summary,
                    original_email={"subject": subject, "body": body},
                    company_data={
                        "name": lead_data_full["company_data"]["name"],
                        "company_short_name": (
                            lead_data_full["company_data"].get("company_short_name") or
                            lead_data_full.get("company_short_name") or
                            lead_data_full["company_data"]["name"].split(" ")[0]
                        ),
                        "city": lead_data_full["company_data"].get("city", ""),
                        "state": lead_data_full["company_data"].get("state", ""),
                        "club_type": lead_data_full["company_data"]["club_type"],
                        "club_info": lead_data_full["company_data"].get("club_info", "")
                    }
                )

                # 10) Personalize with xAI
                personalized_content = personalize_email_with_xai(
                    lead_sheet=lead_data_full,
                    subject=subject,
                    body=body,
                    summary=conversation_summary,
                    context=context
                )

                # 11) Create Gmail draft
                draft_result = create_draft(
                    sender="me",
                    to=email,
                    subject=personalized_content["subject"],
                    message_text=personalized_content["body"]
                )

                if draft_result["status"] == "ok":
                    # 12) Store info in local DB
                    store_lead_email_info(
                        lead_sheet={
                            "lead_data": {
                                "email": email,
                                "properties": {
                                    "hs_object_id": contact_id,
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
                    logger.info(f"Created draft email for {email}")
                else:
                    logger.error(f"Failed to create Gmail draft for contact {contact_id}")

                leads_processed += 1
                logger.info(f"Completed processing contact {contact_id} ({leads_processed}/{LEADS_TO_PROCESS})")

                if leads_processed >= LEADS_TO_PROCESS:
                    logger.info(f"Reached processing limit of {LEADS_TO_PROCESS} contacts")
                    return

    except LeadContextError as e:
        logger.error(f"Lead context error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in main_list_workflow: {str(e)}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# SINGLE ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    logger.debug(f"Starting with CLEAR_LOGS_ON_START={CLEAR_LOGS_ON_START}")
    
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Start scheduler in background
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # We now only use the list-based approach
    main_list_workflow()

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
"""
Database operations for the scheduling service.
"""
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
                    gmail_id           VARCHAR(100),
                    company_short_name VARCHAR(100) NULL
                )
            END
        """)
        
        
        conn.commit()
        
        
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
                     status: str = 'pending',
                     company_short_name: str = None) -> int:
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
    - company_short_name
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
                status = ?,
                company_short_name = ?
            WHERE draft_id = ? AND lead_id = ?
        """, (
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            company_short_name,
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
                draft_id,
                company_short_name
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            lead_id,
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id,
            company_short_name
        ))
        cursor.execute("SELECT SCOPE_IDENTITY()")
        return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")


```

## scheduling\email_sender.py
```python
from datetime import datetime
from utils.gmail_integration import get_gmail_service
from scheduling.database import get_db_connection
import logging
import time
import random

logger = logging.getLogger(__name__)

def send_scheduled_emails():
    """
    Sends emails that are scheduled for now or in the past.
    Updates their status in the database.
    Includes random delays between sends to appear more natural.
    """
    now = datetime.now()
    logger.info(f"Checking for emails to send at {now}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Find emails that should be sent (both draft and reviewed status)
        cursor.execute("""
            SELECT email_id, draft_id, email_address, scheduled_send_date, status
            FROM emails 
            WHERE status IN ('reviewed')
            AND scheduled_send_date <= GETDATE()
            AND actual_send_date IS NULL
            ORDER BY scheduled_send_date ASC
        """)
        
        to_send = cursor.fetchall()
        logger.info(f"Found {len(to_send)} emails to send")

        if not to_send:
            return

        service = get_gmail_service()
        
        for email_id, draft_id, recipient, scheduled_date, status in to_send:
            try:
                logger.debug(f"Attempting to send email_id={email_id} to {recipient} (scheduled for {scheduled_date}, status={status})")
                
                # Send the draft
                message = service.users().drafts().send(
                    userId='me',
                    body={'id': draft_id}
                ).execute()

                # Update database
                cursor.execute("""
                    UPDATE emails 
                    SET status = 'sent',
                        actual_send_date = GETDATE(),
                        gmail_id = ?
                    WHERE email_id = ?
                """, (message['id'], email_id))
                
                conn.commit()
                logger.info(f"Successfully sent email_id={email_id} to {recipient}")

                # Random delay between sends (30-180 seconds)
                delay = random.uniform(5, 30)
                logger.debug(f"Waiting {delay:.1f} seconds before next send")
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Failed to send email_id={email_id} to {recipient}: {str(e)}", exc_info=True)
                conn.rollback()
                
                continue 
```

## scheduling\extended_lead_storage.py
```python
# scheduling/extended_lead_storage.py

from datetime import datetime, timedelta
from utils.logging_setup import logger
from scheduling.database import get_db_connection, store_email_draft

def find_next_available_timeslot(desired_send_date: datetime, preferred_window: dict = None) -> datetime:
    """
    Finds the next available timeslot using 30-minute windows.
    Within each window, attempts to schedule with 2-minute increments.
    If a window is full, moves to the next 30-minute window.
    
    Args:
        desired_send_date: The target date/time
        preferred_window: Optional dict with start/end times to constrain scheduling
    """
    logger.debug(f"Finding next available timeslot starting from: {desired_send_date}")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Round to nearest 30-minute window
        minutes = desired_send_date.minute
        rounded_minutes = (minutes // 30) * 30
        current_window_start = desired_send_date.replace(
            minute=rounded_minutes,
            second=0,
            microsecond=0
        )
        
        while True:
            # Check if we're still within preferred window
            if preferred_window:
                current_time = current_window_start.hour + current_window_start.minute / 60
                if current_time > preferred_window["end"]:
                    # Move to next day at start of preferred window
                    next_day = current_window_start + timedelta(days=1)
                    current_window_start = next_day.replace(
                        hour=int(preferred_window["start"]),
                        minute=int((preferred_window["start"] % 1) * 60)
                    )
                    logger.debug(f"Outside preferred window, moving to next day: {current_window_start}")
                    continue
            
            # Try each 2-minute slot within the current 30-minute window
            for minutes_offset in range(0, 30, 2):
                proposed_time = current_window_start + timedelta(minutes=minutes_offset)
                logger.debug(f"Checking availability for timeslot: {proposed_time}")
                
                # Check if this specific timeslot is available
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM emails
                    WHERE scheduled_send_date = ?
                """, (proposed_time,))
                
                count = cursor.fetchone()[0]
                logger.debug(f"Found {count} existing emails at timeslot {proposed_time}")
                
                if count == 0:
                    logger.debug(f"Selected available timeslot at {proposed_time}")
                    return proposed_time
            
            # If we get here, the current 30-minute window is full
            # Move to the next 30-minute window
            current_window_start += timedelta(minutes=30)
            logger.debug(f"Current window full, moving to next window starting at {current_window_start}")


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
    logger.debug(f"Storing lead email info for draft_id={draft_id}, scheduled_date={scheduled_date}")
    
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
        company_data = lead_sheet.get("company_data", {})

        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        email_address = lead_data.get("email")
        company_short_name = company_data.get("company_short_name", "").strip()

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
            status='draft',
            company_short_name=company_short_name
        )

        conn.commit()
        logger.info(
            f"[store_lead_email_info] Scheduled email for lead_id={lead_id}, email={email_address}, "
            f"company={company_short_name}, draft_id={draft_id} at {scheduled_date}",
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
# scheduling/followup_generation.py
"""
Functions for generating follow-up emails.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.database import get_db_connection
from utils.gmail_integration import create_draft, get_gmail_service, get_gmail_template
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import (
    get_best_outreach_window,
    calculate_send_date
)
from datetime import datetime, timedelta
import random
import base64
from services.gmail_service import GmailService

def get_calendly_template() -> str:
    """Load the Calendly HTML template from file."""
    try:
        template_path = Path(project_root) / 'docs' / 'templates' / 'calendly.html'
        logger.debug(f"Loading Calendly template from: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_html = f.read()
            
        logger.debug(f"Loaded template, length: {len(template_html)}")
        return template_html
    except Exception as e:
        logger.error(f"Error loading Calendly template: {str(e)}")
        return ""

def generate_followup_email_xai(
    lead_id: int, 
    email_id: int = None,
    sequence_num: int = None,
    original_email: dict = None
) -> dict:
    """Generate a follow-up email using xAI and original Gmail message"""
    try:
        logger.debug(f"Starting follow-up generation for lead_id={lead_id}, sequence_num={sequence_num}")
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the most recent email if not provided
        if not original_email:
            cursor.execute("""
                SELECT TOP 1
                    email_address,
                    name,
                    company_short_name,
                    body,
                    gmail_id,
                    scheduled_send_date,
                    draft_id
                FROM emails
                WHERE lead_id = ?
                AND sequence_num = 1
                AND gmail_id IS NOT NULL
                ORDER BY created_at DESC
            """, (lead_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"No original email found for lead_id={lead_id}")
                return None

            email_address, name, company_short_name, body, gmail_id, scheduled_date, draft_id = row
            original_email = {
                'email': email_address,
                'name': name,
                'company_short_name': company_short_name,
                'body': body,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'draft_id': draft_id
            }

        # Get the Gmail service and raw service
        gmail_service = GmailService()
        service = get_gmail_service()
        
        # Get the original message from Gmail
        try:
            message = service.users().messages().get(
                userId='me',
                id=original_email['gmail_id'],
                format='full'
            ).execute()
            
            # Get headers
            headers = message['payload'].get('headers', [])
            orig_subject = next(
                (header['value'] for header in headers if header['name'].lower() == 'subject'),
                'No Subject'
            )
            
            # Get the original sender (me) and timestamp
            from_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'from'),
                'me'
            )
            date_header = next(
                (header['value'] for header in headers if header['name'].lower() == 'date'),
                ''
            )
            
            # Get original HTML content
            original_html = None
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/html':
                        original_html = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break
            elif message['payload'].get('mimeType') == 'text/html':
                original_html = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error fetching Gmail message: {str(e)}", exc_info=True)
            return None

        # Build the new content with proper div wrapper
        venue_name = original_email.get('company_short_name', 'your club')
        subject = f"Re: {orig_subject}"
        
        followup_content = (
            f"<div dir='ltr'>"
            f"<p>I wanted to quickly follow up on my previous email about Swoop. "
            f"We've made setup incredibly easy—just send us a menu, and {venue_name} could be up and running in as little as 24-48 hours to try out.</p>"
            f"<p>Would you have 10 minutes next week for a brief call?</p>"
            f"<p>Ty</p>"
            f"</div>"
        )

        # followup_content = (
        #     f"<div dir='ltr'>"
        #     f"<p>Following up about improving operations at {venue_name}. "
        #     f"Would you have 10 minutes next week for a brief call?</p>"
        #     f"<p>Thanks,<br>Ty</p>"
        #     f"</div>"
        # )
        
        # Get Calendly template
        template_html = get_calendly_template()
        
        # Create blockquote for original email
        blockquote_html = (
            f'<blockquote class="gmail_quote" '
            f'style="margin:0 0 0 .8ex;border-left:1px #ccc solid;padding-left:1ex">\n'
            f'{original_html or ""}'
            f'</blockquote>'
        )
        
        # Combine in correct order with proper HTML structure
        full_html = (
            f"{followup_content}\n"     # New follow-up content first
            f"{template_html}\n"        # Calendly template second
            f"{blockquote_html}"        # Original email last (with Gmail blockquote formatting)
        )

        # Now log the combined HTML after it's created
        logger.debug("\n=== Final Combined HTML ===")
        logger.debug("First 1000 chars:")
        logger.debug(full_html[:1000])
        logger.debug("\nLast 1000 chars:")
        logger.debug(full_html[-1000:] if len(full_html) > 1000 else "N/A")
        logger.debug("=" * 80)

        # Calculate send date using golf_outreach_strategy logic
        send_date = calculate_send_date(
            geography=original_email.get('geographic_seasonality', 'Year-Round Golf'),
            persona="General Manager",  # Could be made dynamic based on contact data
            state=original_email.get('state'),  # Now properly using state from original_email
            sequence_num=sequence_num or 2,
            season_data=None
        )
        logger.debug(f"Using state from lead info: {original_email.get('state')}")
        logger.debug(f"Calculated send date: {send_date}")

        # Ensure minimum 3-day gap from original send date
        orig_scheduled_date = original_email.get('scheduled_send_date', datetime.now())
        logger.debug(f"Original scheduled date: {orig_scheduled_date}")
        while send_date < (orig_scheduled_date + timedelta(days=3)):
            send_date += timedelta(days=1)
            logger.debug(f"Adjusted send date to ensure 3-day gap: {send_date}")

        return {
            'email': original_email.get('email'),
            'subject': subject,
            'body': full_html,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2,
            'lead_id': str(lead_id),
            'company_short_name': original_email.get('company_short_name', ''),
            'in_reply_to': original_email['gmail_id'],
            'thread_id': original_email['gmail_id']
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
from scripts.golf_outreach_strategy import get_best_outreach_window
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

## scripts\find_email_ids_with_responses.py
```python
# test_hubspot_leads_service.py
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def get_random_lead_id():
    """Get a random lead_id from the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Modified query to ensure we get a valid HubSpot contact ID
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY NEWID()
            """)
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database: {lead_id}")
                return lead_id
            logger.warning("No valid lead_id found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting random lead_id: {str(e)}")
        return None

def get_lead_id_for_email(email):
    """Get lead_id for a specific email address."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.email_address = ?
                  AND e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
            """, (email,))
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database for {email}: {lead_id}")
                return lead_id
            logger.warning(f"No valid lead_id found for email: {email}")
            return None
    except Exception as e:
        logger.error(f"Error getting lead_id for email {email}: {str(e)}")
        return None

def get_all_leads():
    """Get all lead IDs and emails from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT e.lead_id, e.email_address, e.name, e.company_short_name
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY e.company_short_name, e.email_address
            """)
            results = cursor.fetchall()
            if results:
                leads = [
                    {
                        'lead_id': str(row[0]),
                        'email': row[1],
                        'name': row[2],
                        'company': row[3]
                    }
                    for row in results
                ]
                logger.debug(f"Found {len(leads)} leads in database")
                return leads
            logger.warning("No valid leads found in database")
            return []
    except Exception as e:
        logger.error(f"Error getting leads: {str(e)}")
        return []

def test_lead_info():
    """Test function to pull HubSpot data for a random lead ID."""
    try:
        # Initialize services in correct order
        data_gatherer = DataGathererService()  # Initialize without parameters
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service  # Set the service after initialization
        leads_service = LeadsService(data_gatherer)
        
        # Get random contact ID from database
        contact_id = get_random_lead_id()
        if not contact_id:
            print("No lead IDs found in database")
            return
            
        print(f"\nFetching info for contact ID: {contact_id}")
        
        # Verify contact exists in HubSpot before proceeding
        try:
            # Test if we can get contact properties
            contact_props = hubspot_service.get_contact_properties(contact_id)
            if not contact_props:
                print(f"Contact ID {contact_id} not found in HubSpot")
                return
        except Exception as e:
            print(f"Error verifying contact in HubSpot: {str(e)}")
            return
            
        # Get lead summary using LeadsService
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error: {lead_info['error']}")
            return
            
        # Print results
        print("\nLead Information:")
        print("=" * 50)
        print(f"Last Reply Date: {lead_info['last_reply_date']}")
        print(f"Lifecycle Stage: {lead_info['lifecycle_stage']}")
        print(f"Company Name: {lead_info['company_name']}")
        print(f"Company Short Name: {lead_info['company_short_name']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_lead_info: {str(e)}", exc_info=True)

def test_specific_lead():
    """Test function to pull HubSpot data for pcrow@chicagohighlands.com."""
    try:
        email = "pcrow@chicagohighlands.com"
        print(f"\nTesting lead info for: {email}")
        
        # Initialize services
        data_gatherer = DataGathererService()
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service
        leads_service = LeadsService(data_gatherer)
        
        # Get contact ID from database
        contact_id = get_lead_id_for_email(email)
        if not contact_id:
            print(f"No lead ID found for email: {email}")
            return
            
        print(f"Found contact ID: {contact_id}")
        
        # Test direct HubSpot API calls
        print("\nTesting direct HubSpot API calls:")
        print("=" * 50)
        
        # Get contact properties
        contact_props = hubspot_service.get_contact_properties(contact_id)
        if contact_props:
            print("Contact properties found:")
            for key, value in contact_props.items():
                print(f"{key}: {value}")
        else:
            print("No contact properties found")
            
        # Get latest emails
        print("\nChecking latest emails:")
        print("=" * 50)
        try:
            latest_emails = hubspot_service.get_latest_emails_for_contact(contact_id)
            if latest_emails:
                print("Latest email interactions:")
                for email in latest_emails:
                    print(f"Type: {email.get('type')}")
                    print(f"Date: {email.get('created_at')}")
                    print(f"Subject: {email.get('subject', 'No subject')}")
                    print("-" * 30)
            else:
                print("No email interactions found")
        except Exception as e:
            print(f"Error getting latest emails: {str(e)}")
        
        # Get lead summary
        print("\nTesting LeadsService summary:")
        print("=" * 50)
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error: {lead_info['error']}")
            return
            
        print("Lead Summary Information:")
        print(f"Last Reply Date: {lead_info.get('last_reply_date', 'None')}")
        print(f"Lifecycle Stage: {lead_info.get('lifecycle_stage', 'None')}")
        print(f"Company Name: {lead_info.get('company_name', 'None')}")
        print(f"Company Short Name: {lead_info.get('company_short_name', 'None')}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_specific_lead: {str(e)}", exc_info=True)

def test_recent_replies():
    """Test function to find all leads who replied in the last month."""
    try:
        print("\nChecking for recent replies from all leads...")
        
        # Initialize services
        data_gatherer = DataGathererService()
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service
        
        # Get all leads
        leads = get_all_leads()
        if not leads:
            print("No leads found in database")
            return
            
        print(f"Found {len(leads)} total leads to check")
        print("=" * 50)
        
        # Track replies
        recent_replies = []
        one_month_ago = datetime.now() - timedelta(days=30)
        
        # Check each lead
        for lead in leads:
            try:
                # Get contact properties
                contact_props = hubspot_service.get_contact_properties(lead['lead_id'])
                if not contact_props:
                    continue
                
                last_reply = contact_props.get('hs_sales_email_last_replied')
                if last_reply:
                    try:
                        reply_date = datetime.fromtimestamp(int(last_reply)/1000)
                        if reply_date > one_month_ago:
                            lead['reply_date'] = reply_date
                            lead['properties'] = contact_props
                            recent_replies.append(lead)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing reply date for {lead['email']}: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Error checking lead {lead['email']}: {str(e)}")
                continue
        
        # Print results
        print(f"\nFound {len(recent_replies)} leads with recent replies:")
        print("=" * 50)
        
        if recent_replies:
            # Sort by reply date, most recent first
            recent_replies.sort(key=lambda x: x['reply_date'], reverse=True)
            
            for lead in recent_replies:
                print(f"\nName: {lead['name']}")
                print(f"Email: {lead['email']}")
                print(f"Company: {lead['company']}")
                print(f"Reply Date: {lead['reply_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Try to get the actual email content
                try:
                    latest_emails = hubspot_service.get_latest_emails_for_contact(lead['lead_id'])
                    if latest_emails:
                        print("Latest Email Details:")
                        for email in latest_emails:
                            if email.get('type') == 'INCOMING':  # Only show replies
                                print(f"Subject: {email.get('subject', 'No subject')}")
                                print(f"Date: {email.get('created_at')}")
                                print(f"Preview: {email.get('body', '')[:200]}...")
                except Exception as e:
                    print(f"Could not fetch email details: {str(e)}")
                
                print("-" * 50)
        else:
            print("No leads found with replies in the last month")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_recent_replies: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Enable debug logging
    logger.setLevel("DEBUG")
    test_recent_replies()



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
# scripts/golf_outreach_strategy.py
# """
# Scripts for determining optimal outreach timing based on club and contact attributes.
# """
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os
import random

logger = logging.getLogger(__name__)

# Update the logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to file
        logging.StreamHandler()          # Log to console
    ]
)

# Add a debug message to verify logging is working
logger.debug("Golf outreach strategy logging initialized")

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
    logger.debug(f"Loaded timezone offsets for {len(offsets)} states")
    return offsets

STATE_OFFSETS = load_state_offsets()


def get_best_month(geography: str, club_type: str = None, season_data: dict = None) -> list:
    """
    Determine best outreach months based on geography/season and club type.
    """
    current_month = datetime.now().month
    logger.debug(f"Determining best month for geography: {geography}, club_type: {club_type}, current month: {current_month}")
    
    # If we have season data, use it as primary decision factor
    if season_data:
        peak_start = season_data.get('peak_season_start', '')
        peak_end = season_data.get('peak_season_end', '')
        logger.debug(f"Using season data - peak start: {peak_start}, peak end: {peak_end}")
        
        if peak_start and peak_end:
            peak_start_month = int(peak_start.split('-')[0])
            peak_end_month = int(peak_end.split('-')[0])
            
            logger.debug(f"Peak season: {peak_start_month} to {peak_end_month}")
            
            # For winter peak season (crossing year boundary)
            if peak_start_month > peak_end_month:
                if current_month >= peak_start_month or current_month <= peak_end_month:
                    logger.debug("In winter peak season, targeting September shoulder season")
                    return [9]  # September (before peak starts)
                else:
                    logger.debug("In winter shoulder season, targeting January")
                    return [1]  # January
            # For summer peak season
            else:
                if peak_start_month <= current_month <= peak_end_month:
                    target = [peak_start_month - 1] if peak_start_month > 1 else [12]
                    logger.debug(f"In summer peak season, targeting month {target}")
                    return target
                else:
                    logger.debug("In summer shoulder season, targeting January")
                    return [1]  # January
    
    # Fallback to geography-based matrix
    month_matrix = {
        "Year-Round Golf": [1, 9],      # January or September
        "Peak Winter Season": [9],       # September
        "Peak Summer Season": [2],       # February
        "Short Summer Season": [1],      # January
        "Shoulder Season Focus": [2, 10]  # February or October
    }
    
    result = month_matrix.get(geography, [1, 9])
    logger.debug(f"Using geography matrix fallback for {geography}, selected months: {result}")
    return result

def get_best_time(persona: str, sequence_num: int, timezone_offset: int = 0) -> dict:
    """
    Determine best time of day based on persona and email sequence number.
    Returns a dict with start time in MST.
    
    Args:
        persona: The recipient's role/persona
        sequence_num: Email sequence number (1 or 2)
        timezone_offset: Hours offset from MST (e.g., 2 for EST)
    """
    logger.debug(f"\n=== Time Window Selection ===")
    logger.debug(f"Input - Persona: {persona}, Sequence: {sequence_num}, Timezone offset: {timezone_offset}")
    
    # Default to General Manager if persona is None or invalid
    if not persona:
        logger.warning("No persona provided, defaulting to General Manager")
        persona = "General Manager"
    
    # Normalize persona string
    try:
        persona = " ".join(word.capitalize() for word in persona.split("_"))
        if persona.lower() in ["general manager", "gm", "club manager", "general_manager"]:
            persona = "General Manager"
        elif persona.lower() in ["f&b manager", "food & beverage manager", "food and beverage manager", "fb_manager"]:
            persona = "Food & Beverage Director"
    except Exception as e:
        logger.error(f"Error normalizing persona: {str(e)}")
        persona = "General Manager"
    
    # Define windows in LOCAL time
    time_windows = {
        "General Manager": {
            # 2: {  # Sequence 2: Morning window only
            #     "start_hour": 8, "start_minute": 30,  # 8:30 AM LOCAL
            #     "end_hour": 10, "end_minute": 30      # 10:30 AM LOCAL
            # },
            1: {  # Sequence 1: Afternoon window only
                "start_hour": 13, "start_minute": 30,  # 1:30 PM LOCAL
                "end_hour": 16, "end_minute": 00      # 4:00 PM LOCAL
            }
        },
        "Food & Beverage Director": {
            # 2: {  # Sequence 2: Morning window only
            #     "start_hour": 9, "start_minute": 30,  # 9:30 AM LOCAL
            #     "end_hour": 11, "end_minute": 30      # 11:30 AM LOCAL
            # },
            1: {  # Sequence 1: Afternoon window only
                "start_hour": 13, "start_minute": 30,  # 1:30 PM LOCAL
                "end_hour": 16, "end_minute": 00      # 4:00 PM LOCAL
            }
        }
    }
    
    # Get time window for the persona and sequence number, defaulting to GM times if not found
    window = time_windows.get(persona, time_windows["General Manager"]).get(sequence_num, time_windows["General Manager"][1])
    
    # Convert LOCAL time to MST by subtracting timezone offset
    start_time = window["start_hour"] + window["start_minute"] / 60
    mst_start = start_time - timezone_offset
    
    logger.debug(f"Selected window (LOCAL): {window['start_hour']}:{window['start_minute']:02d}")
    logger.debug(f"Converted to MST (offset: {timezone_offset}): {int(mst_start)}:{int((mst_start % 1) * 60):02d}")
    
    return {
        "start": mst_start,
        "end": mst_start  # Since we're only using start time, just duplicate it
    }

def get_best_outreach_window(
    persona: str, 
    geography: str, 
    sequence_num: int = 1,
    club_type: str = None, 
    season_data: dict = None,
    timezone_offset: int = 0  # Add timezone_offset parameter
) -> Dict[str, Any]:
    """
    Get the optimal outreach window based on persona and geography.
    """
    logger.debug(f"Getting outreach window for persona: {persona}, geography: {geography}, sequence: {sequence_num}, timezone_offset: {timezone_offset}")
    
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona, sequence_num, timezone_offset)  # Pass timezone_offset
    best_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    logger.debug(f"Calculated base outreach window", extra={
        "persona": persona,
        "geography": geography,
        "sequence": sequence_num,
        "timezone_offset": timezone_offset,
        "best_months": best_months,
        "best_time": best_time,
        "best_days": best_days
    })
    
    return {
        "Best Month": best_months,
        "Best Time": best_time,
        "Best Day": best_days
    }

def calculate_send_date(geography: str, persona: str, state: str, sequence_num: int = 1, season_data: dict = None, day_offset: int = 0) -> datetime:
    try:
        logger.debug(f"\n=== Starting Send Date Calculation ===")
        logger.debug(f"Inputs: geography={geography}, persona={persona}, state={state}, sequence={sequence_num}, day_offset={day_offset}")
        
        # Get timezone offset from MST for this state
        offsets = STATE_OFFSETS.get(state.upper() if state else None)
        logger.debug(f"Found offsets for {state}: {offsets}")
        
        if not offsets:
            logger.warning(f"No offset data for state {state}, using MST")
            timezone_offset = 0
        else:
            # Determine if we're in DST
            dt = datetime.now()
            is_dst = 3 <= dt.month <= 11
            # Invert the offset since we want LOCAL = MST + offset
            timezone_offset = -(offsets['dst'] if is_dst else offsets['std'])
            logger.debug(f"Using timezone offset of {timezone_offset} hours for {state} (DST: {is_dst})")
        
        # Get best outreach window (returns times in MST)
        outreach_window = get_best_outreach_window(
            geography=geography,
            persona=persona,
            sequence_num=sequence_num,
            season_data=season_data,
            timezone_offset=timezone_offset
        )
        
        logger.debug(f"Outreach window result: {outreach_window}")
        
        preferred_time = outreach_window["Best Time"]  # Already in MST
        
        # Use start of window
        start_hour = int(preferred_time["start"])
        start_minutes = int((preferred_time["start"] % 1) * 60)
        
        logger.debug(f"Using MST time: {start_hour}:{start_minutes:02d}")
        
        # Start with today + day_offset
        target_date = datetime.now() + timedelta(days=day_offset)
        target_time = target_date.replace(hour=start_hour, minute=start_minutes, second=0, microsecond=0)
        
        # Find next available preferred day (Tuesday, Wednesday, Thursday)
        while target_time.weekday() not in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
            target_time += timedelta(days=1)
            logger.debug(f"Moved to next preferred day: {target_time}")
        
        logger.debug(f"Final scheduled date (MST): {target_time}")
        return target_time

    except Exception as e:
        logger.error(f"Error calculating send date: {str(e)}", exc_info=True)
        return datetime.now() + timedelta(days=1, hours=10)

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

## scripts\monitor\auto_reply_processor.py
```python
import sys
import os
import re
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Optional
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from scheduling.database import get_db_connection

class AutoReplyProcessor:
    def __init__(self, testing: bool = False):
        """Initialize the auto-reply processor."""
        self.gmail_service = GmailService()
        self.hubspot_service = HubspotService(HUBSPOT_API_KEY)
        self.analyzer = ResponseAnalyzerService()
        self.TESTING = testing
        self.processed_count = 0
        self.AUTO_REPLY_QUERY = """
            (subject:"No longer employed" OR subject:"out of office" OR 
            subject:"automatic reply" OR subject:"auto-reply" OR 
            subject:"automated response" OR subject:"inactive")
            in:inbox newer_than:30d
        """.replace('\n', ' ').strip()

    def extract_new_contact_email(self, message: str) -> Optional[str]:
        """Extract new contact email from message."""
        patterns = [
            r'(?:please\s+)?(?:contact|email|send\s+to|forward\s+to)\s+[\w\s]+:\s*([\w\.-]+@[\w\.-]+\.\w+)',
            r'please\s+(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'(?:new\s+email|new\s+contact|instead\s+email)\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'email\s+([\w\.-]+@[\w\.-]+\.\w+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                return match.group(1)
        return None

    def process_single_reply(self, message_id: str, email: str, subject: str, body: str) -> bool:
        """
        Process a single auto-reply message.
        Returns True if processing was successful, False otherwise.
        """
        try:
            logger.info(f"Processing auto-reply for: {email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body preview: {body[:200]}...")
            success = True

            # Check for out of office first
            if self.analyzer.is_out_of_office(body, subject):
                logger.info("Detected out of office reply - archiving")
                if not self.TESTING:
                    if self.gmail_service.archive_email(message_id):
                        logger.info("✅ Archived out of office message")
                    else:
                        success = False
                        logger.error("❌ Failed to archive out of office message")
                return success

            # Check for employment change
            if self.analyzer.is_employment_change(body, subject):
                logger.info("Detected employment change notification")
                if self.TESTING:
                    logger.info(f"TESTING: Would process employment change for {email}")
                    return True
                # Try to extract new contact email
                new_email = self.extract_new_contact_email(body)
                if new_email:
                    logger.info(f"Found new contact email: {new_email}")
                    # Get contact info for transfer
                    contact = self.hubspot_service.get_contact_by_email(email)
                    if contact:
                        # Create new contact with existing info
                        new_properties = contact.copy()
                        new_properties['email'] = new_email
                        if self.hubspot_service.create_contact(new_properties):
                            logger.info(f"✅ Created new contact: {new_email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to create new contact: {new_email}")

                # Delete old contact
                contact = self.hubspot_service.get_contact_by_email(email)
                if contact:
                    contact_id = contact.get('id')
                    if contact_id:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"✅ Deleted old contact: {email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to delete contact {email}")
                else:
                    logger.info(f"ℹ️ No contact found for {email}")

            # Check for do not contact request
            elif self.analyzer.is_do_not_contact_request(body, subject):
                logger.info("Detected do not contact request")
                if self.TESTING:
                    logger.info(f"TESTING: Would process do not contact request for {email}")
                    return True
                contact = self.hubspot_service.get_contact_by_email(email)
                if contact:
                    contact_id = contact.get('id')
                    if contact_id:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"✅ Deleted contact per request: {email}")
                        else:
                            success = False
                            logger.error(f"❌ Failed to delete contact {email}")
                else:
                    logger.info(f"ℹ️ No contact found for {email}")

            # Check for inactive/bounce
            elif self.analyzer.is_inactive_email(body, subject):
                logger.info("Detected inactive email notification")
                if self.TESTING:
                    logger.info(f"TESTING: Would process inactive email for {email}")
                    return True
                else:
                    notification = {
                        "bounced_email": email,
                        "message_id": message_id
                    }
                    from bounce_processor import BounceProcessor
                    bounce_processor = BounceProcessor(testing=self.TESTING)
                    if bounce_processor.process_single_bounce(notification):
                        logger.info(f"✅ Successfully processed bounce for {email}")
                    else:
                        success = False
                        logger.error(f"❌ Failed to process bounce for {email}")
                    return success

            # If we get here, log why we didn't process it
            logger.info("ℹ️ Message did not match any processing criteria")
            return False

        except Exception as e:
            logger.error(f"❌ Error processing auto-reply for {email}: {str(e)}", exc_info=True)
            return False

    def process_auto_replies(self, target_email: str = None) -> int:
        """
        Process all auto-reply messages, optionally filtered by specific email.
        Returns the number of successfully processed replies.
        """
        logger.info(f"Starting auto-reply processing{' for email: ' + target_email if target_email else ''}")
        self.processed_count = 0

        auto_replies = self.gmail_service.search_messages(self.AUTO_REPLY_QUERY)

        if auto_replies:
            logger.info(f"Found {len(auto_replies)} auto-reply message(s)")

            for message in auto_replies:
                try:
                    message_data = self.gmail_service.get_message(message['id'])
                    if not message_data:
                        continue

                    from_header = self.gmail_service._get_header(message_data, 'from')
                    subject = self.gmail_service._get_header(message_data, 'subject')
                    
                    if not from_header:
                        continue

                    # Extract email from header
                    matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', from_header)
                    if not matches:
                        continue

                    email = matches[0]
                    
                    if target_email and email.lower() != target_email.lower():
                        logger.debug(f"Skipping {email} - not target email {target_email}")
                        continue

                    body = self.gmail_service._get_full_body(message_data)
                    self.process_single_reply(message['id'], email, subject, body)

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue

            if self.processed_count > 0:
                logger.info(f"Successfully processed {self.processed_count} auto-reply message(s)")
            else:
                logger.info("No auto-reply messages were processed")
        else:
            logger.info("No auto-reply messages found")

        return self.processed_count


def main():
    """Main entry point for auto-reply processing."""
    TESTING = False  # Set to False for production
    TARGET_EMAIL = "psanders@rccgolf.com"
    
    processor = AutoReplyProcessor(testing=TESTING)
    if TESTING:
        logger.info("Running in TEST mode - no actual changes will be made")
        logger.info(f"Target Email: {TARGET_EMAIL}")
    
    processed_count = processor.process_auto_replies(target_email=TARGET_EMAIL)
    
    if TESTING:
        logger.info(f"TEST RUN COMPLETE - Would have processed {processed_count} auto-replies")
    else:
        logger.info(f"Processing complete - Successfully processed {processed_count} auto-replies")


if __name__ == "__main__":
    main() 
```

## scripts\monitor\bounce_processor.py
```python
import sys
import os
import re
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Optional
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from scheduling.database import get_db_connection

class BounceProcessor:
    def __init__(self, testing: bool = False):
        """Initialize the bounce processor."""
        self.gmail_service = GmailService()
        self.hubspot_service = HubspotService(HUBSPOT_API_KEY)
        self.TESTING = testing
        self.processed_count = 0
        self.BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'

    def delete_from_database(self, email_address: str) -> bool:
        """
        Delete email records from database.
        Returns True if successful, False otherwise.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Delete from emails table
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email_address,))
            emails_deleted = cursor.rowcount
            
            # Delete from leads table
            cursor.execute("DELETE FROM leads WHERE email = %s", (email_address,))
            leads_deleted = cursor.rowcount
            
            conn.commit()
            logger.info(f"Deleted {emails_deleted} email(s) and {leads_deleted} lead(s) for {email_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from database: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def verify_processing(self, email: str) -> bool:
        """Verify that the contact was properly deleted and emails archived."""
        success = True
        
        # Check if contact still exists in HubSpot
        contact = self.hubspot_service.get_contact_by_email(email)
        if contact:
            logger.error(f"❌ Verification failed: Contact {email} still exists in HubSpot")
            success = False
        else:
            logger.info(f"✅ Verification passed: Contact {email} successfully deleted from HubSpot")
        
        # Check for any remaining emails in inbox
        query = f"from:{email} in:inbox"
        remaining_emails = self.gmail_service.search_messages(query)
        if remaining_emails:
            logger.error(f"❌ Verification failed: Found {len(remaining_emails)} unarchived emails from {email}")
            success = False
        else:
            logger.info(f"✅ Verification passed: No remaining emails from {email} in inbox")
        
        return success

    def process_single_bounce(self, notification: Dict) -> bool:
        """
        Process a single bounce notification.
        Returns True if processing was successful, False otherwise.
        """
        bounced_email = notification.get('bounced_email')
        message_id = notification.get('message_id')
        
        if not bounced_email or not message_id:
            logger.error("Invalid bounce notification - missing email or message ID")
            return False
        
        logger.info(f"Processing bounce notification:")
        logger.info(f"  Email: {bounced_email}")
        logger.info(f"  Message ID: {message_id}")
        
        try:
            success = True
            
            # 1. Delete from database
            if self.TESTING:
                logger.info(f"TESTING: Would delete {bounced_email} from database")
            else:
                if not self.delete_from_database(bounced_email):
                    success = False
                    logger.error(f"Failed to delete {bounced_email} from database")
            
            # 2. Delete from HubSpot
            contact = self.hubspot_service.get_contact_by_email(bounced_email)
            if contact:
                contact_id = contact.get('id')
                if contact_id:
                    if self.TESTING:
                        logger.info(f"TESTING: Would delete contact {bounced_email} from HubSpot")
                    else:
                        if self.hubspot_service.delete_contact(contact_id):
                            logger.info(f"Successfully deleted contact {bounced_email} from HubSpot")
                        else:
                            success = False
                            logger.error(f"Failed to delete contact {bounced_email} from HubSpot")
            else:
                logger.info(f"No HubSpot contact found for {bounced_email}")
            
            # 3. Archive the bounce notification
            if self.TESTING:
                logger.info(f"TESTING: Would archive bounce notification {message_id}")
            else:
                if self.gmail_service.archive_email(message_id):
                    logger.info(f"Successfully archived bounce notification")
                else:
                    success = False
                    logger.error(f"Failed to archive Gmail message {message_id}")
            
            # 4. Verify processing if not in testing mode
            if not self.TESTING and success:
                success = self.verify_processing(bounced_email)
            
            if success:
                self.processed_count += 1
                logger.info(f"✅ Successfully processed bounce for {bounced_email}")
            else:
                logger.error(f"❌ Failed to fully process bounce for {bounced_email}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error processing bounce for {bounced_email}: {str(e)}", exc_info=True)
            return False

    def process_bounces(self, target_email: str = None) -> int:
        """
        Process all bounce notifications, optionally filtered by specific email.
        Returns the number of successfully processed bounces.
        """
        logger.info(f"Starting bounce notification processing{' for email: ' + target_email if target_email else ''}")
        self.processed_count = 0
        
        bounce_notifications = self.gmail_service.get_all_bounce_notifications(inbox_only=True)
        
        if bounce_notifications:
            logger.info(f"Found {len(bounce_notifications)} bounce notification(s)")
            valid_notifications = []
            
            for notification in bounce_notifications:
                email = notification.get('bounced_email')
                if not email:
                    logger.debug("Skipping notification - no email address found")
                    continue
                
                if target_email and email.lower() != target_email.lower():
                    logger.debug(f"Skipping {email} - not target email {target_email}")
                    continue
                
                # Process the bounce notification
                try:
                    logger.info(f"Processing bounce for: {email}")
                    
                    # 1) Delete from SQL database
                    if self.TESTING:
                        logger.info(f"TEST MODE: Would delete {email} from SQL database")
                    else:
                        try:
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email,))
                            cursor.execute("DELETE FROM leads WHERE email = %s", (email,))
                            conn.commit()
                            logger.info(f"Deleted records for {email} from SQL database")
                        except Exception as e:
                            logger.error(f"Error deleting from SQL: {str(e)}")
                        finally:
                            if 'cursor' in locals():
                                cursor.close()
                            if 'conn' in locals():
                                conn.close()

                    # 2) Delete from HubSpot
                    try:
                        hubspot = HubspotService(HUBSPOT_API_KEY)
                        contact = hubspot.get_contact_by_email(email)
                        if contact:
                            contact_id = contact.get('id')
                            if contact_id:
                                if self.TESTING:
                                    logger.info(f"TEST MODE: Would delete contact {email} from HubSpot")
                                else:
                                    if hubspot.delete_contact(contact_id):
                                        logger.info(f"Successfully deleted contact {email} from HubSpot")
                                    else:
                                        logger.error(f"Failed to delete contact {email} from HubSpot")
                            else:
                                logger.warning(f"Contact found but no ID for {email}")
                        else:
                            logger.warning(f"No contact found in HubSpot for {email}")
                    except Exception as e:
                        logger.error(f"Error processing HubSpot deletion: {str(e)}")

                    # 3) Archive the bounce notification
                    if self.TESTING:
                        logger.info(f"TEST MODE: Would archive bounce notification for {email}")
                    else:
                        if self.gmail_service.archive_email(notification['message_id']):
                            logger.info(f"Successfully archived bounce notification for {email}")
                        else:
                            logger.error(f"Failed to archive Gmail message for {email}")

                    self.processed_count += 1
                    logger.info(f"Successfully processed bounce for {email}")
                    
                except Exception as e:
                    logger.error(f"Error processing bounce notification for {email}: {str(e)}")
                    continue

            if self.processed_count > 0:
                logger.info(f"Successfully processed {self.processed_count} bounce notification(s)")
            else:
                logger.info("No bounce notifications were processed")
        else:
            logger.info("No bounce notifications found")
        
        return self.processed_count


def main():
    """Main entry point for bounce processing."""
    TESTING = False  # Set to False for production
    TARGET_EMAIL = "psanders@rccgolf.com"
    
    processor = BounceProcessor(testing=TESTING)
    if TESTING:
        logger.info("Running in TEST mode - no actual changes will be made")
        logger.info(f"Target Email: {TARGET_EMAIL}")
    
    processed_count = processor.process_bounces(target_email=TARGET_EMAIL)
    
    if TESTING:
        logger.info(f"TEST RUN COMPLETE - Would have processed {processed_count} valid bounce(s)")
    else:
        logger.info(f"Processing complete - Successfully processed {processed_count} bounce(s)")


if __name__ == "__main__":
    main() 
```

## scripts\monitor\monitor_email_review_status.py
```python
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from datetime import datetime
import pytz
from utils.gmail_integration import get_gmail_service
from utils.logging_setup import logger
from scheduling.database import get_db_connection
import base64

###############################################################################
#                           CONSTANTS
###############################################################################

TO_REVIEW_LABEL = "to_review"
REVIEWED_LABEL = "reviewed"
SQL_DRAFT_STATUS = "draft"
SQL_REVIEWED_STATUS = "reviewed"

###############################################################################
#                           GMAIL INTEGRATION
###############################################################################

def get_draft_emails() -> list:
    """
    Retrieve all emails with status='draft'.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email_id,
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
                 WHERE status = ?
            """, (SQL_DRAFT_STATUS,))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                record = {
                    'email_id': row[0],
                    'lead_id': row[1],
                    'name': row[2],
                    'email_address': row[3],
                    'sequence_num': row[4],
                    'body': row[5],
                    'scheduled_send_date': str(row[6]) if row[6] else None,
                    'actual_send_date': str(row[7]) if row[7] else None,
                    'created_at': str(row[8]) if row[8] else None,
                    'status': row[9],
                    'draft_id': row[10],
                    'gmail_id': row[11]
                }
                results.append(record)
            return results

    except Exception as e:
        logger.error(f"Error retrieving draft emails: {str(e)}", exc_info=True)
        return []

def get_gmail_draft_by_id(service, draft_id: str) -> dict:
    """Retrieve a specific Gmail draft by its draftId."""
    try:
        draft = service.users().drafts().get(userId="me", id=draft_id).execute()
        return draft.get("message", {})
    except Exception as e:
        logger.error(f"Error fetching draft {draft_id}: {str(e)}")
        return {}

def get_draft_body(message: dict) -> str:
    """Extract the body text from a Gmail draft message."""
    try:
        body = ""
        if 'payload' in message and 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        elif 'payload' in message and 'body' in message['payload']:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
        
        if not body:
            logger.error("No body content found in message")
            return ""
            
        # Log the first few characters to help with debugging
        preview = body[:100] + "..." if len(body) > 100 else body
        logger.debug(f"Retrieved draft body preview: {preview}")
        
        # Check for template placeholders
        if "[firstname]" in body or "{{" in body or "}}" in body:
            logger.warning("Found template placeholders in draft body - draft may not be properly personalized")
            
        return body
        
    except Exception as e:
        logger.error(f"Error extracting draft body: {str(e)}")
        return ""

def translate_label_ids_to_names(service, label_ids: list) -> list:
    """Convert Gmail label IDs to their corresponding names."""
    try:
        labels_response = service.users().labels().list(userId='me').execute()
        all_labels = labels_response.get('labels', [])
        id_to_name = {lbl["id"]: lbl["name"] for lbl in all_labels}

        label_names = []
        for lid in label_ids:
            label_names.append(id_to_name.get(lid, lid))
        return label_names
    except Exception as e:
        logger.error(f"Error translating label IDs: {str(e)}")
        return label_ids

def update_email_status(email_id: int, new_status: str, body: str = None):
    """
    Update the emails table with the new status and optionally update the body.
    Similar to update_email_record in check_for_sent_emails.py
    """
    try:
        if body:
            sql = """
                UPDATE emails
                   SET status = ?,
                       body = ?
                 WHERE email_id = ?
            """
            params = (new_status, body, email_id)
        else:
            sql = """
                UPDATE emails
                   SET status = ?
                 WHERE email_id = ?
            """
            params = (new_status, email_id)
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Email ID {email_id} updated: status='{new_status}'" + 
                       (", body updated" if body else ""))
    except Exception as e:
        logger.error(f"Error updating email ID {email_id}: {str(e)}", exc_info=True)

def delete_orphaned_draft_records():
    """
    Delete records from the emails table that have status='draft' but their
    corresponding Gmail drafts no longer exist.
    """
    try:
        # Get Gmail service
        gmail_service = get_gmail_service()
        if not gmail_service:
            logger.error("Could not initialize Gmail service.")
            return

        # Get all draft records
        drafts = get_draft_emails()
        if not drafts:
            logger.info("No draft emails found to check.")
            return

        logger.info(f"Checking {len(drafts)} draft records for orphaned entries")

        # Get list of all Gmail drafts
        try:
            gmail_drafts = gmail_service.users().drafts().list(userId='me').execute()
            existing_draft_ids = {d['id'] for d in gmail_drafts.get('drafts', [])}
        except Exception as e:
            logger.error(f"Error fetching Gmail drafts: {str(e)}")
            return

        # Track which records to delete
        orphaned_records = []
        for record in drafts:
            if not record['draft_id'] or record['draft_id'] not in existing_draft_ids:
                orphaned_records.append(record['email_id'])
                logger.debug(f"Found orphaned record: email_id={record['email_id']}, draft_id={record['draft_id']}")

        if not orphaned_records:
            logger.info("No orphaned draft records found.")
            return

        # Delete orphaned records
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(orphaned_records))
                cursor.execute(f"""
                    DELETE FROM emails
                     WHERE email_id IN ({placeholders})
                       AND status = ?
                """, (*orphaned_records, SQL_DRAFT_STATUS))
                conn.commit()
                logger.info(f"Deleted {cursor.rowcount} orphaned draft records")
        except Exception as e:
            logger.error(f"Error deleting orphaned records: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in delete_orphaned_draft_records: {str(e)}", exc_info=True)

###############################################################################
#                               MAIN PROCESS
###############################################################################

def main():
    """Check Gmail drafts for labels and update SQL status accordingly."""
    logger.info("=== Starting Gmail label check process ===")

    # First, clean up any orphaned draft records
    delete_orphaned_draft_records()

    # Get all draft emails from SQL
    pending = get_draft_emails()
    if not pending:
        logger.info("No draft emails found. Exiting.")
        return

    logger.info(f"Found {len(pending)} draft emails to process")

    # Get Gmail service
    gmail_service = get_gmail_service()
    if not gmail_service:
        logger.error("Could not initialize Gmail service.")
        return

    # Process each record
    for record in pending:
        email_id = record['email_id']
        draft_id = record['draft_id']

        logger.info(f"\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Draft ID: {draft_id}")

        if not draft_id:
            logger.warning(f"No draft_id found for email_id={email_id}. Skipping.")
            continue

        # Get the Gmail draft and its labels
        message = get_gmail_draft_by_id(gmail_service, draft_id)
        if not message:
            logger.error(f"Failed to retrieve Gmail draft for draft_id={draft_id}")
            continue

        # Get current labels
        current_labels = message.get("labelIds", [])
        if not current_labels:
            logger.info(f"No labels found for draft_id={draft_id}. Skipping.")
            continue

        # Convert label IDs to names
        label_names = translate_label_ids_to_names(gmail_service, current_labels)
        logger.debug(f"Draft {draft_id} has labels: {label_names}")

        # Check for reviewed label and update SQL status
        if REVIEWED_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} marked as '{REVIEWED_LABEL}'. Updating SQL status and body.")
            # Get the draft body text
            body = get_draft_body(message)
            if body:
                logger.debug(f"Retrieved body text ({len(body)} chars) for draft_id={draft_id}")
                # Update both status and body
                update_email_status(email_id, SQL_REVIEWED_STATUS, body)
            else:
                logger.warning(f"Could not retrieve body text for draft_id={draft_id}")
                # Update only status if no body could be retrieved
                update_email_status(email_id, SQL_REVIEWED_STATUS)
        elif TO_REVIEW_LABEL.lower() in [ln.lower() for ln in label_names]:
            logger.info(f"Draft {draft_id} still labeled '{TO_REVIEW_LABEL}'. No action needed.")
        else:
            logger.info(f"Draft {draft_id} has no relevant labels: {label_names}")

    logger.info("\n=== Completed label check process ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()

```

## scripts\monitor\monitor_email_sent_status.py
```python
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import base64
from datetime import datetime
import pytz
from email.utils import parsedate_to_datetime
from thefuzz import fuzz

# Your existing helper imports
from utils.gmail_integration import get_gmail_service, search_messages
from utils.logging_setup import logger
from scheduling.database import get_db_connection
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def parse_sql_datetime(date_str: str) -> datetime:
    """
    Parse a datetime from your SQL table based on typical SQL formats:
      - 'YYYY-MM-DD HH:MM:SS'
      - 'YYYY-MM-DD HH:MM:SS.fff'
    Returns a UTC datetime object or None if parsing fails.
    """
    if not date_str:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Assuming DB times are UTC-naive or local; attach UTC as needed:
            dt_utc = dt.replace(tzinfo=pytz.UTC)
            return dt_utc
        except ValueError:
            continue

    logger.error(f"Unable to parse SQL datetime '{date_str}' with known formats.")
    return None


def parse_gmail_datetime(gmail_date_header: str) -> datetime:
    """
    Parse the Gmail 'Date' header (e.g. 'Wed, 15 Jan 2025 13:01:22 -0500').
    Convert it to a datetime in UTC.
    """
    if not gmail_date_header:
        return None
    try:
        dt = parsedate_to_datetime(gmail_date_header)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception as e:
        logger.error(f"Failed to parse Gmail date '{gmail_date_header}': {str(e)}", exc_info=True)
        return None


def is_within_24_hours(dt1: datetime, dt2: datetime) -> bool:
    """
    Return True if dt1 and dt2 are within 24 hours of each other.
    """
    if not dt1 or not dt2:
        return False
    diff = abs((dt1 - dt2).total_seconds())
    return diff <= 86400  # 86400 seconds = 24 hours


def is_within_days(dt1: datetime, dt2: datetime, days: int = 7) -> bool:
    """Return True if dt1 and dt2 are within X days of each other."""
    if not dt1 or not dt2:
        return False
    diff = abs((dt1 - dt2).total_seconds())
    return diff <= (days * 24 * 60 * 60)  # Convert days to seconds


###############################################################################
#                        DATABASE-RELATED FUNCTIONS
###############################################################################

def get_pending_emails(lead_id: int) -> list:
    """
    Retrieve emails where:
      - lead_id = ?
      - status IN ('pending','draft','sent')
      
    The table columns are:
      email_id, lead_id, name, email_address, sequence_num,
      body, scheduled_send_date, actual_send_date, created_at,
      status, draft_id, gmail_id
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email_id,
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
                 WHERE lead_id = ?
                   AND status IN ('reviewed','draft','sent') 
            """, (lead_id,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            record = {
                'email_id': row[0],
                'lead_id': row[1],
                'name': row[2],
                'email_address': row[3],
                'sequence_num': row[4],
                'body': row[5],
                'scheduled_send_date': str(row[6]) if row[6] else None,
                'actual_send_date': str(row[7]) if row[7] else None,
                'created_at': str(row[8]) if row[8] else None,
                'status': row[9],
                'draft_id': row[10],
                'gmail_id': row[11]
            }
            results.append(record)
        return results

    except Exception as e:
        logger.error(f"Error retrieving pending emails: {str(e)}", exc_info=True)
        return []


def update_email_record(email_id: int, gmail_id: str, actual_send_date_utc: datetime, body: str = None):
    """
    Update the emails table with the matched Gmail ID, set actual_send_date,
    mark status='sent', and optionally update the body.
    """
    try:
        actual_send_date_str = actual_send_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        
        if body:
            sql = """
                UPDATE emails
                   SET gmail_id = ?,
                       actual_send_date = ?,
                       status = 'sent',
                       body = ?
                 WHERE email_id = ?
            """
            params = (gmail_id, actual_send_date_str, body, email_id)
        else:
            sql = """
                UPDATE emails
                   SET gmail_id = ?,
                       actual_send_date = ?,
                       status = 'sent'
                 WHERE email_id = ?
            """
            params = (gmail_id, actual_send_date_str, email_id)
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Email ID {email_id} updated: gmail_id={gmail_id}, status='sent'.")
    except Exception as e:
        logger.error(f"Error updating email ID {email_id}: {str(e)}", exc_info=True)


def update_email_address(email_id: int, email_address: str):
    """Update the email_address field in the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE emails
                   SET email_address = ?
                 WHERE email_id = ?
            """, (email_address, email_id))
            conn.commit()
            logger.info(f"Updated email_id={email_id} with email_address={email_address}")
    except Exception as e:
        logger.error(f"Error updating email address for ID {email_id}: {str(e)}", exc_info=True)


###############################################################################
#                           GMAIL INTEGRATION
###############################################################################

def search_gmail_for_messages(to_address: str, subject: str, scheduled_dt: datetime = None) -> list:
    """
    Search Gmail for messages in 'sent' folder that match the criteria.
    If scheduled_dt is provided, only return messages within 7 days.
    """
    service = get_gmail_service()
    
    logger.info(f"Searching Gmail with to_address={to_address}")
    logger.info(f"Searching Gmail with subject={subject}")
    logger.info(f"Scheduled date: {scheduled_dt}")
    
    # Modify query to be more lenient
    subject_quoted = subject.replace('"', '\\"')  # Escape any quotes in subject
    query = f'in:sent to:{to_address}'
    
    logger.info(f"Gmail search query: {query}")
    messages = search_messages(query)
    
    if messages:
        logger.info(f"Found {len(messages)} Gmail messages matching query: {query}")
        
        # Filter messages by date if scheduled_dt is provided
        if scheduled_dt:
            filtered_messages = []
            for msg in messages:
                details = get_email_details(msg)
                sent_date = details.get('date_parsed')
                if sent_date and is_within_days(sent_date, scheduled_dt):
                    filtered_messages.append(msg)
                    logger.info(f"Found matching message: Subject='{details.get('subject')}', "
                              f"To={details.get('to')}, Date={details.get('date_raw')}")
            
            logger.info(f"Found {len(filtered_messages)} messages within 7 days of scheduled date")
            return filtered_messages
        
        return messages
    else:
        logger.info("No messages found with this query")
        return []


def get_email_details(gmail_message: dict) -> dict:
    """Extract details including body from a Gmail message."""
    try:
        service = get_gmail_service()
        full_message = service.users().messages().get(
            userId='me', 
            id=gmail_message['id'], 
            format='full'
        ).execute()
        
        payload = full_message.get('payload', {})
        headers = payload.get('headers', [])
        
        def find_header(name: str):
            return next((h['value'] for h in headers if h['name'].lower() == name.lower()), None)
        
        # Extract body from multipart message
        def get_body_from_parts(parts):
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    if part.get('body', {}).get('data'):
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part.get('parts'):  # Handle nested parts
                    body = get_body_from_parts(part['parts'])
                    if body:
                        return body
            return None

        # Get body content
        body = None
        if payload.get('mimeType') == 'text/plain':
            if payload.get('body', {}).get('data'):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif payload.get('mimeType', '').startswith('multipart/'):
            body = get_body_from_parts(payload.get('parts', []))

        # Clean up the body - remove signature
        if body:
            body = body.split('Time zone: Eastern Time')[0].strip()
            logger.info("=== Extracted Email Body ===")
            logger.info(body)
            logger.info("=== End of Email Body ===")
        
        date_str = find_header('Date')
        parsed_dt = parse_gmail_datetime(date_str)

        details = {
            'gmail_id':  full_message['id'],
            'thread_id': full_message['threadId'],
            'subject':   find_header('Subject'),
            'from':      find_header('From'),
            'to':        find_header('To'),
            'cc':        find_header('Cc'),
            'bcc':       find_header('Bcc'),
            'date_raw':  date_str,
            'date_parsed': parsed_dt,
            'body':      body
        }
        
        return details
        
    except Exception as e:
        logger.error(f"Error getting email details: {str(e)}", exc_info=True)
        return {}


###############################################################################
#                                MAIN LOGIC
###############################################################################

def get_draft_emails(email_id: int = None) -> list:
    """Retrieve specific draft email or all draft emails that need processing."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if email_id:
                cursor.execute("""
                    SELECT email_id,
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
                     WHERE email_id = ?
                       AND status IN ('draft','reviewed')
                """, (email_id,))
            else:
                cursor.execute("""
                    SELECT email_id,
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
                     WHERE status IN ('draft','reviewed')
                  ORDER BY email_id DESC
                """)
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'email_id': row[0],
                    'lead_id': row[1],
                    'name': row[2],
                    'email_address': row[3],
                    'sequence_num': row[4],
                    'body': row[5],
                    'scheduled_send_date': str(row[6]) if row[6] else None,
                    'actual_send_date': str(row[7]) if row[7] else None,
                    'created_at': str(row[8]) if row[8] else None,
                    'status': row[9],
                    'draft_id': row[10],
                    'gmail_id': row[11]
                })
            return results
    except Exception as e:
        logger.error(f"Error retrieving draft emails: {str(e)}", exc_info=True)
        return []

def main():
    """Process all draft emails."""
    logger.info("=== Starting Gmail match process ===")

    # Get all draft emails (no specific email_id)
    pending = get_draft_emails()
    if not pending:
        logger.info("No draft emails found. Exiting.")
        return

    logger.info(f"Found {len(pending)} draft emails to process")

    # Initialize HubSpot service
    hubspot = HubspotService(HUBSPOT_API_KEY)

    # Process each record
    for record in pending:
        email_id = record['email_id']
        lead_id = record['lead_id']
        to_address = record['email_address']
        scheduled_dt = parse_sql_datetime(record['scheduled_send_date'])
        created_dt = parse_sql_datetime(record['created_at'])

        logger.info("\n=== Processing Record ===")
        logger.info(f"Email ID: {email_id}")
        logger.info(f"Lead ID: {lead_id}")
        logger.info(f"Current email address: {to_address}")
        logger.info(f"Scheduled: {scheduled_dt}")
        logger.info(f"Created: {created_dt}")

        # If no email address, try to get it from HubSpot
        if not to_address:
            try:
                logger.info(f"Fetching contact data from HubSpot for lead_id: {lead_id}")
                contact_props = hubspot.get_contact_properties(str(lead_id))
                
                if contact_props and 'email' in contact_props:
                    email_address = contact_props['email']
                    if email_address:
                        logger.info(f"Found email address in HubSpot: {email_address}")
                        update_email_address(email_id, email_address)
                        to_address = email_address
                    else:
                        logger.warning(f"No email found in HubSpot properties for lead_id={lead_id}")
                        continue
                else:
                    logger.warning(f"No contact data found in HubSpot for lead_id={lead_id}")
                    continue
            except Exception as e:
                logger.error(f"Error fetching HubSpot data: {str(e)}", exc_info=True)
                continue

        # Now proceed with Gmail search - without subject
        messages = search_gmail_for_messages(to_address, "", scheduled_dt)
        if not messages:
            logger.info(f"No matching Gmail messages found for email_id={email_id}.")
            continue

        # Try to find a valid match
        matched_gmail = None
        for msg in messages:
            details = get_email_details(msg)
            dt_sent = details.get('date_parsed')
            if not dt_sent:
                continue

            logger.info(f"\nChecking message sent at {dt_sent}:")
            logger.info(f"Within 7 days of scheduled ({scheduled_dt}): {is_within_days(dt_sent, scheduled_dt)}")
            
            if is_within_days(dt_sent, scheduled_dt):
                matched_gmail = details
                break

        if matched_gmail:
            logger.info(f"\nMatched Gmail ID={matched_gmail['gmail_id']} for email_id={email_id}")
            logger.info(f"Sent date: {matched_gmail['date_parsed']}")
            logger.info(f"To: {matched_gmail['to']}")
            logger.info("Updating database record...")
            
            update_email_record(
                email_id=email_id,
                gmail_id=matched_gmail['gmail_id'],
                actual_send_date_utc=matched_gmail['date_parsed'],
                body=matched_gmail.get('body', '')
            )
        else:
            logger.info(f"\nNo valid match found for email_id={email_id}.")

    logger.info("\n=== Completed email matching process. ===")
    logger.info(f"Processed {len(pending)} draft emails.")

if __name__ == "__main__":
    main()

```

## scripts\monitor\review_email_responses.py
```python
import sys
import os
import re
from datetime import datetime
# Add this at the top of the file, before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List
from services.response_analyzer_service import ResponseAnalyzerService
from services.hubspot_service import HubspotService
from services.gmail_service import GmailService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY
from services.data_gatherer_service import DataGathererService
from scheduling.database import get_db_connection
from utils.xai_integration import analyze_auto_reply, analyze_employment_change

# Define queries for different types of notifications
BOUNCE_QUERY = 'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" in:inbox'
AUTO_REPLY_QUERY = f"""
    (subject:"No longer employed" OR subject:"out of office" OR subject:"automatic reply")
    in:inbox
""".replace('\n', ' ').strip()

TESTING = True  # Set to False for production
REGULAR_RESPONSE_QUERY = '-in:trash -in:spam'  # Search everywhere except trash and spam


def get_first_50_words(text: str) -> str:
    """Get first 50 words of a text string."""
    if not text:
        return "No content"
    words = [w for w in text.split() if w.strip()]
    snippet = ' '.join(words[:50])
    return snippet + ('...' if len(words) > 50 else '')


def process_invalid_email(email: str, analyzer_result: Dict) -> None:
    """
    Process an invalid email by removing the contact from HubSpot.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get the contact
        contact = hubspot.get_contact_by_email(email)
        if not contact:
            logger.info(f"Contact not found in HubSpot for email: {email}")
            return

        contact_id = contact.get('id')
        if not contact_id:
            logger.error(f"Contact found but missing ID for email: {email}")
            return

        # Archive the contact in HubSpot
        try:
            hubspot.archive_contact(contact_id)
            logger.info(
                f"Successfully archived contact {email} (ID: {contact_id}) due to invalid email: {analyzer_result['message']}"
            )
        except Exception as e:
            logger.error(f"Failed to archive contact {email}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing invalid email {email}: {str(e)}")


def process_bounced_email(email: str, gmail_id: str, analyzer_result: Dict) -> None:
    """
    Process a bounced email by deleting from SQL, HubSpot, and archiving the email.
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        gmail = GmailService()
        
        # 1) Delete from SQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM emails WHERE email_address = %s", (email,))
            cursor.execute("DELETE FROM leads WHERE email = %s", (email,))
            conn.commit()
            logger.info(f"Deleted records for {email} from SQL database")
        except Exception as e:
            logger.error(f"Error deleting from SQL: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

        # 2) Delete from HubSpot
        contact = hubspot.get_contact_by_email(email)
        if contact:
            contact_id = contact.get('id')
            if contact_id and hubspot.delete_contact(contact_id):
                logger.info(f"Contact {email} deleted from HubSpot")
        
        # 3) Archive bounce notification (uncomment if you'd like to truly archive)
        # if gmail.archive_email(gmail_id):
        #     logger.info(f"Bounce notification archived in Gmail")
        print(f"Would have archived bounce notification with ID: {gmail_id}")

    except Exception as e:
        logger.error(f"Error processing bounced email {email}: {str(e)}")


def is_out_of_office(message: str, subject: str) -> bool:
    """Check if a message is an out-of-office response."""
    ooo_phrases = [
        "out of office",
        "automatic reply",
        "away from",
        "will be out",
        "on vacation",
        "annual leave",
        "business trip",
        "return to the office",
        "be back",
        "currently away"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    return any(phrase in message_lower or phrase in subject_lower for phrase in ooo_phrases)


def is_no_longer_employed(message: str, subject: str) -> bool:
    """Check if message indicates person is no longer employed."""
    employment_phrases = [
        "no longer employed",
        "no longer with",
        "is not employed",
        "has left",
        "no longer works",
        "no longer at",
        "no longer associated",
        "please remove",
        "has departed"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    return any(phrase in message_lower or phrase in subject_lower for phrase in employment_phrases)


def extract_new_contact_email(message: str) -> str:
    """Extract new contact email from message."""
    import re
    patterns = [
        r'please\s+(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:contact|email|send\s+to|forward\s+to)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'(?:new\s+email|new\s+contact|instead\s+email)\s+([\w\.-]+@[\w\.-]+\.\w+)',
        r'email\s+([\w\.-]+@[\w\.-]+\.\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1)
    return None


def process_employment_change(email: str, message_id: str, gmail_service: GmailService) -> None:
    """Process an employment change notification."""
    try:
        # Get message details
        message_data = gmail_service.get_message(message_id)
        subject = gmail_service._get_header(message_data, 'subject')
        body = gmail_service._get_full_body(message_data)
        
        # Analyze the message
        contact_info = analyze_employment_change(body, subject)
        
        if contact_info and contact_info.get('new_email'):
            hubspot = HubspotService(HUBSPOT_API_KEY)
            
            # Create new contact properties
            new_properties = {
                'email': contact_info['new_email'],
                'firstname': (
                    contact_info['new_contact'].split()[0]
                    if contact_info['new_contact'] != 'Unknown' else ''
                ),
                'lastname': (
                    ' '.join(contact_info['new_contact'].split()[1:])
                    if contact_info['new_contact'] != 'Unknown' else ''
                ),
                'company': contact_info['company'] if contact_info['company'] != 'Unknown' else '',
                'jobtitle': contact_info['new_title'] if contact_info['new_title'] != 'Unknown' else '',
                'phone': contact_info['phone'] if contact_info['phone'] != 'Unknown' else ''
            }
            
            new_contact = hubspot.create_contact(new_properties)
            if new_contact:
                logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
            
            # Try to archive the message
            if gmail_service.archive_email(message_id):
                logger.info("Archived employment change notification")
            
    except Exception as e:
        logger.error(f"Error processing employment change: {str(e)}", exc_info=True)


def is_inactive_email(message: str, subject: str) -> bool:
    """Check if message indicates email is inactive."""
    inactive_phrases = [
        "email is now inactive",
        "email is inactive",
        "no longer active",
        "this address is inactive",
        "email account is inactive",
        "account is inactive",
        "inactive email",
        "email has been deactivated"
    ]
    
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in inactive_phrases)


def is_employment_change(body: str, subject: str) -> bool:
    """Check if the message indicates an employment change."""
    employment_phrases = [
        "no longer with",
        "no longer employed",
        "is no longer",
        "has left",
        "no longer works",
        "no longer at",
        "has departed",
        "is not with"
    ]
    
    message_lower = (body + " " + subject).lower()
    return any(phrase in message_lower for phrase in employment_phrases)


def is_do_not_contact_request(message: str, subject: str) -> bool:
    """Check if message is a request to not be contacted."""
    dnc_phrases = [
        "don't contact",
        "do not contact",
        "stop contacting",
        "please remove",
        "unsubscribe",
        "take me off",
        "remove me from",
        "no thanks",
        "not interested"
    ]
    
    message_lower = message.lower()
    subject_lower = subject.lower()
    return any(phrase in message_lower or phrase in subject_lower for phrase in dnc_phrases)


def delete_email_from_database(email_address):
    """Helper function to delete email records from database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        delete_query = "DELETE FROM dbo.emails WHERE email_address = ?"
        cursor.execute(delete_query, (email_address,))
        conn.commit()
        logger.info(f"Deleted all records for {email_address} from emails table")
    except Exception as e:
        logger.error(f"Error deleting from SQL: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def process_bounce_notification(notification, gmail_service):
    """Process a single bounce notification."""
    bounced_email = notification['bounced_email']
    message_id = notification['message_id']
    logger.info(f"Processing bounce notification - Email: {bounced_email}, Message ID: {message_id}")
    
    if TESTING:
        print(f"Would have deleted {bounced_email} from SQL database")
    else:
        delete_email_from_database(bounced_email)
    
    try:
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        contact = hubspot.get_contact_by_email(bounced_email)
        
        if contact:
            contact_id = contact.get('id')
            if contact_id:
                logger.info(f"Attempting to delete contact {bounced_email} from HubSpot")
                if TESTING:
                    print(f"Would have deleted contact {bounced_email} with ID {contact_id}")
                else:
                    if hubspot.delete_contact(contact_id):
                        logger.info(f"Successfully deleted contact {bounced_email} from HubSpot")
                    else:
                        logger.error(f"Failed to delete contact {bounced_email} from HubSpot")
            else:
                logger.warning(f"Contact found but no ID for {bounced_email}")
        else:
            logger.warning(f"No contact found in HubSpot for {bounced_email}")
        
        if TESTING:
            print(f"Would have archived bounce notification with ID: {message_id}")
        else:
            if gmail_service.archive_email(message_id):
                logger.info(f"Successfully archived bounce notification for {bounced_email}")
            else:
                logger.error(f"Failed to archive Gmail message {message_id}")
            
    except Exception as e:
        logger.error(f"Error processing bounce notification for {bounced_email}: {str(e)}")
        raise


def verify_processing(email: str, gmail_service: GmailService, hubspot_service: HubspotService) -> bool:
    """Verify that the contact was properly deleted and emails archived."""
    if TESTING:
        print(f"TESTING: Skipping verification for {email}")
        return True
        
    success = True
    
    # Check if contact still exists in HubSpot
    contact = hubspot_service.get_contact_by_email(email)
    if contact:
        logger.error(f"❌ Verification failed: Contact {email} still exists in HubSpot")
        success = False
    else:
        logger.info(f"✅ Verification passed: Contact {email} successfully deleted from HubSpot")
    
    # Check for any remaining emails in inbox
    query = f"from:{email} in:inbox"
    remaining_emails = gmail_service.search_messages(query)
    if remaining_emails:
        logger.error(f"❌ Verification failed: Found {len(remaining_emails)} unarchived emails from {email}")
        success = False
    else:
        logger.info(f"✅ Verification passed: No remaining emails from {email} in inbox")
    
    return success


def mark_lead_as_dq(email_address, reason):
    """Mark a lead as disqualified in the database."""
    logger.info(f"Marking {email_address} as DQ. Reason: {reason}")
    # Add your database update logic here


def check_company_responses(email: str, sent_date: str) -> bool:
    """
    Check if there are any responses from the same company domain after the sent date.
    
    - Now logs each inbound email from any contact with the same domain.
    - Returns True if at least one such email is found after the 'sent_date'.
    """
    try:
        domain = email.split('@')[-1]
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Get all contacts with the same domain
        domain_contacts = hubspot.get_contacts_by_company_domain(domain)
        if not domain_contacts:
            logger.info(f"No domain contacts found for {domain}")
            return False
            
        logger.info(f"Found {len(domain_contacts)} contact(s) for domain '{domain}'. Checking inbound emails...")
        
        # Convert sent_date string to datetime for comparison
        sent_datetime = datetime.strptime(sent_date, '%Y-%m-%d %H:%M:%S')
        
        any_responses = False
        
        for c in domain_contacts:
            contact_id = c.get('id')
            if not contact_id:
                continue
            
            contact_email = c.get('properties', {}).get('email', 'Unknown')
            emails = hubspot.get_all_emails_for_contact(contact_id)
            # Filter for inbound/incoming emails after sent_date
            incoming_emails = [
                e for e in emails
                if e.get('direction') in ['INBOUND', 'INCOMING', 'INCOMING_EMAIL']
                and int(e.get('timestamp', 0)) / 1000 > sent_datetime.timestamp()
            ]
            
            for e in incoming_emails:
                # Log each inbound email from the same company domain
                msg_time = datetime.fromtimestamp(int(e.get('timestamp', 0))/1000).strftime('%Y-%m-%d %H:%M:%S')
                snippet = get_first_50_words(e.get('body_text', '') or e.get('text', '') or '')
                logger.info(
                    f"FLAGGED: Response from domain '{domain}' => "
                    f"Contact: {contact_email}, Timestamp: {msg_time}, Snippet: '{snippet}'"
                )
                any_responses = True
        
        return any_responses

    except Exception as e:
        logger.error(f"Error checking company responses for {email}: {str(e)}", exc_info=True)
        return False


def process_email_response(message_id: str, email: str, subject: str, body: str, gmail_service: GmailService) -> None:
    """Process an email response."""
    try:
        logger.info(f"Starting to process email response from {email}")
        message_data = gmail_service.get_message(message_id)
        sent_date = datetime.fromtimestamp(
            int(message_data.get('internalDate', 0)) / 1000
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Message date: {sent_date}")
        
        # ** Updated call: check and flag domain responses
        if check_company_responses(email, sent_date):
            logger.info(f"Found response(s) from the same company domain for {email} after {sent_date}")
            # Example action: update HubSpot contact with a 'company_responded' flag
            hubspot = HubspotService(HUBSPOT_API_KEY)
            contact = hubspot.get_contact_by_email(email)
            if contact:
                properties = {
                    'company_responded': 'true',
                    'last_company_response_date': sent_date
                }
                hubspot.update_contact(contact.get('id'), properties)
                logger.info(f"Updated contact {email} to mark company response")

        # Continue with existing logic...
        from_header = gmail_service._get_header(message_data, 'from')
        full_name = from_header.split('<')[0].strip()
        
        hubspot = HubspotService(HUBSPOT_API_KEY)
        logger.info(f"Processing response for email: {email}")
        
        # Get contact directly
        contact = hubspot.get_contact_by_email(email)
        
        # Check if it's a do-not-contact request
        if is_do_not_contact_request(body, subject):
            logger.info(f"Detected do-not-contact request from {email}")
            if contact:
                company_name = contact.get('properties', {}).get('company', '')
                if hubspot.mark_do_not_contact(email, company_name):
                    logger.info(f"Successfully marked {email} as do-not-contact")
                    delete_email_from_database(email)
                    logger.info(f"Removed {email} from SQL database")
                    if gmail_service.archive_email(message_id):
                        logger.info(f"Archived do-not-contact request email")
            return
        
        # Check if it's an employment change notification
        if is_employment_change(body, subject):
            logger.info(f"Detected employment change notification for {email}")
            process_employment_change(email, message_id, gmail_service)
            return
            
        if contact:
            contact_id = contact.get('id')
            contact_email = contact.get('properties', {}).get('email', '')
            logger.info(f"Found contact in HubSpot: {contact_email} (ID: {contact_id})")
            
            # Check if it's an auto-reply
            if "automatic reply" in subject.lower():
                logger.info("Detected auto-reply, sending to xAI for analysis...")
                contact_info = analyze_auto_reply(body, subject)
                
                if contact_info and contact_info.get('new_email'):
                    if TESTING:
                        print("\nAuto-reply Analysis Results:")
                        print(f"Original Contact: {email}")
                        print(f"Analysis Results: {contact_info}")
                        return
                    
                    new_contact = hubspot.create_contact({
                        'email': contact_info['new_email'],
                        'firstname': (
                            contact_info['new_contact'].split()[0]
                            if contact_info['new_contact'] != 'Unknown'
                            else contact.get('properties', {}).get('firstname', '')
                        ),
                        'lastname': (
                            ' '.join(contact_info['new_contact'].split()[1:])
                            if contact_info['new_contact'] != 'Unknown'
                            else contact.get('properties', {}).get('lastname', '')
                        ),
                        'company': (
                            contact_info['company']
                            if contact_info['company'] != 'Unknown'
                            else contact.get('properties', {}).get('company', '')
                        ),
                        'jobtitle': (
                            contact_info['new_title']
                            if contact_info['new_title'] != 'Unknown'
                            else contact.get('properties', {}).get('jobtitle', '')
                        ),
                        'phone': (
                            contact_info['phone']
                            if contact_info['phone'] != 'Unknown'
                            else contact.get('properties', {}).get('phone', '')
                        )
                    })
                    if new_contact:
                        logger.info(f"Created new contact in HubSpot: {contact_info['new_email']}")
                        
                        # Copy associations
                        old_associations = hubspot.get_contact_associations(contact_id)
                        for association in old_associations:
                            hubspot.create_association(new_contact['id'], association['id'], association['type'])
                        
                        # Delete old contact
                        if hubspot.delete_contact(contact_id):
                            logger.info(f"Deleted old contact: {email}")
                        
                        # Archive the notification
                        if gmail_service.archive_email(message_id):
                            logger.info("Archived employment change notification")
                else:
                    logger.info("No new contact info found, processing as standard auto-reply.")
                    if gmail_service.archive_email(message_id):
                        logger.info("Archived standard auto-reply message")
            
            elif is_inactive_email(body, subject):
                notification = {
                    "bounced_email": contact_email,
                    "message_id": message_id
                }
                process_bounce_notification(notification, gmail_service)
            
            logger.info(f"Processed response for contact: {contact_email}")
        else:
            logger.warning(f"No contact found in HubSpot for email: {email}")
            # Check for employment change before giving up
            if is_employment_change(body, subject):
                logger.info("Processing employment change for non-existent contact")
                process_employment_change(email, message_id, gmail_service)
            else:
                # Still try to archive the message even if no contact found
                if gmail_service.archive_email(message_id):
                    logger.info("Archived message for non-existent contact")

    except Exception as e:
        logger.error(f"Error processing email response: {str(e)}", exc_info=True)


def process_bounce_notifications(target_email: str = None):
    """
    Main function to process all bounce notifications and auto-replies.
    """
    logger.info("Starting bounce notification and auto-reply processing...")
    
    gmail_service = GmailService()
    hubspot_service = HubspotService(HUBSPOT_API_KEY)
    processed_emails = set()
    target_domain = "rainmakersusa.com"  # Set target domain
    
    logger.info(f"Processing emails for domain: {target_domain}")
    
    # 1) Process bounce notifications
    logger.info("=" * 80)
    logger.info("Searching for bounce notifications in inbox...")
    bounce_notifications = gmail_service.get_all_bounce_notifications(inbox_only=True)
    
    if bounce_notifications:
        logger.info(f"Found {len(bounce_notifications)} bounce notifications to process.")
        for notification in bounce_notifications:
            email = notification.get('bounced_email')
            if email:
                domain = email.split('@')[-1].lower()
                if domain == target_domain:
                    logger.info(f"Processing bounce notification for domain email: {email}")
                    process_bounce_notification(notification, gmail_service)
                    processed_emails.add(email)
    
    # 2) Process auto-replies with domain filter
    logger.info("=" * 80)
    logger.info("Searching for auto-reply notifications...")
    logger.info(f"Using auto-reply query: {AUTO_REPLY_QUERY}")
    
    auto_replies = gmail_service.search_messages(AUTO_REPLY_QUERY)
    
    if auto_replies:
        logger.info(f"Found {len(auto_replies)} auto-reply notifications for {target_domain} domain.")
        for message in auto_replies:
            try:
                message_data = gmail_service.get_message(message['id'])
                if message_data:
                    from_header = gmail_service._get_header(message_data, 'from')
                    to_header = gmail_service._get_header(message_data, 'to')
                    subject = gmail_service._get_header(message_data, 'subject')
                    
                    # Extract email addresses from headers
                    email_addresses = []
                    for header in [from_header, to_header]:
                        if header:
                            matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', header)
                            email_addresses.extend(matches)
                    
                    # Find domain-specific emails
                    domain_emails = [
                        email for email in email_addresses 
                        if email.split('@')[-1].lower() == target_domain
                    ]
                    
                    if domain_emails:
                        body = gmail_service._get_full_body(message_data)
                        logger.info(f"Found domain emails in auto-reply: {domain_emails}")
                        
                        for email in domain_emails:
                            logger.info(f"Processing auto-reply for: {email}")
                            logger.info(f"Subject: {subject}")
                            logger.info(f"Preview: {get_first_50_words(body)}")
                            
                            if TESTING:
                                logger.info(f"TESTING MODE: Would process auto-reply for {email}")
                            else:
                                process_email_response(message['id'], email, subject, body, gmail_service)
                                processed_emails.add(email)
                    else:
                        logger.debug(f"No {target_domain} emails found in auto-reply message")
            except Exception as e:
                logger.error(f"Error processing auto-reply: {str(e)}", exc_info=True)
    
    # 3) Process regular responses with broader query
    logger.info("=" * 80)
    logger.info("Searching for regular responses with domain-wide query...")
    
    # Broader query to catch all domain emails
    regular_query = f"""
        (from:@{target_domain} OR to:@{target_domain})
        -in:trash -in:spam -label:sent
        newer_than:30d
    """.replace('\n', ' ').strip()
    
    logger.info(f"Using domain-wide search query: {regular_query}")
    
    
    try:
        regular_responses = gmail_service.search_messages(regular_query)
        
        if regular_responses:
            logger.info(f"Found {len(regular_responses)} potential domain messages.")
            for idx, message in enumerate(regular_responses, 1):
                logger.info("-" * 40)
                logger.info(f"Processing message {idx} of {len(regular_responses)}")
                
                try:
                    message_data = gmail_service.get_message(message['id'])
                    if message_data:
                        # Extract and log all headers
                        from_header = gmail_service._get_header(message_data, 'from')
                        to_header = gmail_service._get_header(message_data, 'to')
                        cc_header = gmail_service._get_header(message_data, 'cc')
                        subject = gmail_service._get_header(message_data, 'subject')
                        date = gmail_service._get_header(message_data, 'date')
                        
                        logger.info(f"Message details:")
                        logger.info(f"  Date: {date}")
                        logger.info(f"  From: {from_header}")
                        logger.info(f"  To: {to_header}")
                        logger.info(f"  CC: {cc_header}")
                        logger.info(f"  Subject: {subject}")
                        
                        # Extract all email addresses from all headers
                        email_addresses = []
                        for header in [from_header, to_header, cc_header]:
                            if header:
                                matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', header)
                                email_addresses.extend(matches)
                        
                        # Find domain-specific emails
                        domain_emails = [
                            email for email in email_addresses 
                            if email.split('@')[-1].lower() == target_domain
                        ]
                        
                        if domain_emails:
                            logger.info(f"Found {len(domain_emails)} domain email(s) in message: {domain_emails}")
                            body = gmail_service._get_full_body(message_data)
                            
                            for email in domain_emails:
                                logger.info(f"Processing response for domain email: {email}")
                                logger.info(f"Message preview: {get_first_50_words(body)}")
                                
                                if TESTING:
                                    logger.info(f"TESTING MODE: Would process message for {email}")
                                else:
                                    process_email_response(message['id'], email, subject, body, gmail_service)
                                    processed_emails.add(email)
                                    logger.info(f"✓ Processed message for {email}")
                        else:
                            logger.info("No domain emails found in message headers")
                
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}", exc_info=True)
                    continue
        else:
            logger.info("No messages found matching domain-wide search criteria.")
    
    except Exception as e:
        logger.error(f"Error during Gmail search: {str(e)}", exc_info=True)

    # 4) Verification
    if processed_emails:
        logger.info("=" * 80)
        logger.info("Verifying processing results...")
        all_verified = True
        for email in processed_emails:
            if not verify_processing(email, gmail_service, hubspot_service):
                all_verified = False
                logger.error(f"❌ Processing verification failed for {email}")
        
        if all_verified:
            logger.info("✅ All processing verified successfully")
        else:
            logger.error("❌ Some processing verifications failed - check logs for details")
    else:
        logger.info("No emails were processed.")


if __name__ == "__main__":
    TARGET_EMAIL = "lvelasquez@rainmakersusa.com"
    if TESTING:
        print("\nRunning in TEST mode - no actual changes will be made\n")
    process_bounce_notifications(TARGET_EMAIL)
    # Or process_bounce_notifications() for all.

```

## scripts\monitor\run_email_monitoring.py
```python
#!/usr/bin/env python3

"""
Email Monitoring System
======================

This script orchestrates the email monitoring process by coordinating two main components:
1. Review Status Monitor (monitor_email_review_status.py)
   - Tracks Gmail drafts for review labels
   - Updates database when drafts are approved

2. Sent Status Monitor (monitor_email_sent_status.py)
   - Verifies when emails are actually sent
   - Updates database with send confirmation and details

Email Status Flow
----------------
1. draft → reviewed → sent
   - draft: Initial state when email is created
   - reviewed: Email has been approved via Gmail labels
   - sent: Email has been sent and confirmed via Gmail API

Configuration
------------
The script supports the following environment variables:
- EMAIL_MONITOR_INTERVAL: Seconds between monitoring runs (default: 300)
- EMAIL_MONITOR_MAX_RETRIES: Max retry attempts per run (default: 3)
- EMAIL_MONITOR_RETRY_DELAY: Seconds between retries (default: 60)
"""

import os
import sys
import time
from datetime import datetime
import pytz
from typing import Optional, Tuple, Dict
import traceback

# Local imports
from utils.logging_setup import logger
import monitor_email_review_status
import monitor_email_sent_status

###############################################################################
#                           CONFIGURATION
###############################################################################

class MonitorConfig:
    """Configuration settings for the monitoring process."""
    
    def __init__(self):
        # Process interval settings
        self.monitor_interval = int(os.getenv('EMAIL_MONITOR_INTERVAL', 300))  # 5 minutes
        self.max_retries = int(os.getenv('EMAIL_MONITOR_MAX_RETRIES', 3))
        self.retry_delay = int(os.getenv('EMAIL_MONITOR_RETRY_DELAY', 60))
        
        # Monitoring flags
        self.check_reviews = True
        self.check_sent = True
        
    def __str__(self) -> str:
        """Return string representation of config."""
        return (
            f"MonitorConfig("
            f"interval={self.monitor_interval}s, "
            f"max_retries={self.max_retries}, "
            f"retry_delay={self.retry_delay}s, "
            f"check_reviews={self.check_reviews}, "
            f"check_sent={self.check_sent})"
        )

###############################################################################
#                           MONITORING FUNCTIONS
###############################################################################

def run_review_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the review status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_review_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_sent_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the sent status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_sent_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_monitoring_cycle(config: MonitorConfig) -> Dict[str, bool]:
    """
    Run a complete monitoring cycle with both checks.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        Dict[str, bool]: Status of each check
    """
    results = {
        'review_check': False,
        'sent_check': False
    }
    
    start_time = datetime.now(pytz.UTC)
    logger.info(f"=== Starting Monitoring Cycle at {start_time} ===")
    
    # Review Status Check
    if config.check_reviews:
        for attempt in range(config.max_retries):
            success, error = run_review_check()
            if success:
                results['review_check'] = True
                break
            else:
                logger.error(f"Review check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    # Brief pause between checks
    time.sleep(1)
    
    # Sent Status Check
    if config.check_sent:
        for attempt in range(config.max_retries):
            success, error = run_sent_check()
            if success:
                results['sent_check'] = True
                break
            else:
                logger.error(f"Sent check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    end_time = datetime.now(pytz.UTC)
    duration = end_time - start_time
    
    logger.info("\n=== Monitoring Cycle Complete ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Results: {results}")
    
    return results

###############################################################################
#                               MAIN PROCESS
###############################################################################

def run_continuous_monitoring(config: MonitorConfig):
    """
    Run the monitoring process continuously with the specified interval.
    
    Args:
        config: MonitorConfig instance with settings
    """
    logger.info(f"Starting continuous monitoring with config: {config}")
    
    while True:
        try:
            run_monitoring_cycle(config)
            logger.info(f"Waiting {config.monitor_interval} seconds until next cycle...")
            time.sleep(config.monitor_interval)
            
        except KeyboardInterrupt:
            logger.info("\nMonitoring interrupted by user")
            break
            
        except Exception as e:
            logger.error("Unexpected error in monitoring cycle:")
            logger.error(traceback.format_exc())
            logger.info(f"Retrying in {config.monitor_interval} seconds...")
            time.sleep(config.monitor_interval)

def run_single_cycle(config: MonitorConfig) -> bool:
    """
    Run a single monitoring cycle.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        bool: True if all enabled checks succeeded
    """
    results = run_monitoring_cycle(config)
    return all(results.values())

def main():
    """
    Main entry point for the email monitoring system.
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = MonitorConfig()
        
        # Check for command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run single cycle
            success = run_single_cycle(config)
            return 0 if success else 1
        else:
            # Run continuous monitoring
            run_continuous_monitoring(config)
            return 0
            
    except Exception as e:
        logger.error(f"Fatal error in monitoring system: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
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

## scripts\run_email_monitoring.py
```python

```

## scripts\run_scheduler.py
```python
import os
import sys
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from scheduling.email_sender import send_scheduled_emails
from utils.logging_setup import logger
import time

def start_scheduler():
    """Start the email scheduler for sending all scheduled emails."""
    logger.info("Starting email scheduler for all emails...")
    
    while True:
        try:
            logger.debug("Checking for emails to send...")
            send_scheduled_emails()
        except Exception as e:
            logger.error(f"Error in email scheduler: {str(e)}")
        
        # Wait 30 seconds before checking again
        time.sleep(30)

if __name__ == "__main__":
    start_scheduler() 
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

## scripts\update_company_names.py
```python
import csv
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.database import get_db_connection
from utils.logging_setup import logger

def update_company_names():
    """Update company_short_name in emails table using lead_data.csv"""
    try:
        # Read the CSV file
        csv_path = project_root / "docs" / "data" / "lead_data.csv"
        email_to_company = {}
        
        logger.info("Reading lead data from CSV...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                email = row['Email'].lower()  # Store emails in lowercase for matching
                company_short_name = row['Company Short Name'].strip()
                if email and company_short_name:  # Only store if both fields have values
                    email_to_company[email] = company_short_name

        logger.info(f"Found {len(email_to_company)} email-to-company mappings")

        # Update the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all emails that need updating
        cursor.execute("""
            SELECT DISTINCT email_address 
            FROM emails 
            WHERE company_short_name IS NULL 
            OR company_short_name = ''
        """)
        
        update_count = 0
        for row in cursor.fetchall():
            email = row[0].lower() if row[0] else ''
            if email in email_to_company:
                cursor.execute("""
                    UPDATE emails 
                    SET company_short_name = ? 
                    WHERE LOWER(email_address) = ?
                """, (email_to_company[email], email))
                update_count += cursor.rowcount

        conn.commit()
        logger.info(f"Updated company_short_name for {update_count} email records")

    except Exception as e:
        logger.error(f"Error updating company names: {str(e)}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting company name update process...")
    update_company_names()
    logger.info("Completed company name update process") 
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
## services/gmail_service.py
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
import re
from services.hubspot_service import HubspotService
import base64

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

    def search_replies(self, thread_id):
        """
        Search for replies in a Gmail thread.
        """
        try:
            service = get_gmail_service()
            query = f'threadId:{thread_id} is:reply'
            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])
            return messages if messages else None
        except Exception as e:
            logger.error(f"Error searching for replies in thread {thread_id}: {str(e)}", exc_info=True)
            return None

    def _get_header(self, message: Dict[str, Any], header_name: str) -> str:
        """Extract header value from Gmail message."""
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header["name"].lower() == header_name.lower():
                return header["value"]
        return ""

    def get_latest_emails_with_bounces(self, email_address: str) -> Dict[str, Any]:
        """Get the latest emails including bounce notifications for a contact from Gmail."""
        try:
            # Search for latest inbound message
            inbound_query = f"from:{email_address}"
            inbound_messages = search_messages(query=inbound_query)
            
            # Search for bounce notifications and outbound messages
            outbound_query = f"(to:{email_address} OR (subject:\"Delivery Status Notification\" from:mailer-daemon@googlemail.com {email_address}))"
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
                    "body_text": self._get_full_body(latest_inbound),
                    "direction": "INCOMING_EMAIL",
                    "gmail_id": latest_inbound["id"]
                }
            
            # Get latest outbound email or bounce notification
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
                
                # Check if it's a bounce notification
                from_header = self._get_header(latest_outbound, "from")
                is_bounce = "mailer-daemon@googlemail.com" in from_header.lower()
                
                latest_emails["outbound"] = {
                    "timestamp": timestamp.isoformat(),
                    "subject": self._get_header(latest_outbound, "subject"),
                    "body_text": self._get_full_body(latest_outbound),
                    "direction": "BOUNCE" if is_bounce else "EMAIL",
                    "gmail_id": latest_outbound["id"],
                    "is_bounce": is_bounce
                }
            
            return latest_emails
            
        except Exception as e:
            logger.error(f"Error getting Gmail messages: {str(e)}")
            return {"inbound": None, "outbound": None}

    def _get_full_body(self, message: Dict[str, Any]) -> str:
        """Extract full message body including all parts."""
        try:
            parts = []
            def extract_parts(payload):
                if 'parts' in payload:
                    for part in payload['parts']:
                        extract_parts(part)
                elif 'body' in payload:
                    if 'data' in payload['body']:
                        parts.append(payload['body']['data'])
                    elif 'attachmentId' in payload['body']:
                        # Handle attachments if needed
                        pass
            
            if 'payload' in message:
                extract_parts(message['payload'])
            
            # Join all parts and decode
            full_body = ''
            for part in parts:
                try:
                    decoded = base64.urlsafe_b64decode(part).decode('utf-8')
                    full_body += decoded + '\n'
                except Exception as e:
                    logger.warning(f"Error decoding message part: {str(e)}")
            
            return full_body
        except Exception as e:
            logger.error(f"Error getting full body: {str(e)}")
            return ""

    def archive_email(self, message_id: str) -> bool:
        """Archive an email in Gmail by removing the INBOX label."""
        try:
            service = get_gmail_service()
            # Remove INBOX label to archive the message
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['INBOX']}
            ).execute()
            logger.info(f"Successfully archived email {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error archiving email {message_id}: {str(e)}")
            return False

    def search_bounce_notifications(self, email_address: str) -> List[str]:
        """Search for bounce notification emails for a specific email address."""
        try:
            # Search for bounce notifications containing the email address
            query = f'from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" "{email_address}"'
            messages = search_messages(query=query)
            
            if messages:
                return [msg['id'] for msg in messages]
            return []
            
        except Exception as e:
            logger.error(f"Error searching bounce notifications: {str(e)}")
            return []

    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Search for messages in Gmail using the given query."""
        try:
            logger.debug(f"Executing Gmail search with query: {query}")
            service = get_gmail_service()
            
            # Execute the search
            result = service.users().messages().list(
                userId='me',
                q=query
            ).execute()
            
            messages = result.get('messages', [])
            logger.debug(f"Found {len(messages)} messages matching query")
            
            return messages
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}", exc_info=True)
            return []

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific message by ID."""
        try:
            service = get_gmail_service()
            message = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()
            return message
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {str(e)}")
            return None

    def get_bounce_message_details(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get details from a bounce notification message including the bounced email address."""
        try:
            logger.debug(f"Getting bounce message details for ID: {message_id}")
            message = self.get_message(message_id)
            if not message:
                return None
            
            body = self._get_full_body(message)
            if not body:
                return None
            
            logger.debug(f"Message body length: {len(body)}")
            logger.debug(f"First 200 characters of body: {body[:200]}")
            
            # Add Office 365 patterns to existing patterns
            patterns = [
                # Existing Gmail patterns
                r'Original-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'Final-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r'To: <([\w\.-]+@[\w\.-]+\.\w+)>',
                r'The email account that you tried to reach[^\n]*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,})',
                r'failed permanently.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                r"message wasn(?:&#39;|\')t delivered to ([\w\.-]+@[\w\.-]+\.\w+)",
                
                # New Office 365 patterns
                r"Your message to ([\w\.-]+@[\w\.-]+\.\w+) couldn't be delivered",
                r"Recipient Address:\s*([\w\.-]+@[\w\.-]+\.\w+)",
                r"550 5\.1\.10.*?Recipient ([\w\.-]+@[\w\.-]+\.\w+) not found",
                r"RESOLVER\.ADR\.RecipientNotFound; Recipient ([\w\.-]+@[\w\.-]+\.\w+) not found",
            ]
            
            # Update bounce query to include Office 365 postmaster
            self.BOUNCE_QUERY = """
                (from:mailer-daemon@googlemail.com subject:"Delivery Status Notification" 
                 OR from:postmaster@*.outbound.protection.outlook.com subject:"Undeliverable")
                in:inbox
            """
            
            for pattern in patterns:
                match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
                if match:
                    bounced_email = match.group(1)
                    if bounced_email and '@' in bounced_email:
                        logger.debug(f"Found bounced email {bounced_email} using pattern: {pattern}")
                        subject = self._get_header(message, 'subject')
                        return {
                            'bounced_email': bounced_email,
                            'subject': subject,
                            'body': body,
                            'message_id': message_id
                        }
                else:
                    logger.debug(f"Pattern did not match: {pattern}")
            
            logger.debug(f"No email found in message {message_id}. Full body: {body[:500]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error getting bounce message details for {message_id}: {str(e)}", exc_info=True)
            return None

    def process_bounce_notification(self, message_id: str) -> Optional[str]:
        """
        Process a bounce notification and extract the bounced email address.
        """
        try:
            details = self.get_bounce_message_details(message_id)
            if details and 'bounced_email' in details:
                # Add HubSpot logging
                logger.info(f"Attempting to update HubSpot for bounced email: {details['bounced_email']}")
                
                # Ensure HubSpot is being called
                hubspot_service = HubspotService()
                success = hubspot_service.mark_contact_as_bounced(details['bounced_email'])
                
                if success:
                    logger.info(f"Successfully updated HubSpot for {details['bounced_email']}")
                else:
                    logger.error(f"Failed to update HubSpot for {details['bounced_email']}")
                
                return details['bounced_email']
            return None
        except Exception as e:
            logger.error(f"Error processing bounce notification {message_id}: {str(e)}")
            return None

    def get_all_bounce_notifications(self, inbox_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all bounce notifications from Gmail.
        """
        try:
            # Create a clean, precise query that matches both direct and "via" senders
            base_query = (
                '(from:"postmaster@*" OR from:"mailer-daemon@*") '
                'subject:("Undeliverable" OR "Delivery Status Notification")'
            )
            
            if inbox_only:
                query = f"{base_query} in:inbox"
            else:
                query = base_query
            
            logger.debug(f"Using Gmail search query: {query}")
            messages = self.search_messages(query)
            
            # Log the raw results for debugging
            logger.debug(f"Raw search results: {messages}")
            
            bounce_notifications = []
            for message in messages:
                message_id = message['id']
                full_message = self.get_message(message_id)
                if full_message:
                    headers = full_message.get('payload', {}).get('headers', [])
                    from_header = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'Unknown')
                    logger.info(f"Processing message - From: {from_header}, Subject: {subject}")
                
                details = self.get_bounce_message_details(message_id)
                if details:
                    bounce_notifications.append(details)
                    logger.info(f"Found bounce notification for: {details.get('bounced_email')}")
                else:
                    logger.warning(f"Could not extract bounce details from message ID: {message_id}")
            
            return bounce_notifications
            
        except Exception as e:
            logger.error(f"Error getting bounce notifications: {str(e)}", exc_info=True)
            return []

    def get_rejection_search_query(self):
        """Returns search query for explicit rejection emails"""
        rejection_phrases = [
            '"no thanks"',
            '"don\'t contact"',
            '"do not contact"',
            '"please remove"',
            '"not interested"',
            '"we use"',
            '"we already use"',
            '"we have"',
            '"we already have"',
            '"please don\'t contact"',
            '"stop contacting"',
            '"remove me"',
            '"unsubscribe"'
        ]
        
        # Combine phrases with OR and add inbox filter
        query = f"({' OR '.join(rejection_phrases)}) in:inbox"
        return query

    def get_gmail_template(self, template_name: str = "salesv2") -> str:
        """
        Fetch HTML email template from Gmail drafts whose subject
        contains `template_name`.

        Args:
            template_name (str): Name to search for in draft subjects. Defaults to "sales".

        Returns:
            str: The template as an HTML string, or "" if no template found.
        """
        try:
            service = get_gmail_service()
            drafts_list = service.users().drafts().list(userId='me').execute()
            if not drafts_list.get('drafts'):
                logger.error("No drafts found in Gmail account.")
                return ""

            # Optional: gather subjects for debugging
            draft_subjects = []
            template_html = ""

            for draft in drafts_list['drafts']:
                msg_id = draft['message']['id']
                msg = service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()

                # Grab the subject
                headers = msg['payload'].get('headers', [])
                subject = next(
                    (h['value'] for h in headers if h['name'].lower() == 'subject'),
                    ''
                ).lower()
                draft_subjects.append(subject)

                # If the draft subject contains template_name, treat that as the template
                if template_name.lower() in subject:
                    logger.debug(f"Found template draft with subject: {subject}")

                    # Extract HTML parts
                    if 'parts' in msg['payload']:
                        for part in msg['payload']['parts']:
                            if part['mimeType'] == 'text/html':
                                template_html = base64.urlsafe_b64decode(
                                    part['body']['data']
                                ).decode('utf-8', errors='ignore')
                                break
                    elif msg['payload'].get('mimeType') == 'text/html':
                        template_html = base64.urlsafe_b64decode(
                            msg['payload']['body']['data']
                        ).decode('utf-8', errors='ignore')

                    if template_html:
                        return template_html

            # If we got here, we never found a match
            logger.error(
                f"No template found with name: {template_name}. "
                f"Available draft subjects: {draft_subjects}"
            )
            return ""

        except Exception as e:
            logger.error(f"Error fetching Gmail template: {str(e)}", exc_info=True)
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
from datetime import datetime


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

    def get_contacts_from_list(self, list_id: str) -> List[Dict[str, Any]]:
        """Get all contacts from a specified HubSpot list."""
        url = f"{self.base_url}/contacts/v1/lists/{list_id}/contacts/all"
        all_contacts = []
        vidOffset = 0
        has_more = True
        
        while has_more:
            try:
                params = {
                    "count": 100,
                    "vidOffset": vidOffset
                }
                response = self._make_hubspot_get(url, params)
                
                contacts = response.get("contacts", [])
                all_contacts.extend(contacts)
                
                has_more = response.get("has-more", False)
                vidOffset = response.get("vid-offset", 0)
                
            except Exception as e:
                logger.error(f"Error fetching contacts from list: {str(e)}")
                break
        
        return all_contacts

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully deleted contact {contact_id} from HubSpot")
            return True
        except Exception as e:
            logger.error(f"Error deleting contact {contact_id}: {str(e)}")
            return False

    def mark_contact_as_bounced(self, email: str) -> bool:
        """Mark a contact as bounced in HubSpot."""
        try:
            # Add detailed logging
            logger.info(f"Marking contact as bounced in HubSpot: {email}")
            
            # Get the contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"Contact not found in HubSpot: {email}")
                return False
            
            # Update the contact properties
            properties = {
                "email_bounced": "true",
                "email_bounced_date": datetime.now().strftime("%Y-%m-%d"),
                "email_bounced_reason": "Hard bounce - Invalid recipient"
            }
            
            # Make the API call
            success = self.update_contact(contact['id'], properties)
            
            if success:
                logger.info(f"Successfully marked contact as bounced in HubSpot: {email}")
            else:
                logger.error(f"Failed to mark contact as bounced in HubSpot: {email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking contact as bounced in HubSpot: {str(e)}")
            return False

    def create_contact(self, properties: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Create a new contact in HubSpot.
        
        Args:
            properties (Dict[str, str]): Dictionary of contact properties
                Required keys: email
                Optional keys: firstname, lastname, company, jobtitle, phone
                
        Returns:
            Optional[Dict[str, Any]]: The created contact data or None if creation fails
        """
        try:
            email = properties.get('email')
            logger.info(f"Creating new contact in HubSpot with email: {email}")
            logger.debug(f"Contact properties: {properties}")
            
            # Validate required email property
            if not email:
                logger.error("Cannot create contact: email is required")
                return None
            
            # Check if contact already exists
            existing_contact = self.get_contact_by_email(email)
            if existing_contact:
                logger.warning(f"Contact already exists with email {email}. Updating instead.")
                # Update existing contact with new properties
                if self.update_contact(existing_contact['id'], properties):
                    logger.info(f"Successfully updated existing contact: {existing_contact['id']}")
                    return existing_contact
                return None
            
            payload = {
                "properties": {
                    key: str(value) for key, value in properties.items() if value and value != 'Unknown'
                }
            }
            
            logger.debug(f"Making create contact request with payload: {payload}")
            response = requests.post(
                self.contacts_endpoint,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 409:
                logger.warning(f"Conflict creating contact {email} - contact may already exist")
                # Try to get the existing contact again (in case it was just created)
                existing_contact = self.get_contact_by_email(email)
                if existing_contact:
                    logger.info(f"Found existing contact after conflict: {existing_contact['id']}")
                    if self.update_contact(existing_contact['id'], properties):
                        return existing_contact
                return None
            
            response.raise_for_status()
            contact_data = response.json()
            logger.info(f"Successfully created new contact: {contact_data.get('id')}")
            logger.debug(f"New contact data: {contact_data}")
            return contact_data
            
        except Exception as e:
            logger.error(f"Error creating contact in HubSpot: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    def get_contact_associations(self, contact_id: str) -> List[Dict[str, Any]]:
        """Get all associations for a contact."""
        url = f"{self.contacts_endpoint}/{contact_id}/associations"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching contact associations: {str(e)}")
            return []

    def create_association(self, from_id: str, to_id: str, association_type: str) -> bool:
        """Create an association between two objects."""
        url = f"{self.contacts_endpoint}/{from_id}/associations/{association_type}/{to_id}"
        
        try:
            response = requests.put(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully created association between {from_id} and {to_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating association: {str(e)}")
            return False

    def update_contact(self, contact_id: str, properties: Dict[str, Any]) -> bool:
        """Update contact properties in HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            payload = {"properties": properties}
            
            response = requests.patch(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating contact {contact_id}: {str(e)}")
            return False

    def mark_do_not_contact(self, email: str, company_name: str = None) -> bool:
        """
        Mark a contact as 'Do Not Contact' and update related properties.
        
        Args:
            email: Contact's email address
            company_name: Optional company name for template
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing do-not-contact request for {email}")
            
            # Get contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"Contact not found for email: {email}")
                return False
            
            contact_id = contact.get('id')
            
            # Update contact properties
            properties = {
                "hs_lead_status": "DQ",  # Set lead status to DQ
                "do_not_contact": "true",
                "do_not_contact_reason": "Customer Request",
                "lifecyclestage": "Other",
                "hs_marketable_reason_id": "UNSUBSCRIBED",
                "hs_marketable_status": "NO",
                "hs_marketable_until_renewal": "false"
            }
            
            # Update the contact
            if not self.update_contact(contact_id, properties):
                logger.error(f"Failed to update contact properties for {email}")
                return False
            
            # Get first name for template
            first_name = contact.get('properties', {}).get('firstname', '')
            company_short = company_name or contact.get('properties', {}).get('company', '')
            
            # Generate response from template
            template_vars = {
                "firstname": first_name,
                "company_short_name": company_short
            }
            
            logger.info(f"Successfully marked {email} as do-not-contact")
            return True
            
        except Exception as e:
            logger.error(f"Error processing do-not-contact request: {str(e)}")
            return False

    def mark_contact_as_dq(self, email: str, reason: str) -> bool:
        """Mark a contact as disqualified in HubSpot."""
        try:
            # First get the contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"No contact found in HubSpot for {email}")
                return False

            contact_id = contact.get('id')
            if not contact_id:
                logger.warning(f"Contact found but no ID for {email}")
                return False

            # Update the contact properties to mark as DQ
            properties = {
                'lifecyclestage': 'disqualified',
                'hs_lead_status': 'DQ',
                'dq_reason': reason,
                'dq_date': datetime.now().strftime('%Y-%m-%d')
            }

            # Update the contact in HubSpot
            url = f"{self.base_url}/objects/contacts/{contact_id}"
            response = requests.patch(
                url,
                json={'properties': properties},
                headers=self.headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully marked {email} as DQ in HubSpot")
                return True
            else:
                logger.error(f"Failed to mark {email} as DQ in HubSpot. Status code: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error marking contact as DQ in HubSpot: {str(e)}")
            return False

    def get_contacts_by_company_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get contacts in HubSpot whose email domain matches the given company domain.
        
        Args:
            domain (str): The company domain to search for (e.g., "example.com")
            
        Returns:
            List[Dict[str, Any]]: List of contact records matching the domain
        """
        try:
            logger.info(f"Searching for contacts with company domain: {domain}")
            url = f"{self.contacts_endpoint}/search"
            
            # Updated payload structure to match HubSpot's API requirements
            payload = {
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "email",
                                "operator": "CONTAINS_TOKEN",  # Changed from CONTAINS to CONTAINS_TOKEN
                                "value": domain
                            }
                        ]
                    }
                ],
                "sorts": [
                    {
                        "propertyName": "createdate",
                        "direction": "DESCENDING"
                    }
                ],
                "properties": [
                    "email", 
                    "firstname", 
                    "lastname", 
                    "company", 
                    "jobtitle",
                    "phone",
                    "createdate",
                    "lastmodifieddate"
                ],
                "limit": 100
            }
            
            try:
                response = self._make_hubspot_post(url, payload)
                
                if not response:
                    logger.warning(f"No response received from HubSpot for domain: {domain}")
                    return []
                    
                results = response.get("results", [])
                
                # Filter results to ensure exact domain match
                filtered_results = []
                for contact in results:
                    contact_email = contact.get("properties", {}).get("email", "")
                    if contact_email and contact_email.lower().endswith(f"@{domain.lower()}"):
                        filtered_results.append(contact)
                
                logger.info(f"Found {len(filtered_results)} contacts for domain {domain}")
                return filtered_results
                
            except Exception as e:
                logger.error(f"Error in HubSpot API request: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"Error getting contacts by company domain: {str(e)}")
            return []

```

## services\leads_service.py
```python
## services/leads_service.py
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
from utils.exceptions import HubSpotError


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

    def get_lead_summary(self, lead_id: str) -> Dict[str, Optional[str]]:
        """
        Get key summary information about a lead.
        
        Args:
            lead_id: HubSpot contact ID
            
        Returns:
            Dict containing:
                - last_reply_date: Date of latest email reply
                - lifecycle_stage: Current lifecycle stage
                - company_short_name: Short name of associated company
                - company_name: Full name of associated company
                - company_state: State of associated company
                - error: Error message if any
        """
        try:
            logger.info(f"Fetching HubSpot properties for lead {lead_id}...")
            contact_props = self.data_gatherer.hubspot.get_contact_properties(lead_id)
            result = {
                'last_reply_date': None,
                'lifecycle_stage': None,
                'company_short_name': None,
                'company_name': None,
                'company_state': None,
                'error': None
            }
            
            # Get basic contact info
            result['last_reply_date'] = contact_props.get('hs_sales_email_last_replied')
            result['lifecycle_stage'] = contact_props.get('lifecyclestage')
            
            # Get company information
            company_id = self.data_gatherer.hubspot.get_associated_company_id(lead_id)
            if company_id:
                company_props = self.data_gatherer.hubspot.get_company_data(company_id)
                if company_props:
                    result['company_name'] = company_props.get('name')
                    result['company_short_name'] = company_props.get('company_short_name')
                    result['company_state'] = company_props.get('state')
            
            return result
            
        except HubSpotError as e:
            if '404' in str(e):
                return {'error': '404 - Lead not found'}
            return {'error': str(e)}

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

## services\response_analyzer_service.py
```python
import re
from typing import Dict, Optional, Tuple, Any
from .data_gatherer_service import DataGathererService
from .gmail_service import GmailService
from .hubspot_service import HubspotService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY

class ResponseAnalyzerService:
    def __init__(self):
        self.data_gatherer = DataGathererService()
        self.gmail_service = GmailService()
        self.hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # Add patterns for different types of responses
        self.out_of_office_patterns = [
            r"out\s+of\s+office",
            r"automatic\s+reply",
            r"auto\s*-?\s*reply",
            r"vacation\s+response",
            r"away\s+from\s+(?:the\s+)?office",
            r"will\s+be\s+(?:away|out)",
            r"not\s+(?:in|available)",
            r"on\s+vacation",
            r"on\s+holiday",
        ]
        
        self.employment_change_patterns = [
            r"no\s+longer\s+(?:with|at)",
            r"has\s+left\s+the\s+company",
            r"(?:email|address)\s+is\s+no\s+longer\s+valid",
            r"(?:has|have)\s+moved\s+on",
            r"no\s+longer\s+employed",
            r"last\s+day",
            r"departed",
            r"resigned",
        ]
        
        self.do_not_contact_patterns = [
            r"do\s+not\s+contact",
            r"stop\s+(?:contact|email)",
            r"unsubscribe",
            r"remove\s+(?:me|from)",
            r"opt\s+out",
            r"take\s+me\s+off",
        ]
        
        self.inactive_email_patterns = [
            r"undeliverable",
            r"delivery\s+failed",
            r"delivery\s+status\s+notification",
            r"failed\s+delivery",
            r"bounce",
            r"not\s+found",
            r"does\s+not\s+exist",
            r"invalid\s+recipient",
            r"recipient\s+rejected",
            r"no\s+longer\s+active",
            r"account\s+disabled",
            r"email\s+is\s+(?:now\s+)?inactive",
        ]

    def analyze_response_status(self, email_address: str) -> Dict:
        """
        Analyze if and how a lead has responded to our emails.
        Returns a dictionary with analysis results.
        """
        try:
            # Use the new method that includes bounce detection
            messages = self.gmail_service.get_latest_emails_with_bounces(email_address)
            
            if not messages:
                return {
                    "status": "NO_MESSAGES",
                    "message": "No email conversation found for this address."
                }

            # Check both inbound and outbound messages
            inbound_msg = messages.get("inbound", {})
            outbound_msg = messages.get("outbound", {})
            
            # Check for bounce notification first
            if outbound_msg and outbound_msg.get("is_bounce"):
                return {
                    "status": "BOUNCED",
                    "response_type": "BOUNCED_EMAIL",
                    "confidence": 0.95,
                    "timestamp": outbound_msg.get("timestamp"),
                    "message": outbound_msg.get("body_text", ""),
                    "subject": outbound_msg.get("subject", "No subject")
                }

            # Then check for regular responses
            if inbound_msg:
                body_text = inbound_msg.get("body_text", "")
                response_type, confidence = self._categorize_response(body_text)
                
                return {
                    "status": "RESPONSE_FOUND",
                    "response_type": response_type,
                    "confidence": confidence,
                    "timestamp": inbound_msg.get("timestamp"),
                    "message": body_text,
                    "subject": inbound_msg.get("subject", "No subject")
                }
            
            return {
                "status": "NO_RESPONSE",
                "message": "No incoming responses found from this lead."
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Error analyzing response: {str(e)}"
            }

    def _get_latest_response(self, messages):
        """Get the most recent incoming message."""
        incoming_messages = [
            msg for msg in messages 
            if msg.get("direction") not in ["EMAIL", "NOTE", "OUTBOUND"]
            and msg.get("body_text")
        ]
        
        return incoming_messages[-1] if incoming_messages else None

    def _categorize_response(self, message: str) -> Tuple[str, float]:
        """
        Categorize the type of response.
        Returns tuple of (response_type, confidence_score)
        """
        # Clean the message
        cleaned_message = message.lower().strip()

        # Add bounce/delivery failure patterns
        bounce_patterns = [
            r"delivery status notification \(failure\)",
            r"address not found",
            r"recipient address rejected",
            r"address couldn't be found",
            r"unable to receive mail",
            r"delivery failed",
            r"undeliverable",
            r"550 5\.4\.1",  # Common SMTP failure code
        ]

        # Check for bounced emails first
        for pattern in bounce_patterns:
            if re.search(pattern, cleaned_message):
                return "BOUNCED_EMAIL", 0.95

        # Existing patterns
        auto_reply_patterns = [
            r"out\s+of\s+office",
            r"automatic\s+reply",
            r"auto\s*-?\s*reply",
            r"vacation\s+response",
            r"away\s+from\s+(?:the\s+)?office",
        ]

        left_company_patterns = [
            r"no\s+longer\s+(?:with|at)",
            r"has\s+left\s+the\s+company",
            r"(?:email|address)\s+is\s+no\s+longer\s+valid",
            r"(?:has|have)\s+moved\s+on",
            r"no\s+longer\s+employed",
        ]

        # Check for auto-replies
        for pattern in auto_reply_patterns:
            if re.search(pattern, cleaned_message):
                return "AUTO_REPLY", 0.9

        # Check for left company messages
        for pattern in left_company_patterns:
            if re.search(pattern, cleaned_message):
                return "LEFT_COMPANY", 0.9

        # If no patterns match, assume it's a genuine response
        # You might want to add more sophisticated analysis here
        return "GENUINE_RESPONSE", 0.7

    def extract_bounced_email_address(self, bounce_message: Dict[str, Any]) -> Optional[str]:
        """
        Extract the original recipient's email address from a bounce notification message.
        
        Args:
            bounce_message: The full Gmail message object of the bounce notification
            
        Returns:
            str: The extracted email address or None if not found
        """
        try:
            # Get the message body
            if 'payload' not in bounce_message:
                return None

            # First try to get it from the subject line
            subject = None
            for header in bounce_message['payload'].get('headers', []):
                if header['name'].lower() == 'subject':
                    subject = header['value']
                    break

            if subject:
                # Look for email pattern in subject
                import re
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                matches = re.findall(email_pattern, subject)
                if matches:
                    # Filter out mailer-daemon address
                    for email in matches:
                        if 'mailer-daemon' not in email.lower():
                            return email

            # If not found in subject, try message body
            def get_text_from_part(part):
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        import base64
                        try:
                            return base64.urlsafe_b64decode(data).decode('utf-8')
                        except:
                            return ''
                return ''

            # Get text from main payload
            message_text = get_text_from_part(bounce_message['payload'])
            
            # If not in main payload, check parts
            if not message_text:
                for part in bounce_message['payload'].get('parts', []):
                    message_text = get_text_from_part(part)
                    if message_text:
                        break

            if message_text:
                # Look for common bounce message patterns
                patterns = [
                    r'Original-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'Final-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'The email account that you tried to reach does not exist.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'failed permanently.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, message_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error extracting bounced email address: {str(e)}")
            return None

    def is_out_of_office(self, body: str, subject: str) -> bool:
        """Check if message is an out of office reply."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.out_of_office_patterns)

    def is_employment_change(self, body: str, subject: str) -> bool:
        """Check if message indicates employment change."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.employment_change_patterns)

    def is_do_not_contact_request(self, body: str, subject: str) -> bool:
        """Check if message is a request to not be contacted."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.do_not_contact_patterns)

    def is_inactive_email(self, body: str, subject: str) -> bool:
        """Check if message indicates an inactive email address."""
        inactive_phrases = [
            "not actively monitored",
            "no longer monitored",
            "inbox is not monitored",
            "email is inactive",
            "mailbox is inactive",
            "account is inactive",
            "email address is inactive",
            "no longer in service",
            "mailbox is not monitored"
        ]
        
        # Convert to lowercase for case-insensitive matching
        body_lower = body.lower()
        subject_lower = subject.lower()
        
        # Check both subject and body for inactive phrases
        return any(phrase in body_lower or phrase in subject_lower 
                  for phrase in inactive_phrases)

def main():
    """Test function to demonstrate usage."""
    analyzer = ResponseAnalyzerService()
    
    while True:
        email = input("\nEnter email address to analyze (or 'quit' to exit): ")
        
        if email.lower() == 'quit':
            break
            
        result = analyzer.analyze_response_status(email)
        
        print("\nAnalysis Results:")
        print("-" * 50)
        
        if result["status"] == "RESPONSE_FOUND":
            print(f"Response Type: {result['response_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Subject: {result['subject']}")
            print("\nMessage Preview:")
            print(result["message"][:200] + "..." if len(result["message"]) > 200 else result["message"])
        else:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

if __name__ == "__main__":
    main() 
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

## utils\data_extraction.py
```python
from typing import Dict

def extract_lead_data(company_props: Dict, lead_props: Dict) -> Dict:
    """
    Extract and organize lead and company data from HubSpot properties.
    
    Args:
        company_props (Dict): Raw company properties from HubSpot
        lead_props (Dict): Raw lead/contact properties from HubSpot
        
    Returns:
        Dict: Organized data structure with company_data and lead_data
    """
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
    # List of primary files focused on email template building and placeholder replacement
    primary_files = [
        'scheduling/database.py',
        'scheduling/followup_generation.py', 
        'scripts/golf_outreach_strategy.py',
        'services/gmail_service.py',
        'services/leads_service.py',
        'tests/test_followup_generation.py',
        'tests/test_hubspot_leads_service.py'

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
from googleapiclient.errors import HttpError

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
        logger.debug(f"Searching for template with name: {template_name}")
        logger.debug("=" * 80)
        
        # First try to list all drafts
        drafts = service.users().drafts().list(userId='me').execute()
        if not drafts.get('drafts'):
            logger.error("No drafts found in Gmail account")
            return ""

        # Log all available drafts for debugging
        draft_subjects = []
        template_html = ""
        
        logger.debug("SCANNING DRAFTS:")
        logger.debug("=" * 80)
        
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
            logger.debug(f"Found draft with subject: {subject}")
            
            # If this is our template
            if template_name.lower() in subject:
                logger.debug(f"FOUND MATCHING TEMPLATE: {subject}")
                logger.debug("=" * 80)
                
                # Extract HTML content
                if 'parts' in msg['payload']:
                    logger.debug("Processing multipart message")
                    for part in msg['payload']['parts']:
                        logger.debug(f"Checking part with mimeType: {part['mimeType']}")
                        if part['mimeType'] == 'text/html':
                            template_html = base64.urlsafe_b64decode(
                                part['body']['data']
                            ).decode('utf-8')
                            logger.debug(f"Found HTML part, length: {len(template_html)}")
                            logger.debug("First 200 chars of raw template:")
                            logger.debug(template_html[:200])
                            
                            # Clean up the template HTML
                            if '<div class="gmail_quote">' in template_html:
                                logger.debug("Found gmail_quote - cleaning template")
                                template_html = template_html.split('<div class="gmail_quote">')[0]
                                logger.debug(f"Template length after cleaning: {len(template_html)}")
                            break
                elif msg['payload']['mimeType'] == 'text/html':
                    logger.debug("Processing single-part HTML message")
                    template_html = base64.urlsafe_b64decode(
                        msg['payload']['body']['data']
                    ).decode('utf-8')
                    logger.debug(f"Found HTML content, length: {len(template_html)}")
                    
                    # Clean up the template HTML
                    if '<div class="gmail_quote">' in template_html:
                        logger.debug("Found gmail_quote - cleaning template")
                        template_html = template_html.split('<div class="gmail_quote">')[0]
                        logger.debug(f"Template length after cleaning: {len(template_html)}")
                
                if template_html:
                    # Add wrapper div if not present
                    if not template_html.strip().startswith('<div'):
                        logger.debug("Adding wrapper div to template")
                        template_html = f'<div class="template-content">{template_html}</div>'
                    
                    logger.debug("FINAL TEMPLATE:")
                    logger.debug("=" * 80)
                    logger.debug(f"Length: {len(template_html)}")
                    logger.debug("First 200 chars:")
                    logger.debug(template_html[:200])
                    logger.debug("Last 200 chars:")
                    logger.debug(template_html[-200:])
                    logger.debug("=" * 80)
                    
                    return template_html

        if not template_html:
            logger.error(
                f"No template found with name: {template_name}. "
                f"Available draft subjects: {draft_subjects}"
            )
        return template_html

    except Exception as e:
        logger.error(f"Error fetching Gmail template: {str(e)}", exc_info=True)
        logger.exception("Full traceback:")
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
        template = get_gmail_template(service, "salesv2")
        
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
        logger.debug(f"Attempting to retrieve thread with ID: {thread_id}")
        thread_data = service.users().threads().get(userId="me", id=thread_id).execute()
        msgs = thread_data.get("messages", [])
        return len(msgs) > 1
    except HttpError as e:
        if e.resp.status == 404:
            logger.error(f"Thread with ID {thread_id} not found. It may have been deleted or moved.")
        else:
            logger.error(f"HttpError occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error retrieving thread {thread_id}: {e}", exc_info=True)
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

def create_followup_draft(
    sender: str,
    to: str,
    subject: str,
    message_text: str,
    lead_id: str = None,
    sequence_num: int = None,
    original_html: str = None,
    in_reply_to: str = None
) -> Dict[str, Any]:
    try:
        logger.debug(f"Creating follow-up draft for lead_id={lead_id}, sequence_num={sequence_num}")
        
        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return {"status": "error", "error": "No Gmail service"}

        # Create message container
        message = MIMEMultipart('alternative')
        message["to"] = to
        message["subject"] = subject
        message["bcc"] = "20057893@bcc.hubspot.com"

        # Add threading headers
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to

        # Split the message text to remove the original content
        new_content = message_text.split("On ", 1)[0].strip()
        
        # Create the HTML parts separately to avoid f-string backslash issues
        html_start = '<div dir="ltr" style="font-family:Arial, sans-serif;">'
        html_content = new_content.replace("\n", "<br>")
        html_quote_start = '<br><br><div class="gmail_quote"><blockquote class="gmail_quote" style="margin:0 0 0 .8ex;border-left:1px solid #ccc;padding-left:1ex">'
        html_quote_content = original_html if original_html else ""
        html_end = '</blockquote></div></div>'

        # Combine HTML parts
        html = html_start + html_content + html_quote_start + html_quote_content + html_end

        # Create both plain text and HTML versions
        text_part = MIMEText(new_content, 'plain')  # Only include new content in plain text
        html_part = MIMEText(html, 'html')

        # Add both parts to the message
        message.attach(text_part)
        message.attach(html_part)

        # Encode and create the draft
        raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8")).decode("utf-8")
        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw_message}}
        ).execute()

        if "id" not in draft:
            return {"status": "error", "error": "No draft ID returned"}

        draft_id = draft["id"]
        
        # Add 'to_review' label
        label_id = get_or_create_label(service, "to_review")
        if label_id:
            service.users().messages().modify(
                userId="me",
                id=draft["message"]["id"],
                body={"addLabelIds": [label_id]},
            ).execute()

        logger.debug(f"Created follow-up draft with id={draft_id}")

        return {
            "status": "ok",
            "draft_id": draft_id,
            "sequence_num": sequence_num,
        }

    except Exception as e:
        logger.error(f"Error in create_followup_draft: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

```

## utils\gmail_service.py
```python
 
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

## utils\main_followups.py
```python
# tests/test_followup_for_hubspot_leads.py
# This script is used to test the follow-up email generation for HubSpot leads.

import sys
import random
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.test_hubspot_leads_service import get_random_lead_id
from scheduling.database import get_db_connection
from scheduling.followup_generation import generate_followup_email_xai
from services.gmail_service import GmailService
from utils.gmail_integration import create_followup_draft
from scheduling.extended_lead_storage import store_lead_email_info
from utils.logging_setup import logger
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from services.leads_service import LeadsService
from config.settings import HUBSPOT_API_KEY

def generate_followup_email_with_injection(lead_id: int, original_email: dict) -> dict:
    """
    Wraps generate_followup_email_xai but specifically places the new follow-up
    text *underneath the top reply* and *above the original email* at the bottom.
    """
    # Call your existing function to fetch the default follow-up structure
    followup = generate_followup_email_xai(
        lead_id=lead_id,
        original_email=original_email
    )
    if not followup:
        return {}

    # The followup already contains everything we need, just return it
    return followup


def create_followup_for_unreplied_leads():
    """
    1. Pull leads from the DB (sequence_num=1).
    2. Check if they replied.
    3. If not replied, generate and store follow-up.
    """
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Initialize services
        data_gatherer = DataGathererService()
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service
        leads_service = LeadsService(data_gatherer)

        # Fetch leads that have a first-sequence email with valid Gmail info
        cursor.execute("""
            SELECT TOP 200
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date,
                sequence_num,
                company_short_name,
                body
            FROM emails
            WHERE sequence_num = 1
              AND gmail_id IS NOT NULL
              AND company_short_name IS NOT NULL
            ORDER BY created_at ASC
        """)

        rows = cursor.fetchall()
        if not rows:
            logger.error("No sequence=1 emails found in the database.")
            return

        logger.info(f"Found {len(rows)} leads. Checking replies & generating follow-ups if needed...")

        gmail_service = GmailService()

        for idx, row in enumerate(rows, start=1):
            (lead_id, email, name, gmail_id, scheduled_date,
             seq_num, company_short_name, body) = row

            logger.info(f"[{idx}] Checking Lead ID: {lead_id} | Email: {email}")

            # Get lead summary from LeadsService (includes company state)
            lead_info = leads_service.get_lead_summary(lead_id)
            if lead_info.get('error'):
                logger.error(f"[{idx}] Failed to get lead info for Lead ID: {lead_id}")
                continue

            # 1) Check if there's a reply in the thread
            replies = gmail_service.search_replies(gmail_id)
            if replies:
                logger.info(f"[{idx}] Lead ID: {lead_id} has replied. Skipping follow-up.")
                continue

            # 2) If no reply, build a dictionary for the original email info
            original_email = {
                'email': email,
                'name': name,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date,
                'company_short_name': company_short_name,
                'body': body,
                'state': lead_info.get('company_state')  # Get state from lead_info
            }

            # 3) Generate the follow-up with your injection in the middle
            followup = generate_followup_email_with_injection(
                lead_id=lead_id,
                original_email=original_email
            )
            if not followup:
                logger.error(f"[{idx}] Failed to generate follow-up for Lead ID: {lead_id}")
                continue

            # 4) Create the Gmail draft
            draft_result = create_followup_draft(
                sender="me",
                to=followup['email'],
                subject=followup['subject'],
                message_text=followup['body'],
                lead_id=str(lead_id),
                sequence_num=followup.get('sequence_num', 2),
                original_html=followup.get('original_html'),
                in_reply_to=followup['in_reply_to']
            )

            # 5) Update DB with the new draft info
            if draft_result.get('draft_id'):
                store_lead_email_info(
                    lead_sheet={
                        'lead_data': {
                            'properties': {'hs_object_id': lead_id},
                            'email': email
                        },
                        'company_data': {
                            'company_short_name': company_short_name
                        }
                    },
                    draft_id=draft_result['draft_id'],
                    scheduled_date=followup['scheduled_send_date'],
                    body=followup['body'],
                    sequence_num=followup.get('sequence_num', 2)
                )
                logger.info(f"[{idx}] Successfully created/stored follow-up draft for Lead ID: {lead_id}")
            else:
                logger.error(f"[{idx}] Failed creating Gmail draft for Lead ID: {lead_id}: "
                             f"{draft_result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error during follow-up creation: {str(e)}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    create_followup_for_unreplied_leads()

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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

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

def fetch_website_html(url: str, timeout: int = 10, retries: int = 3) -> str:
    """
    Fetch HTML content from a website with retries and better error handling.
    
    Args:
        url: The URL to fetch
        timeout: Timeout in seconds for each attempt
        retries: Number of retry attempts
    """
    logger = logging.getLogger(__name__)
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(
            url,
            timeout=timeout,
            verify=False,  # Skip SSL verification
            allow_redirects=True
        )
        response.raise_for_status()
        return response.text
    
    except requests.exceptions.ConnectTimeout:
        logger.warning(f"Connection timed out for {url} - skipping website fetch")
        return ""
    except requests.exceptions.ReadTimeout:
        logger.warning(f"Read timed out for {url} - skipping website fetch")
        return ""
    except requests.exceptions.SSLError:
        # Try again without SSL
        logger.warning(f"SSL error for {url} - attempting without SSL verification")
        try:
            response = session.get(
                url.replace('https://', 'http://'),
                timeout=timeout,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url} without SSL: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return ""
    finally:
        session.close()

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

def clean_html(html_text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    import re
    from html import unescape
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    # Decode HTML entities
    clean_text = unescape(clean_text)
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

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

def analyze_auto_reply(body: str, subject: str) -> Dict[str, str]:
    """
    Analyze auto-reply email to extract contact transition information.
    
    Args:
        body (str): The email body text
        subject (str): The email subject line
        
    Returns:
        Dict with structured contact transition information:
        {
            'original_person': str,
            'new_contact': str,
            'new_email': str,
            'new_title': str,
            'phone': str,
            'company': str,
            'reason': str,
            'permanent': str
        }
    """
    prompt = f"""
Analyze this auto-reply email and extract the following information with precision:

1. **ORIGINAL PERSON**: Who sent the auto-reply?
2. **NEW CONTACT**: Who is the new person to contact? If multiple people are listed, only use the first one mentioned.
3. **NEW EMAIL**: What is their new email address? If multiple emails are listed, only use the first one that matches the first new contact.
4. **NEW TITLE**: What is their job title/role?
5. **PHONE**: Any phone number provided?
6. **COMPANY**: Company name if mentioned
7. **REASON**: Why is the original person no longer available? (Retired, Left Company, etc.)
8. **PERMANENT**: Is this a permanent change? (Yes/No/Unknown)

CRITICAL RULES:
- Only extract information explicitly stated in the message
- Use 'Unknown' if information is not clearly provided
- Do not make assumptions
- For emails, only include if properly formatted (user@domain.com)
- If multiple contacts are listed, only extract info for the first person mentioned

Email Subject: {subject}
Email Body:
{body}

Format your response with these exact headings:
ORIGINAL PERSON:
[Answer]

NEW CONTACT:
[Answer]

NEW EMAIL:
[Answer]

NEW TITLE:
[Answer]

PHONE:
[Answer]

COMPANY:
[Answer]

REASON:
[Answer]

PERMANENT:
[Answer]
"""

    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing auto-reply emails and extracting "
                        "contact information changes. Only return verified information, "
                        "use 'Unknown' if not certain."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_NAME,
            "temperature": 0.0
        }

        response = _send_xai_request(payload)
        logger.debug(f"Raw xAI response for contact extraction:\n{response}")

        # Parse the response
        result = {
            'original_person': '',
            'new_contact': '',
            'new_email': '',
            'new_title': 'Unknown',
            'phone': 'Unknown',
            'company': 'Unknown',
            'reason': 'Unknown',
            'permanent': 'Unknown'
        }

        sections = {
            'original_person': re.search(r'ORIGINAL PERSON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_contact': re.search(r'NEW CONTACT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_email': re.search(r'NEW EMAIL:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_title': re.search(r'NEW TITLE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'phone': re.search(r'PHONE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'company': re.search(r'COMPANY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'reason': re.search(r'REASON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'permanent': re.search(r'PERMANENT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
        }

        for key, match in sections.items():
            if match:
                result[key] = match.group(1).strip().split('\n')[0].strip()

        # Validate email format
        if result['new_email'] and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', result['new_email']):
            result['new_email'] = ''

        logger.debug(f"Parsed contact info: {result}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing auto-reply: {str(e)}")
        return None

def analyze_employment_change(body: str, subject: str) -> Dict[str, str]:
    """
    Analyze an employment change notification email to extract relevant information.
    Similar to analyze_auto_reply but specifically focused on employment changes.
    """
    try:
        # Clean HTML if present
        if '<html>' in body:
            body = clean_html(body)

        prompt = (
            "Analyze this employment change notification email and extract the following information. "
            "Use 'Unknown' if information is not clearly stated.\n\n"
            "Email Subject: " + subject + "\n"
            "Email Body: " + body + "\n\n"
            "Extract and format the information as follows:\n"
            "ORIGINAL PERSON:\n[Name of person who left]\n\n"
            "NEW CONTACT:\n[Name of new contact person]\n\n"
            "NEW EMAIL:\n[Email of new contact]\n\n"
            "NEW TITLE:\n[Title of new contact]\n\n"
            "PHONE:\n[Phone number]\n\n"
            "COMPANY:\n[Company name]\n\n"
            "REASON:\n[Reason for change]\n\n"
            "PERMANENT:\n[Is this a permanent change? Yes/No/Unknown]"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing employment change notifications "
                        "and extracting contact information changes. Only return verified "
                        "information, use 'Unknown' if not certain."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_NAME,
            "temperature": 0.0
        }

        response = _send_xai_request(payload)
        logger.debug(f"Raw xAI response for employment change analysis:\n{response}")

        # Parse the response using the same structure as analyze_auto_reply
        result = {
            'original_person': '',
            'new_contact': '',
            'new_email': '',
            'new_title': 'Unknown',
            'phone': 'Unknown',
            'company': 'Unknown',
            'reason': 'Unknown',
            'permanent': 'Unknown'
        }

        sections = {
            'original_person': re.search(r'ORIGINAL PERSON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_contact': re.search(r'NEW CONTACT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_email': re.search(r'NEW EMAIL:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_title': re.search(r'NEW TITLE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'phone': re.search(r'PHONE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'company': re.search(r'COMPANY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'reason': re.search(r'REASON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'permanent': re.search(r'PERMANENT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
        }

        for key, match in sections.items():
            if match:
                result[key] = match.group(1).strip().split('\n')[0].strip()

        # Validate email format
        if result['new_email'] and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', result['new_email']):
            result['new_email'] = ''

        logger.debug(f"Parsed employment change info: {result}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing employment change: {str(e)}")
        return None
```
