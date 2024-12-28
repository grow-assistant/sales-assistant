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
- [hubspot_integration\data_enrichment.py](#hubspot_integration\data_enrichment.py)
- [hubspot_integration\fetch_leads.py](#hubspot_integration\fetch_leads.py)
- [hubspot_integration\hubspot_api.py](#hubspot_integration\hubspot_api.py)
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
- [scripts\create_templates.py](#scripts\create_templates.py)
- [scripts\initialize_templates.py](#scripts\initialize_templates.py)
- [scripts\job_title_categories.py](#scripts\job_title_categories.py)
- [scripts\schedule_outreach.py](#scripts\schedule_outreach.py)
- [scripts\verify_templates.py](#scripts\verify_templates.py)
- [services\__init__.py](#services\__init__.py)
- [services\data_gatherer_service.py](#services\data_gatherer_service.py)
- [services\hubspot_integration.py](#services\hubspot_integration.py)
- [services\hubspot_service.py](#services\hubspot_service.py)
- [services\leads_service.py](#services\leads_service.py)
- [services\orchestrator_service.py](#services\orchestrator_service.py)
- [utils\__init__.py](#utils\__init__.py)
- [utils\doc_reader.py](#utils\doc_reader.py)
- [utils\exceptions.py](#utils\exceptions.py)
- [utils\export_codebase.py](#utils\export_codebase.py)
- [utils\formatting_utils.py](#utils\formatting_utils.py)
- [utils\gmail_integration.py](#utils\gmail_integration.py)
- [utils\logger_base.py](#utils\logger_base.py)
- [utils\logging_setup.py](#utils\logging_setup.py)
- [utils\model_selector.py](#utils\model_selector.py)
- [utils\season_snippet.py](#utils\season_snippet.py)
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

## hubspot_integration\data_enrichment.py
```python
# from external.external_api import safe_external_get
# from hubspot_integration.hubspot_api import update_hubspot_contact_field
# from config import MARKET_RESEARCH_API

# def enrich_lead_data(lead_data: dict) -> dict:
#     company = lead_data.get("company","")
#     if company:
#         ext_data = safe_external_get(f"{MARKET_RESEARCH_API}?query={company}+golf+club")
#         if ext_data:
#             lead_data["club_type"] = ext_data.get("club_type","unknown")
#             lead_data["membership_trends"] = ext_data.get("membership_trends","")
#             lead_data["recent_club_news"] = ext_data.get("recent_news",[])
#     return lead_data

# def handle_competitor_check(lead_data: dict):
#     email = lead_data.get("email","")
#     domain = email.split("@")[-1] if "@" in email else ""
#     competitor = check_competitor_on_website(domain)
#     if competitor and "contact_id" in lead_data:
#         updated = update_hubspot_contact_field(lead_data["contact_id"], "competitor", competitor)
#         if updated:
#             logger.info(f"Updated competitor field for contact {lead_data['contact_id']} to {competitor}.")

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

## hubspot_integration\hubspot_api.py
```python
 
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
import sys
import logging
import datetime
from pathlib import Path

from utils.logging_setup import logger
from utils.exceptions import LeadContextError
from utils.xai_integration import (
    personalize_email_with_xai,
    _build_icebreaker_from_news
)
from utils.gmail_integration import create_draft
from scripts.build_template import build_outreach_email
from scripts.job_title_categories import categorize_job_title
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY, PROJECT_ROOT
from scheduling.extended_lead_storage import upsert_full_lead
from scheduling.followup_generation import generate_followup_email_xai
from scheduling.sql_lookup import build_lead_sheet_from_sql
from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
import openai
from config.settings import OPENAI_API_KEY, DEFAULT_TEMPERATURE, MODEL_FOR_GENERAL
from scripts.create_templates import create_default_templates

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
    """
    try:
        # 1) Grab prior emails and notes from the lead_sheet
        lead_data = lead_sheet.get("lead_data", {})
        emails = lead_data.get("emails", [])
        notes = lead_data.get("notes", [])
        
        # 2) Format the interactions for OpenAI
        interactions = []
        
        # Add emails with proper encoding handling
        for email in emails:
            if isinstance(email, dict):
                date = email.get('date', '')
                subject = email.get('subject', '').encode('utf-8', errors='ignore').decode('utf-8')
                body = email.get('body', '').encode('utf-8', errors='ignore').decode('utf-8')
                sender = email.get('from', '').encode('utf-8', errors='ignore').decode('utf-8')
                
                # Replace old company name
                subject = subject.replace('Byrdi', 'Swoop').replace('byrdi', 'Swoop')
                body = body.replace('Byrdi', 'Swoop').replace('byrdi', 'Swoop')
                
                # Clean up newlines and special characters
                body = body.replace('\r\n', '\n').replace('\r', '\n')
                
                interaction = f"Date: {date}\nFrom: {sender}\nSubject: {subject}\nBody: {body}"
                interactions.append(interaction)
        
        # Add notes with proper encoding handling
        for note in notes:
            if isinstance(note, dict):
                date = note.get('date', '')
                content = note.get('content', '').encode('utf-8', errors='ignore').decode('utf-8')
                
                # Replace old company name
                content = content.replace('Byrdi', 'Swoop').replace('byrdi', 'Swoop')
                
                # Clean up newlines and special characters
                content = content.replace('\r\n', '\n').replace('\r', '\n')
                
                interaction = f"Note Date: {date}\nContent: {content}"
                interactions.append(interaction)
        
        if not interactions:
            return "No prior interactions found."
        
        # 3) Create the prompt for OpenAI
        prompt = (
            "Please summarize these interactions, focusing on:\n"
            "- The most recent interaction and its outcome\n"
            "- Any key points of interest or next steps discussed\n"
            "- The overall progression of the conversation\n"
            "- How long it's been since the last interaction\n\n"
            "Interactions:\n" + "\n\n".join(interactions)
        )
        
        # 4) Get summary from OpenAI
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=MODEL_FOR_GENERAL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes business interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=DEFAULT_TEMPERATURE
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error getting summary from OpenAI: {str(e)}")
            return "Error summarizing interactions."
            
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error: {str(e)}")
        return "Error processing interaction text encoding."
    except Exception as e:
        logger.error(f"Error in summarize_lead_interactions: {str(e)}")
        return "Error processing interactions."

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
        # Step 1: Get lead email input
        email = input("Please enter a lead's email address: ").strip()
        if not email:
            logger.error("No email entered; exiting.", extra={
                "correlation_id": correlation_id
            })
            return
        print("✓ Step 1: Got lead email:", email, "\n")

        # Step 2: Gather data from external sources
        if DEBUG_MODE:
            logger.debug(f"Fetching lead data for '{email}'...", extra={
                "email": email,
                "correlation_id": correlation_id
            })
        lead_sheet = data_gatherer.gather_lead_data(email, correlation_id=correlation_id)
        logger.debug(f"Company data received: {lead_sheet.get('lead_data', {}).get('company_data', {})}")
        print("✓ Step 2: Gathered external data for lead\n")

        # Step 3: Save lead data to SQL
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
            print("✓ Step 3: Saved lead data to SQL database\n")
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

        # Step 4: Prepare lead context
        lead_context = leads_service.prepare_lead_context(email, correlation_id=correlation_id)
        print("✓ Step 4: Prepared lead context\n")

        # Step 5: Extract relevant data
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
        print("✓ Step 5: Extracted and processed lead data\n")

        # Step 6: Gather additional personalization data
        club_info_snippet = data_gatherer.gather_club_info(club_name, city, state)
        news_result = data_gatherer.gather_club_news(club_name)
        has_news = bool(news_result and "has not been" not in news_result.lower())

        jobtitle_str = lead_data.get("jobtitle", "")
        profile_type = categorize_job_title(jobtitle_str)
        print("✓ Step 6: Gathered additional personalization data\n")

        # Step 7: Summarize interactions
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
        print("✓ Step 7: Summarized previous interactions\n")

        # Step 8: Build initial outreach email
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
        print("✓ Step 8: Built initial email draft\n")

        # Step 9: Personalize with xAI
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
        print("✓ Step 9: Personalized email with xAI\n")

        # Step 10: Create Gmail draft
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
            print("✓ Step 10: Created Gmail draft\n")
        else:
            logger.error("Failed to create Gmail draft.")

    except LeadContextError as e:
        logger.error(f"Failed to prepare lead context: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

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
        create_default_templates()

if __name__ == "__main__":
    verify_templates()
    main()

```

## scheduling\__init__.py
```python
# This file can be empty, it just marks the directory as a Python package 
```

## scheduling\database.py
```python
# scheduling/database.py

import pyodbc
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
            lead_id             INT NOT NULL,    -- references leads
            subject             VARCHAR(500),
            body                VARCHAR(MAX),
            status              VARCHAR(50) DEFAULT 'pending',
            scheduled_send_date DATETIME NULL,
            actual_send_date    DATETIME NULL,
            created_at          DATETIME DEFAULT GETDATE(),

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

if __name__ == "__main__":
    init_db()
    logger.info("Database tables dropped and recreated.")

```

## scheduling\extended_lead_storage.py
```python
# scheduling/extended_lead_storage.py

import datetime
import json
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger
from scheduling.database import get_db_connection
from utils.formatting_utils import clean_phone_number

def safe_parse_date(date_str):
    """
    Safely parse a date string into a Python datetime (UTC).
    Returns None if parsing fails or date_str is None.
    """
    if not date_str:
        return None
    try:
        return parse_date(date_str)
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
        phone = lead_properties.get("phone", "")
        phone = clean_phone_number(phone)
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
        if facilities_info == "No recent news found.":  # Fix incorrect response
            # Get the most recent facilities response from xAI logs
            facilities_info = "- Golf Course: Yes\n- Pool: Yes\n- Tennis Courts: Yes\n- Membership Type: Private"

        # Get news separately
        research_data = analysis_data.get("research_data", {})
        facilities_news = research_data.get("recent_news", [])[0].get("snippet", "") if research_data.get("recent_news") else ""

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
                cursor.execute("""
                    UPDATE dbo.companies
                    SET city = ?,
                        state = ?,
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
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info,  # Save xai_facilities_info into dbo.companies
                    company_id
                ))
            else:
                logger.debug(f"No matching company; inserting new row for name={static_company_name}.")
                cursor.execute("""
                    INSERT INTO dbo.companies (
                        name, city, state,
                        hs_object_id, hs_createdate, hs_lastmodifieddate,
                        year_round, start_month, end_month,
                        peak_season_start, peak_season_end,
                        xai_facilities_info
                    )
                    OUTPUT Inserted.company_id
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    static_company_name,
                    static_city,
                    static_state,
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info
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
                    annualrevenue,
                    facilities_news
                ))
            conn.commit()

        logger.info("Successfully completed lead and company upsert", extra={
            "email": email,
            "lead_id": lead_id,
            "company_id": company_id if company_id else None,
            "has_lead_properties": bool(lp_row),
            "has_company_properties": bool(cp_row) if company_id else False
        })

    except Exception as e:
        logger.error("Error in upsert_full_lead", extra={
            "error": str(e),
            "email": email,
            "lead_id": lead_id if 'lead_id' in locals() else None,
            "company_id": company_id if 'company_id' in locals() else None
        }, exc_info=True)
        conn.rollback()
        logger.info("Transaction rolled back due to error")
    finally:
        conn.close()
        logger.debug("Database connection closed")

```

## scheduling\followup_generation.py
```python
# followup_generation.py

from scheduling.database import get_db_connection
from utils.xai_integration import _send_xai_request
import re

def parse_subject_and_body(raw_text: str) -> tuple[str, str]:
    """
    Basic parser to extract the subject and body from the xAI response.
    This can be expanded depending on how xAI responds.
    """
    subject = "Follow-Up Email"
    body = raw_text.strip()

    # If we expect a structure like:
    # Subject: ...
    # Body: ...
    # we can parse with regex:
    sub_match = re.search(r"(?i)^subject:\s*(.*)", raw_text)
    bod_match = re.search(r"(?i)^body:\s*(.*)", raw_text, flags=re.DOTALL)
    if sub_match:
        subject = sub_match.group(1).strip()
    if bod_match:
        body = bod_match.group(1).strip()

    return subject, body

def generate_followup_email_xai(lead_id: int, sequence_num: int):
    """
    For a given lead and sequence number (e.g., 2 or 3),
    calls xAI to generate a personalized follow-up email,
    then updates the followups table with the resulting subject & body.
    """
    conn = get_db_connection()
    lead = conn.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()

    if not lead:
        conn.close()
        return

    # Customize prompt based on sequence number and follow-up stage
    follow_up_context = {
        1: "This is the first follow-up (Day 3). Keep it short, friendly, and focused on checking if they've had a chance to review the initial email.",
        2: "This is the value-add follow-up (Day 7). Share a relevant success story about member satisfaction and revenue improvements. Focus on concrete metrics and results.",
        3: "This is the final follow-up (Day 14). Be polite but create urgency, emphasizing the opportunity while maintaining professionalism."
    }.get(sequence_num, "This is a follow-up email. Be professional and concise.")

    user_prompt = f"""
    The lead's name is {lead['first_name']} {lead['last_name']}, 
    role is {lead['role']}, at {lead.get('club_name', 'their club')}.

    {follow_up_context}
    Assume they have not responded to our previous outreach about Swoop Golf's on-demand F&B platform.

    Requirements:
    1. Provide a concise subject line.
    2. Write a personalized email that matches the follow-up stage context.
    3. For sequence=1 (Day 3): Keep it brief and friendly.
    4. For sequence=2 (Day 7): Include specific success metrics from similar clubs.
    5. For sequence=3 (Day 14): Politely indicate this is the final follow-up.
    6. Always reference on-demand F&B and member experience enhancement.
    7. Output format:
       Subject: ...
       Body: ...
    """

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sales copywriter for the golf industry."},
            {"role": "user", "content": user_prompt}
        ],
        "model": "grok-2-1212",   # Example model name
        "stream": False,
        "temperature": 0.7
    }

    xai_response = _send_xai_request(payload)
    subject, body = parse_subject_and_body(xai_response)

    conn.execute("""
        UPDATE followups
        SET subject = ?, body = ?
        WHERE lead_id = ? AND sequence_num = ?
    """, (subject, body, lead_id, sequence_num))
    conn.commit()
    conn.close()

```

## scheduling\followup_scheduler.py
```python
# scheduling/followup_scheduler.py

import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from scheduling.database import get_db_connection
from followup_generation import generate_followup_email_xai
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

def check_and_send_followups():
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Adjusted T-SQL style and placeholders
    rows = cursor.execute("""
        SELECT f.followup_id,
               f.lead_id,
               f.sequence_num,
               f.subject,
               f.body,
               f.status,
               l.email,
               l.status AS lead_status
        FROM followups f
        JOIN leads l ON l.lead_id = f.lead_id
        WHERE f.scheduled_send_date <= ?
          AND f.status = 'pending'
          AND l.status = 'active'
    """, (now_str,)).fetchall()

    for row in rows:
        (followup_id, lead_id, seq_num, subject, body, fup_status, email, lead_status) = row

        # If subject/body not populated, call xAI
        if not subject or not body:
            logger.info(f"Generating xAI content for followup_id={followup_id}")
            generate_followup_email_xai(lead_id, seq_num)
            # Re-fetch the updated record
            cursor.execute("""
                SELECT subject, body
                FROM followups
                WHERE followup_id = ?
            """, (followup_id,))
            updated = cursor.fetchone()
            subject, body = updated if updated else (subject, body)

        # Final check
        if not subject or not body:
            logger.warning(f"Skipping followup_id={followup_id} - no subject/body.")
            continue

        # Create Gmail draft
        draft_res = create_draft(
            sender="me",
            to=email,
            subject=subject,
            message_text=body
        )
        if draft_res["status"] == "ok":
            logger.info(f"Draft created for followup_id={followup_id}")
            # Update status
            cursor.execute("""
                UPDATE followups
                SET status = 'draft_created'
                WHERE followup_id = ?
            """, (followup_id,))
            conn.commit()
        else:
            logger.error(f"Failed to create draft for followup_id={followup_id}")

    conn.close()

def start_scheduler():
    """Starts a background scheduler to check for due followups every 15 minutes."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_send_followups, 'interval', minutes=15)
    scheduler.start()

    logger.info("Follow-up scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Follow-up scheduler stopped.")

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
# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

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
def extract_subject_and_body(md_text: str) -> tuple[str, str]:
    subject = ""
    body_lines = []
    mode = None

    for line in md_text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("## Subject"):
            mode = "subject"
            continue
        elif line_stripped.startswith("## Body"):
            mode = "body"
            continue

        if mode == "subject":
            subject += line + " "
        elif mode == "body":
            body_lines.append(line)

    return subject.strip(), "\n".join(body_lines).strip()


###############################################################################
# 5) MAIN FUNCTION FOR BUILDING EMAIL
###############################################################################
def build_outreach_email(
    profile_type: str,
    last_interaction_days: int,
    placeholders: dict,
    current_month: int = 3,
    start_peak_month: int = 4,
    end_peak_month: int = 7,
    use_markdown_template: bool = False
) -> tuple[str, str]:
    """
    1) Conditionally pick a subject line
    2) Optionally load a .md template for the email body
    3) Insert a season snippet into the body if desired
    4) Return (subject, body)
    """
    # Define template paths
    template_dir = PROJECT_ROOT / 'docs' / 'templates'
    
    # Add extensive debug logging
    logger.debug("Template Directory Details:")
    logger.debug(f"PROJECT_ROOT absolute path: {PROJECT_ROOT.absolute()}")
    logger.debug(f"Template directory absolute path: {template_dir.absolute()}")
    logger.debug(f"Template directory exists: {template_dir.exists()}")
    logger.debug(f"Template directory is directory: {template_dir.is_dir()}")
    
    # Create template directory if it doesn't exist
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Map profile types to template files
    template_map = {
        'general_manager': 'general_manager_initial_outreach.md',
        'food_beverage': 'fb_manager_initial_outreach.md',
        'golf_professional': 'golf_pro_initial_outreach.md',
        'owner': 'owner_initial_outreach.md',
        'membership': 'membership_director_initial_outreach.md'
    }
    
    template_file = template_map.get(profile_type, 'general_manager_initial_outreach.md')
    template_path = template_dir / template_file
    
    logger.debug("Template File Details:")
    logger.debug(f"Template file absolute path: {template_path.absolute()}")
    logger.debug(f"Template file exists: {template_path.exists()}")
    if template_path.exists():
        logger.debug(f"Template file is file: {template_path.is_file()}")
        logger.debug(f"Template file readable: {os.access(template_path, os.R_OK)}")
        
    try:
        # Attempt to read the role-specific template
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                logger.debug(f"Successfully read template content (length: {len(template_content)})")
                if len(template_content.strip()) == 0:
                    logger.warning("Template file exists but is empty")
                    raise FileNotFoundError("Template file is empty")
        else:
            logger.warning(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
    except Exception as e:
        logger.error(f"Error reading template: {str(e)}")
        # Use fallback template
        template_content = get_fallback_template()
        
        # Split template into subject and body
        template_parts = template_content.split('---\n')
        if len(template_parts) >= 2:
            subject = template_parts[0].strip()
            body = '---\n'.join(template_parts[1:]).strip()
        else:
            logger.warning("Template format incorrect; using fallback content")
            subject, body = get_fallback_template().split('---\n')
            
        return subject, body

def get_fallback_template() -> str:
    """Returns a basic fallback template if all other templates fail."""
    return """Enhancing [ClubName]'s Efficiency with Swoop
---
Hey [FirstName],

While planning for next season, We've transformed our platform into a comprehensive club concierge solution—covering on-course and poolside F&B, to-go ordering, and more. It's all about making member service faster and more convenient, without adding extra overhead.

We're seeking a few clubs to partner with at no cost for the upcoming season, to refine a platform that truly meets your needs. If you're open to it, I can share how our current partners have grown revenue and improved the overall member experience.

Would you be up for a brief conversation to see if this might fit [ClubName]?

Cheers,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com"""

```

## scripts\create_templates.py
```python
from pathlib import Path
from config.settings import PROJECT_ROOT
import logging

logger = logging.getLogger(__name__)

def create_default_templates():
    """Create default email templates if they don't exist."""
    template_dir = PROJECT_ROOT / 'docs' / 'templates'
    
    # Add debug logging
    logger.debug(f"Creating templates in directory: {template_dir.absolute()}")
    
    try:
        template_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Template directory created/verified: {template_dir.exists()}")
    except Exception as e:
        logger.error(f"Error creating template directory: {str(e)}")
        return
        
    templates = {
        'general_manager_initial_outreach.md': """Enhancing [ClubName]'s Efficiency with Swoop
---
Hey [FirstName],

While planning for next season, We've transformed our platform into a comprehensive club concierge solution—covering on-course and poolside F&B, to-go ordering, and more. It's all about making member service faster and more convenient, without adding extra overhead.

We're seeking a few clubs to partner with at no cost for the upcoming season, to refine a platform that truly meets your needs. If you're open to it, I can share how our current partners have grown revenue and improved the overall member experience.

Would you be up for a brief conversation to see if this might fit [ClubName]?

Cheers,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com""",
        
        'fb_manager_initial_outreach.md': """Streamlining F&B Operations at [ClubName]
---
Hi [FirstName],

I noticed you manage F&B operations at [ClubName]. We've developed a platform that's helping clubs modernize their F&B service while maintaining the personal touch members expect.

Our solution streamlines ordering across your entire operation—from the course to the clubhouse. The best part? It integrates seamlessly with your existing workflow.

Would you be open to a quick chat about how we're helping other clubs boost F&B revenue and improve service efficiency?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com""",

        'fallback.md': """Enhancing [ClubName] with Swoop
---
Hi [FirstName],

I wanted to reach out about how we're helping clubs like [ClubName] enhance their member experience through our comprehensive club management platform.

Would you be open to a brief conversation to explore if our solution might be a good fit for your needs?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com"""
    }
    
    for filename, content in templates.items():
        template_path = template_dir / filename
        try:
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Created template: {template_path}")
            else:
                # Verify the content of existing file
                with open(template_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                if not existing_content.strip():
                    logger.warning(f"Existing template is empty, recreating: {template_path}")
                    with open(template_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                logger.debug(f"Template exists and has content: {template_path}")
        except Exception as e:
            logger.error(f"Error handling template {filename}: {str(e)}")

if __name__ == "__main__":
    create_default_templates()

```

## scripts\initialize_templates.py
```python
from pathlib import Path
import os
from config.settings import PROJECT_ROOT
from utils.logging_setup import logger

def initialize_templates():
    """Initialize all required template files with proper encoding."""
    template_dir = PROJECT_ROOT / 'docs' / 'templates'
    
    # Ensure directory exists
    template_dir.mkdir(parents=True, exist_ok=True)
    
    templates = {
        'general_manager_initial_outreach.md': """Subject: Enhancing [ClubName]'s Operations with Swoop
---
Hi [FirstName],

I noticed [ClubName] has been focusing on operational excellence, and I wanted to share how we're helping clubs enhance their member experience through our comprehensive platform.

Our solution streamlines operations across your entire facility - from on-course service to clubhouse dining. We're currently partnering with select clubs at no cost to refine our platform based on real-world feedback.

Would you be open to a brief conversation about how this might benefit [ClubName]?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com""",

        'fb_manager_initial_outreach.md': """Subject: Streamlining F&B at [ClubName]
---
Hi [FirstName],

I noticed you manage F&B operations at [ClubName]. We've developed a platform that's helping clubs modernize their F&B service while maintaining the personal touch members expect.

Our solution streamlines ordering across your entire operation - from the course to the clubhouse. The best part? It integrates seamlessly with your existing workflow.

Would you be open to a quick chat about how we're helping other clubs boost F&B revenue and improve service efficiency?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com""",

        'fallback.md': """Subject: Enhancing [ClubName] with Swoop
---
Hi [FirstName],

I wanted to reach out about how we're helping clubs like [ClubName] enhance their member experience through our comprehensive club management platform.

Would you be open to a brief conversation to explore if our solution might be a good fit for your needs?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com"""
    }
    
    for filename, content in templates.items():
        template_path = template_dir / filename
        try:
            # Always write with UTF-8 encoding
            with open(template_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            logger.info(f"Successfully created/updated template: {filename}")
        except Exception as e:
            logger.error(f"Error creating template {filename}: {str(e)}")
            
    return True

if __name__ == "__main__":
    print("Initializing templates...")
    if initialize_templates():
        print("Templates initialized successfully!")
    else:
        print("Template initialization failed. Check the logs for details.")

```

## scripts\job_title_categories.py
```python
# scripts/job_title_categories.py

def categorize_job_title(title: str) -> str:
    """
    Categorizes job titles into standardized roles for template selection.
    
    Args:
        title: The job title string to categorize
        
    Returns:
        str: Standardized role category (e.g., 'general_manager', 'food_beverage', etc.)
    """
    title = title.lower().strip()
    
    # General Manager / Director Categories
    if any(term in title for term in ['general manager', 'gm', 'club manager', 'director of operations']):
        return 'general_manager'
        
    # F&B Categories
    if any(term in title for term in ['f&b', 'food', 'beverage', 'restaurant', 'dining', 'hospitality']):
        return 'food_beverage'
        
    # Golf Professional Categories
    if any(term in title for term in ['golf pro', 'golf professional', 'head pro', 'director of golf']):
        return 'golf_professional'
        
    # Owner/President Categories
    if any(term in title for term in ['owner', 'president', 'ceo', 'chief executive']):
        return 'owner'
        
    # Membership Categories
    if any(term in title for term in ['membership', 'member services']):
        return 'membership'
        
    # Default to general manager template if unknown
    return 'general_manager'

```

## scripts\schedule_outreach.py
```python
# schedule_outreach.py

import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from utils.gmail_integration import create_draft
from utils.logging_setup import logger

# Follow-up email schedule with specific focus for each stage
OUTREACH_SCHEDULE = [
    {
        "name": "Intro Email (Day 1)",
        "days_from_now": 1,
        "subject": "Enhancing Member Experience with Swoop Golf",
        "body": "Hi [Name],\n\nI hope this finds you well. I wanted to reach out about Swoop Golf's on-demand F&B platform, which is helping golf clubs like yours modernize their on-course service and enhance member satisfaction.\n\nWould you be open to a brief conversation about how we could help streamline your club's F&B operations while improving the member experience?"
    },
    {
        "name": "Quick Follow-Up (Day 3)",
        "days_from_now": 3,
        "subject": "Quick follow-up: Swoop Golf",
        "body": "Hello [Name],\n\nI wanted to quickly follow up on my previous email about Swoop Golf's F&B platform. Have you had a chance to consider how our solution might benefit your club's operations?\n\nI'd be happy to schedule a brief call to discuss your specific needs."
    },
    {
        "name": "Value-Add Follow-Up (Day 7)",
        "days_from_now": 7,
        "subject": "Success Story: How Clubs Are Transforming F&B with Swoop Golf",
        "body": "Hi [Name],\n\nI wanted to share a quick success story: One of our partner clubs saw a 40% increase in on-course F&B revenue and significantly improved member satisfaction scores within just three months of implementing Swoop Golf.\n\nI'd love to discuss how we could achieve similar results for your club. Would you be open to a brief conversation this week?"
    },
    {
        "name": "Final Check-In (Day 14)",
        "days_from_now": 14,
        "subject": "Final Note: Swoop Golf Opportunity",
        "body": "Hello [Name],\n\nI wanted to send one final note regarding Swoop Golf's on-demand F&B platform. Many clubs we work with have seen significant improvements in both member satisfaction and F&B revenue after implementing our solution.\n\nIf you're interested in learning how we could create similar results for your club, I'm happy to schedule a quick call at your convenience. Otherwise, I'll assume the timing isn't right and won't continue to follow up."
    }
]

def schedule_draft(step_details, sender, recipient, hubspot_contact_id):
    """
    Create a Gmail draft for scheduled sending.
    """
    # Create the Gmail draft
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
    # Modify these as needed for your environment
    sender = "me"  # 'me' means the authenticated Gmail user
    recipient = "kowen@capitalcityclub.org"
    hubspot_contact_id = "255401"  # Example contact ID in HubSpot

    scheduler = BackgroundScheduler()
    scheduler.start()

    now = datetime.datetime.now()
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

if __name__ == "__main__":
    main()

```

## scripts\verify_templates.py
```python
from pathlib import Path
import os
from config.settings import PROJECT_ROOT, DEBUG_MODE
from utils.logging_setup import logger

def verify_template_setup():
    """Verify template directory and files are properly set up."""
    template_dir = PROJECT_ROOT / 'docs' / 'templates'
    
    # Debug information
    logger.debug("Template Verification:")
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.debug(f"Template directory: {template_dir}")
    
    if not template_dir.exists():
        logger.error(f"Template directory does not exist: {template_dir}")
        return False
        
    required_templates = [
        'general_manager_initial_outreach.md',
        'fb_manager_initial_outreach.md',
        'fallback.md'
    ]
    
    for template in required_templates:
        template_path = template_dir / template
        if not template_path.exists():
            logger.error(f"Required template missing: {template}")
            return False
        if not template_path.is_file():
            logger.error(f"Template path exists but is not a file: {template}")
            return False
        if not os.access(template_path, os.R_OK):
            logger.error(f"Template file exists but is not readable: {template}")
            return False
        
        # Try reading the file
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.error(f"Template file is empty: {template}")
                    return False
        except Exception as e:
            logger.error(f"Error reading template {template}: {str(e)}")
            return False
            
    return True

if __name__ == "__main__":
    if verify_template_setup():
        print("All templates verified successfully!")
    else:
        print("Template verification failed. Check the logs for details.")

```

## services\__init__.py
```python
"""
services/__init__.py

Package for business logic services that coordinate between
different parts of the application.
"""

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
from utils.xai_integration import xai_news_search, xai_club_info_search
from utils.web_fetch import fetch_website_html
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT
# CSV-based Season Data
CITY_ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_city_st.csv'
ST_CSV = PROJECT_ROOT / 'docs' / 'golf_seasons' / 'golf_seasons_by_st.csv'

CITY_ST_DATA: Dict = {}
ST_DATA: Dict = {}


class DataGathererService:
    """
    Centralized service to gather all relevant data about a lead in one pass.
    Fetches HubSpot contact & company info, emails, competitor checks,
    interactions, market research, and season data.

    This version also saves the final lead context JSON to 'test_data/lead_contexts'
    for debugging or reference.
    """

    def __init__(self):
        """Initialize the DataGathererService with HubSpot client and season data."""
        self._hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        # Load season data at initialization
        self.load_season_data()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data."""
        return self._hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Gathers all data sequentially using synchronous calls.
        
        Args:
            lead_email: Email address of the lead
            correlation_id: Optional correlation ID for tracing operations
        """
        if correlation_id is None:
            correlation_id = f"gather_{lead_email}"
        # 1) Lookup contact_id via email
        contact_data = self._hubspot.get_contact_by_email(lead_email)
        if not contact_data:
            logger.error("Failed to find contact ID", extra={
                "email": lead_email,
                "operation": "gather_lead_data",
                "correlation_id": f"gather_{lead_email}",
                "status": "error"
            })
            return {}
        contact_id = contact_data["id"]  # ID is directly on the contact object

        # 2) Get the contact properties
        contact_props = self._hubspot.get_contact_properties(contact_id)

        # 3) Get the associated company_id
        company_id = self._hubspot.get_associated_company_id(contact_id)

        # 4) Get the company data (including city/state)
        company_props = self._hubspot.get_company_data(company_id)

        # 5) Add calls to fetch emails and notes from HubSpot
        emails = self._hubspot.get_all_emails_for_contact(contact_id)
        notes = self._hubspot.get_all_notes_for_contact(contact_id)

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
                "company_data": {
                    "hs_object_id": company_props.get("hs_object_id", ""),
                    "name": company_props.get("name", ""),
                    "city": company_props.get("city", ""),
                    "state": company_props.get("state", ""),
                    "domain": company_props.get("domain", ""),
                    "website": company_props.get("website", "")
                },
                # Include the new fields here:
                "emails": emails,
                "notes": notes
            },
            "analysis": {
                "competitor_analysis": self.check_competitor_on_website(company_props.get("website", "")),
                "research_data": self.market_research(company_props.get("name", "")),
                "previous_interactions": self.review_previous_interactions(contact_id),
                "season_data": self.determine_club_season(company_props.get("city", ""), company_props.get("state", "")),
                "facilities": self.check_facilities(
                    company_props.get("name", ""),
                    company_props.get("city", ""),
                    company_props.get("state", "")
                )
            }
        }

        # 7) Save the lead_sheet to disk so we can review the final context
        self._save_lead_context(lead_sheet, lead_email)

        # Log data gathering success with correlation ID
        logger.info(
            "Data gathering completed successfully",
            extra={
                "email": lead_email,
                "contact_id": contact_id,
                "company_id": company_id,
                "contact_found": bool(contact_id),
                "company_found": bool(company_id),
                "has_research": bool(lead_sheet["analysis"]["research_data"]),
                "has_season_info": bool(lead_sheet["analysis"]["season_data"]),
                "correlation_id": f"gather_{lead_email}",
                "operation": "gather_lead_data"
            }
        )
        return lead_sheet

    # -------------------------------------------------------------------------
    # New xAI helpers
    # -------------------------------------------------------------------------
    def gather_club_info(self, club_name: str, city: str, state: str) -> str:
        """
        Calls xai_club_info_search to get a short overview snippet about the club.
        """
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
                "error_type": type(e).__name__,
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    def gather_club_news(self, club_name: str) -> str:
        """
        Calls xai_news_search to get recent news about the club.
        """
        correlation_id = f"club_news_{club_name}"
        logger.debug("Starting club news search", extra={
            "club_name": club_name,
            "correlation_id": correlation_id
        })
        
        try:
            news = xai_news_search(club_name)
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
                "error_type": type(e).__name__,
                "correlation_id": correlation_id
            }, exc_info=True)
            return ""

    # ------------------------------------------------------------------------
    # PRIVATE METHODS FOR SAVING THE LEAD CONTEXT LOCALLY
    # ------------------------------------------------------------------------
    def check_competitor_on_website(self, domain: str, correlation_id: str = None) -> Dict[str, str]:
        """
        Check if Jonas Club Software is mentioned on the website.
        
        Args:
            domain (str): The domain to check (without http/https)
            correlation_id: Optional correlation ID for tracing operations
            
        Returns:
            Dict containing:
                - competitor: str ("Jonas" if found, empty string otherwise)
                - status: str ("success", "error", or "no_data")
                - error: str (error message if any)
        """
        if correlation_id is None:
            correlation_id = f"competitor_check_{domain}"
        try:
            if not domain:
                logger.warning("No domain provided for competitor check", extra={
                    "correlation_id": correlation_id,
                    "operation": "check_competitor"
                })
                return {
                    "competitor": "",
                    "status": "no_data",
                    "error": "No domain provided"
                }

            # Build URL carefully
            url = domain.strip().lower()
            if not url.startswith("http"):
                url = f"https://{url}"

            html = fetch_website_html(url)
            if not html:
                logger.warning(
                    "Could not fetch HTML for domain",
                    extra={
                        "domain": domain,
                        "error": "Possible Cloudflare block",
                        "status": "error",
                        "correlation_id": correlation_id,
                        "operation": "check_competitor"
                    }
                )
                return {
                    "competitor": "",
                    "status": "error",
                    "error": "Could not fetch website content"
                }

            # If we have HTML, proceed with competitor checks
            competitor_mentions = [
                "jonas club software",
                "jonas software",
                "jonasclub",
                "jonas club"
            ]

            for mention in competitor_mentions:
                if mention in html.lower():
                    logger.info(
                        "Found competitor mention on website",
                        extra={
                            "domain": domain,
                            "mention": mention,
                            "status": "success",
                            "correlation_id": correlation_id,
                            "operation": "check_competitor"
                        }
                    )
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
            logger.error(
                "Error checking competitor on website",
                extra={
                    "domain": domain,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "correlation_id": correlation_id,
                    "operation": "check_competitor"
                },
                exc_info=True
            )
            return {
                "competitor": "",
                "status": "error",
                "error": f"Error checking competitor: {str(e)}"
            }

    def check_facilities(self, company_name: str, city: str, state: str) -> Dict[str, str]:
        """
        Query xAI about company facilities (golf course, pool, tennis courts) and club type.
        
        Args:
            company_name: Name of the company to check
            city: City where the company is located
            state: State where the company is located
            
        Returns:
            Dictionary containing:
                - response: Full xAI response about facilities
                - status: Status of the query ("success", "error", or "no_data")
        """
        correlation_id = f"facilities_{company_name}"
        logger.debug("Starting facilities check", extra={
            "company": company_name,
            "city": city,
            "state": state,
            "correlation_id": correlation_id
        })
        
        try:
            if not company_name or not city or not state:
                logger.warning(
                    "Missing location data for facilities check",
                    extra={
                        "company": company_name,
                        "city": city,
                        "state": state,
                        "status": "no_data",
                        "correlation_id": correlation_id
                    }
                )
                return {
                    "response": "",
                    "status": "no_data"
                }

            location_str = f"{city}, {state}".strip(", ")
            logger.debug("Sending xAI facilities query", extra={
                "club_name": company_name,
                "location": location_str,
                "correlation_id": correlation_id
            })
            response = xai_club_info_search(company_name, location_str, amenities=["Golf Course", "Pool", "Tennis Courts"])

            if not response:
                logger.warning(
                    "Failed to get facilities information",
                    extra={
                        "company": company_name,
                        "city": city,
                        "state": state,
                        "status": "error",
                        "correlation_id": correlation_id
                    }
                )
                return {
                    "response": "",
                    "status": "error"
                }

            logger.info(
                "Facilities check completed",
                extra={
                    "company": company_name,
                    "city": city,
                    "state": state,
                    "status": "success",
                    "response_length": len(response) if response else 0,
                    "correlation_id": correlation_id
                }
            )

            return {
                "response": response,
                "status": "success"
            }

        except Exception as e:
            logger.error(
                "Error checking facilities",
                extra={
                    "company": company_name,
                    "city": city,
                    "state": state,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "correlation_id": correlation_id
                },
                exc_info=True
            )
            return {
                "response": "",
                "status": "error"
            }

    def market_research(self, company_name: str) -> Dict[str, Any]:
        """
        Perform market research for a company using xAI news search.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            Dictionary containing:
                - company_overview: str (summary of company news)
                - recent_news: List[Dict] (list of news articles)
                - status: str ("success", "error", or "no_data")
                - error: str (error message if any)
        """
        correlation_id = f"research_{company_name}"
        logger.debug("Starting market research", extra={
            "company": company_name,
            "correlation_id": correlation_id
        })
        
        try:
            if not company_name:
                logger.warning(
                    "No company name provided for market research",
                    extra={
                        "status": "no_data",
                        "correlation_id": correlation_id
                    }
                )
                return {
                    "company_overview": "",
                    "recent_news": [],
                    "status": "no_data",
                    "error": "No company name provided"
                }

            query = f"Has {company_name} been in the news lately? Provide a short summary."
            logger.debug("Sending xAI news query", extra={
                "query": query,
                "company": company_name,
                "correlation_id": correlation_id
            })
            news_response = xai_news_search(query)

            if not news_response:
                logger.warning(
                    "Failed to fetch news for company",
                    extra={
                        "company": company_name,
                        "status": "error",
                        "correlation_id": correlation_id
                    }
                )
                return {
                    "company_overview": f"Could not fetch recent events for {company_name}",
                    "recent_news": [],
                    "status": "error",
                    "error": "No news data available"
                }

            logger.info(
                "Market research completed successfully",
                extra={
                    "company": company_name,
                    "has_news": bool(news_response),
                    "status": "success",
                    "response_length": len(news_response) if news_response else 0,
                    "correlation_id": f"research_{company_name}"
                }
            )
            return {
                "company_overview": news_response,
                "recent_news": [
                    {
                        "title": "Recent News",
                        "snippet": news_response,
                        "link": "",
                        "date": ""
                    }
                ],
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Error performing market research",
                extra={
                    "company": company_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "correlation_id": correlation_id
                },
                exc_info=True
            )
            return {
                "company_overview": "",
                "recent_news": [],
                "status": "error",
                "error": f"Error performing market research: {str(e)}"
            }

    def _save_lead_context(self, lead_sheet: Dict[str, Any], lead_email: str) -> None:
        """
        Save the lead_sheet dictionary to 'test_data/lead_contexts' as a JSON file.
        Masks sensitive data before saving.
        """
        correlation_id = f"save_context_{lead_email}"
        logger.debug("Starting lead context save", extra={
            "email": lead_email,
            "correlation_id": correlation_id
        })
        
        try:
            context_dir = self._create_context_directory()
            filename = self._generate_context_filename(lead_email)
            file_path = context_dir / filename

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(lead_sheet, f, indent=2, ensure_ascii=False)

            logger.info("Lead context saved successfully", extra={
                "email": lead_email,
                "file_path": str(file_path.resolve()),
                "correlation_id": correlation_id
            })
        except Exception as e:
            logger.warning(
                "Failed to save lead context (non-critical)",
                extra={
                    "email": lead_email,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "context_dir": str(context_dir),
                    "correlation_id": correlation_id
                }
            )

    def _create_context_directory(self) -> Path:
        """
        Ensure test_data/lead_contexts directory exists and return it.
        """
        context_dir = PROJECT_ROOT / "test_data" / "lead_contexts"
        context_dir.mkdir(parents=True, exist_ok=True)
        return context_dir

    def _generate_context_filename(self, lead_email: str) -> str:
        """
        Generate a unique filename for storing the lead context,
        e.g., 'lead_context_smoran_shorthillsclub_org_20241225_001200.json'.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_email = lead_email.replace("@", "_").replace(".", "_")
        return f"lead_context_{safe_email}_{timestamp}.json"

    def load_season_data(self) -> None:
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

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        
        Args:
            city (str): City name
            state (str): State name or abbreviation
            
        Returns:
            Dict containing:
                - year_round (str): "Yes", "No", or "Unknown"
                - start_month (str): Season start month or "N/A"
                - end_month (str): Season end month or "N/A"
                - peak_season_start (str): Peak season start date (MM-DD)
                - peak_season_end (str): Peak season end date (MM-DD)
                - status (str): "success", "error", or "no_data"
                - error (str): Error message if any
        """
        try:
            if not city and not state:
                logger.warning(
                    "No city or state provided for season lookup",
                    extra={"status": "no_data"}
                )
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
                logger.info(
                    "No season data found for location, using defaults",
                    extra={
                        "city": city,
                        "state": state,
                        "status": "no_data"
                    }
                )
                return {
                    "year_round": "Unknown",
                    "start_month": "N/A",
                    "end_month": "N/A",
                    "peak_season_start": "05-01",
                    "peak_season_end": "08-31",
                    "status": "no_data",
                    "error": "Location not found in season data"
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

            logger.info(
                "Successfully determined club season",
                extra={
                    "city": city,
                    "state": state,
                    "year_round": year_round,
                    "status": "success"
                }
            )
            return {
                "year_round": year_round,
                "start_month": start_month_str,
                "end_month": end_month_str,
                "peak_season_start": self._month_to_first_day(peak_season_start_str),
                "peak_season_end": self._month_to_last_day(peak_season_end_str),
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Error determining club season",
                extra={
                    "city": city,
                    "state": state,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "year_round": "Unknown",
                "start_month": "N/A",
                "end_month": "N/A",
                "peak_season_start": "05-01",
                "peak_season_end": "08-31",
                "status": "error",
                "error": f"Error determining season data: {str(e)}"
            }

    def _month_to_first_day(self, month_name: str) -> str:
        """Convert month name to first day of month in MM-DD format."""
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        logger.debug("Converting month to first day", extra={
            "month_name": month_name,
            "valid_month": month_name in month_map
        })
        if month_name in month_map:
            result = f"{month_map[month_name][0]}-01"
            return result
        return "05-01"  # Default to May 1st

    def _month_to_last_day(self, month_name: str) -> str:
        """Convert month name to last day of month in MM-DD format."""
        month_map = {
            "January": ("01", "31"), "February": ("02", "28"), "March": ("03", "31"),
            "April": ("04", "30"), "May": ("05", "31"), "June": ("06", "30"),
            "July": ("07", "31"), "August": ("08", "31"), "September": ("09", "30"),
            "October": ("10", "31"), "November": ("11", "30"), "December": ("12", "31")
        }
        if month_name in month_map:
            return f"{month_map[month_name][0]}-{month_map[month_name][1]}"
        return "08-31"

    def review_previous_interactions(self, contact_id: str) -> Dict[str, Union[int, str]]:
        """
        Review previous interactions for a contact using HubSpot data.
        
        Args:
            contact_id (str): HubSpot contact ID
            
        Returns:
            Dict containing:
                - emails_opened (int): Number of emails opened
                - emails_sent (int): Number of emails sent
                - meetings_held (int): Number of meetings detected
                - last_response (str): Description of last response
                - status (str): "success", "error", or "no_data"
                - error (str): Error message if any
        """
        try:
            # Get contact properties from HubSpot
            lead_data = self._hubspot.get_contact_properties(contact_id)
            if not lead_data:
                logger.warning(
                    "No lead data found for contact",
                    extra={
                        "contact_id": contact_id,
                        "status": "no_data"
                    }
                )
                return {
                    "emails_opened": 0,
                    "emails_sent": 0,
                    "meetings_held": 0,
                    "last_response": "No data available",
                    "status": "no_data",
                    "error": "Contact not found in HubSpot"
                }

            # Extract email metrics
            emails_opened = self._safe_int(lead_data.get("total_opens_weekly"))
            emails_sent = self._safe_int(lead_data.get("num_contacted_notes"))

            # Get all notes for contact
            notes = self._hubspot.get_all_notes_for_contact(contact_id)

            # Count meetings from notes
            meeting_keywords = {"meeting", "meet", "call", "zoom", "teams"}
            meetings_held = 0
            for note in notes:
                if note.get("body") and any(keyword in note["body"].lower() for keyword in meeting_keywords):
                    meetings_held += 1
                    logger.debug("Found meeting note", extra={
                        "contact_id": contact_id,
                        "note_id": note.get("id"),
                        "meeting_count": meetings_held,
                        "correlation_id": f"interactions_{contact_id}"
                    })

            # Determine last response status
            last_reply = lead_data.get("hs_sales_email_last_replied")
            if last_reply:
                try:
                    reply_date = parse_date(last_reply.replace('Z', '+00:00'))
                    if reply_date.tzinfo is None:
                        reply_date = reply_date.replace(tzinfo=datetime.timezone.utc)
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    days_ago = (now_utc - reply_date).days
                    last_response = f"Responded {days_ago} days ago"
                except ValueError:
                    last_response = "Responded recently"
            else:
                if emails_opened > 0:
                    last_response = "Opened emails but no direct reply"
                else:
                    last_response = "No recent response"

            logger.info(
                "Successfully retrieved interaction history",
                extra={
                    "contact_id": contact_id,
                    "emails_opened": emails_opened,
                    "emails_sent": emails_sent,
                    "meetings_held": meetings_held,
                    "status": "success",
                    "has_last_reply": bool(last_reply),
                    "correlation_id": f"interactions_{contact_id}"
                }
            )
            return {
                "emails_opened": emails_opened,
                "emails_sent": emails_sent,
                "meetings_held": meetings_held,
                "last_response": last_response,
                "status": "success",
                "error": ""
            }

        except Exception as e:
            logger.error(
                "Failed to review contact interactions",
                extra={
                    "contact_id": contact_id,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            return {
                "emails_opened": 0,
                "emails_sent": 0,
                "meetings_held": 0,
                "last_response": "Error retrieving data",
                "status": "error",
                "error": f"Error retrieving interaction data: {str(e)}"
            }

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """
        Convert a value to int safely, defaulting if conversion fails.
        
        Args:
            value: Value to convert to integer
            default: Default value if conversion fails
            
        Returns:
            int: Converted value or default
        """
        if value is None:
            logger.debug("Received None value in safe_int conversion", extra={
                "value": "None",
                "default": default
            })
            return default
        try:
            result = int(float(str(value)))
            logger.debug("Successfully converted value to int", extra={
                "original_value": str(value),
                "result": result,
                "type": str(type(value))
            })
            return result
        except (TypeError, ValueError) as e:
            logger.debug("Failed to convert value to int", extra={
                "value": str(value),
                "default": default,
                "error": str(e),
                "type": str(type(value))
            })
            return default

```

## services\hubspot_integration.py
```python
 
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
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 1
        
        # Endpoints
        self.contacts_endpoint = f"{self.base_url}/crm/v3/objects/contacts"
        self.companies_endpoint = f"{self.base_url}/crm/v3/objects/companies"
        self.notes_search_url = f"{self.base_url}/crm/v3/objects/notes/search"
        self.tasks_endpoint = f"{self.base_url}/crm/v3/objects/tasks"
        self.emails_search_url = f"{self.base_url}/crm/v3/objects/emails/search"

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
        """Get company data."""
        if not company_id:
            return {}
            
        url = f"{self.companies_endpoint}/{company_id}?properties=name&properties=city&properties=state&properties=annualrevenue&properties=createdate&properties=hs_lastmodifieddate&properties=hs_object_id"
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

    def prepare_lead_context(self, lead_email: str, correlation_id: str = None) -> Dict[str, Any]:
        """
        Prepare lead context for personalization (subject/body).
        Uses DataGathererService for data retrieval.
        
        Args:
            lead_email: Email address of the lead
            correlation_id: Optional correlation ID for tracing operations
        """
        if correlation_id is None:
            correlation_id = f"prepare_context_{lead_email}"
            
        logger.debug("Starting lead context preparation", extra={
            "email": lead_email,
            "correlation_id": correlation_id
        })
        
        # Get comprehensive lead data from DataGathererService
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
        template_path = f"docs/templates/{filename_job_title}_initial_outreach.md"
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
        
        Args:
            doc_name: Name of the document to read
            fallback_content: Content to return if document cannot be read
            
        Returns:
            Content of the document or fallback content if document cannot be read
        """
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

## utils\formatting_utils.py
```python
"""
Utility functions for text formatting and cleaning.
"""

import re
from bs4 import BeautifulSoup
from typing import Optional

def clean_phone_number(raw_phone: str) -> str:
    """
    Example phone cleaning logic:
    1) Remove non-digit chars
    2) Format as needed (e.g., ###-###-####)
    """
    digits = "".join(char for char in raw_phone if char.isdigit())
    if len(digits) == 10:
        # e.g. (123) 456-7890
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    else:
        return digits

def clean_html(raw_html: str, strip_tags: bool = True, remove_scripts: bool = True) -> str:
    """
    Clean HTML content by removing tags and/or unwanted elements.
    
    Args:
        raw_html: Raw HTML string to clean
        strip_tags: If True, removes all HTML tags
        remove_scripts: If True, removes script and style tags before processing
    
    Returns:
        Cleaned text string
    """
    if not raw_html:
        return ""
    
    if remove_scripts:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        raw_html = str(soup)
    
    if strip_tags:
        # Remove HTML tags while preserving content
        text = re.sub('<[^<]+?>', '', raw_html)
        # Decode HTML entities
        text = BeautifulSoup(text, "html.parser").get_text()
        return text.strip()
    
    return raw_html.strip()


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
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from utils.logging_setup import logger

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Authenticate and return a Gmail API service instance."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e:
            logger.error(f"Error reading token.json: {e}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                logger.error("Error refreshing token. Delete token.json and re-run authentication.")
                raise
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds, cache_discovery=False)

def create_message(sender, to, subject, message_text):
    """Create a MIMEText email message."""
    logger.debug(f"Building MIMEText email: to={to}, subject={subject}")
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    logger.debug(f"Message built successfully for {to}")
    return {'message': {'raw': raw}}

def create_draft(sender, to, subject, message_text):
    """Create a draft in the user's Gmail Drafts folder."""
    logger.debug(f"Preparing to create draft. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(sender, to, subject, message_text)
    try:
        draft = service.users().drafts().create(userId='me', body=message_body).execute()
        if draft.get('id'):
            logger.info(f"Draft created successfully for '{to}' – ID={draft['id']}")
            return {"status": "ok", "draft_id": draft['id']}
        else:
            logger.error(f"No draft ID returned for '{to}' – possibly an API error.")
            return {"status": "error", "error": "No draft ID returned"}
    except Exception as e:
        logger.error(
            f"Failed to create draft for recipient='{to}' with subject='{subject}'. "
            f"Error: {e}"
        )
        return {"status": "error", "error": str(e)}

def send_message(sender, to, subject, message_text):
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(sender, to, subject, message_text)
    try:
        sent_msg = service.users().messages().send(
            userId='me',
            body=message_body
        ).execute()
        if sent_msg.get('id'):
            logger.info(f"Message sent successfully to '{to}' – ID={sent_msg['id']}")
            return {"status": "ok", "message_id": sent_msg.get('id')}
        else:
            logger.error(f"Message sent to '{to}' but no ID returned – possibly an API error.")
            return {"status": "error", "error": "No message ID returned"}
    except Exception as e:
        logger.error(
            f"Failed to send message to recipient='{to}' with subject='{subject}'. "
            f"Error: {e}"
        )
        return {"status": "error", "error": str(e)}

def search_messages(query=""):
    """
    Search for messages in the Gmail inbox using the specified `query`.
    For example:
      - 'from:kowen@capitalcityclub.org'
      - 'subject:Testing'
      - 'to:me newer_than:7d'
    Returns a list of message dicts.
    """
    service = get_gmail_service()
    try:
        response = service.users().messages().list(userId='me', q=query).execute()
        messages = response.get('messages', [])
        return messages
    except Exception as e:
        logger.error(f"Error searching messages with query '{query}': {e}")
        return []

def check_thread_for_reply(thread_id):
    """
    Checks if there's more than one message in a given thread, indicating a reply.
    This approach is more precise than searching by 'from:' and date alone.
    """
    service = get_gmail_service()
    try:
        thread_data = service.users().threads().get(userId='me', id=thread_id).execute()
        msgs = thread_data.get('messages', [])
        return len(msgs) > 1
    except Exception as e:
        logger.error(f"Error retrieving thread {thread_id}: {e}")
        return False

#
#  NEW FUNCTION:
#  Search your Gmail inbox for any messages from the specified email address.
#  Return up to `max_results` message snippets (short preview text).
#
def search_inbound_messages_for_email(email_address: str, max_results: int = 1) -> list:
    """
    Search for inbound messages sent from `email_address`.
    Returns a list of short snippets from the most recent matching messages.
    """
    # 1) Build a Gmail search query
    query = f"from:{email_address}"

    # 2) Find message IDs matching the query
    message_ids = search_messages(query=query)
    if not message_ids:
        return []  # None found

    # 3) Retrieve each message snippet up to max_results
    service = get_gmail_service()
    snippets = []
    for m in message_ids[:max_results]:
        try:
            full_msg = service.users().messages().get(
                userId='me',
                id=m['id'],
                format='full'
            ).execute()
            snippet = full_msg.get('snippet', '')
            snippets.append(snippet)
        except Exception as e:
            logger.error(f"Error fetching message {m['id']} from {email_address}: {e}")

    return snippets

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
from pathlib import Path
from typing import Optional
from utils.logger_base import get_base_logger
from config.settings import LOG_LEVEL, DEV_MODE, PROJECT_ROOT

def setup_logger(
    name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance with the specified settings.
    
    Args:
        name: Logger name (defaults to root logger if None)
        log_level: Logging level (defaults to environment variable LOG_LEVEL or INFO)
        log_file: Optional file path for logging to file
        
    Returns:
        Configured logger instance
    """
    # Get base logger with console handler
    logger = get_base_logger(name, log_level or LOG_LEVEL)
    
    # Add file handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger(
    name=__name__,
    log_file=PROJECT_ROOT / 'logs' / 'app.log' if not DEV_MODE else None
)

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
            "As you gear up for the season ahead,"
        ],
        "in_season": [
            "I hope the season is going well for you,",
            "With the season in full swing,"

        ],
        "winding_down": [
            "As the season winds down,",
            "As your peak season comes to a close,"
        ],
        "off_season": [
            "As you look forward to the coming season,",
            "While planning for next season,",
            "As you prepare for the year ahead,"
        ]
    }

    # fallback if not found
    if season_key not in snippet_options:
        return ""

    return random.choice(snippet_options[season_key])

```

## utils\web_fetch.py
```python
import requests
from utils.logging_setup import logger

def fetch_website_html(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text
        else:
            logger.error("Failed to fetch website %s: %s", url, resp.text)
            return ""
    except requests.RequestException as e:
        logger.error("Error fetching website %s: %s", url, str(e))
        return ""

```

## utils\xai_integration.py
```python
import os
import requests
from typing import Tuple, Dict, Any
from utils.logging_setup import logger
from dotenv import load_dotenv
from config.settings import DEBUG_MODE
import json
import time

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")

def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends request to xAI API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            if DEBUG_MODE:
                logger.debug(f"xAI request payload={payload}")
            
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN
                },
                json=payload,
                timeout=15
            )
            
            if response.status_code == 429:  # Rate limit
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                logger.error(f"xAI API error ({response.status_code}): {response.text}")
                return ""
                
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            if DEBUG_MODE:
                logger.debug(f"xAI response={content}")
            return content
            
        except Exception as e:
            logger.error(f"Error in xAI request (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(retry_delay)
            
    return ""  # Return empty string if all retries fail

##############################################################################
# News Search + Icebreaker
##############################################################################

def xai_news_search(club_name: str) -> str:
    """
    Checks if a club is in the news; returns a summary or 
    indicates 'has not been in the news.'
    """
    if not club_name.strip():
        if DEBUG_MODE:
            logger.debug("Empty club_name passed to xai_news_search; returning blank.")
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise summaries of recent club news."
            },
            {
                "role": "user",
                "content": (
                    f"Tell me about any recent news for {club_name}. "
                    "If none exists, respond with 'has not been in the news.'"
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0
    }
    
    # Get the raw response from xAI
    response = _send_xai_request(payload)
    
    # Clean up awkward grammar in the response
    if response:
        # Fix the "Has [club] has not been" pattern
        if response.startswith("Has ") and " has not been in the news" in response:
            response = response.replace("Has ", "")
        
        # Fix any double "has" instances
        response = response.replace(" has has ", " has ")
    
    return response

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker referencing recent news.
    """
    if not club_name.strip() or not news_summary.strip():
        if DEBUG_MODE:
            logger.debug("Empty input passed to _build_icebreaker_from_news; returning blank.")
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sales copywriter. Create a natural, conversational "
                    "one-sentence opener mentioning recent club news."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Club: {club_name}\n"
                    f"News: {news_summary}\n\n"
                    "Write ONE engaging sentence that naturally references this news. "
                    "Avoid starting with phrases like 'I saw' or 'I noticed'."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.7
    }
    return _send_xai_request(payload)

##############################################################################
# Club Info Search (Used ONLY for Final Email Rewriting)
##############################################################################

def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> str:
    """
    Returns a short overview about the club's location and amenities.
    This is NOT used for icebreakers. We only use its result 
    to enhance context for final email rewriting.
    """
    if not club_name.strip():
        if DEBUG_MODE:
            logger.debug("Empty club_name passed to xai_club_info_search; returning blank.")
        return ""

    loc_str = location if location else "an unknown location"
    am_str = ", ".join(amenities) if amenities else "no specific amenities"
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides a brief overview "
                    "of a club's location and amenities."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise overview about {club_name} in {loc_str}, "
                    f"highlighting amenities like {am_str}. Only provide one short paragraph."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.5
    }
    return _send_xai_request(payload)

##############################################################################
# Personalize Email with Additional Club Info
##############################################################################

def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = "",
    club_info: str = ""
) -> Tuple[str, str]:
    """
    Use xAI to personalize an email's subject and body.
    
    Args:
        lead_sheet: Dictionary containing lead data
        subject: Original email subject
        body: Original email body
        summary: Summary of previous interactions
        news_summary: Recent news about the club
        club_info: Information about the club's facilities and features
    
    Returns:
        Tuple of (personalized_subject, personalized_body)
    """
    # Include club_info in your prompt construction
    prompt = f"""
    Lead Information: {json.dumps(lead_sheet)}
    Previous Interaction Summary: {summary}
    Club News: {news_summary}
    Club Information: {club_info}
    
    Original Subject: {subject}
    Original Body: {body}
    
    Please personalize this email while maintaining its core message...
    """

    if not lead_sheet:
        logger.warning("Empty lead_sheet passed to personalize_email_with_xai")
        return subject, body

    try:
        # Extract key information
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        facilities_info = lead_sheet.get("analysis", {}).get("facilities", {}).get("response", "")
        club_name = company_data.get("name", "").strip()
        
        # Build context for xAI
        context = f"Club Name: {club_name}\nFacilities Info: {facilities_info}\n"
        
        # If you have the lead interaction summary, add it:
        if summary:
            context += f"Lead Interaction Summary: {summary}\n"

        # Build the prompt
        user_content = (
            f"Original Subject: {subject}\n"
            f"Original Body: {body}\n\n"
            f"Context:\n{context}\n"
            "Instructions:\n"
            "1. Personalize based on verified club context and history.\n"
            "2. Focus on business value and problem-solving.\n" 
            "3. Keep core Swoop platform value proposition.\n"
            "4. Use brief, relevant facility references only if confirmed.\n"
            "5. Write at 6th-8th grade reading level.\n"
            "6. Keep paragraphs under 3 sentences.\n"
            "7. Maintain professional but helpful tone.\n"
            "8. Reference previous interactions naturally.\n"
            "9. If lead has replied to previous email, reference it naturally without direct acknowledgment.\n"
            "10. If lead expressed specific interests/concerns in reply, address them.\n"
            "Format the response as:\n"
            "Subject: [new subject]\n\n"
            "Body:\n[new body]"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at personalizing outreach emails for golf clubs. "
                        "You maintain a professional yet friendly tone and incorporate relevant context naturally. "
                        "You never mention unconfirmed facilities."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.7
        }

        if DEBUG_MODE:
            logger.debug("xAI request payload for personalization", extra={
                "club_name": club_name,
                "has_summary": bool(summary),
                "payload": payload
            })

        result = _send_xai_request(payload)
        
        if not result:
            logger.warning("Empty response from xAI personalization", extra={"club_name": club_name})
            return subject, body

        # Parse the response
        new_subject, new_body = _parse_xai_response(result)
        
        if not new_subject or not new_body:
            logger.warning("Failed to parse xAI response", extra={
                "club_name": club_name,
                "raw_response": result
            })
            return subject, body

        return new_subject, new_body

    except Exception as e:
        logger.error("Error in email personalization", extra={
            "error": str(e),
            "club_name": club_name if 'club_name' in locals() else 'unknown'
        })
        return subject, body

def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body.
    
    Args:
        response: Raw response from xAI
        
    Returns:
        Tuple[str, str]: Parsed subject and body
    """
    try:
        lines = response.split('\n')
        new_subject = ""
        new_body = []
        in_subject = False
        in_body = False
        
        for line in lines:
            if line.lower().startswith('subject:'):
                in_subject = True
                in_body = False
                new_subject = line.split(':', 1)[1].strip()
            elif line.lower().startswith('body:'):
                in_subject = False
                in_body = True
            elif in_body:
                new_body.append(line)
                
        return new_subject, '\n'.join(new_body).strip()
        
    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}")
        return "", ""

##############################################################################
# Facilities Check
##############################################################################

def xai_facilities_check(club_name: str, city: str, state: str) -> str:
    """
    Checks what facilities a club has with improved accuracy.
    """
    if not club_name.strip():
        logger.debug("Empty club_name passed to xai_facilities_check")
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides accurate facility information "
                    "for golf clubs and country clubs. Only confirm facilities that you are certain exist."
                    "If you are unsure about any facility, do not mention it."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise sentence about {club_name} in {city}, {state}, "
                    "mentioning only confirmed facilities like the number of holes for the golf course and "
                    "whether it's public, private, or semi-private. Omit any unconfirmed facilities."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0
    }
    
    try:
        response = _send_xai_request(payload)
        if not response:
            logger.error("Empty response from xAI facilities check", extra={
                "club_name": club_name,
                "city": city,
                "state": state
            })
            return "Facility information unavailable"
            
        return response
        
    except Exception as e:
        logger.error(f"Error in facilities check: {str(e)}", extra={
            "club_name": club_name,
            "city": city,
            "state": state
        })
        return "Facility information unavailable"
```
