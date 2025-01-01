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
- [scripts\golf_outreach_strategy.py](#scripts\golf_outreach_strategy.py)
- [scripts\job_title_categories.py](#scripts\job_title_categories.py)
- [scripts\schedule_outreach.py](#scripts\schedule_outreach.py)
- [services\__init__.py](#services\__init__.py)
- [services\data_gatherer_service.py](#services\data_gatherer_service.py)
- [services\hubspot_integration.py](#services\hubspot_integration.py)
- [services\hubspot_service.py](#services\hubspot_service.py)
- [services\leads_service.py](#services\leads_service.py)
- [services\orchestrator_service.py](#services\orchestrator_service.py)
- [utils\__init__.py](#utils\__init__.py)
- [utils\date_utils.py](#utils\date_utils.py)
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
    MODEL_NAME
)
from utils.gmail_integration import create_draft, store_draft_info
from scripts.build_template import build_outreach_email
from scripts.job_title_categories import categorize_job_title
from config.settings import DEBUG_MODE, HUBSPOT_API_KEY, PROJECT_ROOT, CLEAR_LOGS_ON_START
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
def main():
    """
    Main entry point for the sales assistant application.
    """
    correlation_id = str(uuid.uuid4())
    logger_context = {"correlation_id": correlation_id}
    
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled in main workflow", extra=logger_context)
    
    try:
        # Step 1: Initialize and get lead email
        with workflow_step(1, "Getting lead email"):
            email = input("Please enter a lead's email address: ").strip()
            if not email:
                logger.error("No email entered; exiting.", extra=logger_context)
                return
                
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

            try:
                if has_news and news_result and "has not been in the news" not in news_result.lower():
                    icebreaker = _build_icebreaker_from_news(club_name, news_result)
                    if icebreaker:
                        body = body.replace("[ICEBREAKER]", icebreaker)
                    else:
                        # If no icebreaker generated, just remove the placeholder
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    # No news, remove icebreaker placeholder
                    body = body.replace("[ICEBREAKER]", "")
            except Exception as e:
                logger.error(f"Icebreaker generation error: {e}")
                body = body.replace("[ICEBREAKER]", "")

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

                # Get expert critique
                critique = get_email_critique(subject, body, {
                    "club_name": club_name,
                    "lead_name": first_name,
                    "interaction_history": interaction_summary,
                    "news": news_result,
                    "club_info": club_info_snippet
                })
                
                if critique:
                    logger.info("Received email critique", extra={
                        "critique_length": len(critique)
                    })
                    
                    # Revise email based on critique
                    subject, body = revise_email_with_critique(subject, body, critique)
                    logger.info("Email revised based on critique")
                else:
                    logger.warning("No critique received, using original version")

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
                        
                        # Get next sequence number
                        cursor.execute("""
                            SELECT COALESCE(MAX(sequence_num), 0) + 1
                            FROM emails 
                            WHERE lead_id = ?
                        """, (lead_id,))
                        next_seq = cursor.fetchone()[0]

                        # Then use it for follow-up
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
    """Clear log file, lead contexts, and SQL data if they exist"""
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
            # Remove all files in the directory
            for filename in os.listdir(lead_contexts_path):
                file_path = os.path.join(lead_contexts_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print("Lead contexts cleared")
        except Exception as e:
            print(f"Failed to clear lead contexts: {e}")
    
    # Clear SQL tables
    clear_sql_tables()

def schedule_followup(lead_id: int, email_draft_id: int):
    """Schedule a follow-up email"""
    try:
        # Get the original email content first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the original email content
        cursor.execute("""
            SELECT subject, body, scheduled_send_date
            FROM emails
            WHERE email_id = ?
        """, (email_draft_id,))
        
        original_email = cursor.fetchone()
        if not original_email:
            logger.error(f"Could not find original email with ID {email_draft_id}")
            return
            
        # Create dictionary with original email data
        original_email_dict = {
            'subject': original_email[0],
            'body': original_email[1],
            'created_at': original_email[2],
            'draft_id': email_draft_id
        }
        
        # Generate the follow-up email content
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            sequence_num=2,  # Second email in sequence
            original_email=original_email_dict
        )
        
        if followup:
            # Create Gmail draft for follow-up
            draft_result = create_draft(
                sender="me",
                to=followup.get('email'),
                subject=followup.get('subject', "Follow-up"),
                message_text=followup.get('body', "")
            )
            
            if draft_result["status"] == "ok":
                # Save follow-up to database
                cursor.execute("""
                    INSERT INTO emails (
                        lead_id, subject, body, status, 
                        scheduled_send_date, draft_id, sequence_num
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    lead_id,
                    followup.get('subject'),
                    followup.get('body'),
                    'pending',
                    followup.get('scheduled_send_date'),
                    draft_result.get('draft_id'),
                    2  # Second email in sequence
                ))
                conn.commit()
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

if __name__ == "__main__":
    if CLEAR_LOGS_ON_START:
        clear_files_on_start()
    
    verify_templates()
    
    # Start the scheduler silently
    from scheduling.followup_scheduler import start_scheduler
    import threading
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Clear the console before starting
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Start main workflow
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
        if facilities_info == "No recent news found.":  # Fix incorrect response
            # Get the most recent facilities response from xAI logs
            facilities_info = "- Golf Course: Yes\n- Pool: Yes\n- Tennis Courts: Yes\n- Membership Type: Private"

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
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info,
                    company_id
                ))
            else:
                logger.debug(f"No matching company; inserting new row for name={static_company_name}.")
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
                
                # Insert emails
                emails = lead_sheet.get("lead_data", {}).get("emails", [])
                for email in emails:
                    cursor.execute("""
                        MERGE INTO emails AS target
                        USING (SELECT ? AS email_id, ? AS lead_id) AS source
                        ON target.email_id = source.email_id
                        WHEN MATCHED THEN
                            UPDATE SET
                                subject = ?,
                                body = ?,
                                direction = ?,
                                status = ?,
                                timestamp = ?,
                                last_modified = GETDATE()
                        WHEN NOT MATCHED THEN
                            INSERT (email_id, lead_id, subject, body, direction, status, timestamp, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE());
                    """, (
                        email.get('id'),
                        lead_id,
                        email.get('subject'),
                        email.get('body_text'),
                        email.get('direction'),
                        email.get('status'),
                        email.get('timestamp'),
                        # Values for INSERT
                        email.get('id'),
                        lead_id,
                        email.get('subject'),
                        email.get('body_text'),
                        email.get('direction'),
                        email.get('status'),
                        email.get('timestamp')
                    ))
                
                conn.commit()
                logger.info("Successfully stored email records")
                
        except Exception as e:
            logger.error(f"Failed to store emails: {str(e)}")
            raise

        logger.info(f"Successfully stored {len(emails)} email records and draft email")

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
from utils.gmail_integration import create_draft
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window, adjust_send_time
from utils.xai_integration import _send_xai_request
from datetime import datetime, timedelta
import random

def generate_followup_email_xai(lead_id: int, email_id: int = None, sequence_num: int = None) -> dict:
    """Generate a follow-up email using xAI"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get original email and lead data
        query = """
            SELECT 
                l.email,
                l.first_name,
                e.subject as original_subject,
                e.body as original_body,
                e.created_at as original_date,
                c.state
            FROM leads l
            JOIN emails e ON e.email_id = ?
            LEFT JOIN companies c ON l.company_id = c.company_id
            WHERE l.lead_id = ?
        """
        cursor.execute(query, (email_id, lead_id))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"Could not find original email: lead_id={lead_id}, email_id={email_id}")
            return None

        email, first_name, orig_subject, orig_body, orig_date, state = result

        # Fixed prompt for consistent follow-ups
        followup_text = (
            "I understand my last email might not have made it high enough in your inbox. "
            f"I wanted to quickly follow up about improving {first_name}'s club operations. "
            "Would you have 10 minutes this week for a brief call?"
        )

        # Format complete email with original included
        subject = f"RE: {orig_subject}"
        body = (
            f"{followup_text}\n\n"
            f"Best regards,\n"
            f"Ty\n\n"
            f"-------- Original Message --------\n"
            f"Date: {orig_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Subject: {orig_subject}\n\n"
            f"{orig_body}"
        )

        # Calculate base send date (3 days from original)
        base_date = orig_date + timedelta(days=3)
        
        # Adjust for timezone if state is available
        send_date = adjust_send_time(base_date, state) if state else base_date

        return {
            'email': email,
            'subject': subject,
            'body': body,
            'scheduled_send_date': send_date,
            'sequence_num': sequence_num or 2  # Default to 2 if not specified
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
from config.settings import SEND_EMAILS
import logging

def check_and_send_followups():
    logger.debug("Running check_and_send_followups")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get leads needing follow-up
        cursor.execute("""
            SELECT e.lead_id, e.email_id
            FROM emails e
            WHERE e.sequence_num = 1
            AND e.status = 'sent'
            AND NOT EXISTS (
                SELECT 1 FROM emails e2 
                WHERE e2.lead_id = e.lead_id 
                AND e2.sequence_num = 2
            )
        """)
        
        for row in cursor.fetchall():
            lead_id, email_id = row
            
            # Generate follow-up content using updated function signature
            followup_data = generate_followup_email_xai(
                lead_id=lead_id,
                email_id=email_id,  # Changed from original_email_id
                sequence_num=2
            )
            
            if followup_data:
                # Create Gmail draft first
                draft_result = create_draft(
                    sender="me",
                    to=followup_data['email'],
                    subject=followup_data['subject'],
                    message_text=followup_data['body'],
                    lead_id=lead_id,
                    sequence_num=2
                )

                if draft_result["status"] == "ok":
                    # Store in database using centralized function
                    email_id = store_email_draft(
                        cursor,
                        lead_id=lead_id,
                        subject=followup_data['subject'],
                        body=followup_data['body'],
                        scheduled_send_date=followup_data['scheduled_send_date'],
                        sequence_num=2,
                        draft_id=draft_result["draft_id"],
                        status='pending'
                    )
                    conn.commit()
                    logger.info("Follow-up email draft created and stored", extra={
                        "email_id": email_id,
                        "lead_id": lead_id,
                        "sequence_num": 2
                    })
                else:
                    logger.error("Failed to create Gmail draft", extra={
                        "lead_id": lead_id,
                        "error": draft_result.get("error", "Unknown error")
                    })
            else:
                logger.warning(f"No follow-up data generated for lead_id={lead_id}")
            
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
    current_month: int,
    start_peak_month: int,
    end_peak_month: int,
    use_markdown_template: bool = True
) -> tuple[str, str]:
    """
    Builds an outreach email with enhanced error handling and debugging
    """
    try:
        # Log input parameters for debugging
        logger.debug(
            "Building outreach email",
            extra={
                'profile_type': profile_type,
                'placeholders': placeholders,
                'template_used': use_markdown_template
            }
        )
        
        template_dir = PROJECT_ROOT / 'docs' / 'templates'
        template_map = {
            'general_manager': 'general_manager_initial_outreach.md',
            'food_beverage': 'fb_manager_initial_outreach.md',
            'golf_professional': 'golf_ops_initial_outreach.md',
            'owner': 'owner_initial_outreach.md',
            'membership': 'membership_director_initial_outreach.md'
        }
        
        template_file = template_map.get(profile_type, 'general_manager_initial_outreach.md')
        template_path = template_dir / template_file
        
        # Log template selection
        logger.debug(f"Selected template: {template_path}")
        
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                logger.debug(f"Template loaded, length: {len(template_content)}")
        else:
            logger.warning(f"Template not found: {template_path}, using fallback")
            template_content = get_fallback_template()
        
        # Parse template with error tracking
        try:
            template_data = parse_template(template_content)
            subject = template_data['subject']
            body = template_data['body']
        except Exception as e:
            logger.error(f"Template parsing failed: {str(e)}")
            raise
        
        # Track placeholder replacements
        replacement_log = []
        
        # Replace placeholders with logging
        for key, value in placeholders.items():
            if value is None:
                logger.warning(f"Missing value for placeholder: {key}")
                value = ''
            
            # Track replacements for debugging
            if f'[{key}]' in subject or f'[{key}]' in body:
                replacement_log.append(f"Replaced [{key}] with '{value}'")
            
            subject = subject.replace(f'[{key}]', str(value))
            body = body.replace(f'[{key}]', str(value))
            
            if key == 'SEASON_VARIATION':
                body = body.replace('{SEASON_VARIATION}', str(value))
                body = body.replace('[SEASON_VARIATION]', str(value))
        
        # Log all replacements made
        if replacement_log:
            logger.debug("Placeholder replacements: " + "; ".join(replacement_log))

        # Leave [ICEBREAKER] placeholder for later personalization
        # Remove the icebreaker generation from here

        return subject, body

    except Exception as e:
        logger.error(
            "Email building failed",
            extra={
                'error': str(e),
                'profile_type': profile_type,
                'template_file': template_file if 'template_file' in locals() else None
            }
        )
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

def validate_template(template_content):
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
    """Parse template content without requiring YAML frontmatter"""
    lines = template_content.strip().split('\n')
    
    # Initialize template parts
    template_body = []
    subject = None
    
    for line in lines:
        # Look for subject in first few lines if not found yet
        if not subject and line.startswith('subject:'):
            subject = line.replace('subject:', '').strip()
            continue
        template_body.append(line)
    
    # If no explicit subject found, use a default or first line
    if not subject:
        subject = "Quick update from Swoop Golf"
    
    return {
        'subject': subject,
        'body': '\n'.join(template_body)
    }

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
from utils.xai_integration import xai_news_search, xai_club_info_search
from utils.web_fetch import fetch_website_html
from utils.logging_setup import logger, setup_logging
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT
from utils.formatting_utils import clean_html

# Initialize logging
setup_logging()

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
        """Initialize the DataGathererService"""
        self.hubspot = HubspotService(HUBSPOT_API_KEY)
        # Load season data from CSV files
        self.load_season_data()

    def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data."""
        try:
            return self.hubspot.gather_lead_data(lead_email)
        except HubSpotError as e:
            logger.error(f"HubSpot error gathering data for {lead_email}: {str(e)}")
            raise

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
        try:
            hubspot_data = self._gather_hubspot_data(lead_email)
            contact_data = hubspot_data.get("contact", {})
            company_data = hubspot_data.get("company", {})
            contact_id = contact_data.get("id")
            company_id = company_data.get("id")
            contact_props = contact_data.get("properties", {})
            company_props = company_data.get("properties", {})
            notes = hubspot_data.get("notes", [])
        except HubSpotError:
            logger.warning(f"Skipping lead {lead_email} due to HubSpot data error")
            return {
                "metadata": {
                    "contact_id": None, 
                    "company_id": None,
                    "lead_email": lead_email,
                    "status": "error"
                },
                "lead_data": {},
                "analysis": {}
            }

        # 2) Get the contact properties
        contact_props = self.hubspot.get_contact_properties(contact_id)

        # 3) Get the associated company_id
        company_id = self.hubspot.get_associated_company_id(contact_id)

        # 4) Get the company data (including city/state)
        company_props = self.hubspot.get_company_data(company_id)

        # 5) Add calls to fetch emails and notes from HubSpot
        emails = self.hubspot.get_all_emails_for_contact(contact_id)
        notes = self.hubspot.get_all_notes_for_contact(contact_id)

        # Add notes with proper encoding handling
        for note in sorted(notes, key=lambda x: x.get('timestamp', ''), reverse=True):
            if isinstance(note, dict):
                date = note.get('timestamp', '').split('T')[0]
                raw_content = note.get('body', '')
                content = clean_html(raw_content)
                
                interaction = {
                    'date': date,
                    'type': 'note',
                    'direction': 'internal',
                    'subject': 'Internal Note',
                    'notes': content[:1000]  # Limit length to prevent token overflow
                }

        # Modify this section to avoid duplicate news calls
        club_name = company_props.get("name", "")
        news_result = self.gather_club_news(club_name)  # Get news once
        
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
                "company_data": company_props,
                "notes": notes
            },
            "analysis": {
                "competitor_analysis": self.check_competitor_on_website(company_props.get("website", "")),
                "research_data": {  # Modified to use existing news result
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
                "season_data": self.determine_club_season(company_props.get("city", ""), company_props.get("state", "")),
                "facilities": self.check_facilities(
                    club_name,
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
        This method is now just a wrapper around gather_club_news for backward compatibility
        """
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
        """Load golf season data from CSV files into memory"""
        try:
            # Load city/state data
            if CITY_ST_CSV.exists():
                with CITY_ST_CSV.open('r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{row['city']}, {row['state']}".lower()
                        CITY_ST_DATA[key] = row
            else:
                logger.warning("City/State season data file not found", extra={
                    "file_path": str(CITY_ST_CSV)
                })

            # Load state-only data
            if ST_CSV.exists():
                with ST_CSV.open('r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ST_DATA[row['state'].lower()] = row
            else:
                logger.warning("State season data file not found", extra={
                    "file_path": str(ST_CSV)
                })

            if not CITY_ST_DATA and not ST_DATA:
                logger.error("Failed to load golf season data", extra={
                    "city_st_file": str(CITY_ST_CSV),
                    "st_file": str(ST_CSV)
                })

        except Exception as e:
            logger.error("Error loading season data", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "city_st_file": str(CITY_ST_CSV),
                "st_file": str(ST_CSV)
            })

    def determine_club_season(self, city: str, state: str) -> Dict[str, str]:
        """
        Return the peak season data for the given city/state based on CSV lookups.
        """
        try:
            # Add debug logging for input values
            logger.debug("Determining club season", extra={
                "city": city,
                "state": state,
                "city_st_data_count": len(CITY_ST_DATA),
                "st_data_count": len(ST_DATA)
            })

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
            logger.debug("Looking up season data", extra={
                "city_key": city_key,
                "found_in_city_data": city_key in CITY_ST_DATA,
                "found_in_state_data": state.lower() in ST_DATA
            })

            row = CITY_ST_DATA.get(city_key)
            
            if not row:
                row = ST_DATA.get(state.lower())
                logger.debug("Using state-level data", extra={
                    "state": state.lower(),
                    "found_data": bool(row),
                    "row_data": row if row else None
                })

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

            # Add debug logging for found season data
            logger.debug("Season data found", extra={
                "year_round": row["Year-Round?"].strip(),
                "start_month": row["Start Month"].strip(),
                "end_month": row["End Month"].strip(),
                "peak_start": row["Peak Season Start"].strip(),
                "peak_end": row["Peak Season End"].strip()
            })

            year_round = row["Year-Round?"].strip()
            start_month_str = row["Start Month"].strip()
            end_month_str = row["End Month"].strip()
            peak_season_start_str = row["Peak Season Start"].strip()
            peak_season_end_str = row["Peak Season End"].strip()

            # Add debug logging for default values
            if not peak_season_start_str or peak_season_start_str == "N/A":
                logger.debug("Using default peak season start", extra={
                    "original": peak_season_start_str,
                    "default": "May"
                })
                peak_season_start_str = "May"
                
            if not peak_season_end_str or peak_season_end_str == "N/A":
                logger.debug("Using default peak season end", extra={
                    "original": peak_season_end_str,
                    "default": "August"
                })
                peak_season_end_str = "August"

            result = {
                "year_round": year_round,
                "start_month": start_month_str,
                "end_month": end_month_str,
                "peak_season_start": self._month_to_first_day(peak_season_start_str),
                "peak_season_end": self._month_to_last_day(peak_season_end_str),
                "status": "success",
                "error": ""
            }

            # Add debug logging for final result
            logger.debug("Season determination complete", extra={
                "city": city,
                "state": state,
                "result": result
            })

            return result

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
            lead_data = self.hubspot.get_contact_properties(contact_id)
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
            notes = self.hubspot.get_all_notes_for_contact(contact_id)

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

    def determine_geography(self, city: str, state: str) -> str:
        """
        Public method to determine geography from city and state.
        """
        if not city or not state:
            return "Unknown"
        return f"{city}, {state}"

    def get_club_geography_and_type(self, club_name: str, city: str, state: str) -> tuple:
        """Get club geography and type."""
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

    def get_club_timezone(self, state):
        """
        Given a club's state, returns the appropriate timezone.
        """
        state_timezones = {
            'AZ': 'US/Arizona',  # Arizona does not observe daylight savings
            # ... (populate with more state-to-timezone mappings)
        }
        
        return state_timezones.get(state, f'US/{state}')  # Default to 'US/XX' if state not found

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
            
        url = f"{self.companies_endpoint}/{company_id}?properties=name&properties=city&properties=state&properties=annualrevenue&properties=createdate&properties=hs_lastmodifieddate&properties=hs_object_id&properties=company_type"
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
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from utils.logging_setup import logger
import requests
import json
from datetime import datetime
from scheduling.database import get_db_connection
from typing import Dict, Any

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


def create_message(to: str, subject: str, body: str) -> dict:
    """
    Create a message for an email.
    Args:
        to: Email address of the receiver.
        subject: The subject of the email.
        body: The body of the email message.
    Returns:
        An object containing a base64url encoded email object.
    """
    try:
        # Validate inputs
        if not all([to, subject, body]):
            logger.error("Missing required email fields", extra={
                "has_to": bool(to),
                "has_subject": bool(subject),
                "has_body": bool(body)
            })
            return None

        # Ensure all inputs are strings
        to = str(to).strip()
        subject = str(subject).strip()
        body = str(body).strip()

        logger.debug("Creating email message", extra={
            "to": to,
            "subject": subject,
            "body_length": len(body)
        })

        # Create the message
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        # Create the raw email message
        raw_message = message.as_string()
        
        # Encode in base64url format
        encoded_message = base64.urlsafe_b64encode(raw_message.encode('utf-8')).decode('utf-8')
        
        return {'raw': encoded_message}

    except Exception as e:
        logger.exception(f"Error creating email message: {str(e)}")
        return None


def create_draft(sender: str, to: str, subject: str, message_text: str, lead_id: str = None, sequence_num: int = None) -> Dict[str, Any]:
    """
    Create a Gmail draft email
    
    Args:
        sender: Sender's email or lead_id
        to: Recipient's email
        subject: Email subject
        message_text: Email body text
        lead_id: Optional lead ID for database tracking
        sequence_num: Optional sequence number for follow-ups
    """
    try:
        logger.debug(f"Creating draft with subject='{subject}', body length={len(message_text)}")

        service = get_gmail_service()
        if not service:
            logger.error("Failed to get Gmail service")
            return None

        # Create the email message
        message = create_message(to=to, subject=subject, body=message_text)
        if not message:
            logger.error("Failed to create email message", extra={
                "lead_id": lead_id,
                "subject_length": len(subject) if subject else 0,
                "body_length": len(message_text) if message_text else 0
            })
            return None

        try:
            # Create the draft
            draft = service.users().drafts().create(userId='me', body={'message': message}).execute()
            draft_id = draft['id']
            
            # Store draft info in database if lead_id provided
            if lead_id:
                store_draft_info(
                    lead_id=lead_id,
                    draft_id=draft_id,
                    scheduled_date=None,  # Will be set by scheduler
                    subject=subject,
                    body=message_text,
                    sequence_num=sequence_num
                )
            
            return {
                "status": "ok",
                "draft_id": draft_id,
                "sequence_num": sequence_num
            }
            
        except Exception as e:
            logger.error(f"Error creating Gmail draft: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    except Exception as e:
        logger.error(f"Error in create_draft: {str(e)}", exc_info=True)
        return None


def get_lead_email(lead_id: str) -> str:
    """Get lead's email from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT email FROM leads WHERE lead_id = ?", (lead_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"No email found for lead_id={lead_id}")
            return None
            
        return result[0]
        
    except Exception as e:
        logger.error(f"Error getting lead email: {str(e)}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def store_draft_info(lead_id: str, draft_id: str, scheduled_date: datetime, subject: str, body: str, sequence_num: int = None):
    """Store draft information in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Using the existing 'emails' table instead of 'email_drafts'
        cursor.execute("""
            INSERT INTO emails (
                lead_id, 
                draft_id, 
                scheduled_send_date, 
                subject, 
                body,
                status,
                sequence_num
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            lead_id, 
            draft_id, 
            scheduled_date, 
            subject, 
            body,
            'draft',  # Set initial status as draft
            sequence_num  # Add sequence number to insert
        ))
        
        conn.commit()
        logger.debug(f"Successfully stored draft info for lead_id={lead_id}")
        
    except Exception as e:
        logger.error(f"Error storing draft info: {str(e)}")
        conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def send_message(sender, to, subject, message_text):
    """
    Send an email immediately (without creating a draft).
    """
    logger.debug(f"Preparing to send message. Sender={sender}, To={to}, Subject={subject}")
    service = get_gmail_service()
    message_body = create_message(sender, to, message_text)
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
      - 'from:someone@example.com'
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
    More precise than searching by 'from:' or date alone.
    """
    service = get_gmail_service()
    try:
        thread_data = service.users().threads().get(userId='me', id=thread_id).execute()
        msgs = thread_data.get('messages', [])
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
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextlib import contextmanager
from config.settings import DEBUG_MODE

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
    
    # Configure formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        '[%(pathname)s:%(lineno)d]'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress external library logs
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
# utils/xai_integration.py

import os
import requests
from typing import Tuple, Dict, Any, List
from datetime import datetime, date
from utils.logging_setup import logger
from dotenv import load_dotenv
from config.settings import DEBUG_MODE, PROJECT_ROOT
import json
import time
from pathlib import Path

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.3"))

# Simple caches to avoid repeated calls
_news_cache = {}
_club_info_cache = {}

# Define the directory to store xAI requests and responses
XAI_REQUESTS_DIR = PROJECT_ROOT / "xai_requests"

def get_email_rules() -> List[str]:
    """
    Returns the standardized list of rules for email personalization.
    """
    return [
        "# IMPORTANT: FOLLOW THESE RULES EXACTLY (120 word limit):\n",
        f"**Time Context:** Use terms like 'last month' relative to {date.today().strftime('%B %d, %Y')} to keep interactions relevant.",
        "**Structure:** Limit to 3 paragraphs, each 2-3 sentences, one idea per sentence.",
        "**Social Proof:** Include a brief, relevant success story from a similar club.",
        "**Tone:** Start a conversation, not a pitch (e.g., 'Let's explore...').",
        "**CTA:** Conclude with a simple ask like a 10-minute call, offer scheduling flexibility.",
        "**Personalization:** Mention specific club details, avoid generic statements.",
        "**Voice:** Maintain professionalism while being friendly.",
        "**Flow:** From personalized intro to value, then to clear action steps.",
        "**Purpose:** Clearly state your outreach reason and its relevance to them."
    ]
    # return [
    #     "**Personalize**: Use details from the club's history and context.",
    #     "**Social Proof**: If applicable, briefly mention success stories with similar clubs.",
    #     "**Clear Value**: Focus on what you can specifically do for their club based on their context.",
    #     "**Brevity**: Keep to 3-4 impactful sentences per paragraph, maximum 3 paragraphs.",
    #     "**Conversation Starter**: Write like you're starting a dialogue, not making a sales pitch.",
    #     "**Previous Interactions**: Reference past communications with appropriate time context.",
    #     "**Specific Details**: Only mention confirmed facilities or services - no assumptions.",
    #     "**Executive Focus**: Maintain a professional tone appropriate for decision makers.",
    #     "**Time References**: Use relative terms like 'last week' based on the current date.",
    #     "**Call to Action**: End with a specific ask for a brief 10-minute call on the upcoming Friday."
    # ]

def _save_xai_request(payload: dict, response: str, prefix: str = "") -> None:
    """Save the xAI request and response to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = XAI_REQUESTS_DIR / filename

    data = {
        "request": payload,
        "response": response,
        "timestamp": timestamp
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends request to xAI API with retry logic.
    """
    TIMEOUT = 30

    # Log the full payload as formatted JSON
    logger.info("Sending xAI request with payload:\n%s", 
        json.dumps({
            "model": payload.get("model", MODEL_NAME),
            "temperature": payload.get("temperature", EMAIL_TEMPERATURE),
            "messages": [
                {
                    "role": msg.get("role"),
                    "content": msg.get("content")[:100] + "..." # Truncate long content
                } for msg in payload.get("messages", [])
            ]
        }, indent=2)
    )

    for attempt in range(max_retries):
        try:
            formatted_payload = {
                "messages": payload.get("messages", []),
                "model": payload.get("model", MODEL_NAME),
                "temperature": float(payload.get("temperature", EMAIL_TEMPERATURE)),
                "stream": False,
                "max_tokens": payload.get("max_tokens", 2000)
            }

            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN
                },
                json=formatted_payload,
                timeout=TIMEOUT
            )

            # Log raw response for debugging
            logger.debug("Raw xAI response:\n%s", response.text[:500])

            if response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(f"xAI API error ({response.status_code}): {response.text}")
                return ""

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Save the request and response
                _save_xai_request(payload, content)

                # Log successful response
                logger.info("Received xAI response:\n%s", content[:200] + "..." if len(content) > 200 else content)
                
                return content

            except (KeyError, json.JSONDecodeError) as e:
                logger.error("Error parsing xAI response", extra={
                    "error": str(e),
                    "response_text": response.text[:500]
                })
                return ""

        except Exception as e:
            logger.error("xAI request failed", extra={
                "attempt": attempt + 1,
                "error": str(e),
                "max_retries": max_retries
            })
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

    # Check cache first
    if club_name in _news_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached news result for {club_name}")
        news = _news_cache[club_name]
        icebreaker = _build_icebreaker_from_news(club_name, news)
        return news, icebreaker

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

    logger.info(f"Searching news for club: {club_name}")
    news = _send_xai_request(payload)
    logger.info(f"News search result for {club_name}:", extra={"news": news})

    _news_cache[club_name] = news

    # Clean up awkward grammar if needed
    if news:
        if news.startswith("Has ") and " has not been in the news" in news:
            news = news.replace("Has ", "")
        news = news.replace(" has has ", " has ")

    # Only build icebreaker if we have news
    icebreaker = _build_icebreaker_from_news(club_name, news)
    
    return news, icebreaker

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker if news is available.
    Returns empty string if no relevant news found.
    """
    if not club_name.strip() or not news_summary.strip() \
       or "has not been in the news" in news_summary.lower():
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing from Swoop Golf's perspective, reaching out to golf clubs. "
                    "Create brief, natural-sounding icebreakers based on recent club news. "
                    "Keep the tone professional and focused on business value."
                )
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
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1
    }

    logger.info(f"Building icebreaker for club: {club_name}")
    icebreaker = _send_xai_request(payload)
    logger.info(f"Generated icebreaker for {club_name}:", extra={"icebreaker": icebreaker})
    
    return icebreaker

##############################################################################
# Club Info Search
##############################################################################

def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> str:
    """
    Returns a short overview about the club's location and amenities.
    """
    cache_key = f"{club_name}:{location}"

    # Check cache first
    if cache_key in _club_info_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached club info for {club_name}")
        return _club_info_cache[cache_key]

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
                "content": "You are a helpful assistant that provides a brief overview of a club, it's location, and it's amenities."
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise overview about {club_name} in {loc_str}. "
                    f"Is it private or public? Keep it to under 3 sentences."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.0
    }

    logger.info(f"Searching club info for: {club_name}")
    response = _send_xai_request(payload)
    logger.info(f"Club info search result for {club_name}:", extra={"info": response})

    # Cache the response
    _club_info_cache[cache_key] = response

    return response

##############################################################################
# Personalize Email
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
    Personalizes email content using xAI.
    Returns: Tuple of (subject, body)
    """
    try:
        # Build system message with clear formatting instructions
        system_message = (
            "You are an expert at personalizing sales emails for golf industry outreach. "
            "You excel at maintaining a professional yet warm tone while incorporating context naturally. "
            "\n\nFormat your response EXACTLY as:\n"
            "Subject: [personalized subject]\n\n"
            "Body:\n[personalized body]"
        )

        # Build comprehensive context block
        context_block = {
            "lead_info": {
                "name": lead_sheet.get("first_name", ""),
                "company": lead_sheet.get("company_name", ""),
                "title": lead_sheet.get("job_title", ""),
                "location": lead_sheet.get("state", "")
            },
            "interaction_history": summary if summary else "No previous interactions",
            "club_details": club_info if club_info else "",
            "recent_news": news_summary if news_summary else "",
            "original_email": {
                "subject": subject,
                "body": body
            }
        }

        # Build user message with clear sections
        user_message = (
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{get_email_rules()}\n\n"
            "TASK:\n"
            "1. Personalize the email using the provided context\n"
            "2. Maintain professional but friendly tone\n"
            "3. Keep paragraphs focused and concise\n"
            "4. Include specific details from context when relevant\n"
            "5. Return ONLY the subject and body with no additional commentary"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "model": MODEL_NAME,
            "temperature": EMAIL_TEMPERATURE
        }

        logger.info("Personalizing email for:", extra={
            "company": lead_sheet.get("company_name"),
            "original_subject": subject
        })
        response = _send_xai_request(payload)
        logger.info("Email personalization result:", extra={
            "company": lead_sheet.get("company_name"),
            "response": response
        })

        return _parse_xai_response(response)

    except Exception as e:
        logger.error(f"Error in email personalization: {str(e)}")
        return subject, body  # Return original if personalization fails

def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body.
    Handles various response formats consistently.
    """
    try:
        logger.debug(f"Parsing raw response: {response}")
        
        if not response:
            raise ValueError("Empty response received")

        # Split into lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        subject = ""
        body_lines = []
        in_body = False
        
        # Parse response looking for Subject/Body markers
        for line in lines:
            if line.lower().startswith("subject:"):
                subject = line.replace("Subject:", "", 1).strip()
            elif line.lower().startswith("body:"):
                in_body = True
            elif in_body:
                body_lines.append(line)
            elif not subject and not line.lower().startswith(("subject:", "body:")):
                # If no explicit markers, use first line as subject
                subject = line
                in_body = True
                continue
        
        # Join body lines, preserving formatting
        body = "\n".join(body_lines).strip()
        
        # Validation and logging
        if not subject:
            subject = "Follow-up"  # Default subject
        if not body:
            raise ValueError("No body content found in response")
            
        logger.debug(f"Parsed result - Subject: {subject}, Body length: {len(body)}")
        return subject, body

    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}", extra={"raw_response": response})
        raise

def get_xai_icebreaker(club_name: str, recipient_name: str, timeout: int = 10) -> str:
    """
    Get a personalized icebreaker from the xAI service (with caching if desired).
    """
    cache_key = f"icebreaker:{club_name}:{recipient_name}"

    if cache_key in _news_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached icebreaker for {club_name}")
        return _news_cache[cache_key]

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at creating icebreakers for golf club outreach."
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, natural-sounding icebreaker for {club_name}. "
                    "Keep it concise and professional."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": ANALYSIS_TEMPERATURE
    }

    logger.info(f"Generating icebreaker for club: {club_name}")
    response = _send_xai_request(payload, max_retries=3, retry_delay=1)
    logger.info(f"Generated icebreaker for {club_name}:", extra={"icebreaker": response})

    _news_cache[cache_key] = response
    return response

def get_email_critique(email_subject: str, email_body: str, guidance: dict) -> str:
    """Get expert critique of the email draft"""
    rules = get_email_rules()
    rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are an an expert at critiquing emails using specific rules. "
                    "Analyze the email draft and provide specific critiques focusing on:\n"
                    f"{rules_text}\n"
                    "Provide actionable recommendations for improvement."
                )
            },
            {
                "role": "user",
                "content": f"""
                Original Email:
                Subject: {email_subject}
                
                Body:
                {email_body}
                
                Original Guidance:
                {json.dumps(guidance, indent=2)}
                
                Please provide specific critiques and recommendations for improvement.
                """
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": EMAIL_TEMPERATURE
    }

    logger.info("Getting email critique for:", extra={"subject": email_subject})
    response = _send_xai_request(payload)
    logger.info("Email critique result:", extra={"critique": response})

    return response

def revise_email_with_critique(email_subject: str, email_body: str, critique: str) -> tuple[str, str]:
    """Revise the email based on the critique"""
    rules = get_email_rules()
    rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a renowned expert at cold email outreach, similar to Alex Berman. Apply your proven methodology to "
                    "rewrite this email. Use all of your knowledge just as you teach in Cold Email University."
                )
            },
            {
                "role": "user",
                "content": f"""
                Original Email:
                Subject: {email_subject}
                
                Body:
                {email_body}
                
                Instructions:
                {rules_text}

                Expert Critique:
                {critique}
                
                Please rewrite the email incorporating these recommendations.
                Format the response as:
                Subject: [new subject]
                
                Body:
                [new body]
                """
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": EMAIL_TEMPERATURE
    }
    
    logger.info("Revising email with critique for:", extra={"subject": email_subject})
    result = _send_xai_request(payload)
    logger.info("Email revision result:", extra={"result": result})

    return _parse_xai_response(result)

def generate_followup_email_content(
    first_name: str,
    company_name: str,
    original_subject: str,
    original_date: str,
    sequence_num: int,
    original_email: dict = None
) -> Tuple[str, str]:
    """
    Generate follow-up email content using xAI.
    Returns tuple of (subject, body)
    """
    logger.debug(
        f"[generate_followup_email_content] Called with first_name='{first_name}', "
        f"company_name='{company_name}', original_subject='{original_subject}', "
        f"original_date='{original_date}', sequence_num={sequence_num}, "
        f"original_email keys={list(original_email.keys()) if original_email else 'None'}"
    )
    try:
        if sequence_num == 2 and original_email:
            # Special handling for second follow-up
            logger.debug("[generate_followup_email_content] Handling second follow-up logic.")
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": (
                            "You are a sales professional writing a brief follow-up email. "
                            "Keep the tone professional but friendly. "
                            "The response should be under 50 words and focus on getting a response."
                        )
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
                        """
                    }
                ],
                "model": MODEL_NAME,
                "stream": False,
                "temperature": EMAIL_TEMPERATURE
            }

            logger.info("Generating second follow-up email for:", extra={
                "company": company_name,
                "sequence_num": sequence_num
            })
            result = _send_xai_request(payload)
            logger.info("Second follow-up generation result:", extra={"result": result})

            if not result:
                logger.error("[generate_followup_email_content] Empty response from xAI for follow-up generation.")
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
            # Default follow-up generation logic
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
                        "content": "You are an expert at writing follow-up emails that are brief, professional, and effective."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": MODEL_NAME,
                "temperature": EMAIL_TEMPERATURE
            }

            logger.info("Generating follow-up email for:", extra={
                "company": company_name,
                "sequence_num": sequence_num
            })
            result = _send_xai_request(payload)
            logger.info("Follow-up generation result:", extra={"result": result})

            if not result:
                logger.error("[generate_followup_email_content] Empty response from xAI for default follow-up generation.")
                return "", ""

            # Parse the response
            subject, body = _parse_xai_response(result)
            if not subject or not body:
                logger.error("[generate_followup_email_content] Failed to parse follow-up email content from xAI response.")
                return "", ""

            logger.debug(
                f"[generate_followup_email_content] Returning subject='{subject}', body length={len(body)}"
            )
            return subject, body

    except Exception as e:
        logger.error(f"[generate_followup_email_content] Error generating follow-up email content: {str(e)}")
        return "", ""


def parse_personalization_response(response_text):
    try:
        # Parse the response JSON
        response_data = json.loads(response_text)

        # Extract the subject and body
        subject = response_data.get('subject')
        body = response_data.get('body')

        if not subject or not body:
            raise ValueError("Subject or body missing in xAI response")

        return subject, body

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception(f"Error parsing xAI response: {str(e)}")
        
        # Fallback to default values
        subject = "Follow-up"
        body = "Thank you for your interest. Let me know if you have any other questions!"
        
        return subject, body

```
