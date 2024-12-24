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
from agents.functions import call_function
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.exceptions import LeadContextError, HubSpotError
from config.settings import DEBUG_MODE, OPENAI_API_KEY
from services.orchestrator_service import OrchestratorService
from services.leads_service import LeadsService
from services.hubspot_service import HubspotService

load_dotenv()
openai.api_key = OPENAI_API_KEY

# Initialize services
_leads_service = LeadsService()
_hubspot_service = HubspotService()
_orchestrator_service = OrchestratorService(_leads_service, _hubspot_service)


class Context(TypedDict):
    lead_id: str
    domain_docs: Dict[str, str]
    messages: list  # conversation messages for GPT
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
    Run the sales leader workflow in two phases:
       1. Preparation: Gather all required info about the lead.
       2. Decision: Let the Sales Leader Agent decide the next steps,
          with pruning to avoid token overload.
    """
    result: OrchestrationResult = {
        'success': False,
        'lead_id': context['lead_id'],
        'actions_taken': [],
        'completion_time': datetime.now(),
        'error': None
    }

    try:
        # 1) Load the Outreach Decision Tree right away
        decision_tree_md = _orchestrator_service.doc_reader.read_doc(
            'templates/swoop_sales_outreach_decision_tree.md',
            fallback_content=""
        )
        context["messages"].append({
            "role": "system",
            "content": (
                "You have access to the Swoop Sales Outreach Decision Tree:\n\n" + decision_tree_md
            )
        })

        logger.info(f"Starting sales leader workflow for lead {context['lead_id']}")

        # 2) Define preparation steps
        preparation_steps = [
            ("get_lead_data_from_hubspot", {"contact_id": context["lead_id"]}),
            ("review_previous_interactions", {"contact_id": context["lead_id"]}),
            ("market_research", {}),
            ("analyze_competitors", {}),
            ("personalize_message", {})
        ]

        # 3) Iterate over the steps
        for func_name, args in preparation_steps:
            if func_name == "market_research":
                company_name = context.get("lead_data", {}).get("company")
                if company_name:
                    args["company_name"] = company_name
                else:
                    logger.warning("No company name found; skipping market research.")
                    continue

            elif func_name == "personalize_message":
                lead_data = context.get("lead_data")
                if lead_data:
                    args["lead_data"] = lead_data
                else:
                    logger.warning("Lead data missing; skipping message personalization")
                    continue

            logger.info(f"Running step: {func_name}")
            try:
                if func_name == "get_lead_data_from_hubspot":
                    result = _orchestrator_service.get_lead_data(args["contact_id"])
                elif func_name == "review_previous_interactions":
                    result = _orchestrator_service.review_interactions(args["contact_id"])
                elif func_name == "market_research":
                    result = _orchestrator_service.analyze_competitors(args["company_name"])
                elif func_name == "personalize_message":
                    result = _orchestrator_service.personalize_message(args["lead_data"])
                context[func_name.replace("get_", "")] = result
            except (LeadContextError, HubSpotError) as e:
                logger.error(f"Error in {func_name}: {e}")
                continue

        # 4) Decision Phase
        logger.info("Preparation complete. Creating context summary and initiating decision phase.")
        summary = _orchestrator_service.create_context_summary(context)
        context["messages"].append({"role": "assistant", "content": summary})
        context["messages"].append({
            "role": "user",
            "content": "Now that we have the info and decision tree, what's the next best step?"
        })

        decision_success = await _orchestrator_service.run_decision_loop(context)
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
    into one condensed message and trim them to keep token usage lower.
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
    or reaches max iterations. Prune messages each iteration to avoid token overload.
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


def create_context_summary(context: Context) -> str:
    """Create a summary of the gathered information for the Sales Leader."""
    lead_data = context.get("lead_data", {})
    previous_interactions = context.get("previous_interactions", {})
    research_data = context.get("research_data", {})
    competitor_data = context.get("competitor_data", {})
    personalized_message = context.get("personalized_message", "")

    summary_lines = [
        "Context Summary:",
        f"Lead Data: {truncate_dict(lead_data)}",
        f"Previous Interactions: {truncate_dict(previous_interactions)}",
        f"Market Research: {truncate_dict(research_data)}",
        f"Competitor Analysis: {truncate_dict(competitor_data)}",
        "Proposed Personalized Message:",
        personalized_message[:300]  # short preview
    ]

    logger.debug("Created context summary.")
    return "\n".join(summary_lines)


def truncate_dict(data: dict, max_len: int = 200) -> str:
    """
    Safely truncate dictionary string representation to avoid huge blocks
    of text in the summary.
    """
    text = str(data)
    if len(text) > max_len:
        text = text[:max_len] + "...(truncated)"
    return text
