"""
services/orchestrator_service.py

Service for managing orchestration-related operations including:
- Lead data fetching
- Previous interaction review
- Market research
- Competitor analysis
- Decision loop management
- Context summarization
"""

import openai
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime

from services.hubspot_service import HubspotService
from external.external_api import (
    market_research,
    review_previous_interactions
)
from utils.logging_setup import logger
from utils.exceptions import (
    LeadContextError,
    HubSpotError,
    ExternalAPIError,
    OpenAIError
)
from utils.doc_reader import DocReader
from config.constants import DEFAULTS

if TYPE_CHECKING:
    from services.leads_service import LeadsService


class OrchestratorService:
    """Service for managing orchestration-related operations."""

    def __init__(self, leads_service: 'LeadsService', hubspot_service: HubspotService):
        """Initialize the orchestrator service."""
        self.leads_service = leads_service
        self.hubspot_service = hubspot_service
        self.logger = logger
        self.doc_reader = DocReader()

    def get_lead_data(self, contact_id: str) -> Dict[str, Any]:
        """Get lead data from HubSpot."""
        try:
            data = self.hubspot_service.get_lead_data_from_hubspot(contact_id)
            return {
                "status": "success",
                "data": data,
                "error": None
            }
        except HubSpotError as e:
            error_msg = f"HubSpot error fetching lead data for {contact_id}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error fetching lead data for {contact_id}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }

    def review_interactions(self, contact_id: str) -> Dict[str, Any]:
        """Review previous interactions for a lead."""
        try:
            data = review_previous_interactions(contact_id)
            return {
                "status": "success",
                "data": data,
                "error": None
            }
        except ExternalAPIError as e:
            error_msg = f"External API error reviewing interactions for {contact_id}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error reviewing interactions for {contact_id}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }

    def analyze_competitors(self, company_name: str) -> Dict[str, Any]:
        """Analyze competitors for a company."""
        try:
            data = market_research(company_name)
            return {
                "status": "success",
                "data": data,
                "error": None
            }
        except ExternalAPIError as e:
            error_msg = f"External API error analyzing competitors for {company_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error analyzing competitors for {company_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "data": {},
                "error": error_msg
            }

    def personalize_message(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize a message for a lead."""
        try:
            lead_email = lead_data.get("email")
            if not lead_email:
                return {
                    "status": "error",
                    "data": {},
                    "error": "No email found in lead data"
                }
            
            # Import only when needed to avoid circular imports
            from services.leads_service import LeadsService
            
            data = self.leads_service.generate_lead_summary(lead_email)
            return {
                "status": "success",
                "data": data,
                "error": None
            }
        except Exception as e:
            self.logger.error("Failed to personalize message", extra={
                "error": str(e),
                "lead_email": lead_data.get("email")
            })
            return {
                "status": "error",
                "data": {},
                "error": str(e)
            }

    def create_context_summary(self, context: Dict[str, Any]) -> str:
        """Create a summary of the gathered information for the Sales Leader."""
        lead_data = context.get("lead_data", {})
        previous_interactions = context.get("previous_interactions", {})
        research_data = context.get("research_data", {})
        competitor_data = context.get("competitor_data", {})
        personalized_message = context.get("personalized_message", "")

        summary_lines = [
            "Context Summary:",
            f"Lead Data: {self._truncate_dict(lead_data)}",
            f"Previous Interactions: {self._truncate_dict(previous_interactions)}",
            f"Market Research: {self._truncate_dict(research_data)}",
            f"Competitor Analysis: {self._truncate_dict(competitor_data)}",
            "Proposed Personalized Message:",
            personalized_message[:300]  # short preview
        ]

        self.logger.debug("Created context summary.")
        return "\n".join(summary_lines)

    def prune_messages(self, messages: List[Dict[str, str]], max_messages: int = 10) -> List[Dict[str, str]]:
        """Prune or summarize message history to avoid token limits."""
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

    async def run_decision_loop(self, context: Dict[str, Any]) -> bool:
        """Run the decision loop with OpenAI model."""
        iteration = 0
        max_iterations = DEFAULTS["MAX_ITERATIONS"]

        self.logger.info("Entering decision loop...")
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"Decision loop iteration {iteration}/{max_iterations}")

            context["messages"] = self.prune_messages(context["messages"])

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=context["messages"],
                    temperature=0
                )
            except openai.error.OpenAIError as e:
                error_msg = f"OpenAI API error in decision loop: {str(e)}"
                self.logger.error(error_msg)
                raise OpenAIError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error in decision loop: {str(e)}"
                self.logger.error(error_msg)
                return False

            assistant_message = response.choices[0].message
            content = assistant_message.get("content", "").strip()
            self.logger.info(f"Assistant response at iteration {iteration}: {content}")

            context["messages"].append({"role": "assistant", "content": content})

            if "we are done" in content.lower():
                self.logger.info("Sales Leader indicated that we are done.")
                return True

            if iteration >= 2:
                self.logger.info("We have recommended next steps. Exiting loop.")
                return True

            context["messages"].append({
                "role": "user",
                "content": "What else should we consider?"
            })

        self.logger.info("Reached max iterations in decision loop without completion.")
        return False

    def _truncate_dict(self, data: dict, max_len: int = 200) -> str:
        """Safely truncate dictionary string representation."""
        text = str(data)
        if len(text) > max_len:
            text = text[:max_len] + "...(truncated)"
        return text
