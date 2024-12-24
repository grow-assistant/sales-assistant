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
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.leads_service import LeadsService
from services.hubspot_service import HubspotService
from external.external_api import (
    market_research,
    review_previous_interactions
)
from utils.logging_setup import logger
from utils.exceptions import LeadContextError, HubSpotError
from utils.doc_reader import DocReader
from config.constants import DEFAULTS


class OrchestratorService:
    """Service for managing orchestration-related operations."""

    def __init__(self, leads_service: LeadsService, hubspot_service: HubspotService):
        """Initialize the orchestrator service."""
        self.leads_service = leads_service
        self.hubspot_service = hubspot_service
        self.logger = logger
        self.doc_reader = DocReader()

    def get_lead_data(self, contact_id: str) -> Dict[str, Any]:
        """Get lead data from HubSpot."""
        return self.hubspot_service.get_lead_data_from_hubspot(contact_id)

    def review_interactions(self, contact_id: str) -> Dict[str, Any]:
        """Review previous interactions for a lead."""
        try:
            return review_previous_interactions(contact_id)
        except Exception as e:
            raise LeadContextError(f"Error reviewing interactions for {contact_id}: {e}")

    def analyze_competitors(self, company_name: str) -> Dict[str, Any]:
        """Analyze competitors for a company."""
        try:
            return market_research(company_name)
        except Exception as e:
            raise LeadContextError(f"Error analyzing competitors for {company_name}: {e}")

    def personalize_message(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize a message for a lead."""
        lead_email = lead_data.get("email")
        if not lead_email:
            raise LeadContextError("No email found in lead data")
        return self.leads_service.generate_lead_summary(lead_email)

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
            except Exception as e:
                self.logger.error(f"Error calling OpenAI API in decision loop: {str(e)}")
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
