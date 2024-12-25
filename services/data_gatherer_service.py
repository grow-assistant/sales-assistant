# services/data_gatherer_service.py

import json
import datetime
from typing import Dict, Any
from pathlib import Path

import asyncio
from services.async_hubspot_service import AsyncHubspotService
from external.external_api import (
    market_research,
    review_previous_interactions,
    determine_club_season
)
from hubspot_integration.data_enrichment import check_competitor_on_website
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY, PROJECT_ROOT


class DataGathererService:
    """
    Centralized service to gather all relevant data about a lead in one pass.
    Fetches HubSpot contact & company info, emails, competitor checks,
    interactions, market research, and season data.

    This version also saves the final lead context JSON to 'test_data/lead_contexts'
    for debugging or reference.
    """

    def __init__(self):
        self._hubspot = AsyncHubspotService(api_key=HUBSPOT_API_KEY)

    async def _gather_hubspot_data(self, lead_email: str) -> Dict[str, Any]:
        """Gather all HubSpot data asynchronously."""
        return await self._hubspot.gather_lead_data(lead_email)

    def gather_lead_data(self, lead_email: str) -> Dict[str, Any]:
        """
        Main entry point for gathering lead data.
        Coordinates async HubSpot calls with other synchronous operations.
        """
        # 1) Gather all HubSpot data asynchronously
        hubspot_data = asyncio.run(self._gather_hubspot_data(lead_email))
        
        contact_id = hubspot_data["id"]
        lead_data = {
            "id": contact_id,
            "properties": hubspot_data["properties"],
            "emails": hubspot_data["emails"]
        }
        
        company_data_raw = hubspot_data.get("company_data", {})

        # --- NEW: Flatten the company JSON for easy access in main.py
        parsed_company_data = {}
        company_id = None
        if company_data_raw:
            # Convert HubSpot's structure (id + properties dict) to a simpler format
            company_id = company_data_raw.get("id", "")
            props = company_data_raw.get("properties", {})
            parsed_company_data = {
                "hs_object_id": company_id,
                "name": props.get("name", ""),
                "city": props.get("city", ""),
                "state": props.get("state", ""),
                "domain": props.get("domain", ""),
                "website": props.get("website", "")
            }

        lead_data["company_data"] = parsed_company_data

        competitor = ""
        if parsed_company_data.get("website"):
            competitor = check_competitor_on_website(parsed_company_data["website"]) or ""

        # 5) External calls: Interactions, market research, season info
        company_name = parsed_company_data.get("name", "").strip()
        research_data = market_research(company_name) if company_name else {}
        interactions = review_previous_interactions(contact_id)
        city = parsed_company_data.get("city", "")
        state = parsed_company_data.get("state", "")
        season_info = determine_club_season(city, state)

        # 6) Build final lead_sheet for consistent usage across the app
        lead_sheet = {
            "metadata": {
                "contact_id": contact_id,
                "company_id": company_id,
                "lead_email": lead_email,
                "status": "success"
            },
            "lead_data": lead_data,
            "analysis": {
                "competitor_analysis": competitor,
                "research_data": research_data,
                "previous_interactions": interactions,
                "season_data": season_info
            }
        }

        # 7) Save the lead_sheet to disk so we can review the final context
        self._save_lead_context(lead_sheet, lead_email)

        logger.info(f"Data gathered successfully for {lead_email} (contact_id={contact_id}).")
        return lead_sheet

    # ------------------------------------------------------------------------
    # PRIVATE METHODS FOR SAVING THE LEAD CONTEXT LOCALLY
    # ------------------------------------------------------------------------
    def _save_lead_context(self, lead_sheet: Dict[str, Any], lead_email: str) -> None:
        """
        Save the lead_sheet dictionary to 'test_data/lead_contexts' as a JSON file.
        """
        try:
            context_dir = self._create_context_directory()
            filename = self._generate_context_filename(lead_email)
            file_path = context_dir / filename

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(lead_sheet, f, indent=2, ensure_ascii=False)

            logger.info(f"Lead context saved at: {file_path.resolve()}")
        except Exception as e:
            logger.warning(f"Failed to save lead context (non-critical): {e}")

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
