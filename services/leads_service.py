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
            
            return result
            
        except HubSpotError as e:
            if '404' in str(e):
                return {'error': '404 - Lead not found'}
            return {'error': str(e)}
