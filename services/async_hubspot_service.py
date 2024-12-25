"""
Asynchronous version of HubSpot service for improved API call efficiency.
"""
import asyncio
from typing import Dict, List, Optional, Any
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed
from config.settings import logger
from utils.exceptions import HubSpotError
from utils.formatting_utils import clean_html


class AsyncHubspotService:
    """Asynchronous service class for HubSpot API operations."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.hubapi.com"):
        """Initialize AsyncHubSpot service with API credentials."""
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

    async def get_contact_by_email(self, session: aiohttp.ClientSession, email: str) -> Optional[str]:
        """Find a contact by email address asynchronously."""
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
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                results = data.get("results", [])
                return results[0].get("id") if results else None
        except Exception as e:
            raise HubSpotError(f"Error searching for contact by email {email}: {str(e)}")

    async def get_contact_properties(self, session: aiohttp.ClientSession, contact_id: str) -> dict:
        """Get properties for a contact asynchronously."""
        props = [
            "email", "jobtitle", "lifecyclestage", "phone",
            "hs_sales_email_last_replied", "firstname", "lastname"
        ]
        query_params = "&".join([f"properties={p}" for p in props])
        url = f"{self.contacts_endpoint}/{contact_id}?{query_params}"

        try:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("properties", {})
        except Exception as e:
            raise HubSpotError(f"Error fetching contact properties: {str(e)}")

    async def get_all_emails_for_contact(self, session: aiohttp.ClientSession, contact_id: str) -> list:
        """Fetch all Email objects for a contact asynchronously."""
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
                async with session.post(self.emails_search_url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
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

    async def get_all_notes_for_contact(self, session: aiohttp.ClientSession, contact_id: str) -> list:
        """Get all notes for a contact asynchronously."""
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
            async with session.post(self.notes_search_url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                results = data.get("results", [])
                return [{
                    "id": note.get("id"),
                    "body": note["properties"].get("hs_note_body", ""),
                    "timestamp": note["properties"].get("hs_timestamp", ""),
                    "last_modified": note["properties"].get("hs_lastmodifieddate", "")
                } for note in results]
        except Exception as e:
            raise HubSpotError(f"Error fetching notes for contact {contact_id}: {str(e)}")

    async def get_associated_company_id(self, session: aiohttp.ClientSession, contact_id: str) -> Optional[str]:
        """Get the associated company ID for a contact asynchronously."""
        url = f"{self.contacts_endpoint}/{contact_id}/associations/company"
        
        try:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                results = data.get("results", [])
                return results[0].get("id") if results else None
        except Exception as e:
            raise HubSpotError(f"Error fetching associated company ID: {str(e)}")

    async def get_company_data(self, session: aiohttp.ClientSession, company_id: str) -> dict:
        """Get company data asynchronously."""
        if not company_id:
            return {}
            
        url = f"{self.companies_endpoint}/{company_id}"
        try:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            raise HubSpotError(f"Error fetching company data: {str(e)}")

    async def gather_lead_data(self, email: str) -> Dict[str, Any]:
        """
        Gather all lead data in parallel using asyncio.
        This method coordinates multiple async API calls efficiently.
        """
        async with aiohttp.ClientSession() as session:
            # 1. Get contact ID
            contact_id = await self.get_contact_by_email(session, email)
            if not contact_id:
                raise HubSpotError(f"No contact found for email: {email}")

            # 2. Fetch multiple data points in parallel
            contact_props, emails, notes, company_id = await asyncio.gather(
                self.get_contact_properties(session, contact_id),
                self.get_all_emails_for_contact(session, contact_id),
                self.get_all_notes_for_contact(session, contact_id),
                self.get_associated_company_id(session, contact_id)
            )

            # 3. Get company data if available
            company_data = await self.get_company_data(session, company_id) if company_id else {}

            # 4. Combine all data
            return {
                "id": contact_id,
                "properties": contact_props,
                "emails": emails,
                "notes": notes,
                "company_data": company_data
            }
