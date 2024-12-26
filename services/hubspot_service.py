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
