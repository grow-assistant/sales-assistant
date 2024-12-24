"""
HubSpot service for managing HubSpot API interactions.
"""
import re
from typing import Dict, List, Optional, Any
import requests
from utils.formatting_utils import clean_html
from datetime import datetime
from dateutil.parser import parse as parse_date
from tenacity import retry, stop_after_attempt, wait_fixed
from config.settings import logger
from utils.exceptions import HubSpotError


class HubspotService:
    """Service class for HubSpot API operations."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.hubapi.com"):
        """
        Initialize HubSpot service with API credentials.
        
        Args:
            api_key: HubSpot API key
            base_url: Base URL for HubSpot API (default: https://api.hubapi.com)
        """
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

    def __init__(self, api_key: str, base_url: str = "https://api.hubapi.com"):
        """
        Initialize HubSpot service with API credentials.
        
        Args:
            api_key: HubSpot API key
            base_url: Base URL for HubSpot API (default: https://api.hubapi.com)
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 1

    @staticmethod
    def format_timestamp(ts: str) -> str:
        """Format ISO8601 timestamp into a more readable format."""
        try:
            dt = parse_date(ts)
            return dt.strftime("%b %d, %Y %I:%M %p")
        except:
            return ts


    def get_hubspot_leads(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch leads from HubSpot.
        
        Args:
            limit: Maximum number of leads to fetch
            
        Returns:
            List of lead records
        """
        url = f"{self.contacts_endpoint}/search"
        payload = {
            "filterGroups": [],
            "properties": ["email", "firstname", "lastname", "company", "jobtitle"],
            "limit": limit
        }

        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            leads = [{
                "id": res.get("id"),
                "email": res.get("properties", {}).get("email"),
                "firstname": res.get("properties", {}).get("firstname"),
                "lastname": res.get("properties", {}).get("lastname"),
                "company": res.get("properties", {}).get("company"),
                "jobtitle": res.get("properties", {}).get("jobtitle")
            } for res in results]
            logger.info(f"Retrieved {len(leads)} lead(s) from HubSpot.")
            return leads
        except requests.RequestException as e:
            raise HubSpotError(f"Error fetching leads: {str(e)}")

    def parse_email_body(self, body: str, recipient_email: str) -> dict:
        """Parse the email body into components: campaign, subject, text."""
        body = body.strip()
        campaign_match = re.search(r'Email (sent|opened) from campaign (.*?)(Subject|Text|$)', body, re.IGNORECASE)
        campaign = campaign_match.group(2).strip() if campaign_match else ""

        subject_match = re.search(r'Subject:\s*(.*?)(Text:|$)', body, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else ""

        text_match = re.search(r'Text:\s*(.*)$', body, re.IGNORECASE | re.DOTALL)
        text = text_match.group(1).strip() if text_match else ""

        if not campaign and "Email sent from campaign" not in body and "Email opened from campaign" not in body:
            text = body
            subject = ""
            campaign = ""

        from_email = "Swoop Golf Team" if "Email sent" in body else "Unknown"
        if "Ryan Donovan" in text:
            from_email = "Ryan Donovan"
        elif "Ty Hayes" in text:
            from_email = "Ty Hayes"

        to_email = recipient_email

        text = clean_html(text)
        subject = clean_html(subject)
        campaign = clean_html(campaign)

        return {
            "from": from_email,
            "to": to_email,
            "subject": subject or "No Subject",
            "body": text,
            "campaign": campaign
        }

    def get_all_emails_for_contact(self, contact_id: str) -> list:
        """
        Fetch all Email objects (sent/received) for a specific contact.
        
        Args:
            contact_id: HubSpot contact ID
            
        Returns:
            List of email records
        """
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
                resp = requests.post(self.emails_search_url, headers=self.headers, json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
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
            
            except requests.RequestException as e:
                raise HubSpotError(f"Error fetching emails for contact {contact_id}: {e}")

        return all_emails

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_contact_by_email(self, email: str) -> Optional[str]:
        """
        Find a contact by email address.
        
        Args:
            email: Contact's email address
            
        Returns:
            Contact ID if found, None otherwise
        """
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
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise HubSpotError(f"Network error searching for contact by email {email}: {str(e)}")

        data = response.json()
        results = data.get("results", [])
        if results:
            return results[0].get("id")
        else:
            logger.info(f"No contact found for {email}")
            return None

    def get_contact_properties(self, contact_id: str) -> dict:
        """
        Get properties for a contact.
        
        Args:
            contact_id: HubSpot contact ID
            
        Returns:
            Dictionary of contact properties
        """
        props = [
            "email", "jobtitle", "lifecyclestage", "phone",
            "hs_sales_email_last_replied", "firstname", "lastname"
        ]
        query_params = "&".join([f"properties={p}" for p in props])
        url = f"{self.contacts_endpoint}/{contact_id}?{query_params}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {})
        except requests.RequestException as e:
            raise HubSpotError(f"Error fetching contact properties: {str(e)}")

    def get_all_notes_for_contact(self, contact_id: str) -> list:
        """
        Get all notes for a contact.
        
        Args:
            contact_id: HubSpot contact ID
            
        Returns:
            List of note records
        """
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
            response = requests.post(self.notes_search_url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            notes = data.get("results", [])
            note_list = []
            for note in notes:
                props = note.get("properties", {})
                raw_html_string = props.get("hs_note_body", "")
                cleaned_text = clean_html(raw_html_string)

                note_list.append({
                    "id": note.get("id"),
                    "body": cleaned_text,
                    "createdate": props.get("hs_timestamp", ""),
                    "lastmodifieddate": props.get("hs_lastmodifieddate", "")
                })
            note_list.sort(key=lambda x: x["createdate"], reverse=True)
            return note_list
        except requests.RequestException as e:
            logger.error(f"Error fetching notes: {str(e)}")
            return []

    def get_associated_company_id(self, contact_id: str) -> Optional[str]:
        """
        Get company ID associated with a contact.
        
        Args:
            contact_id: HubSpot contact ID
            
        Returns:
            Company ID if found, None otherwise
        """
        url = f"{self.contacts_endpoint}/{contact_id}/associations/company"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0].get("id")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching associated company: {str(e)}")
            return None

    def get_company_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company data by ID.
        
        Args:
            company_id: HubSpot company ID
            
        Returns:
            Company data if found, None otherwise
        """
        properties = [
            "domain",
            "name",
            "city",
            "state",
            "hs_lastmodifieddate",
            "industry",
            "website"
        ]
        url = f"{self.companies_endpoint}/{company_id}?properties={','.join(properties)}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching company data: {str(e)}")
            return None

    def get_lead_data_from_hubspot(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive lead data including contact and company information.
        
        Args:
            email: Lead's email address
            
        Returns:
            Dictionary containing lead data if found, None otherwise
        """
        contact_id = self.get_contact_by_email(email)
        if not contact_id:
            return None

        contact_properties = self.get_contact_properties(contact_id)
        company_id = self.get_associated_company_id(contact_id)
        company_data = self.get_company_data(company_id) if company_id else None

        return {
            "contact_id": contact_id,
            "contact": contact_properties,
            "company": company_data
        }
