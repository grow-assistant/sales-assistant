"""
HubSpot service for API operations.
"""
from typing import Dict, List, Optional, Any
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from config.settings import logger
from utils.exceptions import HubSpotError
from utils.formatting_utils import clean_html
from datetime import datetime


class HubspotService:
    """Service class for HubSpot API operations."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.hubapi.com"):
        """Initialize HubSpot service with API credentials."""
        logger.debug("Initializing HubspotService")
        if not api_key:
            logger.error("No API key provided to HubspotService")
            raise ValueError("HubSpot API key is required")
            
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.debug(f"HubspotService initialized with base_url: {self.base_url}")
        
        # Endpoints
        self.contacts_endpoint = f"{self.base_url}/crm/v3/objects/contacts"
        self.companies_endpoint = f"{self.base_url}/crm/v3/objects/companies"
        self.notes_search_url = f"{self.base_url}/crm/v3/objects/notes/search"
        self.tasks_endpoint = f"{self.base_url}/crm/v3/objects/tasks"
        self.emails_search_url = f"{self.base_url}/crm/v3/objects/emails/search"

        # Property mappings
        self.hubspot_property_mapping = {
            "name": "name",
            "company_short_name": "company_short_name",
            "club_type": "club_type",
            "facility_complexity": "facility_complexity",
            "geographic_seasonality": "geographic_seasonality",
            "has_pool": "has_pool",
            "has_tennis_courts": "has_tennis_courts",
            "number_of_holes": "number_of_holes",
            "public_private_flag": "public_private_flag",
            "club_info": "club_info",
            "start_month": "start_month",
            "end_month": "end_month",
            "peak_season_start_month": "peak_season_start_month",
            "peak_season_end_month": "peak_season_end_month",
            "competitor": "competitor",
            "domain": "domain",
            "notes_last_contacted": "notes_last_contacted",
            "num_contacted_notes": "num_contacted_notes",
            "num_associated_contacts": "num_associated_contacts"
        }

        self.property_value_mapping = {
            "club_type": {
                "Private": "Private",
                "Private Course": "Private",
                "Country Club": "Country Club",
                "Public": "Public",
                "Public - Low Daily Fee": "Public - Low Daily Fee",
                "Municipal": "Municipal",
                "Semi-Private": "Semi-Private",
                "Resort": "Resort",
                "Management Company": "Management Company",
                "Unknown": "Unknown"
            },
            "facility_complexity": {
                "Single-Course": "Standard",
                "Multi-Course": "Multi-Course",
                "Resort": "Resort",
                "Unknown": "Unknown"
            },
            "geographic_seasonality": {
                "Year-Round": "Year-Round Golf",
                "Peak Summer Season": "Peak Summer Season",
                "Short Summer Season": "Short Summer Season",
                "Unknown": "Unknown"
            },
            "competitor": {
                "Club Essentials": "Club Essentials",
                "Jonas": "Jonas",
                "Northstar": "Northstar",
                "Unknown": "Unknown"
            }
        }

    def search_country_clubs(self, batch_size: int = 25) -> List[Dict[str, Any]]:
        """Search for Country Club type companies in HubSpot."""
        url = f"{self.companies_endpoint}/search"
        all_results = []
        after = None
        
        while True:
            payload = {
                "limit": batch_size,
                "properties": [
                    "name", "company_short_name", "city", "state",
                    "club_type", "facility_complexity", "geographic_seasonality",
                    "has_pool", "has_tennis_courts", "number_of_holes",
                    "public_private_flag", "club_info",
                    "peak_season_start_month", "peak_season_end_month",
                    "start_month", "end_month", "domain",
                    "notes_last_contacted", "num_contacted_notes",
                    "num_associated_contacts"
                ],
                "filterGroups": [{
                    "filters": [{
                        "propertyName": "club_type",
                        "operator": "NOT_HAS_PROPERTY",
                        "value": None

                    }]
                }]
            }
            
            if after:
                payload["after"] = after
                
            try:
                response = self._make_hubspot_post(url, payload)
                results = response.get("results", [])
                all_results.extend(results)
                
                paging = response.get("paging", {})
                next_link = paging.get("next", {}).get("after")
                if not next_link:
                    break
                after = next_link
                
            except Exception as e:
                logger.error(f"Error fetching Country Clubs: {str(e)}")
                break
                
        return all_results

    def update_company_properties(self, company_id: str, properties: Dict[str, Any]) -> bool:
        """Update company properties in HubSpot."""
        try:
            mapped_updates = {}
            
            # Map and transform properties
            for internal_key, value in properties.items():
                logger.debug(f"Processing property - Key: {internal_key}, Value: {value}")
                
                if value is None or value == "":
                    logger.debug(f"Skipping empty value for key: {internal_key}")
                    continue

                # Get the HubSpot property name from mapping
                hubspot_key = self.hubspot_property_mapping.get(internal_key)
                if not hubspot_key:
                    logger.warning(f"No HubSpot mapping for property: {internal_key}")
                    continue

                # Special handling for company_short_name
                if internal_key == "company_short_name":
                    value = str(value).strip()[:100]  # Ensure it's a string and within length limit
                    mapped_updates[hubspot_key] = value  # Add directly to mapped_updates
                    logger.debug(f"Processed company_short_name: {value}")
                    continue  # Skip the rest of the processing for this field

                # Rest of the property processing...
                try:
                    # Apply enum value transformations
                    if internal_key in self.property_value_mapping:
                        original_value = value
                        value = self.property_value_mapping[internal_key].get(str(value), value)
                        logger.debug(f"Enum transformation for {internal_key}: {original_value} -> {value}")

                    # Type-specific handling
                    if internal_key in ["number_of_holes", "start_month", "end_month", 
                                      "peak_season_start_month", "peak_season_end_month",
                                      "notes_last_contacted", "num_contacted_notes",
                                      "num_associated_contacts"]:
                        logger.debug(f"Converting numeric value for {internal_key}: {value}")
                        value = int(value) if str(value).isdigit() else 0
                    elif internal_key in ["has_pool", "has_tennis_courts"]:
                        logger.debug(f"Converting boolean value for {internal_key}: {value}")
                        value = "Yes" if str(value).lower() in ["yes", "true"] else "No"
                    elif internal_key == "club_info":
                        logger.debug(f"Truncating club_info from length {len(str(value))}")
                        value = str(value)[:5000]

                except Exception as e:
                    logger.error(f"Error transforming {internal_key}: {str(e)}")
                    continue

                mapped_updates[hubspot_key] = value

            # Debug logging
            logger.debug("Final HubSpot payload:")
            logger.debug(f"Company ID: {company_id}")
            logger.debug("Properties:")
            for key, value in mapped_updates.items():
                logger.debug(f"  {key}: {value} (Type: {type(value)})")

            url = f"{self.companies_endpoint}/{company_id}"
            payload = {"properties": mapped_updates}
            
            logger.info(f"Making PATCH request to HubSpot - URL: {url}")
            logger.debug(f"Request payload: {payload}")
            
            response = self._make_hubspot_patch(url, payload)
            success = bool(response)
            logger.info(f"HubSpot update {'successful' if success else 'failed'} for company {company_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating company {company_id}: {str(e)}")
            return False

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
        """
        Get company data, including the 15 fields required:
        name, city, state, annualrevenue, createdate, hs_lastmodifieddate,
        hs_object_id, club_type, facility_complexity, has_pool,
        has_tennis_courts, number_of_holes, geographic_seasonality,
        public_private_flag, club_info, peak_season_start_month,
        peak_season_end_month, start_month, end_month, notes_last_contacted,
        num_contacted_notes, num_associated_contacts.
        """
        if not company_id:
            return {}
            
        url = (
            f"{self.companies_endpoint}/{company_id}?"
            "properties=name"
            "&properties=company_short_name"
            "&properties=city"
            "&properties=state"
            "&properties=annualrevenue"
            "&properties=createdate"
            "&properties=hs_lastmodifieddate"
            "&properties=hs_object_id"
            "&properties=club_type"
            "&properties=facility_complexity"
            "&properties=has_pool"
            "&properties=has_tennis_courts"
            "&properties=number_of_holes"
            "&properties=geographic_seasonality"
            "&properties=public_private_flag"
            "&properties=club_info"
            "&properties=peak_season_start_month"
            "&properties=peak_season_end_month"
            "&properties=start_month"
            "&properties=end_month"
            "&properties=notes_last_contacted"
            "&properties=num_contacted_notes"
            "&properties=num_associated_contacts"
        )
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

    def get_random_contacts(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get a random sample of contact email addresses from HubSpot.
        
        Args:
            count: Number of random contacts to retrieve (default: 3)
            
        Returns:
            List of dicts containing contact info (email, name, etc.)
        """
        try:
            # First, get total count of contacts
            url = f"{self.contacts_endpoint}/search"
            payload = {
                "filterGroups": [],  # No filters to get all contacts
                "properties": ["email", "firstname", "lastname", "company"],
                "limit": 1,  # Just need count
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            total = response.json().get("total", 0)
            
            if total == 0:
                logger.warning("No contacts found in HubSpot")
                return []
            
            # Generate random offset to get different contacts each time
            import random
            random_offset = random.randint(0, max(0, total - count * 2))
            
            # Get a batch starting from random offset
            batch_size = min(count * 2, total)  # Get 2x needed to ensure enough valid contacts
            payload.update({
                "limit": batch_size,
                "after": str(random_offset)  # Add random offset
            })
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            contacts = response.json().get("results", [])
            
            # Randomly sample from the batch
            selected = random.sample(contacts, min(count, len(contacts)))
            
            # Format the results
            results = []
            for contact in selected:
                props = contact.get("properties", {})
                results.append({
                    "id": contact.get("id"),
                    "email": props.get("email"),
                    "first_name": props.get("firstname"),
                    "last_name": props.get("lastname"),
                    "company": props.get("company")
                })
            
            logger.debug(f"Retrieved {len(results)} random contacts from HubSpot (offset: {random_offset})")
            return results
            
        except Exception as e:
            logger.error(f"Error getting random contacts: {str(e)}")
            return []

    def _make_hubspot_post(self, url: str, payload: dict) -> dict:
        """
        Make a POST request to HubSpot API with retries.
        
        Args:
            url: The endpoint URL
            payload: The request payload
            
        Returns:
            dict: The JSON response from HubSpot
        """
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HubSpot API error: {str(e)}")
            raise HubSpotError(f"Failed to make HubSpot POST request: {str(e)}")
            
    def _make_hubspot_get(self, url: str, params: Dict = None) -> Dict[str, Any]:
        """Make a GET request to HubSpot API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _make_hubspot_patch(self, url: str, payload: Dict) -> Any:
        """Make a PATCH request to HubSpot API."""
        try:
            logger.debug(f"Making PATCH request to: {url}")
            logger.debug(f"Headers: {self.headers}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.patch(url, headers=self.headers, json=payload)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response body: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            if hasattr(e.response, 'text'):
                logger.error(f"Response body: {e.response.text}")
            raise HubSpotError(f"PATCH request failed: {str(e)}")

    def get_company_by_id(self, company_id: str, properties: List[str]) -> Dict[str, Any]:
        """Get company by ID with specified properties."""
        try:
            url = f"{self.companies_endpoint}/{company_id}"
            params = {
                "properties": properties
            }
            response = self._make_hubspot_get(url, params=params)
            return response
        except Exception as e:
            logger.error(f"Error getting company {company_id}: {e}")
            return {}

    def get_contacts_from_list(self, list_id: str) -> List[Dict[str, Any]]:
        """Get all contacts from a specified HubSpot list."""
        url = f"{self.base_url}/contacts/v1/lists/{list_id}/contacts/all"
        all_contacts = []
        vidOffset = 0
        has_more = True
        
        while has_more:
            try:
                params = {
                    "count": 100,
                    "vidOffset": vidOffset
                }
                response = self._make_hubspot_get(url, params)
                
                contacts = response.get("contacts", [])
                all_contacts.extend(contacts)
                
                has_more = response.get("has-more", False)
                vidOffset = response.get("vid-offset", 0)
                
            except Exception as e:
                logger.error(f"Error fetching contacts from list: {str(e)}")
                break
        
        return all_contacts

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully deleted contact {contact_id} from HubSpot")
            return True
        except Exception as e:
            logger.error(f"Error deleting contact {contact_id}: {str(e)}")
            return False

    def mark_contact_as_bounced(self, email: str) -> bool:
        """Mark a contact as bounced in HubSpot."""
        try:
            # Add detailed logging
            logger.info(f"Marking contact as bounced in HubSpot: {email}")
            
            # Get the contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"Contact not found in HubSpot: {email}")
                return False
            
            # Update the contact properties
            properties = {
                "email_bounced": "true",
                "email_bounced_date": datetime.now().strftime("%Y-%m-%d"),
                "email_bounced_reason": "Hard bounce - Invalid recipient"
            }
            
            # Make the API call
            success = self.update_contact(contact['id'], properties)
            
            if success:
                logger.info(f"Successfully marked contact as bounced in HubSpot: {email}")
            else:
                logger.error(f"Failed to mark contact as bounced in HubSpot: {email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking contact as bounced in HubSpot: {str(e)}")
            return False

    def create_contact(self, properties: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Create a new contact in HubSpot.
        
        Args:
            properties (Dict[str, str]): Dictionary of contact properties
                Required keys: email
                Optional keys: firstname, lastname, company, jobtitle, phone
                
        Returns:
            Optional[Dict[str, Any]]: The created contact data or None if creation fails
        """
        try:
            email = properties.get('email')
            logger.info(f"Creating new contact in HubSpot with email: {email}")
            logger.debug(f"Contact properties: {properties}")
            
            # Validate required email property
            if not email:
                logger.error("Cannot create contact: email is required")
                return None
            
            # Check if contact already exists
            existing_contact = self.get_contact_by_email(email)
            if existing_contact:
                logger.warning(f"Contact already exists with email {email}. Updating instead.")
                # Update existing contact with new properties
                if self.update_contact(existing_contact['id'], properties):
                    logger.info(f"Successfully updated existing contact: {existing_contact['id']}")
                    return existing_contact
                return None
            
            payload = {
                "properties": {
                    key: str(value) for key, value in properties.items() if value and value != 'Unknown'
                }
            }
            
            logger.debug(f"Making create contact request with payload: {payload}")
            response = requests.post(
                self.contacts_endpoint,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 409:
                logger.warning(f"Conflict creating contact {email} - contact may already exist")
                # Try to get the existing contact again (in case it was just created)
                existing_contact = self.get_contact_by_email(email)
                if existing_contact:
                    logger.info(f"Found existing contact after conflict: {existing_contact['id']}")
                    if self.update_contact(existing_contact['id'], properties):
                        return existing_contact
                return None
            
            response.raise_for_status()
            contact_data = response.json()
            logger.info(f"Successfully created new contact: {contact_data.get('id')}")
            logger.debug(f"New contact data: {contact_data}")
            return contact_data
            
        except Exception as e:
            logger.error(f"Error creating contact in HubSpot: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    def get_contact_associations(self, contact_id: str) -> List[Dict[str, Any]]:
        """Get all associations for a contact."""
        url = f"{self.contacts_endpoint}/{contact_id}/associations"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching contact associations: {str(e)}")
            return []

    def create_association(self, from_id: str, to_id: str, association_type: str) -> bool:
        """Create an association between two objects."""
        url = f"{self.contacts_endpoint}/{from_id}/associations/{association_type}/{to_id}"
        
        try:
            response = requests.put(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully created association between {from_id} and {to_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating association: {str(e)}")
            return False

    def update_contact(self, contact_id: str, properties: Dict[str, Any]) -> bool:
        """Update contact properties in HubSpot."""
        try:
            url = f"{self.contacts_endpoint}/{contact_id}"
            payload = {"properties": properties}
            
            response = requests.patch(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating contact {contact_id}: {str(e)}")
            return False

    def mark_do_not_contact(self, email: str, company_name: str = None) -> bool:
        """
        Mark a contact as 'Do Not Contact' and update related properties.
        
        Args:
            email: Contact's email address
            company_name: Optional company name for template
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing do-not-contact request for {email}")
            
            # Get contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"Contact not found for email: {email}")
                return False
            
            contact_id = contact.get('id')
            
            # Update contact properties
            properties = {
                "hs_lead_status": "DQ",  # Set lead status to DQ
                "do_not_contact": "true",
                "do_not_contact_reason": "Customer Request",
                "lifecyclestage": "Other",
                "hs_marketable_reason_id": "UNSUBSCRIBED",
                "hs_marketable_status": "NO",
                "hs_marketable_until_renewal": "false"
            }
            
            # Update the contact
            if not self.update_contact(contact_id, properties):
                logger.error(f"Failed to update contact properties for {email}")
                return False
            
            # Get first name for template
            first_name = contact.get('properties', {}).get('firstname', '')
            company_short = company_name or contact.get('properties', {}).get('company', '')
            
            # Generate response from template
            template_vars = {
                "firstname": first_name,
                "company_short_name": company_short
            }
            
            logger.info(f"Successfully marked {email} as do-not-contact")
            return True
            
        except Exception as e:
            logger.error(f"Error processing do-not-contact request: {str(e)}")
            return False

    def mark_contact_as_dq(self, email: str, reason: str) -> bool:
        """Mark a contact as disqualified in HubSpot."""
        try:
            # First get the contact
            contact = self.get_contact_by_email(email)
            if not contact:
                logger.warning(f"No contact found in HubSpot for {email}")
                return False

            contact_id = contact.get('id')
            if not contact_id:
                logger.warning(f"Contact found but no ID for {email}")
                return False

            # Update the contact properties to mark as DQ
            properties = {
                'lifecyclestage': 'disqualified',
                'hs_lead_status': 'DQ',
                'dq_reason': reason,
                'dq_date': datetime.now().strftime('%Y-%m-%d')
            }

            # Update the contact in HubSpot
            url = f"{self.base_url}/objects/contacts/{contact_id}"
            response = requests.patch(
                url,
                json={'properties': properties},
                headers=self.headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully marked {email} as DQ in HubSpot")
                return True
            else:
                logger.error(f"Failed to mark {email} as DQ in HubSpot. Status code: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error marking contact as DQ in HubSpot: {str(e)}")
            return False

    def get_contacts_by_company_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get contacts in HubSpot whose email domain matches the given company domain.
        
        Args:
            domain (str): The company domain to search for (e.g., "example.com")
            
        Returns:
            List[Dict[str, Any]]: List of contact records matching the domain
        """
        try:
            logger.info(f"Searching for contacts with company domain: {domain}")
            url = f"{self.contacts_endpoint}/search"
            
            # Updated payload structure to match HubSpot's API requirements
            payload = {
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "email",
                                "operator": "CONTAINS_TOKEN",  # Changed from CONTAINS to CONTAINS_TOKEN
                                "value": domain
                            }
                        ]
                    }
                ],
                "sorts": [
                    {
                        "propertyName": "createdate",
                        "direction": "DESCENDING"
                    }
                ],
                "properties": [
                    "email", 
                    "firstname", 
                    "lastname", 
                    "company", 
                    "jobtitle",
                    "phone",
                    "createdate",
                    "lastmodifieddate"
                ],
                "limit": 100
            }
            
            try:
                response = self._make_hubspot_post(url, payload)
                
                if not response:
                    logger.warning(f"No response received from HubSpot for domain: {domain}")
                    return []
                    
                results = response.get("results", [])
                
                # Filter results to ensure exact domain match
                filtered_results = []
                for contact in results:
                    contact_email = contact.get("properties", {}).get("email", "")
                    if contact_email and contact_email.lower().endswith(f"@{domain.lower()}"):
                        filtered_results.append(contact)
                
                logger.info(f"Found {len(filtered_results)} contacts for domain {domain}")
                return filtered_results
                
            except Exception as e:
                logger.error(f"Error in HubSpot API request: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"Error getting contacts by company domain: {str(e)}")
            return []
