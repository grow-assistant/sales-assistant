import os
import requests
import logging
import re
import html
from datetime import datetime
from dateutil.parser import parse as parse_date
from config.settings import HEADERS
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hubapi.com"
CONTACTS_ENDPOINT = f"{BASE_URL}/crm/v3/objects/contacts"
COMPANIES_ENDPOINT = f"{BASE_URL}/crm/v3/objects/companies"
NOTES_SEARCH_URL = f"{BASE_URL}/crm/v3/objects/notes/search"
TASKS_ENDPOINT = f"{BASE_URL}/crm/v3/objects/tasks"
EMAILS_SEARCH_URL = f"{BASE_URL}/crm/v3/objects/emails/search"


def get_hubspot_leads():
    """
    Retrieve a list of contact IDs from HubSpot.
    This is a placeholder implementation. Adjust the filterGroups 
    and properties as needed to get the desired leads.
    """
    url = f"{CONTACTS_ENDPOINT}/search"
    payload = {
        "filterGroups": [],
        "properties": ["email", "firstname", "lastname", "company", "jobtitle"],
        "limit": 10  # limit can be adjusted as needed
    }

    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        lead_ids = [res.get("id") for res in results if res.get("id")]
        logger.info(f"Retrieved {len(lead_ids)} lead(s) from HubSpot.")
        return lead_ids
    except requests.RequestException as e:
        logger.error(f"Error fetching leads: {str(e)}")
        return []


def clean_html(raw_html: str) -> str:
    """
    Remove HTML tags and decode HTML entities from a given string.
    """
    text_only = re.sub('<[^<]+?>', '', raw_html)
    text_only = html.unescape(text_only)
    text_only = text_only.replace('\u00a0', ' ')
    text_only = re.sub(r'\s+', ' ', text_only)
    return text_only.strip()


def format_timestamp(ts: str) -> str:
    """
    Format ISO8601 timestamp into a more readable format.
    Example: '2024-12-12T16:12:52.812Z' -> 'Dec 12, 2024 11:12 AM'
    """
    try:
        dt = parse_date(ts)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        return ts


def parse_email_body(body: str, recipient_email: str) -> dict:
    """
    Attempt to parse the email body into components: campaign, subject, text.
    """
    body = body.strip()
    campaign_match = re.search(r'Email (sent|opened) from campaign (.*?)(Subject|Text|$)', body, re.IGNORECASE)
    campaign = campaign_match.group(2).strip() if campaign_match else ""

    subject_match = re.search(r'Subject:\s*(.*?)(Text:|$)', body, re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ""

    text_match = re.search(r'Text:\s*(.*)$', body, re.IGNORECASE | re.DOTALL)
    text = text_match.group(1).strip() if text_match else ""

    if not campaign and "Email sent from campaign" not in body and "Email opened from campaign" not in body:
        # Treat entire body as text if no structure found
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


def get_all_emails_for_contact(contact_id: str) -> list:
    """
    Fetch all Email objects (sent/received) for a specific contact via HubSpot's CRM v3.
    Returns a list of dicts, each representing an email engagement.
    """
    all_emails = []
    after = None  # for pagination
    has_more = True
    
    while has_more:
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            # We want Email objects associated with the given contact
                            "propertyName": "associations.contact",  
                            "operator": "EQ",
                            "value": contact_id
                        }
                    ]
                }
            ],
            # You can request more properties if needed:
            "properties": ["hs_email_subject", "hs_email_text", "hs_email_direction", 
                           "hs_email_status", "hs_timestamp"]
        }
        if after:
            payload["after"] = after

        try:
            resp = requests.post(EMAILS_SEARCH_URL, headers=HEADERS, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            for item in results:
                # item["properties"] has the actual fields we requested
                all_emails.append({
                    "id": item.get("id"),
                    "subject": item["properties"].get("hs_email_subject", ""),
                    "body_text": item["properties"].get("hs_email_text", ""), 
                    "direction": item["properties"].get("hs_email_direction", ""),
                    "status": item["properties"].get("hs_email_status", ""),
                    "timestamp": item["properties"].get("hs_timestamp", "")
                })
            
            # Check if there's more pages
            paging = data.get("paging", {}).get("next")
            if paging and paging.get("after"):
                after = paging["after"]
            else:
                has_more = False
        
        except requests.RequestException as e:
            logger.error(f"Error fetching emails for contact {contact_id}: {e}")
            break

    return all_emails


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_contact_by_email(email: str) -> str:
    url = f"{CONTACTS_ENDPOINT}/search"
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
        response = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Network error searching for contact by email {email}: {str(e)}")
        return None

    data = response.json()
    results = data.get("results", [])
    if results:
        return results[0].get("id")
    else:
        logger.info(f"No contact found for {email}")
        return None


def get_contact_properties(contact_id: str) -> dict:
    props = [
        "email", "jobtitle", "lifecyclestage", "phone",
        "hs_sales_email_last_replied", "firstname", "lastname"
    ]
    query_params = "&".join([f"properties={p}" for p in props])
    url = f"{CONTACTS_ENDPOINT}/{contact_id}?{query_params}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("properties", {})
    except requests.RequestException as e:
        logger.error(f"Error fetching contact properties: {str(e)}")
        return {}


def get_all_notes_for_contact(contact_id: str) -> list:
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
        response = requests.post(NOTES_SEARCH_URL, headers=HEADERS, json=payload, timeout=10)
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
        # Sort notes by createdate descending (newest first)
        note_list.sort(key=lambda x: x["createdate"], reverse=True)
        return note_list
    except requests.RequestException as e:
        logger.error(f"Error fetching notes: {str(e)}")
        return []


def get_associated_company_id(contact_id: str) -> str:
    url = f"{CONTACTS_ENDPOINT}/{contact_id}/associations/companies"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            return results[0].get("id")
        else:
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching associated company: {str(e)}")
        return None


def get_company_data(company_id: str) -> dict:
    if not company_id:
        return {}
    url = f"{COMPANIES_ENDPOINT}/{company_id}?properties=name&properties=city&properties=state&properties=annualrevenue&properties=createdate&properties=hs_lastmodifieddate&properties=hs_object_id"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("properties", {})
    except requests.RequestException as e:
        logger.error(f"Error fetching company data: {str(e)}")
        return {}


def get_lead_data_from_hubspot(contact_id: str) -> dict:
    """
    Retrieve lead data by combining contact properties and notes.
    """
    props = get_contact_properties(contact_id)
    notes = get_all_notes_for_contact(contact_id)
    emails = get_all_emails_for_contact(contact_id)
    lead_data = {**props, "contact_id": contact_id, "notes": notes, "emails": emails}
    return lead_data
