import sys
import os
from datetime import datetime
from pprint import pprint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger

def format_timestamp(timestamp: str) -> str:
    """Convert HubSpot timestamp to readable format."""
    try:
        ts = int(timestamp) / 1000  # Convert milliseconds to seconds
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'N/A'

def get_first_50_words(text: str) -> str:
    """Get first 50 words of a text string."""
    if not text:
        return "No content"
    # Split on whitespace and remove empty strings
    words = [w for w in text.split() if w.strip()]
    # Join first 50 words
    return ' '.join(words[:50]) + ('...' if len(words) > 50 else '')

def test_get_company_emails(test_email: str, debug: bool = True) -> None:
    """
    Test function to get all contacts and their incoming emails from the same company domain.
    
    Args:
        test_email (str): The email address to test with
        debug (bool): Whether to print debug information
    """
    try:
        hubspot = HubspotService(HUBSPOT_API_KEY)
        
        # First get the contact
        contact = hubspot.get_contact_by_email(test_email)
        if not contact:
            print(f"❌ No contact found for email: {test_email}")
            return
            
        # Extract the domain from the test email
        domain = test_email.split('@')[-1]
        print(f"\n🔍 Searching for contacts with domain: {domain}")
        
        # Get all contacts with the same domain
        domain_contacts = hubspot.get_contacts_by_company_domain(domain)
        
        if not domain_contacts:
            print(f"❌ No contacts found with domain: {domain}")
            return
            
        # Print results
        print(f"\n✅ Found {len(domain_contacts)} contacts with domain {domain}:")
        print("-" * 80)
        
        # For each contact, get their emails
        for contact in domain_contacts:
            properties = contact.get('properties', {})
            contact_id = contact.get('id')
            
            print(f"\n👤 {properties.get('firstname', '')} {properties.get('lastname', '')} ({properties.get('email', 'N/A')})")
            
            # Get all emails
            emails = hubspot.get_all_emails_for_contact(contact_id)
            
            # Filter for incoming emails
            incoming_emails = [
                email for email in emails 
                if email.get('direction') in ['INBOUND', 'INCOMING', 'INCOMING_EMAIL']
            ]
            
            if incoming_emails:
                print("\n📨 Email Responses:")
                for email in incoming_emails:
                    date = format_timestamp(email.get('timestamp', ''))
                    content = email.get('body_text', '') or email.get('text', '') or email.get('body', '')
                    summary = get_first_50_words(content)
                    
                    print(f"\n📅 {date}")
                    print(f"💬 {summary}")
            else:
                print("\n❌ No email responses found")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        if debug:
            import traceback
            print("\nStacktrace:")
            print(traceback.format_exc())

if __name__ == "__main__":
    TEST_EMAIL = "pbostwick@foxchapelgolfclub.com"
    print(f"🏃 Running company email test for: {TEST_EMAIL}")
    test_get_company_emails(TEST_EMAIL) 