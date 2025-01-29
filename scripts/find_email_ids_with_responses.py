"""
Purpose: Analyzes email interactions and responses in HubSpot.

Key functions:
- get_random_lead_id(): Gets random test lead ID from database
- get_lead_id_for_email(): Looks up lead ID by email address  
- get_all_leads(): Retrieves all lead records
- test_lead_info(): Tests HubSpot data retrieval for random lead
- test_specific_lead(): Tests data retrieval for specific test email
- test_recent_replies(): Finds leads with replies in last 30 days

When run directly, checks all leads for recent replies and generates 
interaction report using HubSpot and data gathering services.
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.leads_service import LeadsService
from services.data_gatherer_service import DataGathererService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def get_random_lead_id():
    """Get a random lead_id from the emails table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Modified query to ensure we get a valid HubSpot contact ID
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY NEWID()
            """)
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database: {lead_id}")
                return lead_id
            logger.warning("No valid lead_id found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting random lead_id: {str(e)}")
        return None

def get_lead_id_for_email(email):
    """Get lead_id for a specific email address."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 1 e.lead_id 
                FROM emails e
                WHERE e.email_address = ?
                  AND e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
            """, (email,))
            result = cursor.fetchone()
            if result and result[0]:
                lead_id = str(result[0])
                logger.debug(f"Found lead_id in database for {email}: {lead_id}")
                return lead_id
            logger.warning(f"No valid lead_id found for email: {email}")
            return None
    except Exception as e:
        logger.error(f"Error getting lead_id for email {email}: {str(e)}")
        return None

def get_all_leads():
    """Get all lead IDs and emails from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT e.lead_id, e.email_address, e.name, e.company_short_name
                FROM emails e
                WHERE e.lead_id IS NOT NULL 
                  AND e.lead_id != ''
                  AND LEN(e.lead_id) > 0
                ORDER BY e.company_short_name, e.email_address
            """)
            results = cursor.fetchall()
            if results:
                leads = [
                    {
                        'lead_id': str(row[0]),
                        'email': row[1],
                        'name': row[2],
                        'company': row[3]
                    }
                    for row in results
                ]
                logger.debug(f"Found {len(leads)} leads in database")
                return leads
            logger.warning("No valid leads found in database")
            return []
    except Exception as e:
        logger.error(f"Error getting leads: {str(e)}")
        return []

def test_lead_info():
    """Test function to pull HubSpot data for a random lead ID."""
    try:
        # Initialize services in correct order
        data_gatherer = DataGathererService()  # Initialize without parameters
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service  # Set the service after initialization
        leads_service = LeadsService(data_gatherer)
        
        # Get random contact ID from database
        contact_id = get_random_lead_id()
        if not contact_id:
            print("No lead IDs found in database")
            return
            
        print(f"\nFetching info for contact ID: {contact_id}")
        
        # Verify contact exists in HubSpot before proceeding
        try:
            # Test if we can get contact properties
            contact_props = hubspot_service.get_contact_properties(contact_id)
            if not contact_props:
                print(f"Contact ID {contact_id} not found in HubSpot")
                return
        except Exception as e:
            print(f"Error verifying contact in HubSpot: {str(e)}")
            return
            
        # Get lead summary using LeadsService
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error: {lead_info['error']}")
            return
            
        # Print results
        print("\nLead Information:")
        print("=" * 50)
        print(f"Last Reply Date: {lead_info['last_reply_date']}")
        print(f"Lifecycle Stage: {lead_info['lifecycle_stage']}")
        print(f"Company Name: {lead_info['company_name']}")
        print(f"Company Short Name: {lead_info['company_short_name']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_lead_info: {str(e)}", exc_info=True)

def test_specific_lead():
    """Test function to pull HubSpot data for pcrow@chicagohighlands.com."""
    try:
        email = "pcrow@chicagohighlands.com"
        print(f"\nTesting lead info for: {email}")
        
        # Initialize services
        data_gatherer = DataGathererService()
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service
        leads_service = LeadsService(data_gatherer)
        
        # Get contact ID from database
        contact_id = get_lead_id_for_email(email)
        if not contact_id:
            print(f"No lead ID found for email: {email}")
            return
            
        print(f"Found contact ID: {contact_id}")
        
        # Test direct HubSpot API calls
        print("\nTesting direct HubSpot API calls:")
        print("=" * 50)
        
        # Get contact properties
        contact_props = hubspot_service.get_contact_properties(contact_id)
        if contact_props:
            print("Contact properties found:")
            for key, value in contact_props.items():
                print(f"{key}: {value}")
        else:
            print("No contact properties found")
            
        # Get latest emails
        print("\nChecking latest emails:")
        print("=" * 50)
        try:
            latest_emails = hubspot_service.get_latest_emails_for_contact(contact_id)
            if latest_emails:
                print("Latest email interactions:")
                for email in latest_emails:
                    print(f"Type: {email.get('type')}")
                    print(f"Date: {email.get('created_at')}")
                    print(f"Subject: {email.get('subject', 'No subject')}")
                    print("-" * 30)
            else:
                print("No email interactions found")
        except Exception as e:
            print(f"Error getting latest emails: {str(e)}")
        
        # Get lead summary
        print("\nTesting LeadsService summary:")
        print("=" * 50)
        lead_info = leads_service.get_lead_summary(contact_id)
        
        if lead_info.get('error'):
            print(f"Error: {lead_info['error']}")
            return
            
        print("Lead Summary Information:")
        print(f"Last Reply Date: {lead_info.get('last_reply_date', 'None')}")
        print(f"Lifecycle Stage: {lead_info.get('lifecycle_stage', 'None')}")
        print(f"Company Name: {lead_info.get('company_name', 'None')}")
        print(f"Company Short Name: {lead_info.get('company_short_name', 'None')}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_specific_lead: {str(e)}", exc_info=True)

def test_recent_replies():
    """Test function to find all leads who replied in the last month."""
    try:
        print("\nChecking for recent replies from all leads...")
        
        # Initialize services
        data_gatherer = DataGathererService()
        hubspot_service = HubspotService(HUBSPOT_API_KEY)
        data_gatherer.hubspot_service = hubspot_service
        
        # Get all leads
        leads = get_all_leads()
        if not leads:
            print("No leads found in database")
            return
            
        print(f"Found {len(leads)} total leads to check")
        print("=" * 50)
        
        # Track replies
        recent_replies = []
        one_month_ago = datetime.now() - timedelta(days=30)
        
        # Check each lead
        for lead in leads:
            try:
                # Get contact properties
                contact_props = hubspot_service.get_contact_properties(lead['lead_id'])
                if not contact_props:
                    continue
                
                last_reply = contact_props.get('hs_sales_email_last_replied')
                if last_reply:
                    try:
                        reply_date = datetime.fromtimestamp(int(last_reply)/1000)
                        if reply_date > one_month_ago:
                            lead['reply_date'] = reply_date
                            lead['properties'] = contact_props
                            recent_replies.append(lead)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing reply date for {lead['email']}: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Error checking lead {lead['email']}: {str(e)}")
                continue
        
        # Print results
        print(f"\nFound {len(recent_replies)} leads with recent replies:")
        print("=" * 50)
        
        if recent_replies:
            # Sort by reply date, most recent first
            recent_replies.sort(key=lambda x: x['reply_date'], reverse=True)
            
            for lead in recent_replies:
                print(f"\nName: {lead['name']}")
                print(f"Email: {lead['email']}")
                print(f"Company: {lead['company']}")
                print(f"Reply Date: {lead['reply_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Try to get the actual email content
                try:
                    latest_emails = hubspot_service.get_latest_emails_for_contact(lead['lead_id'])
                    if latest_emails:
                        print("Latest Email Details:")
                        for email in latest_emails:
                            if email.get('type') == 'INCOMING':  # Only show replies
                                print(f"Subject: {email.get('subject', 'No subject')}")
                                print(f"Date: {email.get('created_at')}")
                                print(f"Preview: {email.get('body', '')[:200]}...")
                except Exception as e:
                    print(f"Could not fetch email details: {str(e)}")
                
                print("-" * 50)
        else:
            print("No leads found with replies in the last month")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Error in test_recent_replies: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Enable debug logging
    logger.setLevel("DEBUG")
    test_recent_replies()


