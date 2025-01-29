"""
Purpose: Retrieves random contact samples from HubSpot for testing/verification.

Key functions:
- main(): Fetches and displays 3 random contacts from HubSpot with their basic info
         (email, name, company)

When run directly, pulls 3 random contacts from HubSpot and prints their details,
useful for spot-checking contact data quality and API functionality.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import HUBSPOT_API_KEY
from services.hubspot_service import HubspotService

def main():
    # Initialize HubSpot service
    hubspot = HubspotService(HUBSPOT_API_KEY)
    
    # Get 3 random contacts
    contacts = hubspot.get_random_contacts(count=3)
    
    # Print results
    print("\nRandom Contacts from HubSpot:")
    print("=" * 40)
    for contact in contacts:
        print(f"\nEmail: {contact['email']}")
        print(f"Name: {contact['first_name']} {contact['last_name']}")
        print(f"Company: {contact['company']}")
    print("\n")

if __name__ == "__main__":
    main() 