import json
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now we can import project modules
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.logging_setup import logger

class HubspotFieldFinder:
    def __init__(self):
        # Initialize HubspotService with API key from settings
        self.hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        
    def get_all_properties(self, object_type: str) -> List[Dict]:
        """Get all properties for a given object type (company or contact)."""
        try:
            url = f"{self.hubspot.base_url}/crm/v3/properties/{object_type}"
            response = self.hubspot._make_hubspot_get(url)
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Error getting {object_type} properties: {str(e)}")
            return []

    def search_property(self, search_term: str, object_type: str) -> List[Dict]:
        """Search for properties containing the search term."""
        properties = self.get_all_properties(object_type)
        matches = []
        
        search_term = search_term.lower()
        for prop in properties:
            if (search_term in prop.get("label", "").lower() or
                search_term in prop.get("name", "").lower() or
                search_term in prop.get("description", "").lower()):
                
                matches.append({
                    "internal_name": prop.get("name"),
                    "label": prop.get("label"),
                    "type": prop.get("type"),
                    "description": prop.get("description"),
                    "group_name": prop.get("groupName"),
                    "options": prop.get("options", [])
                })
        
        return matches

def print_matches(matches: List[Dict], search_term: str, object_type: str):
    """Pretty print the matching properties."""
    if not matches:
        print(f"\nNo matches found for '{search_term}' in {object_type} properties.")
        return
        
    print(f"\nFound {len(matches)} matches for '{search_term}' in {object_type} properties:")
    print("=" * 80)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. Internal Name: {match['internal_name']}")
        print(f"   Label: {match['label']}")
        print(f"   Type: {match['type']}")
        print(f"   Group: {match['group_name']}")
        if match['description']:
            print(f"   Description: {match['description']}")
        if match['options']:
            print("   Options:")
            for opt in match['options']:
                print(f"     - {opt.get('label')} ({opt.get('value')})")
        print("-" * 80)

def save_results(matches: List[Dict], object_type: str):
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hubspot_fields_{object_type}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(matches, f, indent=2)
    print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Search HubSpot fields by name or description')
    parser.add_argument('search_term', nargs='?', help='Term to search for in field names/descriptions')
    parser.add_argument('--type', '-t', choices=['companies', 'contacts'], default='companies',
                      help='Object type to search (companies or contacts)')
    parser.add_argument('--save', '-s', action='store_true',
                      help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Only show internal names (useful for scripting)')
    
    args = parser.parse_args()
    
    # If no search term provided, enter interactive mode
    if not args.search_term:
        print("\nHubSpot Field Finder")
        print("1. Search Company Fields")
        print("2. Search Contact Fields")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "3":
            return
            
        if choice not in ["1", "2"]:
            print("Invalid choice. Please try again.")
            return
            
        args.type = "companies" if choice == "1" else "contacts"
        args.search_term = input(f"\nEnter search term for {args.type} fields: ")
    
    finder = HubspotFieldFinder()
    matches = finder.search_property(args.search_term, args.type)
    
    if args.quiet:
        # Only print internal names, one per line
        for match in matches:
            print(match['internal_name'])
    else:
        print_matches(matches, args.search_term, args.type)
        
    if args.save:
        save_results(matches, args.type)

if __name__ == "__main__":
    main() 