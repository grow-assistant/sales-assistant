import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hubspot_service import HubspotService
from utils.enrich_hubspot_company_data import (
    get_facility_info,
    determine_facility_type,
    determine_seasonality,
    update_company_properties
)
from utils.logging_setup import logger


def test_company_enrichment():
    # Get API key from environment variable
    api_key = os.getenv('HUBSPOT_API_KEY')
    if not api_key:
        print("Please set HUBSPOT_API_KEY environment variable")
        return

    # Initialize HubSpot service
    hubspot = HubspotService(api_key)

    # Prompt for company ID
    company_id = input("Please enter the HubSpot company ID to enrich: ").strip()
    if not company_id:
        print("No company ID provided. Exiting...")
        return
    
    try:
        print(f"\nProcessing company ID: {company_id}")
        print("\n=== Initial Company Data ===")
        print("-" * 50)
        
        # Get initial company data
        company_data = hubspot.get_company_data(company_id)
        
        # Extract data from company_data
        name = company_data.get("name", "")
        company_short_name = company_data.get("company_short_name", "")
        city = company_data.get("city", "")
        state = company_data.get("state", "")
        club_type = company_data.get("club_type", "Unknown")
        facility_complexity = company_data.get("facility_complexity", "Unknown")
        has_pool = company_data.get("has_pool", "No")
        has_tennis_courts = company_data.get("has_tennis_courts", "No")
        number_of_holes = company_data.get("number_of_holes", 0)
        geographic_seasonality = company_data.get("geographic_seasonality", "Unknown")
        public_private_flag = company_data.get("public_private_flag", "Unknown")
        club_info = company_data.get("club_info", "")

        # Print initial data
        initial_data = {
            'name': name,
            'company_short_name': company_short_name,
            'city': city,
            'state': state,
            'club_type': club_type,
            'facility_complexity': facility_complexity,
            'has_pool': has_pool,
            'has_tennis_courts': has_tennis_courts,
            'number_of_holes': number_of_holes,
            'geographic_seasonality': geographic_seasonality,
            'public_private_flag': public_private_flag,
            'club_info': club_info
        }

        for key, value in initial_data.items():
            print(f"{key}: {value}")

        print("\n=== Enriching Data ===")
        print("-" * 50)

        # Get enriched facility data
        if name and state:
            club_info = determine_facility_type(name, state)
            season_data = determine_seasonality(state)
            
            # Keep existing values if new ones are Unknown
            has_pool = club_info.get("has_pool", "")
            if has_pool.lower() == "unknown":
                has_pool = has_pool if has_pool else "No"
                
            has_tennis = club_info.get("has_tennis_courts", "")
            if has_tennis.lower() == "unknown":
                has_tennis = has_tennis_courts if has_tennis_courts else "No"
            
            # Combine all updates with stricter value handling
            confirmed_updates = {
                "name": club_info.get("name", name),
                "company_short_name": club_info.get("company_short_name", company_short_name),
                "club_type": club_info.get("club_type", "Unknown"),
                "facility_complexity": "Standard" if club_info.get("facility_complexity") == "Single-Course" else club_info.get("facility_complexity", "Unknown"),
                "geographic_seasonality": season_data["geographic_seasonality"],
                "has_pool": "Yes" if str(has_pool).lower() == "yes" else "No",
                "has_tennis_courts": "Yes" if str(has_tennis).lower() == "yes" else "No",
                "number_of_holes": club_info.get("number_of_holes", 0),
                "public_private_flag": public_private_flag
            }
            
            # Only add season data if it's not empty or "1"
            if season_data.get("start_month") and season_data.get("start_month") != "1":
                confirmed_updates.update({
                    "start_month": season_data["start_month"],
                    "end_month": season_data["end_month"],
                    "peak_season_start_month": season_data["peak_season_start_month"],
                    "peak_season_end_month": season_data["peak_season_end_month"]
                })

            # Only add club_info if it exists and isn't empty
            new_club_info = club_info.get("club_info")
            if new_club_info and str(new_club_info).strip():
                # Only update if new info is longer or current is empty
                if not club_info or len(new_club_info) > len(club_info):
                    confirmed_updates["club_info"] = new_club_info

            # Print enriched data
            print("\nEnriched Data to be Updated:")
            for key, value in confirmed_updates.items():
                print(f"{key}: {value}")

            # Update HubSpot using HubspotService
            print("\n=== Updating HubSpot ===")
            print("-" * 50)
            success = hubspot.update_company_properties(company_id, confirmed_updates)
            
            if success:
                print("✓ Successfully updated HubSpot properties")
            else:
                print("✗ Failed to update HubSpot properties")
        else:
            print("Unable to enrich data - missing company name or location")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    test_company_enrichment() 