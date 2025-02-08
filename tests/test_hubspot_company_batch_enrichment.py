import sys
import os
import random
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.hubspot_service import HubspotService
from utils.enrich_hubspot_company_data import (
    determine_facility_type,
    determine_seasonality,
    _search_companies_with_filters
)
from utils.logging_setup import logger

# Constants for batch processing
TOTAL_COMPANIES = 2500
BATCH_SIZE = 100

def test_company_batch_enrichment():
    api_key = os.getenv('HUBSPOT_API_KEY')
    if not api_key:
        print("Please set HUBSPOT_API_KEY environment variable")
        return

    hubspot = HubspotService(api_key)

    try:
        print("\n=== Fetching Companies ===")
        print("-" * 10)

        # Get companies in batches
        processed_count = 0
        while processed_count < TOTAL_COMPANIES:
            # Calculate remaining companies to process
            remaining = TOTAL_COMPANIES - processed_count
            current_batch_size = min(BATCH_SIZE, remaining)
            
            print(f"\nFetching batch of {current_batch_size} companies (Processed: {processed_count}/{TOTAL_COMPANIES})")
            
            # Get batch of companies
            companies_batch = _search_companies_with_filters(hubspot, batch_size=current_batch_size)

            if not companies_batch:
                print("No more companies found.")
                break

            print(f"Retrieved {len(companies_batch)} companies in this batch")

            # Process each company in the batch
            for i, company in enumerate(companies_batch):
                company_id = company.get("id")
                if not company_id:
                    print("Skipping company with no ID")
                    continue

                overall_count = processed_count + i + 1
                print(f"\n=== Processing Company {overall_count}/{TOTAL_COMPANIES} (ID: {company_id}) ===")

                # --- Get Initial Data ---
                company_data = hubspot.get_company_data(company_id)
                name = company_data.get("name", "")
                state = company_data.get("state", "")
                club_info_initial = company_data.get("club_info", "")
                club_type_initial = company_data.get("club_type", "Unknown")
                print(f"  Initial club_type: {club_type_initial}")

                print("\nInitial Data:")
                print(f"  Name: {name}")
                print(f"  State: {state}")
                print(f"  Initial club_info: {club_info_initial}")

                # --- Enrich Data ---
                if name and state:
                    club_info_enriched = determine_facility_type(name, state)
                    season_data = determine_seasonality(state)

                    # --- Prepare Updates ---
                    confirmed_updates = {
                        "name": club_info_enriched.get("name", name),
                        "company_short_name": club_info_enriched.get("company_short_name", ""),
                        "club_type": club_info_enriched.get("club_type", "Unknown"),
                        "facility_complexity": "Standard" if club_info_enriched.get("facility_complexity") == "Single-Course" else club_info_enriched.get("facility_complexity", "Unknown"),
                        "geographic_seasonality": season_data.get("geographic_seasonality", "Unknown"),
                        "has_pool": "Yes" if str(club_info_enriched.get("has_pool", "")).lower() == "yes" else "No",
                        "has_tennis_courts": "Yes" if str(club_info_enriched.get("has_tennis_courts", "")).lower() == "yes" else "No",
                        "number_of_holes": club_info_enriched.get("number_of_holes", 0),
                        "start_month": season_data.get("start_month", ""),
                        "end_month": season_data.get("end_month", ""),
                        "peak_season_start_month": season_data.get("peak_season_start_month", ""),
                        "peak_season_end_month": season_data.get("peak_season_end_month", "")
                    }

                    # --- Handle club_info ---
                    new_club_info = club_info_enriched.get("club_info")
                    if new_club_info and str(new_club_info).strip():
                        if not club_info_initial or len(new_club_info) > len(club_info_initial):
                            confirmed_updates["club_info"] = new_club_info

                    # --- Update HubSpot ---
                    print("\nEnriched Data to be Updated:")
                    for key, value in confirmed_updates.items():
                        print(f"  {key}: {value}")

                    print("\nAll Retrieved Values:")
                    print(f"  Company ID: {company_id}")
                    for key, value in company_data.items():
                        print(f"  {key}: {value}")

                    success = hubspot.update_company_properties(company_id, confirmed_updates)

                    if success:
                        print("✓ Successfully updated HubSpot properties")
                    else:
                        print("✗ Failed to update HubSpot properties")

                    print("\nUpdated Company Data:")
                    updated_company_data = hubspot.get_company_data(company_id)
                    for key, value in updated_company_data.items():
                        print(f"  {key}: {value}")
                else:
                    print("Unable to enrich data - missing company name or location")

            # Update processed count after batch completion
            processed_count += len(companies_batch)
            print(f"\nCompleted batch. Total processed: {processed_count}/{TOTAL_COMPANIES}")

        print("\n=== Completed processing all companies ===")
        print(f"Total companies processed: {processed_count}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    test_company_batch_enrichment() 