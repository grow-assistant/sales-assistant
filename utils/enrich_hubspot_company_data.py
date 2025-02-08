# tests/test_hubspot_company_type.py

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import os
import csv
import re

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.company_enrichment_service import CompanyEnrichmentService
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from utils.exceptions import HubSpotError
from utils.xai_integration import (
    xai_club_segmentation_search,
    get_club_summary
)
from utils.logging_setup import logger
from scripts.golf_outreach_strategy import get_best_outreach_window
from utils.web_fetch import fetch_website_html

# Update these constants after imports
###########################
# CONFIG / CONSTANTS
###########################
TEST_MODE = False  # Set to False for production
TEST_LIMIT = 3    # Number of companies to process in test mode
BATCH_SIZE = 10  # Companies per API request (increased from 25 to 100)
TEST_COMPANY_ID = ""  # Set this to a specific company ID to test just that company
TOTAL_COMPANIES = 200  # Maximum number of companies to process


# Define target states
TARGET_STATES = [
    "FL",  # Year-Round Golf
    "MN",  # Short Summer Season
    "WI",  # Short Summer Season
    "MI",  # Short Summer Season
    "ME",  # Short Summer Season
    "VT",  # Short Summer Season
]


def get_facility_info(company_id: str) -> tuple[
    str, str, str, str, str, str, str, str, str, str, str, int, str, str, str, str
]:
    """
    Fetches the company's properties from HubSpot.
    Returns a tuple of company properties including club_info and company_short_name.
    """
    try:
        company_enricher = CompanyEnrichmentService(api_key=HUBSPOT_API_KEY)
        company_data = company_enricher._get_facility_info(company_id)

        # Check domain for competitor software with shorter timeouts
        domain = company_data.get('properties', {}).get('domain', '')
        competitor = company_data.get('properties', {}).get('competitor', 'Unknown')
        
        if domain:
            # Try different URL variations with shorter timeout
            urls_to_try = []
            base_url = domain.strip().lower()
            if not base_url.startswith('http'):
                urls_to_try.extend([f"https://{base_url}", f"http://{base_url}"])
            else:
                urls_to_try.append(base_url)
            
            # Add www. version if not present
            urls_to_try.extend([url.replace('://', '://www.') for url in urls_to_try])
            
            for url in urls_to_try:
                try:
                    # Reduced timeout from 10 seconds to 5 seconds
                    html_content = fetch_website_html(url, timeout=5)
                    if html_content:
                        html_lower = html_content.lower()
                        # Check for Club Essentials mentions first
                        clubessential_mentions = [
                            "copyright clubessential",
                            "clubessential, llc",
                            "www.clubessential.com",
                            "http://www.clubessential.com",
                            "clubessential"
                        ]
                        for mention in clubessential_mentions:
                            if mention in html_lower:
                                competitor = "Club Essentials"
                                logger.debug(f"Found Club Essentials on {url}")
                                break
                                
                        # Check for Jonas mentions if not Club Essentials
                        if competitor == "Unknown":
                            jonas_mentions = ["jonas club software", "jonas software", "jonasclub"]
                            for mention in jonas_mentions:
                                if mention in html_lower:
                                    competitor = "Jonas"
                                    logger.debug(f"Found Jonas on {url}")
                                    break
                        
                        # If we found a competitor, no need to check other URLs
                        if competitor != "Unknown":
                            break
                            
                except Exception as e:
                    logger.debug(f"Error checking {url}: {str(e)}")
                    continue  # Try next URL if this one fails

        return (
            company_data.get("name", ""),
            company_data.get("company_short_name", ""),
            company_data.get("city", ""),
            company_data.get("state", ""),
            company_data.get("annual_revenue", ""),
            company_data.get("create_date", ""),
            company_data.get("last_modified", ""),
            company_data.get("object_id", ""),
            company_data.get("club_type", "Unknown"),
            company_data.get("facility_complexity", "Unknown"),
            company_data.get("has_pool", "Unknown"),
            company_data.get("has_tennis_courts", "Unknown"),
            company_data.get("number_of_holes", 0),
            company_data.get("geographic_seasonality", "Unknown"),
            company_data.get("public_private_flag", "Unknown"),
            company_data.get("club_info", ""),
            competitor
        )

    except Exception as e:
        print(f"Error fetching company data: {e}")
        return ("", "", "", "", "", "", "", "", "", "No", "No", 0, "", "", "", "")


def determine_facility_type(company_name: str, location: str) -> dict:
    """
    Uses xAI to determine the facility type and official name based on company info.
    """
    if not company_name or not location:
        return {}

    # Get segmentation data
    segmentation_info = xai_club_segmentation_search(company_name, location)
    
    # Get summary for additional context
    club_summary = get_club_summary(company_name, location)

    # Map facility complexity correctly
    facility_complexity = segmentation_info.get("facility_complexity", "Unknown")
    if facility_complexity.lower() in ["single-course", "standard"]:
        facility_complexity = "Standard"
    elif facility_complexity.lower() == "multi-course":
        facility_complexity = "Multi-Course"
    elif facility_complexity.lower() == "resort":
        facility_complexity = "Resort"

    # Map geographic seasonality correctly
    seasonality = segmentation_info.get("geographic_seasonality", "Unknown")
    if seasonality.lower() in ["seasonal", "standard season"]:
        seasonality = "Standard Season"
    elif seasonality.lower() == "year-round":
        seasonality = "Year-Round Golf"
    elif seasonality.lower() == "short summer season":
        seasonality = "Short Summer Season"

    # Map club type correctly
    club_type = segmentation_info.get("club_type", "Unknown")
    if "country club" in club_type.lower():
        club_type = "Country Club"
    elif club_type.lower() == "private course" or club_type.lower() == "private":
        club_type = "Private"

    # Set public/private flag based on club type
    public_private_flag = "Unknown"
    if club_type in ["Private", "Country Club"]:
        public_private_flag = "Private"
    elif any(t in club_type for t in ["Public", "Municipal"]):
        public_private_flag = "Public"

    # Keep original values for has_pool and has_tennis_courts
    has_pool = segmentation_info.get("has_pool", "Unknown")
    has_tennis = segmentation_info.get("has_tennis_courts", "Unknown")

    # Don't convert Unknown to No
    if has_pool.lower() == "unknown":
        has_pool = "Unknown"
    if has_tennis.lower() == "unknown":
        has_tennis = "Unknown"

    full_info = {
        "name": segmentation_info.get("name", ""),
        "company_short_name": segmentation_info.get("company_short_name", ""),
        "club_type": club_type,
        "facility_complexity": facility_complexity,
        "geographic_seasonality": seasonality,
        "has_pool": has_pool,
        "has_tennis_courts": has_tennis,
        "number_of_holes": segmentation_info.get("number_of_holes", 0),
        "public_private_flag": public_private_flag,
        "club_info": club_summary
    }

    return full_info


def update_company_properties(company_id: str, club_info: dict, confirmed_updates: dict) -> bool:
    """
    Updates the company's properties in HubSpot based on club segmentation info.
    """
    try:
        company_enricher = CompanyEnrichmentService(api_key=HUBSPOT_API_KEY)
        
        # Check club_info for pool mentions before processing
        club_info_text = str(confirmed_updates.get('club_info', '')).lower()
        if 'pool' in club_info_text and confirmed_updates.get('has_pool') in ['Unknown', 'No']:
            logger.debug(f"Found pool mention in club_info, updating has_pool to Yes")
            confirmed_updates['has_pool'] = 'Yes'

        # Debug input values
        logger.debug("Input values for update:")
        for key, value in confirmed_updates.items():
            logger.debug(f"Field: {key}, Value: {value}, Type: {type(value)}")

        # Use the CompanyEnrichmentService's update method
        success = company_enricher.update_company_properties(company_id, confirmed_updates)
        return success

    except Exception as e:
        logger.error(f"Error updating company properties: {str(e)}")
        return False


def determine_seasonality(state: str, city: str = None) -> dict:
    """
    Determine the seasonality based on city/state or state lookup.
    Prioritizes city/state data over state-only data.
    """
    logger.debug(f"Determining seasonality for city: {city}, state: {state}")
    
    if not state:
        logger.warning("No state provided")
        return _get_default_seasonality()

    season_data = None
    
    # First try city/state lookup if city is provided
    if city:
        season_data = _lookup_city_state_data(city, state)
        if season_data:
            logger.debug(f"Found city/state data for {city}, {state}: {season_data}")
            return season_data

    # If no city data found, try state-level lookup
    season_data = _lookup_state_data(state)
    if season_data:
        logger.debug(f"Found state-level data for {state}: {season_data}")
        return season_data

    # Fallback to default if no data found
    logger.warning(f"No seasonality data found for {state}, using default")
    return _get_default_seasonality()

def _lookup_city_state_data(city: str, state: str) -> dict:
    """Look up seasonality data by city and state."""
    logger.debug(f"Looking up city/state data for {city}, {state}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    city_state_path = os.path.join(project_root, 'docs', 'golf_seasons', 'golf_seasons_by_city_st.csv')
    
    try:
        with open(city_state_path, 'r', encoding='utf-8-sig') as file:  # Changed to utf-8-sig
            reader = csv.DictReader(file)
            for row in reader:
                # Clean any potential BOM from column names
                cleaned_row = {k.strip('\ufeff'): v for k, v in row.items()}
                if (cleaned_row['State'].upper() == state.upper() and 
                    cleaned_row['City'].lower() == city.lower()):
                    logger.debug(f"Found city/state match: {cleaned_row}")
                    return _parse_season_data(cleaned_row)
    except Exception as e:
        logger.error(f"Error reading city/state data: {str(e)}")
        logger.debug(f"CSV Path: {city_state_path}")
    return None

def _lookup_state_data(state: str) -> dict:
    """Look up seasonality data by state."""
    logger.debug(f"Looking up state data for {state}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    state_path = os.path.join(project_root, 'docs', 'golf_seasons', 'golf_seasons_by_st.csv')
    
    try:
        with open(state_path, 'r', encoding='utf-8-sig') as file:  # Changed to utf-8-sig
            reader = csv.DictReader(file)
            for row in reader:
                # Clean any potential BOM from column names
                cleaned_row = {k.strip('\ufeff'): v for k, v in row.items()}
                if cleaned_row['State'].upper() == state.upper():
                    logger.debug(f"Found state match: {cleaned_row}")
                    return _parse_season_data(cleaned_row)
    except Exception as e:
        logger.error(f"Error reading state data: {str(e)}")
        logger.debug(f"CSV Path: {state_path}")
    return None

def _parse_season_data(row: dict) -> dict:
    """Parse season data from CSV row."""
    is_year_round = str(row.get('Year-Round?', '')).lower() == 'yes'
    
    # Convert month names to numbers if needed
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    def parse_month(month_str):
        if not month_str or month_str == 'N/A':
            return ''
        # Try as number first
        try:
            return str(int(month_str))
        except ValueError:
            # Try as month name
            return str(month_map.get(month_str.lower(), ''))

    start_month = parse_month(row.get('Start Month'))
    end_month = parse_month(row.get('End Month'))
    peak_start = parse_month(row.get('Peak Season Start'))
    peak_end = parse_month(row.get('Peak Season End'))

    # Determine seasonality type
    if is_year_round:
        seasonality = "Year-Round Golf"
    elif start_month == '3' or start_month == '4':  # March or April start
        seasonality = "Peak Summer Season"
    else:
        seasonality = "Short Summer Season"

    return {
        "geographic_seasonality": seasonality,
        "start_month": start_month,
        "end_month": end_month,
        "peak_season_start_month": peak_start,
        "peak_season_end_month": peak_end
    }

def _get_default_seasonality() -> dict:
    """Return default seasonality values."""
    return {
        "geographic_seasonality": "Short Summer Season",
        "start_month": "5",  # May
        "end_month": "9",    # September
        "peak_season_start_month": "6",  # June
        "peak_season_end_month": "8"     # August
    }


def process_company(company_id: str) -> None:
    """Process a single company for enrichment."""
    try:
        print(f"\n=== Processing Company ID: {company_id} ===")
        company_enricher = CompanyEnrichmentService(api_key=HUBSPOT_API_KEY)

        # Use the enrichment service to process the company
        enrichment_result = company_enricher.enrich_company(company_id)
        if enrichment_result.get("success", False):
            print("✓ Successfully updated HubSpot properties")
            print(f"Updated properties: {enrichment_result.get('data', {})}")
        else:
            print("✗ Failed to update HubSpot properties")
            if "error" in enrichment_result:
                print(f"Error: {enrichment_result['error']}")

    except Exception as e:
        logger.error(f"Error processing company {company_id}: {str(e)}")
        logger.exception("Full error details:")


def _search_companies_with_filters(hubspot: HubspotService, batch_size=100) -> List[Dict[str, Any]]:
    """
    Search for companies in HubSpot that need club type enrichment.
    Processes companies in batches with state filtering.
    """
    logger.info(f"Searching for companies (Test Mode: {TEST_MODE})")
    company_enricher = CompanyEnrichmentService(api_key=HUBSPOT_API_KEY)
    url = f"{company_enricher.hubspot.base_url}/crm/v3/objects/companies/search"
    after = None
    all_results = []
    
    while True and (not TEST_MODE or len(all_results) < TEST_LIMIT):
        # Build request payload
        payload = {
            "limit": min(batch_size, TEST_LIMIT) if TEST_MODE else batch_size,
            "properties": [
                "name", "company_short_name", "city", "state",
                "club_type", "annualrevenue", "facility_complexity",
                "geographic_seasonality", "has_pool", "has_tennis_courts",
                "number_of_holes", "public_private_flag", "start_month",
                "end_month", "peak_season_start_month", "peak_season_end_month",
                "notes_last_contacted", "num_contacted_notes",
                "num_associated_contacts"
            ],
            "filterGroups": [
                {
                    "filters": [
                        # {
                        #     "propertyName": "state", 
                        #     "operator": "IN",
                        #     "values": TARGET_STATES
                        # },
                        {
                            "propertyName": "club_type",
                            "operator": "NOT_HAS_PROPERTY",
                            "value": None
                        }
                    ]
                }
            ]
        }
        
        if after:
            payload["after"] = after

        try:
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break

            results = response.get("results", [])
            all_results.extend(results)
            
            logger.info(f"Retrieved {len(all_results)} total companies")

            # Handle pagination
            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link

            # Check if we've hit the test limit or total limit
            if TEST_MODE and len(all_results) >= TEST_LIMIT:
                logger.info(f"Test mode: Reached limit of {TEST_LIMIT} companies")
                break
            if len(all_results) >= TOTAL_COMPANIES:
                logger.info(f"Reached total limit of {TOTAL_COMPANIES} companies")
                break

        except Exception as e:
            logger.error(f"Error fetching companies from HubSpot: {str(e)}")
            break

    # Ensure we don't exceed limits
    if TEST_MODE:
        all_results = all_results[:TEST_LIMIT]
    else:
        all_results = all_results[:TOTAL_COMPANIES]
    
    logger.info(f"Returning {len(all_results)} companies total")
    return all_results


def main():
    """Main function to process companies needing enrichment."""
    try:
        company_enricher = CompanyEnrichmentService(api_key=HUBSPOT_API_KEY)
        
        # Check if we're processing a single test company
        if TEST_COMPANY_ID:
            print(f"\n=== Processing Single Test Company: {TEST_COMPANY_ID} ===\n")
            process_company(TEST_COMPANY_ID)
            print("\n=== Completed processing test company ===")
            return
            
        # Regular batch processing
        processed_count = 0
        while processed_count < TOTAL_COMPANIES:
            # Get next batch of companies
            companies_batch = _search_companies_with_filters(company_enricher.hubspot)
            
            if not companies_batch:
                print("No more companies found needing enrichment")
                break
                
            print(f"\n=== Processing batch of {len(companies_batch)} companies ===\n")
            
            for i, company in enumerate(companies_batch, 1):
                company_id = company.get("id")
                if not company_id:
                    continue
                    
                overall_count = processed_count + i
                print(f"\nProcessing company {overall_count} of {TOTAL_COMPANIES}")
                process_company(company_id)
            
            processed_count += len(companies_batch)
            print(f"\nCompleted batch. Total processed: {processed_count}/{TOTAL_COMPANIES}")
        
        print("\n=== Completed processing all companies ===")
        print(f"Total companies processed: {processed_count}")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()