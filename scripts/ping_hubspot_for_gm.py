# File: scripts/ping_hubspot_for_gm.py

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random

# Imports from your codebase
from services.hubspot_service import HubspotService
from config.settings import HUBSPOT_API_KEY
from scripts.golf_outreach_strategy import (
    get_best_month, 
    get_best_outreach_window,
    adjust_send_time
)
from services.data_gatherer_service import DataGathererService  # optional if you want to reuse
from utils.logging_setup import logger

# Initialize data gatherer service
data_gatherer = DataGathererService()  # Add this line

###########################
# CONFIG / CONSTANTS
###########################
BATCH_SIZE = 25  # Keep original batch size
DEFAULT_GEOGRAPHY = "Year-Round Golf"  # Default geography if none specified
MIN_REVENUE = 1000000  # Minimum annual revenue filter
EXCLUDED_STATES = []    # List of states to exclude from search
TEST_MODE = True  # Add test mode flag
TEST_LIMIT = 10  # Number of companies to process in test mode

###########################
# SCRIPT START
###########################

def calculate_send_date(geography, profile_type, state, preferred_days, preferred_time):
    """
    Calculate optimal send date based on geography and profile.
    
    Args:
        geography: Geographic region of the club
        profile_type: Type of contact profile (e.g. General Manager)
        state: Club's state location
        preferred_days: List of preferred weekdays for sending
        preferred_time: Dict with start/end hours for sending
        
    Returns:
        datetime: Optimal send date and time
    """
    # Start with tomorrow
    base_date = datetime.now() + timedelta(days=1)
    
    # Find next preferred day
    while base_date.weekday() not in preferred_days:
        base_date += timedelta(days=1)
        
    # Set time within preferred window
    send_hour = preferred_time["start"]
    if random.random() < 0.5:  # 50% chance to use later hour
        send_hour += 1
        
    return base_date.replace(
        hour=send_hour,
        minute=random.randint(0, 59),
        second=0,
        microsecond=0
    )

def _calculate_lead_score(
    revenue: float,
    club_type: str,
    geography: str,
    current_month: int,
    best_months: List[int],
    season_data: Dict[str, Any]
) -> float:
    """
    Calculate a score for lead prioritization based on multiple factors.
    
    Args:
        revenue: Annual revenue
        club_type: Type of club
        geography: Geographic region
        current_month: Current month number
        best_months: List of best months for outreach
        season_data: Dictionary containing peak season information
        
    Returns:
        float: Score from 0-100
    """
    score = 0.0
    
    # Revenue scoring (up to 40 points)
    if revenue >= 5000000:
        score += 40
    elif revenue >= 2000000:
        score += 30
    elif revenue >= 1000000:
        score += 20

    # Club type scoring (up to 30 points)
    club_type_scores = {
        "Private": 30,
        "Semi-Private": 25,
        "Resort": 20,
        "Public": 15
    }
    score += club_type_scores.get(club_type, 0)

    # Timing/Season scoring (up to 30 points)
    if current_month in best_months:
        score += 30
    elif abs(current_month - min(best_months)) <= 1:
        score += 15

    # Geography bonus (up to 10 points)
    if geography == "Year-Round Golf":
        score += 10
    elif geography in ["Peak Summer Season", "Peak Winter Season"]:
        score += 5

    return score

def main():
    """
    Enhanced version with seasonal intelligence and lead scoring.
    """
    try:
        logger.info("==== Starting enhanced ping_hubspot_for_gm workflow ====")
        
        # Initialize services
        hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
        data_gatherer = DataGathererService()

        # Get current timing info
        current_month = datetime.now().month
        current_date = datetime.now()
        
        # Fetch companies with geographic data
        all_companies = _search_companies_with_filters(
            hubspot,
            batch_size=BATCH_SIZE,
            min_revenue=MIN_REVENUE,
            excluded_states=EXCLUDED_STATES
        )
        
        logger.info(f"Retrieved {len(all_companies)} companies matching initial criteria")

        # Store scored and prioritized leads
        scored_leads = []

        for company in all_companies:
            props = company.get("properties", {})
            geography_data = company.get("geography_data", {})
            
            # Get seasonal data
            season_data = {
                'peak_season_start': props.get('peak_season_start'),
                'peak_season_end': props.get('peak_season_end')
            }
            
            # Get outreach window
            outreach_window = get_best_outreach_window(
                persona="General Manager",
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                club_type=geography_data.get('club_type'),
                season_data=season_data
            )
            
            # Calculate lead score
            lead_score = _calculate_lead_score(
                revenue=float(props.get('annualrevenue', 0) or 0),
                club_type=geography_data.get('club_type', ''),
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                current_month=current_month,
                best_months=outreach_window["Best Month"],
                season_data=season_data
            )
            
            # Skip low-scoring leads (optional)
            if lead_score < 20:  # Adjust threshold as needed
                logger.debug(f"Skipping {props.get('name')}: Low score ({lead_score})")
                continue
            
            # Calculate optimal send time
            send_date = calculate_send_date(
                geography=geography_data.get('geography', DEFAULT_GEOGRAPHY),
                profile_type="General Manager",
                state=props.get('state', ''),
                preferred_days=outreach_window["Best Day"],
                preferred_time=outreach_window["Best Time"]
            )
            
            adjusted_send_time = adjust_send_time(send_date, props.get('state'))
            
            # Process contacts
            company_id = company.get("id")
            if not company_id:
                continue
                
            associated_contacts = _get_contacts_for_company(hubspot, company_id)
            if not associated_contacts:
                continue

            for contact in associated_contacts:
                c_props = contact.get("properties", {})
                jobtitle = c_props.get("jobtitle", "")
                
                if not jobtitle or not is_general_manager_jobtitle(jobtitle):
                    continue
                    
                email = c_props.get("email", "missing@noemail.com")
                first_name = c_props.get("firstname", "")
                last_name = c_props.get("lastname", "")
                
                scored_leads.append({
                    "score": lead_score,
                    "email": email,
                    "name": f"{first_name} {last_name}".strip(),
                    "company": props.get("name", ""),
                    "jobtitle": jobtitle,
                    "geography": geography_data.get('geography', DEFAULT_GEOGRAPHY),
                    "best_months": outreach_window["Best Month"],
                    "optimal_send_time": adjusted_send_time,
                    "club_type": geography_data.get('club_type', ''),
                    "peak_season": season_data,
                    "revenue": props.get('annualrevenue', 'N/A')
                })

        # Sort leads by score (highest first) and then by send time
        scored_leads.sort(key=lambda x: (-x["score"], x["optimal_send_time"]))
        
        logger.info(f"Found {len(scored_leads)} scored and prioritized GM leads")
        
        # Print results with scores
        for lead in scored_leads:
            print(
                f"Score: {lead['score']:.1f} | "
                f"Send Time: {lead['optimal_send_time'].strftime('%Y-%m-%d %H:%M')} | "
                f"{lead['name']} | "
                f"{lead['company']} | "
                f"Revenue: ${float(lead['revenue'] or 0):,.0f} | "
                f"Type: {lead['club_type']} | "
                f"Geography: {lead['geography']}"
            )

    except Exception as e:
        logger.exception(f"Error in enhanced ping_hubspot_for_gm: {str(e)}")


###########################
# HELPER FUNCTIONS
###########################

def is_general_manager_jobtitle(title: str) -> bool:
    """
    Returns True if the jobtitle indicates 'General Manager'.
    
    Args:
        title: Job title string to check
        
    Returns:
        bool: True if title contains 'general manager'
    """
    title_lower = title.lower()
    # simple approach
    if "general manager" in title_lower:
        return True
    return False


def _search_companies_in_batches(hubspot: HubspotService, batch_size=25, max_pages=1) -> List[Dict[str, Any]]:
    """
    Searches for companies in HubSpot using the CRM API with pagination.
    
    Args:
        hubspot: HubspotService instance
        batch_size: Number of records per request
        max_pages: Maximum number of pages to fetch
        
    Returns:
        List of company records
    """
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    after = None
    all_results = []
    pages_fetched = 0

    while pages_fetched < max_pages:
        # Build request payload
        payload = {
            "limit": batch_size,
            "properties": ["name", "city", "state", "annualrevenue", "company_type"],
            # you can add filters if you want to only get certain companies
            # "filterGroups": [...],
        }
        if after:
            payload["after"] = after

        try:
            # Make API request
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break

            # Process results
            results = response.get("results", [])
            all_results.extend(results)

            # Handle pagination
            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            else:
                after = next_link

            pages_fetched += 1
        except Exception as e:
            logger.error(f"Error fetching companies from HubSpot page={pages_fetched}: {str(e)}")
            break

    return all_results


def _get_contacts_for_company(hubspot: HubspotService, company_id: str) -> List[Dict[str, Any]]:
    """
    Find all associated contacts for a company.
    
    Args:
        hubspot: HubspotService instance
        company_id: ID of company to get contacts for
        
    Returns:
        List of contact records
    """
    # HubSpot API: GET /crm/v3/objects/companies/{companyId}/associations/contacts
    url = f"{hubspot.base_url}/crm/v3/objects/companies/{company_id}/associations/contacts"
    
    try:
        # Get contact associations
        response = hubspot._make_hubspot_get(url)
        if not response:
            return []
        contact_associations = response.get("results", [])
        # Each association looks like: {"id": <contactId>, "type": "company_to_contact"}
        
        if not contact_associations:
            return []

        # Collect the contact IDs
        contact_ids = [assoc["id"] for assoc in contact_associations if assoc.get("id")]

        # Bulk fetch each contact's properties
        contact_records = []
        for cid in contact_ids:
            # Reuse hubspot.get_contact_properties (which returns minimal)
            # or do a direct GET / search for that contact object.
            contact_data = _get_contact_by_id(hubspot, cid)
            if contact_data:
                contact_records.append(contact_data)
        
        return contact_records

    except Exception as e:
        logger.error(f"Error fetching contacts for company_id={company_id}: {str(e)}")
        return []


def _get_contact_by_id(hubspot: HubspotService, contact_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single contact by ID with all relevant properties.
    
    Args:
        hubspot: HubspotService instance
        contact_id: ID of contact to retrieve
        
    Returns:
        Contact record dict or None if error
    """
    url = f"{hubspot.base_url}/crm/v3/objects/contacts/{contact_id}"
    query_params = {
        "properties": ["email", "firstname", "lastname", "jobtitle"], 
        "archived": "false"
    }
    try:
        response = hubspot._make_hubspot_get(url, params=query_params)
        return response
    except Exception as e:
        logger.error(f"Error fetching contact_id={contact_id}: {str(e)}")
        return None


def _search_companies_with_filters(
    hubspot: HubspotService,
    batch_size: int = 25,
    min_revenue: float = 1000000,
    excluded_states: List[str] = None
) -> List[Dict[str, Any]]:
    """Enhanced company search with filtering."""
    url = f"{hubspot.base_url}/crm/v3/objects/companies/search"
    after = None
    all_results = []

    while True and (not TEST_MODE or len(all_results) < TEST_LIMIT):
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "annualrevenue",
                            "operator": "GTE",
                            "value": str(min_revenue)
                        }
                    ]
                }
            ],
            "properties": [
                "name", "city", "state", "annualrevenue",
                "company_type", "industry", "website",
                "peak_season_start", "peak_season_end"
            ],
            "limit": min(batch_size, TEST_LIMIT) if TEST_MODE else batch_size
        }
        
        if after:
            payload["after"] = after

        try:
            logger.info(f"Fetching companies (Test Mode: {TEST_MODE})")
            response = hubspot._make_hubspot_post(url, payload)
            if not response:
                break

            results = response.get("results", [])
            
            # Filter out excluded states
            if excluded_states:
                results = [
                    r for r in results 
                    if r.get("properties", {}).get("state") not in excluded_states
                ]

            all_results.extend(results)
            
            logger.info(f"Retrieved {len(all_results)} companies so far")

            # Break if we've reached test limit
            if TEST_MODE and len(all_results) >= TEST_LIMIT:
                logger.info(f"Test mode: Reached limit of {TEST_LIMIT} companies")
                break

            paging = response.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link

        except Exception as e:
            logger.error(f"Error fetching companies: {str(e)}")
            break

    # Ensure we don't exceed test limit
    if TEST_MODE:
        all_results = all_results[:TEST_LIMIT]
        logger.info(f"Test mode: Returning {len(all_results)} companies")

    return all_results


if __name__ == "__main__":
    main()
