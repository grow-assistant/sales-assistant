"""
Scripts for determining optimal outreach timing based on club and contact attributes.
"""
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

def load_state_offsets():
    """Load state hour offsets from CSV file."""
    offsets = {}
    csv_path = os.path.join('docs', 'data', 'state_timezones.csv')
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = row['state_code']
            offsets[state] = {
                'dst': int(row['daylight_savings']),
                'std': int(row['standard_time'])
            }
    return offsets

STATE_OFFSETS = load_state_offsets()

def adjust_send_time(send_time: datetime, state_code: str) -> datetime:
    """Adjust send time based on state's hour offset."""
    if not state_code:
        logger.warning("No state code provided, using original time")
        return send_time
        
    offsets = STATE_OFFSETS.get(state_code.upper())
    if not offsets:
        logger.warning(f"No offset data for state {state_code}, using original time")
        return send_time
    
    # Determine if we're in DST
    is_dst = datetime.now().astimezone().dst() != timedelta(0)
    offset_hours = offsets['dst'] if is_dst else offsets['std']
    
    # Apply offset
    adjusted_time = send_time + timedelta(hours=offset_hours)
    logger.debug(f"Adjusted time from {send_time} to {adjusted_time} for state {state_code} (offset: {offset_hours}h)")
    return adjusted_time

def get_best_month(geography: str, club_type: str = None, season_data: dict = None) -> list:
    """
    Determine best outreach months based on geography/season and club type.
    """
    current_month = datetime.now().month
    
    # If we have season data, use it as primary decision factor
    if season_data:
        peak_start = season_data.get('peak_season_start', '')
        peak_end = season_data.get('peak_season_end', '')
        
        if peak_start and peak_end:
            peak_start_month = int(peak_start.split('-')[0])
            peak_end_month = int(peak_end.split('-')[0])
            
            logger.debug(f"Peak season: {peak_start_month} to {peak_end_month}")
            
            # For winter peak season (crossing year boundary)
            if peak_start_month > peak_end_month:
                if current_month >= peak_start_month or current_month <= peak_end_month:
                    # We're in peak season, target shoulder season
                    return [9]  # September (before peak starts)
                else:
                    # We're in shoulder season
                    return [1]  # January
            # For summer peak season
            else:
                if peak_start_month <= current_month <= peak_end_month:
                    # We're in peak season, target shoulder season
                    return [peak_start_month - 1] if peak_start_month > 1 else [12]
                else:
                    # We're in shoulder season
                    return [1]  # January
    
    # Fallback to geography-based matrix
    month_matrix = {
        "Year-Round Golf": [1, 9],      # January or September
        "Peak Winter Season": [9],       # September
        "Peak Summer Season": [2],       # February
        "Short Summer Season": [1],      # January
        "Shoulder Season Focus": [2, 10]  # February or October
    }
    
    return month_matrix.get(geography, [1, 9])

def get_best_time(persona: str) -> dict:
    """
    Determine best time of day based on persona.
    Returns a dict with start and end hours in 24-hour format.
    """
    time_windows = {
        "General Manager": {"start": 9, "end": 11},  # 9am-11am
        "Membership Director": {"start": 13, "end": 15},  # 1pm-3pm
        "Food & Beverage Director": {"start": 10, "end": 12},  # 10am-12pm
        "Golf Professional": {"start": 8, "end": 10},  # 8am-10am
        "Superintendent": {"start": 7, "end": 9}  # 7am-9am
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    return time_windows.get(persona, {"start": 9, "end": 11})

def get_best_day(persona: str) -> list:
    """
    Determine best days of week based on persona.
    Returns a list of day numbers (0 = Monday, 6 = Sunday)
    """
    day_mappings = {
        "General Manager": [1, 3],  # Tuesday, Thursday
        "Membership Director": [2, 3],  # Wednesday, Thursday
        "Food & Beverage Director": [2, 3],  # Wednesday, Thursday
        "Golf Professional": [1, 2],  # Tuesday, Wednesday
        "Superintendent": [1, 2]  # Tuesday, Wednesday
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    return day_mappings.get(persona, [1, 3])

def get_best_outreach_window(persona: str, geography: str, club_type: str = None, season_data: dict = None) -> Dict[str, Any]:
    """Get the optimal outreach window based on persona and geography."""
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona)
    best_days = get_best_day(persona)
    
    logger.debug(f"Calculated base outreach window", extra={
        "persona": persona,
        "geography": geography,
        "best_months": best_months,
        "best_time": best_time,
        "best_days": best_days
    })
    
    return {
        "Best Month": best_months,
        "Best Time": best_time,
        "Best Day": best_days
    }
