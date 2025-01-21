"""
Scripts for determining optimal outreach timing based on club and contact attributes.
"""
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os
import random

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
    logger.debug(f"Loaded timezone offsets for {len(offsets)} states")
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
    logger.debug(f"Adjusted time from {send_time} to {adjusted_time} for state {state_code} (offset: {offset_hours}h, DST: {is_dst})")
    return adjusted_time

def get_best_month(geography: str, club_type: str = None, season_data: dict = None) -> list:
    """
    Determine best outreach months based on geography/season and club type.
    """
    current_month = datetime.now().month
    logger.debug(f"Determining best month for geography: {geography}, club_type: {club_type}, current month: {current_month}")
    
    # If we have season data, use it as primary decision factor
    if season_data:
        peak_start = season_data.get('peak_season_start', '')
        peak_end = season_data.get('peak_season_end', '')
        logger.debug(f"Using season data - peak start: {peak_start}, peak end: {peak_end}")
        
        if peak_start and peak_end:
            peak_start_month = int(peak_start.split('-')[0])
            peak_end_month = int(peak_end.split('-')[0])
            
            logger.debug(f"Peak season: {peak_start_month} to {peak_end_month}")
            
            # For winter peak season (crossing year boundary)
            if peak_start_month > peak_end_month:
                if current_month >= peak_start_month or current_month <= peak_end_month:
                    logger.debug("In winter peak season, targeting September shoulder season")
                    return [9]  # September (before peak starts)
                else:
                    logger.debug("In winter shoulder season, targeting January")
                    return [1]  # January
            # For summer peak season
            else:
                if peak_start_month <= current_month <= peak_end_month:
                    target = [peak_start_month - 1] if peak_start_month > 1 else [12]
                    logger.debug(f"In summer peak season, targeting month {target}")
                    return target
                else:
                    logger.debug("In summer shoulder season, targeting January")
                    return [1]  # January
    
    # Fallback to geography-based matrix
    month_matrix = {
        "Year-Round Golf": [1, 9],      # January or September
        "Peak Winter Season": [9],       # September
        "Peak Summer Season": [2],       # February
        "Short Summer Season": [1],      # January
        "Shoulder Season Focus": [2, 10]  # February or October
    }
    
    result = month_matrix.get(geography, [1, 9])
    logger.debug(f"Using geography matrix fallback for {geography}, selected months: {result}")
    return result

def get_best_time(persona: str) -> dict:
    """
    Determine best time of day based on persona.
    Returns a dict with start and end hours/minutes in 24-hour format.
    Times are aligned to 30-minute windows.
    """
    logger.debug(f"Getting best time for persona: {persona}")
    
    time_windows = {
        "General Manager": [
            {
                "start_hour": 8, "start_minute": 30,
                "end_hour": 10, "end_minute": 30
            },  # 8:30-10:30 AM
            # {
            #     "start_hour": 15, "start_minute": 0,
            #     "end_hour": 16, "end_minute": 30
            # }   # 3:00-4:30 PM
        ],
        "Food & Beverage Director": [
            {
                "start_hour": 9, "start_minute": 30,
                "end_hour": 11, "end_minute": 30
            },  # 9:30-11:30 AM
            # {
            #     "start_hour": 15, "start_minute": 0,
            #     "end_hour": 16, "end_minute": 30
            # }   # 3:00-4:30 PM
        ],
        "Golf Professional": [
            {
                "start_hour": 8, "start_minute": 0,
                "end_hour": 10, "end_minute": 0
            }   # 8:00-10:00 AM
        ]
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    logger.debug(f"Normalized persona: {persona}")
    
    # Get time windows for the persona, defaulting to GM times if not found
    windows = time_windows.get(persona, time_windows["General Manager"])
    if persona not in time_windows:
        logger.debug(f"No specific time window for {persona}, using General Manager defaults")
    
    # Randomly select between morning and afternoon windows if multiple exist
    selected_window = random.choice(windows)
    logger.debug(f"Selected time window: {selected_window['start_hour']}:{selected_window['start_minute']} - {selected_window['end_hour']}:{selected_window['end_minute']}")
    
    # Update calculate_send_date function expects start/end format
    return {
        "start": selected_window["start_hour"] + selected_window["start_minute"] / 60,
        "end": selected_window["end_hour"] + selected_window["end_minute"] / 60
    }

def get_best_outreach_window(persona: str, geography: str, club_type: str = None, season_data: dict = None) -> Dict[str, Any]:
    """Get the optimal outreach window based on persona and geography."""
    logger.debug(f"Getting outreach window for persona: {persona}, geography: {geography}, club_type: {club_type}")
    
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona)
    best_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday (0 = Monday, 6 = Sunday)
    
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

def calculate_send_date(geography: str, persona: str, state: str, season_data: dict = None) -> datetime:
    """Calculate the next appropriate send date based on outreach window."""
    logger.debug(f"Calculating send date for: geography={geography}, persona={persona}, state={state}")
    
    outreach_window = get_best_outreach_window(geography, persona, season_data=season_data)
    best_months = outreach_window["Best Month"]
    preferred_time = outreach_window["Best Time"]
    preferred_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    # Find the next preferred day of week
    now = datetime.now()
    today_weekday = now.weekday()
    
    # Only use today if it's a preferred day
    if today_weekday in preferred_days:
        target_date = now
        logger.debug(f"Using today ({target_date}) as it's a preferred day (weekday: {today_weekday})")
    else:
        days_ahead = [(day - today_weekday) % 7 for day in preferred_days]
        next_preferred_day = min(days_ahead)
        target_date = now + timedelta(days=next_preferred_day)
        logger.debug(f"Using future date ({target_date}) as today isn't a preferred day (weekday: {today_weekday})")
    
    # Apply preferred time
    start_hour = int(preferred_time["start"])
    start_minutes = int((preferred_time["start"] % 1) * 60)
    target_date = target_date.replace(hour=start_hour, minute=start_minutes)
    logger.debug(f"Applied preferred time: {target_date}")
    
    # Adjust for state timezone
    final_date = adjust_send_time(target_date, state)
    logger.debug(f"Final scheduled date after timezone adjustment: {final_date}")
    
    return final_date
