# scripts/golf_outreach_strategy.py
# """
# Scripts for determining optimal outreach timing based on club and contact attributes.
# """
from typing import Dict, Any
import csv
import logging
from datetime import datetime, timedelta
import os
import random

logger = logging.getLogger(__name__)

# Update the logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to file
        logging.StreamHandler()          # Log to console
    ]
)

# Add a debug message to verify logging is working
logger.debug("Golf outreach strategy logging initialized")

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

def get_best_time(persona: str, sequence_num: int, timezone_offset: int = 0) -> dict:
    """
    Determine best time of day based on persona and email sequence number.
    Returns a dict with start time in MST.
    
    Args:
        persona: The recipient's role/persona
        sequence_num: Email sequence number (1 or 2)
        timezone_offset: Hours offset from MST (e.g., 2 for EST)
    """
    logger.debug(f"\n=== Time Window Selection ===")
    logger.debug(f"Input - Persona: {persona}, Sequence: {sequence_num}, Timezone offset: {timezone_offset}")
    
    # Default to General Manager if persona is None or invalid
    if not persona:
        logger.warning("No persona provided, defaulting to General Manager")
        persona = "General Manager"
    
    # Normalize persona string
    try:
        persona = " ".join(word.capitalize() for word in persona.split("_"))
        if persona.lower() in ["general manager", "gm", "club manager", "general_manager"]:
            persona = "General Manager"
        elif persona.lower() in ["f&b manager", "food & beverage manager", "food and beverage manager", "fb_manager"]:
            persona = "Food & Beverage Director"
    except Exception as e:
        logger.error(f"Error normalizing persona: {str(e)}")
        persona = "General Manager"
    
    # Define windows in LOCAL time
    time_windows = {
        "General Manager": {
            # 2: {  # Sequence 2: Morning window only
            #     "start_hour": 8, "start_minute": 30,  # 8:30 AM LOCAL
            #     "end_hour": 10, "end_minute": 30      # 10:30 AM LOCAL
            # },
            1: {  # Sequence 1: Afternoon window only
                "start_hour": 13, "start_minute": 30,  # 1:30 PM LOCAL
                "end_hour": 16, "end_minute": 00      # 4:00 PM LOCAL
            }
        },
        "Food & Beverage Director": {
            # 2: {  # Sequence 2: Morning window only
            #     "start_hour": 9, "start_minute": 30,  # 9:30 AM LOCAL
            #     "end_hour": 11, "end_minute": 30      # 11:30 AM LOCAL
            # },
            1: {  # Sequence 1: Afternoon window only
                "start_hour": 13, "start_minute": 30,  # 1:30 PM LOCAL
                "end_hour": 16, "end_minute": 00      # 4:00 PM LOCAL
            }
        }
    }
    
    # Get time window for the persona and sequence number, defaulting to GM times if not found
    window = time_windows.get(persona, time_windows["General Manager"]).get(sequence_num, time_windows["General Manager"][1])
    
    # Convert LOCAL time to MST by subtracting timezone offset
    start_time = window["start_hour"] + window["start_minute"] / 60
    mst_start = start_time - timezone_offset
    
    logger.debug(f"Selected window (LOCAL): {window['start_hour']}:{window['start_minute']:02d}")
    logger.debug(f"Converted to MST (offset: {timezone_offset}): {int(mst_start)}:{int((mst_start % 1) * 60):02d}")
    
    return {
        "start": mst_start,
        "end": mst_start  # Since we're only using start time, just duplicate it
    }

def get_best_outreach_window(
    persona: str, 
    geography: str, 
    sequence_num: int = 1,
    club_type: str = None, 
    season_data: dict = None,
    timezone_offset: int = 0  # Add timezone_offset parameter
) -> Dict[str, Any]:
    """
    Get the optimal outreach window based on persona and geography.
    """
    logger.debug(f"Getting outreach window for persona: {persona}, geography: {geography}, sequence: {sequence_num}, timezone_offset: {timezone_offset}")
    
    best_months = get_best_month(geography, club_type, season_data)
    best_time = get_best_time(persona, sequence_num, timezone_offset)  # Pass timezone_offset
    best_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    logger.debug(f"Calculated base outreach window", extra={
        "persona": persona,
        "geography": geography,
        "sequence": sequence_num,
        "timezone_offset": timezone_offset,
        "best_months": best_months,
        "best_time": best_time,
        "best_days": best_days
    })
    
    return {
        "Best Month": best_months,
        "Best Time": best_time,
        "Best Day": best_days
    }

def calculate_send_date(geography: str, persona: str, state: str, sequence_num: int = 1, season_data: dict = None, day_offset: int = 0) -> datetime:
    try:
        logger.debug(f"\n=== Starting Send Date Calculation ===")
        logger.debug(f"Inputs: geography={geography}, persona={persona}, state={state}, sequence={sequence_num}, day_offset={day_offset}")
        
        # Get timezone offset from MST for this state
        offsets = STATE_OFFSETS.get(state.upper() if state else None)
        logger.debug(f"Found offsets for {state}: {offsets}")
        
        if not offsets:
            logger.warning(f"No offset data for state {state}, using MST")
            timezone_offset = 0
        else:
            # Determine if we're in DST
            dt = datetime.now()
            is_dst = 3 <= dt.month <= 11
            # Invert the offset since we want LOCAL = MST + offset
            timezone_offset = -(offsets['dst'] if is_dst else offsets['std'])
            logger.debug(f"Using timezone offset of {timezone_offset} hours for {state} (DST: {is_dst})")
        
        # Get best outreach window (returns times in MST)
        outreach_window = get_best_outreach_window(
            geography=geography,
            persona=persona,
            sequence_num=sequence_num,
            season_data=season_data,
            timezone_offset=timezone_offset
        )
        
        logger.debug(f"Outreach window result: {outreach_window}")
        
        preferred_time = outreach_window["Best Time"]  # Already in MST
        
        # Use start of window
        start_hour = int(preferred_time["start"])
        start_minutes = int((preferred_time["start"] % 1) * 60)
        
        logger.debug(f"Using MST time: {start_hour}:{start_minutes:02d}")
        
        # Start with today + day_offset
        target_date = datetime.now() + timedelta(days=day_offset)
        target_time = target_date.replace(hour=start_hour, minute=start_minutes, second=0, microsecond=0)
        
        # Find next available preferred day (Tuesday, Wednesday, Thursday)
        while target_time.weekday() not in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
            target_time += timedelta(days=1)
            logger.debug(f"Moved to next preferred day: {target_time}")
        
        logger.debug(f"Final scheduled date (MST): {target_time}")
        return target_time

    except Exception as e:
        logger.error(f"Error calculating send date: {str(e)}", exc_info=True)
        return datetime.now() + timedelta(days=1, hours=10)
