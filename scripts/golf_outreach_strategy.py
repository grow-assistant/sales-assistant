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
    # In US, DST is from second Sunday in March to first Sunday in November
    dt = datetime.now()
    is_dst = 3 <= dt.month <= 11  # True if between March and November
    
    # Get the offset relative to Arizona time
    offset_hours = offsets['dst'] if is_dst else offsets['std']
    
    # Apply offset from Arizona time
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

def get_best_time(persona: str, sequence_num: int) -> dict:
    """
    Determine best time of day based on persona and email sequence number.
    Returns a dict with start and end hours/minutes in 24-hour format.
    Times are aligned to 30-minute windows.
    """
    logger.debug(f"Getting best time for persona: {persona}, sequence_num: {sequence_num}")
    
    time_windows = {
        "General Manager": {
            2: [  # Sequence 2: Morning or afternoon hours
                {
                    "start_hour": 8, "start_minute": 30,
                    "end_hour": 10, "end_minute": 30
                },
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ],
            1: [  # Sequence 1: Afternoon hours only
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ]
        },
        "Food & Beverage Director": {
            2: [  # Sequence 2: Morning or afternoon hours
                {
                    "start_hour": 9, "start_minute": 30,
                    "end_hour": 11, "end_minute": 30
                },
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ],
            1: [  # Sequence 1: Afternoon hours only
                {
                    "start_hour": 15, "start_minute": 0,
                    "end_hour": 16, "end_minute": 30
                }
            ]
        }
    }
    
    # Convert persona to title case to handle different formats
    persona = " ".join(word.capitalize() for word in persona.split("_"))
    logger.debug(f"Normalized persona: {persona}")
    
    # Get time windows for the persona and sequence number, defaulting to GM times if not found
    windows = time_windows.get(persona, time_windows["General Manager"]).get(sequence_num, time_windows["General Manager"][1])
    if persona not in time_windows or sequence_num not in time_windows[persona]:
        logger.debug(f"No specific time window for {persona} with sequence {sequence_num}, using General Manager defaults")
    
    # For sequence 2, randomly choose between morning and afternoon windows
    if sequence_num == 2:
        selected_window = random.choice(windows)
        logger.debug(f"Sequence 2: Randomly selected window: {selected_window['start_hour']}:{selected_window['start_minute']} - {selected_window['end_hour']}:{selected_window['end_minute']}")
    else:
        selected_window = windows[0]  # For sequence 1, use the single defined window
    
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
    best_time = get_best_time(persona, 1)
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

def calculate_send_date(geography: str, persona: str, state: str, sequence_num: int, season_data: dict = None) -> datetime:
    """Calculate the next appropriate send date based on outreach window."""
    logger.debug(f"Calculating send date for: geography={geography}, persona={persona}, state={state}, sequence_num={sequence_num}")
    
    outreach_window = get_best_outreach_window(geography, persona, season_data=season_data)
    best_months = outreach_window["Best Month"]
    preferred_time = get_best_time(persona, sequence_num)
    preferred_days = [1, 2, 3]  # Tuesday, Wednesday, Thursday
    
    # Get current time and adjust it for target state's timezone first
    now = datetime.now()
    state_now = adjust_send_time(now, state)
    
    # Calculate target time for today
    start_hour = int(preferred_time["start"])
    start_minutes = int((preferred_time["start"] % 1) * 60)
    target_time = state_now.replace(hour=start_hour, minute=start_minutes, second=0, microsecond=0)
    
    # If target time is in the past, move to next available time slot
    if target_time <= state_now:
        # If we're past today's target time, try tomorrow at the same time
        target_time += timedelta(days=1)
    
    # Find next available preferred day
    while target_time.weekday() not in preferred_days:
        target_time += timedelta(days=1)
    
    # Final timezone adjustment back from state time
    final_date = adjust_send_time(target_time, state)
    logger.debug(f"Final scheduled date after timezone adjustment: {final_date}")
    
    # Verify final date is in the future
    if final_date <= datetime.now():
        logger.warning("Final date was not in future, adding one day")
        final_date += timedelta(days=1)
        while final_date.weekday() not in preferred_days:
            final_date += timedelta(days=1)
    
    logger.info(f"Scheduled send date and time: {final_date}")
    return final_date
