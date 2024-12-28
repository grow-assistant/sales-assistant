"""
utils/timezone_utils.py

Handles timezone conversions and scheduling adjustments based on state location.
Uses state_timezones.csv for offset data.
"""

import csv
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from utils.logging_setup import logger

PROJECT_ROOT = Path(__file__).parent.parent
TIMEZONE_CSV = PROJECT_ROOT / 'docs' / 'data' / 'state_timezones.csv'

# Cache for timezone data
STATE_TIMEZONE_DATA: Dict[str, Dict[str, str]] = {}


def load_timezone_data() -> None:
    """Load timezone data from CSV if not already loaded."""
    if not STATE_TIMEZONE_DATA:
        try:
            with open(TIMEZONE_CSV, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    STATE_TIMEZONE_DATA[row['State Code']] = {
                        'timezone': row['Time Zone'],
                        'dst_offset': row['DST Offset from AZ'],
                        'std_offset': row['Standard Offset from AZ']
                    }
            logger.info("Timezone data loaded successfully", extra={
                "states_loaded": len(STATE_TIMEZONE_DATA),
                "source": str(TIMEZONE_CSV)
            })
        except Exception as e:
            logger.error("Failed to load timezone data", extra={
                "error": str(e),
                "source": str(TIMEZONE_CSV)
            })


def get_state_timezone_info(state_code: str) -> Optional[Dict[str, str]]:
    """
    Get timezone information for a given state.
    
    Args:
        state_code: Two-letter state code (e.g., 'CA', 'NY')
        
    Returns:
        Dictionary with timezone name and offsets, or None if state not found
    """
    if not STATE_TIMEZONE_DATA:
        load_timezone_data()
    
    state_code = state_code.upper()
    timezone_info = STATE_TIMEZONE_DATA.get(state_code)
    
    if not timezone_info:
        logger.warning("No timezone data found for state", extra={
            "state_code": state_code,
            "available_states": list(STATE_TIMEZONE_DATA.keys())
        })
        return None
    
    return timezone_info


def adjust_for_timezone(proposed_time: datetime.datetime, state_code: str) -> datetime.datetime:
    """
    Adjust a proposed time based on the lead's state timezone.
    
    Args:
        proposed_time: The datetime to adjust
        state_code: Two-letter state code
        
    Returns:
        Adjusted datetime accounting for timezone differences
    """
    timezone_info = get_state_timezone_info(state_code)
    if not timezone_info:
        logger.warning("Using default timezone (no adjustment)", extra={
            "state": state_code,
            "original_time": str(proposed_time)
        })
        return proposed_time
    
    # Determine if we're in DST
    is_dst = _is_dst(proposed_time)
    offset_str = timezone_info['dst_offset'] if is_dst else timezone_info['std_offset']
    
    # Parse the offset string (e.g., "-2 hours" or "-1 hour")
    try:
        offset_hours = int(offset_str.split()[0])
        adjusted_time = proposed_time + datetime.timedelta(hours=offset_hours)
        
        logger.info("Adjusted time for timezone", extra={
            "state": state_code,
            "timezone": timezone_info['timezone'],
            "is_dst": is_dst,
            "offset_applied": offset_str,
            "original_time": str(proposed_time),
            "adjusted_time": str(adjusted_time)
        })
        
        return adjusted_time
    except (ValueError, IndexError) as e:
        logger.error("Failed to parse timezone offset", extra={
            "state": state_code,
            "offset_str": offset_str,
            "error": str(e)
        })
        return proposed_time


def _is_dst(dt: datetime.datetime) -> bool:
    """
    Determine if a given datetime is in DST.
    This is a simplified version - in production you'd want to use pytz or similar.
    """
    # Rough DST approximation for US (2nd Sunday in March to 1st Sunday in November)
    year = dt.year
    dst_start = datetime.datetime(year, 3, 8) + datetime.timedelta(days=(6 - datetime.datetime(year, 3, 8).weekday()))
    dst_end = datetime.datetime(year, 11, 1) + datetime.timedelta(days=(6 - datetime.datetime(year, 11, 1).weekday()))
    
    return dst_start <= dt.replace(tzinfo=None) < dst_end
