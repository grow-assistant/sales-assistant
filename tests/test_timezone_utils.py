"""
tests/test_timezone_utils.py

Tests for timezone adjustment functionality.
"""

import pytest
import datetime
from utils.timezone_utils import (
    load_timezone_data,
    get_state_timezone_info,
    adjust_for_timezone,
    _is_dst
)

def test_load_timezone_data():
    """Test that timezone data loads correctly from CSV."""
    load_timezone_data()
    from utils.timezone_utils import STATE_TIMEZONE_DATA
    
    # Check some key states
    assert "CA" in STATE_TIMEZONE_DATA
    assert "NY" in STATE_TIMEZONE_DATA
    assert "FL" in STATE_TIMEZONE_DATA
    
    # Verify data structure
    ca_data = STATE_TIMEZONE_DATA["CA"]
    assert "timezone" in ca_data
    assert "dst_offset" in ca_data
    assert "std_offset" in ca_data
    assert ca_data["timezone"] == "Pacific"

def test_get_state_timezone_info():
    """Test timezone info retrieval for different states."""
    # Test valid state (California)
    ca_info = get_state_timezone_info("CA")
    assert ca_info is not None
    assert ca_info["timezone"] == "Pacific"
    assert ca_info["dst_offset"] == "-1 hour"
    assert ca_info["std_offset"] == "-1 hour"
    
    # Test case insensitive (New York)
    ny_info = get_state_timezone_info("ny")
    assert ny_info is not None
    assert ny_info["timezone"] == "Eastern"
    assert ny_info["dst_offset"] == "-3 hours"
    assert ny_info["std_offset"] == "-2 hours"
    
    # Test Arizona (reference state)
    az_info = get_state_timezone_info("AZ")
    assert az_info is not None
    assert az_info["timezone"] == "Mountain"
    assert az_info["dst_offset"] == "-1 hour"
    assert az_info["std_offset"] == "0 hours"
    
    # Test invalid state
    invalid_info = get_state_timezone_info("XX")
    assert invalid_info is None

def test_adjust_for_timezone():
    """Test timezone adjustments for different states and seasons."""
    # Test summer time (DST) adjustment for California relative to Arizona
    summer_time = datetime.datetime(2024, 7, 15, 10, 0)  # 10 AM
    ca_adjusted = adjust_for_timezone(summer_time, "CA")
    assert ca_adjusted == summer_time - datetime.timedelta(hours=1)  # -1 hour from AZ in DST
    
    # Test winter time adjustment for New York relative to Arizona
    winter_time = datetime.datetime(2024, 1, 15, 14, 0)  # 2 PM
    ny_adjusted = adjust_for_timezone(winter_time, "NY")
    assert ny_adjusted == winter_time - datetime.timedelta(hours=2)  # -2 hours from AZ in standard time
    
    # Test Arizona itself (no adjustment needed)
    az_time = datetime.datetime(2024, 7, 15, 10, 0)  # 10 AM
    az_adjusted = adjust_for_timezone(az_time, "AZ")
    assert az_adjusted == az_time + datetime.timedelta(hours=-1)  # -1 hour in DST
    
    # Test invalid state (should return original time)
    invalid_adjusted = adjust_for_timezone(summer_time, "XX")
    assert invalid_adjusted == summer_time

def test_dst_detection():
    """Test DST detection logic."""
    # Test summer (should be DST)
    summer = datetime.datetime(2024, 7, 1)
    assert _is_dst(summer) is True
    
    # Test winter (should not be DST)
    winter = datetime.datetime(2024, 1, 1)
    assert _is_dst(winter) is False
    
    # Test DST start (second Sunday in March)
    dst_start = datetime.datetime(2024, 3, 10)
    assert _is_dst(dst_start) is True
    
    # Test DST end (first Sunday in November)
    dst_end = datetime.datetime(2024, 11, 3)
    assert _is_dst(dst_end) is False
