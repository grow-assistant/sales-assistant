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
    """Test that timezone data loads correctly from the CSV file."""
    load_timezone_data()
    from utils.timezone_utils import STATE_TIMEZONE_DATA

    # Spot-check a few states
    assert "CA" in STATE_TIMEZONE_DATA, "California must be in the dataset"
    assert "NY" in STATE_TIMEZONE_DATA, "New York must be in the dataset"
    assert "FL" in STATE_TIMEZONE_DATA, "Florida must be in the dataset"
    assert "AZ" in STATE_TIMEZONE_DATA, "Arizona must be in the dataset"

    ca_data = STATE_TIMEZONE_DATA["CA"]
    assert ca_data["timezone"] == "Pacific"
    assert "dst_offset" in ca_data
    assert "std_offset" in ca_data

def test_get_state_timezone_info():
    """Test retrieving state timezone info by code."""
    info_ca = get_state_timezone_info("CA")
    assert info_ca is not None
    assert info_ca["timezone"] == "Pacific"

    info_ny = get_state_timezone_info("ny")
    assert info_ny is not None
    assert info_ny["timezone"] == "Eastern"

    info_az = get_state_timezone_info("AZ")
    assert info_az is not None
    assert info_az["timezone"] == "Mountain"

    info_none = get_state_timezone_info("XX")
    assert info_none is None, "Should return None for invalid state codes"

def test_adjust_for_timezone():
    """Test adjusting a datetime for a lead's state time offset from AZ."""
    # Summer test for CA (DST)
    summer_time = datetime.datetime(2024, 7, 15, 10, 0)
    ca_adjusted = adjust_for_timezone(summer_time, "CA")
    # CA is typically -1 hour from AZ in DST in this data
    assert ca_adjusted == summer_time - datetime.timedelta(hours=1)

    # Winter test for NY
    winter_time = datetime.datetime(2024, 1, 15, 14, 0)
    ny_adjusted = adjust_for_timezone(winter_time, "NY")
    # In the provided CSV, NY standard offset might be -2 from AZ, etc.
    assert ny_adjusted == winter_time - datetime.timedelta(hours=2)

    # If invalid state
    invalid_adjusted = adjust_for_timezone(summer_time, "XX")
    assert invalid_adjusted == summer_time, "No adjustment for invalid states"

def test__is_dst():
    """Test the rough DST detection logic."""
    # A day in July
    july_date = datetime.datetime(2024, 7, 1)
    assert _is_dst(july_date) is True, "July should be DST"

    # A day in January
    jan_date = datetime.datetime(2024, 1, 1)
    assert _is_dst(jan_date) is False, "January should not be DST"

    # DST starts 2nd Sunday in March
    march_dst = datetime.datetime(2024, 3, 10)
    assert _is_dst(march_dst) is True

    # DST ends 1st Sunday in November
    nov_dst = datetime.datetime(2024, 11, 3)
    assert _is_dst(nov_dst) is False
