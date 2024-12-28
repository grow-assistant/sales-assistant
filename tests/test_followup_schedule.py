import pytest
from pathlib import Path
import csv
from scripts.schedule_outreach import OUTREACH_SCHEDULE, get_best_outreach_window
from scheduling.followup_generation import generate_followup_email_xai, parse_subject_and_body
from scheduling.database import get_db_connection
import datetime

# Load test data from Golf_Outreach_Strategy.csv
STRATEGY_CSV = Path(__file__).parent.parent / 'docs' / 'data' / 'Golf_Outreach_Strategy.csv'
STRATEGY_DATA = []
with STRATEGY_CSV.open('r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    STRATEGY_DATA = list(reader)

def test_followup_schedule_configuration():
    """Test that the follow-up schedule is configured correctly with all required entries."""
    # Verify we have all expected follow-ups
    assert len(OUTREACH_SCHEDULE) == 4, "Should have 4 entries (initial + 3 follow-ups)"
    
    # Verify day offsets
    expected_days = [0, 3, 7, 14]
    actual_days = [entry["days_from_now"] for entry in OUTREACH_SCHEDULE]
    assert actual_days == expected_days, f"Expected days {expected_days}, got {actual_days}"
    
    # Verify all entries have required fields
    required_fields = ["name", "days_from_now", "subject", "body"]
    for entry in OUTREACH_SCHEDULE:
        for field in required_fields:
            assert field in entry, f"Missing required field {field} in entry {entry['name']}"
            assert entry[field] is not None, f"Field {field} is None in entry {entry['name']}"

def test_followup_names_and_subjects():
    """Test that follow-up names and subjects are properly configured."""
    expected_names = [
        "Email #1 (Day 0)",
        "Follow-Up #1 (Day 3–4)",
        "Follow-Up #2 (Day 7)",
        "Follow-Up #3 (Day 14)"
    ]
    actual_names = [entry["name"] for entry in OUTREACH_SCHEDULE]
    assert actual_names == expected_names, f"Expected names {expected_names}, got {actual_names}"

def test_followup_body_content():
    """Test that follow-up email bodies contain appropriate content for their sequence."""
    # Test day 7 follow-up content
    day_7_entry = [entry for entry in OUTREACH_SCHEDULE if entry["days_from_now"] == 7][0]
    assert "week since we reached out" in day_7_entry["body"], "Day 7 follow-up should mention timing"
    assert "member experiences" in day_7_entry["body"], "Day 7 follow-up should mention member experience"
    
    # Test day 14 follow-up content
    day_14_entry = [entry for entry in OUTREACH_SCHEDULE if entry["days_from_now"] == 14][0]
    assert "final note" in day_14_entry["body"].lower(), "Day 14 follow-up should indicate it's the final one"
    assert "improvements in both member satisfaction" in day_14_entry["body"], "Day 14 follow-up should mention improvements"

def test_role_based_scheduling():
    """Test that scheduling windows are correctly determined based on role."""
    test_cases = [
        {
            "role": "Membership Director",
            "club_type": "Private Clubs",
            "geography": "Year-Round Golf",
            "expected": {
                "Best Time": "Early afternoon (1–3 PM)",
                "Best Day": "Wednesday or Thursday",
                "Best Month": "January or September"
            }
        },
        {
            "role": "General Manager",
            "club_type": "Public Courses",
            "geography": "Year-Round Golf",
            "expected": {
                "Best Time": "Mid-morning (9–11 AM)",
                "Best Day": "Tuesday or Thursday",
                "Best Month": "January or September"
            }
        },
        {
            "role": "Food & Beverage Director",
            "club_type": "Resorts",
            "geography": "Year-Round Golf",
            "expected": {
                "Best Time": "Late morning (10–12 AM)",
                "Best Day": "Wednesday or Thursday",
                "Best Month": "January or September"
            }
        }
    ]
    
    for case in test_cases:
        window = get_best_outreach_window(case["role"], case["geography"], case["club_type"])
        assert window == case["expected"], \
            f"Mismatch for {case['role']} at {case['club_type']}: expected {case['expected']}, got {window}"

def test_default_scheduling_values():
    """Test that scheduling handles missing or default values appropriately."""
    # Test with missing role (should default to General Manager)
    default_role_window = get_best_outreach_window("", "Year-Round Golf", "Public Courses")
    expected_default_role = {
        "Best Time": "Mid-morning (9–11 AM)",
        "Best Day": "Tuesday or Thursday",
        "Best Month": "January or September"
    }
    assert default_role_window == expected_default_role, \
        "Missing role should default to General Manager schedule"
    
    # Test with missing club type (should default to Public Courses)
    default_club_window = get_best_outreach_window("General Manager", "Year-Round Golf", "")
    expected_default_club = {
        "Best Time": "Mid-morning (9–11 AM)",
        "Best Day": "Tuesday or Thursday",
        "Best Month": "January or September"
    }
    assert default_club_window == expected_default_club, \
        "Missing club type should default to Public Courses schedule"

from scripts.golf_outreach_strategy import build_outreach_strategy_csv, get_best_outreach_window

def test_strategy_data_integration():
    """Test that scheduling uses data from Golf_Outreach_Strategy.csv correctly."""
    # Create test data file
    test_file = "test_strategy.csv"
    build_outreach_strategy_csv(file_path=test_file)
    
    # Test specific combinations
    test_cases = [
        {
            "persona": "Membership Director",
            "geography": "Year-Round Golf",
            "club_type": "Private Clubs",
            "expected": {
                "Best Month": "January or September",
                "Best Time": "Early afternoon (1–3 PM)",
                "Best Day": "Wednesday or Thursday"
            }
        },
        {
            "persona": "General Manager",
            "geography": "Peak Summer Season",
            "club_type": "Public Courses",
            "expected": {
                "Best Month": "February",
                "Best Time": "Mid-morning (9–11 AM)",
                "Best Day": "Tuesday or Thursday"
            }
        }
    ]
    
    for case in test_cases:
        result = get_best_outreach_window(
            case["persona"],
            case["geography"],
            case["club_type"],
            file_path=test_file
        )
        assert result == case["expected"], \
            f"Mismatch for {case['persona']} at {case['club_type']}: expected {case['expected']}, got {result}"
