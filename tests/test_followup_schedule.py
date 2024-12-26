import pytest
from scripts.schedule_outreach import OUTREACH_SCHEDULE
from scheduling.followup_generation import generate_followup_email_xai, parse_subject_and_body
from scheduling.database import get_db_connection
import datetime

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
