import pytest
import csv
from pathlib import Path
import datetime

from scripts.schedule_outreach import OUTREACH_SCHEDULE, get_best_outreach_window
from scheduling.followup_generation import generate_followup_email_xai, parse_subject_and_body
from scheduling.database import get_db_connection
from utils.logging_setup import logger

# Load test data from Golf_Outreach_Strategy.csv
STRATEGY_CSV = Path(__file__).parent.parent / 'docs' / 'data' / 'Golf_Outreach_Strategy.csv'
STRATEGY_DATA = []
if STRATEGY_CSV.exists():
    with STRATEGY_CSV.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        STRATEGY_DATA = list(reader)

def test_followup_schedule_configuration():
    """Test that the follow-up schedule is configured correctly with all required entries."""
    # Check for day 0
    day_0_entry = [entry for entry in OUTREACH_SCHEDULE if entry["days_from_now"] == 0]
    assert day_0_entry, "Must have a day 0 outreach step."
    # Check for day 14 final note
    day_14_entry = [entry for entry in OUTREACH_SCHEDULE if entry["days_from_now"] == 14]
    assert day_14_entry, "Must have a day 14 final outreach step."

def test_followup_body_content():
    """Ensure each follow-up step body references the correct timeline."""
    day_7_entry = [entry for entry in OUTREACH_SCHEDULE if entry["days_from_now"] == 7][0]
    assert "about a week" in day_7_entry["body"].lower(), "Day 7 email should mention it's been about a week"

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
        assert window == case["expected"], (
            f"Mismatch for {case['role']} at {case['club_type']}: "
            f"expected {case['expected']}, got {window}"
        )

def test_default_scheduling_values():
    """Test that scheduling handles missing or default values appropriately."""
    # Missing role
    default_role_window = get_best_outreach_window("", "Year-Round Golf", "Public Courses")
    expected_default_role = {
        "Best Time": "Mid-morning (9–11 AM)",
        "Best Day": "Tuesday or Thursday",
        "Best Month": "January or September"
    }
    assert default_role_window == expected_default_role, "Missing role should default to GM schedule"

    # Missing club type
    default_club_window = get_best_outreach_window("General Manager", "Year-Round Golf", "")
    expected_default_club = {
        "Best Time": "Mid-morning (9–11 AM)",
        "Best Day": "Tuesday or Thursday",
        "Best Month": "January or September"
    }
    assert default_club_window == expected_default_club, "Missing club type should default to Public Courses"

def test_strategy_data_integration():
    """Test that scheduling uses data from Golf_Outreach_Strategy.csv correctly."""
    if not STRATEGY_CSV.exists():
        pytest.skip("Golf_Outreach_Strategy.csv not present, skipping integration test.")
    # Simple check: if we loaded data, ensure certain rows exist
    assert len(STRATEGY_DATA) > 0, "Strategy CSV is empty or not loaded."

def test_xai_followup_generation():
    """Test generating a follow-up email with xAI."""
    # Insert a test lead in DB
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO leads (email, first_name, last_name, role, status)
        OUTPUT Inserted.lead_id
        VALUES (?, ?, ?, ?, 'active')
    """, ("test_user@example.com", "Test", "User", "General Manager"))
    result = cursor.fetchone()
    lead_id = result[0]
    conn.commit()

    # Insert followup row
    sequence_num = 1
    cursor.execute("""
        INSERT INTO followups (lead_id, sequence_num, subject, body, status)
        VALUES (?, ?, '', '', 'pending')
    """, (lead_id, sequence_num))
    conn.commit()

    # Now generate
    generate_followup_email_xai(lead_id, sequence_num)

    # Verify it was updated
    row = cursor.execute("""
        SELECT subject, body, status FROM followups WHERE lead_id = ? AND sequence_num = ?
    """, (lead_id, sequence_num)).fetchone()

    assert row is not None
    assert row.subject, "Subject must be populated by xAI"
    assert row.body, "Body must be populated by xAI"
    assert row.status == "generated"

    # Cleanup
    cursor.execute("DELETE FROM followups WHERE lead_id = ?", (lead_id,))
    cursor.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
    conn.commit()
    conn.close()
