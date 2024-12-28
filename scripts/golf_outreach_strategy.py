"""
scripts/golf_outreach_strategy.py

Generates a CSV table recommending the best outreach month, time of day,
and day of week for each combination of:
- Persona (e.g., GM, Membership, F&B)
- Geography/season (e.g., Peak Winter, etc.)
- Club type (e.g., Private, Public, etc.)

Provides two main functions:
1) build_outreach_strategy_csv(file_path): creates/updates the table
2) get_best_outreach_window(persona, geography, club_type, file_path):
   returns a dictionary with "Best Month", "Best Time", and "Best Day"
   for the specified persona/geography/club_type.
"""

import csv
from pathlib import Path
from typing import Dict
from utils.logging_setup import logger

def build_outreach_strategy_csv(file_path: str = "docs/data/Golf_Outreach_Strategy.csv") -> None:
    """
    Create or overwrite a CSV table with recommended months, times, and days
    for each combination of persona, geography, and club type.
    """
    personas = ["General Manager", "Membership Director", "Food & Beverage Director"]
    geographies = [
        "Year-Round Golf",
        "Peak Winter Season",
        "Peak Summer Season",
        "Short Summer Season",
        "Shoulder Season Focus"
    ]
    club_types = ["Private Clubs", "Public Courses", "Resorts", "Management Companies"]

    months = {
        "Year-Round Golf": "January or September",
        "Peak Winter Season": "September",
        "Peak Summer Season": "February",
        "Short Summer Season": "January",
        "Shoulder Season Focus": "February or October"
    }

    times_of_day = {
        "General Manager": "Mid-morning (9–11 AM)",
        "Membership Director": "Early afternoon (1–3 PM)",
        "Food & Beverage Director": "Late morning (10–12 AM)"
    }

    days_of_week = {
        "General Manager": "Tuesday or Thursday",
        "Membership Director": "Wednesday or Thursday",
        "Food & Beverage Director": "Wednesday or Thursday"
    }

    columns = [
        "Persona",
        "Geography/Golf Season",
        "Club Type",
        "Best Month to Start Outreach",
        "Best Time of Day",
        "Best Day of the Week"
    ]

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for persona in personas:
            for geography in geographies:
                for club_type in club_types:
                    writer.writerow([
                        persona,
                        geography,
                        club_type,
                        months[geography],
                        times_of_day[persona],
                        days_of_week[persona]
                    ])

    logger.info(f"Outreach strategy CSV created or updated at: {file_path}")


def get_best_outreach_window(
    persona: str,
    geography: str,
    club_type: str,
    file_path: str = "docs/data/Golf_Outreach_Strategy.csv"
) -> dict:
    """
    Returns a dictionary with keys "Best Month", "Best Time", and "Best Day"
    from the CSV for the given persona, geography, and club type.

    1) Attempt to match row in Golf_Outreach_Strategy.csv
    2) If no match, return defaults
    """

    # If no persona or club_type, set defaults
    if not persona:
        persona = "General Manager"
        logger.warning("No role provided, using default", extra={"default_role": persona})
    if not club_type:
        club_type = "Public Courses"
        logger.warning("No club type provided, using default", extra={"default_club_type": club_type})

    try:
        # Ensure CSV exists
        build_outreach_strategy_csv(file_path)
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row["Persona"].lower() == persona.lower()
                    and row["Geography/Golf Season"] == geography
                    and row["Club Type"].lower() == club_type.lower()):
                    logger.info("Found matching outreach strategy", extra={
                        "role": persona,
                        "club_type": club_type,
                        "best_time": row["Best Time of Day"],
                        "best_day": row["Best Day of the Week"],
                        "best_month": row["Best Month to Start Outreach"]
                    })
                    return {
                        "Best Month": row["Best Month to Start Outreach"],
                        "Best Time": row["Best Time of Day"],
                        "Best Day": row["Best Day of the Week"]
                    }
    except FileNotFoundError:
        # Create it and re-check
        build_outreach_strategy_csv(file_path)
        return get_best_outreach_window(persona, geography, club_type, file_path)
    except Exception as e:
        logger.error(f"Error reading outreach strategy: {str(e)}")

    # If no match
    default_strategy = {
        "Best Month": "January or September",
        "Best Time": "Mid-morning (9–11 AM)",
        "Best Day": "Tuesday or Thursday"
    }
    logger.warning("No matching strategy found, using defaults", extra={
        "role": persona,
        "club_type": club_type,
        "default_strategy": default_strategy
    })
    return default_strategy


if __name__ == "__main__":
    # Quick demonstration
    build_outreach_strategy_csv()
    test_window = get_best_outreach_window("General Manager", "Peak Summer Season", "Private Clubs")
    print("Example Best Outreach Window:", test_window)
