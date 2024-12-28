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

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
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

    # Recommended months for each geography
    months = {
        "Year-Round Golf": "January or September",
        "Peak Winter Season": "September",
        "Peak Summer Season": "February",
        "Short Summer Season": "January",
        "Shoulder Season Focus": "February or October"
    }

    # Best time of day for each persona
    times_of_day = {
        "General Manager": "Mid-morning (9–11 AM)",
        "Membership Director": "Early afternoon (1–3 PM)",
        "Food & Beverage Director": "Late morning (10–12 AM)"
    }

    # Best days of the week for each persona
    days_of_week = {
        "General Manager": "Tuesday or Thursday",
        "Membership Director": "Wednesday or Thursday",
        "Food & Beverage Director": "Wednesday or Thursday"
    }

    data = []
    for persona in personas:
        for geography in geographies:
            for club_type in club_types:
                data.append([
                    persona,
                    geography,
                    club_type,
                    months[geography],
                    times_of_day[persona],
                    days_of_week[persona]
                ])

    columns = [
        "Persona",
        "Geography/Golf Season",
        "Club Type",
        "Best Month to Start Outreach",
        "Best Time of Day",
        "Best Day of the Week"
    ]
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(file_path, index=False)
    logger.info(f"Outreach strategy CSV created or updated at: {file_path}")


def get_best_outreach_window(
    persona: str,
    geography: str,
    club_type: str,
    file_path: str = "/mnt/data/Golf_Outreach_Strategy.csv"
) -> dict:
    """
    Returns a dictionary with keys "Best Month", "Best Time", and "Best Day"
    from the CSV for the given persona, geography, and club type.

    Steps:
    1) Ensure the CSV table is built by calling build_outreach_strategy_csv().
    2) Load the CSV and filter to the single row matching the persona,
       geography, and club type. If none found, return defaults.
    """

    # Rebuild or update the CSV if missing
    build_outreach_strategy_csv(file_path=file_path)

    df = pd.read_csv(file_path)
    match = df[
        (df["Persona"] == persona) &
        (df["Geography/Golf Season"] == geography) &
        (df["Club Type"] == club_type)
    ]

    if match.empty:
        logger.warning(
            f"No matching outreach window found for {persona}, {geography}, {club_type}. Returning N/A."
        )
        return {
            "Best Month": "N/A",
            "Best Time": "N/A",
            "Best Day": "N/A"
        }

    row = match.iloc[0]
    return {
        "Best Month": row["Best Month to Start Outreach"],
        "Best Time": row["Best Time of Day"],
        "Best Day": row["Best Day of the Week"]
    }


if __name__ == "__main__":
    # Example usage when run directly
    build_outreach_strategy_csv()
    
    # Test getting outreach window for a specific combination
    test_persona = "General Manager"
    test_geography = "Peak Summer Season" 
    test_club_type = "Private Clubs"
    
    result = get_best_outreach_window(test_persona, test_geography, test_club_type)
    print(f"\nBest outreach window for {test_persona} at {test_club_type} in {test_geography}:")
    print(f"Month: {result['Best Month']}")
    print(f"Time: {result['Best Time']}")
    print(f"Day: {result['Best Day']}")
