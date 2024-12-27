# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
CONDITION_SUBJECTS = {
    "general_manager": [
        "Quick Question for [FirstName]",
        "New Ways to Elevate [ClubName]'s Operations",
        "Boost [ClubName]'s Efficiency with Swoop",
        "Need Assistance with [Task]? – [FirstName]"
    ],
    "fnb_manager": [
        "Ideas for Increasing F&B Revenue at [ClubName]",
        "Quick Note for [FirstName] about On-Demand Service",
        "A Fresh Take on [ClubName]'s F&B Operations"
    ],
    "golf_ops": [
        "Keeping [ClubName] Rounds on Pace: Quick Idea",
        "New Golf Ops Tools for [ClubName]",
        "Quick Question for [FirstName] – On-Course Efficiency"
    ],
    # New line: If job title doesn't match any known category,
    # we map it to this fallback template
    "fallback": [
        "Enhancing Your Club's Efficiency with Swoop",
        "Is [ClubName] Looking to Modernize?"
    ]
}


###############################################################################
# 2) PICK SUBJECT LINE BASED ON LEAD ROLE & LAST INTERACTION
###############################################################################
def pick_subject_line_based_on_lead(
    lead_role: str,
    last_interaction_days: int,
    placeholders: dict
) -> str:
    """
    Choose a subject line from CONDITION_SUBJECTS based on the lead role
    and the days since last interaction. Then replace placeholders.
    """
    # 1) Decide which subject lines to use based on role
    if lead_role in CONDITION_SUBJECTS:
        subject_variations = CONDITION_SUBJECTS[lead_role]
    else:
        subject_variations = CONDITION_SUBJECTS["fallback"]

    # 2) Example condition: if lead is "older" than 60 days, pick the first subject
    #    otherwise pick randomly.
    if last_interaction_days > 60:
        chosen_template = subject_variations[0]
    else:
        chosen_template = random.choice(subject_variations)

    # 3) Replace placeholders in the subject
    for key, val in placeholders.items():
        chosen_template = chosen_template.replace(f"[{key}]", val)

    return chosen_template


###############################################################################
# 3) SEASON VARIATION LOGIC (OPTIONAL)
###############################################################################
def apply_season_variation(email_text: str, snippet: str) -> str:
    """
    Replaces {SEASON_VARIATION} in an email text with the chosen snippet.
    """
    return email_text.replace("{SEASON_VARIATION}", snippet)


###############################################################################
# 4) OPTION: READING AN .MD TEMPLATE (BODY ONLY)
###############################################################################
def extract_subject_and_body(md_text: str) -> tuple[str, str]:
    subject = ""
    body_lines = []
    mode = None

    for line in md_text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("## Subject"):
            mode = "subject"
            continue
        elif line_stripped.startswith("## Body"):
            mode = "body"
            continue

        if mode == "subject":
            subject += line + " "
        elif mode == "body":
            body_lines.append(line)

    return subject.strip(), "\n".join(body_lines).strip()


###############################################################################
# 5) MAIN FUNCTION FOR BUILDING EMAIL
###############################################################################
def build_outreach_email(
    profile_type: str,
    last_interaction_days: int,
    placeholders: dict,
    current_month: int = 3,
    start_peak_month: int = 4,
    end_peak_month: int = 7,
    use_markdown_template: bool = False
) -> tuple[str, str]:
    """
    1) Conditionally pick a subject line
    2) Optionally load a .md template for the email body
    3) Insert a season snippet into the body if desired
    4) Return (subject, body)
    """
    # First, pick the subject line
    subject_line = pick_subject_line_based_on_lead(
        profile_type, last_interaction_days, placeholders
    )

    md_body = ""
    if use_markdown_template:
        # Attempt to match a .md file based on profile_type
        template_map = {
            "general_manager": "templates/gm_initial_outreach.md",
            "fnb_manager": "templates/fnb_initial_outreach.md",
            "golf_ops": "templates/golf_ops_initial_outreach.md",
            # NEW: Provide a fallback path for unrecognized roles
            "fallback": "templates/fallback.md"
        }
        file_path = template_map.get(profile_type)
        if file_path:
            reader = DocReader()
            logger.debug(f"Attempting to load markdown template: {file_path}")
            md_content = reader.read_doc(file_path, fallback_content="")
            if md_content.strip():
                md_subject, md_body = extract_subject_and_body(md_content)
                logger.debug(
                    "Markdown template loaded successfully.",
                    extra={
                        "parsed_subject": md_subject.strip(),
                        "body_length": len(md_body)
                    }
                )
            else:
                logger.warning(
                    f"No content found in {file_path}. Using fallback body..."
                )
        else:
            logger.warning(
                f"No .md file mapped for profile_type='{profile_type}'. Using fallback body..."
            )
    else:
        logger.debug("use_markdown_template=False, using inline fallback body.")

    # If we couldn't load a .md body, use an inline fallback
    if not md_body.strip():
        logger.warning("Markdown body is empty. Using default fallback body.")
        md_body = (
            "Hey [FirstName],\n\n"
            "{SEASON_VARIATION} I'd like to discuss how Swoop can significantly "
            "enhance your club's operational efficiency. Our solutions are designed to:\n\n"
            "- Automate booking and scheduling to reduce administrative workload.\n"
            "- Improve member engagement through personalized communications.\n"
            "- Optimize resource management for better cost control.\n\n"
            "Could we schedule a brief call this week to explore how these benefits "
            "could directly address your club's specific needs?\n\n"
            "Best,\n[YourName]"
        )

    # Replace placeholders in the body
    for key, val in placeholders.items():
        md_body = md_body.replace(f"[{key}]", str(val))

    # Insert the season snippet
    season_key = get_season_variation_key(current_month, start_peak_month, end_peak_month)
    snippet = pick_season_snippet(season_key)
    final_body = md_body.replace("{SEASON_VARIATION}", snippet)

    return subject_line, final_body
