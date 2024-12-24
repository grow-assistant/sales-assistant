# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
# Each key (e.g., "general_manager", "fnb_manager") corresponds to a lead role.
# Updated to include the new subject lines provided.
###############################################################################

CONDITION_SUBJECTS = {
    "general_manager": [
        "Question for [FirstName]",
        "Quick Question",
        "Can I help with [specific goal/pain point]?",
        "Quick Question for [FirstName]",
        "Quick Question, [FirstName]",
        "Need Assistance with [specific challenge]?"
    ],
    "fnb_manager": [
        "Question for [FirstName]",
        "Quick Question",
        "Can I help with [specific goal/pain point]?",
        "Quick Question for [FirstName]",
        "Quick Question, [FirstName]",
        "Need Assistance with [specific challenge]?"
    ],
    # If no matching role, use these fallback subject lines:
    "fallback": [
        "Question for [FirstName]",
        "Quick Question",
        "Can I help with [specific goal/pain point]?",
        "Quick Question for [FirstName]",
        "Quick Question, [FirstName]",
        "Need Assistance with [specific challenge]?"
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

    # 3) Placeholder replacement
    for key, val in placeholders.items():
        chosen_template = chosen_template.replace(f"[{key}]", val)

    return chosen_template


###############################################################################
# 3) SEASON VARIATION LOGIC (OPTIONAL)
###############################################################################

def get_season_variation_key(current_month, start_peak_month, end_peak_month):
    """
    Example from original code:
    0–2 months away from start => approaching
    within start-end => in_season
    if less than 1 month left => winding_down
    else => off_season
    """
    if start_peak_month <= current_month <= end_peak_month:
        if (end_peak_month - current_month) < 1:
            return "winding_down"
        else:
            return "in_season"

    months_away = start_peak_month - current_month
    if months_away < 0:
        months_away += 12

    if 1 <= months_away <= 2:
        return "approaching"
    else:
        return "off_season"


def pick_season_snippet(season_key):
    snippet_options = {
        "approaching": [
            "As you prepare for the upcoming season,",
            "With the season just around the corner,"
        ],
        "in_season": [
            "I hope the season is going well for you,",
            "With the season in full swing,"
        ],
        "winding_down": [
            "As the season winds down,",
            "As your peak season comes to a close,"
        ],
        "off_season": [
            "While planning for the upcoming season,",
            "As you look forward to next season,"
        ]
    }
    if season_key not in snippet_options:
        return ""
    return random.choice(snippet_options[season_key])


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
    subject_line = pick_subject_line_based_on_lead(
        profile_type, last_interaction_days, placeholders
    )

    md_body = ""
    if use_markdown_template:
        template_map = {
            "general_manager": "templates/gm_initial_outreach.md",
            "fnb_manager": "templates/fnb_initial_outreach.md",
            "golf_ops": "templates/golf_ops_initial_outreach.md",
        }
        file_path = template_map.get(profile_type)
        if file_path:
            reader = DocReader()
            logger.debug(f"Attempting to load markdown template: {file_path}")
            md_content = reader.read_doc(file_path, fallback_content="")
            if md_content.strip():
                md_subject, md_body = extract_subject_and_body(md_content)
                logger.debug(
                    f"Markdown template loaded successfully. "
                    f"Parsed subject=[{md_subject.strip()}], body length={len(md_body)}"
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

    if not md_body.strip():
        logger.warning("Markdown body is empty. Using default fallback body.")
        md_body = (
            "Hello [FirstName],\n\n"
            "{SEASON_VARIATION} I'd love to connect on a quick call to "
            "discuss how Swoop can streamline operations at [ClubName]. "
            "Let me know if you have a few minutes this week.\n\n"
            "Best,\nYour Name"
        )

    # Replace placeholders in the body
    for key, val in placeholders.items():
        md_body = md_body.replace(f"[{key}]", val)

    # Insert the season snippet
    season_key = get_season_variation_key(current_month, start_peak_month, end_peak_month)
    snippet = pick_season_snippet(season_key)
    final_body = apply_season_variation(md_body, snippet)

    return subject_line, final_body


if __name__ == "__main__":
    # Example placeholders
    placeholders_demo = {
        "FirstName": "Taylor",
        "ClubName": "Pinetree CC",
        "DeadlineDate": "Oct 15th",
        "Role": "F&B Manager",
        "Task": "Staff Onboarding",
        "Topic": "On-Course Ordering"
    }

    subject, body = build_outreach_email(
        profile_type="general_manager",
        last_interaction_days=75,
        placeholders=placeholders_demo,
        current_month=9,
        start_peak_month=5,
        end_peak_month=8,
        use_markdown_template=False
    )

    print("Subject:", subject)
    print("Body:\n", body)
