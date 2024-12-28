# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path
from config.settings import PROJECT_ROOT
from utils.xai_integration import get_xai_icebreaker

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
    current_month: int,
    start_peak_month: int,
    end_peak_month: int,
    use_markdown_template: bool = True
) -> tuple[str, str]:
    """
    Builds an outreach email with enhanced error handling and debugging
    """
    try:
        # Log input parameters for debugging
        logger.debug(
            "Building outreach email",
            extra={
                'profile_type': profile_type,
                'placeholders': placeholders,
                'template_used': use_markdown_template
            }
        )
        
        template_dir = PROJECT_ROOT / 'docs' / 'templates'
        template_map = {
            'general_manager': 'general_manager_initial_outreach.md',
            'food_beverage': 'fb_manager_initial_outreach.md',
            'golf_professional': 'golf_ops_initial_outreach.md',
            'owner': 'owner_initial_outreach.md',
            'membership': 'membership_director_initial_outreach.md'
        }
        
        template_file = template_map.get(profile_type, 'general_manager_initial_outreach.md')
        template_path = template_dir / template_file
        
        # Log template selection
        logger.debug(f"Selected template: {template_path}")
        
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                logger.debug(f"Template loaded, length: {len(template_content)}")
        else:
            logger.warning(f"Template not found: {template_path}, using fallback")
            template_content = get_fallback_template()
        
        # Parse template with error tracking
        try:
            template_data = parse_template(template_content)
            subject = template_data['subject']
            body = template_data['body']
        except Exception as e:
            logger.error(f"Template parsing failed: {str(e)}")
            raise
        
        # Track placeholder replacements
        replacement_log = []
        
        # Replace placeholders with logging
        for key, value in placeholders.items():
            if value is None:
                logger.warning(f"Missing value for placeholder: {key}")
                value = ''
            
            # Track replacements for debugging
            if f'[{key}]' in subject or f'[{key}]' in body:
                replacement_log.append(f"Replaced [{key}] with '{value}'")
            
            subject = subject.replace(f'[{key}]', str(value))
            body = body.replace(f'[{key}]', str(value))
            
            if key == 'SEASON_VARIATION':
                body = body.replace('{SEASON_VARIATION}', str(value))
                body = body.replace('[SEASON_VARIATION]', str(value))
        
        # Log all replacements made
        if replacement_log:
            logger.debug("Placeholder replacements: " + "; ".join(replacement_log))
        
        # Handle icebreaker with detailed logging
        if '[ICEBREAKER]' in body:
            try:
                icebreaker = get_xai_icebreaker(
                    placeholders.get('ClubName', ''),
                    placeholders.get('FirstName', '')
                )
                if icebreaker:
                    logger.debug(f"Using xAI icebreaker: {icebreaker}")
                    body = body.replace('[ICEBREAKER]', icebreaker)
                else:
                    logger.warning("Using fallback icebreaker")
                    body = body.replace('[ICEBREAKER]', "I hope this email finds you well.")
            except Exception as e:
                logger.error(f"Icebreaker generation failed: {str(e)}")
                body = body.replace('[ICEBREAKER]', "I hope this email finds you well.")

        return subject, body

    except Exception as e:
        logger.error(
            "Email building failed",
            extra={
                'error': str(e),
                'profile_type': profile_type,
                'template_file': template_file if 'template_file' in locals() else None
            }
        )
        return get_fallback_template().split('---\n', 1)

def get_fallback_template() -> str:
    """Returns a basic fallback template if all other templates fail."""
    return """Connecting About Club Services
---
Hi [FirstName],

I wanted to reach out about how we're helping clubs like [ClubName] enhance their member experience through our comprehensive platform.

Would you be open to a brief conversation to explore if our solution might be a good fit for your needs?

Best regards,
[YourName]
Swoop Golf
480-225-9702
swoopgolf.com"""

def validate_template(template_content):
    """Validate template format and structure"""
    if not template_content:
        raise ValueError("Template content cannot be empty")
    
    # Basic validation that template has some content
    lines = template_content.strip().split('\n')
    if len(lines) < 2:  # At least need greeting and body
        raise ValueError("Template must have sufficient content")
    
    return True

def build_template(template_path):
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
            
        # Validate template before processing
        validate_template(template_content)
        
        # Rest of the template building logic...
        # ...
    except Exception as e:
        logger.error(f"Error building template from {template_path}: {str(e)}")
        raise

def parse_template(template_content):
    """Parse template content without requiring YAML frontmatter"""
    lines = template_content.strip().split('\n')
    
    # Initialize template parts
    template_body = []
    subject = None
    
    for line in lines:
        # Look for subject in first few lines if not found yet
        if not subject and line.startswith('subject:'):
            subject = line.replace('subject:', '').strip()
            continue
        template_body.append(line)
    
    # If no explicit subject found, use a default or first line
    if not subject:
        subject = "Quick update from Swoop Golf"
    
    return {
        'subject': subject,
        'body': '\n'.join(template_body)
    }

def build_email(template_path, parameters):
    """Build email from template and parameters"""
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    template_data = parse_template(template_content)
    
    # Replace parameters in both subject and body
    subject = template_data['subject']
    body = template_data['body']
    
    for key, value in parameters.items():
        subject = subject.replace(f'[{key}]', str(value))
        body = body.replace(f'[{key}]', str(value))
        # Handle season variation differently since it uses curly braces
        if key == 'SEASON_VARIATION':
            body = body.replace('{SEASON_VARIATION}', str(value))
    
    return {
        'subject': subject,
        'body': body
    }