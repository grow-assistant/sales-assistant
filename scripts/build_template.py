# scripts/build_template.py

import os
import random
from utils.doc_reader import DocReader
from utils.logging_setup import logger
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path
from config.settings import PROJECT_ROOT
from utils.xai_integration import (
    _build_icebreaker_from_news,
    _send_xai_request
)
from scripts.job_title_categories import categorize_job_title
from datetime import datetime

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
SUBJECT_TEMPLATES = [
    "Quick Chat, [FirstName]?",
    "Quick Question, [FirstName]?",
    "Question about 2025",
    "Quick Question"
]


###############################################################################
# 2) PICK SUBJECT LINE BASED ON LEAD ROLE & LAST INTERACTION
###############################################################################
def pick_subject_line_based_on_lead(
    profile_type: str,
    placeholders: dict
) -> str:
    """
    Choose a subject line from SUBJECT_TEMPLATES.
    """
    # Debug logging
    logger.debug("Selecting subject line from simplified templates")
    
    # Pick randomly from available templates
    chosen_template = random.choice(SUBJECT_TEMPLATES)
    logger.debug(f"Selected template: {chosen_template}")

    # Replace placeholders in the subject
    for key, val in placeholders.items():
        if val:  # Only replace if value exists
            chosen_template = chosen_template.replace(f"[{key}]", str(val))
            logger.debug(f"Replaced placeholder [{key}] with {val}")

    return chosen_template


###############################################################################
# 3) SEASON VARIATION LOGIC (OPTIONAL)
###############################################################################
def apply_season_variation(email_text: str, snippet: str) -> str:
    """Replaces {SEASON_VARIATION} in an email text with the chosen snippet."""
    logger.debug("Applying season variation:", extra={
        "original_length": len(email_text),
        "snippet_length": len(snippet),
        "has_placeholder": "{SEASON_VARIATION}" in email_text
    })
    
    result = email_text.replace("{SEASON_VARIATION}", snippet)
    
    logger.debug("Season variation applied:", extra={
        "final_length": len(result),
        "successful": result != email_text
    })
    
    return result


###############################################################################
# 4) OPTION: READING AN .MD TEMPLATE (BODY ONLY)
###############################################################################
def extract_subject_and_body(template_content: str) -> tuple[str, str]:
    """
    Extract body from template content, treating the entire content as body.
    Subject will be handled separately via CONDITION_SUBJECTS.
    """
    try:
        # Clean up the template content
        body = template_content.strip()
        
        logger.debug(f"Template content length: {len(template_content)}")
        logger.debug(f"Cleaned body length: {len(body)}")
        
        if len(body) == 0:
            logger.warning("Warning: Template body is empty")
            
        # Return empty subject since it's handled elsewhere
        return "", body
        
    except Exception as e:
        logger.error(f"Error processing template: {e}")
        return "", ""


###############################################################################
# 5) MAIN FUNCTION FOR BUILDING EMAIL
###############################################################################
def build_outreach_email(
    template_path: str = None,
    profile_type: str = None,
    last_interaction_days: int = None,
    placeholders: dict = None,
    current_month: int = 9,  # Default from main_old.py
    start_peak_month: int = 5,  # Default from main_old.py
    end_peak_month: int = 8,  # Default from main_old.py
    use_markdown_template: bool = True
) -> tuple[str, str]:  # Return tuple like main_old.py expects
    """Build email content from template."""
    try:
        placeholders = placeholders or {}
        
        print(f"Starting build_outreach_email with template_path: {template_path}")
        print(f"Placeholders: {placeholders}")
        
        if template_path and Path(template_path).exists():
            print(f"Template file exists at {template_path}")
            with open(template_path, 'r', encoding='utf-8') as f:
                body = f.read().strip()
            print(f"Raw template body length: {len(body)}")
            
            # 1. Handle season variation first
            if "[SEASON_VARIATION]" in body:
                print("Found [SEASON_VARIATION] placeholder")
                season_key = get_season_variation_key(
                    current_month=current_month,
                    start_peak_month=start_peak_month,
                    end_peak_month=end_peak_month
                )
                logger.debug(f"Generated season key: {season_key}")
                print(f"Season key: {season_key}")
                
                season_snippet = pick_season_snippet(season_key)
                logger.debug(f"Selected season snippet: {season_snippet}")
                print(f"Season snippet: {season_snippet}")
                
                body = body.replace("[SEASON_VARIATION]", season_snippet)
                logger.debug(f"Applied season variation: {season_snippet}")
                print("Applied season variation")
            
            # 2. Handle icebreaker exactly like main_old.py
            try:
                has_news = placeholders.get('has_news', False)
                news_result = placeholders.get('news_text', '')
                club_name = placeholders.get('ClubName', '')
                
                print(f"Icebreaker params - has_news: {has_news}, club_name: {club_name}")
                print(f"News result length: {len(news_result)}")
                
                if has_news and news_result and "has not been in the news" not in news_result.lower():
                    icebreaker = _build_icebreaker_from_news(club_name, news_result)
                    if icebreaker:
                        print(f"Generated icebreaker: {icebreaker}")
                        body = body.replace("[ICEBREAKER]", icebreaker)
                    else:
                        print("No icebreaker generated, removing placeholder")
                        body = body.replace("[ICEBREAKER]\n\n", "")
                        body = body.replace("[ICEBREAKER]\n", "")
                        body = body.replace("[ICEBREAKER]", "")
                else:
                    print("No valid news, removing icebreaker placeholder")
                    body = body.replace("[ICEBREAKER]\n\n", "")
                    body = body.replace("[ICEBREAKER]\n", "")
                    body = body.replace("[ICEBREAKER]", "")
            except Exception as e:
                logger.error(f"Icebreaker generation error: {e}")
                print(f"Error generating icebreaker: {e}")
                body = body.replace("[ICEBREAKER]\n\n", "")
                body = body.replace("[ICEBREAKER]\n", "")
                body = body.replace("[ICEBREAKER]", "")
            
            # 3. Clean up multiple newlines
            print("Cleaning up multiple newlines")
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")
            
            # 4. Replace remaining placeholders
            print("Replacing remaining placeholders")
            for key, value in placeholders.items():
                if value:
                    print(f"Replacing [{key}] with {value}")
                    body = body.replace(f"[{key}]", str(value))
            
            # Add Byrdi to Swoop replacement
            body = body.replace("Byrdi", "Swoop")
            print("Replaced 'Byrdi' with 'Swoop' in email body")
            
            # 5. Get subject (will be ignored by main_old.py but included for completeness)
            subject = pick_subject_line_based_on_lead(profile_type, placeholders)
            print(f"Generated subject line: {subject}")
            
            print(f"Final email body length: {len(body)}")
            return subject, body
            
    except Exception as e:
        logger.error(f"Error building email: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        print(f"Error in build_outreach_email: {e}")
        return "", ""  # Return empty strings on error

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

def validate_template(template_content: str) -> bool:
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

def get_xai_icebreaker(club_name: str, recipient_name: str) -> str:
    """
    Get personalized icebreaker from xAI with proper error handling and debugging
    """
    try:
        # Log the request parameters
        logger.debug(f"Requesting xAI icebreaker for club: {club_name}, recipient: {recipient_name}")
        
        # Create the payload for xAI request
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are writing from Swoop Golf's perspective, reaching out to golf clubs about our technology platform."
                },
                {
                    "role": "user",
                    "content": f"Create a brief, professional icebreaker for {club_name}. Focus on improving their operations and member experience. Keep it concise."
                }
            ],
            "model": "grok-2-1212",
            "stream": False,
            "temperature": 0.1
        }
        
        # Use _send_xai_request directly with recipient email
        response = _send_xai_request(payload, timeout=10)
        
        if not response:
            raise ValueError("Empty response from xAI service")
            
        cleaned_response = response.strip()
        
        if len(cleaned_response) < 10:
            raise ValueError(f"Response too short ({len(cleaned_response)} chars)")
            
        if '[' in cleaned_response or ']' in cleaned_response:
            logger.warning("Response contains unresolved template variables")
        
        if cleaned_response.lower().startswith(('hi', 'hello', 'dear')):
            logger.warning("Response appears to be a full greeting instead of an icebreaker")
            
        return cleaned_response
        
    except Exception as e:
        logger.warning(
            "Failed to get xAI icebreaker",
            extra={
                'error': str(e),
                'club_name': club_name,
                'recipient_name': recipient_name
            }
        )
        return "I wanted to reach out about enhancing your club's operations"  # Fallback icebreaker

def parse_template(template_content):
    """Parse template content - subject lines are handled separately via CONDITION_SUBJECTS"""
    logger.debug(f"Parsing template content of length: {len(template_content)}")
    lines = template_content.strip().split('\n')
    logger.debug(f"Template contains {len(lines)} lines")
    
    # Just return the body content directly
    result = {
        'subject': None,  # Subject will be set from CONDITION_SUBJECTS
        'body': template_content.strip()
    }
    
    logger.debug(f"Parsed template - Body length: {len(result['body'])}")
    return result

def build_email(template_path, parameters):
    """Build email from template and parameters"""
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
    
        template_data = parse_template(template_content)
        
        # Get the body text
        body = template_data['body']
        
        # Replace parameters in body
        for key, value in parameters.items():
            if value:  # Only replace if value is not None/empty
                body = body.replace(f'[{key}]', str(value))
                # Handle season variation differently since it uses curly braces
                if key == 'SEASON_VARIATION':
                    body = body.replace('{SEASON_VARIATION}', str(value))
        
        return body  # Return just the body text, not a dictionary
        
    except Exception as e:
        logger.error(f"Error building email: {str(e)}")
        return ""  # Return empty string on error

def extract_template_body(template_content: str) -> str:
    """Extract body from template content."""
    try:
        # Simply clean up the template content
        body = template_content.strip()
        
        logger.debug(f"Extracted body length: {len(body)}")
        if len(body) == 0:
            logger.warning("Warning: Extracted body is empty")
        
        return body
        
    except Exception as e:
        logger.error(f"Error extracting body: {e}")
        return ""

def process_template(template_path):
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.debug(f"Raw template content length: {len(content)}")
            return extract_template_body(content)
    except Exception as e:
        logger.error(f"Error reading template file: {e}")
        return ""

def get_template_path(club_type: str, role: str, sequence_num: int = 1) -> str:
    """Get the appropriate template path based on club type and role."""
    try:
        # Normalize inputs
        club_type = club_type.lower().strip().replace(" ", "_")
        role_category = categorize_job_title(role)
        
        # Map role categories to template names
        template_map = {
            "fb_manager": "fb_manager_initial_outreach",
            "membership_director": "membership_director_initial_outreach",
            "golf_operations": "golf_operations_initial_outreach",
            "general_manager": "general_manager_initial_outreach"
        }
        
        # Get template name or default to general manager
        template_name = template_map.get(role_category, "general_manager_initial_outreach")
        
        # Build template path
        template_path = os.path.join(
            PROJECT_ROOT,
            "docs",
            "templates",
            club_type,
            f"{template_name}_{sequence_num}.md"
        )
        
        logger.debug(f"Selected template path: {template_path} for role: {role} (category: {role_category})")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template not found: {template_path}, falling back to general manager template")
            template_path = os.path.join(
                PROJECT_ROOT,
                "docs",
                "templates",
                club_type,
                f"general_manager_initial_outreach_{sequence_num}.md"
            )
        
        return template_path
        
    except Exception as e:
        logger.error(f"Error getting template path: {str(e)}")
        # Fallback to general manager template
        return os.path.join(
            PROJECT_ROOT,
            "docs",
            "templates",
            "country_club",
            f"general_manager_initial_outreach_{sequence_num}.md"
        )