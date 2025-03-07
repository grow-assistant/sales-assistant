# File: scripts/build_template.py
"""
Purpose: Builds email templates by combining various components and applying customizations.

Logical steps:
1. Define subject line templates and selection logic
2. Apply seasonal variations to email content
3. Generate personalized icebreakers based on news/context
4. Build complete outreach email by combining components
5. Handle template validation and fallback options
6. Process placeholders and customize content
"""

import os
import random
import traceback
from utils.doc_reader import DocReader
from utils.logging_setup import logger, setup_logging
from utils.season_snippet import get_season_variation_key, pick_season_snippet
from pathlib import Path
from config.settings import PROJECT_ROOT
from utils.xai_integration import (
    _build_icebreaker_from_news,
    _send_xai_request,
    xai_news_search,
    get_xai_icebreaker
)
from scripts.job_title_categories import categorize_job_title
from datetime import datetime
import re
from typing import Dict, Any
import json

###############################################################################
# 1) ROLE-BASED SUBJECT-LINE DICTIONARY
###############################################################################
SUBJECT_TEMPLATES = [
    "Question about 2025"
]

# Create a dedicated logger for template building
template_logger = setup_logging(
    log_name='template_builder',
    console_level='WARNING',  # Only show warnings and errors in console
    file_level='DEBUG',       # Log all details to file
    max_bytes=5242880        # 5MB
)

###############################################################################
# 2) PICK SUBJECT LINE BASED ON LEAD ROLE & LAST INTERACTION
###############################################################################
def pick_subject_line_based_on_lead(profile_type: str, placeholders: dict) -> str:
    """Choose a subject line from SUBJECT_TEMPLATES."""
    try:
        logger.debug("Selecting subject line from simplified templates")
        chosen_template = random.choice(SUBJECT_TEMPLATES)
        logger.debug(f"Selected template: {chosen_template}")

        # Normalize placeholder keys to match template format exactly
        normalized_placeholders = {
            "firstname": placeholders.get("firstname", ""),  # Match exact case
            "LastName": placeholders.get("lastname", ""),
            "companyname": placeholders.get("company_name", ""),
            "company_short_name": placeholders.get("company_short_name", ""),
            # Add other placeholders as needed
        }
        
        # Replace placeholders in the subject
        for key, val in normalized_placeholders.items():
            if val:  # Only replace if value exists
                placeholder = f"[{key}]"
                if placeholder in chosen_template:
                    chosen_template = chosen_template.replace(placeholder, str(val))
                    logger.debug(f"Replaced placeholder {placeholder} with {val}")
                
        if "[" in chosen_template or "]" in chosen_template:
            logger.warning(f"Unreplaced placeholders in subject: {chosen_template}")
            
        return chosen_template
        
    except Exception as e:
        logger.error(f"Error picking subject line: {str(e)}")
        return "Quick Question"  # Safe fallback


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

def generate_icebreaker(has_news: bool, club_name: str, news_text: str = None) -> str:
    """Generate an icebreaker based on news availability."""
    try:
        # Log parameters for debugging
        logger.debug(f"Icebreaker params - has_news: {has_news}, club_name: {club_name}, news_text: {news_text}")
        
        if not club_name.strip():
            logger.debug("No club name provided for icebreaker")
            return ""
            
        # Try news-based icebreaker first
        if has_news and news_text:
            news, icebreaker = xai_news_search(club_name)
            if icebreaker:
                return icebreaker
        
        # Fallback to general icebreaker if no news
        icebreaker = get_xai_icebreaker(
            club_name=club_name,
            recipient_name=""  # Leave blank as we don't have it at this stage
        )
        
        return icebreaker if icebreaker else ""
            
    except Exception as e:
        logger.debug(f"Error in icebreaker generation: {str(e)}")
        return ""

def build_outreach_email(
    template_path: str = None,
    profile_type: str = None,
    placeholders: dict = None,
    current_month: int = 9,
    start_peak_month: int = 5,
    end_peak_month: int = 8,
    use_markdown_template: bool = True
) -> tuple[str, str]:
    """Build email content from template."""
    try:
        placeholders = placeholders or {}
        
        template_logger.info(f"Building email for {profile_type}")
        template_logger.debug(f"Received placeholders: {placeholders}")
        
        if template_path and Path(template_path).exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                body = f.read().strip()
            
            # Log the initial template content
            template_logger.debug(f"Template content length: {len(body)}")
            template_logger.debug(f"Template preview: {body[:200]}...")
            
            # 1. Replace standard placeholders first
            standard_replacements = {
                "[firstname]": placeholders.get("firstname", ""),
                "[LastName]": placeholders.get("LastName", ""),
                "[companyname]": placeholders.get("companyname", ""),
                "[company_short_name]": placeholders.get("company_short_name", ""),
                "[JobTitle]": placeholders.get("JobTitle", ""),
                "[clubname]": placeholders.get("companyname", "")
            }
            
            # Log the replacements being made
            template_logger.debug("Standard replacements:")
            for key, value in standard_replacements.items():
                template_logger.debug(f"  {key}: '{value}'")
            
            for key, value in standard_replacements.items():
                if value:
                    body = body.replace(key, str(value))
                    template_logger.debug(f"Replaced {key} with '{value}'")

            # 1. Handle season variation first
            if "[SEASON_VARIATION]" in body:
                season_key = get_season_variation_key(
                    current_month=current_month,
                    start_peak_month=start_peak_month,
                    end_peak_month=end_peak_month
                )
                season_snippet = pick_season_snippet(season_key)
                body = body.replace("[SEASON_VARIATION]", season_snippet)
                template_logger.debug(f"Applied season variation: {season_snippet}")
            
            # 2. Handle icebreaker
            try:
                has_news = placeholders.get('has_news', False)
                news_result = placeholders.get('news_text', '')
                club_name = placeholders.get('clubname', '')
                
                template_logger.debug(f"Icebreaker params - has_news: {has_news}, club_name: {club_name}")
                
                if has_news and news_result and "has not been in the news" not in news_result.lower():
                    icebreaker = _build_icebreaker_from_news(club_name, news_result)
                    if icebreaker:
                        body = body.replace("[ICEBREAKER]", icebreaker)
                        template_logger.debug(f"Added news-based icebreaker: {icebreaker}")
                    else:
                        body = body.replace("[ICEBREAKER]\n\n", "")
                        body = body.replace("[ICEBREAKER]\n", "")
                        body = body.replace("[ICEBREAKER]", "")
                        template_logger.debug("Removed icebreaker placeholder (no content)")
                else:
                    body = body.replace("[ICEBREAKER]\n\n", "")
                    body = body.replace("[ICEBREAKER]\n", "")
                    body = body.replace("[ICEBREAKER]", "")
                    template_logger.debug("Removed icebreaker placeholder (no news)")
            except Exception as e:
                template_logger.error(f"Icebreaker generation error: {e}")
                body = body.replace("[ICEBREAKER]\n\n", "")
                body = body.replace("[ICEBREAKER]\n", "")
                body = body.replace("[ICEBREAKER]", "")
            
            # 3. Clean up multiple newlines
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")
            
            # 4. Replace remaining placeholders
            for key, value in placeholders.items():
                if value:
                    placeholder = f"[{key}]"
                    if placeholder in body:
                        body = body.replace(placeholder, str(value))
                        template_logger.debug(f"Replaced additional placeholder {placeholder} with {value}")
            
            # Add Byrdi to Swoop replacement
            body = body.replace("Byrdi", "Swoop")
                      
            # Clean up any double newlines
            while "\n\n\n" in body:
                body = body.replace("\n\n\n", "\n\n")
            
            # Get subject
            subject = pick_subject_line_based_on_lead(profile_type, placeholders)
            template_logger.debug(f"Selected subject: {subject}")
            
            # Remove signature as it's in the HTML template
            body = body.split("\n\nCheers,")[0].strip()
            
            if body:
                template_logger.info("Successfully built email template")
                template_logger.debug(f"Final email preview: {body[:200]}...")
            else:
                template_logger.error("Failed to build email template")
                
            return subject, body
            
    except Exception as e:
        template_logger.error(f"Error building email: {str(e)}")
        template_logger.error("Full traceback:", exc_info=True)
        return "", ""

def get_fallback_template() -> str:
    """Returns a basic fallback template if all other templates fail."""
    return """Connecting About Club Services
---
Hi [firstname],

I wanted to reach out about how we're helping clubs like [clubname] enhance their member experience through our comprehensive platform.

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
        
        # Add this line to call replace_placeholders
        body = replace_placeholders(body, parameters)
        logger.debug(f"After placeholder replacement - Preview: {body[:200]}")
        
        return body
        
    except Exception as e:
        logger.error(f"Error building email: {str(e)}")
        return ""

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

def get_template_path(club_type: str, role: str) -> str:
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
        
        # Randomly select sequence number (1 or 2)
        sequence_num = random.randint(1, 2)
        logger.debug(f"Selected template variation {sequence_num} for {template_name} (role: {role})")
        
        # Build template path
        template_path = os.path.join(
            PROJECT_ROOT,
            "docs",
            "templates",
            club_type,
            f"{template_name}_{sequence_num}.md"
        )
        
        logger.debug(f"Using template path: {template_path}")
        
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

def replace_placeholders(text: str, data: Dict[str, Any]) -> str:
    """Replace placeholders in template with actual values."""
    try:
        # Detailed logging of input text
        logger.debug("=" * 80)
        logger.debug("STARTING PLACEHOLDER REPLACEMENT")
        logger.debug(f"Original text length: {len(text)}")
        logger.debug(f"First 100 chars of text: {text[:100]}")
        logger.debug("=" * 80)
        
        # Log full data structure
        logger.debug("Input data structure:")
        logger.debug(json.dumps(data, indent=2, default=str))
        
        # Normalize data structure with detailed logging
        lead_data = data.get("lead_data", {}) if isinstance(data.get("lead_data"), dict) else data
        company_data = data.get("company_data", {}) if isinstance(data.get("company_data"), dict) else data
        
        logger.debug("Normalized data structures:")
        logger.debug(f"Lead data: {json.dumps(lead_data, indent=2, default=str)}")
        logger.debug(f"Company data: {json.dumps(company_data, indent=2, default=str)}")
        
        # Detailed company name logging
        company_short_name = company_data.get("company_short_name", "")
        company_full_name = company_data.get("name", "")
        logger.debug("=" * 80)
        logger.debug("COMPANY NAME DETAILS")
        logger.debug(f"company_short_name from company_data: '{company_short_name}'")
        logger.debug(f"company_short_name direct from data: '{data.get('company_short_name', '')}'")
        logger.debug(f"company_full_name: '{company_full_name}'")
        logger.debug("=" * 80)
        
        # Build replacements dict with detailed logging
        replacements = {
            "[firstname]": (lead_data.get("firstname", "") or 
                          lead_data.get("first_name", "") or 
                          data.get("firstname", "")),
            
            "[LastName]": (lead_data.get("lastname", "") or 
                          lead_data.get("last_name", "") or 
                          data.get("lastname", "")),
            
            "[companyname]": (company_data.get("name", "") or 
                             data.get("company_name", "") or 
                             company_full_name),
            
            "[company_short_name]": (company_short_name or 
                                   data.get("company_short_name", "") or 
                                   company_full_name),
            
            "[City]": company_data.get("city", ""),
            "[State]": company_data.get("state", "")
        }
        
        logger.debug("=" * 80)
        logger.debug("REPLACEMENT DICTIONARY")
        for key, value in replacements.items():
            logger.debug(f"{key}: '{value}'")
        logger.debug("=" * 80)
        
        # Do the replacements with detailed logging
        result = text
        for placeholder, value in replacements.items():
            if placeholder in result:
                logger.debug(f"Found '{placeholder}' in text, replacing with '{value}'")
                before_count = result.count(placeholder)
                result = result.replace(placeholder, value)
                after_count = result.count(placeholder)
                logger.debug(f"Replaced {before_count - after_count} instances")
            else:
                logger.debug(f"Placeholder '{placeholder}' not found in text")
        
        # Check for any remaining placeholders
        remaining = re.findall(r'\[([^\]]+)\]', result)
        if remaining:
            logger.warning(f"WARNING: Unreplaced placeholders found: {remaining}")
        
        logger.debug("=" * 80)
        logger.debug("FINAL RESULT")
        logger.debug(f"Final text length: {len(result)}")
        logger.debug(f"First 100 chars: {result[:100]}")
        logger.debug("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in replace_placeholders: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return text