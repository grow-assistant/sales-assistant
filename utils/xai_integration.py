# utils/xai_integration.py

import os
import re
import json
import time
import random
import requests

from typing import Tuple, Dict, Any, List
from datetime import datetime, date
from dotenv import load_dotenv

from utils.logging_setup import logger
from config.settings import DEBUG_MODE

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.2"))

# Simple caches to avoid repeated calls
_cache = {
    "news": {},
    "club_segmentation": {},
    "icebreakers": {},
}

def clean_html(html_text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    import re
    from html import unescape
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    # Decode HTML entities
    clean_text = unescape(clean_text)
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

def get_email_rules() -> List[str]:
    """
    Returns the standardized list of rules for email personalization.
    """
    return [
        "# IMPORTANT: FOLLOW THESE RULES:\n",
        f"**Time Context:** Use relative date terms compared to Today's date of {date.today().strftime('%B %d, %Y')}.",
        "**Tone:** Professional but conversational, focusing on starting a dialogue.",
        "**Closing:** End emails directly after your call-to-action.",
        "**Previous Contact:** If no prior replies, do not reference previous emails or special offers.",
        "**Signature:** DO NOT include a signature block - this will be added later.",
    ]


def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends a request to the xAI API with retry logic.
    Logs request/response details for debugging.
    """
    TIMEOUT = 30
    logger.debug(
        "Full xAI Request Payload:",
        extra={
            "extra_data": {
                "request_details": {
                    "model": payload.get("model", MODEL_NAME),
                    "temperature": payload.get("temperature", EMAIL_TEMPERATURE),
                    "max_tokens": payload.get("max_tokens", 1000),
                    "messages": [
                        {"role": msg.get("role"), "content": msg.get("content")}
                        for msg in payload.get("messages", [])
                    ],
                }
            }
        },
    )

    for attempt in range(max_retries):
        try:
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN,
                },
                json=payload,
                timeout=TIMEOUT,
            )

            logger.debug(
                "Full xAI Response:",
                extra={
                    "extra_data": {
                        "response_details": {
                            "status_code": response.status_code,
                            "response_body": json.loads(response.text)
                            if response.text
                            else None,
                            "attempt": attempt + 1,
                            "headers": dict(response.headers),
                        }
                    }
                },
            )

            if response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time}s before retry"
                )
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(
                    f"xAI API error ({response.status_code}): {response.text}"
                )
                return ""

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                logger.debug(
                    "Received xAI response:\n%s",
                    content[:200] + "..." if len(content) > 200 else content,
                )
                return content

            except (KeyError, json.JSONDecodeError) as e:
                logger.error(
                    "Error parsing xAI response",
                    extra={
                        "error": str(e),
                        "response_text": response.text[:500],
                    },
                )
                return ""

        except Exception as e:
            logger.error(
                "xAI request failed",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "payload": payload,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return ""


##############################################################################
# News Search + Icebreaker
##############################################################################
def xai_news_search(club_name: str) -> tuple[str, str]:
    """
    Checks if a club is in the news and returns both news and icebreaker.
    Returns: Tuple of (news_summary, icebreaker)
    """
    if not club_name.strip():
        return "", ""

    if club_name in _cache["news"]:
        if DEBUG_MODE:
            logger.debug(f"Using cached news result for {club_name}")
        news = _cache["news"][club_name]
        icebreaker = _build_icebreaker_from_news(club_name, news)
        return news, icebreaker

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise summaries of recent club news.",
            },
            {
                "role": "user",
                "content": (
                    f"Tell me about any recent news for {club_name}. "
                    "If none exists, respond with 'has not been in the news.'"
                ),
            },
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0,
    }

    logger.info(f"Searching news for club: {club_name}")
    news = _send_xai_request(payload)
    logger.debug(f"News search result for {club_name}:")

    _cache["news"][club_name] = news

    if news:
        if news.startswith("Has ") and " has not been in the news" in news:
            news = news.replace("Has ", "")
        news = news.replace(" has has ", " has ")

    icebreaker = _build_icebreaker_from_news(club_name, news)
    return news, icebreaker


def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker if news is available.
    Returns an empty string if no relevant news found.
    """
    if (
        not club_name.strip()
        or not news_summary.strip()
        or "has not been in the news" in news_summary.lower()
    ):
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing from Swoop Golf's perspective, reaching out to golf clubs. "
                    "Create brief, natural-sounding icebreakers based on recent club news. "
                    "Keep the tone professional and focused on business value."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, natural-sounding icebreaker about {club_name} "
                    f"based on this news: {news_summary}\n\n"
                    "Requirements:\n"
                    "1. Single sentence only\n"
                    "2. Focus on business impact\n"
                    "3. No generic statements\n"
                    "4. Must relate to the news provided"
                ),
            },
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1,
    }

    logger.info(f"Building icebreaker for club: {club_name}")
    icebreaker = _send_xai_request(payload)
    logger.debug(f"Generated icebreaker for {club_name}:")

    return icebreaker


##############################################################################
# Personalize Email
##############################################################################
def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = "",
    context: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Personalizes email content using xAI.
    Returns a dictionary with 'subject' and 'body' keys.
    """
    try:
        # Ensure lead_sheet is a dictionary
        if not isinstance(lead_sheet, dict):
            logger.warning(f"Invalid lead_sheet type: {type(lead_sheet)}. Using empty dict.")
            lead_sheet = {}

        # Create a filtered company_data with only specific fields
        company_data = lead_sheet.get("company_data", {})
        allowed_fields = ['name', 'city', 'state', 'has_pool']
        filtered_company_data = {
            k: v for k, v in company_data.items() 
            if k in allowed_fields
        }
        
        # Update lead_sheet with filtered company data
        filtered_lead_sheet = {
            "lead_data": lead_sheet.get("lead_data", {}),
            "company_data": filtered_company_data,
            "analysis": lead_sheet.get("analysis", {})
        }
        
        # Use filtered_lead_sheet in the rest of the function
        previous_interactions = filtered_lead_sheet.get("analysis", {}).get("previous_interactions", {})
        has_prior_emails = bool(lead_sheet.get("lead_data", {}).get("emails", []))
        logger.debug(f"Has the lead previously emailed us? {has_prior_emails}")

        objection_handling = ""
        if has_prior_emails:
            with open("docs/templates/objection_handling.txt", "r") as f:
                objection_handling = f.read()
            logger.debug("Objection handling content loaded")
        else:
            logger.debug("Objection handling content not loaded (lead has not emailed us)")

        system_message = (
            "You are a helpful assistant that personalizes outreach emails for golf clubs, focusing on business value and relevant solutions. "
            "IMPORTANT: Do not include any signature block - this will be added later."
        )

        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_sheet.get("company_data", {})
        
        # Use provided context if available, otherwise build it
        if context is None:
            context_block = build_context_block(
                interaction_history=summary if summary else "No previous interactions",
                objection_handling=objection_handling if has_prior_emails else "",
                original_email={"subject": subject, "body": body},
                company_data=filtered_company_data
            )
        else:
            # Add filtered company data to existing context
            context.update({"company_data": filtered_company_data})
            context_block = context
            
        logger.debug(f"Context block: {json.dumps(context_block, indent=2)}")

        rules_text = "\n".join(get_email_rules())
        user_message = (
            "You are an expert at personalizing sales emails for golf industry outreach. "
            "CRITICAL RULES:\n"
            "1. DO NOT modify the subject line\n"
            "2. DO NOT reference weather or seasonal conditions unless specifically provided\n" 
            "3. DO NOT reference any promotions from previous emails\n"
            "4. Focus on business value and problem-solving aspects\n"
            "5. Avoid presumptive descriptions of club features\n"
            "6. Keep club references brief and relevant to the service\n"
            "7. Keep tone professional and direct\n"
            "8. ONLY modify the first paragraph of the email - leave the rest unchanged\n"
            "Format response as:\n"
            "Subject: [keep original subject]\n\n"
            "Body:\n[personalized body]\n\n"
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{rules_text}\n\n"
            "TASK:\n"
            "1. Focus on one key benefit relevant to the club\n"
            "2. Maintain professional tone\n"
            "3. Return ONLY the subject and body\n"
            "4. Only modify the first paragraph after the greeting - keep all other paragraphs exactly as provided"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "model": MODEL_NAME,
            "temperature": 0.3,
        }

        logger.info("Personalizing email for:")
        response = _send_xai_request(payload)
        logger.debug(f"Received xAI response:\n{response}")

        personalized_subject, personalized_body = _parse_xai_response(response)
        
        # Ensure we're returning strings, not dictionaries
        final_subject = personalized_subject if personalized_subject else subject
        final_body = personalized_body if personalized_body else body
        
        if isinstance(final_body, dict):
            final_body = final_body.get('body', body)
            
        # Replace Byrdi with Swoop in response
        if isinstance(final_body, str):
            final_body = final_body.replace("Byrdi", "Swoop")
        if isinstance(final_subject, str):
            final_subject = final_subject.replace("Byrdi", "Swoop")

        # Check for any remaining placeholders (for debugging)
        remaining_placeholders = check_for_placeholders(final_subject) + check_for_placeholders(final_body)
        if remaining_placeholders:
            logger.warning(f"Unreplaced placeholders found: {remaining_placeholders}")

        return {
            "subject": final_subject,
            "body": final_body
        }

    except Exception as e:
        logger.error(f"Error in email personalization: {str(e)}")
        return {
            "subject": subject,
            "body": body
        }


def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body.
    Handles various response formats consistently.
    """
    try:
        if not response:
            raise ValueError("Empty response received")

        lines = [line.strip() for line in response.split("\n") if line.strip()]

        subject = ""
        body_lines = []
        in_body = False

        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith("subject:"):
                subject = line.replace("Subject:", "", 1).strip()
            elif lower_line.startswith("body:"):
                in_body = True
            elif in_body:
                # Simple grouping into paragraphs/signature
                if line.startswith(("Hey", "Hi", "Dear")):
                    body_lines.append(f"{line}\n\n")
                else:
                    body_lines.append(f"{line}\n\n")

        body = "".join(body_lines)

        while "\n\n\n" in body:
            body = body.replace("\n\n\n", "\n\n")
        body = body.rstrip() + "\n"

        if not subject:
            subject = "Follow-up"

        logger.debug(
            f"Parsed result - Subject: {subject}, Body length: {len(body)}"
        )
        return subject, body

    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}")
        raise


def get_xai_icebreaker(club_name: str, recipient_name: str, timeout: int = 10) -> str:
    """Get a personalized icebreaker from the xAI service."""
    try:
        if not club_name.strip():
            logger.debug("Empty club name provided")
            return ""

        cache_key = f"icebreaker:{club_name}:{recipient_name}"
        if cache_key in _cache["icebreakers"]:
            logger.debug(f"Using cached icebreaker for {club_name}")
            return _cache["icebreakers"][cache_key]

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at creating icebreakers for golf club outreach.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a brief, natural-sounding icebreaker for {club_name}. "
                        "Keep it concise and professional."
                    ),
                },
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": ANALYSIS_TEMPERATURE,
        }

        logger.info(f"Generating icebreaker for club: {club_name}")
        response = _send_xai_request(payload, max_retries=3, retry_delay=1)
        
        if not response:
            logger.debug(f"Empty response from xAI for {club_name}")
            return ""
            
        # Clean and validate response
        icebreaker = response.strip()
        if not icebreaker:
            logger.debug(f"Empty icebreaker after cleaning for {club_name}")
            return ""

        logger.debug(f"Generated icebreaker for {club_name}: {icebreaker[:100]}")
        _cache["icebreakers"][cache_key] = icebreaker
        return icebreaker

    except Exception as e:
        logger.debug(f"Error generating icebreaker: {str(e)}")
        return ""

def xai_club_segmentation_search(club_name: str, location: str) -> Dict[str, Any]:
    """
    Returns a dictionary with the club's likely segmentation profile:
      - club_type
      - facility_complexity
      - geographic_seasonality
      - has_pool
      - has_tennis_courts
      - number_of_holes
      - analysis_text
      - company_short_name
    """
    if "club_segmentation" not in _cache:
        _cache["club_segmentation"] = {}

    cache_key = f"{club_name}_{location}"
    if cache_key in _cache["club_segmentation"]:
        logger.debug(f"Using cached segmentation result for {club_name} in {location}")
        return _cache["club_segmentation"][cache_key]

    logger.info(f"Searching for club segmentation info: {club_name} in {location}")

    prompt = f"""
Classify {club_name} in {location} with precision:

0. **OFFICIAL NAME**: What is the correct, official name of this facility?
1. **SHORT NAME**: Create a brief, memorable name by removing common terms like "Country Club", "Golf Club", "Golf Course", etc. Keep it under 100 characters.
2. **CLUB TYPE**: Is it Private, Public - High Daily Fee, Public - Low Daily Fee, Municipal, Resort, Country Club, or Unknown?
3. **FACILITY COMPLEXITY**: Single-Course, Multi-Course, or Unknown?
4. **GEOGRAPHIC SEASONALITY**: Year-Round or Seasonal?
5. **POOL**: ONLY answer 'Yes' if you find clear, direct evidence of a pool.
6. **TENNIS COURTS**: ONLY answer 'Yes' if there's explicit evidence.
7. **GOLF HOLES**: Verify from official sources or consistent user mentions.

CRITICAL RULES:
- **Do not assume amenities based on the type or perceived status of the club.**
- **Confirm amenities only with solid evidence; otherwise, use 'Unknown'.**
- **Use specific references for each answer where possible.**
- **For SHORT NAME: Keep it professional and recognizable while being concise.**

Format your response with these exact headings:
OFFICIAL NAME:
[Answer]

SHORT NAME:
[Answer]

CLUB TYPE:
[Answer]

FACILITY COMPLEXITY:
[Answer]

GEOGRAPHIC SEASONALITY:
[Answer]

POOL:
[Answer]

TENNIS COURTS:
[Answer]

GOLF HOLES:
[Answer]
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at segmenting golf clubs for marketing outreach. "
                    "CRITICAL: Only state amenities as present if verified with certainty. "
                    "Use 'Unknown' if not certain."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "model": MODEL_NAME,
        "temperature": 0.0,
    }

    response = _send_xai_request(payload)
    logger.debug(f"Club segmentation result for {club_name}:")

    parsed_segmentation = _parse_segmentation_response(response)
    _cache["club_segmentation"][cache_key] = parsed_segmentation
    return parsed_segmentation


def _parse_segmentation_response(response: str) -> Dict[str, Any]:
    """Parse the structured response from xAI segmentation search."""
    def clean_value(text: str) -> str:
        if "**Evidence**:" in text:
            text = text.split("**Evidence**:")[0]
        elif "- **Evidence**:" in text:
            text = text.split("- **Evidence**:")[0]
        return text.strip().split('\n')[0].strip()

    result = {
        'name': '',
        'company_short_name': '',
        'club_type': 'Unknown',
        'facility_complexity': 'Unknown',
        'geographic_seasonality': 'Unknown',
        'has_pool': 'Unknown',
        'has_tennis_courts': 'Unknown',
        'number_of_holes': 0,
        'analysis_text': ''
    }
    
    # Add name and short name detection patterns
    name_match = re.search(r'(?:OFFICIAL NAME|NAME):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    short_name_match = re.search(r'SHORT NAME:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    
    if name_match:
        result['name'] = clean_value(name_match.group(1))
    if short_name_match:
        result['company_short_name'] = clean_value(short_name_match.group(1))
    
    logger.debug(f"Raw segmentation response:\n{response}")
    
    sections = {
        'club_type': re.search(r'CLUB TYPE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'facility_complexity': re.search(r'FACILITY COMPLEXITY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'geographic_seasonality': re.search(r'GEOGRAPHIC SEASONALITY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'pool': re.search(r'(?:POOL|HAS POOL):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'tennis': re.search(r'(?:TENNIS COURTS|HAS TENNIS):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
        'holes': re.search(r'GOLF HOLES:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    }

    # Process club type with better matching
    if sections['club_type']:
        club_type = clean_value(sections['club_type'].group(1))
        logger.debug(f"Processing club type: '{club_type}'")
        
        club_type_lower = club_type.lower()
        
        # Handle combined types first
        if any(x in club_type_lower for x in ['private', 'semi-private']) and 'country club' in club_type_lower:
            result['club_type'] = 'Country Club'
        # Handle public course variations
        elif 'public' in club_type_lower:
            if 'high' in club_type_lower and 'daily fee' in club_type_lower:
                result['club_type'] = 'Public - High Daily Fee'
            elif 'low' in club_type_lower and 'daily fee' in club_type_lower:
                result['club_type'] = 'Public - Low Daily Fee'
            else:
                result['club_type'] = 'Public Course'
        # Then handle other types
        elif 'country club' in club_type_lower:
            result['club_type'] = 'Country Club'
        elif 'private' in club_type_lower:
            result['club_type'] = 'Private Course'
        elif 'resort' in club_type_lower:
            result['club_type'] = 'Resort Course'
        elif 'municipal' in club_type_lower:
            result['club_type'] = 'Municipal Course'
        elif 'semi-private' in club_type_lower:
            result['club_type'] = 'Semi-Private Course'
        elif 'management company' in club_type_lower:
            result['club_type'] = 'Management Company'

    # Keep existing pool detection
    if sections['pool']:
        pool_text = clean_value(sections['pool'].group(1)).lower()
        logger.debug(f"Found pool text in section: {pool_text}")
        if 'yes' in pool_text:
            result['has_pool'] = 'Yes'
            logger.debug("Pool found in standard section")

    # Add additional pool detection patterns
    if result['has_pool'] != 'Yes':  # Only check if we haven't found a pool yet
        pool_patterns = [
            r'AMENITIES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)',
            r'FACILITIES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)',
            r'(?:^|\n)-\s*(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$)',
            r'FEATURES:.*?(?:^|\s)(?:pool|swimming pool|pools|swimming pools|aquatic)(?:\s|$).*?(?=\n[A-Z ]+?:|$)'
        ]
        
        for pattern in pool_patterns:
            pool_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if pool_match:
                logger.debug(f"Found pool in additional pattern: {pool_match.group(0)}")
                result['has_pool'] = 'Yes'
                break

    # Process geographic seasonality
    if sections['geographic_seasonality']:
        seasonality = clean_value(sections['geographic_seasonality'].group(1)).lower()
        if 'year' in seasonality or 'round' in seasonality:
            result['geographic_seasonality'] = 'Year-Round'
        elif 'seasonal' in seasonality:
            result['geographic_seasonality'] = 'Seasonal'

    # Process number of holes with better validation
    if sections['holes']:
        holes_text = clean_value(sections['holes'].group(1)).lower()
        logger.debug(f"Processing holes text: '{holes_text}'")
        
        # First check for explicit mentions of multiple courses
        if 'three' in holes_text and '9' in holes_text:
            result['number_of_holes'] = 27
        elif 'two' in holes_text and '9' in holes_text:
            result['number_of_holes'] = 18
        elif '27' in holes_text:
            result['number_of_holes'] = 27
        elif '18' in holes_text:
            result['number_of_holes'] = 18
        elif '9' in holes_text:
            result['number_of_holes'] = 9
        else:
            # Try to extract any other number
            number_match = re.search(r'(\d+)', holes_text)
            if number_match:
                try:
                    result['number_of_holes'] = int(number_match.group(1))
                    logger.debug(f"Found {result['number_of_holes']} holes")
                except ValueError:
                    logger.warning(f"Could not convert {number_match.group(1)} to integer")

    # Process facility complexity
    if sections['facility_complexity']:
        complexity = clean_value(sections['facility_complexity'].group(1)).lower()
        logger.debug(f"Processing facility complexity: '{complexity}'")
        
        if 'single' in complexity or 'single-course' in complexity:
            result['facility_complexity'] = 'Single-Course'
        elif 'multi' in complexity or 'multi-course' in complexity:
            result['facility_complexity'] = 'Multi-Course'
        elif complexity and complexity != 'unknown':
            # Log unexpected values for debugging
            logger.warning(f"Unexpected facility complexity value: {complexity}")
            
    logger.debug(f"Parsed segmentation result: {result}")

    # Enhanced tennis detection
    tennis_found = False
    # First check standard TENNIS section
    if sections['tennis']:
        tennis_text = clean_value(sections['tennis'].group(1)).lower()
        logger.debug(f"Found tennis text: {tennis_text}")
        if 'yes' in tennis_text:
            result['has_tennis_courts'] = 'Yes'
            tennis_found = True
            logger.debug("Tennis courts found in standard section")
    
    # If no tennis found in standard section, check additional patterns
    if not tennis_found:
        tennis_patterns = [
            r'TENNIS COURTS:\s*(.+?)(?=\n[A-Z ]+?:|$)',
            r'HAS TENNIS:\s*(.+?)(?=\n[A-Z ]+?:|$)',
            r'AMENITIES:.*?(?:tennis|tennis courts?).*?(?=\n[A-Z ]+?:|$)'
        ]
        for pattern in tennis_patterns:
            tennis_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if tennis_match:
                tennis_text = clean_value(tennis_match.group(1) if tennis_match.groups() else tennis_match.group(0)).lower()
                logger.debug(f"Found additional tennis text: {tennis_text}")
                if any(word in tennis_text for word in ['yes', 'tennis']):
                    result['has_tennis_courts'] = 'Yes'
                    logger.debug("Tennis courts found in additional patterns")
                    break

    return result


def get_club_summary(club_name: str, location: str) -> str:
    """
    Get a one-paragraph summary of the club using xAI.
    """
    if not club_name or not location:
        return ""

    # Only get segmentation info
    segmentation_info = xai_club_segmentation_search(club_name, location)

    # Create system prompt based on verified info
    verified_info = {
        'type': segmentation_info.get('club_type', 'Unknown'),
        'holes': segmentation_info.get('number_of_holes', 0),
        'has_pool': segmentation_info.get('has_pool', 'No'),
        'has_tennis': segmentation_info.get('has_tennis_courts', 'No')
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a club analyst. Provide a factual one paragraph summary "
                    "based on verified information. Do not make assumptions."
                )
            },
            {
                "role": "user", 
                "content": f"Give me a one paragraph summary about {club_name} in {location}, "
                          f"using these verified facts: {verified_info}"
            }
        ],
        "model": MODEL_NAME,
        "temperature": 0.0,
    }

    response = _send_xai_request(payload)
    return response.strip()


def build_context_block(interaction_history=None, objection_handling=None, original_email=None, company_data=None):
    """Build context block for email personalization."""
    context = {}
    
    if interaction_history:
        context["interaction_history"] = interaction_history
        
    if objection_handling:
        context["objection_handling"] = objection_handling
        
    if original_email:
        context["original_email"] = original_email if isinstance(original_email, dict) else {
            "subject": original_email[0],
            "body": original_email[1]
        }
    
    if company_data:
        # Explicitly get company_short_name, fallback to name if not available
        company_short_name = company_data.get("company_short_name")
        if not company_short_name:
            # If no short name, try to create one from the full name
            full_name = company_data.get("name", "")
            # Take first 100 chars, strip trailing spaces and common suffixes
            company_short_name = re.sub(r'\s*(Golf Club|Country Club|Golf Course|Club|GC)\s*$', '', full_name[:100], flags=re.IGNORECASE)
        
        context["company_data"] = {
            "name": company_data.get("name", ""),
            "company_short_name": company_short_name,
            "city": company_data.get("city", ""),
            "state": company_data.get("state", ""),
            "has_pool": company_data.get("has_pool", "No"),
            "club_type": company_data.get("club_type", ""),
            "club_info": company_data.get("club_info", "")
        }
        
        logger.debug(f"Company short name used: {company_short_name}")
    
    return context


def check_for_placeholders(text: str) -> List[str]:
    """Check for any remaining placeholders in the text."""
    import re
    pattern = r'\[([^\]]+)\]'
    return re.findall(pattern, text)

def analyze_auto_reply(body: str, subject: str) -> Dict[str, str]:
    """
    Analyze auto-reply email to extract contact transition information.
    
    Args:
        body (str): The email body text
        subject (str): The email subject line
        
    Returns:
        Dict with structured contact transition information:
        {
            'original_person': str,
            'new_contact': str,
            'new_email': str,
            'new_title': str,
            'phone': str,
            'company': str,
            'reason': str,
            'permanent': str
        }
    """
    prompt = f"""
Analyze this auto-reply email and extract the following information with precision:

1. **ORIGINAL PERSON**: Who sent the auto-reply?
2. **NEW CONTACT**: Who is the new person to contact? If multiple people are listed, only use the first one mentioned.
3. **NEW EMAIL**: What is their new email address? If multiple emails are listed, only use the first one that matches the first new contact.
4. **NEW TITLE**: What is their job title/role?
5. **PHONE**: Any phone number provided?
6. **COMPANY**: Company name if mentioned
7. **REASON**: Why is the original person no longer available? (Retired, Left Company, etc.)
8. **PERMANENT**: Is this a permanent change? (Yes/No/Unknown)

CRITICAL RULES:
- Only extract information explicitly stated in the message
- Use 'Unknown' if information is not clearly provided
- Do not make assumptions
- For emails, only include if properly formatted (user@domain.com)
- If multiple contacts are listed, only extract info for the first person mentioned

Email Subject: {subject}
Email Body:
{body}

Format your response with these exact headings:
ORIGINAL PERSON:
[Answer]

NEW CONTACT:
[Answer]

NEW EMAIL:
[Answer]

NEW TITLE:
[Answer]

PHONE:
[Answer]

COMPANY:
[Answer]

REASON:
[Answer]

PERMANENT:
[Answer]
"""

    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing auto-reply emails and extracting "
                        "contact information changes. Only return verified information, "
                        "use 'Unknown' if not certain."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_NAME,
            "temperature": 0.0
        }

        response = _send_xai_request(payload)
        logger.debug(f"Raw xAI response for contact extraction:\n{response}")

        # Parse the response
        result = {
            'original_person': '',
            'new_contact': '',
            'new_email': '',
            'new_title': 'Unknown',
            'phone': 'Unknown',
            'company': 'Unknown',
            'reason': 'Unknown',
            'permanent': 'Unknown'
        }

        sections = {
            'original_person': re.search(r'ORIGINAL PERSON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_contact': re.search(r'NEW CONTACT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_email': re.search(r'NEW EMAIL:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_title': re.search(r'NEW TITLE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'phone': re.search(r'PHONE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'company': re.search(r'COMPANY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'reason': re.search(r'REASON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'permanent': re.search(r'PERMANENT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
        }

        for key, match in sections.items():
            if match:
                result[key] = match.group(1).strip().split('\n')[0].strip()

        # Validate email format
        if result['new_email'] and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', result['new_email']):
            result['new_email'] = ''

        logger.debug(f"Parsed contact info: {result}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing auto-reply: {str(e)}")
        return None

def analyze_employment_change(body: str, subject: str) -> Dict[str, str]:
    """
    Analyze an employment change notification email to extract relevant information.
    Similar to analyze_auto_reply but specifically focused on employment changes.
    """
    try:
        # Clean HTML if present
        if '<html>' in body:
            body = clean_html(body)

        prompt = (
            "Analyze this employment change notification email and extract the following information. "
            "Use 'Unknown' if information is not clearly stated.\n\n"
            "Email Subject: " + subject + "\n"
            "Email Body: " + body + "\n\n"
            "Extract and format the information as follows:\n"
            "ORIGINAL PERSON:\n[Name of person who left]\n\n"
            "NEW CONTACT:\n[Name of new contact person]\n\n"
            "NEW EMAIL:\n[Email of new contact]\n\n"
            "NEW TITLE:\n[Title of new contact]\n\n"
            "PHONE:\n[Phone number]\n\n"
            "COMPANY:\n[Company name]\n\n"
            "REASON:\n[Reason for change]\n\n"
            "PERMANENT:\n[Is this a permanent change? Yes/No/Unknown]"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing employment change notifications "
                        "and extracting contact information changes. Only return verified "
                        "information, use 'Unknown' if not certain."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_NAME,
            "temperature": 0.0
        }

        response = _send_xai_request(payload)
        logger.debug(f"Raw xAI response for employment change analysis:\n{response}")

        # Parse the response using the same structure as analyze_auto_reply
        result = {
            'original_person': '',
            'new_contact': '',
            'new_email': '',
            'new_title': 'Unknown',
            'phone': 'Unknown',
            'company': 'Unknown',
            'reason': 'Unknown',
            'permanent': 'Unknown'
        }

        sections = {
            'original_person': re.search(r'ORIGINAL PERSON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_contact': re.search(r'NEW CONTACT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_email': re.search(r'NEW EMAIL:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'new_title': re.search(r'NEW TITLE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'phone': re.search(r'PHONE:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'company': re.search(r'COMPANY:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'reason': re.search(r'REASON:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL),
            'permanent': re.search(r'PERMANENT:\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
        }

        for key, match in sections.items():
            if match:
                result[key] = match.group(1).strip().split('\n')[0].strip()

        # Validate email format
        if result['new_email'] and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', result['new_email']):
            result['new_email'] = ''

        logger.debug(f"Parsed employment change info: {result}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing employment change: {str(e)}")
        return None