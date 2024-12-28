import os
import requests
from typing import Tuple, Dict, Any
from utils.logging_setup import logger
from dotenv import load_dotenv
from config.settings import DEBUG_MODE
import json
import time

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")

# Add a simple cache
_news_cache = {}

def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends request to xAI API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            if DEBUG_MODE:
                logger.debug(f"xAI request payload={payload}")
            
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN
                },
                json=payload,
                timeout=15
            )
            
            if response.status_code == 429:  # Rate limit
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                logger.error(f"xAI API error ({response.status_code}): {response.text}")
                return ""
                
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            if DEBUG_MODE:
                logger.debug(f"xAI response={content}")
            return content
            
        except Exception as e:
            logger.error(f"Error in xAI request (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(retry_delay)
            
    return ""  # Return empty string if all retries fail

##############################################################################
# News Search + Icebreaker
##############################################################################

def xai_news_search(club_name: str) -> str:
    """
    Checks if a club is in the news with caching
    """
    if not club_name.strip():
        return ""
        
    # Check cache first
    if club_name in _news_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached news result for {club_name}")
        return _news_cache[club_name]

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise summaries of recent club news."
            },
            {
                "role": "user",
                "content": (
                    f"Tell me about any recent news for {club_name}. "
                    "If none exists, respond with 'has not been in the news.'"
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0
    }
    
    # Get the raw response from xAI
    response = _send_xai_request(payload)
    
    # Cache the response
    _news_cache[club_name] = response
    
    # Clean up awkward grammar in the response
    if response:
        # Fix the "Has [club] has not been" pattern
        if response.startswith("Has ") and " has not been in the news" in response:
            response = response.replace("Has ", "")
        
        # Fix any double "has" instances
        response = response.replace(" has has ", " has ")
    
    return response

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker referencing recent news.
    """
    if not club_name.strip() or not news_summary.strip():
        if DEBUG_MODE:
            logger.debug("Empty input passed to _build_icebreaker_from_news; returning blank.")
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sales copywriter. Create a natural, conversational "
                    "one-sentence opener mentioning recent club news."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Club: {club_name}\n"
                    f"News: {news_summary}\n\n"
                    "Write ONE engaging sentence that naturally references this news. "
                    "Avoid starting with phrases like 'I saw' or 'I noticed'."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.7
    }
    return _send_xai_request(payload)

##############################################################################
# Club Info Search (Used ONLY for Final Email Rewriting)
##############################################################################

def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> str:
    """
    Returns a short overview about the club's location and amenities.
    This is NOT used for icebreakers. We only use its result 
    to enhance context for final email rewriting.
    """
    if not club_name.strip():
        if DEBUG_MODE:
            logger.debug("Empty club_name passed to xai_club_info_search; returning blank.")
        return ""

    loc_str = location if location else "an unknown location"
    am_str = ", ".join(amenities) if amenities else "no specific amenities"
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides a brief overview of a club's location and amenities."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise overview about {club_name} in {loc_str}. "
                    f"Is it private or public? Keep it to under 3 sentences."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.0
    }
    return _send_xai_request(payload)
    
##############################################################################
# Personalize Email with Additional Club Info
##############################################################################

def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = "",
    club_info: str = ""
) -> Tuple[str, str]:
    """
    Use xAI to personalize an email's subject and body.
    
    Args:
        lead_sheet: Dictionary containing lead data
        subject: Original email subject
        body: Original email body
        summary: Summary of previous interactions
        news_summary: Recent news about the club
        club_info: Information about the club's facilities and features
    
    Returns:
        Tuple of (personalized_subject, personalized_body)
    """
    # Include club_info in your prompt construction
    prompt = f"""
    Lead Information: {json.dumps(lead_sheet)}
    Previous Interaction Summary: {summary}
    Club News: {news_summary}
    Club Information: {club_info}
    
    Original Subject: {subject}
    Original Body: {body}
    
    Please personalize this email while maintaining its core message...
    """

    if not lead_sheet:
        logger.warning("Empty lead_sheet passed to personalize_email_with_xai")
        return subject, body

    try:
        # Extract key information
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        facilities_info = lead_sheet.get("analysis", {}).get("facilities", {}).get("response", "")
        club_name = company_data.get("name", "").strip()
        
        # Build context for xAI
        context = f"Club Name: {club_name}\nFacilities Info: {facilities_info}\n"
        
        # If you have the lead interaction summary, add it:
        if summary:
            context += f"Lead Interaction Summary: {summary}\n"

        # Build the prompt
        user_content = (
            f"Original Subject: {subject}\n"
            f"Original Body: {body}\n\n"
            f"Context:\n{context}\n"
            "Instructions:\n"
            "1. Personalize based on verified club context and history.\n"
            "2. Focus on business value and problem-solving.\n" 
            "3. Keep core Swoop platform value proposition.\n"
            "4. Use brief, relevant facility references only if confirmed.\n"
            "5. Write at 6th-8th grade reading level.\n"
            "6. Keep paragraphs under 3 sentences.\n"
            "7. Maintain professional but helpful tone.\n"
            "8. Reference previous interactions naturally.\n"
            "9. If lead has replied to previous email, reference it naturally without direct acknowledgment.\n"
            "10. If lead expressed specific interests/concerns in reply, address them.\n"
            "Format the response as:\n"
            "Subject: [new subject]\n\n"
            "Body:\n[new body]"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at personalizing outreach emails for golf clubs. "
                        "You maintain a professional yet friendly tone and incorporate relevant context naturally. "
                        "You never mention unconfirmed facilities."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.7
        }

        if DEBUG_MODE:
            logger.debug("xAI request payload for personalization", extra={
                "club_name": club_name,
                "has_summary": bool(summary),
                "payload": payload
            })

        result = _send_xai_request(payload)
        
        if not result:
            logger.warning("Empty response from xAI personalization", extra={"club_name": club_name})
            return subject, body

        # Parse the response
        new_subject, new_body = _parse_xai_response(result)
        
        if not new_subject or not new_body:
            logger.warning("Failed to parse xAI response", extra={
                "club_name": club_name,
                "raw_response": result
            })
            return subject, body

        return new_subject, new_body

    except Exception as e:
        logger.error("Error in email personalization", extra={
            "error": str(e),
            "club_name": club_name if 'club_name' in locals() else 'unknown'
        })
        return subject, body

def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body with improved robustness.
    
    Args:
        response: Raw response from xAI
        
    Returns:
        Tuple[str, str]: Parsed subject and body
    """
    try:
        # Split response into lines
        lines = response.split('\n')
        subject = ""
        body_lines = []
        in_body = False
        
        # Find subject line
        for i, line in enumerate(lines):
            line = line.strip()
            if line.lower().startswith('subject:'):
                subject = line.split(':', 1)[1].strip()
                # Skip this line and start body from next line
                in_body = True
                continue
            
            if in_body:
                # Skip empty lines at the start of body
                if not line and not body_lines:
                    continue
                body_lines.append(lines[i])
        
        # If no explicit subject found, try to find it in the first non-empty line
        if not subject:
            for line in lines:
                if line.strip():
                    subject = line.strip()
                    break
        
        # Join body lines, removing signature if present
        body = '\n'.join(body_lines)
        
        # Clean up common signatures
        signatures = ['Cheers,', 'Best,', 'Swoop Golf', '480-225-9702', 'swoopgolf.com']
        for sig in signatures:
            if sig in body:
                body = body.split(sig)[0].strip()
        
        # Final cleanup
        body = body.strip()
        subject = subject.strip()
        
        if not subject or not body:
            raise ValueError("Failed to parse subject or body")
            
        return subject, body
        
    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}", extra={
            "raw_response": response
        })
        return "", ""

##############################################################################
# Facilities Check
##############################################################################

def xai_facilities_check(club_name: str, city: str, state: str) -> str:
    """
    Checks what facilities a club has with improved accuracy.
    """
    if not club_name.strip():
        logger.debug("Empty club_name passed to xai_facilities_check")
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides accurate facility information "
                    "for golf clubs and country clubs. Only confirm facilities that you are certain exist."
                    "If you are unsure about any facility, do not mention it."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise sentence about {club_name} in {city}, {state}, "
                    "mentioning only confirmed facilities like the number of holes for the golf course and "
                    "whether it's public, private, or semi-private. Omit any unconfirmed facilities."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0
    }
    
    try:
        response = _send_xai_request(payload)
        if not response:
            logger.error("Empty response from xAI facilities check", extra={
                "club_name": club_name,
                "city": city,
                "state": state
            })
            return "Facility information unavailable"
            
        return response
        
    except Exception as e:
        logger.error(f"Error in facilities check: {str(e)}", extra={
            "club_name": club_name,
            "city": city,
            "state": state
        })
        return "Facility information unavailable"