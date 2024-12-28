# utils/xai_integration.py

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
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))

# Simple caches to avoid repeated calls
_news_cache = {}
_club_info_cache = {}

def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends request to xAI API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            if DEBUG_MODE:
                # Log only the essential parts of the payload
                logger.debug("xAI request", extra={
                    "attempt": attempt + 1,
                    "model": payload.get("model"),
                    "temperature": payload.get("temperature")
                })

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
                wait_time = retry_delay * (attempt + 1)  # Simple backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(f"xAI API error ({response.status_code}): {response.text}")
                return ""

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            if DEBUG_MODE:
                # Log only length of response instead of full content
                logger.debug("xAI response received", extra={
                    "content_length": len(content),
                    "attempt": attempt + 1
                })
            return content

        except Exception as e:
            logger.error(f"Error in xAI request (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return ""  # Return empty string if all retries fail

##############################################################################
# News Search + Icebreaker
##############################################################################

def xai_news_search(club_name: str) -> str:
    """
    Checks if a club is in the news (with caching).
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

    response = _send_xai_request(payload)

    # Cache the response
    _news_cache[club_name] = response

    # Clean up awkward grammar if needed
    if response:
        if response.startswith("Has ") and " has not been in the news" in response:
            response = response.replace("Has ", "")
        response = response.replace(" has has ", " has ")

    return response

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker if news is available.
    """
    if not club_name.strip() or not news_summary.strip() \
       or "has not been in the news" in news_summary.lower():
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are writing from Swoop Golf's perspective, reaching out to golf clubs about our technology."
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, natural-sounding icebreaker about {club_name} "
                    f"based on this news: {news_summary}. Keep it concise and professional."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1
    }
    return _send_xai_request(payload)

def get_default_icebreaker(club_name: str) -> str:
    """
    Generate a generic icebreaker if no news is found.
    """
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are writing from Swoop Golf's perspective, reaching out to golf clubs."
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, professional icebreaker for {club_name}. "
                    "Focus on improving their operations and member experience. Keep it concise."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1
    }
    return _send_xai_request(payload)

##############################################################################
# Club Info Search
##############################################################################

def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> str:
    """
    Returns a short overview about the club's location and amenities.
    """
    cache_key = f"{club_name}:{location}"

    # Check cache first
    if cache_key in _club_info_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached club info for {club_name}")
        return _club_info_cache[cache_key]

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
                "content": "You are a helpful assistant that provides a brief overview of a club, it's location, and it's amenities."
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
    response = _send_xai_request(payload)


    # Cache the response
    _club_info_cache[cache_key] = response

    return response

##############################################################################
# Personalize Email
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
    """
    if not lead_sheet:
        logger.warning("Empty lead_sheet passed to personalize_email_with_xai")
        return subject, body

    try:
        # Extract data for additional context
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        facilities_info = lead_sheet.get("analysis", {}).get("facilities", {}).get("response", "")
        club_name = company_data.get("name", "").strip()

        # Build context
        context = f"Club Name: {club_name}\nFacilities Info: {facilities_info}\n"
        if summary:
            context += f"Lead Interaction Summary: {summary}\n"

        # Build user prompt
        user_content = (
            f"Original Subject: {subject}\n"
            f"Original Body: {body}\n\n"
            f"Context:\n{context}\n"
            "Instructions:\n"
            "IMPORTANT: YOU MUST FOLLOW ALL RULES BELOW EXACTLY.\n\n"
            "1. Personalize based on verified club context and history.\n"
            "2. Use brief, relevant facility references only if confirmed.\n"
            "3. Write at 6th-8th grade reading level.\n"
            "4. Keep paragraphs under 3 sentences.\n"
            "5. Maintain professional but helpful tone.\n"
            "6. Reference previous interactions naturally.\n"
            "7. If lead has replied, reference it carefully.\n"
            "8. Avoid generic or cliché references (e.g., seasons, local scenery).\n"
            "9. If lead expressed specific concerns in replies, address them.\n"
            "Format:\n"
            "Subject: [new subject]\n\n"
            "Body:\n[new body]"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at personalizing outreach emails for golf clubs. "
                        "Maintain a professional yet friendly tone and incorporate relevant context naturally. "
                        "Never mention unconfirmed facilities."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.3
        }

        if DEBUG_MODE:
            logger.debug("xAI request payload for personalization", extra={
                "club_name": club_name,
                "has_summary": bool(summary),
                "model": MODEL_NAME,
                "temperature": ANALYSIS_TEMPERATURE
            })

        result = _send_xai_request(payload)
        if not result:
            logger.warning("Empty response from xAI personalization", extra={"club_name": club_name})
            return subject, body

        # Parse the AI response
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
            "lead_sheet": lead_sheet
        })
        return subject, body


def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the AI response into subject and body, ensuring we capture
    all lines after "Body:".
    Expects a format like:
      Subject: ...
      
      Body:
      ...
    """
    try:
        lines = response.splitlines()
        subject = ""
        body_lines = []
        in_body = False

        for line in lines:
            stripped_line = line.strip()

            # Identify the subject line
            if stripped_line.lower().startswith("subject:"):
                subject = stripped_line.split(":", 1)[1].strip()
                continue

            # Identify the start of the body
            if stripped_line.lower().startswith("body:"):
                in_body = True
                continue

            # If we're in the body section, collect *all* lines (blank or not)
            if in_body:
                body_lines.append(line)

        # If we never found "Subject:" in the text, fallback:
        if not subject:
            # Use the first non-empty line as subject or fallback
            for line in lines:
                if line.strip():
                    subject = line.strip()
                    break
            if not subject:
                subject = "No Subject Provided"

        # Combine all body lines (including any blank lines)
        body = "\n".join(body_lines).rstrip("\n")

        # Final cleanup
        subject = subject.strip()
        body = body.strip()

        # If something is still empty, raise
        if not subject or not body:
            raise ValueError("Failed to parse subject or body from xAI response")

        return subject, body

    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}", extra={
            "raw_response": response
        })
        return "", ""

def get_xai_icebreaker(club_name: str, recipient_name: str, timeout: int = 10) -> str:
    """
    Get a personalized icebreaker from the xAI service (with caching if desired).
    """
    cache_key = f"icebreaker:{club_name}:{recipient_name}"

    if cache_key in _news_cache:
        if DEBUG_MODE:
            logger.debug(f"Using cached icebreaker for {club_name}")
        return _news_cache[cache_key]

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at creating icebreakers for golf club outreach."
            },
            {
                "role": "user",
                "content": (
                    f"Create a brief, natural-sounding icebreaker for {club_name}. "
                    "Keep it concise and professional."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": ANALYSIS_TEMPERATURE
    }

    response = _send_xai_request(payload, max_retries=3, retry_delay=1)
    _news_cache[cache_key] = response
    return response
