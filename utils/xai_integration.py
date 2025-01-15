# utils/xai_integration.py

import os
import re
import json
import time
import random
import requests

from typing import Tuple, Dict, Any, List
from datetime import datetime, date
from pathlib import Path
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

SUBJECT_TEMPLATES = [
    "Quick Chat, [FirstName]?",
    "Quick Question, [FirstName]?",
    "Question about 2025",
    "Quick Question",
]


def get_random_subject_template() -> str:
    """Returns a random subject line template from the predefined list."""
    return random.choice(SUBJECT_TEMPLATES)


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

                logger.info(
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
    logger.info(
        "News search result for %s:",
        club_name,
        extra={"news": news},
    )

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
    logger.info(
        "Generated icebreaker for %s:",
        club_name,
        extra={"icebreaker": icebreaker},
    )

    return icebreaker


##############################################################################
# Personalize Email
##############################################################################
def personalize_email_with_xai(
    lead_sheet: Dict[str, Any],
    subject: str,
    body: str,
    summary: str = "",
    news_summary: str = ""
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

        # Create a filtered company_data without club_info
        company_data = lead_sheet.get("company_data", {})
        filtered_company_data = {
            k: v for k, v in company_data.items() 
            if k not in ['club_info', 'overview']  # Add any other fields you want to exclude
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
            "You are a helpful assistant that personalizes outreach emails for golf clubs, focusing on business value and relevant solutions."
        )

        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_sheet.get("company_data", {})
        
        context_block = build_context_block(
            interaction_history=summary if summary else "No previous interactions",
            objection_handling=objection_handling if has_prior_emails else "",
            original_email={"subject": subject, "body": body},
        )
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
            "8. Keep emails concise - under 200 words\n"
            "9. Focus on one key value proposition\n"
            "10. End with a clear call-to-action\n\n"
            "Required Email Structure:\n"
            "1. Start with 'Hey [FirstName],'\n"
            "2. Quick intro paragraph\n"
            "3. Value proposition paragraph\n"
            "4. CTA paragraph\n"
            "5. Signature\n\n"
            "Format response as:\n"
            "Subject: [keep original subject]\n\n"
            "Body:\n[personalized body]\n\n"
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{rules_text}\n\n"
            "TASK:\n"
            "1. Follow the required email structure exactly\n"
            "2. Focus on one key benefit relevant to the club\n"
            "3. Maintain professional tone\n"
            "4. Keep response under 200 words\n"
            "5. Return ONLY the subject and body"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "model": MODEL_NAME,
            "temperature": EMAIL_TEMPERATURE,
        }

        logger.info(
            "Personalizing email for:",
            extra={
                "company": lead_sheet.get("company_name"),
                "original_subject": subject,
            },
        )
        response = _send_xai_request(payload)
        logger.info("Email personalization result:", extra={
            "company": lead_sheet.get("company_name"),
            "response": response
        })

        personalized_subject, personalized_body = _parse_xai_response(response)
        
        # Ensure we're returning strings, not dictionaries
        final_subject = personalized_subject if personalized_subject else subject
        final_body = personalized_body if personalized_body else body
        
        if isinstance(final_body, dict):
            final_body = final_body.get('body', body)
            
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
                elif line in ["Best regards,", "Sincerely,", "Regards,"]:
                    body_lines.append(f"\n{line}")
                elif line == "Ty":
                    body_lines.append(f" {line}\n\n")
                elif line == "Swoop Golf":
                    body_lines.append(f"{line}\n")
                elif line == "480-225-9702":
                    body_lines.append(f"{line}\n")
                elif line == "swoopgolf.com":
                    body_lines.append(line)
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
    """
    Get a personalized icebreaker from the xAI service (with caching if desired).
    """
    cache_key = f"icebreaker:{club_name}:{recipient_name}"
    if cache_key in _cache["icebreakers"]:
        if DEBUG_MODE:
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
    logger.info(
        "Generated icebreaker for %s:",
        club_name,
        extra={"icebreaker": response},
    )

    _cache["icebreakers"][cache_key] = response
    return response


def generate_followup_email_content(
    first_name: str,
    company_name: str,
    original_subject: str,
    original_date: str,
    sequence_num: int,
    original_email: dict = None,
) -> Tuple[str, str]:
    """
    Generate follow-up email content using xAI.
    Returns (subject, body).
    """
    logger.debug(
        f"[generate_followup_email_content] Called with first_name='{first_name}', "
        f"company_name='{company_name}', original_subject='{original_subject}', "
        f"original_date='{original_date}', sequence_num={sequence_num}, "
        f"original_email keys={list(original_email.keys()) if original_email else 'None'}"
    )
    try:
        if sequence_num == 2 and original_email:
            logger.debug("[generate_followup_email_content] Handling second follow-up logic.")

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a sales professional writing a brief follow-up email. "
                            "Keep the tone professional but friendly. "
                            "The response should be under 50 words and focus on getting a response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Original Email:
                        Subject: {original_email['subject']}
                        Body: {original_email['body']}

                        Write a brief follow-up that starts with:
                        "I understand my last email might not have made it high enough in your inbox."

                        Requirements:
                        - Keep it under 50 words
                        - Be concise and direct
                        - End with a clear call to action
                        - Don't repeat information from the original email
                        """,
                    },
                ],
                "model": MODEL_NAME,
                "stream": False,
                "temperature": EMAIL_TEMPERATURE,
            }

            logger.info(
                "Generating second follow-up email for:",
                extra={"company": company_name, "sequence_num": sequence_num},
            )
            result = _send_xai_request(payload)
            logger.info("Second follow-up generation result:", extra={"result": result})

            if not result:
                logger.error(
                    "[generate_followup_email_content] Empty response from xAI for follow-up generation."
                )
                return "", ""

            follow_up_body = result.strip()
            subject = f"RE: {original_email['subject']}"
            body = (
                f"{follow_up_body}\n\n"
                f"Best regards,\n"
                f"Ty\n\n"
                f"-------- Original Message --------\n"
                f"Subject: {original_email['subject']}\n"
                f"Sent: {original_email.get('created_at', 'Unknown date')}\n\n"
                f"{original_email['body']}"
            )

            logger.debug(
                f"[generate_followup_email_content] Returning second follow-up subject='{subject}', "
                f"body length={len(body)}"
            )
            return subject, body

        else:
            logger.debug("[generate_followup_email_content] Handling default follow-up logic.")
            prompt = f"""
            Generate a follow-up email for:
            - Name: {first_name}
            - Company: {company_name}
            - Original Email Subject: {original_subject}
            - Original Send Date: {original_date}

            This is follow-up #{sequence_num}. Keep it brief and focused on scheduling a call.

            Rules:
            1. Keep it under 3 short paragraphs
            2. Reference the original email naturally
            3. Add new value proposition or insight
            4. End with clear call to action
            5. Maintain professional but friendly tone

            Format the response with 'Subject:' and 'Body:' labels.
            """

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at writing follow-up emails that are brief, "
                            "professional, and effective."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "model": MODEL_NAME,
                "temperature": EMAIL_TEMPERATURE,
            }

            logger.info(
                "Generating follow-up email for:",
                extra={"company": company_name, "sequence_num": sequence_num},
            )
            result = _send_xai_request(payload)
            logger.info("Follow-up generation result:", extra={"result": result})

            if not result:
                logger.error(
                    "[generate_followup_email_content] Empty response from xAI for default follow-up generation."
                )
                return "", ""

            subject, body = _parse_xai_response(result)
            if not subject or not body:
                logger.error(
                    "[generate_followup_email_content] Failed to parse follow-up email content."
                )
                return "", ""

            logger.debug(
                f"[generate_followup_email_content] Returning subject='{subject}', body length={len(body)}"
            )
            return subject, body

    except Exception as e:
        logger.error(
            f"[generate_followup_email_content] Error generating follow-up email content: {str(e)}"
        )
        return "", ""


def parse_personalization_response(response_text):
    """
    Parse JSON-based email personalization response text from xAI.
    Returns (subject, body).
    Falls back to defaults on parsing error.
    """
    try:
        response_data = json.loads(response_text)
        subject = response_data.get("subject")
        body = response_data.get("body")

        if not subject or not body:
            raise ValueError("Subject or body missing in xAI response")

        return subject, body

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception(f"Error parsing xAI response: {str(e)}")
        return "Follow-up", (
            "Thank you for your interest. Let me know if you have any other questions!"
        )


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
1. **CLUB TYPE**: Is it Private, Public - High Daily Fee, Public - Low Daily Fee, Municipal, Resort, Country Club, or Unknown?
2. **FACILITY COMPLEXITY**: Single-Course, Multi-Course, or Unknown?
3. **GEOGRAPHIC SEASONALITY**: Year-Round or Seasonal?
4. **POOL**: ONLY answer 'Yes' if you find clear, direct evidence of a pool.
5. **TENNIS COURTS**: ONLY answer 'Yes' if there's explicit evidence.
6. **GOLF HOLES**: Verify from official sources or consistent user mentions.

CRITICAL RULES:
- **Do not assume amenities based on the type or perceived status of the club.**
- **Confirm amenities only with solid evidence; otherwise, use 'Unknown'.**
- **Use specific references for each answer where possible.**

Format your response with these exact headings:
OFFICIAL NAME:
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
    logger.info(
        "Club segmentation result for %s:",
        club_name,
        extra={"segmentation": response},
    )

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
        'club_type': 'Unknown',
        'facility_complexity': 'Unknown',
        'geographic_seasonality': 'Unknown',
        'has_pool': 'Unknown',
        'has_tennis_courts': 'Unknown',
        'number_of_holes': 0,
        'analysis_text': ''
    }
    
    # Add name detection pattern
    name_match = re.search(r'(?:OFFICIAL NAME|NAME):\s*(.+?)(?=\n[A-Z ]+?:|$)', response, re.DOTALL)
    if name_match:
        result['name'] = clean_value(name_match.group(1))
    
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


def build_context_block(interaction_history=None, objection_handling=None, original_email=None):
    context = {}
    
    if interaction_history:
        context["interaction_history"] = interaction_history
        
    if objection_handling:
        context["objection_handling"] = objection_handling
        
    if original_email:
        context["original_email"] = {
            "subject": original_email.get("subject", ""),
            "body": original_email.get("body", "")
        }
    
    return context