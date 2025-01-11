# utils/xai_integration.py

import os
import requests
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, date
from utils.logging_setup import logger
from dotenv import load_dotenv
from config.settings import DEBUG_MODE
import json
import time
from pathlib import Path
import random
import re

load_dotenv()

XAI_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
XAI_BEARER_TOKEN = f"Bearer {os.getenv('XAI_TOKEN', '')}"
MODEL_NAME = os.getenv("XAI_MODEL", "grok-2-1212")
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.2"))

# Simple caches to avoid repeated calls
_cache = {
    'news': {},
    'club_info': {},
    'icebreakers': {}
}

SUBJECT_TEMPLATES = [
    "Quick Chat, [FirstName]?",
    "Quick Question, [FirstName]?",
    "Swoop: [ClubName]'s Edge?",
    "Question about 2025",
    "Quick Question"
]

def get_random_subject_template() -> str:
    """Returns a random subject line template from the predefined list"""
    return random.choice(SUBJECT_TEMPLATES)

def get_email_rules() -> List[str]:
    """
    Returns the standardized list of rules for email personalization.
    """
    return [
        "# IMPORTANT: FOLLOW THESE RULES:\n",
        "**Amenities:** ONLY reference amenities that are explicitly listed in club_details. Do not assume or infer any additional amenities.",
        f"**Personalization:** Use only verified information from club_details.",
        f"**Time Context:** Use relative date terms compared to Todays date to {date.today().strftime('%B %d, %Y')}.",
        "**Tone:** Professional but conversational, focusing on starting a dialogue.",
        "**Closing:** End emails directly after your call-to-action.",
        "**Previous Contact:** If no prior replies, do not reference previous emails or special offers.",
    ]

def _send_xai_request(payload: dict, max_retries: int = 3, retry_delay: int = 1) -> str:
    """
    Sends request to xAI API with retry logic.
    """
    TIMEOUT = 30

    # Log the full payload with complete messages
    logger.debug("Full xAI Request Payload:", extra={
        'extra_data': {
            'request_details': {
                'model': payload.get('model', MODEL_NAME),
                'temperature': payload.get('temperature', EMAIL_TEMPERATURE),
                'max_tokens': payload.get('max_tokens', 2000),
                'messages': [
                    {
                        'role': msg.get('role'),
                        'content': msg.get('content')  # No truncation
                    } 
                    for msg in payload.get('messages', [])
                ]
            }
        }
    })

    for attempt in range(max_retries):
        try:
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN
                },
                json=payload,
                timeout=TIMEOUT
            )

            # Log complete response
            logger.debug("Full xAI Response:", extra={
                'extra_data': {
                    'response_details': {
                        'status_code': response.status_code,
                        'response_body': json.loads(response.text) if response.text else None,
                        'attempt': attempt + 1,
                        'headers': dict(response.headers)
                    }
                }
            })

            if response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(f"xAI API error ({response.status_code}): {response.text}")
                return ""

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Log successful response
                logger.info("Received xAI response:\n%s", content[:200] + "..." if len(content) > 200 else content)
                
                return content

            except (KeyError, json.JSONDecodeError) as e:
                logger.error("Error parsing xAI response", extra={
                    "error": str(e),
                    "response_text": response.text[:500]
                })
                return ""

        except Exception as e:
            logger.error("xAI request failed", extra={
                'extra_data': {
                    'error': str(e),
                    'attempt': attempt + 1,
                    'max_retries': max_retries,
                    'payload': payload
                }
            })
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

    # Check cache first
    if club_name in _cache['news']:
        if DEBUG_MODE:
            logger.debug(f"Using cached news result for {club_name}")
        news = _cache['news'][club_name]
        icebreaker = _build_icebreaker_from_news(club_name, news)
        return news, icebreaker

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

    logger.info(f"Searching news for club: {club_name}")
    news = _send_xai_request(payload)
    logger.info(f"News search result for {club_name}:", extra={"news": news})

    _cache['news'][club_name] = news

    # Clean up awkward grammar if needed
    if news:
        if news.startswith("Has ") and " has not been in the news" in news:
            news = news.replace("Has ", "")
        news = news.replace(" has has ", " has ")

    # Only build icebreaker if we have news
    icebreaker = _build_icebreaker_from_news(club_name, news)
    
    return news, icebreaker

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
    """
    Build a single-sentence icebreaker if news is available.
    Returns empty string if no relevant news found.
    """
    if not club_name.strip() or not news_summary.strip() \
       or "has not been in the news" in news_summary.lower():
        return ""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are writing from Swoop Golf's perspective, reaching out to golf clubs. "
                    "Create brief, natural-sounding icebreakers based on recent club news. "
                    "Keep the tone professional and focused on business value."
                )
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
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.1
    }

    logger.info(f"Building icebreaker for club: {club_name}")
    icebreaker = _send_xai_request(payload)
    logger.info(f"Generated icebreaker for {club_name}:", extra={"icebreaker": icebreaker})
    
    return icebreaker

##############################################################################
# Club Info Search
##############################################################################

def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> Dict[str, Any]:
    """
    Search for club information using xAI.
    
    Args:
        club_name: Name of the club
        location: Club location 
        amenities: Optional list of known amenities
        
    Returns:
        Dict containing parsed club information:
        {
            'overview': str,  # Full response text
            'facility_type': str,  # Classified facility type
            'has_pool': str,  # Yes/No
            'amenities': List[str]  # Extracted amenities
        }
    """
    cache_key = f"{club_name}_{location}"
    if cache_key in _cache['club_info']:
        logger.debug(f"Using cached club info for {club_name} in {location}")
        return _cache['club_info'][cache_key]

    logger.info(f"Searching for club info: {club_name} in {location}")
    amenity_str = ", ".join(amenities) if amenities else ""
    
    prompt = f"""
    Please provide a brief overview of {club_name} located in {location}. Include key facts such as:
    - Type of facility (Public, Private, Municipal, Semi-Private, Country Club, Resort, Management Company)
    - Does the club have a pool? (Answer with 'Yes' or 'No')
    - Notable amenities or features (DO NOT include pro shop, fitness center, or dining facilities)
    - Any other relevant information
    
    Format your response with these exact headings:
    OVERVIEW:
    [Overview text]
    
    FACILITY TYPE:
    [Type]
    
    HAS POOL:
    [Yes/No]
    
    AMENITIES:
    - [amenity 1]
    - [amenity 2]
    """

    payload = {
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are a factual assistant that provides objective, data-focused overviews of clubs. "
                    "Focus only on verifiable facts like location, type, amenities, etc. "
                    "CRITICAL: DO NOT list golf course or pro shop as amenities as these are assumed. "
                    "Only list additional amenities that are explicitly verified. "
                    "Avoid subjective descriptions or flowery language."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": MODEL_NAME,
        "temperature": ANALYSIS_TEMPERATURE
    }

    response = _send_xai_request(payload)
    logger.info(f"Club info search result for {club_name}:", extra={"info": response})
    
    # Parse structured response
    parsed_info = _parse_club_response(response)
    
    # Cache the response
    _cache['club_info'][cache_key] = parsed_info
    
    return parsed_info

def _parse_club_response(response: str) -> Dict[str, Any]:
    """
    Parse structured club information response.
    
    Args:
        response: Raw response text from xAI
        
    Returns:
        Dict containing parsed information
    """
    result = {
        'overview': '',
        'facility_type': 'Other',
        'has_pool': 'No',
        'amenities': []
    }
    
    # Extract sections using regex
    sections = {
        'overview': re.search(r'OVERVIEW:\s*(.+?)(?=FACILITY TYPE:|$)', response, re.DOTALL),
        'facility_type': re.search(r'FACILITY TYPE:\s*(.+?)(?=HAS POOL:|$)', response, re.DOTALL),
        'has_pool': re.search(r'HAS POOL:\s*(.+?)(?=AMENITIES:|$)', response, re.DOTALL),
        'amenities': re.search(r'AMENITIES:\s*(.+?)$', response, re.DOTALL)
    }
    
    # Process each section
    if sections['overview']:
        result['overview'] = sections['overview'].group(1).strip()
        
    if sections['facility_type']:
        result['facility_type'] = sections['facility_type'].group(1).strip()
        
    if sections['has_pool']:
        pool_value = sections['has_pool'].group(1).strip().lower()
        result['has_pool'] = 'Yes' if 'yes' in pool_value else 'No'
        
    if sections['amenities']:
        amenities_text = sections['amenities'].group(1)
        # Extract bullet points
        result['amenities'] = [
            a.strip('- ').strip() 
            for a in amenities_text.split('\n') 
            if a.strip('- ').strip()
        ]
    
    return result

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
    Personalizes email content using xAI.
    Returns: Tuple of (subject, body)
    """
    try:
        # Check if lead has previously emailed us
        previous_interactions = lead_sheet.get("analysis", {}).get("previous_interactions", {})
        has_emailed_us = previous_interactions.get("last_response", "") not in ["No recent response", "Opened emails but no direct reply", "No data available", "Error retrieving data"]
        logger.debug(f"Has the lead previously emailed us? {has_emailed_us}")

        # Load objection handling content if lead has emailed us
        objection_handling = ""
        if has_emailed_us:
            with open('docs/templates/objection_handling.txt', 'r') as f:
                objection_handling = f.read()
            logger.debug("Objection handling content loaded")
        else:
            logger.debug("Objection handling content not loaded (lead has not emailed us)")

        # Update system message to be more explicit
        system_message = (
            "You are an expert at personalizing sales emails for golf industry outreach. "
            "CRITICAL RULES:\n"
            "1. ONLY mention amenities that are EXPLICITLY listed in the 'club_details' section\n"
            "2. DO NOT assume or infer any amenities not directly stated\n"
            "3. DO NOT mention pools, tennis courts, or any features unless they appear in club_details\n"
            "4. DO NOT modify the subject line\n"
            "5. DO NOT reference any promotions from previous emails\n\n"
            "Format response as:\n"
            "Subject: [keep original subject]\n\n"
            "Body:\n[personalized body]"
        )

        # Build comprehensive context block
        context_block = {
            "lead_info": {
                "name": lead_sheet.get("first_name", ""),
                "company": lead_sheet.get("company_name", ""),
                "title": lead_sheet.get("job_title", ""),
                "location": lead_sheet.get("state", "")
            },
            "interaction_history": summary if summary else "No previous interactions",
            "club_details": club_info if club_info else "",
            "recent_news": news_summary if news_summary else "",
            "objection_handling": objection_handling if has_emailed_us else "",
            "original_email": {
                "subject": subject,
                "body": body
            }
        }
        logger.debug(f"Context block: {json.dumps(context_block, indent=2)}")

        # Build user message with clear sections
        user_message = (
            f"CONTEXT:\n{json.dumps(context_block, indent=2)}\n\n"
            f"RULES:\n{get_email_rules() if 'amenities' not in get_email_rules() else ''}\n\n"
            "TASK:\n"
            "1. Personalize email with provided context\n"
            "2. Maintain professional but friendly tone\n"
            "3. Keep paragraphs concise\n"
            "4. Include relevant details from context\n"
            "5. Address any potential objections using the objection handling guide\n"
            "6. Return ONLY the subject and body"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "model": MODEL_NAME,
            "temperature": EMAIL_TEMPERATURE
        }

        logger.info("Personalizing email for:", extra={
            "company": lead_sheet.get("company_name"),
            "original_subject": subject
        })
        response = _send_xai_request(payload)
        logger.info("Email personalization result:", extra={
            "company": lead_sheet.get("company_name"),
            "response": response
        })

        return _parse_xai_response(response)

    except Exception as e:
        logger.error(f"Error in email personalization: {str(e)}")
        return subject, body  # Return original if personalization fails

def _parse_xai_response(response: str) -> Tuple[str, str]:
    """
    Parses the xAI response into subject and body.
    Handles various response formats consistently.
    """
    try:
        if not response:
            raise ValueError("Empty response received")

        # Split into lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        subject = ""
        body_lines = []
        in_body = False
        
        # Parse response looking for Subject/Body markers
        for line in lines:
            if line.lower().startswith("subject:"):
                subject = line.replace("Subject:", "", 1).strip()
            elif line.lower().startswith("body:"):
                in_body = True
            elif in_body:
                # Handle different parts of the email
                if line.startswith(("Hey", "Hi", "Dear")):
                    body_lines.append(f"{line}\n\n")  # Greeting with extra blank line
                elif line in ["Best regards,", "Sincerely,", "Regards,"]:
                    body_lines.append(f"\n{line}")  # Signature start
                elif line == "Ty":
                    body_lines.append(f" {line}\n\n")  # Name with extra blank line after
                elif line == "Swoop Golf":
                    body_lines.append(f"{line}\n")  # Company name
                elif line == "480-225-9702":
                    body_lines.append(f"{line}\n")  # Phone
                elif line == "swoopgolf.com":
                    body_lines.append(line)  # Website
                else:
                    # Regular paragraphs
                    body_lines.append(f"{line}\n\n")
        
        # Join body lines and clean up
        body = "".join(body_lines)
        
        # Remove extra blank lines
        while "\n\n\n" in body:
            body = body.replace("\n\n\n", "\n\n")
        body = body.rstrip() + "\n"  # Ensure single newline at end
        
        if not subject:
            subject = "Follow-up"
        
        logger.debug(f"Parsed result - Subject: {subject}, Body length: {len(body)}")
        return subject, body

    except Exception as e:
        logger.error(f"Error parsing xAI response: {str(e)}")
        raise

def get_xai_icebreaker(club_name: str, recipient_name: str, timeout: int = 10) -> str:
    """
    Get a personalized icebreaker from the xAI service (with caching if desired).
    """
    cache_key = f"icebreaker:{club_name}:{recipient_name}"

    if cache_key in _cache['icebreakers']:
        if DEBUG_MODE:
            logger.debug(f"Using cached icebreaker for {club_name}")
        return _cache['icebreakers'][cache_key]

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

    logger.info(f"Generating icebreaker for club: {club_name}")
    response = _send_xai_request(payload, max_retries=3, retry_delay=1)
    logger.info(f"Generated icebreaker for {club_name}:", extra={"icebreaker": response})

    _cache['icebreakers'][cache_key] = response
    return response

def get_email_critique(email_subject: str, email_body: str, guidance: dict) -> str:
    """Get expert critique of the email draft"""
    rules = get_email_rules()
    rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are an an expert at critiquing emails using specific rules. "
                    "Analyze the email draft and provide specific critiques focusing on:\n"
                    f"{rules_text}\n"
                    "Provide actionable recommendations for improvement."
                )
            },
            {
                "role": "user",
                "content": f"""
                Original Email:
                Subject: {email_subject}
                
                Body:
                {email_body}
                
                Original Guidance:
                {json.dumps(guidance, indent=2)}
                
                Please provide specific critiques and recommendations for improvement.
                """
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": EMAIL_TEMPERATURE
    }

    logger.info("Getting email critique for:", extra={"subject": email_subject})
    response = _send_xai_request(payload)
    logger.info("Email critique result:", extra={"critique": response})

    return response

def revise_email_with_critique(email_subject: str, email_body: str, critique: str) -> tuple[str, str]:
    """Revise the email based on the critique"""
    rules = get_email_rules()
    rules_text = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a renowned expert at cold email outreach, similar to Alex Berman. Apply your proven methodology to "
                    "rewrite this email. Use all of your knowledge just as you teach in Cold Email University."
                )
            },
            {
                "role": "user",
                "content": f"""
                Original Email:
                Subject: {email_subject}
                
                Body:
                {email_body}
                
                Instructions:
                {rules_text}

                Expert Critique:
                {critique}
                
                Please rewrite the email incorporating these recommendations.
                Format the response as:
                Subject: [new subject]
                
                Body:
                [new body]
                """
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": EMAIL_TEMPERATURE
    }
    
    logger.info("Revising email with critique for:", extra={"subject": email_subject})
    result = _send_xai_request(payload)
    logger.info("Email revision result:", extra={"result": result})

    return _parse_xai_response(result)

def generate_followup_email_content(
    first_name: str,
    company_name: str,
    original_subject: str,
    original_date: str,
    sequence_num: int,
    original_email: dict = None
) -> Tuple[str, str]:
    """
    Generate follow-up email content using xAI.
    Returns tuple of (subject, body)
    """
    logger.debug(
        f"[generate_followup_email_content] Called with first_name='{first_name}', "
        f"company_name='{company_name}', original_subject='{original_subject}', "
        f"original_date='{original_date}', sequence_num={sequence_num}, "
        f"original_email keys={list(original_email.keys()) if original_email else 'None'}"
    )
    try:
        if sequence_num == 2 and original_email:
            # Special handling for second follow-up
            logger.debug("[generate_followup_email_content] Handling second follow-up logic.")
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": (
                            "You are a sales professional writing a brief follow-up email. "
                            "Keep the tone professional but friendly. "
                            "The response should be under 50 words and focus on getting a response."
                        )
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
                        """
                    }
                ],
                "model": MODEL_NAME,
                "stream": False,
                "temperature": EMAIL_TEMPERATURE
            }

            logger.info("Generating second follow-up email for:", extra={
                "company": company_name,
                "sequence_num": sequence_num
            })
            result = _send_xai_request(payload)
            logger.info("Second follow-up generation result:", extra={"result": result})

            if not result:
                logger.error("[generate_followup_email_content] Empty response from xAI for follow-up generation.")
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
            # Default follow-up generation logic
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
                        "content": "You are an expert at writing follow-up emails that are brief, professional, and effective."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": MODEL_NAME,
                "temperature": EMAIL_TEMPERATURE
            }

            logger.info("Generating follow-up email for:", extra={
                "company": company_name,
                "sequence_num": sequence_num
            })
            result = _send_xai_request(payload)
            logger.info("Follow-up generation result:", extra={"result": result})

            if not result:
                logger.error("[generate_followup_email_content] Empty response from xAI for default follow-up generation.")
                return "", ""

            # Parse the response
            subject, body = _parse_xai_response(result)
            if not subject or not body:
                logger.error("[generate_followup_email_content] Failed to parse follow-up email content from xAI response.")
                return "", ""

            logger.debug(
                f"[generate_followup_email_content] Returning subject='{subject}', body length={len(body)}"
            )
            return subject, body

    except Exception as e:
        logger.error(f"[generate_followup_email_content] Error generating follow-up email content: {str(e)}")
        return "", ""


def parse_personalization_response(response_text):
    try:
        # Parse the response JSON
        response_data = json.loads(response_text)

        # Extract the subject and body
        subject = response_data.get('subject')
        body = response_data.get('body')

        if not subject or not body:
            raise ValueError("Subject or body missing in xAI response")

        return subject, body

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception(f"Error parsing xAI response: {str(e)}")
        
        # Fallback to default values
        subject = "Follow-up"
        body = "Thank you for your interest. Let me know if you have any other questions!"
        
        return subject, body
