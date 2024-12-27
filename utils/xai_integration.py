import requests
from typing import Tuple
from utils.logging_setup import logger
from config.settings import DEBUG_MODE, XAI_API_URL, XAI_TOKEN, XAI_MODEL

XAI_BEARER_TOKEN = f"Bearer {XAI_TOKEN}"
MODEL_NAME = XAI_MODEL

def _send_xai_request(payload: dict) -> str:
    """
    Sends request to xAI API and returns response content or empty string on error.
    """
    logger.info("Initiating xAI API request")
    
    # Log the full prompt being sent
    messages = payload.get("messages", [])
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.info(f"xAI {role} message: {content}")

    try:
        response = requests.post(
            XAI_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": XAI_BEARER_TOKEN
            },
            json=payload,
            timeout=15
        )
        
        if response.status_code != 200:
            logger.error(f"xAI API error ({response.status_code}): {response.text}")
            return ""
            
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip() if data.get("choices") else ""
        
        # Log the actual response content
        logger.info(f"xAI response received: {content}")
        
        return content
        
    except Exception as e:
        logger.error(f"Error in xAI request: {str(e)}")
        return ""

##############################################################################
# News Search + Icebreaker
##############################################################################

def xai_news_search(club_name: str) -> str:
    """
    Searches for recent news about the club.
    Returns only news-related information.
    """
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a news research assistant. Your task is to ONLY report if "
                    "the club has been in recent news. Do not mention facilities or "
                    "club information. If no news is found, respond exactly with: "
                    "'No recent news found.'"
                )
            },
            {
                "role": "user",
                "content": f"Has {club_name} been in the news lately?"
            }
        ],
        "model": MODEL_NAME,
        "temperature": 0
    }
    return _send_xai_request(payload)

def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
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

def xai_club_info_search(club_name: str, location_str: str = "", amenities: list = None) -> str:
    """
    Searches for club facilities and amenities information.
    Returns only facilities-related information.
    """
    if not club_name.strip():
        if DEBUG_MODE:
            logger.debug("Empty club_name passed to xai_club_info_search; returning blank.")
        return ""

    loc_str = f" in {location_str}" if location_str else ""
    am_str = ", ".join(amenities) if amenities else "golf course, pool, or tennis courts"

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a club facilities researcher. Your job is to provide factual "
                    "information about the club's facilities and whether it's private. "
                    "Focus ONLY on facilities and membership type. "
                    "Do not include any news information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Does {club_name}{loc_str} have a {am_str}? "
                    "Is it a private club? "
                    "Format response as a bulleted list of facilities and membership type only."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.5
    }

    # Log the exact prompts being sent
    logger.debug("Facilities Search - System Prompt:", extra={
        "prompt": payload["messages"][0]["content"]
    })
    logger.debug("Facilities Search - User Prompt:", extra={
        "prompt": payload["messages"][1]["content"]
    })

    response = _send_xai_request(payload)
    logger.debug("Facilities Search - Response:", extra={
        "response": response
    })
    return response

def xai_facilities_search(club_name: str, location_str: str = "") -> str:
    """
    Searches for club facilities information.
    Returns only facilities-related information.
    """
    if not club_name.strip():
        logger.debug("Empty club_name passed to xai_facilities_search; returning blank.")
        return ""

    loc_str = f" in {location_str}" if location_str else ""
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a club facilities researcher. Your ONLY task is to list "
                    "the following information in bullet points:\n"
                    "- Golf course (yes/no)\n"
                    "- Pool (yes/no)\n"
                    "- Tennis courts (yes/no)\n"
                    "- Club type (private/public)\n\n"
                    "If you don't know, respond with 'Facilities information not available.'"
                )
            },
            {
                "role": "user",
                "content": f"What facilities does {club_name}{loc_str} have?"
            }
        ],
        "model": MODEL_NAME,
        "temperature": 0
    }

    # Log the exact prompts and response
    logger.debug("Facilities Search - System Prompt:", extra={"content": payload["messages"][0]["content"]})
    logger.debug("Facilities Search - User Prompt:", extra={"content": payload["messages"][1]["content"]})
    
    response = _send_xai_request(payload)
    
    logger.debug("Facilities Search - Response:", extra={"content": response})
    
    # Validate response format
    if not any(marker in response.lower() for marker in ["golf course", "pool", "tennis", "club type"]):
        logger.warning(f"Invalid facilities response format: {response}")
        return "Facilities information not available."
        
    return response

##############################################################################
# Personalize Email with xAI
##############################################################################

def personalize_email_with_xai(lead_sheet: dict, subject: str, body: str) -> Tuple[str, str]:
    """
    1) Fetch a short 'club info' snippet from xai_club_info_search.
    2) Incorporate that snippet into user_content for final rewriting.
    3) Use the result to rewrite subject and body.
    """
    logger.info("Starting email personalization with xAI")
    lead_data = lead_sheet.get("lead_data", {})
    company_data = lead_data.get("company_data", {})

    club_name = company_data.get("name", "")
    city = company_data.get("city", "")
    state = company_data.get("state", "")
    location_str = f"{city}, {state}".strip(", ")
    amenities = lead_sheet.get("analysis", {}).get("amenities", [])

    # 1) Use xai_club_info_search to gather context
    club_info_snippet = xai_club_info_search(club_name, location_str, amenities)

    # 2) Build user_content for rewriting
    facilities_info = lead_sheet.get("analysis", {}).get("facilities", {}).get("response", "")
    
    user_content = (
        f"Original Subject: {subject}\n"
        f"Original Body: {body}\n\n"
        f"Lead Info:\n"
        f"- First Name: {lead_data.get('firstname', '')}\n"
        f"- Company: {club_name}\n"
        f"- Role: {lead_data.get('jobtitle', '')}\n\n"
        "Additional Club Context:\n"
        f"{club_info_snippet}\n\n"
        "Club Facilities:\n"
        f"{facilities_info}\n\n"
        "Instructions:\n"
        "1. Keep the core structure and Swoop info.\n"
        "2. Focus on the business value and problem-solving aspects.\n"
        "3. Use verified club facilities information when available.\n"
        "4. Keep references to club specifics brief and relevant to the service.\n"
        "5. Use a helpful, positive tone.\n"
        "6. Write at a 6th-8th grade reading level.\n"
        "7. Keep paragraphs under 3 sentences each.\n"
        "8. Return in the format:\n"
        "    Subject: <subject>\n"
        "    Body: <body>"
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that personalizes outreach emails for golf clubs, "
                    "focusing on business value and relevant solutions."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0
    }

    # 3) Send to xAI for rewriting
    result = _send_xai_request(payload)
    if not result:
        logger.warning("No content returned from xAI. Falling back to original subject/body.")
        return subject, body

    lines = result.splitlines()
    new_subject, new_body = subject, []
    in_body = False

    for line in lines:
        lower = line.lower()
        if lower.startswith("subject:"):
            new_subject = line.split(":", 1)[1].strip()
        elif lower.startswith("body:"):
            in_body = True
        elif in_body:
            new_body.append(line)

    final_body = "\n".join(new_body).strip() or body

    if DEBUG_MODE:
        logger.debug("Completed xAI personalization rewrite", extra={
            "new_subject": new_subject,
            "new_body_preview": final_body[:150] + "..." if len(final_body) > 150 else final_body
        })

    logger.info("Email personalization completed successfully")
    return (new_subject if new_subject.strip() else subject), final_body
