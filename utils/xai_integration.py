import aiohttp
from typing import Tuple
from utils.logging_setup import logger
from config.settings import DEBUG_MODE, XAI_API_URL, XAI_TOKEN, XAI_MODEL

XAI_BEARER_TOKEN = f"Bearer {XAI_TOKEN}"
MODEL_NAME = XAI_MODEL

async def _send_xai_request(payload: dict) -> str:
    """
    Sends request to xAI API and returns response content or empty string on error.
    """
    if DEBUG_MODE:
        logger.debug("Sending xAI request payload", extra={"payload": payload})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                XAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": XAI_BEARER_TOKEN
                },
                json=payload,
                timeout=15
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"xAI API error ({response.status}): {error_text}")
                    return ""
                data = await response.json()
                content = data["choices"][0]["message"]["content"].strip() if data.get("choices") else ""
                if DEBUG_MODE:
                    logger.debug("xAI response received", extra={"content": content})
                return content
    except Exception as e:
        logger.error(f"Error in xAI request: {str(e)}")
        return ""

##############################################################################
# News Search + Icebreaker
##############################################################################

async def xai_news_search(club_name: str) -> str:
    if not club_name.strip():
        if DEBUG_MODE:
            logger.debug("Empty club_name passed to xai_news_search; returning blank.")
        return ""

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
    return await _send_xai_request(payload)

async def _build_icebreaker_from_news(club_name: str, news_summary: str) -> str:
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
    return await _send_xai_request(payload)

##############################################################################
# Club Info Search (Used ONLY for Final Email Rewriting)
##############################################################################

async def xai_club_info_search(club_name: str, location: str, amenities: list = None) -> str:
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
                    "You are a helpful assistant that provides a brief overview "
                    "of a club's location and amenities."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please provide a concise overview about {club_name} in {loc_str}, "
                    f"highlighting amenities like {am_str}. Only provide one short paragraph."
                )
            }
        ],
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.5
    }
    return await _send_xai_request(payload)

##############################################################################
# Personalize Email with xAI
##############################################################################

async def personalize_email_with_xai(lead_sheet: dict, subject: str, body: str) -> Tuple[str, str]:
    """
    1) Fetch a short 'club info' snippet from xai_club_info_search.
    2) Incorporate that snippet into user_content for final rewriting.
    3) Use the result to rewrite subject and body.
    """
    lead_data = lead_sheet.get("lead_data", {})
    company_data = lead_data.get("company_data", {})

    club_name = company_data.get("name", "")
    city = company_data.get("city", "")
    state = company_data.get("state", "")
    location_str = f"{city}, {state}".strip(", ")
    amenities = lead_sheet.get("analysis", {}).get("amenities", [])

    # 1) Use xai_club_info_search to gather context
    club_info_snippet = await xai_club_info_search(club_name, location_str, amenities)

    # 2) Build user_content for rewriting
    user_content = (
        f"Original Subject: {subject}\n"
        f"Original Body: {body}\n\n"
        f"Lead Info:\n"
        f"- First Name: {lead_data.get('firstname', '')}\n"
        f"- Company: {club_name}\n"
        f"- Role: {lead_data.get('jobtitle', '')}\n\n"
        "Additional Club Context:\n"
        f"{club_info_snippet}\n\n"
        "Instructions:\n"
        "1. Replace placeholders like [ClubName] with the actual name.\n"
        "2. Use the person's real first name if applicable.\n"
        "3. Keep the core structure and Swoop info.\n"
        "4. Focus on the business value and problem-solving aspects.\n"
        "5. Avoid presumptive descriptions of club features or scenery.\n"
        "6. Keep references to club specifics brief and relevant to the service.\n"
        "7. Return in the format:\n"
        "   Subject: <subject>\n"
        "   Body: <body>"
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
    result = await _send_xai_request(payload)
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

    return (new_subject if new_subject.strip() else subject), final_body
