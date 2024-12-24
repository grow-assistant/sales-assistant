# followup_generation.py

from scheduling.database import get_db_connection
from utils.xai_integration import _send_xai_request
import re

def parse_subject_and_body(raw_text: str) -> tuple[str, str]:
    """
    Basic parser to extract the subject and body from the xAI response.
    This can be expanded depending on how xAI responds.
    """
    subject = "Follow-Up Email"
    body = raw_text.strip()

    # If we expect a structure like:
    # Subject: ...
    # Body: ...
    # we can parse with regex:
    sub_match = re.search(r"(?i)^subject:\s*(.*)", raw_text)
    bod_match = re.search(r"(?i)^body:\s*(.*)", raw_text, flags=re.DOTALL)
    if sub_match:
        subject = sub_match.group(1).strip()
    if bod_match:
        body = bod_match.group(1).strip()

    return subject, body

def generate_followup_email_xai(lead_id: int, sequence_num: int):
    """
    For a given lead and sequence number (e.g., 2 or 3),
    calls xAI to generate a personalized follow-up email,
    then updates the followups table with the resulting subject & body.
    """
    conn = get_db_connection()
    lead = conn.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()

    if not lead:
        conn.close()
        return

    # Example user prompt to xAI
    user_prompt = f"""
    The lead's name is {lead['first_name']} {lead['last_name']}, 
    role is {lead['role']}, at ??? (club_name placeholder).

    This is follow-up email #{sequence_num}. 
    Assume they have not responded to our previous outreach about Swoop Golf.

    Requirements:
    1. Provide a concise subject line.
    2. Write a short 2-paragraph body referencing on-demand F&B for golf clubs.
    3. Be polite, mention the previous email, and show a sense of urgency.
    4. Output should be in the format:
       Subject: ...
       Body: ...
    """

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sales copywriter for the golf industry."},
            {"role": "user", "content": user_prompt}
        ],
        "model": "grok-beta",   # Example model name
        "stream": False,
        "temperature": 0.7
    }

    xai_response = _send_xai_request(payload)
    subject, body = parse_subject_and_body(xai_response)

    conn.execute("""
        UPDATE followups
        SET subject = ?, body = ?
        WHERE lead_id = ? AND sequence_num = ?
    """, (subject, body, lead_id, sequence_num))
    conn.commit()
    conn.close()
