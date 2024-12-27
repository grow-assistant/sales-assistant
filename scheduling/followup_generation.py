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

    # Customize prompt based on sequence number and follow-up stage
    follow_up_context = {
        1: "This is the first follow-up (Day 3). Keep it short, friendly, and focused on checking if they've had a chance to review the initial email.",
        2: "This is the value-add follow-up (Day 7). Share a relevant success story about member satisfaction and revenue improvements. Focus on concrete metrics and results.",
        3: "This is the final follow-up (Day 14). Be polite but create urgency, emphasizing the opportunity while maintaining professionalism."
    }.get(sequence_num, "This is a follow-up email. Be professional and concise.")

    user_prompt = f"""
    The lead's name is {lead['first_name']} {lead['last_name']}, 
    role is {lead['role']}, at {lead.get('club_name', 'their club')}.

    {follow_up_context}
    Assume they have not responded to our previous outreach about Swoop Golf's on-demand F&B platform.

    Requirements:
    1. Provide a concise subject line.
    2. Write a personalized email that matches the follow-up stage context.
    3. For sequence=1 (Day 3): Keep it brief and friendly.
    4. For sequence=2 (Day 7): Include specific success metrics from similar clubs.
    5. For sequence=3 (Day 14): Politely indicate this is the final follow-up.
    6. Always reference on-demand F&B and member experience enhancement.
    7. Output format:
       Subject: ...
       Body: ...
    """

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sales copywriter for the golf industry."},
            {"role": "user", "content": user_prompt}
        ],
        "model": "grok-2-1212",   # Example model name
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
