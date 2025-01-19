# scheduling/extended_lead_storage.py

from datetime import datetime
from utils.logging_setup import logger
from scheduling.database import get_db_connection, store_email_draft

def store_lead_email_info(
    lead_sheet: dict, 
    draft_id: str = None,
    scheduled_date: datetime = None,
    body: str = None,
    sequence_num: int = None,
    correlation_id: str = None
) -> None:
    """
    Store all email-related information for a lead in the emails table.
    
    Table schema:
    - email_id (auto-generated)
    - lead_id
    - name
    - email_address
    - sequence_num
    - body
    - scheduled_send_date
    - actual_send_date (auto-managed)
    - created_at (auto-managed)
    - status
    - draft_id
    - gmail_id (managed elsewhere)
    """
    if correlation_id is None:
        correlation_id = f"store_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Extract basic lead info
        lead_data = lead_sheet.get("lead_data", {})
        lead_props = lead_data.get("properties", {})

        # Get required fields for emails table
        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        email_address = lead_data.get("email")

        # Insert into emails table with all info
        store_email_draft(
            cursor,
            lead_id=lead_id,
            name=name,
            email_address=email_address,
            sequence_num=sequence_num,
            body=body,
            scheduled_send_date=scheduled_date,
            draft_id=draft_id,
            status='draft'
        )

        conn.commit()
        logger.info("Successfully stored lead email info", extra={
            "email": email_address,
            "correlation_id": correlation_id
        })

    except Exception as e:
        logger.error(f"Error storing lead email info: {str(e)}", extra={
            "correlation_id": correlation_id
        })
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
