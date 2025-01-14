# scheduling/extended_lead_storage.py

from datetime import datetime
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def store_lead_email_info(
    lead_sheet: dict, 
    draft_id: str = None,
    scheduled_date: datetime = None,
    subject: str = None,
    body: str = None,
    sequence_num: int = None,
    correlation_id: str = None
) -> None:
    """
    Store all email-related information for a lead in the emails table.
    """
    if correlation_id is None:
        correlation_id = f"store_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Extract basic lead info
        lead_data = lead_sheet.get("lead_data", {})
        lead_props = lead_data.get("properties", {})
        company_data = lead_data.get("company_data", {})

        # Get required fields for emails table
        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        company_name = company_data.get("name", "")
        company_city = company_data.get("city", "")
        company_st = company_data.get("state", "")
        company_type = company_data.get("company_type", "")

        # Insert into emails table with all info
        cursor.execute("""
            INSERT INTO emails (
                lead_id, name, company_name, company_city, 
                company_st, company_type, draft_id, 
                scheduled_send_date, subject, body,
                status, sequence_num, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
        """, (
            lead_id, name, company_name, company_city,
            company_st, company_type, draft_id,
            scheduled_date, subject, body,
            "draft" if draft_id else None, sequence_num
        ))

        conn.commit()
        logger.info("Successfully stored lead email info", extra={
            "email": lead_data.get("email"),
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
