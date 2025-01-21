# scheduling/extended_lead_storage.py

from datetime import datetime, timedelta
from utils.logging_setup import logger
from scheduling.database import get_db_connection, store_email_draft

def find_next_available_timeslot(desired_send_date: datetime) -> datetime:
    """
    Moves 'desired_send_date' forward if needed so that:
      1) It's at least 2 minutes after the last scheduled email
      2) We never exceed 15 emails in any rolling 3-minute window
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        while True:
            # 1) Ensure at least 2 minutes from the last scheduled email
            cursor.execute("""
                SELECT TOP 1 scheduled_send_date
                FROM emails
                WHERE scheduled_send_date IS NOT NULL
                ORDER BY scheduled_send_date DESC
            """)
            row = cursor.fetchone()
            if row:
                last_scheduled = row[0]
                min_allowed = last_scheduled + timedelta(minutes=2)
                if desired_send_date < min_allowed:
                    desired_send_date = min_allowed

            # 2) Check how many are scheduled in the 3-minute window prior to 'desired_send_date'
            window_start = desired_send_date - timedelta(minutes=3)
            cursor.execute("""
                SELECT COUNT(*)
                FROM emails
                WHERE scheduled_send_date BETWEEN ? AND ?
            """, (window_start, desired_send_date))
            count_in_3min = cursor.fetchone()[0]

            # If we already have 15 or more in that window, push out by 2 more minutes and repeat
            if count_in_3min >= 15:
                desired_send_date += timedelta(minutes=2)
            else:
                break

        return desired_send_date


def store_lead_email_info(
    lead_sheet: dict, 
    draft_id: str = None,
    scheduled_date: datetime = None,
    body: str = None,
    sequence_num: int = None,
    correlation_id: str = None
) -> None:
    """
    Store all email-related information for a lead in the 'emails' table.

    New logic enforces:
      - No more than 15 emails in any rolling 3-minute window
      - Each new email at least 2 minutes after the previously scheduled one
    """
    if correlation_id is None:
        correlation_id = f"store_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"

    try:
        # Default to 'now + 10 minutes' if no scheduled_date was provided
        if scheduled_date is None:
            scheduled_date = datetime.now() + timedelta(minutes=10)

        # ---- Enforce our scheduling constraints ----
        scheduled_date = find_next_available_timeslot(scheduled_date)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Extract basic lead info
        lead_data = lead_sheet.get("lead_data", {})
        lead_props = lead_data.get("properties", {})

        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        email_address = lead_data.get("email")

        # Insert into emails table with the adjusted 'scheduled_date'
        email_id = store_email_draft(
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
        logger.info(
            f"[store_lead_email_info] Scheduled email for lead_id={lead_id}, email={email_address}, "
            f"draft_id={draft_id} at {scheduled_date}",
            extra={"correlation_id": correlation_id}
        )

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
