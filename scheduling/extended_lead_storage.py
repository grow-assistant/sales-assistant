# scheduling/extended_lead_storage.py

from datetime import datetime, timedelta
from utils.logging_setup import logger
from scheduling.database import get_db_connection, store_email_draft

def find_next_available_timeslot(desired_send_date: datetime, preferred_window: dict = None) -> datetime:
    """
    Finds the next available timeslot using 30-minute windows.
    Within each window, attempts to schedule with 2-minute increments.
    If a window is full, moves to the next 30-minute window.
    
    Args:
        desired_send_date: The target date/time
        preferred_window: Optional dict with start/end times to constrain scheduling
    """
    logger.debug(f"Finding next available timeslot starting from: {desired_send_date}")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Round to nearest 30-minute window
        minutes = desired_send_date.minute
        rounded_minutes = (minutes // 30) * 30
        current_window_start = desired_send_date.replace(
            minute=rounded_minutes,
            second=0,
            microsecond=0
        )
        
        while True:
            # Check if we're still within preferred window
            if preferred_window:
                current_time = current_window_start.hour + current_window_start.minute / 60
                if current_time > preferred_window["end"]:
                    # Move to next day at start of preferred window
                    next_day = current_window_start + timedelta(days=1)
                    current_window_start = next_day.replace(
                        hour=int(preferred_window["start"]),
                        minute=int((preferred_window["start"] % 1) * 60)
                    )
                    logger.debug(f"Outside preferred window, moving to next day: {current_window_start}")
                    continue
            
            # Try each 2-minute slot within the current 30-minute window
            for minutes_offset in range(0, 30, 2):
                proposed_time = current_window_start + timedelta(minutes=minutes_offset)
                logger.debug(f"Checking availability for timeslot: {proposed_time}")
                
                # Check if this specific timeslot is available
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM emails
                    WHERE scheduled_send_date = ?
                """, (proposed_time,))
                
                count = cursor.fetchone()[0]
                logger.debug(f"Found {count} existing emails at timeslot {proposed_time}")
                
                if count == 0:
                    logger.debug(f"Selected available timeslot at {proposed_time}")
                    return proposed_time
            
            # If we get here, the current 30-minute window is full
            # Move to the next 30-minute window
            current_window_start += timedelta(minutes=30)
            logger.debug(f"Current window full, moving to next window starting at {current_window_start}")


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
        company_data = lead_sheet.get("company_data", {})

        lead_id = lead_props.get("hs_object_id")
        name = f"{lead_props.get('firstname', '')} {lead_props.get('lastname', '')}".strip()
        email_address = lead_data.get("email")
        company_short_name = company_data.get("company_short_name", "").strip()

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
            status='draft',
            company_short_name=company_short_name
        )

        conn.commit()
        logger.info(
            f"[store_lead_email_info] Scheduled email for lead_id={lead_id}, email={email_address}, "
            f"company={company_short_name}, draft_id={draft_id} at {scheduled_date}",
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
