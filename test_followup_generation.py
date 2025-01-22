from scheduling.followup_generation import generate_followup_email_xai
from scheduling.database import get_db_connection
from utils.logging_setup import logger
from utils.gmail_integration import create_draft, create_followup_draft
from scheduling.extended_lead_storage import store_lead_email_info

def test_followup_generation():
    """Test generating a follow-up email from an existing email in the database"""
    try:
        # Get a sample email from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent sent email
        cursor.execute("""
            SELECT TOP 1
                lead_id,
                email_address,
                name,
                gmail_id,
                scheduled_send_date
            FROM emails
            WHERE gmail_id IS NOT NULL
            AND sequence_num = 1
            ORDER BY created_at DESC
        """)
        
        row = cursor.fetchone()
        if not row:
            logger.error("No sent emails found in database")
            return
            
        lead_id, email, name, gmail_id, scheduled_date = row
        logger.info(f"Found original email to {email} (Lead ID: {lead_id})")
        
        # Generate follow-up email
        followup = generate_followup_email_xai(
            lead_id=lead_id,
            original_email={
                'email': email,
                'name': name,
                'gmail_id': gmail_id,
                'scheduled_send_date': scheduled_date
            }
        )
        
        if not followup:
            logger.error("Failed to generate follow-up email")
            return
            
        # Create draft in Gmail using the new follow-up function
        draft_result = create_followup_draft(
            sender="me",
            to=followup['email'],
            subject=followup['subject'],
            message_text=followup['body'],
            lead_id=followup['lead_id'],
            sequence_num=followup['sequence_num'],
            original_html=followup.get('original_html'),       # <--- Pass original HTML
            in_reply_to=followup.get('in_reply_to')            # <--- Pass in_reply_to
        )
        
        if draft_result["status"] == "ok":
            logger.info(f"Created follow-up draft for {email}")
            logger.info(f"Draft ID: {draft_result['draft_id']}")
            logger.info(f"Scheduled send date: {followup['scheduled_send_date']}")
            
            # Store the follow-up in database
            store_lead_email_info(
                lead_sheet={
                    "lead_data": {
                        "properties": {
                            "hs_object_id": followup['lead_id'],
                            "firstname": name.split()[0] if name else "",
                            "lastname": name.split()[1] if name and len(name.split()) > 1 else ""
                        }
                    },
                    "company_data": {
                        "name": name
                    }
                },
                draft_id=draft_result['draft_id'],
                scheduled_date=followup['scheduled_send_date'],
                body=followup['body'],
                sequence_num=followup['sequence_num']
            )
            logger.info("Stored follow-up in database")
        else:
            logger.error(f"Failed to create Gmail draft: {draft_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in test_followup_generation: {str(e)}", exc_info=True)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    print("\nStarting follow-up generation test...")
    test_followup_generation() 