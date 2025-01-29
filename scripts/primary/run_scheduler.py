import os
import sys
import logging

# Add project root to Python path (going up two levels from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Now import using absolute imports
from scripts.primary.email_sender import send_scheduled_emails
from utils.logging_setup import setup_logging
import time

# Create specialized loggers for the scheduler
scheduler_logger = setup_logging(
    log_name='scheduler',
    console_level='INFO',
    file_level='DEBUG',
    max_bytes=10485760  # 10MB
)

email_logger = setup_logging(
    log_name='email_service',
    console_level='WARNING',  # Only show warnings and errors in console
    file_level='INFO',
    max_bytes=5242880  # 5MB
)

def start_scheduler():
    """Start the email scheduler for sending all scheduled emails."""
    scheduler_logger.info("Starting email scheduler for all emails...")
    
    while True:
        try:
            scheduler_logger.debug("Checking for emails to send...")
            
            # Log email processing
            email_logger.info("Starting scheduled email processing cycle")
            send_scheduled_emails()
            email_logger.info("Completed email processing cycle")
            
        except Exception as e:
            # Log errors to both loggers
            scheduler_logger.error(f"Error in email scheduler: {str(e)}")
            email_logger.error(f"Failed to process scheduled emails: {str(e)}")
        
        # Log the wait period
        scheduler_logger.debug("Waiting 30 seconds before next check...")
        time.sleep(30)

if __name__ == "__main__":
    scheduler_logger.info("Initializing scheduler service...")
    start_scheduler() 