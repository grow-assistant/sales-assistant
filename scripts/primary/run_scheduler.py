import os
import sys
import logging

# Add project root to Python path (going up two levels from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Now import using absolute imports
from scripts.primary.email_sender import send_scheduled_emails
from utils.logging_setup import logger
import time

def start_scheduler():
    """Start the email scheduler for sending all scheduled emails."""
    logger.info("Starting email scheduler for all emails...")
    
    while True:
        try:
            logger.debug("Checking for emails to send...")
            send_scheduled_emails()
        except Exception as e:
            logger.error(f"Error in email scheduler: {str(e)}")
        
        # Wait 30 seconds before checking again
        time.sleep(30)

if __name__ == "__main__":
    start_scheduler() 