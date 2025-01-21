#!/usr/bin/env python3

"""
Email Monitoring System
======================

This script orchestrates the email monitoring process by coordinating two main components:
1. Review Status Monitor (monitor_email_review_status.py)
   - Tracks Gmail drafts for review labels
   - Updates database when drafts are approved

2. Sent Status Monitor (monitor_email_sent_status.py)
   - Verifies when emails are actually sent
   - Updates database with send confirmation and details

Email Status Flow
----------------
1. draft → reviewed → sent
   - draft: Initial state when email is created
   - reviewed: Email has been approved via Gmail labels
   - sent: Email has been sent and confirmed via Gmail API

Configuration
------------
The script supports the following environment variables:
- EMAIL_MONITOR_INTERVAL: Seconds between monitoring runs (default: 300)
- EMAIL_MONITOR_MAX_RETRIES: Max retry attempts per run (default: 3)
- EMAIL_MONITOR_RETRY_DELAY: Seconds between retries (default: 60)
"""

import os
import sys
import time
from datetime import datetime
import pytz
from typing import Optional, Tuple, Dict
import traceback

# Local imports
from utils.logging_setup import logger
import monitor_email_review_status
import monitor_email_sent_status

###############################################################################
#                           CONFIGURATION
###############################################################################

class MonitorConfig:
    """Configuration settings for the monitoring process."""
    
    def __init__(self):
        # Process interval settings
        self.monitor_interval = int(os.getenv('EMAIL_MONITOR_INTERVAL', 300))  # 5 minutes
        self.max_retries = int(os.getenv('EMAIL_MONITOR_MAX_RETRIES', 3))
        self.retry_delay = int(os.getenv('EMAIL_MONITOR_RETRY_DELAY', 60))
        
        # Monitoring flags
        self.check_reviews = True
        self.check_sent = True
        
    def __str__(self) -> str:
        """Return string representation of config."""
        return (
            f"MonitorConfig("
            f"interval={self.monitor_interval}s, "
            f"max_retries={self.max_retries}, "
            f"retry_delay={self.retry_delay}s, "
            f"check_reviews={self.check_reviews}, "
            f"check_sent={self.check_sent})"
        )

###############################################################################
#                           MONITORING FUNCTIONS
###############################################################################

def run_review_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the review status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_review_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_sent_check() -> Tuple[bool, Optional[Exception]]:
    """
    Execute the sent status check process.
    
    Returns:
        Tuple[bool, Optional[Exception]]: (success, error)
    """
    try:
        monitor_email_sent_status.main()
        return True, None
    except Exception as e:
        return False, e

def run_monitoring_cycle(config: MonitorConfig) -> Dict[str, bool]:
    """
    Run a complete monitoring cycle with both checks.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        Dict[str, bool]: Status of each check
    """
    results = {
        'review_check': False,
        'sent_check': False
    }
    
    start_time = datetime.now(pytz.UTC)
    logger.info(f"=== Starting Monitoring Cycle at {start_time} ===")
    
    # Review Status Check
    if config.check_reviews:
        for attempt in range(config.max_retries):
            success, error = run_review_check()
            if success:
                results['review_check'] = True
                break
            else:
                logger.error(f"Review check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    # Brief pause between checks
    time.sleep(1)
    
    # Sent Status Check
    if config.check_sent:
        for attempt in range(config.max_retries):
            success, error = run_sent_check()
            if success:
                results['sent_check'] = True
                break
            else:
                logger.error(f"Sent check attempt {attempt + 1} failed: {error}")
                if attempt + 1 < config.max_retries:
                    logger.info(f"Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
    
    end_time = datetime.now(pytz.UTC)
    duration = end_time - start_time
    
    logger.info("\n=== Monitoring Cycle Complete ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Results: {results}")
    
    return results

###############################################################################
#                               MAIN PROCESS
###############################################################################

def run_continuous_monitoring(config: MonitorConfig):
    """
    Run the monitoring process continuously with the specified interval.
    
    Args:
        config: MonitorConfig instance with settings
    """
    logger.info(f"Starting continuous monitoring with config: {config}")
    
    while True:
        try:
            run_monitoring_cycle(config)
            logger.info(f"Waiting {config.monitor_interval} seconds until next cycle...")
            time.sleep(config.monitor_interval)
            
        except KeyboardInterrupt:
            logger.info("\nMonitoring interrupted by user")
            break
            
        except Exception as e:
            logger.error("Unexpected error in monitoring cycle:")
            logger.error(traceback.format_exc())
            logger.info(f"Retrying in {config.monitor_interval} seconds...")
            time.sleep(config.monitor_interval)

def run_single_cycle(config: MonitorConfig) -> bool:
    """
    Run a single monitoring cycle.
    
    Args:
        config: MonitorConfig instance with settings
        
    Returns:
        bool: True if all enabled checks succeeded
    """
    results = run_monitoring_cycle(config)
    return all(results.values())

def main():
    """
    Main entry point for the email monitoring system.
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = MonitorConfig()
        
        # Check for command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run single cycle
            success = run_single_cycle(config)
            return 0 if success else 1
        else:
            # Run continuous monitoring
            run_continuous_monitoring(config)
            return 0
            
    except Exception as e:
        logger.error(f"Fatal error in monitoring system: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 