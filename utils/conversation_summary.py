from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger

def get_latest_email_date(emails: List[Dict[str, Any]]) -> Optional[datetime]:
    """
    Find the most recent email date in a conversation thread.
    
    Args:
        emails: List of email dictionaries with timestamp and direction
        
    Returns:
        datetime object of latest email or None if no emails found
    """
    try:
        if not emails:
            return None
            
        # Sort emails by timestamp in descending order
        sorted_emails = sorted(
            [e for e in emails if e.get('timestamp')],
            key=lambda x: parse_date(x['timestamp']),
            reverse=True
        )
        
        if sorted_emails:
            return parse_date(sorted_emails[0]['timestamp'])
        return None
        
    except Exception as e:
        logger.error(f"Error getting latest email date: {str(e)}")
        return None

def summarize_lead_interactions(lead_sheet: Dict) -> str:
    """
    Get a simple summary of when we last contacted the lead.
    
    Args:
        lead_sheet: Dictionary containing lead data and interactions
        
    Returns:
        String summary of last contact date
    """
    try:
        latest_date = get_latest_email_date(lead_sheet.get('emails', []))
        if latest_date:
            return f"Last contact: {latest_date.strftime('%Y-%m-%d')}"
        return "No previous contact found"
        
    except Exception as e:
        logger.error(f"Error summarizing interactions: {str(e)}")
        return "Error getting interaction summary" 