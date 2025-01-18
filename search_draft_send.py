import os
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def get_all_draft_ids(specific_lead_id: int) -> list:
    """
    Get draft ID for a specific lead from the emails table.
    
    Args:
        specific_lead_id: Lead ID to search for
        
    Returns:
        list: List of draft info dictionaries
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT draft_id, lead_id, subject, name, company_name
                FROM emails 
                WHERE lead_id = ? AND draft_id IS NOT NULL
            """, (specific_lead_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'draft_id': row[0],
                    'lead_id': row[1],
                    'subject': row[2],
                    'name': row[3],
                    'company': row[4]
                })
            
            return results
            
    except Exception as e:
        logger.error(f"Error getting draft IDs: {str(e)}")
        return []

def main():
    lead_id = 51651
    draft_infos = get_all_draft_ids(lead_id)
    
    if not draft_infos:
        print(f"No draft emails found for lead ID {lead_id}")
        return
    
    print("\nFound drafts:")
    print("-" * 50)
    for draft_info in draft_infos:
        print(f"Draft ID: {draft_info['draft_id']}")
        print(f"Lead ID: {draft_info['lead_id']}")
        print(f"Name: {draft_info['name'] or 'N/A'}")
        print(f"Company: {draft_info['company'] or 'N/A'}")
        print(f"Subject: {draft_info['subject']}")
        print("-" * 50)

if __name__ == "__main__":
    main()