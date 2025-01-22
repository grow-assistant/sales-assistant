import csv
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scheduling.database import get_db_connection
from utils.logging_setup import logger

def update_company_names():
    """Update company_short_name in emails table using lead_data.csv"""
    try:
        # Read the CSV file
        csv_path = project_root / "docs" / "data" / "lead_data.csv"
        email_to_company = {}
        
        logger.info("Reading lead data from CSV...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                email = row['Email'].lower()  # Store emails in lowercase for matching
                company_short_name = row['Company Short Name'].strip()
                if email and company_short_name:  # Only store if both fields have values
                    email_to_company[email] = company_short_name

        logger.info(f"Found {len(email_to_company)} email-to-company mappings")

        # Update the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all emails that need updating
        cursor.execute("""
            SELECT DISTINCT email_address 
            FROM emails 
            WHERE company_short_name IS NULL 
            OR company_short_name = ''
        """)
        
        update_count = 0
        for row in cursor.fetchall():
            email = row[0].lower() if row[0] else ''
            if email in email_to_company:
                cursor.execute("""
                    UPDATE emails 
                    SET company_short_name = ? 
                    WHERE LOWER(email_address) = ?
                """, (email_to_company[email], email))
                update_count += cursor.rowcount

        conn.commit()
        logger.info(f"Updated company_short_name for {update_count} email records")

    except Exception as e:
        logger.error(f"Error updating company names: {str(e)}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting company name update process...")
    update_company_names()
    logger.info("Completed company name update process") 