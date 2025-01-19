#!/usr/bin/env python3
# scripts/migrate_emails_table.py

import sys
from pathlib import Path
import pyodbc
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_setup import logger
from config.settings import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    """Get database connection."""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD}"
    )
    return pyodbc.connect(conn_str)

def migrate_emails_table():
    """
    Migrate the emails table to the new schema.
    Steps:
    1. Create new table with correct schema
    2. Copy data from old table to new table
    3. Drop old table
    4. Rename new table to original name
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        logger.info("Starting emails table migration...")
        
        # Step 1: Create new table with correct schema
        cursor.execute("""
            CREATE TABLE emails_new (
                email_id            INT IDENTITY(1,1) PRIMARY KEY,
                lead_id            INT NOT NULL,
                name               VARCHAR(100),
                email_address      VARCHAR(255),
                sequence_num       INT NULL,
                body              VARCHAR(MAX),
                scheduled_send_date DATETIME NULL,
                actual_send_date   DATETIME NULL,
                created_at         DATETIME DEFAULT GETDATE(),
                status             VARCHAR(50) DEFAULT 'pending',
                draft_id           VARCHAR(100) NULL,
                gmail_id           VARCHAR(100)
            )
        """)
        
        # Step 2: Copy data from old table to new table
        logger.info("Copying data to new table...")
        cursor.execute("""
            INSERT INTO emails_new (
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                actual_send_date,
                created_at,
                status,
                draft_id,
                gmail_id
            )
            SELECT 
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                actual_send_date,
                created_at,
                status,
                draft_id,
                gmail_id
            FROM emails
        """)
        
        # Get row counts for verification
        cursor.execute("SELECT COUNT(*) FROM emails")
        old_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM emails_new")
        new_count = cursor.fetchone()[0]
        
        logger.info(f"Old table row count: {old_count}")
        logger.info(f"New table row count: {new_count}")
        
        if old_count != new_count:
            raise ValueError(f"Row count mismatch: old={old_count}, new={new_count}")
        
        # Step 3: Drop old table
        logger.info("Dropping old table...")
        cursor.execute("DROP TABLE emails")
        
        # Step 4: Rename new table to original name
        logger.info("Renaming new table...")
        cursor.execute("EXEC sp_rename 'emails_new', 'emails'")
        
        conn.commit()
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}", exc_info=True)
        conn.rollback()
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def verify_migration():
    """Verify the migration was successful."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'emails'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        logger.info("\nTable structure verification:")
        for col in columns:
            logger.info(f"Column: {col[0]}, Type: {col[1]}, Length: {col[2]}")
        
        # Check for any NULL values in required fields
        cursor.execute("""
            SELECT COUNT(*) 
            FROM emails 
            WHERE lead_id IS NULL
        """)
        null_leads = cursor.fetchone()[0]
        
        if null_leads > 0:
            logger.warning(f"Found {null_leads} rows with NULL lead_id")
        else:
            logger.info("No NULL values found in required fields")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting migration process...")
    
    # Confirm with user
    response = input("This will modify the emails table. Are you sure you want to continue? (y/N): ")
    if response.lower() != 'y':
        logger.info("Migration cancelled by user")
        sys.exit(0)
    
    try:
        migrate_emails_table()
        verify_migration()
        logger.info("Migration and verification completed successfully!")
    except Exception as e:
        logger.error("Migration failed!", exc_info=True)
        sys.exit(1) 