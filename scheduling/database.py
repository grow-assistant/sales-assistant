# scheduling/database.py

import sys
from pathlib import Path
import pyodbc
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_setup import logger
from config.settings import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

SERVER = DB_SERVER
DATABASE = DB_NAME
UID = DB_USER
PWD = DB_PASSWORD

def get_db_connection():
    """Get database connection."""
    logger.debug("Connecting to SQL Server", extra={
        "database": DATABASE,
        "server": SERVER,
        "masked_credentials": True
    })
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={UID};"
        f"PWD={PWD}"
    )
    try:
        conn = pyodbc.connect(conn_str)
        logger.debug("SQL connection established successfully.")
        return conn
    except pyodbc.Error as ex:
        logger.error("Error connecting to SQL Server", extra={
            "error": str(ex),
            "error_type": type(ex).__name__,
            "database": DATABASE,
            "server": SERVER
        }, exc_info=True)
        raise

def init_db():
    """Initialize database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        logger.info("Starting init_db...")

        # Create emails table if it doesn't exist
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                         WHERE object_id = OBJECT_ID(N'[dbo].[emails]') 
                         AND type in (N'U'))
            BEGIN
                CREATE TABLE dbo.emails (
                    email_id            INT IDENTITY(1,1) PRIMARY KEY,
                    lead_id            INT NOT NULL,
                    name               VARCHAR(100),
                    email_address      VARCHAR(255),
                    sequence_num       INT NULL,
                    body               VARCHAR(MAX),
                    scheduled_send_date DATETIME NULL,
                    actual_send_date   DATETIME NULL,
                    created_at         DATETIME DEFAULT GETDATE(),
                    status             VARCHAR(50) DEFAULT 'pending',
                    draft_id           VARCHAR(100) NULL,
                    gmail_id           VARCHAR(100),
                    company_short_name VARCHAR(100) NULL
                )
            END
        """)
        
        
        conn.commit()
        
        
    except Exception as e:
        logger.error("Error in init_db", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "database": DATABASE
        }, exc_info=True)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def clear_tables():
    """Clear the emails table in the database."""
    try:
        with get_db_connection() as conn:
            logger.debug("Clearing emails table")
            
            query = "DELETE FROM dbo.emails"
            logger.debug(f"Executing: {query}")
            conn.execute(query)
                
            logger.info("Successfully cleared emails table")

    except Exception as e:
        logger.exception(f"Failed to clear emails table: {str(e)}")
        raise e

def store_email_draft(cursor, lead_id: int, name: str = None,
                     email_address: str = None,
                     sequence_num: int = None,
                     body: str = None,
                     scheduled_send_date: datetime = None,
                     draft_id: str = None,
                     status: str = 'pending',
                     company_short_name: str = None) -> int:
    """
    Store email draft in database. Returns email_id.
    
    Table schema:
    - email_id (auto-generated)
    - lead_id
    - name
    - email_address
    - sequence_num
    - body
    - scheduled_send_date
    - actual_send_date (auto-managed)
    - created_at (auto-managed)
    - status
    - draft_id
    - gmail_id (managed elsewhere)
    - company_short_name
    """
    # First check if this draft_id already exists
    cursor.execute("""
        SELECT email_id FROM emails 
        WHERE draft_id = ? AND lead_id = ?
    """, (draft_id, lead_id))
    
    existing = cursor.fetchone()
    if existing:
        # Update existing record instead of creating new one
        cursor.execute("""
            UPDATE emails 
            SET name = ?,
                email_address = ?,
                sequence_num = ?,
                body = ?,
                scheduled_send_date = ?,
                status = ?,
                company_short_name = ?
            WHERE draft_id = ? AND lead_id = ?
        """, (
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            company_short_name,
            draft_id,
            lead_id
        ))
        return existing[0]
    else:
        # Insert new record
        cursor.execute("""
            INSERT INTO emails (
                lead_id,
                name,
                email_address,
                sequence_num,
                body,
                scheduled_send_date,
                status,
                draft_id,
                company_short_name
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            lead_id,
            name,
            email_address,
            sequence_num,
            body,
            scheduled_send_date,
            status,
            draft_id,
            company_short_name
        ))
        cursor.execute("SELECT SCOPE_IDENTITY()")
        return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")

