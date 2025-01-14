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
                    company_name       VARCHAR(200),
                    company_city       VARCHAR(100),
                    company_st         VARCHAR(2),
                    company_type       VARCHAR(50),
                    subject            VARCHAR(500),
                    body               VARCHAR(MAX),
                    status             VARCHAR(50) DEFAULT 'pending',
                    scheduled_send_date DATETIME NULL,
                    actual_send_date   DATETIME NULL,
                    created_at         DATETIME DEFAULT GETDATE(),
                    sequence_num       INT NULL,
                    draft_id           VARCHAR(100) NULL
                )
            END
        """)
        conn.commit()
        logger.info("init_db completed successfully. Emails table created if it didn't exist.")
        
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

def store_email_draft(cursor, lead_id: int, subject: str, body: str, 
                     name: str = None,
                     company_name: str = None,
                     company_city: str = None,
                     company_st: str = None,
                     company_type: str = None,
                     scheduled_send_date: datetime = None, 
                     sequence_num: int = None,
                     draft_id: str = None,
                     status: str = 'pending') -> int:
    """
    Store email draft in database. Returns email_id.
    """
    cursor.execute("""
        INSERT INTO emails (
            lead_id, name, company_name, company_city, company_st, company_type,
            subject, body, status, scheduled_send_date, created_at,
            sequence_num, draft_id
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, GETDATE(),
            ?, ?
        )
    """, (
        lead_id, name, company_name, company_city, company_st, company_type,
        subject, body, status, scheduled_send_date,
        sequence_num, draft_id
    ))
    cursor.execute("SELECT SCOPE_IDENTITY()")
    return cursor.fetchone()[0]

if __name__ == "__main__":
    init_db()
    logger.info("Database table created.")
