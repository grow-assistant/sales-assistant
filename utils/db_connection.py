import pyodbc
from utils.logging_setup import logger
from config.settings import DB_CONNECTION_STRING

def get_db_connection():
    """Get a connection to the SQL Server database."""
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        logger.debug("SQL connection established successfully.")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise 