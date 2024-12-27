# scheduling/database.py

import pyodbc
from utils.logging_setup import logger
from config.settings import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

SERVER = DB_SERVER
DATABASE = DB_NAME
UID = DB_USER
PWD = DB_PASSWORD

def get_db_connection():
    logger.debug(f"Connecting to SQL Server: SERVER={SERVER}, DATABASE={DATABASE}, UID={UID}, PWD={PWD}")
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
        logger.error(f"Error connecting to SQL Server: {ex}")
        raise

def init_db():
    """
    Recreate all tables, with season data columns in the 'companies' table:
      - year_round, start_month, end_month, peak_season_start, peak_season_end
    Remove them from 'company_properties' so we don't store them twice.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        logger.info("Starting init_db...")

        # 1) Drop all foreign key constraints
        logger.info("Dropping foreign key constraints...")
        logger.debug("About to drop foreign key constraints via T-SQL")
        cursor.execute("""
            DECLARE @SQL NVARCHAR(MAX) = '';
            SELECT @SQL += 'ALTER TABLE ' + QUOTENAME(OBJECT_SCHEMA_NAME(parent_object_id))
                + '.' + QUOTENAME(OBJECT_NAME(parent_object_id))
                + ' DROP CONSTRAINT ' + QUOTENAME(name) + ';'
            FROM sys.foreign_keys;
            EXEC sp_executesql @SQL;
        """)
        logger.debug("Foreign key constraints dropped successfully")
        conn.commit()
        logger.debug("Changes committed after dropping foreign key constraints")

        # 2) Drop existing tables
        logger.info("Dropping existing tables if they exist...")
        logger.debug("About to drop existing tables if they exist")
        cursor.execute("""
            IF OBJECT_ID('dbo.emails', 'U') IS NOT NULL
                DROP TABLE dbo.emails;

            IF OBJECT_ID('dbo.lead_properties', 'U') IS NOT NULL
                DROP TABLE dbo.lead_properties;

            IF OBJECT_ID('dbo.leads', 'U') IS NOT NULL
                DROP TABLE dbo.leads;

            IF OBJECT_ID('dbo.company_properties', 'U') IS NOT NULL
                DROP TABLE dbo.company_properties;

            IF OBJECT_ID('dbo.companies', 'U') IS NOT NULL
                DROP TABLE dbo.companies;
        """)
        logger.debug("Existing tables dropped successfully")
        conn.commit()
        logger.debug("Changes committed after dropping tables")

        ################################################################
        # companies (static) – with new season data columns
        ################################################################
        logger.debug("About to create companies table")
        cursor.execute("""
        CREATE TABLE dbo.companies (
            hs_object_id         VARCHAR(50) NOT NULL PRIMARY KEY,
            name                 VARCHAR(255) NOT NULL,
            city                 VARCHAR(255),
            state                VARCHAR(255),
            created_at           DATETIME DEFAULT GETDATE(),

            -- HubSpot data:
            hs_createdate        DATETIME NULL,
            hs_lastmodifieddate  DATETIME NULL,

            -- NEW SEASON DATA COLUMNS:
            year_round           VARCHAR(10),
            start_month          VARCHAR(20),
            end_month            VARCHAR(20),
            peak_season_start    VARCHAR(10),
            peak_season_end      VARCHAR(10),

            -- xAI Facilities Data:
            xai_facilities_info  VARCHAR(MAX)
        );
        """)
        logger.debug("Companies table created successfully")
        conn.commit()
        logger.debug("Changes committed after creating companies table")

        ################################################################
        # company_properties (other dynamic fields, without season data)
        ################################################################
        logger.debug("About to create company_properties table")
        cursor.execute("""
        CREATE TABLE dbo.company_properties (
            property_id          INT IDENTITY(1,1) PRIMARY KEY,
            hs_object_id         VARCHAR(50) NOT NULL,
            annualrevenue        VARCHAR(50),
            xai_facilities_news  VARCHAR(MAX),
            last_modified        DATETIME DEFAULT GETDATE(),

            CONSTRAINT FK_company_props
                FOREIGN KEY (hs_object_id) REFERENCES dbo.companies(hs_object_id)
        );
        """)
        logger.debug("Company_properties table created successfully")
        conn.commit()
        logger.debug("Changes committed after creating company_properties table")

        ################################################################
        # leads (static)
        ################################################################
        logger.debug("About to create leads table")
        cursor.execute("""
        CREATE TABLE dbo.leads (
            hs_object_id           VARCHAR(50) NOT NULL PRIMARY KEY,
            email                  VARCHAR(255) NOT NULL,
            first_name             VARCHAR(255),
            last_name              VARCHAR(255),
            role                   VARCHAR(255),
            status                 VARCHAR(50) DEFAULT 'active',
            created_at             DATETIME DEFAULT GETDATE(),
            company_hs_id          VARCHAR(50) NULL,

            hs_createdate          DATETIME NULL,
            hs_lastmodifieddate    DATETIME NULL,

            CONSTRAINT UQ_leads_email UNIQUE (email),
            CONSTRAINT FK_leads_companies
                FOREIGN KEY (company_hs_id) REFERENCES dbo.companies(hs_object_id)
        );
        """)
        logger.debug("Leads table created successfully")
        conn.commit()
        logger.debug("Changes committed after creating leads table")

        ################################################################
        # lead_properties (dynamic/refreshable)
        ################################################################
        logger.debug("About to create lead_properties table")
        cursor.execute("""
        CREATE TABLE dbo.lead_properties (
            property_id           INT IDENTITY(1,1) PRIMARY KEY,
            hs_object_id          VARCHAR(50) NOT NULL,
            phone                 VARCHAR(50),
            lifecyclestage        VARCHAR(50),
            competitor_analysis   VARCHAR(MAX),
            last_response_date    DATETIME,
            last_modified         DATETIME DEFAULT GETDATE(),

            CONSTRAINT FK_lead_properties
                FOREIGN KEY (hs_object_id) REFERENCES dbo.leads(hs_object_id)
        );
        """)
        logger.debug("Lead_properties table created successfully")
        conn.commit()
        logger.debug("Changes committed after creating lead_properties table")

        ################################################################
        # emails (tracking)
        ################################################################
        logger.debug("About to create emails table")
        cursor.execute("""
        CREATE TABLE dbo.emails (
            email_id            INT IDENTITY(1,1) PRIMARY KEY,
            hs_object_id        VARCHAR(50) NOT NULL,    -- references leads
            subject             VARCHAR(500),
            body                VARCHAR(MAX),
            status              VARCHAR(50) DEFAULT 'pending',
            scheduled_send_date DATETIME NULL,
            actual_send_date    DATETIME NULL,
            created_at          DATETIME DEFAULT GETDATE(),

            CONSTRAINT FK_emails_leads
                FOREIGN KEY (hs_object_id) REFERENCES dbo.leads(hs_object_id)
        );
        """)
        logger.debug("Emails table created successfully")
        conn.commit()
        logger.debug("Changes committed after creating emails table")

        logger.info("init_db completed successfully. All tables dropped and recreated.")
    except Exception as e:
        logger.error(f"Error in init_db: {str(e)}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_db()
    logger.info("Database tables dropped and recreated.")
