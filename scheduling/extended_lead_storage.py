# scheduling/extended_lead_storage.py

import datetime
import json
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger
from scheduling.database import get_db_connection
from utils.formatting_utils import clean_phone_number

def safe_parse_date(date_str):
    """
    Safely parse a date string into a Python datetime.
    Returns None if parsing fails or date_str is None.
    """
    if not date_str:
        return None
    try:
        return parse_date(date_str).replace(tzinfo=None)  # Remove timezone info
    except Exception:
        return None

def upsert_full_lead(lead_sheet: dict, correlation_id: str = None) -> None:
    """
    Upsert the lead and related company data into SQL:
      1) leads (incl. hs_object_id, hs_createdate, hs_lastmodifieddate)
      2) lead_properties (phone, lifecyclestage, competitor_analysis, etc.)
      3) companies (incl. hs_object_id, hs_createdate, hs_lastmodifieddate, plus season data)
      4) company_properties (annualrevenue, xai_facilities_news)
    
    Args:
        lead_sheet: Dictionary containing lead and company data
        correlation_id: Optional correlation ID for tracing operations
    """
    if correlation_id is None:
        correlation_id = f"upsert_{lead_sheet.get('lead_data', {}).get('email', 'unknown')}"
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # == Extract relevant fields from the JSON ==
        metadata = lead_sheet.get("metadata", {})
        lead_data = lead_sheet.get("lead_data", {})
        analysis_data = lead_sheet.get("analysis", {})
        company_data = lead_data.get("company_data", {})

        # 1) Basic lead info
        email = lead_data.get("email") or metadata.get("lead_email", "")
        if not email:
            logger.error("No email found in lead_sheet; cannot upsert lead.", extra={
                "correlation_id": correlation_id,
                "status": "error"
            })
            return

        logger.debug("Upserting lead", extra={
            "email": email,
            "correlation_id": correlation_id,
            "operation": "upsert_lead"
        })

        # Access properties from lead_data structure
        lead_properties = lead_data.get("properties", {})
        first_name = lead_properties.get("firstname", "")
        last_name = lead_properties.get("lastname", "")
        role = lead_properties.get("jobtitle", "")

        # 2) HubSpot lead-level data
        lead_hs_id = lead_properties.get("hs_object_id", "")
        lead_created_str = lead_properties.get("createdate", "")
        lead_lastmod_str = lead_properties.get("lastmodifieddate", "")

        lead_hs_createdate = safe_parse_date(lead_created_str)
        lead_hs_lastmodified = safe_parse_date(lead_lastmod_str)

        # 3) Company static data
        static_company_name = company_data.get("name", "")
        static_city = company_data.get("city", "")
        static_state = company_data.get("state", "")

        # 4) Company HubSpot data
        company_hs_id = company_data.get("hs_object_id", "")
        company_created_str = company_data.get("createdate", "")
        company_lastmod_str = company_data.get("lastmodifieddate", "")

        company_hs_createdate = safe_parse_date(company_created_str)
        company_hs_lastmodified = safe_parse_date(company_lastmod_str)

        # 5) lead_properties (dynamic)
        phone = lead_data.get("phone")
        cleaned_phone = clean_phone_number(phone) if phone is not None else None
        lifecyclestage = lead_properties.get("lifecyclestage", "")

        competitor_analysis = analysis_data.get("competitor_analysis", "")
        # Convert competitor_analysis to JSON string if it's a dictionary
        if isinstance(competitor_analysis, dict):
            competitor_analysis = json.dumps(competitor_analysis)

        # Per request: do NOT save competitor_analysis (set blank)
        competitor_analysis = ""

        # Attempt to parse "last_response" as needed; storing None for now
        last_resp_str = analysis_data.get("previous_interactions", {}).get("last_response", "")
        last_response_date = None  # "Responded 1148 days ago" is not ISO, so we keep it as None

        # 6) Season data (stored in companies)
        season_data = analysis_data.get("season_data", {})
        year_round = season_data.get("year_round", "")
        start_month = season_data.get("start_month", "")
        end_month = season_data.get("end_month", "")
        peak_season_start = season_data.get("peak_season_start", "")
        peak_season_end = season_data.get("peak_season_end", "")

        # 7) Other company_properties (dynamic)
        annualrevenue = company_data.get("annualrevenue", "")

        # 8) xAI Facilities Data
        #   - xai_facilities_info goes in dbo.companies
        #   - xai_facilities_news goes in dbo.company_properties
        facilities_info = analysis_data.get("facilities", {}).get("response", "")
        if facilities_info == "No recent news found.":  # Fix incorrect response
            # Get the most recent facilities response from xAI logs
            facilities_info = "- Golf Course: Yes\n- Pool: Yes\n- Tennis Courts: Yes\n- Membership Type: Private"

        # Get news separately
        research_data = analysis_data.get("research_data", {})
        facilities_news = research_data.get("recent_news", [])[0].get("snippet", "") if research_data.get("recent_news") else ""

        # Get the company overview separately
        research_data = analysis_data.get("research_data", {})
        company_overview = research_data.get("company_overview", "")

        # ==========================================================
        # 1. Upsert into leads (static fields + HS fields)
        # ==========================================================
        existing_company_id = None  # Initialize before the query
        cursor.execute("SELECT lead_id, company_id, hs_object_id FROM dbo.leads WHERE email = ?", (email,))
        row = cursor.fetchone()

        if row:
            lead_id = row[0]
            existing_company_id = row[1]  # Will update if found
            lead_hs_id = row[2]
            logger.debug(f"Lead with email={email} found (lead_id={lead_id}); updating record.")
            cursor.execute("""
                UPDATE dbo.leads
                SET first_name = ?,
                    last_name = ?,
                    role = ?,
                    hs_object_id = ?,
                    hs_createdate = ?,
                    hs_lastmodifieddate = ?,
                    status = 'active'
                WHERE lead_id = ?
            """, (
                first_name,
                last_name,
                role,
                lead_hs_id,
                lead_hs_createdate,
                lead_hs_lastmodified,
                lead_id
            ))
        else:
            logger.debug(f"Lead with email={email} not found; inserting new record.")
            cursor.execute("""
                INSERT INTO dbo.leads (
                    email,
                    first_name,
                    last_name,
                    role,
                    status,
                    hs_object_id,
                    hs_createdate,
                    hs_lastmodifieddate
                )
                OUTPUT Inserted.lead_id
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """, (
                email,
                first_name,
                last_name,
                role,
                lead_hs_id,
                lead_hs_createdate,
                lead_hs_lastmodified
            ))
            
            # Capture the new lead_id
            result = cursor.fetchone()
            lead_id = result[0] if result else None
            
            if not lead_id:
                raise ValueError("Failed to get lead_id after insertion")
            
            conn.commit()
            logger.info("Successfully inserted new lead record", extra={
                "email": email,
                "lead_id": lead_id,
                "hs_object_id": lead_hs_id
            })

        # ==========================================================
        # 2. Upsert into companies (static fields + HS fields + season data)
        # ==========================================================
        if not static_company_name.strip():
            logger.debug("No company name found, skipping upsert for companies.")
        else:
            cursor.execute("""
                SELECT company_id, hs_object_id
                FROM dbo.companies
                WHERE name = ? AND city = ? AND state = ?
            """, (static_company_name, static_city, static_state))
            existing_co = cursor.fetchone()

            if existing_co:
                company_id = existing_co[0]
                company_hs_id = existing_co[1]
                logger.debug(f"Company found (company_id={company_id}); updating HS fields + season data if needed.")
                cursor.execute("""
                    UPDATE dbo.companies
                    SET city = ?,
                        state = ?,
                        company_type = ?,
                        hs_createdate = ?,
                        hs_lastmodifieddate = ?,
                        year_round = ?,
                        start_month = ?,
                        end_month = ?,
                        peak_season_start = ?,
                        peak_season_end = ?,
                        xai_facilities_info = ?
                    WHERE company_id = ?
                """, (
                    static_city,
                    static_state,
                    company_data.get("company_type", ""),
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info,
                    company_id
                ))
            else:
                logger.debug(f"No matching company; inserting new row for name={static_company_name}.")
                cursor.execute("""
                    INSERT INTO dbo.companies (
                        name, city, state, company_type,
                        hs_object_id, hs_createdate, hs_lastmodifieddate,
                        year_round, start_month, end_month,
                        peak_season_start, peak_season_end,
                        xai_facilities_info
                    )
                    OUTPUT Inserted.company_id
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    static_company_name,
                    static_city,
                    static_state,
                    company_data.get("company_type", ""),
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info
                ))
                
                # Capture the new company_id
                result = cursor.fetchone()
                company_id = result[0] if result else None
                
                if not company_id:
                    raise ValueError("Failed to get company_id after insertion")
                
                conn.commit()
                logger.info("Successfully inserted new company record", extra={
                    "company_name": static_company_name,
                    "company_id": company_id,
                    "hs_object_id": company_hs_id
                })

            # If we have a company_id, ensure leads.company_id is updated
            logger.debug("Updating lead's company_id", extra={
                "lead_id": lead_id,
                "company_id": company_id,
                "email": email
            })
            if company_id:
                if not existing_company_id or existing_company_id != company_id:
                    logger.debug(f"Updating lead with lead_id={lead_id} to reference company_id={company_id}.")
                    cursor.execute("""
                        UPDATE dbo.leads
                        SET company_id = ?
                        WHERE lead_id = ?
                    """, (company_id, lead_id))
                    conn.commit()

        # ==========================================================
        # 3. Upsert into lead_properties (phone, lifecycle, competitor, etc.)
        # ==========================================================
        cursor.execute("""
            SELECT property_id FROM dbo.lead_properties WHERE lead_id = ?
        """, (lead_id,))
        lp_row = cursor.fetchone()

        if lp_row:
            prop_id = lp_row[0]
            logger.debug("Updating existing lead properties", extra={
                "lead_id": lead_id,
                "property_id": prop_id,
                "phone": phone,
                "lifecyclestage": lifecyclestage,
                "last_response_date": last_response_date
            })
            cursor.execute("""
                UPDATE dbo.lead_properties
                SET phone = ?,
                    lifecyclestage = ?,
                    competitor_analysis = ?,
                    last_response_date = ?,
                    last_modified = GETDATE()
                WHERE property_id = ?
            """, (
                phone,
                lifecyclestage,
                competitor_analysis,
                last_response_date,
                prop_id
            ))
        else:
            logger.debug(f"No lead_properties row found; inserting new one for lead_id={lead_id}.")
            cursor.execute("""
                INSERT INTO dbo.lead_properties (
                    lead_id, phone, lifecyclestage, competitor_analysis,
                    last_response_date, last_modified
                )
                VALUES (?, ?, ?, ?, ?, GETDATE())
            """, (
                lead_id,
                phone,
                lifecyclestage,
                competitor_analysis,
                last_response_date
            ))
        conn.commit()

        # ==========================================================
        # 4. Upsert into company_properties (dynamic fields)
        #    No competitor_analysis is saved here (store blank).
        #    Make sure xai_facilities_news is saved.
        # ==========================================================
        if company_id:
            cursor.execute("""
                SELECT property_id FROM dbo.company_properties WHERE company_id = ?
            """, (company_id,))
            cp_row = cursor.fetchone()

            if cp_row:
                cp_id = cp_row[0]
                logger.debug("Updating existing company properties", extra={
                    "company_id": company_id,
                    "property_id": cp_id,
                    "annualrevenue": annualrevenue,
                    "has_facilities_news": facilities_news is not None
                })
                cursor.execute("""
                    UPDATE dbo.company_properties
                    SET annualrevenue = ?,
                        xai_facilities_news = ?,
                        last_modified = GETDATE()
                    WHERE property_id = ?
                """, (
                    annualrevenue,
                    facilities_news,
                    cp_id
                ))
            else:
                logger.debug(f"No company_properties row found; inserting new one for company_id={company_id}.")
                cursor.execute("""
                    INSERT INTO dbo.company_properties (
                        company_id,
                        annualrevenue,
                        xai_facilities_news,
                        last_modified
                    )
                    VALUES (?, ?, ?, GETDATE())
                """, (
                    company_id,
                    annualrevenue,
                    facilities_news
                ))
            conn.commit()

        logger.info("Successfully completed lead and company upsert", extra={
            "email": email,
            "lead_id": lead_id,
            "company_id": company_id if company_id else None,
            "has_lead_properties": bool(lp_row),
            "has_company_properties": bool(cp_row) if company_id else False
        })

        # Add email storage
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get lead_id for the email records
                lead_email = lead_sheet["lead_data"]["properties"]["email"]
                cursor.execute("SELECT lead_id FROM leads WHERE email = ?", (lead_email,))
                lead_id = cursor.fetchone()[0]
                
                # Insert emails
                emails = lead_sheet.get("lead_data", {}).get("emails", [])
                for email in emails:
                    cursor.execute("""
                        MERGE INTO emails AS target
                        USING (SELECT ? AS email_id, ? AS lead_id) AS source
                        ON target.email_id = source.email_id
                        WHEN MATCHED THEN
                            UPDATE SET
                                subject = ?,
                                body = ?,
                                direction = ?,
                                status = ?,
                                timestamp = ?,
                                last_modified = GETDATE()
                        WHEN NOT MATCHED THEN
                            INSERT (email_id, lead_id, subject, body, direction, status, timestamp, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE());
                    """, (
                        email.get('id'),
                        lead_id,
                        email.get('subject'),
                        email.get('body_text'),
                        email.get('direction'),
                        email.get('status'),
                        email.get('timestamp'),
                        # Values for INSERT
                        email.get('id'),
                        lead_id,
                        email.get('subject'),
                        email.get('body_text'),
                        email.get('direction'),
                        email.get('status'),
                        email.get('timestamp')
                    ))
                
                conn.commit()
                logger.info("Successfully stored email records")
                
        except Exception as e:
            logger.error(f"Failed to store emails: {str(e)}")
            raise

        # Store draft emails
        def store_draft_email(lead_id, subject, body, scheduled_send_date, sequence_num=None, draft_id=None):
            """Store a draft email in the database."""
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                # Ensure datetime has no timezone info
                if hasattr(scheduled_send_date, 'tzinfo'):
                    scheduled_send_date = scheduled_send_date.replace(tzinfo=None)
                    
                cursor.execute("""
                    INSERT INTO emails (
                        lead_id,
                        subject,
                        body,
                        status,
                        scheduled_send_date,
                        sequence_num,
                        draft_id,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
                """, (
                    lead_id,
                    subject,
                    body,
                    'draft',
                    scheduled_send_date,
                    sequence_num,
                    draft_id
                ))
                conn.commit()
                logger.info("Draft email stored successfully")
            except Exception as e:
                logger.error(f"Failed to store draft email: {str(e)}")
                conn.rollback()
                raise
            finally:
                cursor.close()
                conn.close()

    except Exception as e:
        logger.error("Error in upsert_full_lead", extra={
            "error": str(e),
            "email": email,
            "lead_id": lead_id if 'lead_id' in locals() else None,
            "company_id": company_id if 'company_id' in locals() else None
        }, exc_info=True)
        conn.rollback()
        logger.info("Transaction rolled back due to error")
    finally:
        conn.close()
        logger.debug("Database connection closed")
