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
        - dbo.leads
        - dbo.lead_properties
        - dbo.companies
        - dbo.company_properties
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

        # Basic lead info
        email = lead_data.get("email") or metadata.get("lead_email", "")
        if not email:
            logger.error("No email in lead_sheet; cannot upsert lead.", extra={
                "correlation_id": correlation_id,
                "status": "error"
            })
            return

        # Access contact properties
        lead_props = lead_data.get("properties", {})
        first_name = lead_props.get("firstname", "")
        last_name = lead_props.get("lastname", "")
        role = lead_props.get("jobtitle", "")

        # HubSpot lead-level data
        lead_hs_id = lead_props.get("hs_object_id", "")
        lead_created_str = lead_props.get("createdate", "")
        lead_lastmod_str = lead_props.get("lastmodifieddate", "")
        lead_hs_createdate = safe_parse_date(lead_created_str)
        lead_hs_lastmodified = safe_parse_date(lead_lastmod_str)

        # -- Company data extraction --
        static_company_name = company_data.get("name", "")
        static_city = company_data.get("city", "")
        static_state = company_data.get("state", "")
        company_type = company_data.get("company_type", "Country Club")
        annualrevenue = company_data.get("annualrevenue", "")
        
        # Season and facility data
        season_data = analysis_data.get("season_data", {})
        year_round = season_data.get("year_round", False)
        year_round_str = "1" if year_round else "0"
        start_month = season_data.get("start_month", "")
        end_month = season_data.get("end_month", "")
        peak_season_start = season_data.get("peak_season_start", "")
        peak_season_end = season_data.get("peak_season_end", "")
        
        # Facility features
        has_pool = company_data.get("has_pool", 0)
        has_tennis = company_data.get("has_tennis_courts", 0)
        num_golf_holes = company_data.get("number_of_holes")
        facility_type = company_data.get("facility_type", "")
        facilities_info = analysis_data.get("facilities", {}).get("response", "")

        # Company HubSpot data
        company_hs_id = company_data.get("hs_object_id", "")
        company_created_str = company_data.get("createdate", "")
        company_lastmod_str = company_data.get("lastmodifieddate", "")
        company_hs_createdate = safe_parse_date(company_created_str)
        company_hs_lastmodified = safe_parse_date(company_lastmod_str)

        # lead_properties
        phone = lead_data.get("phone")
        cleaned_phone = clean_phone_number(phone) if phone else None
        lifecyclestage = lead_props.get("lifecyclestage", "")

        competitor_analysis = analysis_data.get("competitor_analysis", {}).get("competitor", "")

        # If we have research_data -> recent_news -> snippet
        facilities_news = ""
        research_data = analysis_data.get("research_data", {})
        if research_data.get("recent_news"):
            try:
                snippet = research_data["recent_news"][0].get("snippet", "")
                facilities_news = snippet[:500]  # Keep it under 500 chars
            except (IndexError, AttributeError):
                pass

        # ---------------------------------------------------------
        # 1) UPSERT into leads
        # ---------------------------------------------------------
        cursor.execute("SELECT lead_id, company_id FROM dbo.leads WHERE email = ?", (email,))
        row = cursor.fetchone()
        if row:
            lead_id, existing_company_id = row
            # update
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
                first_name, last_name, role,
                lead_hs_id, lead_hs_createdate, lead_hs_lastmodified,
                lead_id
            ))
        else:
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
            inserted_row = cursor.fetchone()
            lead_id = inserted_row[0] if inserted_row else None
            existing_company_id = None

        if not lead_id:
            raise ValueError("Could not retrieve or insert lead_id in dbo.leads.")

        # ---------------------------------------------------------
        # 2) UPSERT into companies if we have a company name
        # ---------------------------------------------------------
        company_id = existing_company_id
        if static_company_name.strip():
            cursor.execute("""
                SELECT company_id
                FROM dbo.companies
                WHERE name = ? AND city = ? AND state = ?
            """, (static_company_name, static_city, static_state))
            co_row = cursor.fetchone()

            if co_row:
                company_id = co_row[0]
                # update
                cursor.execute("""
                    UPDATE dbo.companies
                    SET
                        company_type = ?,
                        hs_object_id = ?,
                        hs_createdate = ?,
                        hs_lastmodifieddate = ?,
                        year_round = ?,
                        start_month = ?,
                        end_month = ?,
                        peak_season_start = ?,
                        peak_season_end = ?,
                        has_pool = ?,
                        has_tennis = ?,
                        num_golf_holes = ?,
                        xai_facilities_info = ?
                    WHERE company_id = ?
                """, (
                    company_type,
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round_str,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    company_data.get("has_pool", 0),
                    company_data.get("has_tennis_courts", 0),
                    company_data.get("number_of_holes", None),
                    facilities_info,
                    company_id
                ))
            else:
                # insert
                cursor.execute("""
                    INSERT INTO dbo.companies (
                        name,
                        city,
                        state,
                        company_type,
                        hs_object_id,
                        hs_createdate,
                        hs_lastmodifieddate,
                        year_round,
                        start_month,
                        end_month,
                        peak_season_start,
                        peak_season_end,
                        xai_facilities_info
                    )
                    OUTPUT Inserted.company_id
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    static_company_name,
                    static_city,
                    static_state,
                    company_type,
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round_str,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info
                ))
                inserted_co = cursor.fetchone()
                company_id = inserted_co[0] if inserted_co else None

            # Link lead to company if not already
            if company_id and lead_id:
                cursor.execute("""
                    UPDATE dbo.leads
                    SET company_id = ?
                    WHERE lead_id = ?
                """, (company_id, lead_id))

        # ---------------------------------------------------------
        # 3) lead_properties
        # ---------------------------------------------------------
        cursor.execute("""
            SELECT property_id FROM dbo.lead_properties WHERE lead_id = ?
        """, (lead_id,))
        lp_row = cursor.fetchone()

        if lp_row:
            cursor.execute("""
                UPDATE dbo.lead_properties
                SET
                    phone = ?,
                    lifecyclestage = ?,
                    competitor_analysis = ?,
                    last_modified = GETDATE()
                WHERE lead_id = ?
            """, (
                cleaned_phone,
                lifecyclestage,
                competitor_analysis,
                lead_id
            ))
        else:
            cursor.execute("""
                INSERT INTO dbo.lead_properties (
                    lead_id,
                    phone,
                    lifecyclestage,
                    competitor_analysis,
                    last_modified
                )
                VALUES (?, ?, ?, ?, GETDATE())
            """, (
                lead_id,
                cleaned_phone,
                lifecyclestage,
                competitor_analysis
            ))

        # ---------------------------------------------------------
        # 4) company_properties
        # ---------------------------------------------------------
        if company_id:
            cursor.execute("""
                SELECT property_id FROM dbo.company_properties WHERE company_id = ?
            """, (company_id,))
            cp_row = cursor.fetchone()

            if cp_row:
                cursor.execute("""
                    UPDATE dbo.company_properties
                    SET
                        annualrevenue = ?,
                        xai_facilities_news = ?,
                        last_modified = GETDATE()
                    WHERE company_id = ?
                """, (
                    annualrevenue,
                    facilities_news,
                    company_id
                ))
            else:
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
        logger.info("Successfully completed lead upsert", extra={
            "email": email,
            "correlation_id": correlation_id,
            "lead_id": lead_id,
            "company_id": company_id
        })

    except Exception as e:
        logger.error(f"Error in upsert_full_lead: {str(e)}", extra={
            "correlation_id": correlation_id,
            "lead_sheet_email": email
        })
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
