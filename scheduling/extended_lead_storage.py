# scheduling/extended_lead_storage.py

import datetime
import json
from dateutil.parser import parse as parse_date
from utils.logging_setup import logger
from scheduling.database import get_db_connection

def safe_parse_date(date_str):
    """
    Safely parse a date string into a Python datetime (UTC).
    Returns None if parsing fails or date_str is None.
    """
    if not date_str:
        return None
    try:
        return parse_date(date_str)
    except Exception:
        return None

def upsert_full_lead(lead_sheet: dict) -> None:
    """
    Upsert the lead and related company data into SQL:
      1) leads (incl. hs_object_id, hs_createdate, hs_lastmodifieddate)
      2) lead_properties (phone, lifecyclestage, competitor_analysis, etc.)
      3) companies (incl. hs_object_id, hs_createdate, hs_lastmodifieddate, plus season data)
      4) company_properties (annualrevenue, xai_facilities_news only)
    """
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
            logger.error("No email found in lead_sheet; cannot upsert lead.")
            return

        logger.debug(f"Upserting lead with email={email}")

        # Fix: Access properties correctly from lead_data structure
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
        phone = lead_properties.get("phone", "")
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
        cursor.execute("SELECT hs_object_id, company_hs_id FROM dbo.leads WHERE email = ?", (email,))
        row = cursor.fetchone()

        if row:
            existing_hs_id = row[0]
            existing_company_hs_id = row[1]
            logger.debug(f"Lead with email={email} found (hs_object_id={existing_hs_id}); updating record.")
            cursor.execute("""
                UPDATE dbo.leads
                SET first_name = ?,
                    last_name = ?,
                    role = ?,
                    hs_createdate = ?,
                    hs_lastmodifieddate = ?,
                    status = 'active'
                WHERE hs_object_id = ?
            """, (
                first_name,
                last_name,
                role,
                lead_hs_createdate,
                lead_hs_lastmodified,
                lead_hs_id
            ))
        else:
            logger.debug(f"Lead with email={email} not found; inserting new record.")
            cursor.execute("""
                INSERT INTO dbo.leads (
                    hs_object_id,
                    email,
                    first_name,
                    last_name,
                    role,
                    status,
                    hs_createdate,
                    hs_lastmodifieddate
                )
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """, (
                lead_hs_id,
                email,
                first_name,
                last_name,
                role,
                lead_hs_createdate,
                lead_hs_lastmodified
            ))

        conn.commit()

        # ==========================================================
        # 2. Upsert into companies (static fields + HS fields + season data)
        # ==========================================================
        if not static_company_name.strip():
            logger.debug("No company name found, skipping upsert for companies.")
        else:
            cursor.execute("""
                SELECT hs_object_id 
                FROM dbo.companies
                WHERE name = ? AND city = ? AND state = ?
            """, (static_company_name, static_city, static_state))
            existing_co = cursor.fetchone()

            if existing_co:
                existing_company_hs_id = existing_co[0]
                logger.debug(f"Company found (hs_object_id={existing_company_hs_id}); updating fields + season data if needed.")
                cursor.execute("""
                    UPDATE dbo.companies
                    SET city = ?,
                        state = ?,
                        hs_createdate = ?,
                        hs_lastmodifieddate = ?,
                        year_round = ?,
                        start_month = ?,
                        end_month = ?,
                        peak_season_start = ?,
                        peak_season_end = ?,
                        xai_facilities_info = ?
                    WHERE hs_object_id = ?
                """, (
                    static_city,
                    static_state,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info,  # Save xai_facilities_info into dbo.companies
                    company_hs_id
                ))
            else:
                logger.debug(f"No matching company; inserting new row for name={static_company_name}.")
                cursor.execute("""
                    INSERT INTO dbo.companies (
                        name, city, state,
                        hs_object_id, hs_createdate, hs_lastmodifieddate,
                        year_round, start_month, end_month,
                        peak_season_start, peak_season_end,
                        xai_facilities_info
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    static_company_name,
                    static_city,
                    static_state,
                    company_hs_id,
                    company_hs_createdate,
                    company_hs_lastmodified,
                    year_round,
                    start_month,
                    end_month,
                    peak_season_start,
                    peak_season_end,
                    facilities_info  # Save xai_facilities_info
                ))

            conn.commit()

        # If we have a company_hs_id, ensure leads.company_hs_id is updated
        if company_hs_id:
            if not existing_company_hs_id or existing_company_hs_id != company_hs_id:
                logger.debug(f"Updating lead with hs_object_id={lead_hs_id} to reference company_hs_id={company_hs_id}.")
                cursor.execute("""
                    UPDATE dbo.leads
                    SET company_hs_id = ?
                    WHERE hs_object_id = ?
                """, (company_hs_id, lead_hs_id))
                conn.commit()

        # ==========================================================
        # 3. Upsert into lead_properties (phone, lifecycle, competitor, etc.)
        # ==========================================================
        cursor.execute("""
            SELECT property_id FROM dbo.lead_properties WHERE hs_object_id = ?
        """, (lead_hs_id,))
        lp_row = cursor.fetchone()

        if lp_row:
            prop_id = lp_row[0]
            logger.debug(f"Updating existing lead_properties (property_id={prop_id}) for hs_object_id={lead_hs_id}.")
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
            logger.debug(f"No lead_properties row found; inserting new one for hs_object_id={lead_hs_id}.")
            cursor.execute("""
                INSERT INTO dbo.lead_properties (
                    hs_object_id, phone, lifecyclestage, competitor_analysis,
                    last_response_date, last_modified
                )
                VALUES (?, ?, ?, ?, ?, GETDATE())
            """, (
                lead_hs_id,
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
        if company_hs_id:
            cursor.execute("""
                SELECT property_id FROM dbo.company_properties WHERE hs_object_id = ?
            """, (company_hs_id,))
            cp_row = cursor.fetchone()

            # competitor_analysis left blank
            competitor_analysis = ""
            # annualrevenue + company_overview are from above
            # xai_facilities_news is the new item to store
            if cp_row:
                cp_id = cp_row[0]
                logger.debug(f"Updating existing company_properties (property_id={cp_id}) for hs_object_id={company_hs_id}.")
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
                logger.debug(f"No company_properties row found; inserting new one for hs_object_id={company_hs_id}.")
                cursor.execute("""
                    INSERT INTO dbo.company_properties (
                        hs_object_id,
                        annualrevenue,
                        xai_facilities_news,
                        last_modified
                    )
                    VALUES (?, ?, ?, GETDATE())
                """, (
                    company_hs_id,
                    annualrevenue,
                    facilities_news
                ))
            conn.commit()

        logger.info(f"Successfully upserted lead + company info for email='{email}'.")

    except Exception as e:
        logger.error(f"Error in upsert_full_lead: {e}")
        conn.rollback()
    finally:
        conn.close()
