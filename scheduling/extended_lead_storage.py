# scheduling/extended_lead_storage.py

import datetime
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
      4) company_properties (annualrevenue, competitor_analysis, company_overview, etc. but NO season data)
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

        first_name = lead_data.get("firstname", "")
        last_name = lead_data.get("lastname", "")
        role = lead_data.get("jobtitle", "")

        # 2) HubSpot lead-level data
        lead_hs_id = lead_data.get("hs_object_id", "")
        lead_created_str = lead_data.get("createdate", "")
        lead_lastmod_str = lead_data.get("lastmodifieddate", "")

        lead_hs_createdate = safe_parse_date(lead_created_str)
        lead_hs_lastmodified = safe_parse_date(lead_lastmod_str)

        # 3) Company static data
        static_company_name = company_data.get("name", "")
        static_city = company_data.get("city", "")
        static_state = company_data.get("state", "")

        # 4) Company HubSpot data
        company_hs_id = company_data.get("hs_object_id", "")
        company_created_str = company_data.get("createdate", "")
        company_lastmod_str = company_data.get("hs_lastmodifieddate", "")

        company_hs_createdate = safe_parse_date(company_created_str)
        company_hs_lastmodified = safe_parse_date(company_lastmod_str)

        # 5) lead_properties (dynamic)
        phone = lead_data.get("phone", "")
        lifecyclestage = lead_data.get("lifecyclestage", "")
        competitor_analysis = analysis_data.get("competitor_analysis", "")

        # Attempt to parse "last_response" if it’s a date
        last_resp_str = analysis_data.get("previous_interactions", {}).get("last_response", "")
        last_response_date = safe_parse_date(last_resp_str)

        # 6) Season data (now stored in the COMPANIES table)
        season_data = analysis_data.get("season_data", {})
        year_round = season_data.get("year_round", "")            # e.g. "No"
        start_month = season_data.get("start_month", "")          # e.g. "May"
        end_month = season_data.get("end_month", "")              # e.g. "October"
        peak_season_start = season_data.get("peak_season_start", "")  # "06-01"
        peak_season_end = season_data.get("peak_season_end", "")      # "08-31"

        # 7) Other company_properties (dynamic)
        annualrevenue = company_data.get("annualrevenue", "")
        company_overview = analysis_data.get("research_data", {}).get("company_overview", "")
        
        # 8) xAI Facilities Data
        facilities_info = analysis_data.get("facilities_info", "")
        facilities_news = analysis_data.get("facilities_news", "")

        # ==========================================================
        # 1. Upsert into leads (static fields + HS fields)
        # ==========================================================
        cursor.execute("SELECT lead_id, company_id FROM dbo.leads WHERE email = ?", (email,))
        row = cursor.fetchone()

        if row:
            lead_id = row[0]
            existing_company_id = row[1]
            logger.debug(f"Lead with email={email} found (lead_id={lead_id}); updating record.")
            cursor.execute("""
                UPDATE dbo.leads
                SET first_name = ?,
                    last_name = ?,
                    role = ?,
                    hs_object_id = ?,
                    hs_createdate = ?,
                    hs_lastmodifieddate = ?
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
                    email, first_name, last_name, role,
                    hs_object_id, hs_createdate, hs_lastmodifieddate
                )
                OUTPUT Inserted.lead_id
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
            lead_id = inserted_row[0]
            existing_company_id = None

        conn.commit()

        # ==========================================================
        # 2. Upsert into companies (static fields + HS fields + season data)
        # ==========================================================
        company_id = None
        if not static_company_name.strip():
            logger.debug("No company name found, skipping upsert for companies.")
        else:
            # Try to find if there's already a row for this company by name + city + state
            cursor.execute("""
                SELECT company_id 
                FROM dbo.companies
                WHERE name = ? AND city = ? AND state = ?
            """, (static_company_name, static_city, static_state))
            existing_co = cursor.fetchone()

            if existing_co:
                company_id = existing_co[0]
                logger.debug(f"Company found (company_id={company_id}); updating HS fields + season data if needed.")
                cursor.execute("""
                    UPDATE dbo.companies
                    SET city = ?,
                        state = ?,
                        hs_object_id = ?,
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
                    company_hs_id,
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
                        name, city, state,
                        hs_object_id, hs_createdate, hs_lastmodifieddate,
                        year_round, start_month, end_month,
                        peak_season_start, peak_season_end,
                        xai_facilities_info
                    )
                    OUTPUT Inserted.company_id
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
                    facilities_info
                ))
                inserted_co = cursor.fetchone()
                company_id = inserted_co[0]

            conn.commit()

        # If we have a company_id, ensure leads.company_id is updated
        if company_id:
            if not existing_company_id or existing_company_id != company_id:
                logger.debug(f"Updating lead_id={lead_id} to reference company_id={company_id}.")
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
            logger.debug(f"Updating existing lead_properties (property_id={prop_id}) for lead_id={lead_id}.")
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
        #    (Removed peak_season_start / peak_season_end references)
        # ==========================================================
        if company_id:
            cursor.execute("""
                SELECT property_id FROM dbo.company_properties WHERE company_id = ?
            """, (company_id,))
            cp_row = cursor.fetchone()

            # competitor_analysis might be stored in either place, but we keep it here
            research_data = analysis_data.get("research_data", {})
            annualrevenue = company_data.get("annualrevenue", "")
            company_overview = research_data.get("company_overview", "")

            if cp_row:
                cp_id = cp_row[0]
                logger.debug(f"Updating existing company_properties (property_id={cp_id}) for company_id={company_id}.")
                cursor.execute("""
                    UPDATE dbo.company_properties
                    SET annualrevenue = ?,
                        competitor_analysis = ?,
                        company_overview = ?,
                        xai_facilities_news = ?,
                        last_modified = GETDATE()
                    WHERE property_id = ?
                """, (
                    annualrevenue,
                    competitor_analysis,
                    company_overview,
                    facilities_news,
                    cp_id
                ))
            else:
                logger.debug(f"No company_properties row found; inserting new one for company_id={company_id}.")
                cursor.execute("""
                    INSERT INTO dbo.company_properties (
                        company_id,
                        annualrevenue,
                        competitor_analysis,
                        company_overview,
                        xai_facilities_news,
                        last_modified
                    )
                    VALUES (?, ?, ?, ?, ?, GETDATE())
                """, (
                    company_id,
                    annualrevenue,
                    competitor_analysis,
                    company_overview,
                    facilities_news
                ))
            conn.commit()

        logger.info(f"Successfully upserted lead + company info for email='{email}'.")

    except Exception as e:
        logger.error(f"Error in upsert_full_lead: {e}")
        conn.rollback()
    finally:
        conn.close()
