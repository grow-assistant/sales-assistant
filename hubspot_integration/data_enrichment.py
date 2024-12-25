# from external.external_api import safe_external_get
# from hubspot_integration.hubspot_api import update_hubspot_contact_field
# from config import MARKET_RESEARCH_API

# def enrich_lead_data(lead_data: dict) -> dict:
#     company = lead_data.get("company","")
#     if company:
#         ext_data = safe_external_get(f"{MARKET_RESEARCH_API}?query={company}+golf+club")
#         if ext_data:
#             lead_data["club_type"] = ext_data.get("club_type","unknown")
#             lead_data["membership_trends"] = ext_data.get("membership_trends","")
#             lead_data["recent_club_news"] = ext_data.get("recent_news",[])
#     return lead_data

# def handle_competitor_check(lead_data: dict):
#     email = lead_data.get("email","")
#     domain = email.split("@")[-1] if "@" in email else ""
#     competitor = check_competitor_on_website(domain)
#     if competitor and "contact_id" in lead_data:
#         updated = update_hubspot_contact_field(lead_data["contact_id"], "competitor", competitor)
#         if updated:
#             logger.info(f"Updated competitor field for contact {lead_data['contact_id']} to {competitor}.")
