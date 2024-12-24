# agents/personalization.py

from typing import Dict, Any
from hubspot_integration.hubspot_api import (
    get_contact_by_email,
    get_contact_properties,
    get_all_notes_for_contact,
    get_associated_company_id,
    get_company_data,
    # Update functions assumed to be implemented:
    # update_hubspot_contact_property,
    # update_hubspot_company_property
)
from external.external_api import market_research, determine_club_season
from utils.logging_setup import logger
import os
from utils.doc_reader import read_doc


def generate_lead_summary(lead_email: str) -> Dict[str, Any]:
    """
    Gather and summarize lead's info according to the personalization steps.
    Checks HubSpot fields first to avoid unnecessary recomputation.
    Also attempts to load the correct outreach template based on job title.
    """
    contact_id = get_contact_by_email(lead_email)
    if not contact_id:
        logger.warning("No contact found for given email.")
        return {}

    # Fetch contact and related company data
    props = get_contact_properties(contact_id)
    notes = get_all_notes_for_contact(contact_id)
    company_id = get_associated_company_id(contact_id)
    company_data = get_company_data(company_id) if company_id else {}

    # Pull job title (we'll use this to select the template)
    job_title = props.get("jobtitle", "").strip()  # e.g. "Golf Operations Manager"

    # Convert the job title to a filename-safe string
    # e.g., "Golf Operations Manager" -> "golf_operations_manager"
    filename_job_title = (
        job_title.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "")
    )

    template_path = f"docs/templates/{filename_job_title}_initial_outreach.md"

    # Try to read the template file
    # If we cannot read it, we'll log a warning and use fallback content
    try:
        template_content = read_doc(template_path)
        # The doc_reader can return a dict or raw text, depending on your setup
        # For example, if you store "Subject: ..." / "Body: ..." in the Markdown,
        # your read_doc might parse and return them. Let’s assume you do so:
        subject = template_content.get("subject", "Default Subject")
        body = template_content.get("body", "Default Body")
    except Exception as e:
        logger.warning(
            f"Could not read document '{template_path}', using fallback content. "
            f"Reason: {str(e)}"
        )
        subject = "Fallback Subject"
        body = "Fallback Body..."

    # Check for existing stable properties at Company level
    club_type = company_data.get("club_type")
    competitor_presence = company_data.get("competitor_presence")
    peak_season_start = company_data.get("peak_season_start")
    peak_season_end = company_data.get("peak_season_end")

    # Check for existing stable properties at Contact level
    known_pain_points = props.get("known_pain_points")  # Multi-select stored as semicolon or comma-separated
    engagement_focus = props.get("engagement_focus")
    last_personalization_update = props.get("last_personalization_update_date")

    # Determine club type if not set
    if not club_type:
        derived_club_type = infer_club_type(company_data)
        club_type = derived_club_type
        # Optionally update HubSpot if you implement this function:
        # update_hubspot_company_property(company_id, "club_type", club_type)
    else:
        logger.info(f"Club type already known: {club_type}")

    # Determine season data (if not already present)
    # Even if peak season is stored, you might still use determine_club_season for validation
    city = company_data.get("city", "")
    state = company_data.get("state", "")
    season_data = determine_club_season(city, state)
    if peak_season_start:
        season_data["peak_season_start"] = peak_season_start
    if peak_season_end:
        season_data["peak_season_end"] = peak_season_end

    # Competitor presence: If not set, you could consider triggering competitor analysis elsewhere
    # For now, we just reuse if already present:
    if not competitor_presence:
        # If you have a competitor analysis step, call it here or store once determined.
        # For this example, assume we don't re-check here.
        # competitor_presence = analyze_competitors_synchronously(...) 
        # update_hubspot_company_property(company_id, "competitor_presence", competitor_presence)
        pass
    else:
        logger.info(f"Competitor presence already known: {competitor_presence}")

    # Extract engagement points if not already stored
    # If known_pain_points or engagement_focus are missing or empty, recalculate
    needs_pain_points = not known_pain_points or known_pain_points.strip() == ""
    needs_engagement_focus = not engagement_focus or engagement_focus.strip() == ""

    engagement_points = {}
    if needs_pain_points or needs_engagement_focus:
        engagement_points = extract_engagement_points(props, notes)

        # If we found new pain points and they differ from stored data, update them
        if needs_pain_points and engagement_points.get("pain_points"):
            new_pain_points_val = ";".join(engagement_points["pain_points"])
            # update_hubspot_contact_property(contact_id, "known_pain_points", new_pain_points_val)
            known_pain_points = new_pain_points_val

        # If we found new engagement focus items and they differ from stored data, update them
        if needs_engagement_focus and engagement_points.get("engagement_focus"):
            new_engagement_val = ";".join(engagement_points["engagement_focus"])
            # update_hubspot_contact_property(contact_id, "engagement_focus", new_engagement_val)
            engagement_focus = new_engagement_val

        # Optionally update last_personalization_update_date to current date
        # update_hubspot_contact_property(contact_id, "last_personalization_update_date", current_date_str())
    else:
        # Already have data stored, so just parse them into lists
        engagement_points = {
            "pain_points": known_pain_points.split(";") if known_pain_points else [],
            "engagement_focus": engagement_focus.split(";") if engagement_focus else [],
            "assets": [],  # 'assets' previously derived, can remain empty or re-derive if needed
            "tone": "Neutral"  # If needed, tone can be stored or defaulted
        }

    # External Insights
    company_name = company_data.get("name", "")
    # If stable data like competitor presence or club type rarely changes, no need to re-run market_research often
    # But we still do it here to get industry trends if needed.
    external_insights = market_research(company_name) if company_name else {"recent_news": []}
    industry_trends = external_insights.get("recent_news", [])

    # Construct bullet points about the lead
    # Use engagement_points from memory or calculation
    engaged_assets = engagement_points.get("assets", [])
    pain_points_list = engagement_points.get("pain_points", [])
    tone = engagement_points.get("tone", "Neutral")

    lead_care_about = [
        f"- Engaged with: {', '.join(engaged_assets) if engaged_assets else 'N/A'}",
        f"- Pain Points: {', '.join(pain_points_list) if pain_points_list else 'Not explicitly mentioned'}",
        f"- Interaction Tone: {tone}",
    ]

    # Construct bullet points about the club context
    top_trend_title = industry_trends[0]['title'] if industry_trends and 'title' in industry_trends[0] else 'N/A'
    club_context = [
        f"- Club Type: {club_type or 'Unknown'}",
        f"- Peak Season: {season_data.get('peak_season_start', 'Unknown')} to {season_data.get('peak_season_end', 'Unknown')}",
        f"- Industry Trend Mentioned: {top_trend_title}"
    ]

    return {
        "lead_summary": lead_care_about,
        "club_context": club_context,
        "subject": subject,
        "body": body
    }


def infer_club_type(company_data: Dict[str, Any]) -> str:
    """
    Infer the club type from the company's name or other properties.
    This runs only if the `club_type` property is not already set.
    """
    name = company_data.get("name", "").lower()
    if "country" in name:
        return "Country Club"
    elif "resort" in name:
        return "Resort Course"
    else:
        return "Golf Club"


def extract_engagement_points(props: Dict[str, Any], notes: Any) -> Dict[str, Any]:
    """
    Analyze properties and notes to identify engaged assets and pain points
    only if needed (i.e., if they're not already stored in HubSpot).
    """
    engaged_assets = []
    pain_points = []
    engagement_focus = []
    tone = "Neutral"

    # Example logic for engaged_assets:
    # If multiple page views:
    if int(props.get("hs_analytics_num_page_views", 0)) > 5:
        engaged_assets.append("Pricing Page")

    # Check notes for common pain point keywords:
    pain_keywords = ["slow beverage service", "staffing issues", "low member engagement", "limited menu options", "high operational costs", "inefficient technology systems", "long wait times on course"]

    note_bodies = [n.get("body", "").lower() for n in notes if n.get("body")]
    for nb in note_bodies:
        for kw in pain_keywords:
            if kw in nb and kw not in pain_points:
                pain_points.append(kw)

    # Determine tone:
    if any("thanks" in nb or "appreciate" in nb for nb in note_bodies):
        tone = "Friendly"

    # Derive engagement_focus from known behaviors:
    # If they viewed pricing (already added engaged asset), we might also mark them as Pricing Interest
    if "Pricing Page" in engaged_assets:
        engagement_focus.append("Pricing Interest")

    # If we detect references to case studies in notes or properties (just an example):
    # if something_detected_case_study: engagement_focus.append("Case Study Interest")

    return {
        "assets": engaged_assets,
        "pain_points": pain_points,
        "tone": tone,
        "engagement_focus": engagement_focus
    }


def current_date_str():
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d")
