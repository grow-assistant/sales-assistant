# scripts/build_template.py

from utils.template_manager import TemplateManager
from utils.logging_setup import logger


###############################################################################
# MAIN FUNCTION FOR BUILDING EMAIL
###############################################################################
def build_outreach_email(
    profile_type: str,
    last_interaction_days: int,
    placeholders: dict,
    current_month: int = 3,
    start_peak_month: int = 4,
    end_peak_month: int = 7,
    use_markdown_template: bool = True
) -> tuple[str, str]:
    """
    Use TemplateManager to build an outreach email with proper personalization.
    """
    template_manager = TemplateManager()
    
    # Get template with proper personalization
    subject, body = template_manager.get_template_for_role(
        job_title=profile_type,
        last_interaction_days=last_interaction_days,
        placeholders=placeholders,
        city=placeholders.get("city", ""),
        state=placeholders.get("state", "")
    )
    
    logger.debug(
        "Email template generated successfully",
        extra={
            "profile_type": profile_type,
            "subject_length": len(subject),
            "body_length": len(body)
        }
    )
    
    return subject, body
