import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.gmail_integration import create_followup_draft
from utils.logging_setup import logger

def get_calendly_template() -> str:
    """Load the Calendly HTML template from file."""
    try:
        template_path = Path(project_root) / 'docs' / 'templates' / 'calendly.html'
        logger.debug(f"Loading Calendly template from: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_html = f.read()
            
        logger.debug(f"Loaded template, length: {len(template_html)}")
        logger.debug("First 200 chars of template:")
        logger.debug(template_html[:200])
        return template_html
    except Exception as e:
        logger.error(f"Error loading Calendly template: {str(e)}")
        return ""

def test_calendly_insert():
    """Create a test draft with Calendly HTML inserted."""
    try:
        # Load the Calendly template
        template_html = get_calendly_template()
        if not template_html:
            logger.error("Failed to load Calendly template")
            return

        # Create simple test content
        followup_content = (
            "<div dir='ltr'>"
            "<p>This is a test email to verify Calendly template insertion.</p>"
            "<p>Thanks,<br>Ty</p>"
            "</div>"
        )

        # Combine the content
        full_html = f"{followup_content}\n{template_html}"
        
        logger.debug("=" * 80)
        logger.debug("COMBINED HTML:")
        logger.debug("-" * 40)
        logger.debug(f"Total length: {len(full_html)}")
        logger.debug("First 500 chars:")
        logger.debug(full_html[:500])
        logger.debug("-" * 40)
        logger.debug("=" * 80)

        # Create the draft
        draft_result = create_followup_draft(
            sender="me",
            to="ty.hayes@swoopgolf.com",  # Replace with your email
            subject="TEST - Calendly Template Insert",
            message_text=full_html,
            lead_id="TEST",
            sequence_num=1
        )

        if draft_result.get('draft_id'):
            logger.info(f"Successfully created test draft with ID: {draft_result['draft_id']}")
        else:
            logger.error(f"Failed to create draft: {draft_result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error in test_calendly_insert: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    test_calendly_insert() 