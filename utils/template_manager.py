"""
utils/template_manager.py

Centralizes template selection and reading logic.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import random
from datetime import datetime

from utils.doc_reader import DocReader
from utils.logging_setup import logger
from external.external_api import determine_club_season


class TemplateManager:
    """
    Centralizes template selection and reading logic.
    Handles finding the right template based on job title and reading its content.
    """

    # Role-based subject line variations
    SUBJECT_LINES = {
        "general_manager": [
            "Quick Question for [FirstName]",
            "New Ways to Elevate [ClubName]'s Operations",
            "Boost [ClubName]'s Efficiency with Swoop",
            "Need Assistance with [Task]? – [FirstName]"
        ],
        "fnb_manager": [
            "Ideas for Increasing F&B Revenue at [ClubName]",
            "Quick Note for [FirstName] about On-Demand Service",
            "A Fresh Take on [ClubName]'s F&B Operations"
        ],
        "golf_ops": [
            "Keeping [ClubName] Rounds on Pace: Quick Idea",
            "New Golf Ops Tools for [ClubName]",
            "Quick Question for [FirstName] – On-Course Efficiency"
        ],
        "fallback": [
            "Enhancing Your Club's Efficiency with Swoop",
            "Is [ClubName] Looking to Modernize?"
        ]
    }

    def __init__(self, docs_dir: Optional[str] = None):
        """Initialize with optional custom docs directory."""
        self._doc_reader = DocReader(docs_dir)

    def get_template_for_role(
        self,
        job_title: str,
        last_interaction_days: int = 0,
        placeholders: Optional[Dict[str, str]] = None,
        city: str = "",
        state: str = ""
    ) -> Tuple[str, str]:
        """
        Get the appropriate template (subject and body) for a given job title.
        
        Args:
            job_title: The lead's job title
            last_interaction_days: Days since last interaction
            placeholders: Dict of placeholder replacements (e.g., FirstName, ClubName)
        
        Returns:
            tuple[str, str]: (subject line, email body)
        """
        placeholders = placeholders or {}
        
        # 1) Clean job title for file matching
        clean_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )
        
        # 2) Get subject line based on role
        subject = self._get_subject_line(clean_title, last_interaction_days)
        
        # 3) Get email body from template file
        body = self._get_template_body(clean_title)
        
        # 4) Get season variation
        season_data = determine_club_season(city, state)
        current_month = datetime.now().month
        peak_start = season_data.get("peak_season_start_month", 3)  # Default to March
        peak_end = season_data.get("peak_season_end_month", 10)    # Default to October
        
        season_variation = self._get_season_variation(current_month, peak_start, peak_end)
        
        # 5) Replace all placeholders
        for key, val in placeholders.items():
            subject = subject.replace(f"[{key}]", str(val))
            body = body.replace(f"[{key}]", str(val))
            
        # Replace season variation specifically
        body = body.replace("{SEASON_VARIATION}", season_variation)
        
        return subject, body
        
    def _get_season_variation(self, current_month: int, peak_start: int, peak_end: int) -> str:
        """Get appropriate season variation text based on current month and peak season."""
        if peak_start <= current_month <= peak_end:
            return "I hope your peak season is going well."
        elif (current_month + 1) % 12 == peak_start % 12:
            return "With peak season approaching,"
        elif (current_month - 1) % 12 == peak_end % 12:
            return "As you wind down from peak season,"
        else:
            return "During this off-season period,"

    def _get_subject_line(self, role: str, last_interaction_days: int) -> str:
        """Get appropriate subject line variation based on role."""
        if role in self.SUBJECT_LINES:
            variations = self.SUBJECT_LINES[role]
        else:
            variations = self.SUBJECT_LINES["fallback"]
            
        # If "older" lead (>60 days), use first subject, else random
        if last_interaction_days > 60:
            return variations[0]
        return random.choice(variations)

    def _get_template_body(self, role: str) -> str:
        """Get email body from template file, with fallback content."""
        template_path = f"templates/{role}_initial_outreach.md"
        
        try:
            content = self._doc_reader.read_doc(template_path)
            if not content.strip():
                raise ValueError("Empty template content")
                
            # Extract body section from markdown
            _, body = self._extract_sections(content)
            if body.strip():
                logger.debug(f"Successfully loaded template for role: {role}")
                return body
                
        except Exception as e:
            logger.warning(
                f"Could not load template for role '{role}'. Using fallback. Error: {e}"
            )
            
        # Fallback template if anything fails
        return (
            "Hey [FirstName],\n\n"
            "{SEASON_VARIATION} I'd like to discuss how Swoop can significantly "
            "enhance your club's operational efficiency. Our solutions are designed to:\n\n"
            "- Automate booking and scheduling to reduce administrative workload.\n"
            "- Improve member engagement through personalized communications.\n"
            "- Optimize resource management for better cost control.\n\n"
            "Could we schedule a brief call this week to explore how these benefits "
            "could directly address your club's specific needs?\n\n"
            "Best,\n[YourName]"
        )

    def _extract_sections(self, content: str) -> Tuple[str, str]:
        """Extract subject and body sections from markdown content."""
        subject = ""
        body_lines = []
        mode = None
        
        for line in content.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("## Subject"):
                mode = "subject"
                continue
            elif line_stripped.startswith("## Body"):
                mode = "body"
                continue
                
            if mode == "subject":
                subject += line + " "
            elif mode == "body":
                body_lines.append(line)
                
        return subject.strip(), "\n".join(body_lines).strip()
