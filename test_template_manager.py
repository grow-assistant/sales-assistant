"""
test_template_manager.py

Tests the template manager functionality with a focus on personalization.
"""

import os
from services.leads_service import LeadsService
from utils.template_manager import TemplateManager


def test_template_personalization():
    """Test that templates include required personalization."""
    email = "smoran@shorthillsclub.org"
    
    # Initialize services
    leads_service = LeadsService()
    
    # Get lead summary (which uses template manager)
    summary = leads_service.generate_lead_summary(email)
    
    # Check required personalization
    subject = summary.get("subject", "")
    body = summary.get("body", "")
    
    print("\nTemplate Test Results:")
    print("=====================")
    print(f"Subject: {subject}")
    print("\nBody:")
    print(body)
    
    # Verify personalization
    personalization_checks = {
        "FirstName": False,
        "ClubName": False
    }
    
    for field in personalization_checks:
        if f"[{field}]" not in body and f"[{field}]" not in subject:
            personalization_checks[field] = True
        else:
            print(f"\nWARNING: {field} placeholder not replaced!")
            
    return all(personalization_checks.values())


if __name__ == "__main__":
    success = test_template_personalization()
    if success:
        print("\nSUCCESS: All personalization fields were properly replaced!")
    else:
        print("\nFAILURE: Some personalization fields were not replaced properly!")
        exit(1)
