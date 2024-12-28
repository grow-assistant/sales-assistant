# test_create_draft.py

import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gmail_integration import create_draft
from utils.logging_setup import logger

def test_draft():
    try:
        result = create_draft(
            sender="me", 
            to="test@example.com",
            subject="Test Draft",
            message_text="""Dear Team,

I hope this email finds you well. I wanted to provide a comprehensive update on the progress of our Q4 software development initiatives. Over the past month, our team has made significant strides in implementing the new feature set we discussed during our last planning session.

The core functionality has been successfully developed and has passed initial testing phases. Our QA team has identified some minor optimization opportunities, but overall, the system is performing above expectations. The user interface improvements have received positive feedback from our beta testing group, with particular praise for the streamlined workflow and intuitive navigation.

We've also made substantial progress on the backend infrastructure upgrades. The database optimization project is now 85% complete, resulting in a 40% improvement in query response times. Additionally, we've implemented enhanced security protocols as planned, including two-factor authentication and improved data encryption methods.

Looking ahead to next quarter, we've identified several areas for potential enhancement based on user feedback and market analysis. I've prepared a detailed report with specific recommendations, which I'll be sharing in our upcoming team meeting.

Please review these updates and let me know if you have any questions or concerns. I'm available for a more detailed discussion at your convenience.

Best regards,
Test"""
        )
        print(f"Draft creation result: {result}")
    except Exception as e:
        print(f"Error creating draft: {e}")

if __name__ == "__main__":
    test_draft()
