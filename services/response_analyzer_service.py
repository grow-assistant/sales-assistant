import re
from typing import Dict, Optional, Tuple
from .data_gatherer_service import DataGathererService
from .gmail_service import GmailService
from .hubspot_service import HubspotService
from utils.logging_setup import logger
from config.settings import HUBSPOT_API_KEY

class ResponseAnalyzerService:
    def __init__(self):
        self.data_gatherer = DataGathererService()
        self.gmail_service = GmailService()
        self.hubspot = HubspotService(HUBSPOT_API_KEY)

    def analyze_response_status(self, email_address: str) -> Dict:
        """
        Analyze if and how a lead has responded to our emails.
        Returns a dictionary with analysis results.
        """
        try:
            # Use the new method that includes bounce detection
            messages = self.gmail_service.get_latest_emails_with_bounces(email_address)
            
            if not messages:
                return {
                    "status": "NO_MESSAGES",
                    "message": "No email conversation found for this address."
                }

            # Check both inbound and outbound messages
            inbound_msg = messages.get("inbound", {})
            outbound_msg = messages.get("outbound", {})
            
            # Check for bounce notification first
            if outbound_msg and outbound_msg.get("is_bounce"):
                return {
                    "status": "BOUNCED",
                    "response_type": "BOUNCED_EMAIL",
                    "confidence": 0.95,
                    "timestamp": outbound_msg.get("timestamp"),
                    "message": outbound_msg.get("body_text", ""),
                    "subject": outbound_msg.get("subject", "No subject")
                }

            # Then check for regular responses
            if inbound_msg:
                body_text = inbound_msg.get("body_text", "")
                response_type, confidence = self._categorize_response(body_text)
                
                return {
                    "status": "RESPONSE_FOUND",
                    "response_type": response_type,
                    "confidence": confidence,
                    "timestamp": inbound_msg.get("timestamp"),
                    "message": body_text,
                    "subject": inbound_msg.get("subject", "No subject")
                }
            
            return {
                "status": "NO_RESPONSE",
                "message": "No incoming responses found from this lead."
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Error analyzing response: {str(e)}"
            }

    def _get_latest_response(self, messages):
        """Get the most recent incoming message."""
        incoming_messages = [
            msg for msg in messages 
            if msg.get("direction") not in ["EMAIL", "NOTE", "OUTBOUND"]
            and msg.get("body_text")
        ]
        
        return incoming_messages[-1] if incoming_messages else None

    def _categorize_response(self, message: str) -> Tuple[str, float]:
        """
        Categorize the type of response.
        Returns tuple of (response_type, confidence_score)
        """
        # Clean the message
        cleaned_message = message.lower().strip()

        # Add bounce/delivery failure patterns
        bounce_patterns = [
            r"delivery status notification \(failure\)",
            r"address not found",
            r"recipient address rejected",
            r"address couldn't be found",
            r"unable to receive mail",
            r"delivery failed",
            r"undeliverable",
            r"550 5\.4\.1",  # Common SMTP failure code
        ]

        # Check for bounced emails first
        for pattern in bounce_patterns:
            if re.search(pattern, cleaned_message):
                return "BOUNCED_EMAIL", 0.95

        # Existing patterns
        auto_reply_patterns = [
            r"out\s+of\s+office",
            r"automatic\s+reply",
            r"auto\s*-?\s*reply",
            r"vacation\s+response",
            r"away\s+from\s+(?:the\s+)?office",
        ]

        left_company_patterns = [
            r"no\s+longer\s+(?:with|at)",
            r"has\s+left\s+the\s+company",
            r"(?:email|address)\s+is\s+no\s+longer\s+valid",
            r"(?:has|have)\s+moved\s+on",
            r"no\s+longer\s+employed",
        ]

        # Check for auto-replies
        for pattern in auto_reply_patterns:
            if re.search(pattern, cleaned_message):
                return "AUTO_REPLY", 0.9

        # Check for left company messages
        for pattern in left_company_patterns:
            if re.search(pattern, cleaned_message):
                return "LEFT_COMPANY", 0.9

        # If no patterns match, assume it's a genuine response
        # You might want to add more sophisticated analysis here
        return "GENUINE_RESPONSE", 0.7


def main():
    """Test function to demonstrate usage."""
    analyzer = ResponseAnalyzerService()
    
    while True:
        email = input("\nEnter email address to analyze (or 'quit' to exit): ")
        
        if email.lower() == 'quit':
            break
            
        result = analyzer.analyze_response_status(email)
        
        print("\nAnalysis Results:")
        print("-" * 50)
        
        if result["status"] == "RESPONSE_FOUND":
            print(f"Response Type: {result['response_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Subject: {result['subject']}")
            print("\nMessage Preview:")
            print(result["message"][:200] + "..." if len(result["message"]) > 200 else result["message"])
        else:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

if __name__ == "__main__":
    main() 