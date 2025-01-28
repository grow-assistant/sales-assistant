import re
from typing import Dict, Optional, Tuple, Any
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
        
        # Add patterns for different types of responses
        self.out_of_office_patterns = [
            r"out\s+of\s+office",
            r"automatic\s+reply",
            r"auto\s*-?\s*reply",
            r"vacation\s+response",
            r"away\s+from\s+(?:the\s+)?office",
            r"will\s+be\s+(?:away|out)",
            r"not\s+(?:in|available)",
            r"on\s+vacation",
            r"on\s+holiday",
        ]
        
        self.employment_change_patterns = [
            r"no\s+longer\s+(?:with|at)",
            r"has\s+left\s+the\s+company",
            r"(?:email|address)\s+is\s+no\s+longer\s+valid",
            r"(?:has|have)\s+moved\s+on",
            r"no\s+longer\s+employed",
            r"last\s+day",
            r"departed",
            r"resigned",
        ]
        
        self.do_not_contact_patterns = [
            r"do\s+not\s+contact",
            r"stop\s+(?:contact|email)",
            r"unsubscribe",
            r"remove\s+(?:me|from)",
            r"opt\s+out",
            r"take\s+me\s+off",
        ]
        
        self.inactive_email_patterns = [
            r"undeliverable",
            r"delivery\s+failed",
            r"delivery\s+status\s+notification",
            r"failed\s+delivery",
            r"bounce",
            r"not\s+found",
            r"does\s+not\s+exist",
            r"invalid\s+recipient",
            r"recipient\s+rejected",
            r"no\s+longer\s+active",
            r"account\s+disabled",
            r"email\s+is\s+(?:now\s+)?inactive",
        ]

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

    def extract_bounced_email_address(self, bounce_message: Dict[str, Any]) -> Optional[str]:
        """
        Extract the original recipient's email address from a bounce notification message.
        
        Args:
            bounce_message: The full Gmail message object of the bounce notification
            
        Returns:
            str: The extracted email address or None if not found
        """
        try:
            # Get the message body
            if 'payload' not in bounce_message:
                return None

            # First try to get it from the subject line
            subject = None
            for header in bounce_message['payload'].get('headers', []):
                if header['name'].lower() == 'subject':
                    subject = header['value']
                    break

            if subject:
                # Look for email pattern in subject
                import re
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                matches = re.findall(email_pattern, subject)
                if matches:
                    # Filter out mailer-daemon address
                    for email in matches:
                        if 'mailer-daemon' not in email.lower():
                            return email

            # If not found in subject, try message body
            def get_text_from_part(part):
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        import base64
                        try:
                            return base64.urlsafe_b64decode(data).decode('utf-8')
                        except:
                            return ''
                return ''

            # Get text from main payload
            message_text = get_text_from_part(bounce_message['payload'])
            
            # If not in main payload, check parts
            if not message_text:
                for part in bounce_message['payload'].get('parts', []):
                    message_text = get_text_from_part(part)
                    if message_text:
                        break

            if message_text:
                # Look for common bounce message patterns
                patterns = [
                    r'Original-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'Final-Recipient:.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'The email account that you tried to reach does not exist.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?',
                    r'failed permanently.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, message_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error extracting bounced email address: {str(e)}")
            return None

    def is_out_of_office(self, body: str, subject: str) -> bool:
        """Check if message is an out of office reply."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.out_of_office_patterns)

    def is_employment_change(self, body: str, subject: str) -> bool:
        """Check if message indicates employment change."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.employment_change_patterns)

    def is_do_not_contact_request(self, body: str, subject: str) -> bool:
        """Check if message is a request to not be contacted."""
        text = f"{subject} {body}".lower()
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.do_not_contact_patterns)

    def is_inactive_email(self, body: str, subject: str) -> bool:
        """Check if message indicates an inactive email address."""
        inactive_phrases = [
            "not actively monitored",
            "no longer monitored",
            "inbox is not monitored",
            "email is inactive",
            "mailbox is inactive",
            "account is inactive",
            "email address is inactive",
            "no longer in service",
            "mailbox is not monitored"
        ]
        
        # Convert to lowercase for case-insensitive matching
        body_lower = body.lower()
        subject_lower = subject.lower()
        
        # Check both subject and body for inactive phrases
        return any(phrase in body_lower or phrase in subject_lower 
                  for phrase in inactive_phrases)

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