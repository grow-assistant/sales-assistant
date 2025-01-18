import os
from datetime import datetime
import pytz
import base64
from utils.gmail_integration import get_gmail_service, search_messages
from utils.logging_setup import logger

def get_email_details(message) -> dict:
    """Extract detailed information from a Gmail message."""
    headers = message['payload']['headers']
    
    # Get all relevant headers
    details = {
        'subject': next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject'),
        'from': next((h['value'] for h in headers if h['name'].lower() == 'from'), 'No From'),
        'to': next((h['value'] for h in headers if h['name'].lower() == 'to'), 'No To'),
        'cc': next((h['value'] for h in headers if h['name'].lower() == 'cc'), None),
        'bcc': next((h['value'] for h in headers if h['name'].lower() == 'bcc'), None),
        'date': next((h['value'] for h in headers if h['name'].lower() == 'date'), 'No Date'),
        'message_id': next((h['value'] for h in headers if h['name'].lower() == 'message-id'), 'No Message-ID'),
        'thread_id': message['threadId'],
        'gmail_id': message['id'],
        'internal_date': datetime.fromtimestamp(
            int(message['internalDate']) / 1000,
            tz=pytz.UTC
        ).strftime('%Y-%m-%d %H:%M:%S %Z')
    }
    
    # Get message body
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                details['body'] = base64.urlsafe_b64decode(
                    part['body']['data'].encode('UTF-8')
                ).decode('utf-8')
                break
    elif 'body' in message['payload'] and 'data' in message['payload']['body']:
        details['body'] = base64.urlsafe_b64decode(
            message['payload']['body']['data'].encode('UTF-8')
        ).decode('utf-8')
    else:
        details['body'] = 'No Body'
    
    return details

def search_sent_emails(email_address: str = "dhelfrick@secessiongolf.com") -> None:
    """
    Search Gmail for sent emails to a specific address.
    
    Args:
        email_address: The recipient's email address
    """
    try:
        service = get_gmail_service()
        
        # Search for emails sent TO this address
        sent_query = f"to:{email_address} in:sent"
        sent_messages = search_messages(sent_query)
        
        print(f"\nSent Email History for: {email_address}")
        print("=" * 60)
        
        if sent_messages:
            print(f"\nFound {len(sent_messages)} sent emails:")
            print("-" * 50)
            
            for msg in sent_messages:
                message = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()
                
                details = get_email_details(message)
                
                print(f"Internal Date: {details['internal_date']}")
                print(f"Header Date: {details['date']}")
                print(f"From: {details['from']}")
                print(f"To: {details['to']}")
                if details['cc']:
                    print(f"CC: {details['cc']}")
                if details['bcc']:
                    print(f"BCC: {details['bcc']}")
                print(f"Subject: {details['subject']}")
                print(f"Message ID: {details['gmail_id']}")
                print(f"Thread ID: {details['thread_id']}")
                print(f"RFC Message ID: {details['message_id']}")
                print("\nBody Preview:")
                print(details['body'][:200] + "..." if len(details['body']) > 200 else details['body'])
                print("-" * 50)
                print()
        else:
            print(f"\nNo sent emails found for {email_address}")
            
    except Exception as e:
        logger.error(f"Error searching email history: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    import sys
    
    # Use command line argument if provided, otherwise use default
    email = sys.argv[1] if len(sys.argv) > 1 else "dhelfrick@secessiongolf.com"
    
    search_sent_emails(email)

if __name__ == "__main__":
    main()