import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def test_gmail_auth():
    """Test Gmail authentication and token generation."""
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    
    # Check if token.json exists
    if os.path.exists('token.json'):
        print("Found existing token.json")
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if creds and creds.valid:
                print("Token is valid")
                return True
        except Exception as e:
            print(f"Error reading token.json: {e}")
    
    # Check if credentials.json exists
    if os.path.exists('credentials.json'):
        print("Found credentials.json")
        try:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            print("Successfully loaded credentials.json")
            return True
        except Exception as e:
            print(f"Error reading credentials.json: {e}")
    else:
        print("credentials.json not found")
    
    return False

if __name__ == '__main__':
    result = test_gmail_auth()
    print(f"\nAuthentication test result: {'Success' if result else 'Failed'}")
