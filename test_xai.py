import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def call_xai(club_name, city, state):
    """
    Simulate an xAI call for club amenities information
    """
    
    # Get credentials from environment variables
    xai_token = os.getenv('XAI_TOKEN')
    xai_api_url = os.getenv('XAI_API_URL')
    #xai_model = os.getenv('XAI_MODEL')
    xai_model = "grok-2-1212"
    
    # Construct the payload
    payload = {
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant that provides a brief overview of a club's location and amenities."
        }, {
            "role": "user",
            "content": f"Please provide a concise overview about {club_name} in {city}, {state}. Is it private or public? Keep it to under 3 sentences."
        }],
        "model": xai_model,
        "stream": False,
        "temperature": 0.0
    }

    # Log the request
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{timestamp} - DEBUG - xAI request payload={json.dumps(payload)}")

    try:
        response = requests.post(
            xai_api_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {xai_token}"
            }
        )
        
        # Log the response
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        print(f"{timestamp} - DEBUG - xAI response={response.text}")
        
        return response.text

    except Exception as e:
        print(f"Error calling xAI: {str(e)}")
        return None

def main():
    # Test cases
    clubs = [
        {
            "name": "Starfire Golf Club",
            "city": "Scottsdale",
            "state": "AZ"
        },
        {
            "name": "Pine Valley Golf Club",
            "city": "Pine Valley",
            "state": "NJ"
        }
        # Add more test cases as needed
    ]
    
    for club in clubs:
        print("\nTesting club:", club["name"])
        print("-" * 50)
        response = call_xai(club["name"], club["city"], club["state"])
        print("-" * 50)

if __name__ == "__main__":
    main()