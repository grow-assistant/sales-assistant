import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_club_info_search():
    """
    Test xAI club info search functionality
    """
    # Test data
    club_name = "Riverwood Golf Club"
    city = "Port Charlotte"
    state = "FL"
    
    # Get credentials from environment variables
    xai_token = os.getenv('XAI_TOKEN')
    xai_api_url = os.getenv('XAI_API_URL')
    xai_model = "grok-2-1212"
    
    # Construct the payload
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a factual assistant that provides objective, data-focused overviews of golf clubs and country clubs. "
                "Focus only on verifiable facts like location, type (private/public), number of holes, and amenities. "
                "Do not mention any amenities that are not explicitly confirmed."
            },
            {
                "role": "user", 
                "content": f"""
Please provide a brief overview of {club_name} located in {city}, {state}. Include key facts such as:
- Type of facility (private, semi-private, public, resort, country club, etc.) 
- Notable amenities or features (only mention confirmed amenities)
- Any other relevant information

Also, please classify the facility into one of the following types at the end of your response:
- Private Course
- Semi-Private Course 
- Public Course
- Country Club
- Resort
- Other

Provide the classification on a new line starting with "Facility Type: ".
"""
            }
        ],
        "model": xai_model,
        "stream": False,
        "temperature": 0.2
    }

    try:
        response = requests.post(
            xai_api_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {xai_token}"
            }
        )
        
        # Only print the response content
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print("\nxAI Response:")
            print("-" * 50)
            print(content)
            print("-" * 50)
        
        return response.status_code == 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_club_info_search()