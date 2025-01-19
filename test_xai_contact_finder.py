import requests
from dotenv import load_dotenv
import os
import time

load_dotenv()

def query_xai(max_retries=3, retry_delay=1):
    xai_token = os.getenv('XAI_TOKEN')
    xai_api_url = os.getenv('XAI_API_URL')
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts contact information from webpages. When asked about specific roles, only return information that is explicitly stated on the page."
            },
            {
                "role": "user",
                "content": "Who is the Director of Food and Beverage from this webpage? What is their name and email address? https://www.ccroswell.com/web/pages/contact-us"
            }
        ],
        "model": "grok-2-1212",  # Ensure this model version is stable
        "stream": False,
        "temperature": 0  # Set to 0 for maximum determinism
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                xai_api_url,
                json=payload,
                headers={"Authorization": f"Bearer {xai_token}"},
                timeout=10  # Timeout to catch hanging requests
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Here, we check if the content matches our expected response
                expected_response = "The Director of Food and Beverage from the Country Club of Roswell's contact page is **Jason Kramer**. His email address is **jasonkramer@ccroswell.com**."
                if content.strip() == expected_response.strip():
                    print(f"\nResponse: {content}")
                    return True
                else:
                    print(f"\nUnexpected response: {content}")
                    print(f"Expected: {expected_response}")
                    return False
            else:
                print(f"Request failed with status code: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue

        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed due to: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                return False

    print("All retry attempts failed.")
    return False

if __name__ == "__main__":
    query_xai()