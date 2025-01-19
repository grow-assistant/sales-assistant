import os
import re
import random
import requests
import openai
import urllib3
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Comment
from typing import Optional
from urllib.parse import urlparse

# Disable SSL verification warnings (useful for development; remove for production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of common user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

def decode_cloudflare_email(encoded_email: str) -> str:
    """
    Decode Cloudflare-protected email addresses.
    Encoded emails are in the format: <a class="__cf_email__" data-cfemail="encoded_value"></a>
    """
    try:
        # Convert hex-encoded string into readable email
        hex_str = encoded_email[2:]  # Remove the "0x" prefix
        r = int(hex_str[:2], 16)  # First byte is the XOR key
        email = ''.join(chr(int(hex_str[i:i + 2], 16) ^ r) for i in range(2, len(hex_str), 2))
        return email
    except Exception as e:
        print(f"[DEBUG] Failed to decode Cloudflare email: {e}")
        return ""

def fetch_website_html(url: str) -> Optional[str]:
    """
    Fetch HTML content from a website with proper headers and error handling.
    Returns None if unable to fetch from all URL variations.
    """
    if not url:
        print("Error: No URL provided")
        return None

    # Clean up the URL
    url = url.strip().lower()
    print(f"[DEBUG] Original URL: {url}")

    # Parse base domain
    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc.replace('www.', '')

    # Try a few protocol/domain variations
    urls_to_try = [
        f"https://www.{base_domain}{parsed_url.path}",
        f"https://{base_domain}{parsed_url.path}",
        f"http://www.{base_domain}{parsed_url.path}",
        f"http://{base_domain}{parsed_url.path}"
    ]
    print(f"[DEBUG] Will try URLs: {urls_to_try}")

    # Setup headers with a random user agent
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',  # Do Not Track
        'Pragma': 'no-cache'
    }

    session = requests.Session()

    for attempt_url in urls_to_try:
        try:
            print(f"[DEBUG] Trying to fetch: {attempt_url}")
            response = session.get(
                attempt_url,
                headers=headers,
                timeout=10,
                verify=False,  # skip SSL verification (not recommended for production)
                allow_redirects=True
            )

            print(f"[DEBUG] Response code: {response.status_code} for URL: {attempt_url}")

            if response.status_code == 200:
                content = response.text
                print(f"[DEBUG] Successfully fetched content from: {attempt_url}")
                return content
            else:
                print(f"[DEBUG] Non-200 status code: {response.status_code}")

        except requests.RequestException as e:
            print(f"Error fetching {attempt_url}: {str(e)}")
            continue

    print(f"Failed to fetch website content after trying multiple variations of {url}")
    return None

def clean_html_content(html_content: str) -> str:
    """
    Aggressively clean HTML content to focus only on essential contact information.
    """
    if not html_content:
        return ""

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # First pass: Find and decode any Cloudflare-protected emails
        for email_elem in soup.find_all("a", class_="__cf_email__"):
            if 'data-cfemail' in email_elem.attrs:
                decoded_email = decode_cloudflare_email(email_elem['data-cfemail'])
                if decoded_email:
                    email_elem.replace_with(decoded_email)

        # Remove all unnecessary elements
        for element in soup.find_all(['script', 'style', 'meta', 'link', 'nav', 
                                    'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()

        # Find contact-specific sections first
        contact_keywords = ['contact', 'staff', 'team', 'directory', 'personnel']
        contact_content = None

        # Try to find contact section by ID or class
        for keyword in contact_keywords:
            contact_content = (
                soup.find(id=re.compile(keyword, re.I)) or
                soup.find(class_=re.compile(keyword, re.I)) or
                soup.find('section', string=re.compile(keyword, re.I)) or
                soup.find('div', string=re.compile(keyword, re.I))
            )
            if contact_content:
                break

        if not contact_content:
            # Fallback to main content
            contact_content = soup.find('main') or soup.find('body') or soup

        # Extract text content
        text_content = []
        
        # Extract all text nodes and their parent tags
        for element in contact_content.stripped_strings:
            text = element.strip()
            if text and len(text) > 1:  # Ignore single characters
                text_content.append(text)

        # Join with spaces and clean up
        text = ' '.join(text_content)

        # Preserve email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'&[a-zA-Z]+;', '', text)  # Remove HTML entities
        
        # Clean phone numbers while preserving structure
        text = re.sub(r'(?<!\d)(\d{3})[)\s.-]+(\d{3})[)\s.-]+(\d{4})(?!\d)', 
                     r'\1-\2-\3', text)

        # Remove navigation-like short phrases
        text = ' '.join(line for line in text.split() 
                       if len(line) > 2 or line in ['of', 'at'])

        # Reinsert emails
        for email in emails:
            if email not in text:
                text += f" {email}"

        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Debug output
        print(f"\n[DEBUG] Found emails: {emails}")
        print(f"[DEBUG] Final text length: {len(text)}")

        return text

    except Exception as e:
        print(f"Error cleaning HTML: {str(e)}")
        return ""

def test_contact_info_search():
    """
    Test OpenAI contact information extraction functionality.
    """
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not set in environment.")
        return

    url = "https://www.ccroswell.com/web/pages/contact-us"
    print(f"\nAttempting to fetch content from: {url}")

    html_content = fetch_website_html(url)
    if not html_content:
        print("Failed to fetch website content")
        return

    cleaned_content = clean_html_content(html_content)
    
    # More focused system prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise contact information extractor. When asked about staff, "
                "return ONLY the exact name and email address in the specified format. "
                "If information is missing or unclear, respond with 'Information not found'."
            )
        },
        {
            "role": "user",
            "content": (
                "Who is the Director of Food and Beverage? "
                "Format your response exactly like this:\n"
                "Name: [full name]\n"
                "Email: [email address]"
            )
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.0  # Use 0 temperature for most consistent results
        )

        if response.choices:
            print("\nOpenAI Response:")
            print("-" * 50)
            print(response.choices[0].message.content)
            print("-" * 50)
        else:
            print("No response from OpenAI")

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")

def main():
    test_contact_info_search()

if __name__ == "__main__":
    main()
