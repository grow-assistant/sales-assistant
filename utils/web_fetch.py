import requests
from typing import Optional
from utils.logging_setup import logger
import urllib3
import random

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of common user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

def fetch_website_html(url: str) -> Optional[str]:
    """Fetch HTML content from a website with proper headers and error handling."""
    if not url:
        return None
        
    # Clean up the URL
    url = url.strip().lower()
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    # Setup headers
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
        'Cache-Control': 'max-age=0'
    }
    
    urls_to_try = [
        url,
        url.replace('https://', 'http://'),
        url.replace('http://', 'https://'),
        url.replace('://', '://www.'),
        url.replace('://www.', '://')
    ]
    
    for attempt_url in urls_to_try:
        try:
            response = requests.get(
                attempt_url,
                headers=headers,
                timeout=10,
                verify=False,  # Disable SSL verification
                allow_redirects=True
            )
            
            # Check if we got a successful response
            if response.status_code == 200:
                return response.text
            else:
                logger.debug(f"Failed to fetch {attempt_url} with status code {response.status_code}")
                
        except requests.RequestException as e:
            logger.debug(f"Error fetching {attempt_url}: {str(e)}")
            continue
            
    logger.error(f"Failed to fetch website content after trying multiple URLs for {url}")
    return None
