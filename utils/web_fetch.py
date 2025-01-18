import requests
from typing import Optional
from utils.logging_setup import logger
import urllib3
import random
from urllib.parse import urlparse, urlunparse

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of common user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

def sanitize_url(url: str) -> str:
    """Sanitize and normalize URL format."""
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    if not parsed.netloc.startswith("www."):
        parsed = parsed._replace(netloc=f"www.{parsed.netloc}")
    return urlunparse(parsed)

def fetch_website_html(url: str) -> Optional[str]:
    """Fetch HTML content from a website with proper headers and error handling."""
    if not url:
        logger.error("No URL provided")
        return None
        
    # Clean up the URL
    url = url.strip().lower()
    logger.debug(f"Original URL: {url}")
    
    # Generate URL variations
    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc.replace('www.', '')
    
    urls_to_try = [
        f"https://www.{base_domain}{parsed_url.path}",
        f"https://{base_domain}{parsed_url.path}",
        f"http://www.{base_domain}{parsed_url.path}",
        f"http://{base_domain}{parsed_url.path}"
    ]
    
    logger.debug(f"Will try URLs: {urls_to_try}")
    
    # Setup headers with additional browser-like headers
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
            logger.debug(f"Trying to fetch: {attempt_url}")
            
            response = session.get(
                attempt_url,
                headers=headers,
                timeout=10,
                verify=False,  # Still keeping verify=False for testing
                allow_redirects=True
            )
            
            logger.debug(f"Response code: {response.status_code} for URL: {attempt_url}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Check if we got a successful response
            if response.status_code == 200:
                content = response.text
                logger.debug(f"Content preview: {content[:200]}")
                return content
            else:
                logger.debug(f"Failed with status code {response.status_code}")
                if response.text:
                    logger.debug(f"Error response preview: {response.text[:200]}")
                
        except requests.RequestException as e:
            logger.error(f"Error fetching {attempt_url}: {str(e)}", exc_info=True)
            continue
            
    logger.error(f"Failed to fetch website content after trying multiple URLs for {url}")
    return None
