import requests
from typing import Optional
from utils.logging_setup import logger
import urllib3
import random
from urllib.parse import urlparse, urlunparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

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

def fetch_website_html(url: str, timeout: int = 10, retries: int = 3) -> str:
    """
    Fetch HTML content from a website with retries and better error handling.
    
    Args:
        url: The URL to fetch
        timeout: Timeout in seconds for each attempt
        retries: Number of retry attempts
    """
    logger = logging.getLogger(__name__)
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(
            url,
            timeout=timeout,
            verify=False,  # Skip SSL verification
            allow_redirects=True
        )
        response.raise_for_status()
        return response.text
    
    except requests.exceptions.ConnectTimeout:
        logger.warning(f"Connection timed out for {url} - skipping website fetch")
        return ""
    except requests.exceptions.ReadTimeout:
        logger.warning(f"Read timed out for {url} - skipping website fetch")
        return ""
    except requests.exceptions.SSLError:
        # Try again without SSL
        logger.warning(f"SSL error for {url} - attempting without SSL verification")
        try:
            response = session.get(
                url.replace('https://', 'http://'),
                timeout=timeout,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url} without SSL: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return ""
    finally:
        session.close()
