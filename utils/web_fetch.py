import requests
from utils.logging_setup import logger

def fetch_website_html(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text
        else:
            logger.error("Failed to fetch website %s: %s", url, resp.text)
            return ""
    except requests.RequestException as e:
        logger.error("Error fetching website %s: %s", url, str(e))
        return ""
