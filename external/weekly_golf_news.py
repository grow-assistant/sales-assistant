# external/weekly_golf_news.py

import os
import sys
import time
import feedparser
import logging
import requests
import pandas as pd
import openai
import json
from utils.formatting_utils import extract_text_from_html
from datetime import datetime

# If you have a .env, load it:
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Setup logging (already done in utils.logging_setup, but kept here for clarity)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_golf_news():
    """
    Fetch RSS feeds, store basic article metadata to JSON.
    Skips or removes known-bad feeds, shortens request timeout, 
    and logs before/after each feed to pinpoint slowdowns.
    """

    logger.info(f"Fetching golf news... {datetime.now()}")

    # Feeds that were working or partially working:
    feeds = {
        #"Golf.com": "https://golf.com/feed",
        #"Golf Business News": "https://golfbusinessnews.com/feed",
        #"Golf Course Industry": "https://www.golfcourseindustry.com/rss",
        #"Golf News Magazine": "https://www.golfnews.co.uk/feed",
        "Golf One Media": "https://www.golfonemedia.com/feed"
        #"Golf Canada": "https://www.golfcanada.ca/feed",
        #"Golf365": "https://www.golf365.com/feed",
        #"Golf Content Network": "https://www.golfcontentnetwork.com/feed",
        #"Golf Industry Central": "https://www.golfindustrycentral.com.au/feed",
        #"Global Golf Post": "https://www.globalgolfpost.com/feed",
        #"Inside Golf": "https://www.insidegolf.com.au/feed",
        # FEEDS COMMENTED OUT DUE TO ERRORS (404, 403, SSL, etc.):
        # "Golf Channel": "https://www.nbcsports.com/feed",     # 404
        # "Women's Golf": "https://womensgolf.com/feed",        # 500
        # "Golf Digest": "https://www.golfdigest.com/feed",     # 403
        # "Golf Week": "https://golfweek.com/feed",             # 404
        # "National Club Golfer": "https://www.nationalclubgolfer.com/feed", # Timeout
        # "JC Golf Blog": "https://www.jcgolf.com/news",        # Malformed
        # "Irish Golf Desk": "https://www.irishgolfdesk.com/news-files", # Syntax error
        # "Twin Cities Golf": "https://www.twincitiesgolf.com/news",      # Syntax error
        # "Mizuno Golf Europe": "https://golf.mizunoeurope.com/blog/feed" # SSL Cert expired
    }

    # Custom headers to reduce 403 issues
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/117.0.5938.62 Safari/537.36'
        )
    }

    all_articles = []
    for source, url in feeds.items():
        logger.info(f"Fetching feed from {source}: {url} ...")
        try:
            # Optional: verify=False to skip SSL checks (not recommended in production)
            response = requests.get(url, headers=headers, timeout=5, verify=True)
            response.raise_for_status()

            feed = feedparser.parse(response.content)
            if feed.bozo == 1:
                logger.warning(f"Feed parsing error: {source} => {feed.bozo_exception}")
                continue

            for entry in feed.entries[:10]:
                all_articles.append({
                    'source': source,
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', '')[:500],
                    'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        except requests.exceptions.HTTPError as he:
            logger.error(f"HTTP Error fetching {source} ({url}): {he}")
        except requests.exceptions.RequestException as re:
            logger.error(f"Request Error fetching {source} ({url}): {re}")
        except Exception as e:
            logger.error(f"Unknown error fetching {source} ({url}): {e}")
        finally:
            # Always log after each feed
            logger.info(f"Finished attempting {source}.\n")

        # Brief pause between feeds
        time.sleep(1)

    if not all_articles:
        logger.warning("No articles fetched. Possibly all feeds failed.")
        return None

    json_path = os.path.join(os.path.dirname(__file__), 'golf_news.json')
    with open(json_path, 'w') as f:
        json.dump(all_articles, f, indent=2)
    logger.info(f"Golf news articles saved to {json_path}")

    return pd.DataFrame(all_articles)


def fetch_article_text(url: str) -> str:
    """
    Fetch HTML text from an article URL, removing scripts/styles.
    """
    try:
        logger.info(f"Fetching article content: {url}")
        resp = requests.get(url, timeout=5, verify=True)
        resp.raise_for_status()

        return extract_text_from_html(resp.text, preserve_newlines=True)
    except Exception as e:
        logger.error(f"Error fetching article text ({url}): {e}")
        return ""


def summarize_and_extract_clubs(article_text: str) -> dict:
    """
    Use OpenAI to summarize the article text and identify clubs mentioned.
    Returns {"summary": "...", "clubs": [...]}
    """
    if not article_text.strip():
        return {"summary": "No article text available.", "clubs": []}

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Read the given text about golf news "
            "and produce the following:\n"
            "1. A concise 1-2 paragraph summary.\n"
            "2. A list of any golf clubs mentioned (by name). If none, say 'none found'."
        )
    }
    user_prompt = {
        "role": "user",
        # Truncate large articles to avoid token overload
        "content": article_text[:7000]
    }

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",    # or "gpt-3.5-turbo"
            messages=[system_prompt, user_prompt],
            max_tokens=400,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()

        # Minimal parse: separate summary from clubs
        lines = answer.split("\n")
        summary_lines = []
        clubs_lines = []
        clubs_section = False

        for line in lines:
            lower_line = line.lower()
            # Check if we've reached clubs heading
            if "club" in lower_line and "mentioned" in lower_line:
                clubs_section = True
                continue

            if not clubs_section:
                summary_lines.append(line)
            else:
                clubs_lines.append(line)

        summary = "\n".join(summary_lines).strip() or answer
        clubs = [c.strip("-* ").strip() for c in clubs_lines if c.strip()]

        if not clubs:
            clubs = ["none found"]

        return {"summary": summary, "clubs": clubs}

    except Exception as e:
        logger.error(f"OpenAI summarization error: {e}")
        return {"summary": "Summarization failed.", "clubs": []}


def summarize_all_articles():
    """
    Reads 'golf_news.json', fetches each article's content, 
    and uses OpenAI to get summary + clubs. Saves results back to JSON.
    """
    json_path = os.path.join(os.path.dirname(__file__), 'golf_news.json')
    if not os.path.exists(json_path):
        logger.error(f"No JSON found at {json_path}. Run fetch_golf_news first.")
        return

    with open(json_path, 'r') as f:
        articles = json.load(f)

    for article in articles:
        url = article.get('link', '')
        article_text = fetch_article_text(url)
        result = summarize_and_extract_clubs(article_text)
        article['openai_summary'] = result["summary"]
        article['clubs_mentioned'] = result["clubs"]
        time.sleep(1)  # brief pause to avoid OpenAI rate limiting

    with open(json_path, 'w') as f:
        json.dump(articles, f, indent=2)
    logger.info("Updated JSON with summaries and clubs mentioned.")
    return pd.DataFrame(articles)


if __name__ == "__main__":
    # Example manual run:
    # 1) Fetch the latest news
    df_fetched = fetch_golf_news()
    # 2) Summarize articles (requires OPENAI_API_KEY)
    if df_fetched is not None:
        summarize_all_articles()
