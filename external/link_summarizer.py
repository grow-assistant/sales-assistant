# external/link_summarizer.py
import requests
from utils.formatting_utils import extract_text_from_html
import openai
from typing import Dict, List
from utils.logging_setup import logger
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def fetch_page_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return extract_text_from_html(resp.text, preserve_newlines=True)
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return ""

def summarize_text(text: str) -> str:
    if not text:
        return "No content to summarize."
    system_prompt = {
        "role": "system",
        "content": "You are a concise assistant that summarizes webpage content."
    }
    user_prompt = {
        "role": "user",
        "content": f"Summarize this webpage:\n\n{text}"
    }
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[system_prompt, user_prompt],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "Error summarizing text."

def summarize_recent_news(recent_news_list: List[Dict]) -> Dict[str, str]:
    summaries = {}
    for item in recent_news_list:
        link = item.get("link", "")
        if not link:
            summaries[link] = "No link available."
            continue
        text = fetch_page_text(link)
        if not text:
            summaries[link] = "No content or error fetching the page."
            continue
        summaries[link] = summarize_text(text)
    return summaries
