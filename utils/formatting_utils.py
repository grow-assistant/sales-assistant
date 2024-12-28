"""
Utility functions for text formatting and cleaning.
"""

import re
from bs4 import BeautifulSoup
from typing import Optional

def clean_phone_number(raw_phone: str) -> str:
    """
    Example phone cleaning logic:
    1) Remove non-digit chars
    2) Format as needed (e.g., ###-###-####)
    """
    if raw_phone is None:
        return None
        
    digits = "".join(char for char in raw_phone if char.isdigit())
    if len(digits) == 10:
        # e.g. (123) 456-7890
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    else:
        return digits

def clean_html(raw_html: str, strip_tags: bool = True, remove_scripts: bool = True) -> str:
    """
    Clean HTML content by removing tags and/or unwanted elements.
    
    Args:
        raw_html: Raw HTML string to clean
        strip_tags: If True, removes all HTML tags
        remove_scripts: If True, removes script and style tags before processing
    
    Returns:
        Cleaned text string
    """
    if not raw_html:
        return ""
    
    if remove_scripts:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        raw_html = str(soup)
    
    if strip_tags:
        # Remove HTML tags while preserving content
        text = re.sub('<[^<]+?>', '', raw_html)
        # Decode HTML entities
        text = BeautifulSoup(text, "html.parser").get_text()
        return text.strip()
    
    return raw_html.strip()


def extract_text_from_html(html_content: str, preserve_newlines: bool = False) -> str:
    """
    Extract readable text from HTML content, removing scripts and styling.
    Useful for content analysis and summarization.
    
    Args:
        html_content: HTML content to process
        preserve_newlines: If True, uses newlines as separator, else uses space
    
    Returns:
        Extracted text content
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Get text content
    separator = "\n" if preserve_newlines else " "
    text = soup.get_text(separator=separator, strip=True)
    
    # Remove excessive whitespace while preserving single newlines if needed
    if preserve_newlines:
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        # Replace all whitespace (including newlines) with single space
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
