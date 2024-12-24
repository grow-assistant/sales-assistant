# scripts/job_title_categories.py

def categorize_job_title(job_title_str: str) -> str:
    """
    Map a raw job title to a simplified category string,
    which we then use to pick an outreach email template.
    """
    title = (job_title_str or "").lower().strip()

    # 1) Food & Beverage
    if any(keyword in title for keyword in ["food", "f&b", "beverage"]):
        return "fnb_manager"
    
    # 2) Golf Operations
    if any(keyword in title for keyword in ["golf", "director of golf", "head golf professional"]):
        return "golf_ops"
    
    # 3) General Manager/COO (catch “Manager,” “GM,” “COO,” “Assistant General Manager,” etc.)
    if "manager" in title or "gm" in title or "coo" in title:
        return "general_manager"
    
    # 4) Executive or Owner (CEO, President, Owner, Founder, CFO, etc.)
    if any(keyword in title for keyword in [
        "owner", "president", "ceo", "chief executive",
        "chief financial", "cfo", "founder", "vice president",
        "partner", "chief revenue officer"
    ]):
        return "executive_or_owner"

    # 5) Fallback / Default
    return "fallback"
