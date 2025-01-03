# scripts/job_title_categories.py

def categorize_job_title(title: str) -> str:
    """
    Categorizes job titles to map to one of four templates:
    - general_manager_initial_outreach.md
    - fb_manager_initial_outreach.md
    - golf_ops_initial_outreach.md
    - fallback.md (default)
    """
    title = title.lower().strip()
    
    # General Manager Category
    if any(term in title for term in [
        'general manager', 'gm', 'club manager', 
        'director of operations', 'coo', 'president', 
        'owner', 'ceo', 'chief executive'
    ]):
        return 'general_manager'
        
    # F&B Category
    if any(term in title for term in [
        'f&b', 'food', 'beverage', 'restaurant', 
        'dining', 'hospitality', 'culinary'
    ]):
        return 'fb_manager'
        
    # Golf Operations Category
    if any(term in title for term in [
        'golf', 'pro shop', 'course', 'professional',
        'head pro', 'assistant pro', 'director of golf'
    ]):
        return 'golf_ops'
    
    # Default to fallback template
    return 'fallback'
