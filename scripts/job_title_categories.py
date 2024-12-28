# scripts/job_title_categories.py

def categorize_job_title(title: str) -> str:
    """
    Categorizes job titles into standardized roles for template selection.
    
    Args:
        title: The job title string to categorize
        
    Returns:
        str: Standardized role category (e.g., 'general_manager', 'food_beverage', etc.)
    """
    title = title.lower().strip()
    
    # General Manager / Director Categories
    if any(term in title for term in ['general manager', 'gm', 'club manager', 'director of operations']):
        return 'general_manager'
        
    # F&B Categories
    if any(term in title for term in ['f&b', 'food', 'beverage', 'restaurant', 'dining', 'hospitality']):
        return 'food_beverage'
        
    # Golf Professional Categories
    if any(term in title for term in ['golf pro', 'golf professional', 'head pro', 'director of golf']):
        return 'golf_professional'
        
    # Owner/President Categories
    if any(term in title for term in ['owner', 'president', 'ceo', 'chief executive']):
        return 'owner'
        
    # Membership Categories
    if any(term in title for term in ['membership', 'member services']):
        return 'membership'
        
    # Default to general manager template if unknown
    return 'general_manager'
