# scripts/job_title_categories.py

def categorize_job_title(title: str) -> str:
    """Categorize job title into standardized roles."""
    if not title:
        return "general_manager"  # Default fallback
    
    title = title.lower().strip()
    
    # F&B Related Titles
    fb_titles = [
        "f&b", "food", "beverage", "dining", "restaurant", "culinary",
        "chef", "kitchen", "hospitality", "catering", "banquet"
    ]
    
    # Membership Related Titles
    membership_titles = [
        "member", "membership", "marketing", "sales", "business development"
    ]
    
    # Golf Operations Titles
    golf_ops_titles = [
        "golf pro", "pro shop", "golf operations", "head pro",
        "director of golf", "golf director", "pga", "golf professional"
    ]
    
    # Check categories in order of specificity
    for fb_term in fb_titles:
        if fb_term in title:
            return "fb_manager"
            
    for membership_term in membership_titles:
        if membership_term in title:
            return "membership_director"
            
    for golf_term in golf_ops_titles:
        if golf_term in title:
            return "golf_operations"
    
    # Default to general manager for other titles
    return "general_manager"
