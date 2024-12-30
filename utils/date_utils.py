from datetime import datetime, timedelta

def convert_to_club_timezone(dt, state_offsets):
    """
    Adjusts datetime based on state's hour offset.
    
    Args:
        dt: datetime object to adjust
        state_offsets: dict with 'dst' and 'std' hour offsets
    """
    if not state_offsets:
        return dt
        
    # Determine if we're in DST (simple check - could be enhanced)
    is_dst = datetime.now().month in [3,4,5,6,7,8,9,10]
    offset_hours = state_offsets['dst'] if is_dst else state_offsets['std']
    
    # Apply offset
    return dt + timedelta(hours=offset_hours) 