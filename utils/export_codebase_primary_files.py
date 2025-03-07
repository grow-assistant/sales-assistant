import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    # Normalize path separators 
    filepath = filepath.replace('\\', '/')
    # List of primary files focused on email template building and placeholder replacement
    primary_files = [
        'scheduling/database.py',
        'scheduling/followup_generation.py', 
        'scripts/golf_outreach_strategy.py',
        'services/gmail_service.py',
        'services/leads_service.py',
        'tests/test_followup_generation.py',
        'tests/test_hubspot_leads_service.py'

    ]
    # Check if file is in primary files list
    for primary_file in primary_files:
        if filepath.endswith(primary_file):
            return True
            
    return False

def get_file_content(filepath):
    """Read and return file content with proper markdown formatting."""
    try:
        # Try UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
    except UnicodeDecodeError:
        try:
            # Fallback to cp1252 if UTF-8 fails
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If both fail, try with errors='ignore'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(__file__)))
    
    return f"""
## {rel_path}

{content}
"""

def export_files(output_path='exported_codebase.md'):
    """Export all primary files to a markdown file."""
    all_content = []
    
    # Get list of all files in project
    for root, _, files in os.walk(os.path.dirname(os.path.dirname(__file__))):
        for file in files:
            filepath = os.path.join(root, file)
            filepath = filepath.replace('\\', '/')
            
            if should_include_file(filepath):
                content = get_file_content(filepath)
                all_content.append(content)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_content))
    
    return output_path

if __name__ == '__main__':
    output_file = export_files()
    print(f'Codebase exported to: {output_file}')
