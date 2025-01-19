import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    # Normalize path separators
    filepath = filepath.replace('\\', '/')
    # List of primary files to include based on most frequently referenced
    primary_files = [
        'main.py',
        'scripts/golf_outreach_strategy.py',
        'scheduling/database.py',
        'scheduling/extended_lead_storage.py', 
        'scheduling/followup_scheduler.py',
        'scheduling/followup_generation.py',
        'utils/gmail_integration.py',
        'utils/xai_integration.py',
        'scripts/build_template.py'
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
            
            # For leads_list.csv, only include top 10 records
            if filepath.endswith('leads_list.csv'):
                lines = content.splitlines()
                content = '\n'.join(lines[:11]) # Header + 10 records
                
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
