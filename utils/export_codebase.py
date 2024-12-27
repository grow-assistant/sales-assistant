import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    # Normalize path separators
    filepath = filepath.replace('\\', '/')
    
    # Exclude patterns
    exclude_patterns = [
        '*/__pycache__/*',
        '*/test_*',
        '*/.git/*',
        '*.pyc',
        '*.log',
        '*.env*',
        '*.yml',
        '*.txt',
        '*.json',
        'config/*',
    ]
    
    # Check if file matches any exclude pattern
    for pattern in exclude_patterns:
        if glob.fnmatch.fnmatch(filepath, pattern):
            return False
    
    # Include only Python files from core functionality
    include_dirs = [
        'agents',
        'services',
        'external',
        'utils',
        'scripts',
        'hubspot_integration',
        'scheduling'
    ]
    
    # More flexible directory matching
    for dir_name in include_dirs:
        if f'/{dir_name}/' in filepath or filepath.startswith(f'{dir_name}/'):
            return True
            
    # Include main.py
    if filepath.endswith('main.py'):
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
```python
{content}
```
"""

def export_codebase(root_dir, output_file):
    """Export core codebase to a markdown file."""
    # Get all Python files
    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if should_include_file(filepath):
                    all_files.append(filepath)
    
    # Sort files for consistent output
    all_files.sort()
    
    # Generate markdown content
    content = """# Sales Assistant Codebase

This document contains the core functionality of the Sales Assistant project.

## Table of Contents
"""
    
    # Add TOC
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, start=root_dir)
        content += f"- [{rel_path}](#{rel_path.replace('/', '-')})\n"
    
    # Add file contents
    for filepath in all_files:
        content += get_file_content(filepath)
    
    # Write output file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'codebase_export.md')
    
    export_codebase(project_root, output_path)
    print(f"Codebase exported to: {output_path}")
