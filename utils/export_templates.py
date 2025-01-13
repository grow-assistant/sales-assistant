import os
import glob
from pathlib import Path

def should_include_file(filepath):
    """Determine if a file should be included in the export."""
    filepath = filepath.replace('\\', '/')
    return 'docs/templates/country_club' in filepath

def get_file_content(filepath):
    """Read and return file content with proper markdown formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.dirname(__file__)))
    extension = os.path.splitext(filename)[1].lower()
    lang = 'markdown' if extension == '.md' else 'text'
    
    return f"## {rel_path}\n```{lang}\n{content}\n```\n"

def export_codebase(root_dir, output_file):
    """Export country club templates to a markdown file."""
    all_files = []
    country_club_path = os.path.join(root_dir, 'docs', 'templates', 'country_club')
    
    for root, _, files in os.walk(country_club_path):
        for file in files:
            filepath = os.path.join(root, file)
            if should_include_file(filepath):
                all_files.append(filepath)
    
    all_files.sort()
    
    content = "# Country Club Email Templates\n\nThis document contains all country club email templates.\n\n## Table of Contents\n"
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, start=root_dir)
        content += f"- [{rel_path}](#{rel_path.replace('/', '-')})\n"
    
    for filepath in all_files:
        content += get_file_content(filepath)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'country_club_templates.md')
    
    export_codebase(project_root, output_path)
    print(f"Country club templates exported to: {output_path}")
