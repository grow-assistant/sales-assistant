import os
from pathlib import Path
import sys
from typing import List, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change relative import to absolute import
from utils.exceptions import ConfigurationError

def get_template_paths() -> Dict[str, List[str]]:
    """
    Gets all template directory and file paths under the docs/templates directory.
    
    Returns:
        Dict[str, List[str]]: Dictionary with keys 'directories' and 'files' containing
                             lists of all template directory and file paths respectively
    
    Raises:
        ConfigurationError: If templates directory cannot be found
    """
    # Get the project root directory (parent of utils/)
    root_dir = Path(__file__).parent.parent
    templates_dir = root_dir / "docs" / "templates"
    
    if not templates_dir.exists():
        raise ConfigurationError(
            "Templates directory not found",
            {"expected_path": str(templates_dir)}
        )

    template_paths = {
        "directories": [],
        "files": []
    }
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(templates_dir):
        # Convert to relative paths from project root
        rel_root = os.path.relpath(root, root_dir)
        
        # Add directory paths
        template_paths["directories"].append(rel_root)
        
        # Add file paths
        for file in files:
            rel_path = os.path.join(rel_root, file)
            template_paths["files"].append(rel_path)
            
    return template_paths

if __name__ == "__main__":
    try:
        print("\nScanning for templates...")
        paths = get_template_paths()
        
        print("\nTemplate Directories:")
        for directory in paths["directories"]:
            print(f"  - {directory}")
            
        print("\nTemplate Files:")
        for file in paths["files"]:
            print(f"  - {file}")
            
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e.message}")
        if e.details:
            print("Details:", e.details)
    except Exception as e:
        print(f"\nError: {str(e)}")
