"""
Purpose:
Recursively analyzes Python files to find every local file that main.py depends on,
including those imported via dotted module names (e.g., services.hubspot_service).

Key functions:
- find_python_file(): Locates a Python file in the directory and subdirectories
- extract_imports(): Parses a Python file to find import statements, preserving dotted paths
- get_recursive_dependencies(): Builds a dependency tree recursively from main.py
- print_dependency_chain(): Prints all files in the dependency chain of main.py

Usage:
1. Place this script anywhere within your project's directory structure.
2. Adjust `project_root` in main() if your project layout differs.
3. Run the script directly to see a list of Python files that main.py depends on.
"""

import os
import ast
from typing import Set, Dict

def find_python_file(filename: str, start_path: str = '.') -> str:
    """
    Search for a specific Python file within the given directory and its subdirectories.
    Returns the first matching path or None if not found.
    """
    for root, _, files in os.walk(start_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def extract_imports(file_path: str) -> Set[str]:
    """
    Extract all import statements from a Python file, preserving the full dotted module path.
    For example: 'from services.hubspot_service import HubspotService'
    will yield "services.hubspot_service" rather than just "services".
    """
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # 'import x' statements (x can be dotted, e.g., something.more)
                    for name in node.names:
                        imports.add(name.name)
                
                elif isinstance(node, ast.ImportFrom):
                    # 'from x.y.z import A, B, C' statements
                    if node.module:
                        imports.add(node.module)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return imports

def get_recursive_dependencies(file_path: str, start_path: str = '.', processed_files: Set[str] = None) -> Dict[str, Set[str]]:
    """
    Recursively builds a dependency mapping from the given file.
    The dictionary looks like: {relative_file_path: {set_of_imported_modules_as_strings}}.
    """
    if processed_files is None:
        processed_files = set()
    
    dependency_map = {}
    
    # If we've already processed this file, skip to avoid cycles
    if file_path in processed_files:
        return dependency_map
    
    processed_files.add(file_path)
    
    # Extract direct imports from this file
    dependencies = extract_imports(file_path)
    relative_path = os.path.relpath(file_path, start_path)
    dependency_map[relative_path] = dependencies
    
    # For each import, see if it maps to a local file
    for dep in dependencies:
        # Convert dotted module path to a relative file path (services.hubspot_service -> services/hubspot_service.py)
        dep_filename = dep.replace('.', os.sep) + ".py"
        dep_path = find_python_file(dep_filename, start_path)
        
        if dep_path and dep_path not in processed_files:
            nested_deps = get_recursive_dependencies(dep_path, start_path, processed_files)
            dependency_map.update(nested_deps)
    
    return dependency_map

def print_dependency_chain(dependency_map: Dict[str, Set[str]]) -> None:
    """
    Prints all files in the dependency chain (including main.py),
    along with the modules each file directly imports.
    """
    print("\nDependency Chain (from main.py):")
    print("-" * 50)
    for file_path in sorted(dependency_map.keys()):
        print(f"- {file_path}")
        for dep in sorted(dependency_map[file_path]):
            print(f"    └─ {dep}")

def main():
    """
    Main function to run the dependency analysis:
    1. Determine the project root and find main.py.
    2. Build the recursive dependency map starting from main.py.
    3. Print the chain of files in which main.py depends.
    """
    # Adjust this logic if your project layout differs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # E.g., main.py might be one level up
    
    main_path = find_python_file("main.py", project_root)
    if not main_path:
        print("Error: main.py not found in the project.")
        return
        
    dependency_map = get_recursive_dependencies(main_path, project_root)
    print_dependency_chain(dependency_map)

if __name__ == "__main__":
    main()
