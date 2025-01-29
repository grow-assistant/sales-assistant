"""
Purpose:
1. Recursively finds every Python file in the project (including subdirectories).
2. Recursively analyzes imports starting from main.py (handling dotted imports like "services.hubspot_service").
3. Prints all files that main.py depends on, and identifies which files are not used by main.py.

Key functions:
- find_python_file(filename, start_path): Searches for a Python file by name under the start_path.
- find_all_python_files(start_path): Gathers all *.py files in the project.
- extract_imports(file_path): Parses a Python file's AST to extract imports (preserving dotted paths).
- get_recursive_dependencies(file_path, start_path): Recursively builds a dependency tree from a given file.
- print_analysis(dependency_map, all_files): Prints which files are used vs. which are unused.

Usage:
1. Adjust project_root in main() if needed.
2. Run this script directly. It prints the dependency chain from main.py,
   then lists all .py files not used by main.py.
"""

import os
import ast
from typing import Set, Dict, List
import sys

def find_python_file(filename: str, start_path: str = '.') -> str:
    """
    Search for a specific Python file within the given directory and its subdirectories.
    Returns the first matching path or None if not found.
    """
    for root, _, files in os.walk(start_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_all_python_files(start_path: str = '.') -> Set[str]:
    """
    Recursively locate every .py file in the given directory tree.
    Returns a set of paths, each relative to start_path.
    """
    python_files = set()
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start_path)
                python_files.add(rel_path)
    return python_files

def extract_imports(file_path: str) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())
            
        for node in ast.walk(tree):
            # Handle regular imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            
            # Handle from ... import ...
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                for name in node.names:
                    if name.name != '*':
                        imports.add(name.name.split('.')[0])
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return imports

def get_recursive_dependencies(file_path: str, start_path: str = '.', processed_files: Set[str] = None) -> Dict[str, Set[str]]:
    """
    Recursively build a dependency map starting from a given file.
    Returns {relative_file_path: {dotted_import_strings}}.
    """
    if processed_files is None:
        processed_files = set()
    
    dependency_map = {}
    
    # Avoid re-processing the same file
    if file_path in processed_files:
        return dependency_map
    
    processed_files.add(file_path)
    
    # Get direct imports
    dependencies = extract_imports(file_path)
    relative_path = os.path.relpath(file_path, start_path)
    dependency_map[relative_path] = dependencies
    
    # Convert dotted imports to local file paths and recurse
    for dep in dependencies:
        dep_filename = dep.replace('.', os.sep) + ".py"
        dep_path = find_python_file(dep_filename, start_path)
        
        if dep_path and dep_path not in processed_files:
            nested_deps = get_recursive_dependencies(dep_path, start_path, processed_files)
            dependency_map.update(nested_deps)
    
    return dependency_map

def print_analysis(dependency_map: Dict[str, Set[str]], all_files: Set[str], project_root: str) -> None:
    """Print the complete dependency analysis, showing what files main.py depends on."""
    # Find main.py's entry in the dependency map
    main_file = next((f for f in dependency_map.keys() if f.endswith('main.py')), None)
    if not main_file:
        print("Error: main.py not found in dependency map")
        return

    # Get direct imports from main.py
    try:
        with open(os.path.join(project_root, main_file), 'r', encoding='utf-8') as file:
            content = file.read()
            tree = ast.parse(content)
            
            direct_imports = {
                'standard_lib': set(),
                'third_party': set(),
                'project': set()
            }
            
            project_prefixes = {'scripts.', 'utils.', 'services.', 'models.', 'config.', 'scheduling.'}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_name = name.name.split('.')[0]
                        if any(name.name.startswith(prefix) for prefix in project_prefixes):
                            direct_imports['project'].add(name.name)
                        elif module_name in sys.stdlib_module_names:
                            direct_imports['standard_lib'].add(name.name)
                        else:
                            direct_imports['third_party'].add(name.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if any(node.module.startswith(prefix) for prefix in project_prefixes):
                            direct_imports['project'].add(node.module)
                        elif module_name in sys.stdlib_module_names:
                            direct_imports['standard_lib'].add(node.module)
                        else:
                            direct_imports['third_party'].add(node.module)
    
    except Exception as e:
        print(f"Error analyzing main.py imports: {e}")
        return

    print("\nDirect Imports in main.py:")
    print("-" * 50)
    
    print("\nStandard Library Imports:")
    for imp in sorted(direct_imports['standard_lib']):
        print(f" - {imp}")
    
    print("\nThird-Party Library Imports:")
    for imp in sorted(direct_imports['third_party']):
        print(f" - {imp}")
    
    print("\nProject-Specific Imports:")
    for imp in sorted(direct_imports['project']):
        print(f" - {imp}")

    # Rest of the dependency analysis...
    # Files used are the keys in the dependency_map
    used_files = set(dependency_map.keys())
    unused_files = all_files - used_files

    print("\nUnused Python Files:")
    print("-" * 50)
    for f in sorted(unused_files):
        print(f"- {f}")
    
    print(f"\nSummary:\n  Total Python files: {len(all_files)}")
    print(f"  Used (dependency chain): {len(used_files)}")
    print(f"  Unused: {len(unused_files)}")

def analyze_files(file_paths: List[str]) -> Set[str]:
    """Analyze multiple Python files and return unique dependencies."""
    all_imports = set()
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            file_imports = extract_imports(file_path)
            all_imports.update(file_imports)
        else:
            print(f"File not found: {file_path}")
    
    # Use sys.stdlib_module_names instead of stdlib_list
    std_libs = set(sys.stdlib_module_names)
    external_imports = {imp for imp in all_imports if imp not in std_libs}
    
    return external_imports

def main():
    """Main function to analyze file usage."""
    # Get all Python files in project
    all_files = find_all_python_files('.')
    
    # Look for imports of ping_hubspot_for_gm
    target_file = "ping_hubspot_for_gm"
    files_using_target = []
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if target_file in content:
                    files_using_target.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"\nFiles importing or using {target_file}:")
    print("-" * 50)
    if files_using_target:
        for file in files_using_target:
            print(f"- {file}")
    else:
        print("No files found using this module")

if __name__ == "__main__":
    main()
