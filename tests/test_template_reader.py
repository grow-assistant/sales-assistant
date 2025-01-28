# Filename: test_template_reader.py

import sys
import logging
from pathlib import Path

# Adjust these if your project structure differs
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))  # Ensure we can import from the project root

# Import your DocReader convenience function
from utils.doc_reader import read_doc

def main():
    doc_name = "templates/general_manager_initial_outreach.md"  # no leading 'docs/'!

    # Attempt to read the template
    content = read_doc(doc_name)
    if content.strip():
        print(f"✓ Successfully read template: {doc_name}\n")
        print(content)
    else:
        print(f"✗ Failed to read or empty content for: {doc_name}")

if __name__ == "__main__":
    # Optionally set up more verbose logging:
    logging.basicConfig(level=logging.DEBUG)
    main()
