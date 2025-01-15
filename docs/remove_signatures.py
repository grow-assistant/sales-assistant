import os

def remove_signature(content):
    # Define the signature block to remove (with more flexible newline matching)
    signatures = [
        "\n\nCheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com",  # Double newline, markdown spaces
        "\nCheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com",     # Single newline, markdown spaces
        "Cheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com",       # No newline, markdown spaces
        # Original patterns without spaces
        "\n\nCheers,\nTy\n\nSwoop Golf\n480-225-9702\nswoopgolf.com",
        "\nCheers,\nTy\n\nSwoop Golf\n480-225-9702\nswoopgolf.com",
        "Cheers,\nTy\n\nSwoop Golf\n480-225-9702\nswoopgolf.com",
        # With trailing newlines
        "\n\nCheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com\n",
        "\nCheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com\n",
        "Cheers,\nTy\n\nSwoop Golf  \n480-225-9702  \nswoopgolf.com\n"
    ]
    
    # Print original content for debugging
    print("\nORIGINAL CONTENT:")
    print(repr(content[-200:]))  # Show last 200 chars with escape sequences visible
    
    # Try to remove each signature variant
    cleaned_content = content
    for sig in signatures:
        print("\nLooking for signature pattern:")
        print(repr(sig))
        if sig in cleaned_content:
            print("Found signature!")
            cleaned_content = cleaned_content.replace(sig, '')
        else:
            print("Signature pattern not found")
    
    # Remove any trailing blank lines and ensure no trailing whitespace
    cleaned_content = cleaned_content.rstrip()
    
    # Print cleaned content for debugging
    print("\nCLEANED CONTENT:")
    print(repr(cleaned_content[-200:]))  # Show last 200 chars
    
    return cleaned_content  # No extra newline added

def process_templates():
    # Path to templates directory
    templates_dir = 'templates'  # Changed to relative path
    
    print(f"Looking for files in: {os.path.abspath(templates_dir)}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(templates_dir):
        print(f"\nChecking directory: {root}")
        print(f"Found files: {files}")
        
        for filename in files:
            if filename.endswith(('.txt', '.md')):  # Process both .txt and .md files
                file_path = os.path.join(root, filename)
                print(f"\nProcessing file: {file_path}")
                
                try:
                    # First try UTF-8
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        print("Successfully read with UTF-8")
                except UnicodeDecodeError:
                    try:
                        # If UTF-8 fails, try UTF-16
                        with open(file_path, 'r', encoding='utf-16') as file:
                            content = file.read()
                            print("Successfully read with UTF-16")
                    except UnicodeDecodeError:
                        # If both fail, use UTF-8 with error handling
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                            print("Successfully read with UTF-8 (ignore errors)")
                
                # Remove signature
                updated_content = remove_signature(content)
                
                # Write back using UTF-8
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(updated_content)
                
                print(f"Finished processing: {file_path}")

if __name__ == "__main__":
    print("Starting signature removal process...")
    process_templates()
    print("\nFinished processing all files!")