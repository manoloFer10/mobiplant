import os
import sys

def process_directory(directory_path, prefix):
    """
    Process all PDF files in the specified directory.
    Each file is expected to have the naming convention:
      "{year} {doi_suffix}.pdf"
    The function renames the file to insert the given prefix before the DOI suffix.
    """
    # Replace any forward slashes in the prefix with hyphens
    safe_prefix = prefix.replace('/', '_')
    print(f"Using safe prefix: '{safe_prefix}'")

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            name_without_ext, ext = os.path.splitext(filename)
            # Expected filename format: "<year> <doi_suffix>"
            parts = name_without_ext.split(" ", 1)
            if len(parts) != 2:
                print(f"Skipping file with unexpected format: {filename}")
                continue
            year, doi_suffix = parts
            new_filename = f"{year} {safe_prefix}{doi_suffix}{ext}"
            new_file_path = os.path.join(directory_path, new_filename)
            
            if os.path.exists(new_file_path):
                print(f"Target file already exists, skipping: {new_filename}")
                continue

            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python rename_pdfs.py <directory> <prefix>")
        print("Example: python rename_pdfs.py \"data\\paper_collection\\(falta prefijo doi) JexB\" \"10.1093/jxb/\"")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    prefix = sys.argv[2]
    
    print(f"Processing directory: {directory_path} with prefix: {prefix}")
    process_directory(directory_path, prefix)

if __name__ == "__main__":
    main()