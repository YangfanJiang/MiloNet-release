import os
import json

# Root directory (change this as needed)
ROOT_DIR = os.path.abspath("Original_documents")

# Stores mapping from old paths to new paths
name_mapping = {}
counter = 1  # Unique number counter

def get_unique_name():
    """ Returns a unique number as a string. """
    global counter
    unique_name = str(counter)
    counter += 1
    return unique_name

def rename_directories():
    """ Renames directories from the deepest level first. """
    directories = sorted(
        [os.path.join(dp, d) for dp, dn, _ in os.walk(ROOT_DIR) for d in dn],
        key=lambda x: -x.count(os.sep)  # Sort so we process the deepest directories first
    )

    for old_path in directories:
        parent_dir = os.path.dirname(old_path)
        new_name = get_unique_name()
        new_path = os.path.join(parent_dir, new_name)

        if os.path.exists(old_path):
            print(f"Renaming directory: {old_path} -> {new_path}")
            os.rename(old_path, new_path)
            name_mapping[old_path] = new_path  # Store mapping

def rename_files():
    """ Renames all files to unique numbers. """
    files = [
        os.path.join(dp, f) for dp, _, fn in os.walk(ROOT_DIR) for f in fn
    ]

    for old_path in files:
        parent_dir = os.path.dirname(old_path)
        ext = os.path.splitext(old_path)[1]  # Keep file extension
        new_name = get_unique_name() + ext  # Unique number with extension
        new_path = os.path.join(parent_dir, new_name)

        if os.path.exists(old_path):
            print(f"Renaming file: {old_path} -> {new_path}")
            os.rename(old_path, new_path)
            name_mapping[old_path] = new_path  # Store mapping

if __name__ == "__main__":
    rename_directories()  # Step 1: Rename directories first
    rename_files()        # Step 2: Rename files after directories

    # Save the mapping to a JSON file
    with open("name_mapping.json", "w", encoding="utf-8") as f:
        json.dump(name_mapping, f, indent=4)

    print("All folder and file names have been converted to numbers.")
    print("Mapping saved in name_mapping.json.")