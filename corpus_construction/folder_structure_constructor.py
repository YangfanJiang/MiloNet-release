import os
import shutil
import contextlib
import json

@contextlib.contextmanager
def change_directory(new_dir):
    """
    A context manager that changes the current working directory to new_dir.
    It saves the current directory using a file descriptor and then uses os.fchdir to return to it.
    """
    old_fd = os.open('.', os.O_RDONLY)
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.fchdir(old_fd)
        os.close(old_fd)

def load_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def find_original_relative_path(file_id, original_map, default_ext=".md"):
    """
    Given a file ID (e.g. "0_1_1_4"), append the default extension to form "0_1_1_4.md".
    Then iterate over the original_map (which maps original relative paths to processed full paths)
    and return the normalized original relative path (the key) whose processed full path's basename
    matches the target.
    """
    target_name = file_id + default_ext
    for orig_rel_path, processed_full_path in original_map.items():
        # Process the processed_full_path to get its basename properly.
        if os.path.basename(processed_full_path.split("\\")[-1]) == target_name:
            # Normalize and convert any mixed separators.
            return os.path.normpath(orig_rel_path.replace('\\', os.sep).replace('/', os.sep))
    return None

def create_new_structure_cd(node, original_map, original_documents_folder, default_ext=".md"):
    """
    Recursively creates the new folder structure by changing the current working directory ("cd")
    into each created folder. For folder nodes, it creates the folder in the current working directory,
    then processes its children relative to that folder. For file nodes, it looks up the original relative path
    via the file ID, constructs the absolute source path (by joining original_documents_folder),
    and copies the file into the current working directory (keeping its original file name).
    """
    node_type = node.get("type")
    node_name = node.get("name")

    if node_type == "folder":
        # Create the folder in the current working directory.
        os.makedirs(node_name, exist_ok=True)
        print("Creating folder:", os.path.abspath(node_name))
        # Change directory into the new folder.
        with change_directory(node_name):
            for child in node.get("children", []):
                create_new_structure_cd(child, original_map, original_documents_folder, default_ext)

    elif node_type == "file":
        file_id = node_name
        orig_rel_path = find_original_relative_path(file_id, original_map, default_ext)
        if not orig_rel_path:
            print(f"WARNING: Could not find original file for file ID '{file_id}'.")
            return
        # Build the absolute source file path.
        src_file_path = os.path.abspath(os.path.join(original_documents_folder, os.path.normpath(orig_rel_path)))
        # Destination file name remains the original file's base name.
        dest_file_name = os.path.basename(orig_rel_path)

        # Report the resolved copy operation.
        print("Current working directory:", os.getcwd())
        print(f"Copying file:\n  FROM: {src_file_path}\n  TO:   {os.path.abspath(dest_file_name)}")
        try:
            shutil.copy2(src_file_path, dest_file_name)
        except Exception as e:
            print(e)
        # Verify the file was copied
        if os.path.exists(dest_file_name):
            print(f"File {dest_file_name} copied successfully.")
        else:
            print(f"ERROR: File {dest_file_name} not found after copy.")
def build_folder_tree_cd(new_structure, original_map, output_root, original_documents_folder, default_ext=".md"):
    """
    Creates the output root folder and then changes into that folder to start building the new folder structure.
    Both output_root and original_documents_folder are converted to absolute and normalized paths.
    """
    # Normalize and convert to absolute paths.
    output_root_abs = os.path.abspath(os.path.normpath(output_root))
    original_documents_folder_abs = os.path.abspath(os.path.normpath(original_documents_folder))
    os.makedirs(output_root_abs, exist_ok=True)
    with change_directory(output_root_abs):
        create_new_structure_cd(new_structure, original_map, original_documents_folder_abs, default_ext)

# Command-line entry point.
if __name__ == "__main__":
    original_map = load_json_from_file("processed_files_map.json")
    new_structure = load_json_from_file("organized_structure_no_extras.json")
    # These are provided as relative paths but will be converted:
    original_documents_folder = r"Original_documents"
    output_root = r"New_Folder_Structure"

    build_folder_tree_cd(new_structure, original_map, output_root, original_documents_folder, default_ext=".md")
    print("New folder structure built successfully.")
