import json


def load_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Collect file identifiers recursively.
def extract_file_names(folder):
    file_names = []
    if "children" in folder:
        for child in folder["children"]:
            if child["type"] == "file":
                file_names.append(child["name"])
            elif child["type"] == "folder":
                file_names.extend(extract_file_names(child))
    return file_names

def remove_diff_files(node: dict, valid_filenames: set) -> dict:
    """
    Recursively remove file nodes whose name is not in valid_filenames.
    If a folder ends up empty, remove that folder as well (return None).
    """
    # If it's a file node, check if it's in valid_filenames
    if node.get("type") == "file":
        if node["name"] not in valid_filenames:
            return None  # Remove this "extra" file
        return node  # Keep it

    # If it's a folder, iterate its children
    if node.get("type") == "folder":
        cleaned_children = []
        for child in node.get("children", []):
            result = remove_diff_files(child, valid_filenames)
            if result is not None:
                cleaned_children.append(result)
        node["children"] = cleaned_children

        # If this folder is now empty, remove it
        if not node["children"]:
            return None
        return node

    # Preserve untyped nodes for compatibility with the staged workflow.
    return node

# Load the candidate hierarchy and valid document identifiers.
file_path = "organized_structure_no_extras.json"
ori_file_path = "doc_summary_dict.json"
json_data = load_json_from_file(file_path)
ori_json_data = load_json_from_file(ori_file_path)

valid_keys = set(ori_json_data.keys())
# cleaned_structure = remove_diff_files(json_data, valid_keys)
# Compare identifiers in the hierarchy with the valid identifier set.
file_names = extract_file_names(json_data)
diff_1= valid_keys - set(file_names)
diff_2= set(file_names)- valid_keys
print(diff_1, diff_2)

# with open("organized_structure_no_extras.json", "w", encoding="utf-8") as out:
#     json.dump(cleaned_structure, out, indent=2, ensure_ascii=False)


# # Report duplicate file identifiers.
# file_counts = Counter(file_names)
# duplicates = {name: count for name, count in file_counts.items() if count > 1}
#
# for name, count in duplicates.items():
#     print(f"{name}: {count}")
