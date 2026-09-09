import traceback
import os
import json
import re
import time
import pickle
import requests
from typing import Dict, List, Optional
import copy

# For semantic similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# For OpenAI
from openai import OpenAI
from validateStructure import extract_file_names
# For reading .env
from dotenv import load_dotenv
load_dotenv()


class Config:
    """
    Main configuration settings for chunk size, retries, similarity thresholds, etc.
    """
    MAX_TOKENS_PER_CHUNK = 40000
    MAX_RETRIES = 3
    CHECKPOINT_INTERVAL = 5

    # Merge subfolders whose similarity exceeds this threshold.
    SIMILARITY_THRESHOLD = 0.7

    # Shared client for hierarchy-organization calls.
    client = OpenAI()


# ----------------------
# File Chunker
# ----------------------
class FileChunker:
    @staticmethod
    def chunk_files(input_path: str) -> List[Dict]:
        """
        Splits the input JSON into smaller chunks to avoid exceeding the token limit.
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            all_files = json.load(f)

        chunks = []
        current_chunk = {}
        current_token_count = 0

        for filename, desc in all_files.items():
            # Estimate token usage (roughly 1.3 tokens per char + overhead)
            token_cost = int((len(filename) + len(desc)) * 1.3) + 10

            if current_token_count + token_cost > Config.MAX_TOKENS_PER_CHUNK:
                chunks.append(current_chunk)
                current_chunk = {}
                current_token_count = 0

            current_chunk[filename] = desc
            current_token_count += token_cost

        # Append the last chunk if it has any items
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


# ----------------------
# Semantic Validator
# ----------------------
class SemanticValidator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def max_similarity(self, texts: List[str]) -> float:
        """
        Returns the maximum similarity among all pairs in 'texts'.
        """
        if len(texts) < 2:
            return 0.0

        embeddings = self.model.encode(texts)
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        return max(similarities) if similarities else 0.0

    def most_similar_pair(self, texts: List[str]) -> (float, (str, str)):
        """
        Returns (max_similarity_value, (nameA, nameB)) for the pair with the highest similarity.
        """
        if len(texts) < 2:
            return 0.0, ("", "")

        embeddings = self.model.encode(texts)

        max_sim = 0.0
        pair = ("", "")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                if sim > max_sim:
                    max_sim = sim
                    pair = (texts[i], texts[j])
        return max_sim, pair


# ----------------------
# API Processor (with retries)
# ----------------------
class APIProcessor:
    def __init__(self):
        self.validator = SemanticValidator()

    def _repair_json(self, broken_json_str: str) -> str:
        """
        Calls the LLM to fix the broken JSON.
        Returns a string that (hopefully) is valid JSON.

        Prompt Strategy:
        - System: "You fix JSON, output ONLY valid JSON, no extra text."
        - User: Provide the broken JSON and ask for a corrected version.
        """
        system_prompt = (
            "You are a JSON repair assistant. Your task is to correct the user's JSON so it is valid. "
            "Output ONLY the fixed JSON, with no extra keys or commentary. All data from the original must be preserved. "
            "Make sure the final result is valid JSON."
        )

        user_prompt = f"The following JSON is invalid:\n\n{broken_json_str}\n\nPlease return a valid JSON version:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = Config.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )
            # Extract the repaired JSON text.
            fixed_json_str = completion.choices[0].message.content.strip()
            return fixed_json_str
        except requests.RequestException as e:
            raise ValueError("HTTP error during JSON repair call") from e


    def _send_fix_request(self, prev_structure: Dict, fix_prompt: str) -> Dict:

        system_prompt = (

            "You must ONLY return the pure JSON structure. No additional explanation.\n"
            "It must follow the same schema: { name, type, children }.\n"
            "All missing files must be included.\n"
        )


        prev_json_str = json.dumps(prev_structure, ensure_ascii=False)

        user_content = (
            f"Here is your previous JSON:\n{prev_json_str}\n\n"
            f"{fix_prompt}\n\n"
            "Remember to output only valid JSON, with all correct files included."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        completion = Config.client.chat.completions.create(
            model="o3-mini",
            messages=messages
        )
        response_content = completion.choices[0].message.content.strip()


        try:
            result = json.loads(response_content)
        except json.decoder.JSONDecodeError:
            fixed_json_str = self._repair_json(response_content)
            result = json.loads(fixed_json_str)
            # decoder = json.JSONDecoder()
            # result, idx = decoder.raw_decode(response_content)
            # if idx < len(response_content):
            #     print(f"Warning: Extra data found after JSON object at position {idx}.")
        return result

    def _extract_filenames_in_chunk(self, structure: Dict, chunk_keys: set) -> set:

        result = set()

        def dfs(node):
            if node.get("type") == "file":
                filename = node.get("name", "")

                if filename in chunk_keys:
                    result.add(filename)
            elif node.get("type") == "folder":
                for child in node.get("children", []):
                    dfs(child)

        dfs(structure)
        return result

    def process_chunk(self, chunk: Dict, base_structure: Optional[Dict] = None) -> Dict:
        all_files_in_chunk = set(chunk.keys())
        attempt = 0

        while attempt < Config.MAX_RETRIES:
            attempt += 1
            try:
                structure = self._send_api_request(chunk, base_structure)


                extracted_names_in_chunk = self._extract_filenames_in_chunk(structure, all_files_in_chunk)

                missing = all_files_in_chunk - extracted_names_in_chunk

                if not missing:

                    self._attach_descriptions(structure, chunk)
                    return structure
                else:
                    print(f"[Attempt {attempt}] Model missed {len(missing)} files. Retrying fix...")

                    fix_prompt = (
                            "Your previous JSON missed the following files:\n"
                            + "\n".join(list(missing))
                            + "\nPlease return a corrected JSON that includes all missing files, "
                              "without removing or altering the existing correct files."
                    )

                    structure = self._send_fix_request(structure, fix_prompt)

                    # Re-extract the file identifiers and verify coverage.
                    extracted_names_in_chunk = self._extract_filenames_in_chunk(structure, all_files_in_chunk)
                    still_missing = all_files_in_chunk - extracted_names_in_chunk

                    if not still_missing:
                        self._attach_descriptions(structure, chunk)
                        return structure
                    else:
                        print(f"Still missing {len(still_missing)} files after fix attempt {attempt}.")

            except (requests.Timeout, requests.ConnectionError) as e:
                print(f"Attempt {attempt} failed: {str(e)}")
                time.sleep(2 ** (attempt - 1))

        raise Exception("Max retries exceeded - still missing files from chunk.")

    def _strip_descriptions(self, node: Dict):
        """
        Recursively remove the 'desc' key from any file nodes,
        and from any folder's children, in-place.
        """
        if node.get("type") == "folder":
            for child in node["children"]:
                self._strip_descriptions(child)
        elif node.get("type") == "file":
            node.pop("desc", None)  # remove if present

    def _build_messages(self, chunk: Dict, base_structure: Optional[Dict]) -> List[Dict]:
        """
        Constructs the system/user prompts for the model.
        """
        system_prompt = (
            "You must and can only return a valid JSON structure strictly following these rules:\n"
            "1. ***Output ONLY the pure and exact JSON format with NO any additional natural language explanations or comments. NEVER OUTPUT ANY EXTRA TEXTS APART FROM JSON STRUCTURE\n***"
            "2. The structure must follow the schema:\n"
            "{\n"
            '  \"name\": \"Root\",\n'
            '  \"type\": \"folder\",\n'
            '  \"children\": [\n'
            '    {\"name\": \"FileName\", \"type\": \"file\"}\n'
            "    OR\n"
            '    {\"name\": \"FolderName\", \"type\": \"folder\", \"children\": [...]}\n'
            "  ]\n"
            "}\n"
            "3. All folder nodes must have meaningful semantic names representing their main topic;\n"
            "   file nodes use their original ID or filename.\n"
            "4. Ensure all strings use double quotes.\n"
            "5. You MUST output the ENTIRE structure without omitting any document (file node)."
            "6. ***You MUST output a valid Json format WITHOUT missing commas between key-value pairs, or missing or unmatched brackets/braces.***"
            "7. ***You MUST output a valid Json format where all JSON elements must be nested in the correct order.***"
            "8. ***YOU MUST CONTAIN ALL FILES TO BE INTEGRATED IN THE OUTPUT JSON.***"
            "9. After you finish constructing the JSON, re-check that every bracket and brace is properly closed and that no trailing commas exist. The final output must be valid JSON. Do not include any additional text, only the JSON."
        )

        example_structure = {
            "name": "ExampleClassification",
            "type": "folder",
            "children": [
                {"name": "LiteraryStudies.pdf", "type": "file"},
                {
                    "name": "PerformingArts",
                    "type": "folder",
                    "children": [
                        {"name": "TheatreResearch.md", "type": "file"}
                    ]
                }
            ]
        }

        user_content = "File list to be integrated:\n" + "\n".join(f"- {k}: {v}" for k, v in chunk.items())
        user_content += "\n\nPlease return the structure strictly in the following format:\n"
        user_content += json.dumps(example_structure, indent=2)
        user_content += "After you construct the JSON, re-check to ensure every single filename from the list is present in the final structure."

        if base_structure:
            stripped_structure = copy.deepcopy(base_structure)
            self._strip_descriptions(stripped_structure)
            user_content = (
                f"Current structure:\n{json.dumps(stripped_structure, ensure_ascii=False)}\n\n{user_content}"
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _send_api_request(self, chunk: Dict, base_structure: Optional[Dict]) -> Dict:
        messages = self._build_messages(chunk, base_structure)
        completion = Config.client.chat.completions.create(
            model="o3-mini",
            messages=messages
        )
        response_content = completion.choices[0].message.content.strip()

        try:
            result = json.loads(response_content)
        except json.decoder.JSONDecodeError:
            fixed_json_str = self._repair_json(response_content)
            result = json.loads(fixed_json_str)
        return result

    def _attach_descriptions(self, structure: Dict, chunk: Dict):
        """
        Traverse the final structure (folders/files) and attach 'desc' to each file node
        by matching the file name in 'chunk'.

        This step is crucial so we can generate a meaningful subfolder name if we exceed 3 files.
        """
        def traverse_and_attach(node: Dict):
            if node["type"] == "folder":
                for child in node["children"]:
                    traverse_and_attach(child)
            else:
                # node["type"] == "file"
                file_name = node["name"]
                # If the LLM hasn't renamed the file, this should match the chunk keys
                if file_name in chunk:
                    node["desc"] = chunk[file_name]
                else:
                    node["desc"] = ""

        traverse_and_attach(structure)


# ----------------------
# Structure Merger
# ----------------------
class StructureMerger:
    """
    Merges the newly returned JSON classification into the existing structure
    while enforcing:
      - Unlimited subfolders
      - Subfolders at the same level must not be too similar
      - Each folder can have at most 3 files (4th+ file is moved into a new subfolder with a semantic name).
      - No similarity checks among files themselves.
    """
    def __init__(self):
        self.validator = SemanticValidator()

    # def merge(self, base: Optional[Dict], new: Dict) -> Dict:
    #     if not base:
    #         return new
    #     self._merge_children(base['children'], new['children'])
    #     return base

    def merge(self, base: Optional[Dict], new: Dict) -> Dict:
        if not base:
            # Adopt the new tree as the initial structure and enforce constraints.
            if new["type"] == "folder":
                self._enforce_file_limit(new)
            return new
        # Merge into the existing structure.
        self._merge_children(base['children'], new['children'])
        return base

    def _merge_children(self, base_children: List, new_children: List):
        """
        Recursively merges children, ensuring each folder has:
         - unlimited subfolders
         - at most 3 files
         - subfolders are semantically distinct
        """
        # Index existing children by name
        existing_nodes = {node['name']: node for node in base_children}

        for new_node in new_children:
            if new_node['name'] in existing_nodes:
                # Merge if both are folders
                self._merge_existing_node(existing_nodes[new_node['name']], new_node)
            else:
                self._add_new_node(base_children, new_node)

        # After adding everything, check subfolder similarity
        self._validate_merged_children(base_children)

    def _merge_existing_node(self, existing: Dict, new: Dict) -> Dict:
        # all folders
        if existing['type'] == 'folder' and new['type'] == 'folder':
            if not existing.get("children") and new.get("children"):
                existing["children"] = new["children"]
            elif existing.get("children") and not new.get("children"):
                pass
            else:
                self._merge_children(existing['children'], new['children'])
            self._enforce_file_limit(existing)
            return existing
        # one file and other one is folder
        elif existing['type'] != new['type']:
            new_folder = {
                "name": existing['name'],
                "type": "folder",
                "children": []
            }
            if existing['type'] == 'file':
                new_folder["children"].append({
                    "name": existing["name"] + "_file",
                    "type": "file",
                    "desc": existing.get("desc", "")
                })
            else:
                new_folder["children"].extend(existing.get("children", []))
            if new['type'] == 'file':
                new_folder["children"].append({
                    "name": new["name"] + "_file",
                    "type": "file",
                    "desc": new.get("desc", "")
                })
            else:
                new_folder["children"].extend(new.get("children", []))
            self._enforce_file_limit(new_folder)
            print(f"Warning: Merged type conflict for node '{existing['name']}' into a folder.")
            return new_folder


    def _enforce_file_limit(self, folder_node: Dict):
        """
        Recursively ensures that each folder has at most 3 file nodes.
        If there are more than 3 files, the extra files are grouped into a single new subfolder.
        Also prunes any empty folders.
        """
        if folder_node["type"] != "folder":
            return

        # Get the current children (default to empty list if missing)
        children = folder_node.get("children", [])
        # Separate file nodes and folder nodes
        files = [c for c in children if c["type"] == "file"]
        subfolders = [c for c in children if c["type"] == "folder"]

        # If there are more than 3 file nodes, group the extra ones in a new subfolder.
        if len(files) > 3:
            remaining_files = files[:3]
            extra_files = files[3:]
            # Generate a name for the new subfolder using the first extra file.
            new_folder_name = self._generate_subfolder_name_for_file(extra_files[0])
            new_subfolder = {
                "name": new_folder_name,
                "type": "folder",
                "children": extra_files
            }
            subfolders.append(new_subfolder)
            files = remaining_files

        # Rebuild the children list with the (up to 3) files first, then subfolders.
        folder_node["children"] = files + subfolders

        # Recursively enforce file limit on all subfolders.
        for sf in subfolders:
            self._enforce_file_limit(sf)

        self._prune_empty_folders(folder_node["children"])

    def _prune_empty_folders(self, children: List[Dict]):
        pruned = []
        for node in children:
            if node["type"] == "folder":
                if "children" in node and node["children"]:
                    self._prune_empty_folders(node["children"])
                    if node["children"]:
                        pruned.append(node)
                else:
                    continue
            else:
                pruned.append(node)
        children[:] = pruned
    def _add_new_node(self, children: List[Dict], new_node: Dict):
        if new_node["type"] == "file":
            current_file_count = sum(1 for c in children if c["type"] == "file")
            if current_file_count >= 3:
                target_folder = None
                for c in children:
                    if c["type"] == "folder" and c["name"].startswith("Theme_"):
                        target_folder = c
                        break
                if target_folder:
                    target_folder["children"].append(new_node)
                else:
                    subfolder_name = self._generate_subfolder_name_for_file(new_node)
                    new_folder = {
                        "name": subfolder_name,
                        "type": "folder",
                        "children": [new_node]
                    }
                    children.append(new_folder)
            else:
                children.append(new_node)
        else:
            self._enforce_file_limit(new_node)
            children.append(new_node)

    def _validate_merged_children(self, children: List[Dict]):
        merged_dict = {}
        for node in children:
            name = node['name']
            if name in merged_dict:
                existing = merged_dict[name]
                if node['type'] == 'folder' and existing['type'] == 'folder':
                    if not existing.get("children") and node.get("children"):
                        existing["children"] = node["children"]
                    else:
                        existing_children = existing.get('children', [])
                        new_children = node.get('children', [])
                        existing['children'] = existing_children + new_children
            else:
                merged_dict[name] = node
        children[:] = list(merged_dict.values())

        folder_nodes = [c for c in children if c['type'] == 'folder']
        if len(folder_nodes) > 1:
            folder_names = [c['name'] for c in folder_nodes]
            max_sim, pair = self.validator.most_similar_pair(folder_names)
            if max_sim >= Config.SIMILARITY_THRESHOLD:
                merged_folder_name = self._make_merged_folder_name(pair)
                print(f"Warning: High similarity (max_sim={max_sim:.2f}) detected. Merging into folder '{merged_folder_name}'.")
                files = [c for c in children if c['type'] == 'file']
                subfolders = [c for c in children if c['type'] == 'folder']
                children.clear()
                children.append({
                    "name": merged_folder_name,
                    "type": "folder",
                    "children": subfolders
                })
                children.extend(files)


    # -----------------------------------------------------
    # Utility for generating a subfolder name from a file's desc
    # -----------------------------------------------------
    def _generate_subfolder_name_for_file(self, file_node: Dict) -> str:
        # """
        # Creates a subfolder name that reflects the file's content/description.
        # For example: "Theme_CloudComputing".
        # """
        desc_text = file_node.get("desc", "") or ""
        # short_phrase = self._extract_short_phrase(desc_text)
        # if not short_phrase:
        #     short_phrase = "Misc"
        # return f"Theme_{short_phrase}"

        system_prompt = (
            "You are a naming assistant. Your goal is to generate a short, semantic folder name "
            "based on the provided file description. Avoid filler words like 'document', 'details', "
            "'this', etc. Use underscores to separate keywords, and title-case each keyword. "
            "If you have no meaningful words, return 'Theme_Misc'. Limit to ~25 characters. "
            "Do not write explanations—just the final name."
        )

        user_prompt = f"File description: {desc_text}\n\nPlease provide only the folder name."

        response = Config.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        # Extract the generated folder name.
        folder_name = response.choices[0].message.content.strip()

        return folder_name
    def _make_merged_folder_name(self, pair: (str, str), max_len: int = 30) -> str:
        """
        Create a short, meaningful name for the auto-merged subfolder based on two subfolder names.
        """
        nameA =pair[0]
        nameB = pair[1]

        system_prompt = (
          """
          You are a naming assistant. Your task is to generate a concise, semantic folder name for a merged folder based on two provided folder names. 
          The output must be representative of both names, meaningful, and limited to around 25 characters. 
          Do not include any explanations—only provide the final folder name.
          """
        )

        user_prompt = f"File A name: {nameA} and File B name: {nameB} \n\n Please provide only the merged folder name."

        response = Config.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        # Extract the generated merged-folder name.
        merged_folder_name = response.choices[0].message.content.strip()

        return merged_folder_name


# ----------------------
# Checkpoint Manager
# ----------------------
class CheckpointManager:
    @staticmethod
    def save(structure: Dict, path: str = "checkpoint.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(structure, f)

    @staticmethod
    def load(path: str = "checkpoint.pkl") -> Optional[Dict]:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

def remove_duplicate_files(node: Dict, seen_filenames: set):
    if node["type"] != "folder":
        return

    cleaned_children = []
    for child in node["children"]:
        if child["type"] == "file":
            fname = child["name"]
            if fname in seen_filenames:
                continue
            else:
                seen_filenames.add(fname)
                cleaned_children.append(child)
        else:
            remove_duplicate_files(child, seen_filenames)
            cleaned_children.append(child)


    node["children"] = cleaned_children

def prune_empty_folders(node: dict) -> dict:
    """
    Recursively remove any folder nodes that have an empty children list.
    Returns None if this node is an empty folder and should be pruned from the tree,
    or the updated node otherwise.
    """
    # If it's not a folder, just return it as-is
    if node.get("type") != "folder":
        return node

    # Recursively prune all children first
    pruned_children = []
    for child in node.get("children", []):
        pruned_child = prune_empty_folders(child)
        # Only keep children that are not None
        if pruned_child is not None:
            pruned_children.append(pruned_child)

    # Update this folder's children
    node["children"] = pruned_children

    # If it's a folder but now has no children, return None so the parent knows to drop it
    if len(node["children"]) == 0:
        return None

    # Otherwise, return the updated folder node
    return node
# ----------------------
# Main Process
# ----------------------
def main(input_path: str, output_path: str):
    chunker = FileChunker()
    processor = APIProcessor()
    merger = StructureMerger()
    # checkpoint = CheckpointManager.load()
    #
    # if checkpoint:
    #     final_structure = checkpoint["structure"]
    #     processed_chunks = checkpoint.get("processed_chunks", 0)
    #     chunks = chunker.chunk_files(input_path)[processed_chunks:]
    # else:
    final_structure = None
    chunks = chunker.chunk_files(input_path)
    #
    try:
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1} of {len(chunks)}")

            chunk_structure = processor.process_chunk(chunk, final_structure)

            print(len(set(extract_file_names(chunk_structure))))

            final_structure = merger.merge(final_structure, chunk_structure)



            # Print an intermediate result
            print(f"Updated structure after chunk {idx + 1}:")
            print(json.dumps(final_structure, ensure_ascii=False, indent=2))

            if idx == len(chunks) - 1:
                CheckpointManager.save({
                    "structure": final_structure,
                    "processed_chunks": idx + 1
                })
        file_path = "organized_structure_duplicates.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            final_structure = json.load(file)
        # Save final structure
        seen = set()
        remove_duplicate_files(final_structure, seen)
        final_structure = prune_empty_folders(final_structure)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_structure, f, indent=2, ensure_ascii=False)

        print("\nProcessing complete! Organized structure saved to", output_path)

    except Exception as e:
        print(f"\nProcessing interrupted: {str(e)}",  traceback.print_exc())

        if final_structure:
            try:
                with open("partial_result.json", 'w', encoding='utf-8') as f:
                    json.dump(final_structure, f, indent=2, ensure_ascii=False)
                print("Partial result saved to partial_result.json")
            except Exception as save_error:
                print(f"Failed to save partial result: {str(save_error)}")
        CheckpointManager.save({"structure": final_structure})


# ----------------------
# Visualization Utility
# ----------------------
def visualize(structure: Dict, indent: int = 0):
    """Print the folder structure as a tree."""
    prefix = '│   ' * (indent - 1) + '├── ' if indent > 0 else ''
    print(f"{prefix}{structure['name']} ({structure['type']})")
    if structure['type'] == 'folder':
        for child in structure['children']:
            visualize(child, indent + 1)


if __name__ == "__main__":
    input_file = "doc_summary_dict.json"
    output_file = "organized_structure_processed.json"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        exit(1)

    main(input_file, output_file)

    # Optional: visualize the final structure
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                result_structure = json.load(f)
            print("\nFinal Structure Visualization:\n")
            visualize(result_structure)
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
    else:
        print("Warning: Final result file was not generated.")
