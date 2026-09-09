from mlx_vlm import load
from mlx_vlm.utils import load_config
import traceback
import os
from dotenv import load_dotenv
from threading import Lock
import nltk
from fontTools.ttLib.tables.ttProgram import tt_instructions_error
from collections import defaultdict
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import core_prompts as core_prompts

# import shutil
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ChatPromptTemplate,
)

from phoenix.otel import register
import hashlib
import pdb
import math
# from helper import check_memory, LimitedDict, embed_model
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.llms.openai import OpenAI

if os.environ.get("MILONET_COST_LOG_PATH"):
    from cost_logging import install_llama_index_cost_logging_from_env

    install_llama_index_cost_logging_from_env()
from llama_index.core.callbacks import CallbackManager
from pathlib import Path
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.agent.openai import OpenAIAgent
import json
import pickle
import helper
from typing import Sequence, List, Tuple, Optional, Dict, Any
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from datetime import datetime
import gradio as gr
from pathlib import Path
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.packs.agents_coa import CoAAgentPack
from llama_index.agent.coa import CoAAgentWorker
from safe_coa_parser import SafeChainOfAbstractionParser
from llama_index.llms.openai import OpenAI
from llama_index.packs.subdoc_summary import SubDocSummaryPack
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import sys, threading, time, io, re
import pandas as pd
from pptx import Presentation
from llama_index.agent.llm_compiler.step import LLMCompilerAgentWorker
import llama_index.core.callbacks as callbacks
from crawl4ai import AsyncWebCrawler
import asyncio
from googlesearch import search
import time
from pathlib import Path
from gradio.themes import Base, Size
from llama_parse import LlamaParse
from llama_index.core.storage.chat_store import SimpleChatStore
# import joblib
from pycallgraph2 import PyCallGraph, Config
from pycallgraph2.output import GraphvizOutput
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from helper import read_tool_access_info
from UI import GradioInterface
from helper import ModeLoader, QueryTracker, read_all_doc_tool_info, id_stamped_print, parse_ids_from_query
# from unidecode import unidecode
# from llama_index.readers.pdf_marker.base import PDFMarkerReader
import configparser
from local_pdf_parser_mac import convert_to_markdown_and_save

# from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
import torch
from transformers import AutoModelForCausalLM

# from UI import reset_raw_responses
from ollama import chat
from ollama import ChatResponse
from llama_index.core import ServiceContext

from llama_index.core.chat_engine.types import AgentChatResponse

# from llama_index.core.callbacks import CallbackManager
# from helper import PerDocumentContextCallback, register_handler

from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
# from llama_index.embeddings.ollama import OllamaEmbedding
import uuid
from helper import TraceNodeTap


# --- Helper to separate main query content from "Additional focus" directives ---
def _split_query_and_focus(text: str) -> Tuple[str, List[str]]:
    if not text:
        return "", []

    focus_items: List[str] = []
    remaining_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"(?i)^additional\s+focus\s*:\s*(.+)$", line)
        if match:
            value = match.group(1).strip()
            if value:
                focus_items.append(value)
        else:
            remaining_lines.append(line)

    main_query = " ".join(remaining_lines).strip()
    return main_query, focus_items


# --- Simple query sanitizer for rare polluted plan lines ---
def _sanitize_query_text(q: str) -> str:
    """Best-effort cleanup for occasional polluted tool inputs.

    - Removes common injected boilerplate like "* The document does not contain..." or "In summary, ..."
    - Flattens newlines/extra quotes/backticks
    - Trims to a reasonable length, preferring to cut after a question mark
    """
    if not isinstance(q, str):
        try:
            q = str(q)
        except Exception:
            return ""

    s = q.replace("\n", " ").replace("\r", " ")
    additional_focus = ""
    match = re.search(r"(Additional focus\s*:\s*.*)", s, flags=re.IGNORECASE)
    if match:
        additional_focus = match.group(1).strip()
        s = s[: match.start()]
    # Drop typical injected segments (case-insensitive)
    s = re.split(r"\*\s*The document does not contain", s, flags=re.IGNORECASE)[0]
    s = re.split(r"In summary,", s, flags=re.IGNORECASE)[0]
    # Remove stray leading bullets/asterisks/backticks/quotes
    s = s.lstrip("* `\"").strip()
    # Prefer keeping up to last question mark if present
    if "?" in s:
        s = s[: s.rfind("?") + 1]
    # Final trim and length cap
    s = s.strip('`" ').strip()
    if len(s) > 256:
        s = s[:256].rstrip()
    if additional_focus:
        focus_clean = additional_focus.strip('`" ').strip()
        if focus_clean:
            return f"{s}\n{focus_clean}" if s else focus_clean
    return s

# apply enhanced OpenAI model support patch
try:
    from enhanced_model_patch import comprehensive_patch
    comprehensive_patch()
    print("🚀 using enhanced patch system")
except ImportError:
    # backup: use original patch
    try:
        from openai_model_patch import patch_openai_model_support
        patch_openai_model_support()
        print("🔄 using basic patch system")
    except ImportError:
        print("⚠️ cannot find model patch file")
except Exception as e:
    print(f"⚠️ model patch application failed: {e}")

config = configparser.ConfigParser()
config.read('parameters.ini')
CHUNK_SIZE = int(config['INDEXING']['ChunkSize'])
CHUNK_OVERLAP = int(config['INDEXING']['ChunkOverlap'])
SIMILARITY_CUTOFF = float(config['RETRIEVER']['SimilarityCutOff'])
RETRIEVER_TOP_K = int(config['RETRIEVER']['TopK'])
AGENT_CHUNK_SIZE = int(config['AGENT_CHUNK_SIZE']['AgentMaxNum'])
load_dotenv()
base_dir = Path(__file__).parent

allowed_extensions = ('.docx', '.pdf', '.txt', '.xlsx', '.md', '.pptx')
original_documents_folder = (base_dir / 'Original_documents').resolve()
processed_documents_folder = (base_dir / 'Processed_documents').resolve()
global_stop = False
global_id_map = {}
# global_id_map_lock = threading.RLock()

tool_long_tool_description_file = "tool_long_tool_description.txt"
lock_tool_long_tool_description_file = threading.RLock()

filename_map = {}
if os.path.exists("hashing_name_mapping.json"):
    with open("hashing_name_mapping.json", "r", encoding="utf-8") as f:
        filename_map = json.load(f)

def generate_folder_hash(path_id, context=None):
    """generate a consistent hash value for the path ID, ensuring that all files in the same parent path share the same hash prefix"""
    
    if re.match(r'^[a-f0-9]{8}$', path_id):
        # if the path_id is already a valid hash, return it
        return path_id
    
    folder_id = path_id

    if context == "document":
        parts = path_id.split('_')
        if len(parts) > 1:
            folder_id = '_'.join(parts[:-1])
    
    # if len(folder_id) <= 35:
    #     # if the path_id is less than 35 characters, it is already a valid hash
    #     return folder_id
    
    with helper._file_read_lock:
        if folder_id in global_id_map:
            return global_id_map[folder_id]
    
    # generate a hash for the entire path
    folder_hash = hashlib.md5(folder_id.encode('utf-8')).hexdigest()[:8]
    with helper._file_read_lock:
        global_id_map[folder_id] = folder_hash
    return folder_hash



def get_original_id(shortened_id):
    global global_id_map
    
    # create a lock for the file read
    with helper._file_read_lock:
        reverse_map = {v: k for k, v in global_id_map.items()}
        return reverse_map.get(shortened_id)

def get_shortened_id(original_id):
    global global_id_map
    
    with helper._file_read_lock:
        if original_id in global_id_map:
            return global_id_map[original_id]

def load_processed_files_map(processed_files_map_path):
    if os.path.exists(processed_files_map_path):
        with open(processed_files_map_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_files_map(processed_files_map_path, processed_files_map):
    with open(processed_files_map_path, 'w', encoding="utf-8") as f:
        json.dump(processed_files_map, f)




def sanitize_name(name):
    base, ext = os.path.splitext(name)
    sanitized_base = re.sub(r'[^a-zA-Z0-9]', '_', base).replace(' ', '_')
    return sanitized_base + ext


def sanitize_paths_in_directory(directory):
    print(f"Sanitizing paths in directory: {directory}")
    for root, dirs, files in os.walk(directory, topdown=False):
        filtered_files = [f for f in files if "ds_store" not in f.lower()]
        for file in filtered_files:
            original_file_path = os.path.join(root, file)
            sanitized_file_name = sanitize_name(file)
            sanitized_file_path = os.path.join(root, sanitized_file_name)
            if original_file_path != sanitized_file_path:
                os.rename(original_file_path, sanitized_file_path)
                print(f"Renamed file from {original_file_path} to {sanitized_file_path}")

        filtered_dirs = [d for d in dirs if "ds_store" not in d.lower()]
        for dir in filtered_dirs:
            original_dir_path = os.path.join(root, dir)
            sanitized_dir_name = sanitize_name(dir)
            sanitized_dir_path = os.path.join(root, sanitized_dir_name)
            if original_dir_path != sanitized_dir_path:
                os.rename(original_dir_path, sanitized_dir_path)
                print(f"Renamed directory from {original_dir_path} to {sanitized_dir_path}")


def mirror_folder_structure_with_indexing(original_folder, processed_folder, index_map, processed_files_map,
                                          processed_files_map_path):
    # model_path = "deepseek-ai/deepseek-vl-7b-chat"
    # vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    # vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
    model, processor = load(model_path)
    config = load_config(model_path)

    for root, dirs, files in os.walk(original_folder):
        files = [f for f in files if "ds_store" not in f.lower()]
        dirs[:] = [d for d in dirs if "ds_store" not in d.lower()]

        relative_path = os.path.relpath(root, original_folder)
        indexed_relative_path = index_map.get(relative_path, None)
        if indexed_relative_path is None:
            print(f"Warning: No indexed path found for {relative_path}")
            indexed_relative_path = relative_path

        sanitized_indexed_relative_path = sanitize_name(str(indexed_relative_path))
        processed_path = os.path.join(processed_folder, sanitized_indexed_relative_path)

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)  # create processed folder here
            print(f"Created directory: {processed_path}")

        # for subdir in ['document_store', 'nodes_storage', 'subdoc_db', 'summary_db', 'vector_db']:
        for subdir in ['nodes_storage', 'subdoc_db', 'summary_db', 'vector_db']:
            subdir_path = os.path.join(processed_path, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Created directory: {subdir_path}")


        document_store_path = os.path.join(processed_path, 'document_store')
        if not os.path.exists(document_store_path):
            os.makedirs(document_store_path)
            print(f"Created directory: {document_store_path}")

        for file in files:
            if "ds_store" in file.lower():
                continue
            # filename_without_extension = os.path.splitext(file)[0]
            src_file = os.path.join(root, file)
            relative_file_path = os.path.relpath(src_file, original_folder)

            if relative_file_path in processed_files_map:
                print(f"File already processed: {relative_file_path}")
                continue

            indexed_file_name = index_map.get(relative_file_path, None)
            if indexed_file_name is None:
                print(f"Warning: No indexed name found for {relative_file_path}")
                indexed_file_name = file

            # file_extension = os.path.splitext(file)[1]
            sanitized_indexed_file_name = sanitize_name(str(indexed_file_name))

            # document_store_path = os.path.join(processed_path, 'document_store')
            # if not os.path.exists(document_store_path):
            #     os.makedirs(document_store_path)
            #     # print(f"Created directory: {document_store_path}")

            markdown_filepath = convert_to_markdown_and_save(src_file, document_store_path, sanitized_indexed_file_name,
                                                             model, processor, config)
            processed_files_map[relative_file_path] = markdown_filepath

    save_processed_files_map(processed_files_map_path, processed_files_map)


def print_directory_contents(folder_path):
    folder_path = str(folder_path)  # Ensure folder_path is a string
    for root, dirs, files in os.walk(folder_path):
        files = [f for f in files if "ds_store" not in f.lower()]
        dirs[:] = [d for d in dirs if "ds_store" not in d.lower()]
        root = str(root)  # Ensure root is a string
        level = root.replace(folder_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(sub_indent, f))


def create_indexed_path_map(file_paths_with_positions):
    index_map = {}
    for index, path in file_paths_with_positions:
        index_map[path] = index
    return index_map


def create_tool_access_file(file_paths_with_positions, output_file="tool_access_info.txt",
                            all_doc_tools_file="all_doc_tool.txt",
                            original_documents_root=None):
    global global_id_map
    source_root = original_documents_root or original_documents_folder
    # Sort the list to ensure folders are listed before their files
    sorted_file_paths_with_positions = sorted(file_paths_with_positions, key=lambda x: x[0])

    # Dictionary to map tools to their files
    tool_access_map = {}

    # Iterate over the sorted list and organize tools and their accessible files
    for position, path in sorted_file_paths_with_positions:
        if os.path.isdir(os.path.join(source_root, path)):
            # Initialize a new tool entry
            tool_access_map[position] = []
        else:
            # Extract the folder (tool) position and add the file to that tool's list
            parent_position = "_".join(position.split("_")[:-1])
            if parent_position in tool_access_map:
                shortened_code = get_shortened_id(position)
                if not shortened_code and len(position) > 35:
                    print(f"Warning: Long ID '{position}' not hashed to a shortened code.")
                tool_access_map[parent_position].append((os.path.basename(path), shortened_code))

    with open(output_file, "w", encoding='utf-8') as f:
        # Write the header information
        f.write(
            'Below is information on the files (documents) available to each TOOL. '
            'Each file has an ORIGINAL NAME and also a UNIQUE CODE which is used by the TOOLS. '
            'Each TOOL may have access to multiple files and each file is listed as '
            '[original filename="example.md", UNIQUE CODE=shortened_id_or_original_id]\n\n'
        )

        # Write the tool access information
        for tool, files in tool_access_map.items():
            if files:
                files_list = []
                for filename, code in files:
                    original_code = code 
                    files_list.append(f'[original filename="{os.path.splitext(filename)[0]}.md", UNIQUE CODE="{original_code}"]')
                
                files_text = ", ".join(files_list)

                shortened_tool_id = get_shortened_id(tool)
                

                if shortened_tool_id:
                    if shortened_tool_id.startswith("tool_"):
                        tool_display_id = shortened_tool_id
                    else:
                        tool_display_id = f"tool_{shortened_tool_id}"
                else:
                    tool_display_id = f"tool_{tool}"
                f.write(f'{tool_display_id} has access to: {files_text}\n')

            else:
                shortened_tool_id = get_shortened_id(tool)
                if shortened_tool_id:
                    tool_display_id = f"tool_{shortened_tool_id}"
                else:
                    tool_display_id = f"tool_{tool}"
                    
                f.write(f'{tool_display_id} has access to: []\n')

    with open(all_doc_tools_file, "w", encoding='utf-8') as f:
        # Write the header information
        f.write(
            'Below is information on the files (documents) available to the top doc TOOL. '
            'Each file has an ORIGINAL NAME and also a UNIQUE CODE which is used by the TOOL. '
            'Each file is listed as '
            '[original filename="example.md", UNIQUE CODE=shortened_id_or_original_id]\n\n'
        )

        # Write the tool access information
        total_files = []
        for _, files in tool_access_map.items():
            total_files.extend(files)
        if total_files:
            files_list = ", ".join(
                [f'[original filename="{os.path.splitext(filename)[0]}.md", UNIQUE CODE="{code}"]' for
                 filename, code in total_files])
            f.write(f'{files_list}\n')
        else:
            f.write(f'[]\n')



log_window_update_interval = 1  # Update the log window every 1 second





def list_files_recursive_with_numbering(folder_name, max_depth=None):
    file_paths_with_positions = []
    folder_paths_with_positions = []
    folder_index_map = {}
    # ignore_dirs = {'document_store'}

    def traverse_folder(current_folder, current_position, current_depth):
        folder_index_map[current_folder] = current_position
        relative_folder_path = os.path.relpath(current_folder, start=absolute_folder_path)
        folder_paths_with_positions.append((current_position, relative_folder_path))

        if max_depth is not None and current_depth >= max_depth:
            return
        for root, dirs, files in os.walk(current_folder):
            files = [f for f in files if "ds_store" not in f.lower()]
            dirs[:] = [d for d in dirs if "ds_store" not in d.lower()]
            # dirs[:] = [d for d in dirs if "ds_store" not in d.lower() and d.lower() not in ignore_dirs]
            items = sorted([(name, 'dir') for name in dirs] + [(name, 'file') for name in files])
            for index, (name, item_type) in enumerate(items, start=1):
                item_path = os.path.join(root, name)
                item_position = f"{current_position}_{index}"

                if item_type == 'dir':
                    traverse_folder(item_path, item_position, current_depth + 1)

                else:
                    relative_file_path = os.path.relpath(item_path, start=absolute_folder_path)
                    file_paths_with_positions.append((item_position, relative_file_path))
            break  # Prevents os.walk from going deeper as recursion handled manually

    absolute_folder_path = os.path.abspath(folder_name)
    traverse_folder(absolute_folder_path, '0', 0)

    return folder_paths_with_positions + file_paths_with_positions



def check_stop() -> bool:
    if global_stop:
        return True
    else:
        return False


def write_response(response):
    with open("raw_responses.txt", "a", encoding='utf-8') as f:
        f.write(response + "\n")


def read_file_structure() -> str:
    """This tool obtains information on the file structure of the documents used to inform the document tools."""
    with open("file_structure.txt", "r", encoding='utf-8') as f:
        return f.read()



async def _async_read_webpage_content(url: str) -> str:
    """Asynchronously read the content of a webpage and return it as a string."""
    if not url.startswith("https://"):
        url = "https://" + url

    # Create an asynchronous crawler instance
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)

    return result.markdown


def read_webpage_content(url: str) -> str:
    """Read the content of a webpage and return it as a string. Use this if a web page address is provided."""
    return asyncio.run(_async_read_webpage_content(url))




def do_google_search(query: str) -> str:
    """YOU DO HAVE THE ABILITY TO DO WEB SEARCHING USING Google. Use this tool to perform a Google search and return the top 3 search results as a single string.
    ALWAYS state that you are using Google search for this information and that it DOES NOT come from the documents provided."""

    # if not isinstance(query, str):
    #     query = str(query)

    max_retries = 3
    attempt = 0
    search_results = []

    while attempt < max_retries:
        try:
            search_results = search(query, num_results=3)
            if search_results:  # Check if search_results is not empty
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

        attempt += 1
        time.sleep(1)  # Wait for 1 second before retrying

    if not search_results:
        return "Failed to retrieve search results after 3 attempts."

    # Initialize an empty string to hold the concatenated results
    search_results_str = ""

    # Iterate through the search results and append each result to the string
    for result in search_results:
        search_results_str += result + "\n"

    return search_results_str



class BaseAgent:
    def __init__(self, llm_model_name="gpt-4.1-mini", embedding_model_name="text-embedding-ada-002"):
    # def __init__(self, llm_model_name="gpt-5.1-mini", embedding_model_name="nomic-embed-text"):
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.llm = OpenAI(model=llm_model_name, timeout=600, max_retries=20, reuse_client=False, temperature=0)  # remove temperature=0.0, let the patch handle it
        # self.embed_model = OpenAIEmbedding(model=embedding_model_name, timeout=600)
        # self.embed_model = OllamaEmbedding(model_name=embedding_model_name,  request_timeout=600)
        
        # print(f"BaseAgent (id: {id(self)}): Requesting shared Ollama embedding model: {self.embedding_model_name}")
        # self.embed_model = helper.get_shared_ollama_embedding(model_name=self.embedding_model_name)
        self.embed_model = helper.get_shared_openai_embedding(model_name=self.embedding_model_name)
        
        # Configure settings directly
        # Settings.llm = self.llm
        # Settings.embed_model = self.embed_model
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        # Settings.embed_batch_size = 8
        
        self.storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )


class DocumentAgent(BaseAgent):
    def __init__(self, doc_title, folder_paths):
        super().__init__()
        # self.chat_memory = chat_memory
        file_name, file_extension = doc_title
        self.doc_title = file_name
        self.doc_extension = file_extension
        self.folder_paths = folder_paths  # Store the folder paths dictionary
        print(f"Initializing DocumentAgent for {doc_title}")

        # self.context_handler = PerDocumentContextCallback(document_identifier=self.doc_title)
        # register_handler(self.context_handler) # Register it globally
        # self.internal_callback_manager = CallbackManager([self.context_handler])

        self.nodes = self.load_or_create_nodes()
        # self.vector_index = self.load_or_create_vector_index()
        # self.summary_index = self.load_or_create_summary_index()
        self.subdoc_summary_pack = self.load_or_create_subdoc_summary_pack()
        self.document_agent = self.create_document_agent()

        # Create summary_agent only when we actually need to generate summaries
        # Check both short and long summary files
        folder_summary_path = f'{self.folder_paths["summary_db_folder"]}/file_summary.json'
        folder_long_summary_path = f'{self.folder_paths["summary_db_folder"]}/file_long_summary.json'

        need_short_summary = not os.path.exists(folder_summary_path)
        need_long_summary = not os.path.exists(folder_long_summary_path)

        if need_short_summary or need_long_summary:
            print(f"Summary files missing (short: {need_short_summary}, long: {need_long_summary}). Creating summary agent for {self.doc_title}")
            self.summary_agent = self.create_summary_agent()
        else:
            print(f"All summary files exist for {self.doc_title}, skipping summary_agent creation")
            self.summary_agent = None

    def reset(self):
        """Resets the internal chat memory of the agent."""
        if hasattr(self.document_agent, 'reset'):
            # print(f"Resetting state for DocumentAgent: {self.doc_title}")
            self.document_agent.reset()

    def load_or_create_nodes(self):
        nodes_path = f'{self.folder_paths["nodes_storage_folder"]}/{sanitize_name(self.doc_title)}_nodes.pkl'
        print(f"Loading or creating nodes for {self.doc_title}")
        if os.path.exists(nodes_path):
            print(f"Nodes file exists. Loading nodes for {self.doc_title}")
            with open(nodes_path, 'rb') as f:
                nodes = pickle.load(f)
        else:
            print(f"Nodes file does not exist. Creating nodes for {self.doc_title}")
            input_files = [
                f'{self.folder_paths["document_store_folder"]}/{sanitize_name(self.doc_title)}{self.doc_extension}']
            input_docs = SimpleDirectoryReader(input_files=input_files).load_data()

            node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


            nodes = node_parser.get_nodes_from_documents(input_docs)


            os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
            with open(nodes_path, 'wb') as f:
                pickle.dump(nodes, f)
        return nodes

    def create_document_agent(self):
        processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        trace_tap = TraceNodeTap()
                # detailed_answer_template_str = f"""\
        # The following context is from document '{self.doc_title}'.
        # ---------------------
        # {{context_str}}
        # ---------------------
        # Given this information and not any prior knowledge, please provide a comprehensive and detailed answer to the query: {{query_str}}

        # **IMPORTANT: Your answer should include both a direct response to the query AND all relevant background information from the document that relates to the query's subject matter. This is especially critical for lists of people, organizations, events, or other entities - you must include the complete list, not partial summaries.**

        # **CRITICAL RULES:**
        # 1.  **Complete Information Extraction:** You MUST include ALL relevant details from the context, not just a direct answer. If the document mentions names, lists, details, or facts related to the query's subject matter, include them ALL. For example, if asked about "a person in movie X" and the document lists all people in that movie, you MUST list all of them, even if none directly answers the query.
        # 2.  **Specific Over General:** When the context provides specific names, dates, numbers, or details, you MUST include them rather than using vague terms. Do NOT use generalizations.
        # 3.  **Cross-Reference for Contradictions:** Compare the user's query against the context. If the query contains facts that are contradicted by the context, you MUST point this out in your answer (e.g., "While the query mentions X, the document states Y.").
        # 4.  **Synthesize, Don't Hallucinate:** You MUST combine all relevant details from the context to form a complete, coherent paragraph.
        # 5.  **Fact-Based Only:** Your answer MUST be based *exclusively* on the information provided in the context.
        # 6.  **No External Information:** DO NOT infer, assume, or add any external information. If the context is insufficient, state that the document does not provide a complete answer.

        # **EXPECTED FORMAT EXAMPLE:**
        # If asked about "a person in movie X" and the document states:
        # "movie X features key figures including John, Mary, Bob, and Susan..."
        # Your answer should include: "The document does not provide information about [specific query details]. However, movie X features the following key figures: John, Mary, Bob, and Susan..."

        # Answer: """
        # detailed_answer_template_str = f"""\
        # You are a precise information extraction bot. Your job is to extract and report ALL relevant information from the context.

        # **CONTEXT:**
        # ---------------------
        # {{context_str}}
        # ---------------------
        # **QUERY:** {{query_str}}

        # **INSTRUCTIONS:**

        # **Step 1: Identify all subjects mentioned in the query.**

        # **Step 2: Scan the context for ANY mention of these subjects.**

        # **Step 3: Report findings using EXACTLY this format:**

        # **Information found in the document:**
        # - [List ALL information about any query subject found EXPLICITLY and DIRECTLY stated in the context, even if partial. NEVER include implied, inferred, or suggested information.]

        # **Document silence (neutral):**
        # - [List what specific information is missing to fully answer the query]

        detailed_answer_template_str = f"""\
        You are a precise information extraction bot. Your job is to extract and synthesize ALL relevant information from the provided context chunks to answer the user's query.

        **CONTEXT:**
        ---------------------
        {{context_str}}  <-- This may contain multiple document chunks.
        ---------------------
        **QUERY:** {{query_str}}

        **INSTRUCTIONS:**

        **CORE DIRECTIVE: EXHAUSTIVE AND VERIFICATIVE REPORTING**
        - Your primary mission is to act as an exhaustive archivist for the provided document context. The user's query is a guide to identify key subjects of interest (e.g., people, organizations, creative works).
        - You MUST scan the entire document context and extract **EVERY SINGLE affirmative fact** related to these key subjects.
        - **VERIFY, DON'T ASSUME:** Even if the query seems to already state a fact (e.g., "What did leader X do?"), your job is to find the sentence in the document that **verifies** that X is indeed the leader. You must extract this verifying fact. Do not omit facts just because they seem to align with the query's premise. Your role is to confirm what the document says, not to skip over known information.
        - After extracting all relevant facts, your synthesis should then attempt to answer the user's question based on this complete set of extracted information.

        **CORE DIRECTIVE: ZERO HALLUCINATION & STRICT ADHERENCE**
        - **Your answer MUST be derived STRICTLY and EXPLICITLY from the text in the provided context chunks. This is your highest priority rule.**
        - If the context chunks do NOT contain information about a subject in the query, you MUST state "Information not found in the document(s):" followed by a description of what is missing.
        - **You are STRICTLY FORBIDDEN from inventing facts, names, dates, or relationships that are not explicitly present in the text.** It is a critical failure to report information about entity A if the document is about entity B.
        - If a connection between two entities is not explicitly stated, you MUST report that the connection is not specified.
        - **STRICT QUOTATION RULE:** Every fact you report MUST quote the exact sentence(s) from the context verbatim, wrapped in double quotes. Do NOT paraphrase or restate in your own words. If you cannot point to the exact sentence, you must state that the information is missing instead of guessing.
        - **CRITICAL: DO NOT merge time/scope qualifiers from different sentences.** If one sentence has a qualifier (e.g., "as of 2017", "during World War II", "in 1990") and another sentence does not, you MUST report them as separate facts. NEVER attach a qualifier from one sentence to a different sentence. **CRITICAL: Every qualifier in your fact MUST exactly match the qualifier in the quoted evidence. If the evidence quote does not contain a qualifier, your fact MUST NOT contain that qualifier either.**
            *   **Example:** If context says "Entity A has attribute X. Entity B has attribute Y as of 2020.", you MUST report them as TWO separate facts: "Entity A has attribute X." AND "Entity B has attribute Y as of 2020." NOT as "Entity A has attribute X as of 2020."
        - **NO NEW INFERENCES:** Do not merge or combine sentences from different parts of the context unless the context itself explicitly links them. Report each supporting quote separately.

        **Step 1: Identify all subjects and attributes in the query.** (e.g., Subject: "Emmanuelle Vaugier", Attribute: "genre", Context: "supporting role movie")
        - When listing subjects, you MUST quote them exactly as they appear in the query. If the query uses a description instead of a proper noun (e.g., "the 13th American President"), you MUST keep that description verbatim unless the context chunk explicitly states the resolved name. Do NOT substitute with outside knowledge.

        **Step 2: Scan ALL provided context chunks for these elements.**
        - Extract EVERY mention of entities from the query, even if not directly linked to the main subject. Include all surrounding sentences that provide context, quoted verbatim.
        - **MANDATORY SUBJECT CONFIRMATION:** For every subject extracted from the query, if any sentence in the context explicitly names that subject, you MUST quote that sentence. If the relationship is spread across consecutive sentences (e.g., the first names the subject and the next states what it operates or does), quote both sentences together so the link is preserved.
        
        **Step 3: Synthesize and Report.**
        - **PRIMARY GOAL: Attempt to build a logical chain based on retrived contexts ONLY.** If one chunk links the Subject to an intermediate entity (e.g., a movie title), and another chunk describes an attribute of that EXACT same entity, you MUST connect these facts. This is your main task.
        - **MANDATORY PLACEMENT RULE:** ALL positive facts, including direct verbatim quotes from context, MUST be placed in "**Information found in the document(s):**" as exact quotes. Even if a fact is combined from sentences, quote each supporting sentence separately in Found. Synthesis section is ONLY for the final synthesized answer summary, not for facts. Direct facts are NOT derived—place them verbatim in Found first.
        - Report the full, synthesized answer if possible.
        - When quoting to establish relationships (e.g., "Subject X operates Venue Y"), include the sentence that names the subject as well as the sentence describing the relationship, even if that requires quoting two consecutive sentences.
        - If a full answer cannot be synthesized, then report the partial facts you found.
        - You MUST base every statement solely on exact wording present in the context chunks. If the context does not state a fact (e.g., the true name behind a description), do NOT supply it.
        - **Do NOT merge entities without explicit evidence.** If the context does not explicitly state that the described subject in the query is the same as an entity named in the document, you must treat them as separate. Report the document facts independently and then state that the connection to the query subject is unspecified.

        **EXAMPLE OF SYNTHESIS:**
        - Query: "What genre did Emmanuelle Vaugier play a supporting role in?"
        - Context Chunk 1: "Emmanuelle Vaugier had a supporting role in '40 Days and 40 Nights'."
        - Context Chunk 2: "'40 Days and 40 Nights' is a romantic comedy."
        - **Your Correct Output:** "Information found: Emmanuelle Vaugier played a supporting role in the romantic comedy '40 Days and 40 Nights'."

        **Step 4: Report findings using EXACTLY this format:**

        **Information found in the document(s):**
        - [For each fact, quote the exact supporting sentence(s) verbatim, e.g., "<exact sentence from context>." Do not paraphrase.]

        **Document silence (neutral):**
        - [List what specific information is still missing after attempting synthesis.]
        - When describing missing information, refer to the subject exactly as worded in the context or query. Do NOT invent or resolve entity names that are not present in the context.
        - Explicitly note when the document fails to confirm a relationship between the query subject and any entities mentioned in the context.

        **IMPORTANT PRINCIPLE:** Always report partial information. 
        - If the context mentions Person X but not in relation to Organization Y, report what IS known about Person X and then state what is NOT known about Person X's relationship to Organization Y.
        - CRITICAL ENTITY MATCHING RULE: By default, treat names with clearly conflicting parts (e.g., different surnames) as different entities. You may assume two mentions refer to the same entity under the following, strictly defined exceptions:
             - the document explicitly states or implies equivalence (e.g., "John William Cheever, also known as John Cheever").
             - the difference is limited to middle names/initials, accents/diacritics, capitalization, punctuation, hyphenation, or honorifics (e.g., "John William Cheever" vs. "John Cheever"; "José" vs. "Jose").
             - one name is a direct translation or transliteration of the other and the surrounding text clearly anchors them to the same role/location/timeframe.
             - a name uses a last name only (e.g., "Cheever") while the other uses the full name, **and** the nearby context provides enough evidence (shared occupation, family relation, or identical event) to justify the linkage.
        - When in doubt, report both names separately and note that their equivalence is not confirmed.
        - Never dismiss relevant information just because it doesn't fully answer the query.

        Answer: """

        detailed_answer_template = PromptTemplate(detailed_answer_template_str)

        # Build the answer query engine once and keep a reference for direct calls
        self.answer_query_engine = self.subdoc_summary_pack.vector_index.as_query_engine(
            llm=self.llm,
            similarity_top_k=RETRIEVER_TOP_K,
            node_postprocessors=[processor, trace_tap],
            embed_model=self.embed_model,
            text_qa_template=detailed_answer_template,
        )

        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.answer_query_engine,
                metadata=ToolMetadata(
                    name="answer_from_document_tool",
                    description=(
                        f"""Extract ALL information related to the query from '{self.doc_title}', including partial matches and contextual information. No implied, inferred, or suggested information. Only report explicit facts from the document."""
                    ),
                ),
            ),
        ]

        system_prompt = f"""\
        You are a senior editor agent for document '{self.doc_title}'. Your task is to produce a comprehensive, accurate answer based on a draft provided by your internal tool.
        **ABSOLUTE OVERRIDING RULE: NEVER infer, imply, or suggest any information that is NOT EXPLICITLY and DIRECTLY stated in the retrieved contexts by your tool. This includes all forms of implicit references, inferences, cross-temporal connections, or drawing conclusions from separate facts unless the document explicitly links them. Specifically, do NOT infer actions, relationships (e.g., "opposing"), or conflicts from mere listings of entities or general affiliations. These must be explicitly stated or clear connective phrases in the document. The document is the SOLE source of truth.**
        **Everything you conclude MUST BE from the retrieved contexts by your tool, you CANNOT use any other knowledges INCLUDING the texts or information from the Question.
        **CLASSIFICATION PRESERVATION RULE**: Preserve qualifiers verbatim; do not group differently qualified entities into one category.
            -   **Correct**: "Allied forces of Romania and France" and "Second Polish Republic forces" are listed separately.
            -   **Incorrect**: "Romania, France, and the Second Polish Republic were Allied forces".

        **EVIDENCE-BOUND ASSERTIONS**: Use only predicates present in the source; if a predicate like "opposed/were Allied forces/participated" does not appear verbatim, do NOT use it.
            -   **Correct**: "The document states 'Also involved were the Allied forces of Romania and France.'"
            -   **Incorrect**: "Romania and France opposed the Central Powers."
        
        **VERBATIM QUOTATION RULE**: When reporting facts, prefer direct quotes from the context to avoid altering meaning. This is especially true for entity names and their relationships.
            -   **Correct**: The context mentions "'Ukrainian nationalists,' 'anarchists,' 'Bolsheviks,' 'Central Powers forces of Germany and Austria-Hungary,' 'White Russian Volunteer Army,' and 'Second Polish Republic forces'."
            -   **Incorrect**: The context lists countries like Ukraine, Russia, Poland...
        **EXPLICIT SUBJECT RESTORE RULE**: Whenever a sourced sentence begins with context-dependent wording (e.g., "Also", "Additionally", pronouns like "It", "They", "This"), you MUST prepend or restate the nearest explicit subject from the same document so that each sentence you provide is self-contained and unambiguous. Do not leave sentences starting with bare pronouns or connective adverbs in your final answer.
        **Step 1: Generate Draft Answer**
        - For any user query, your first and only action is to call the `answer_from_document_tool`. This tool will provide a detailed draft answer.

        **Step 2: Critical Review and Finalize (MANDATORY)**
        After the tool returns the draft answer, you have a final, critical duty before presenting it to the user:

        **VERBATIM PASS-THROUGH RULE:** You MUST preserve every quoted sentence exactly as produced by `answer_from_document_tool`. You may reorder or label the tool output to fit the required structure, but you are strictly forbidden from paraphrasing, trimming, or adding new wording to those sentences.

        **ABSOLUTE FIDELITY RULE: Your final output MUST include EVERY SINGLE POSITIVE FACT reported by the `answer_from_document_tool`. You are NOT a summarizer; you are a reporter. Do not omit any details, especially explicit attributes, even if they only partially answer the user's main question. The tool's output is the ground truth.**
        **FULL NAME PRESERVATION RULE: When the tool output provides a person's full name, your final response MUST use that exact full name in its first mention of the person. Do NOT shorten or simplify it to just a last name unless the tool output itself only provides the shortened version. Preserve the full name to ensure maximum precision and avoid ambiguity.**

        **CRITICAL ANALYSIS RULES:**

        1. **Assess Information Availability:**
        - If the tool found NO information about ANY subject mentioned in the query, then state that the document does not contain the requested information.
        - If the tool found information about SOME but not ALL subjects in the query, you MUST report what was found and clearly state what was missing.
        - NEVER dismiss partial information as "not found" - partial answers are valuable.

        2. **Completeness and Transparency Check:**
        - Ensure ALL relevant information from the document is included, even if it only partially relates to the query.
        - If the document mentions Person X but not their relationship to Organization Y, report what IS known about Person X and state what is NOT known about the relationship.
        - Pay special attention to complete lists of people, organizations, events, or other entities mentioned in the document that relate to any query subject.
        
        3. **Fact-Check Against Query:**
        - Compare factual details in the user's original query against the facts presented in the tool's draft response.
        - Identify any assumptions or claims in the query that cannot be verified from the document.
        - CRITICAL ENTITY MATCHING RULE: By default, treat names with clearly conflicting parts (e.g., different surnames) as different entities. You may assume two mentions refer to the same entity under the following, strictly defined exceptions:
             - the document explicitly states or implies equivalence (e.g., "John William Cheever, also known as John Cheever").
             - the difference is limited to middle names/initials, accents/diacritics, capitalization, punctuation, hyphenation, or honorifics.
             - one name is a direct translation or transliteration of the other and the surrounding text clearly anchors them to the same role/timeframe/event.
             - one mention uses only the surname while another uses the full name **and** nearby context (occupation, family relation, identical event) clearly indicates they are the same person.
        - When you rely on these relaxed matches, state the linkage explicitly (e.g., "The context refers to 'John Cheever', which matches 'John William Cheever' by shared surname and identical role"). If evidence is insufficient, report the names separately and note the uncertainty.
        - **CRITICAL: Avoid Cross-Contextual or Cross-Temporal Inference (HIGHEST PRIORITY)**: NEVER infer connections or relationships between entities or events if the document does not explicitly state them, especially across different time periods or unrelated topics. This rule **OVERRIDE** any instruction to report "ALL information" or "partial information" if such information requires inference. Do NOT use information from one context (e.g., WWII) to make implicit claims about another (e.g., WWI) unless the document explicitly draws that connection. Report only what is explicitly stated for the IMMEDIATE query context.
          - **Example of FORBIDDEN Inference**: If the document states "Percival served in WWI" and "Percival commanded British Commonwealth forces in WWII", do NOT infer or state that "British Commonwealth was a participant in WWI" based on Percival's British nationality in WWI. This is an invalid cross-temporal inference.
        - **CRITICAL: Prohibition on Inferring Actions/Relationships**: Do NOT infer any actions, relationships (e.g., "opposing", "supporting", "allied against"), or conflicts between entities from mere listings of participants or general affiliations. These specific actions, relationships, or conflicts must be explicitly stated with verbs or clear connective phrases within the document. Specifically, if the document only states an entity was "involved" or an "Allied force", do NOT infer their direct opposition to a specific power (e.g., Central Powers) unless the document explicitly states "opposed [specific power]" or "allied against [specific power]". Do NOT categorize military units or forces (e.g., "White Russian Volunteer Army") as "countries" unless the document explicitly refers to them as such. Report only the explicit classification and actions mentioned in the document.
        - Never dismiss relevant information just because it doesn't fully answer the query.

        4. **Highlight Discrepancies:**
        - Point out any contradictions between the user's query assumptions and the document's facts.
        - Use format: "The query suggests [assumption], but the document states [actual fact]."

        5. **Final Output Structure - Choose the appropriate response pattern:**

       **PRIORITY RULE: Always Report Found Information**
        - If the tool found ANY information related to the query subjects (even if incomplete), you MUST use Case C (Partial Information Found) rather than Case A.
        - Example: If asked about "classes of X" but only found general information about "X legislation", report what was found about the legislation and note the specific classes were not mentioned.
       **Case A: Empty or No Response from Tool**
        - If the tool returns an empty response, "Empty Response", or indicates no information was found about ALL query subjects:
        "The document does not contain any information about the subjects mentioned in the query."

        **Case B: Complete Information Found**
        - If the document contains sufficient information to fully answer the query:
        "Based on the document: [complete answer with all relevant information]."

        **Case C: Partial Information Found**
        - If the document contains some but not all information needed to answer the query:
        "Based on the document, here is what I found about [query subjects]: [all relevant information]. However, the document does not contain information about [missing specific elements needed for the complete answer]."
        **Example for your case:**
        "Based on the document, here is what I found about Chinese's tax code: The Income Tax Assessment Act 1963 is a key Chinese statute for calculating income tax and is gradually being rewritten into the Income Tax Assessment Act 2000. However, the document does not specifically mention the three categories under which income is levied."

        **Case D: Information Found but Contradicts Query**
        - If the document contains information that contradicts assumptions in the query:
        "Based on the document: [factual information from document]. This differs from the query's suggestion that [query assumption]."

        +**Case E: Multi-Hop or Multi-Component Query**
            - For queries with multiple linked components (e.g., "author of X's city after Y"):
                - Structure output as:
                    Hop 1: Confirm [first component, e.g., author of X is Z] (Evidence: [direct quote or "Not stated"])
                    Hop 2: Confirm [second component, e.g., Z's city after Y is W] (Evidence: [direct quote or "Not stated"])
                    - Synthesis: [Full answer only if all hops explicitly confirmed; else report gaps, e.g., "Author confirmed as Z, but city not stated."]
            - Always verify and report each hop separately to retain all original query elements. Do not skip or assume any component.

        **DO NOT use external knowledge (including common knowledge). Stick strictly to document content while ensuring transparency about both what IS and what IS NOT available.**
        """


        # system_prompt = f"""\
        #         You are a senior editor agent for document '{self.doc_title}'. Your task is to produce a final, polished answer based on a draft provided by your internal tool.

        #         **Step 1: Generate Draft Answer**
        #         - For any user query, your first and only action is to call the `answer_from_document_tool`. This tool will provide a detailed draft answer.

        #         **Step 2: Critical Review and Finalize (MANDATORY)**
        #         After the tool returns the draft answer, you have a final, critical duty before presenting it to the user:

        #         **CRITICAL RULE: If the tool returns an empty response, "Empty Response", or "None", indicating that the information is not in the document, your final answer MUST state that the document does not contain the requested information. DO NOT use your own knowledge or any external information to answer the query.**

        #         Otherwise, if the tool provided a substantive draft, then proceed with the following checks:
        #         1.  **Completeness Check:** Ensure the draft answer includes ALL relevant information from the document related to the query's subject matter, not just a direct answer. If the document contains names, lists, or details related to the topic, they should all be included. Pay special attention to complete lists of people, organizations, events, or other entities mentioned in the document that relate to the query.
        #         2.  **Fact-Check Against Query:** You MUST meticulously compare the factual details in the user's **original query** against the facts presented in the tool's draft response.
        #         3.  **Ensure Contradictions are Highlighted:** The document is the single source of truth. The draft answer should already point out any contradictions between the user's query and the document's facts (e.g., the query states "Fact A is X", but the document states "Fact A is Y"). Your job is to ensure this critical check has been performed and is clearly stated in the final answer. If the draft misses a contradiction, you must add it.
        #         4.  **Approve for Final Output:** Once you have verified the answer is detailed, factually accurate according to the document, correctly highlights any discrepancies with the user's query, and includes all relevant contextual information, you will provide it as the final response.
        #         """
        document_agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=self.llm,
            # temperature=0,
            verbose=True,
            system_prompt=system_prompt,
            return_message=True,
            additional_kwargs={
            "tool_choice": "auto",
            }
        )

        return document_agent

    def create_summary_agent(self):
        # Temporarily load indices locally within this method
        vector_index = self.load_or_create_vector_index()
        summary_index = self.load_or_create_summary_index()
        
        vector_query_engine = vector_index.as_query_engine(llm=self.llm)
        summary_query_engine = summary_index.as_query_engine(llm=self.llm)
        subdoc_summary_engine = self.subdoc_summary_pack.vector_index.as_query_engine(llm=self.llm)
        query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="summary_agent_vector_tool",
                description=(
                    f"""Use this tool by default for precise, targeted queries requiring specific factual data or numerical information about the document 
                    {self.doc_title}. Ideal for questions seeking exact figures, dates, names, or other specific details.
                    """
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_agent_summary_tool",
                description=(
                    f"""Use this tool exclusively for generating summaries of the document {self.doc_title}. This includes concise overviews, key points, and main ideas. "
                    "Avoid using this tool for detailed analysis or multi-part questions; instead, use the subdoc_summary_tool for those.
                    """
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=subdoc_summary_engine,
            metadata=ToolMetadata(
                name="summary_agent_subdoc_summary_tool",
                description=(
                    f"""Default tool for comprehensive queries requiring a mix of general overview and detailed information. Ideal for multi-part, complex questions
                    about document {self.doc_title} that necessitate understanding broader context as well as specific details.
                    """
                ),
                ),
            ),
        ]

        system_prompt=f"""\
                You are an EXPERT AI and your TOOL number is summary_agent_tool_{self.doc_title}. You are keen to help answer questions.
                *ALWAYS USE THE TOOLS PROVIDED* to gather information so you can fully and carefully answer questions. DO NOT rely on your prior knowledge.
                Think step-by-step, provide thoughtful DETAILED reasoning steps as you think through your response.
                Always provide FULL JUSTIFICATION for your response, based on the information available to you, including QUANTIFIED DATA, EXAMPLE CASES if applicable.
                DETAILED responses are always better than short summaries.
                
                * IMPORTANT: DO NOT rely on your prior knowledge.*
                \
                """
        summary_agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=self.llm,
            verbose=True,
            system_prompt=system_prompt,
        )
        return summary_agent

    def load_or_create_vector_index(self):
        vector_db_path = f'{self.folder_paths["vector_db_folder"]}/{sanitize_name(self.doc_title)}'
        print(f"Loading or creating vector index for {self.doc_title}")
        if os.path.exists(vector_db_path):
            print(f"Vector index exists. Loading vector index for {self.doc_title}")
            storage_context = StorageContext.from_defaults(persist_dir=vector_db_path)
            vector_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            print(f"Vector index does not exist. Creating vector index for {self.doc_title}")
            vector_index = VectorStoreIndex(self.nodes, embed_model=self.embed_model)
            vector_index.storage_context.persist(persist_dir=vector_db_path)
        return vector_index

    def load_or_create_summary_index(self):
        summary_db_path = f'{self.folder_paths["summary_db_folder"]}/{sanitize_name(self.doc_title)}'
        if os.path.exists(summary_db_path):
            print(f"Loading or summary index for {self.doc_title}")
            storage_context = StorageContext.from_defaults(persist_dir=summary_db_path)
            summary_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            print(f"Summary index does not exist. Creating summary index for {self.doc_title}")
            summary_index = SummaryIndex(self.nodes, embed_model=self.embed_model)
            summary_index.storage_context.persist(persist_dir=summary_db_path)
        return summary_index

    def load_or_create_subdoc_summary_pack(self):
        subdoc_db_path = f'{self.folder_paths["subdoc_db_folder"]}/{sanitize_name(self.doc_title)}_subdoc_summary_pack.pkl'
        if os.path.exists(subdoc_db_path):
            print(f"Loading subdoc summary pack for {self.doc_title}")
            with open(subdoc_db_path, 'rb') as f:
                subdoc_summary_pack = pickle.load(f)

            if getattr(subdoc_summary_pack, "embed_model", None) is not self.embed_model:
                print(
                    f"Updating embed_model for loaded SubDocSummaryPack "
                    f"'{self.doc_title}' to this agent-specific instance."
                )
                subdoc_summary_pack.embed_model = self.embed_model

        else:
            print(f"Creating subdoc summary pack for {self.doc_title}")
            input_file_path = f'{self.folder_paths["document_store_folder"]}/{sanitize_name(self.doc_title)}{self.doc_extension}'
            input_docs = SimpleDirectoryReader(input_files=[input_file_path]).load_data()
            subdoc_summary_pack = SubDocSummaryPack(
                input_docs,
                parent_chunk_size=8192,  # default 8192
                child_chunk_size=CHUNK_SIZE,  # default 512
                # parent_chunk_size=2560,
                # child_chunk_size=256,
                llm=OpenAI(model="gpt-4.1-mini", timeout=600, temperature=0),
                # embed_model=OpenAIEmbedding(timeout=600),
                # embed_model=OllamaEmbedding(model_name="nomic-embed-text", request_timeout=600),
                embed_model=self.embed_model,
            )
            with open(subdoc_db_path, 'wb') as f:
                pickle.dump(subdoc_summary_pack, f)
            # with open(subdoc_db_path, 'rb') as f:
            #     subdoc_summary_pack = pickle.load(f)
        return subdoc_summary_pack


class GeneralFolderAgent(BaseAgent):
    def __init__(self, folder_paths, current_folder):
        super().__init__()
        # Use defaultdict for shortened names with empty dict as default
        # self.shortened_names = defaultdict(dict)
        self.shortened_names = {}
        self.validator_extractor_llm = OpenAI(model="gpt-5-nano", timeout=120, max_retries=1)
        self.tool_counter = 1
        self.tool_numbers = {}
        self.folder_paths = folder_paths
        self.dict_summary_path = f'{self.folder_paths["summary_db_folder"]}/file_summary.json'
        self.dict_long_summary_path = f'{self.folder_paths["summary_db_folder"]}/file_long_summary.json'
        self.dict_summary_lock = threading.RLock()
        self.file_operation_lock = threading.RLock()
        self.enable_fact_subject_rewrite = True
        

        self.doc_titles = self.get_doc_titles()

        self.document_agents = self.create_document_agents()



        for doc_title_tuple in self.doc_titles:
            self.generate_tool_name(doc_title_tuple[0])

        
        self.dict_summary_tool = dict()
        self.dict_long_summary_tool = dict()

        self.folder_summary = self.load_or_create_folder_summary(self.shortened_names)
        
        # self.general_folder_agent, self.doc_tools = self.create_general_folder_agent()

        print(f"current_folder: {current_folder}")   
        self.general_folder_agent, self.doc_tools = self.create_general_folder_agent_internal(self.shortened_names)

        self.save_final_tool_name_map() 

        self.folder_agent_name = current_folder
        self.setup_tools()
        # self.setup_advanced_agents()

    # def load_or_create_folder_summary(self):
    #     summary_path = f'{self.folder_paths["summary_db_folder"]}/folder_summary.txt'
    #     if not os.path.exists(summary_path):
    #         print(f"Generating folder summary for {self.folder_paths}")
    #         folder_summary = self.generate_folder_summary()
    #         print("Starting to save folder summary to ", summary_path)
    #         os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    #         with open(summary_path, 'w', encoding="utf-8") as f:
    #             f.write(folder_summary)
    #     else:
    #         print(f"Loading summary for {self.folder_paths}")
    #         with open(summary_path, 'r') as f:
    #             folder_summary = f.read().strip()
    #         with open(self.dict_summary_path, 'r') as file:
    #             self.dict_summary_tool = json.load(file)
        
    #     print("Folder summary generated and saved to ", summary_path)
    #     return folder_summary

    def reset(self):
        """Resets this folder agent and all its document sub-agents."""
        if hasattr(self, 'general_folder_agent') and hasattr(self.general_folder_agent, 'reset'):
            # print(f"Resetting GeneralFolderAgent for: {self.folder_agent_name}")
            self.general_folder_agent.reset()
        
        if hasattr(self, 'document_agents'):
            for doc_agent in self.document_agents.values():
                if hasattr(doc_agent, 'reset'):
                    doc_agent.reset()
                    
    def save_final_tool_name_map(self):
        global global_id_map
        if self.shortened_names:
           
            with helper._file_read_lock:
                global_id_map.update(self.shortened_names)
            
            mapping_path = f'{self.folder_paths["summary_db_folder"]}/shortened_names.json'
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, 'w', encoding="utf-8") as f:
                json.dump(self.shortened_names, f, indent=2)
            
            with helper._file_read_lock:
                current_global_map = global_id_map.copy()  
            
            global_mapping_path = "global_id_map.json"

            with helper._file_read_lock:
                with open(global_mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(current_global_map, f, indent=2)
            
            print(f"Saved {len(self.shortened_names)} shortened name mappings to local folder")
            print(f"Saved {len(current_global_map)} total mappings to global map file")
            
            # helper.update_tool_list()

    def regenerate_failed_long_summaries(self):
        """Actually trigger the regeneration of failed long summaries"""
        print("🔄 Starting to regenerate failed long summaries...")

        # Count total folders that need processing
        processed_docs_path = self.processed_documents_folder

        # Validate that the path exists
        if not os.path.exists(processed_docs_path):
            print(f"❌ Error: Processed documents folder does not exist: {processed_docs_path}")
            return 0

        folders_to_process = []

        for root, dirs, files in os.walk(processed_docs_path):
            if "summary_db" in dirs:
                summary_db_path = os.path.join(root, "summary_db")
                long_summary_file_path = os.path.join(summary_db_path, "file_long_summary.json")
                
                # Case 1: The long summary file does not exist
                if not os.path.exists(long_summary_file_path):
                    folders_to_process.append(root)
                    print(f"📝 Found missing 'file_long_summary.json' in: {os.path.basename(root)}")
                    continue # Move to the next directory
                
                # Case 2: The long summary file exists, check for failures
                try:
                    with open(long_summary_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        has_failed = any("No long summary generated" in str(value) for value in data.values())
                        if has_failed:
                            folders_to_process.append(root)
                            print(f"📝 Found failed summaries in: {os.path.basename(root)}")
                except Exception as e:
                    print(f"Error reading {long_summary_file_path}: {e}")
                    folders_to_process.append(root) # Still try to process if file is corrupted

        if not folders_to_process:
            print("✅ No missing or failed long summaries found!")
            return 0

        print(f"🚀 Processing {len(folders_to_process)} folders with missing or failed summaries...")

        # Process each folder that has failed summaries
        total_regenerated = 0
        for folder_path in folders_to_process:
            try:
                print(f"\n📂 Processing folder: {os.path.basename(folder_path)}")

                # Create folder paths for this specific folder
                folder_paths = {
                    "document_store_folder": os.path.join(folder_path, "document_store"),
                    "summary_db_folder": os.path.join(folder_path, "summary_db"),
                    "nodes_storage_folder": os.path.join(folder_path, "nodes_storage"),
                    "subdoc_db_folder": os.path.join(folder_path, "subdoc_db"),
                    "vector_db_folder": os.path.join(folder_path, "vector_db")
                }

                # Validate that required folders exist
                required_folders = ["document_store_folder", "summary_db_folder"]
                missing_folders = []
                for folder_key in required_folders:
                    if not os.path.exists(folder_paths[folder_key]):
                        missing_folders.append(folder_key)

                if missing_folders:
                    print(f"⚠️  Skipping folder {os.path.basename(folder_path)} - missing required folders: {missing_folders}")
                    continue

                # Create a temporary GeneralFolderAgent for this folder
                temp_agent = GeneralFolderAgent(folder_paths, os.path.dirname(folder_path))

                # Load existing short summaries if available
                short_summary_path = os.path.join(folder_path, "summary_db", "file_summary.json")
                if os.path.exists(short_summary_path):
                    with open(short_summary_path, 'r', encoding='utf-8') as f:
                        temp_agent.dict_summary_tool = json.load(f)
                    print(f"✅ Loaded existing short summaries")

                # Load existing long summaries to identify which ones failed
                long_summary_path = os.path.join(folder_path, "summary_db", "file_long_summary.json")
                failed_docs = []
                existing_data = {}

                if os.path.exists(long_summary_path):
                    try:
                        with open(long_summary_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            for doc_id, summary in existing_data.items():
                                summary_str = str(summary)
                                # Check for both old and new failure patterns
                                is_failed = self.is_failed_summary(summary_str, doc_id)
                                if is_failed:
                                    failed_docs.append(doc_id)
                                    print(f"📝 Detected failed summary for {doc_id}: {summary_str[:100]}...")
                    except Exception as e:
                        print(f"Warning: Could not load existing long summary data: {e}")
                        existing_data = {}
                else:
                    print(f"Warning: Long summary file not found: {long_summary_path}")

                if failed_docs:
                    print(f"🔧 Regenerating {len(failed_docs)} failed long summaries...")

                    # Filter to only process failed documents that have corresponding .md files
                    valid_failed_docs = []
                    for doc_id in failed_docs:
                        doc_path = os.path.join(folder_paths["document_store_folder"], f"{doc_id}.md")
                        if os.path.exists(doc_path):
                            valid_failed_docs.append(doc_id)
                        else:
                            print(f"⚠️  Skipping {doc_id} - corresponding .md file not found")

                    if not valid_failed_docs:
                        print(f"ℹ️  No valid documents found to process in {os.path.basename(folder_path)}")
                        continue

                    temp_agent.doc_titles = [(doc_id, '.md') for doc_id in valid_failed_docs]

                    # Create document agents only for valid failed documents
                    temp_agent.document_agents = {}
                    for doc_title in temp_agent.doc_titles:
                        doc_agent = DocumentAgent(doc_title, folder_paths)
                        temp_agent.document_agents[doc_title] = doc_agent

                    # Generate folder summary (only long summaries for failed docs)
                    # Create an empty tool name map since we only need long summaries
                    empty_tool_name_map = {}
                    temp_agent.generate_folder_summary(empty_tool_name_map, generate_short_summaries=False)

                    # Update the original file with regenerated summaries
                    updated_count = 0
                    for doc_id in failed_docs:
                        if doc_id in temp_agent.dict_long_summary_tool:
                            existing_data[doc_id] = temp_agent.dict_long_summary_tool[doc_id]
                            updated_count += 1

                    # Save updated data
                    with open(long_summary_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)

                    print(f"✅ Updated {updated_count} long summaries in {os.path.basename(folder_path)}")
                    total_regenerated += updated_count
                else:
                    print(f"ℹ️  No failed summaries found in {os.path.basename(folder_path)}")

            except Exception as e:
                print(f"❌ Error processing folder {os.path.basename(folder_path)}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n🎉 Regeneration completed!")
        print(f"📊 Total long summaries regenerated: {total_regenerated}")
        return total_regenerated

    def is_failed_summary(self, summary_str, doc_id):
        """Check if a summary string indicates failure."""
        return (
            "No long summary generated" in summary_str or
            ": None" in summary_str or
            "LLM query completely failed" in summary_str or
            "All LLM queries failed" in summary_str or
            summary_str.endswith(": None") or
            summary_str == f"{doc_id}: None"
        )

    def load_or_create_folder_summary(self, final_tool_name_map):
        summary_path = f'{self.folder_paths["summary_db_folder"]}/folder_summary.txt'
        dict_summary_path = self.dict_summary_path

        # Check if files exist
        file_summary_exists = os.path.exists(dict_summary_path)
        file_long_summary_exists = os.path.exists(self.dict_long_summary_path)
        folder_summary_exists = os.path.exists(summary_path)

        # Check if long_summary file exists but has failed entries
        need_long_summary_regeneration = False
        if file_long_summary_exists:
            try:
                with open(self.dict_long_summary_path, 'r', encoding='utf-8') as f:
                    long_summary_data = json.load(f)
                    # Check for various failure patterns
                    has_failed_entries = False
                    # Treat empty dict as failed (needs regeneration)
                    if isinstance(long_summary_data, dict) and len(long_summary_data) == 0:
                        has_failed_entries = True
                    # Also treat None or non-dict/str as failed
                    if long_summary_data is None:
                        has_failed_entries = True
                    for doc_id, summary in long_summary_data.items():
                        summary_str = str(summary)
                        is_failed = self.is_failed_summary(summary_str, doc_id)
                        if is_failed:
                            has_failed_entries = True
                            break

                    if has_failed_entries:
                        need_long_summary_regeneration = True
                        print(f"Found failed long summary entries in {self.dict_long_summary_path}. Will regenerate.")
            except Exception as e:
                print(f"Error reading long summary file: {e}")
                need_long_summary_regeneration = True

        # ALWAYS handle long_summary independently - if it doesn't exist OR has failed entries, we must generate it
        if not file_long_summary_exists or need_long_summary_regeneration:
            print(f"Long summary file not found at {self.dict_long_summary_path}. Must generate it.")
            # We need to generate long summaries regardless of other files
            if file_summary_exists:
                # Load existing summaries to avoid regenerating everything
                with open(dict_summary_path, 'r', encoding='utf-8') as file:
                    self.dict_summary_tool = json.load(file)
                print(f"Loaded existing summary file with {len(self.dict_summary_tool)} entries")

            # Force re-create summary agents so that prompt changes take effect
            # (If we reuse existing summary_agent instances, they may still carry old prompts.)
            try:
                for dt, agent in self.document_agents.items():
                    if hasattr(agent, 'summary_agent'):
                        agent.summary_agent = None
                print("Reset summary_agent for all DocumentAgents to apply latest prompts")
            except Exception as e:
                print(f"Warning: failed to reset summary_agent: {e}")

            # Generate folder summary which will also generate long summaries
            # If we have existing summaries, we only need to generate long summaries
            need_short_summaries = not file_summary_exists

            # If we don't need to generate short summaries, load existing ones for the generate_doc_summary function
            if not need_short_summaries:
                print("Loading existing short summaries for long summary generation...")
                with open(dict_summary_path, 'r', encoding='utf-8') as file:
                    self.dict_summary_tool = json.load(file)
                print(f"Loaded {len(self.dict_summary_tool)} existing summaries")

            folder_summary = self.generate_folder_summary(final_tool_name_map, generate_short_summaries=need_short_summaries)

            # Save folder summary
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, 'w', encoding="utf-8") as f:
                f.write(folder_summary)
            print(f"Saved newly generated folder_summary.txt")

        else:
            # Long summary exists, load it
            with open(self.dict_long_summary_path, 'r', encoding='utf-8') as file:
                self.dict_long_summary_tool = json.load(file)
            print(f"Loaded long summary file with {len(self.dict_long_summary_tool)} entries")

            # Handle other files normally
            with self.file_operation_lock:
                if folder_summary_exists and file_summary_exists:
                    print(f"Loading existing folder_summary.txt and file_summary.json")
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        folder_summary = f.read().strip()
                    with open(dict_summary_path, 'r', encoding='utf-8') as file:
                        self.dict_summary_tool = json.load(file)

                elif not folder_summary_exists and file_summary_exists:
                    print(f"folder_summary.txt not found but file_summary.json exists. Generating folder summary.")
                    with open(dict_summary_path, 'r', encoding='utf-8') as file:
                        self.dict_summary_tool = json.load(file)

                    # Generate folder summary from existing summaries
                    detailed_summary = []
                    for doc_id, summary in self.dict_summary_tool.items():
                        if doc_id in [doc_tuple[0] for doc_tuple in self.doc_titles]:
                            detailed_summary.append(summary)

                    detailed_summary_text = "\n".join(detailed_summary)
                    folder_summary = self.condense_summary(detailed_summary_text, final_tool_name_map)

                    # Save the newly generated folder summary
                    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                    with open(summary_path, 'w', encoding="utf-8") as f:
                        f.write(folder_summary)
                    print(f"Saved newly generated folder_summary.txt")

                else:
                    print(f"file_summary.json doesn't exist. Generating from scratch.")
                    folder_summary = self.generate_folder_summary(final_tool_name_map, generate_short_summaries=True)

                    # Save folder summary
                    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                    with open(summary_path, 'w', encoding="utf-8") as f:
                        f.write(folder_summary)

        print(f"Folder summary loading/generation complete")
        return folder_summary


    def generate_folder_summary(self, final_tool_name_map, generate_short_summaries=True):
        print("Starting generate_folder_summary")
        if not generate_short_summaries:
            print("Only generating long summaries, skipping short summaries")
        summary_content = []

        def generate_doc_summary(doc_title):
            print(f"Starting to generate summary for document: {doc_title}")
            doc_agent = self.document_agents[doc_title]
            summary_str = ""
            long_summary_str = ""
            
            def _qe_query_with_default_k(prompt_str: str, template=None, top_k_floor: int = 2) -> str:
                try:
                    vec_index = doc_agent.subdoc_summary_pack.vector_index
                    retr = vec_index.as_retriever()
                    # pdb.set_trace()
                    default_k = getattr(retr, "similarity_top_k", 2)
                    k = max(default_k, top_k_floor)

                    # Optional debug preview of retrieved contexts
                    # env_flag = str(os.getenv("SUMMARY_DEBUG", "true")).lower()
                    # print(f"[SummaryDebug] env={env_flag} doc={doc_title[0]} k={k}")
                    # if env_flag in ("1", "true", "yes"):
                    #     try:
                    #         preview_nodes = vec_index.as_retriever(similarity_top_k=k).retrieve(prompt_str)
                    #         print(f"[SummaryDebug] doc={doc_title[0]} top_k={k} retrieved={len(preview_nodes)}")
                    #         for i, nws in enumerate(preview_nodes[:3]):
                    #             node_obj = getattr(nws, "node", nws)
                    #             txt = getattr(node_obj, "text", None)
                    #             if not txt and hasattr(node_obj, "get_content"):
                    #                 try:
                    #                     txt = node_obj.get_content()
                    #                 except Exception:
                    #                     txt = str(node_obj)
                    #             safe = (txt or "").replace("\n", " ")
                    #             print(f"[SummaryDebug] ctx[{i}]: {safe[:280]}")
                    #     except Exception as dbg_e:
                    #         print(f"[SummaryDebug] failed to preview contexts: {dbg_e}")

                    qe = vec_index.as_query_engine(
                        llm=doc_agent.llm,
                        similarity_top_k=k,
                        embed_model=doc_agent.embed_model,
                        text_qa_template=template,
                    )
                    res = qe.query(prompt_str)
                    print("qe query response: ", res, "prompt_str: ", prompt_str)
                    return getattr(res, "response", str(res))
                except Exception as e:
                    print(f"QE query failed for {doc_title[0]}: {e}")
                    return ""
            try:
                # Ensure summary_agent exists if we need to generate summaries
                need_summary_agent = generate_short_summaries or True  # Always need for long summary
                if need_summary_agent and doc_agent.summary_agent is None:
                    print(f"Creating summary_agent for {doc_title[0]} during generation")
                    doc_agent.summary_agent = doc_agent.create_summary_agent()

                if generate_short_summaries:
                    # Generate short summary with retry mechanism
                    max_retries = 3
                    retry_delay = 2
                    summary_str = ""

                    for attempt in range(max_retries):
                        try:
                            print(f"Attempting to generate short summary for {doc_title[0]} (attempt {attempt + 1}/{max_retries})")

                            summary_prompt = f"""
                                Create a COMPREHENSIVE summary of this document that includes:
                                1. The main topic and purpose of the document
                                2. ALL key people mentioned (names, titles, roles)
                                3. ALL specific dates, years, and time periods
                                4. ALL organizations, companies, teams, or institutions
                                5. ALL locations, venues, or places
                                6. ALL significant events, films, books, or works discussed
                                7. SPECIFIC numerical data or statistics if present

                                Format your response as:
                                {doc_title[0]}; summary: [Your comprehensive summary including ALL key details listed above]

                                CRITICAL: 
                                - Include ALL proper nouns, dates, and specific identifiers that someone might search for.
                                - **Personal Names Rule: You MUST replicate every person's name EXACTLY as it is written in the source document, every single time the name appears. Do not alter, reorder, interpret, or omit ANY part of the name. Your 'common sense' about what a name should be is irrelevant; copy the source text verbatim.**
                                    - **Example:** If the source says "John Louis Tommy", you MUST write exactly "John Louis Tommy". Do NOT write "John Tommy" or "J. L. Tommy".
                                - If the document contains any road/route relationship with a location phrase (e.g., "connects to / connecting to / interchange / junction / exit" + "in / at / within a place"), you MUST preserve that phrase inside the summary, including the place.
                                - Do not generalize such clauses (never drop the preposition + place).
                                - Your summary should be ***less than 350 characters*** in length.
                                """

                            # Enforce exact-copy behavior at the engine layer to avoid normalization of names
                            copy_only_template = PromptTemplate("""
                            You must copy ALL person names, titles, organizations and other proper nouns EXACTLY as they appear in the CONTEXT. Do not normalize, shorten, translate, or omit any tokens. Use only strings that occur in CONTEXT.
                            When multiple surface forms of the same name appear (e.g., "John Louis Tommy"), you MUST use the LONGEST contiguous surface form and prefer the EARLIEST occurrence in CONTEXT. Do not drop modifiers like "Louis", "Jr.", "III" etc.

                            **VERY IMPORTANT**: Some names may contain words that are also common nouns (e.g., 'Husband', 'Baker', 'Cook'). You MUST treat these as part of the name and copy them verbatim. For example, if the context says "Husband Jonathan Bryan", you MUST write "Husband Jonathan Bryan", not "Jonathan Bryan". Your assumption about what constitutes a typical name is not relevant; you must follow the source text.

                            CONTEXT:
                            {context_str}
    
                            TASK:
                            {query_str}
                            """)

                            resp_text = _qe_query_with_default_k(summary_prompt, template=copy_only_template)
                            if resp_text:
                                summary_str = resp_text
                                print(f"✅ Successfully generated short summary for {doc_title[0]}: {summary_str[:100]}...")
                                break
                            else:
                                # Handle None case and other invalid responses
                                print(f"⚠️  Attempt {attempt + 1} failed for {doc_title[0]}: Empty response")
                                if attempt < max_retries - 1:
                                    print(f"⏳  Retrying in {retry_delay} seconds...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 1.5

                        except Exception as e:
                            print(f"❌ Attempt {attempt + 1} failed for {doc_title[0]}: {e}")
                            if attempt < max_retries - 1:
                                print(f"⏳  Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 1.5

                    # Fallback if all retries failed
                    if not summary_str:
                        print(f"💥 All {max_retries} attempts failed for short summary of {doc_title[0]}")
                        summary_str = f"{doc_title[0]}: Failed to generate summary after {max_retries} retries"
                else:
                    # Use existing summary from dict_summary_tool
                    existing_summary = self.dict_summary_tool.get(doc_title[0])
                    if existing_summary:
                        summary_str = existing_summary
                        print(f"Using existing summary for {doc_title[0]}: {summary_str[:100]}...")
                    else:
                        print(f"Warning: No existing summary found for {doc_title[0]}, using empty string")
                        summary_str = ""

                # Always generate long summary (since that's what we're here for)
                # Implement retry mechanism for LLM queries
                max_retries = 3
                retry_delay = 2  # seconds
                long_summary_str = None

                for attempt in range(max_retries):
                    try:
                        print(f"Attempting to generate long_summary for {doc_title[0]} (attempt {attempt + 1}/{max_retries})")

                        long_prompt = f"""
                            Create a COMPREHENSIVE summary of this document that includes:
                            1. The main topic and purpose of the document
                            2. ALL key people mentioned (names, titles, roles)
                            3. ALL specific dates, years, and time periods
                            4. ALL organizations, companies, teams, or institutions
                            5. ALL locations, venues, or places
                            6. ALL significant events, films, books, or works discussed
                            7. SPECIFIC numerical data or statistics if present, including any quantities written in words (e.g., "four acts", "two volumes")

                            CRITICAL: Treat spelled-out numbers as numerical details. If such quantities appear, you MUST include them in the summary instead of stating that no significant numerical data exists.

                            Format your response as:
                            {doc_title[0]}; summary: [Your comprehensive summary including ALL key details listed above]

                            CRITICAL: 
                            - Include ALL proper nouns, dates, and specific identifiers that someone might search for.
                            - **Personal Names Rule: You MUST replicate every person's name EXACTLY as it is written in the source document, every single time the name appears. Do not alter, reorder, interpret, or omit ANY part of the name. Your 'common sense' about what a name should be is irrelevant; copy the source text verbatim.**
                                - **Example:** If the source says "John Louis Tommy", you MUST write exactly "John Louis Tommy". Do NOT write "John Tommy" or "J. L. Tommy".
                            - If the document contains any road/route relationship with a location phrase (e.g., "connects to / connecting to / interchange / junction / exit" + "in / at / within a place"), you MUST preserve that phrase inside the summary, including the place.
                            - Do not generalize such clauses (never drop the preposition + place).
                            """

                        # Use the same strict template for long summary and ensure sufficient top-k for early paragraphs
                        copy_only_template = PromptTemplate("""
                        You must copy ALL person names, titles, organizations and other proper nouns EXACTLY as they appear in the CONTEXT. Do not normalize, shorten, translate, or omit any tokens. Use only strings that occur in CONTEXT.
                        When multiple surface forms of the same name appear, you MUST use the LONGEST contiguous surface form and prefer the EARLIEST occurrence in CONTEXT. Do not drop modifiers.

                        **VERY IMPORTANT**: Some names may contain words that are also common nouns (e.g., 'Husband', 'Baker', 'Cook'). You MUST treat these as part of the name and copy them verbatim. For example, if the context says "Husband Jonathan Bryan", you MUST write "Husband Jonathan Bryan", not "Jonathan Bryan". Your assumption about what constitutes a typical name is not relevant; you must follow the source text.

                        CONTEXT:
                        {context_str}

                        TASK:
                        {query_str}
                        """)
                        long_text = _qe_query_with_default_k(long_prompt, template=copy_only_template)

                        # Safely extract response - handle None case properly
                        if long_text:
                            long_summary_str = long_text
                            print(f"✅ Successfully generated long_summary for {doc_title[0]}: {long_summary_str[:100]}...")
                            break  # Success, exit retry loop
                        else:
                            # Handle None case and other invalid responses
                            print(f"⚠️  Attempt {attempt + 1} failed for {doc_title[0]}: Empty response")

                            if attempt < max_retries - 1:
                                print(f"⏳  Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 1.5  # Exponential backoff

                    except Exception as e:
                        print(f"❌ Attempt {attempt + 1} failed for {doc_title[0]}: {e}")
                        if attempt < max_retries - 1:
                            print(f"⏳  Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff

                # Final fallback if all retries failed
                if long_summary_str is None:
                    # Try fallback to existing short summary if available
                    existing_short = self.dict_summary_tool.get(doc_title[0]) if hasattr(self, 'dict_summary_tool') else None
                    if existing_short and isinstance(existing_short, str):
                        try:
                            if " summary:" in existing_short:
                                # Normalize previously saved format "<id>: <content>" or "<id>; summary: <content>"
                                # Prefer the part after " summary:" if present
                                short_part = existing_short.split(" summary:", 1)[1].strip()
                                long_summary_str = f"{doc_title[0]}; summary: {short_part}"
                            else:
                                # If it's already in "<id>: <content>" format, keep content
                                content = existing_short.split(": ", 1)[1] if ": " in existing_short else existing_short
                                long_summary_str = f"{doc_title[0]}; summary: {content.strip()}"
                            print(f"🔁 Fallback to existing short summary for {doc_title[0]}")
                        except Exception:
                            print(f"⚠️ Failed to parse existing short summary for fallback: {existing_short[:100]}")
                            long_summary_str = f"{doc_title[0]}: All LLM queries failed - unable to generate long summary"
                    else:
                        print(f"💥 All {max_retries} attempts failed for {doc_title[0]}. No fallback available.")
                        long_summary_str = f"{doc_title[0]}: All LLM queries failed - unable to generate long summary"

            except Exception as e:
                print(f"Error generating summary for {doc_title}: {e}")
                traceback.print_exc()

            return summary_str, long_summary_str, doc_title

        # print(f"Starting to process {len(self.doc_titles)} documents with ThreadPoolExecutor")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(generate_doc_summary, doc_title) for doc_title in self.doc_titles]
            for future in futures:
                try:
                    summary, long_summary, doc_title = future.result()
                    with self.dict_summary_lock:
                        # Safely handle summary parsing - if it already contains the full format from existing file, use it as-is
                        if generate_short_summaries:
                            # Newly generated summary - extract content
                            if summary and " summary:" in summary:
                                try:
                                    summary_content_part = summary.split(" summary:", 1)[1].strip()
                                    self.dict_summary_tool[doc_title[0]] = doc_title[0] + ": " + summary_content_part
                                except (IndexError, AttributeError) as e:
                                    print(f"Warning: Failed to parse summary for {doc_title[0]}: {e}")
                                    self.dict_summary_tool[doc_title[0]] = doc_title[0] + ": " + str(summary)
                            else:
                                print(f"Warning: Invalid summary format for {doc_title[0]}: {summary[:100] if summary else 'None'}")
                                self.dict_summary_tool[doc_title[0]] = doc_title[0] + ": " + str(summary) if summary else doc_title[0] + ": No summary generated"
                        else:
                            # Using existing summary - it should already be in the correct format
                            if summary:
                                self.dict_summary_tool[doc_title[0]] = summary
                            else:
                                print(f"Warning: Empty summary for {doc_title[0]}")
                                self.dict_summary_tool[doc_title[0]] = doc_title[0] + ": No summary available"

                        # Store long summary as well
                        if long_summary and " summary:" in long_summary:
                            try:
                                long_summary_content = doc_title[0] + ": " + long_summary.split(" summary:", 1)[1].strip()
                            except (IndexError, AttributeError) as e:
                                print(f"Warning: Failed to parse long_summary for {doc_title[0]}: {e}")
                                long_summary_content = doc_title[0] + ": " + str(long_summary)
                        else:
                            print(f"Warning: Invalid long_summary format for {doc_title[0]}: {long_summary[:100] if long_summary else 'None'}")
                            long_summary_content = doc_title[0] + ": " + str(long_summary) if long_summary else doc_title[0] + ": No long summary generated"

                        self.dict_long_summary_tool[f"{doc_title[0]}"] = long_summary_content

                        # summary_content.append with safe parsing
                        doc_title_id = global_id_map.get(doc_title[0])
                        if summary and " summary:" in summary:
                            try:
                                summary_part = summary.split(" summary:", 1)[1].strip()
                                summary_content.append(f"{doc_title_id}: {summary_part}")
                            except (IndexError, AttributeError) as e:
                                print(f"Warning: Failed to parse summary_content for {doc_title[0]}: {e}")
                                summary_content.append(f"{doc_title_id}: {summary}")
                        else:
                            summary_content.append(f"{doc_title_id}: {summary}" if summary else f"{doc_title_id}: No summary")

                    print(f"Successfully processed summary for {doc_title}")
                except Exception as e:
                    print(f"An error occurred while processing {doc_title}: {e}")
                    traceback.print_exc()

        # print("Writing summary files to disk")
        with self.file_operation_lock:
            try:
                # Only save short summaries if they were generated
                if generate_short_summaries:
                    with open(self.dict_summary_path, 'w', encoding="utf-8") as dict_summary_file:
                        json.dump(self.dict_summary_tool, dict_summary_file)
                    print(f"Saved {len(self.dict_summary_tool)} summaries to {self.dict_summary_path}")
                else:
                    print(f"Skipped saving short summaries (using existing ones)")

                # Always save long summaries (since that's what we're here for)
                with open(self.dict_long_summary_path, 'w', encoding="utf-8") as dict_long_summary_file:
                    json.dump(self.dict_long_summary_tool, dict_long_summary_file)
                print(f"Saved {len(self.dict_long_summary_tool)} long summaries to {self.dict_long_summary_path}")

            except Exception as e:
                print(f"Error writing summary files: {e}")
                traceback.print_exc()
        # print(self.dict_summary_tool)

        print("\n Setup summary using multiple threads for " + str(self.doc_titles) + "\n")
        try:
            # print("Attempting to join summary content")
            detailed_summary = "\n".join(summary_content)
            # print(f"Successfully joined summary content. Length: {len(detailed_summary)}")
            
            # print("Starting to generate condensed summary")
            try:
                condensed_summary = self.condense_summary(detailed_summary, final_tool_name_map)
                # print(f"Successfully generated condensed summary. Length: {len(condensed_summary)}")
            except Exception as e:
                print(f"Error in condense_summary: {e}")
                traceback.print_exc()
                condensed_summary = "Error generating condensed summary"
            
            print(f"\nCondensed Folder summary: {condensed_summary}\n")
            return condensed_summary
        except Exception as e:
            print(f"Error in final summary processing: {e}")
            traceback.print_exc()
            return "Error processing summary"

    def condense_summary(self, detailed_summary, final_tool_name_map):
        # Use the first document agent to condense the summary
         
        if self.doc_titles:
            # doc_ids = [doc_tuple[0] for doc_tuple in self.doc_titles]
            final_tool_names_in_order = [final_tool_name_map.get(dt[0], f"{dt[0]}") for dt in self.doc_titles]
            condense_summary_count = 1024
            while condense_summary_count >= 1000:
                print(f"Condensing summary for {len(final_tool_names_in_order)} documents")
                response: ChatResponse = chat(model='deepseek-r1:32b', messages=[
                    {
                        'role': 'user',
                        'content': f"""You are given a detailed summary of all files in a folder. 
                        Your task is to combine all the file summaries into one cohesive folder summary, preserving the key information from each file.

                        
                        OUTPUT FORMAT:
                        - It contains the following files: 
                            1. Document {final_tool_names_in_order[0]}: [Summary of document content]
                            ... and so on.
                        
                        FORMAT REQUIREMENTS:
                        - Output MUST BE IN the SAME LANGUAGE as the input content.
                        - ***ALWAYS USE THE EXACT DOCUMENT IDs provided from {final_tool_names_in_order}, NEVER USE descriptive names.***
                        - THE DOCUMENT ID MUST BE EXACTLY AS PROVIDED IN {final_tool_names_in_order}.
                        - DO NOT repeat content or duplicate entries.
                        - For each document overview, prioritize keeping proper nouns, names, and unique identifiers, but also keep it concise.

                        CHARACTER COUNT LIMIT:
                        - ***HARD LIMIT: Final output MUST be **700** characters or less***
                        - If the summary is too long, you MUST condense it to the limit even if it means losing some information.

                        CRITICAL: 
                        - ***YOU MUST NOT EXCEED THE ***700*** characters limit in the final output. DO NOT INCLUDE ANYTHING ELSE APART FROM the OUTPUT FORMAT.***
                        - Never include the character count in the final output.
                        - ***NEVER include Chinese characters in the final output unless these Chinese characters are from the input content.***
                        
                        Here is the input content to summarize: {detailed_summary}
                        """,
                    },
                ])
                condensed_summary = response.message.content.split("</think>")[1].strip()
                # Hard-fix any LLM drift in document IDs by enforcing the known list in order
                condensed_summary = self._fix_doc_ids_in_summary(condensed_summary, final_tool_names_in_order)
                condense_summary_count = len(condensed_summary)
            
            return condensed_summary
        else:
            return "No summary: no documents were available to summarize."

    def _fix_doc_ids_in_summary(self, text: str, doc_ids_in_order):
        """Ensure each enumerated 'Document <ID>:' line uses the exact ID from doc_ids_in_order.

        We rely on the enumeration order (1., 2., 3., …). If the model mutated an ID
        (e.g., added extra characters), we replace it with the known ID for that position.
        """
        try:
            lines = text.splitlines()
            out = []
            idx = 0
            pattern = re.compile(r'^(\s*\d+\.\s*Document\s+)([^:]+)(:)(.*)$')
            for ln in lines:
                m = pattern.match(ln)
                if m and idx < len(doc_ids_in_order):
                    prefix, _bad_id, colon, rest = m.groups()
                    correct_id = str(doc_ids_in_order[idx])
                    out.append(f"{prefix}{correct_id}{colon}{rest}")
                    idx += 1
                else:
                    out.append(ln)
            return "\n".join(out)
        except Exception:
            return text

    # NOT USED CURRENTLY
    # def create_summary_agent(self):
    #     query_engine_tool = QueryEngineTool(
    #         query_engine=self.general_folder_agent,
    #         metadata=ToolMetadata(
    #             name="summary_tool",
    #             description="Tool to condense detailed summaries into concise sentences.",
    #         ),
    #     )

    #     summary_agent = OpenAIAgent.from_tools(
    #         [query_engine_tool],
    #         llm=self.llm,
    #         verbose=True,
    #         system_prompt="""\
    #             You are an expert summarizer. Your task is to condense detailed content into a single concise sentence.
    #             *ALWAYS USE THE TOOL PROVIDED* to gather information so you can fully and carefully summarize the content.
    #             Think step-by-step, provide thoughtful and concise reasoning steps as you think through your summary.
    #             \
    #             """,
    #     )
    #     return summary_agent

    def get_doc_titles(self):
        document_store_folder = self.folder_paths["document_store_folder"]
        # if "0_1_2_1_1_3_3_4_2_1_10_1_19_1" in document_store_folder:
        #     pdb.set_trace()
        doc_titles = [(os.path.splitext(f)[0], os.path.splitext(f)[1]) for f in os.listdir(document_store_folder) if
                      f.endswith(allowed_extensions)]
        return doc_titles

    def create_document_agents(self):
        document_agents = {}
        print(f"Creating DocumentAgents for {self.doc_titles} using multi-threads\n")

        def create_single_document_agent(doc_title):
            # document_agents = {}
            # for doc_title in self.doc_titles:
            print(f"Creating DocumentAgent for {doc_title}")
            document_agent = DocumentAgent(doc_title, self.folder_paths)
            # if doc_title == "0_1_2_1_1_3_3_4_2_1_10_1_19_1":
            #     pdb.set_trace()
            return (doc_title, document_agent)
            # document_agents[doc_title] = document_agent

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_doc_title = {
                executor.submit(create_single_document_agent, doc_title): doc_title
                for doc_title in self.doc_titles
            }
            initialization_errors = []
            for future, submitted_doc_title in future_to_doc_title.items():
                try:
                    doc_title, document_agent = future.result()  # Get the result from the future
                    document_agents[doc_title] = document_agent
                except Exception as e:
                    initialization_errors.append((submitted_doc_title, e))
                    print(f"An error occurred while processing document {submitted_doc_title}: {e}")

            if initialization_errors and os.environ.get("MILONET_STRICT_INITIALIZATION") == "1":
                failed_docs = ", ".join(str(doc_title) for doc_title, _ in initialization_errors)
                raise RuntimeError(f"Document-agent initialization failed for: {failed_docs}")



        print(f"Finished creating DocumentAgents for {self.doc_titles} using multi-threads\n")
        print("EXPECTED total_document_agent num:", len(self.doc_titles))
        print("ACTUAL total_document_agent num:", len(document_agents))
        return document_agents

    def create_general_folder_agent_internal(self, final_tool_name_map):
       
        all_tools = []
        doc_tools = []
        
        for doc_title in self.doc_titles:
           
            tool_name = final_tool_name_map.get(doc_title[0], f"tool_{doc_title[0]}")
            tool_id = self.dict_summary_tool.get(doc_title[0]).split(": ")[0]
            # tool_id = tool_name.split("tool_")[1] if "tool_" in tool_name else tool_name
            
            try:
                # tool_aim = f"Provides access to document content summarized as: {self.dict_summary_tool.get(doc_title[0], 'No summary available.')}. Use this tool via its name: {tool_name}"
                tool_description = f"""Provides access to the document content of {doc_title[0]} with the document id to answer the query: {tool_id}. Use this tool via its name: {tool_name}.
                Sometimes, you can only find the general answer from the document, but you can still return the answer, even if it is not specific enough."""
                tool_aim = tool_description
                # Wrap each document agent with a lightweight sanitizer to guard against
                # rare polluted plan lines leaking into tool inputs.
                doc_agent = self.document_agents[doc_title]

                def _make_doc_fn(agent_ref, tool_name_str: str, doc_identifier: str):
                    def _to_text(resp_obj):
                        try:
                            if resp_obj is None:
                                return ""
                            if hasattr(resp_obj, "response") and resp_obj.response is not None:
                                return str(resp_obj.response)
                            # LlamaIndex Response has get_response() sometimes
                            if hasattr(resp_obj, "get_response"):
                                return str(resp_obj.get_response())
                            return str(resp_obj)
                        except Exception:
                            return ""

                    def _call(input: str) -> str:
                        clean = _sanitize_query_text(input)
                        if not clean:
                            clean = input if isinstance(input, str) else ""
                        result_text = ""
                        try:
                            # Prefer direct query to the underlying query engine to avoid
                            # agent finalization returning None.
                            if hasattr(agent_ref, "answer_query_engine") and agent_ref.answer_query_engine is not None:
                                resp_obj = agent_ref.answer_query_engine.query(clean)
                                if hasattr(resp_obj, "source_nodes") and resp_obj.source_nodes:
                                    merged_entries: list[tuple[str, str]] = []
                                    last_index_by_doc: dict[str, int] = {}

                                    def _merge_text(prev_text: str, new_text: str) -> str:
                                        prev = prev_text.rstrip()
                                        addition = new_text.lstrip()
                                        if prev and not prev.endswith(('.', '!', '?', '"')):
                                            prev += " "
                                        return f"{prev}{addition}"

                                    for src in resp_obj.source_nodes:
                                        node_obj = src.node
                                        text_val = ""
                                        if hasattr(node_obj, "text") and node_obj.text:
                                            text_val = node_obj.text
                                        elif hasattr(node_obj, "get_content"):
                                            try:
                                                text_val = node_obj.get_content()
                                            except Exception:
                                                text_val = ""
                                        if not text_val:
                                            text_val = str(node_obj)
                                        snippet = text_val.strip()
                                        if not snippet:
                                            continue
                                        parts = [part.strip() for part in re.split(r"[\r\n]+", snippet) if part.strip()]
                                        if not parts:
                                            continue
                                        for part in parts:
                                            if doc_identifier in last_index_by_doc:
                                                idx = last_index_by_doc[doc_identifier]
                                                prev_doc, prev_text = merged_entries[idx]
                                                merged_entries[idx] = (prev_doc, _merge_text(prev_text, part))
                                            else:
                                                merged_entries.append((doc_identifier, part))
                                                last_index_by_doc[doc_identifier] = len(merged_entries) - 1

                                    if merged_entries:
                                        lines = []
                                        seen = set()
                                        for doc_id_entry, snippet_entry in merged_entries:
                                            formatted = f'- [Document {doc_id_entry}]: "{snippet_entry}"'
                                            if formatted not in seen:
                                                seen.add(formatted)
                                                lines.append(formatted)
                                        if lines:
                                            result_text = "\n".join(lines)
                                if not result_text:
                                    text = _to_text(resp_obj).strip()
                                    if text:
                                        result_text = text
                            if not result_text:
                                # Fallback to the agent if needed
                                resp = agent_ref.document_agent.chat(clean)
                                if hasattr(resp, "response") and resp.response:
                                    result_text = str(resp.response)
                                elif hasattr(resp, "message") and getattr(resp.message, "content", None):
                                    result_text = str(resp.message.content)
                                else:
                                    result_text = str(resp)
                        except Exception as e:
                            result_text = f"Tool execution error: {e}"

                        if result_text is None:
                            result_text = ""
                        if not isinstance(result_text, str):
                            result_text = str(result_text)
                        result_text = result_text.strip()

                        print("--- Raw document agent answer ---")
                        print(f"Tool: {tool_name_str}")
                        if result_text:
                            print(result_text)
                        else:
                            print("<empty response>")
                        print("----------------------------------")

                        return result_text
                    return _call

                shortened_id = helper.get_shortened_id_from_file(tool_id) or tool_id
                sanitized_fn = _make_doc_fn(doc_agent, tool_name, shortened_id)
                doc_tool = FunctionTool.from_defaults(
                    fn=sanitized_fn,
                    name=tool_name,
                    description=tool_aim,
                )
                # print(f"tool {tool_name} description: {len(tool_aim)} characters")
                if len(tool_aim) > 1024:
                        print(f"Warning: document tool {tool_name} description is too long!")
                        with open(tool_long_tool_description_file, "a") as f:
                            f.write(f"{tool_name}: {tool_aim}\n\n")
                all_tools.append(doc_tool)
                print(f"Added {doc_tool} into {all_tools}")
                doc_tools.append(doc_tool)
            except Exception as e:
                print(f"Error creating tool for {doc_title}: {e}")
        
        
        '''Below is the original folder agent internal coa system prompt'''
        # folder_agent_internal_coa_system_prompt = f"""\
        # You are an internal query processing agent for this Folder. Your goal is to answer the query using ONLY your **available document-specific tools**: {[t.metadata.name for t in doc_tools]}.
        # **IMPORTANT: NEVER use any tools or call any functions not in your available document-specific tools. NEVER MAKE UP ANY NEW TOOLS OR FUNCTIONS.**

        # **ULTRA-CRITICAL OUTPUT REQUIREMENT: REPORT ALL VERIFIED FINDINGS, EVEN IF PARTIAL.**
        # - Even if you cannot answer the entire original query posed to you, your primary responsibility is to:
        # - **Synthesize and RETURN ALL VERIFIABLE FACTS and pieces of information retrieved from your document tools that are relevant to ANY part of the original query or ANY entities mentioned in it.**
        # - If you find information about one part of the query but not another, you MUST still report the part you found.
        # - Your output should clearly state what was found and, if applicable, what related information was *not* found within your documents.
        # - **DO NOT discard information simply because it doesn't form a complete answer to the overall query given to you. Your goal is to extract and pass on all relevant factual nuggets from your documents.**

        # **CORE WORKFLOW & FAILURE HANDLING:**
        # 1.  **Tool Execution:**
        #     *   NEVER CALL YOURSELF, only call the available document-specific tools.
        #     *   ALL the relevant document tools MUST from your available document-specific tools.
        #     *   You MUST EXECUTE ALL the selected document tools and use the output from these tools to answer the user's query.
        #     *   NEVER wrap [FUNC …] lines in quotation marks, code blocks, or any other wrapper. They must appear at top level so the execution engine can execute them.
        #     *   Execute ALL relevant document tools with the query (or a refined version) with following requirements strictly:
        #         -   **CRITICAL: NEVER use PLACEHOLDER tool names in your plan or tool calls, always use the actual tool name from your *available document-specific tools**.
        #         -   **CRITICAL: NEVER use PLACEHOLDER query strings in your plan or tool calls, always use the actual query string**.
        #         -   ALWAYS use the actual query string with double quotes as the parameter for the tool call, do not use placeholders in the parameter.
        #         -   **The tool execution MUST be in the format of `[FUNC tool_name("query string")]`, the tool name MUST be replaced by the actual tool name in your available document-specific tools and query string MUST be replaced by the actual query string.**
        #         -   When you call (execute) any document tool, you MUST actually execute the tool, lead to the document tool to use its one of query engines to answer the query.
        #         -   If you call (execute) any tools, you MUST carry on until get responses from the document tool's one of query engines, OTHERWISE, it is NOT an execution or call.
        # 2.  **Result Processing & Synthesis:**
        #     *   **NO QUERY CONTAMINATION IN FINAL RESPONSE (ZERO TOLERANCE):** DO NOT introduce any phrasing, claims, entities, or relationships from the user's original query into your final response unless that exact phrasing, claim, entity, or relationship is also explicitly evidenced in the document content by your tools. For example, if the user query mentions 'red cars' and your tools only return information about 'cars', your final response must only talk about 'cars'.
        #     *   Carefully examine the output from EACH called tool.
        #     *   Your final answer MUST be based EXCLUSIVELY on the literal text content successfully returned by these tools.
        #     *   You final answer should be the synthesized textual answer from the document tools' outputs, NOT a plan.
        #     *   Make sure you have actually executed the document tools, lead to the document tool to use its one of query engines to answer the query.
        # 3.  **Output Requirements (If usable information WAS found):**
        #     *   Your ENTIRE and final output MUST be the synthesized textual answer from the outputs from the selected document tools.
        #     *   If you only can find the general answer from the documents, you MUST still return the answer, even if it is not specific enough.
        #     *   ABSOLUTELY NO internal planning steps, NO "==EXECUTING PLAN==", NO "[FUNC tool_name(...)]" traces are allowed in the final output.
        #     *   Cite the document tool's name or reference number if it provided verified information.
        #     *   NEVER use external knowledge or fabricate information.
        # 4.  If your output does not include the outputs from the selected document tools, you MUST reexecute the plan with the outputs from the selected document tools.

        # Your final deliverable is EITHER the synthesized textual result (if any verifiable information for any part of the query was found after tool execution) OR the standard "not found" message (if relevance check fails).
        # """
        '''Above is the end of the original folder agent internal coa system prompt'''

        """Below is the new folder agent internal coa system prompt"""

        FOLDER_AGENT_RULES_FINAL = f"""
        ## CRITICAL RULES (EXPLICITLY ENFORCED):
        - **RELEVANT ENTITY ATTRIBUTE EXTRACTION**: If a document provides a key attribute (like a definition, date range, or location) for a named entity (e.g., "The Bronze Age") that is a component of the overall query context, you MUST extract this attribute as a fact. Your job is to collect all potentially useful evidence, even if the document does not re-state the query's main subject in the same sentence. The final connection will be made by a later agent.
        - **ADDITIONAL FOCUS COMPLIANCE**: When the Combined Input includes an **Additional Focus** section, you MUST treat every item listed there as mandatory scope. Every document tool call and every synthesized statement must explicitly address those focus points—do NOT drop them even if the primary question already seems answered.
        - **EVIDENCE-BOUND ASSERTIONS FIRST**: Your primary duty is to report ONLY what the documents explicitly state.
            - **1. SOURCE ATTRIBUTION PRESERVATION**: If the source text attributes a fact, quote, or description to a specific publication, person, or document, you MUST preserve this attribution in your output. For example, if the input is "The castle is described in 'The Big Book of Castles' as 'very old'", your output MUST include "'The Big Book of Castles' describes the castle as 'very old'" or "The castle is described as 'very old', according to 'The Big Book of Castles'".
            - **2. CLASSIFICATION PRESERVATION**: Preserve qualifiers (e.g., "Allied forces of") verbatim. NEVER group differently qualified entities.
                - **Example**: "Allied forces of Romania and France" + "Second Polish Republic forces" MUST NOT become "Romania, France, and the Second Polish Republic were Allied forces". They must be reported separately.
            - **3. NO SCOPE MERGING**: Keep entity scopes separate. "X forces" (a military unit) is NOT the same as "X" (a country), unless the document explicitly equates them.
            - **4. PREDICATE VERIFICATION**: Only use a verb (predicate) like "opposed" or "participated in" if that EXACT verb is present in the source text for that subject. If not, you MUST quote the source text instead (e.g., "The document states '...Second Polish Republic forces were also involved.'").
            - **5. HARD GATE — SUBJECT·PREDICATE·TIMEFRAME binding**:
                - Any final statement that contains a [subject, predicate, timeframe/alignment] triple MUST be explicitly present 
                for the same subject in the evidence. If not, you MUST downgrade by quoting the weaker original wording 
                (e.g., "also involved"/"mentioned") and/or mark timeframe/alignment as **unspecified**.
        - **SUMMARY GROUNDING RULE**: Final answers must be based exclusively on document facts; do not mention or reuse any 'Additional Focus' wording from the input in the final summary.        
        - Do NOT make any assumptions or inferences to bridge gaps between the document tool's output and the user's query, just report exact texts and state its limitations. For example, do NOT assume "anti-militarist" imply "opposing the Central Powers" unless a document explicitly states it.
        - **Descriptor Preservation:** If the Original Query references an entity only by a descriptor or title (e.g., "the 13th Speaker of the House"), you MUST repeat that descriptor verbatim unless a document tool explicitly provides the resolved name. NEVER replace query wording with presumed identities or external knowledge.
        - **Name Equivalence Guidance:** When two mentions share the same surname and context but differ only by middle names/initials, capitalization, diacritics, punctuation, or honorifics, you may treat them as the same person **only if** the surrounding evidence (e.g., identical occupation, relationships, or explicit alias statements) supports it. If the evidence is weak, list them separately and label the connection as unconfirmed.
        - **Extension for Spelling Variations:** For common spelling variations (e.g., "Ishqbaaz" vs. "Ishqbaaaz", or "color" vs. "colour"), treat them as the same entity **only if** they are highly similar (differ by 1-2 characters) AND the context (e.g., same topic, description, or attributes) strongly supports equivalence. If linked, explicitly note in the fact: "(Linked via spelling variant and matching context)". If uncertain, list separately and mark as "possible variant, unconfirmed". NEVER assume equivalence based on external knowledge—rely solely on document content.
        - **ANTI-HALLUCINATION PRIORITY** 
        - You MUST NOT imply any information or draw conclusions that are not explicitly stated in the retrieved documents. Your response MUST be limited to direct textual evidence.
        - If a document does **not mention** a piece of information (e.g., a country's participation),  
        you **MUST** mark it explicitly as "**UNKNOWN**" or "**unspecified**".  
        You **MUST NOT** imply the absence (NO) of that information.

        - If you have details (e.g., location or participants) confirmed for only one event in a series,  
        you **MUST NOT** generalize these details to the entire series.  
        Each event in a series **MUST** have its own separate confirmation.

        - If the question explicitly asks "**which countries**" and your retrieved document texts  
        **do not explicitly list any countries**,  
        you **MUST** answer clearly with "**The documents do not list any countries.**"

        - "**Neutral countries**" and "**Entente/Allied countries**" are **distinct scopes**.  
        You **MUST NOT** mix or imply equivalence unless explicitly stated.  
        If both are mentioned separately, state clearly "**These are separate contexts, not contradictions**".

        
        **CRITICAL: YOU MUST CALL EVERY SINGLE TOOL IN THE FILTER SET** 
        
        You are an EXPERT FOLDER AGENT. Your PRIMARY TASK is to:
        1. **CALL ALL TOOLS** in the FILTER SET - NO EXCEPTIONS
        2. Use the SAME query string for ALL tools to ensure comprehensive coverage
        3. Generate an abstract plan using placeholders y1, y2, y3, etc.
        
        **MANDATORY RULE**: If the FILTER SET contains 3 tools, your plan MUST contain exactly 3 function calls.
        **MANDATORY RULE**: If the FILTER SET contains 5 tools, your plan MUST contain exactly 5 function calls.
        
        Your task is to generate an abstract plan of reasoning that:
        * Uses **ALL** functions from the FILTER SET - missing any tool is FORBIDDEN
        * Uses placeholders for the specific values and function calls needed
        * The placeholders should be labeled y1, y2, etc. 
        * Function calls should be represented as inline strings like [FUNC {{function_name}}({{input1}}, {{input2}}, ...) = {{output_placeholder}}]

        **MANDATORY OPERATING PROTOCOL:**
        1.  **INTERPRET INPUT & PLAN:**
            *   **FIRST AND MOST IMPORTANT**: Count the tools in the FILTER SET and ensure your plan calls EXACTLY that many tools. If FILTER SET = {{A, B, C}}, you MUST have [FUNC A(...) = y1], [FUNC B(...) = y2], [FUNC C(...) = y3].
            *   The input you receive, under "Combined Input", contains a **CRITICAL DIRECTIVE** with a **FILTER SET** of document tools, and the **Original Query** from the higher-level agent.
            *   **DOCUMENT FIDELITY:** Your SOLE source of information is the EXPLICIT TEXT from your DOCUMENT TOOLS.
            *   **ADHERE TO THE FILTER SET:** Your plan must use ALL tools listed in the **CRITICAL DIRECTIVE'S FILTER SET**. You MUST IGNORE all other tools in the "Available functions" list.
            *   **MANDATORY TOOL COVERAGE**: You MUST create a plan that calls **EVERY SINGLE TOOL** specified in the **FILTER SET**. Missing any tool from the FILTER SET is strictly forbidden, as different tools may contain different parts of the answer.
            *   **COMPREHENSIVE SEARCH STRATEGY**: Since answers may be distributed across multiple document tools, you MUST query ALL available tools in the FILTER SET to ensure complete information retrieval.
            *   Critical: When the original query contains multiple entities, you MUST construct a document-tool query including all entities that are present in the original query for each document-tool call, do NOT split the entities into separate queries because it will miss relevant information. 
                - For example, if the original query is "Are A, B and C films", you MUST make sure every document-tool query including all of "Is A a film", "Is B a film", "Is C a film" in the query string, do NOT split the A, B and C into separate queries or separate document-tool calls.
            *   Do not repeat the same query with the same document tool call in the plan.

        2.  **MANDATORY FUNCTION CALL SYNTAX (CRITICAL for planning):**
            *   **COUNT CHECK**: Before writing your plan, count the tools in FILTER SET. Your plan MUST have exactly that many [FUNC ...] calls.
            *   **COMPLETE TOOL USAGE**: Your plan MUST include a function call for EVERY tool listed in the FILTER SET. Do not skip any tools, as each may contain unique information.
            *   **EXAMPLE**: If FILTER SET = {'tool_A', 'tool_B', 'tool_C'}, your plan MUST contain:
                [FUNC tool_A("your query") = y1]
                [FUNC tool_B("your query") = y2] 
                [FUNC tool_C("your query") = y3]
            * HARD SYNTAX GATE:
                * **AVOID DUPLICATES placeholder in PLAN:** Ensure EVERY FUNC call in your plan has a UNIQUE placeholder (y1, y2, etc.). NEVER call the same tool with the same query string, NEVER use the same yN variable twice.
                * Only output lines that match this regex exactly, one per line: `^\\[FUNC [A-Za-z0-9_]+\\("([^"\\\\]|\\\\.)*"\\) = y[1-9]\\d*\\]$`
                * If any line fails the regex, reformat and retry until ALL lines pass. Do NOT print anything else.
                * Use ASCII double quotes " for the query string; escape internal quotes as \". No smart quotes.
                * Balanced delimiters: one leading [ at start and one trailing ] at end; exactly one (/) pair.
                * The tool name must be EXACTLY one from the FILTER SET (case-sensitive).
                * Placeholders must be sequential and unique: y1, y2, …, yN with no gaps.

            *   Your plan MUST be a sequence of function calls using the strict single-line inline string format: `[FUNC actual_tool_name("query string") = yX]`, where `yX` is a placeholder like `y1`, `y2`, etc.
            *   The `actual_tool_name` MUST be the EXACT name from the "Available functions" list (and your FILTER SET).
            *   The `"query string"` MUST be a self-contained and specific question according to the document tool description, enclosed in DOUBLE QUOTES.
            *   GOOD EXAMPLE: [FUNC actual_tool_name("query string") = yX], where `yX` is a placeholder like `y1`, `y2`, etc.
            *   BAD EXAMPLE: [actual_tool_name("query string") = yX], where `yX` is a placeholder like `y1`, `y2`, etc. THIS IS WRONG, IT MISSES `FUNC` keyword before actual_tool_name.   
            *   **CRITICAL:** You MUST use the `= yX` placeholder at the end of each function call.
            *   **QUERY STRING QUALITY:** Ensure each "query string" is a clear, grammatically correct sentence or phrase with proper spacing, punctuation, and structure. Break down complex queries into logical parts (e.g., "A's residence city after moving to England and A's relation to book B"). Avoid concatenated words without spaces; rephrase for clarity to improve retrieval accuracy.

        3.  **ANTI-HALLUCINATION & VERIFICATION (APPLIES TO EVERY PART OF YOUR RESPONSE):**
            *   NEVER use external/prior knowledge. Base answers ONLY on text returned by CALLED document tools.
            *   **CRITICAL PRINCIPLE:** The Original Query is for context only. You MUST NOT QUOTE, PARAPHRASE, OR REFERENCE the Original Query in your response in a way that implies unconfirmed details are true, UNLESS that exact phrasing is EXPLICITLY CONFIRMED by your document tools.
            *   **EXAMPLE:**
                *   Query: "Information about **green elephants** sighted in **Paris**."
                *   Tool Output: "Document A mentions elephants. It does not mention their color or sightings in Paris."
                *   **INCORRECT RESPONSE:** "Details about **green elephants** sighted in **Paris** could not be confirmed."
                *   **CORRECT RESPONSE:** "Document A mentions elephants. However, details regarding their color or specific sightings in Paris are not provided."
            *   **PRESERVE NEGATIVE INFORMATION:** If any tool explicitly states "does not provide," "does not address," or "no information about" a connection, you MUST preserve this negative information and NOT create that connection even if other tools mention related concepts. When summarizing, you MUST clearly separate information from different contexts and avoid language that implies connections that tools explicitly denied.
            *   **REPORTING PARTIAL INFO:** 
                *   If your tools provide a VERIFIED but general answer, you MUST report this factual information WITH stating its limitations.
                *   If your tools provide a VERIFIED but partial answer which it is related to the query but may not answer the full query, you MUST report this factual information WITH stating its limitations.
        """

        FOLDER_AGENT_STRUCTURE_FINAL = """
        *Available functions (This list contains ALL of this agent's possible document tools, but you must obey the FILTER SET provided in the Combined Input):*
        ```python
        {functions}
        ```
        Combined Input (Directives + Original Query):
        {question}

        Abstract plan of reasoning:
        """
        folder_coa_content_final = f"{FOLDER_AGENT_RULES_FINAL}\n\n{FOLDER_AGENT_STRUCTURE_FINAL}"
        folder_coa_system_prompt_final = ChatMessage(role=MessageRole.SYSTEM, content=folder_coa_content_final)
        FOLDER_COA_REASONING_PROMPT_TEMPLATE_FINAL = ChatPromptTemplate(message_templates=[folder_coa_system_prompt_final])

        folder_coa_refine_prompt_content_final = """You are a highly structured reasoning agent. Your task is to answer a question by synthesizing information from multiple document tool outputs. You MUST follow a strict two-phase process: first extract all facts, then synthesize an answer based ONLY on those extracted facts, using the original question for guidance.

        **--- PHASE 1: INDEPENDENT FACT EXTRACTION ---**

        First, you will meticulously review the 'Previous reasoning' section, which contains outputs from various document tools (y1, y2, etc.). Your goal is to create a comprehensive "Verified Facts List".

        **Rules for Phase 1:**
        1.  **Process Sequentially and Independently:** Go through each tool output one by one. Do NOT let information from one document influence your extraction from another.
        2.  **Resolve Pronouns:** Before listing a fact, resolve any pronouns (e.g., "she", "he", "it") by replacing them with the specific entity name from the context. If a sentence begins with context-dependent wording such as "Also", "Additionally", "Furthermore", "Moreover", "Then", "Later", "Such", "So", or similar connectors, rewrite it so that it explicitly starts with the named subject. Every fact you list must be self-contained.
        3.  **Preserve Scope When Splitting (CRITICAL):** If a sentence depends on an earlier timeframe, location, or other scope (e.g., "during World War II", "with its assignment to...", "as of 2017"), only keep the qualifier when the source sentence itself contains that wording. **ABSOLUTELY FORBIDDEN: You MUST NOT take a qualifier from one sentence and attach it to a different sentence as a fact.** 
            *   **CRITICAL: EVERY QUALIFIER IN YOUR FACT MUST EXACTLY MATCH THE QUALIFIER IN THE EVIDENCE QUOTE.** If the evidence quote does not contain a qualifier, your fact statement MUST NOT contain that qualifier either.
            *   **Example of VIOLATION:**
                *   Document text: "Entity A has attribute X. Entity B has attribute Y as of 2020."
                *   WRONG: [Document X]: Entity A has attribute X as of 2020. (Evidence: "Entity A has attribute X.")
                *   CORRECT: [Document X]: Entity A has attribute X. (Evidence: "Entity A has attribute X.") AND [Document X]: Entity B has attribute Y as of 2020. (Evidence: "Entity B has attribute Y as of 2020.")
            *   **CRITICAL RULE:** If the scope is stated in a different sentence or the linkage is ambiguous, quote the exact evidence sentence verbatim instead of trying to reconstruct or supplement the qualifier. Never invent or merge scope language across sentences.
        4.  **Extract Verbatim (STRICT):** For each document, identify all concrete, positive facts. Use the "Evidence-First Protocol": find the exact supporting quote before listing the fact. **CRITICAL: When copying entity names or descriptions, preserve them EXACTLY as they appear in the source, including all details, aliases, or clarifications (whether in parentheses, after commas, or in any other format).** Do NOT abbreviate, simplify, or paraphrase. For example, if the source says "Entity A (also known as Entity B)", copy the entire phrase; do NOT shorten it to just "Entity A" or paraphrase it to a generic description.
        5.  **Document Support Required:** If you cannot locate a verbatim or near-verbatim supporting quote in the current document output, you MUST instead report `Document <Document ID> did not provide relevant information.` Never cite the Original Query, user instructions, or your own assumptions as evidence. You may reuse entity names or descriptors for clarity, but you MUST NOT promote any factual claim that appears only in the Original Query; when no document support exists, output the fallback sentence exactly as written.
        6.  **List Facts with Source:** Compile all extracted information into a single list under the heading `## Verified Facts List`. Each item MUST be prefixed with its original Document ID.
        7.  **NO SYNTHESIS ALLOWED:** During this phase, you are strictly forbidden from summarizing, comparing, or combining information. You are only a fact collector.

        **Example for Phase 1:**

        *Previous reasoning:*
        [FUNC doc_A("query") = y1] -> Result: "Gertrude Stein was an American novelist. She moved to Paris in 1903."
        [FUNC doc_B("query") = y2] -> Result: "The work of Mina Loy was admired by Gertrude Stein."
        [FUNC doc_C("query") = y3] -> Result: "This document contains no relevant information."

        *Your Correct Phase 1 Output (internal thought process, not shown to user):*
        ## Verified Facts List
        - [Document doc_A]: Gertrude Stein was an American novelist. (Evidence: "Gertrude Stein was an American novelist.")
        - [Document doc_A]: Gertrude Stein moved to Paris in 1903. (Evidence: "She moved to Paris in 1903.")
        - [Document doc_B]: The work of Mina Loy was admired by Gertrude Stein. (Evidence: "The work of Mina Loy was admired by Gertrude Stein.")
        - [Document doc_C]: Contains no relevant information.

        **--- PHASE 2: GROUNDED SYNTHESIS ---**

        After you have completed the "Verified Facts List", you will now synthesize a final answer. This is a critical step where you must distinguish between your goal and your evidence.

        **Rules for Phase 2:**
        1.  **Structure Your Answer (HIGHEST PRIORITY):** Your final response MUST begin by **restating verbatim** ALL the positive, affirmative facts from your `Verified Facts List` that are relevant to any part of the `Question` (even if they don't fully answer the question or they are already provided in the question). **CRITICAL: When restating facts, preserve entity names and descriptions EXACTLY as they appear in the Verified Facts List.** Do NOT paraphrase specific entity names into generic terms (e.g., do NOT change "Entity X (also known as Entity Y)" to other generic terms). This includes facts that verify identity, relationships, and attributes, even if they don't fully answer the question. After restating what is known verbatim, you may then state what specific information required to fully answer the `Question` is missing.
        2.  **CRITICAL - Never Omit Positive Facts:** It is a critical failure to omit a relevant, positive fact from your summary just because you cannot fully answer the question. Your primary duty is to report what IS known before reporting what ISN'T.
        3.  **INCLUDE ALL POSITIVE FACTS, EVEN IF TANGENTIAL:** If a fact in your `Verified Facts List` is positive/affirmative but only indirectly related to the question (e.g., information about a different but mentioned entity), you MUST still include it verbatim with the correct citation. Do not drop such facts simply because they seem weakly relevant.
        4.  **Understand Your Goal vs. Your Evidence:**
            -   **Goal Guidance (from `Question`):** You MUST look at the original `{question}` to understand what the user is asking about. This is your target. It tells you what topics to address and what aspects to prioritize (like an `additional focus`).
            -   **Factual Source (from `Verified Facts List`):** You MUST build your answer **exclusively** using the facts from the "Verified Facts List" you created in Phase 1. This is your only source of truth.
        5.  **CRITICAL GROUNDING RULE:** It is a critical failure to use any detail, claim, or entity from the `{question}` (especially from an `additional focus`) as a fact in your answer unless that exact detail is also present in your "Verified Facts List".
            -   **Example:** If the `Question` asks about "events in 1905" but your `Verified Facts List` contains no information about "1905", your answer MUST state that this information is not available. You CANNOT assume or invent facts to satisfy the `additional focus`.
        6.  **Synthesize and Reconcile:** Combine facts from different sources in your list to construct a comprehensive answer. If facts are contradictory, report the contradiction. If the facts only partially answer the question, state what is known and what is missing.
        7.  **Cite Sources with Mandatory Formatting:** Every factual statement in your final answer MUST be explicitly attributed to its source using the format: `"Document [Document ID] states that [fact]"`. This is a non-negotiable output format. Do not group facts from different documents into a single sentence.
            -   **Correct Example:** `Document doc_A states that X was a novelist. According to Document doc_B, her work was admired by Y.`
            -   **Incorrect Example:** `A was a novelist whose work was admired by Y (Sources: doc_A, doc_B).`
        8.  **MINIMAL BRIDGING VIA TEXTUAL LOGIC:** From the Verified Facts List, you MAY bridge facts using direct logical connections inherent in the verbatim text's grammar or structure (e.g., a possessive modifier linking ownership, or a direct appositive providing equivalence), but ONLY if the bridge arises purely from the wording itself without any external knowledge or assumptions. Always cite the exact quotes and state "This logical bridge is based solely on the textual structure in [quote]". If the connection requires any interpretation beyond the explicit wording, report facts separately and note "logical connection not directly supported by text".
        9.  **FORBIDDEN INFERENTIAL LANGUAGE (YOUR OWN INTERPRETATIONS):** Do NOT add your own inferential words (indicating/suggesting/implying/establishing/confirming) AFTER stating what a document says. These words signal YOUR interpretation, not the document's content. ONLY report what documents explicitly contain using neutral language ("Document X states that..."). Exception: If the original document text itself contains these words, preserve them when quoting verbatim.

        ---
        **Your Turn**
        -----------
        **Question:**
        {question}

        **Previous reasoning:**
        {prev_reasoning}
        
        **I will now begin Phase 1 to construct the Verified Facts List. Once complete, I will proceed to Phase 2, using the Question for guidance and the Verified Facts List as my only source of truth.**
        **Response:**
        """
        # folder_coa_refine_prompt_final = ChatMessage(role=MessageRole.USER, content=folder_coa_refine_prompt_content_final)
        FOLDER_COA_REFINE_PROMPT_TEMPLATE_FINAL = PromptTemplate(folder_coa_refine_prompt_content_final)
        # FOLDER_COA_REFINE_PROMPT_TEMPLATE_FINAL = ChatPromptTemplate(message_templates=[folder_coa_refine_prompt_final])
        """Above is the end of the new folder agent internal coa system prompt"""

        # folder_agent_internal_coa_system_prompt = """Testing system prompt, only return "test"."""
        CoA_worker = CoAAgentWorker.from_tools(
            tools=all_tools,
            llm=self.llm,
            verbose=True,
            reasoning_prompt_template = FOLDER_COA_REASONING_PROMPT_TEMPLATE_FINAL,
            refine_reasoning_prompt_template = FOLDER_COA_REFINE_PROMPT_TEMPLATE_FINAL,
            output_parser=SafeChainOfAbstractionParser(verbose=True, auto_fix_duplicates=True),
            # system_prompt=folder_agent_internal_coa_system_prompt
        )

        CoA_agent = CoA_worker.as_agent(memory=None)
        print(f"Set {all_tools} into {CoA_agent}")
        return CoA_agent, doc_tools

    # def create_general_folder_agent(self):
    #     all_tools = []
    #     doc_tools = []
        
    #     for doc_title in self.doc_titles:
    #         tool_name = self.generate_tool_name(doc_title[0])
    #         # tool_id = tool_name.split("tool_")[1]
            
    #         try:
    #             doc_tool = QueryEngineTool(
    #                 query_engine=self.document_agents[doc_title].document_agent,
    #                 metadata=ToolMetadata(
    #                     name=tool_name,
    #                     # Use the actual tool name in the description for clarity,
    #                     # and fetch the summary using the correct original key.
    #                     description=f"Provides access to document content summarized as: {self.dict_summary_tool.get(doc_title[0], 'No summary available.')}. Use this tool via its name: {tool_name}",
    #                 ),
    #             )
    #             all_tools.append(doc_tool)
    #             print(f"Added {doc_tool} into {all_tools}")
    #             doc_tools.append(doc_tool)
    #         except Exception as e:
    #             print(f"Error creating tool for {doc_title}: {e}")
        
    #     # Save shortened name mappings if any were created
    #     if hasattr(self, 'shortened_names') and self.shortened_names:
    #         self.save_shortened_names()

    #     folder_agent_internal_coa_system_prompt = f"""\
    #     You are an internal query processing agent for this Folder. Your goal is to answer the query using ONLY your **available document-specific tools**: {[t.metadata.name for t in doc_tools]}.
    #     **IMPORTANT: NEVER use any tools or call any functions not in your available document-specific tools. NEVER MAKE UP ANY NEW TOOLS OR FUNCTIONS.**

    #     **CORE WORKFLOW & FAILURE HANDLING:**
    #     1.  **Relevance Check (MANDATORY FIRST STEP):**
    #         *   NEVER use any tools or call any functions not in your available document-specific tools. NEVER call yourself or other folder agents.
    #         *   Analyze the user's query. Determine if any document tool descriptions in your available document-specific tools are SEMANTICALLY RELEVANT.
    #         *   If NO tools are relevant, YOU MUST ONLY RETURN: "Related information was not found in the documents within the folder".
    #         *   If no relevant tools in the filter list: {helper.tool_list}. DO NOT PROCEED FOLLOWING STEPS, YOU MUST ONLY RETURN: "Related information was not found in the documents within the folder after the filter check".
    #     2.  **Tool Execution (If relevant tools found):**
    #         *   NEVER CALL YOURSELF, only call the available document-specific tools.
    #         *   ALL the relevant document tools MUST from your available document-specific tools and ALSO MUST IN the filter tool list.
    #         *   Execute ALL relevant document tools with the query (or a refined version) with following requirements strictly:
    #             -   **CRITICAL: DO NOT USE PLACEHOLDER CALLS, always use the actual tool name**
    #             -   ALWAYS use the actual query string with double quotes as the parameter for the tool call, do not use placeholder in the parameter.
    #             -   **The tool execution MUST be in the format of `[FUNC tool_name("query string")]`, the tool name MUST be in your available document-specific tools and query string MUST be the actual query string.**
    #     3.  **Result Processing & Synthesis:**
    #         *   Carefully examine the output from EACH called tool.
    #         *   Your final answer MUST be based EXCLUSIVELY on the literal text content successfully returned by these tools.
    #         *   If all tools explicitly stated "information not found", "document does not contain", or a similar negative response, your final output MUST be: "Related information was not found in the documents within the folder."
    #     4.  **Output Requirements (If usable information WAS found):**
    #         *   Your ENTIRE output MUST be the synthesized textual answer ONLY.
    #         *   ABSOLUTELY NO internal planning steps, NO "==EXECUTING PLAN==", NO "[FUNC tool_name(...)]" traces are allowed in the final output.
    #         *   Cite the document tool's name or reference number if it provided verified information.
    #         *   NEVER use external knowledge or fabricate information.

    #     Your final deliverable is EITHER the synthesized textual result (if successful) OR the standard "not found" message (if relevance check fails OR all tool calls result in unusable outputs).
    #     """




        # folder_agent_internal_coa_system_prompt = f"""\
        # You are an internal query processing agent for a General Folder.
        # Your ONLY available tools are document-specific query engines: {[t.metadata.name for t in doc_tools]}.
        # Your SOLE TASK is to answer the user's query by meticulously using ONLY these document tools.

        # **ULTRA-CRITICAL INITIAL RELEVANCE CHECK:**
        # Before executing ANY tools, you MUST first determine if ANY of your available tools are SEMANTICALLY RELEVANT to the user's query. A tool is relevant ONLY if its description contains terms or concepts DIRECTLY related to the query's core subject.
        # - If NO tools are semantically relevant to the query, you MUST immediately return: "Related information was not found in the documents within the folder or no relevant tools passed the filter." DO NOT call ANY tools in this case.
        # - NEVER USE PLACEHOLDER CALL, always use the actual tool name.
        # - If at least one tool is relevant, proceed with the execution below.

        # **YOUR ROLE WHEN ACTIVATED (e.g., via a .chat() call):**
        # When semantically relevant tools exist, you MUST:
        # 1. IMMEDIATELY call those relevant document tools with the appropriate query
        # 2. Synthesize ONLY the information explicitly returned by those tools
        # 3. If all called tools return "information not found" responses, your final output MUST be: "Related information was not found in the documents within the folder."

        # **CRITICAL INSTRUCTIONS FOR PROCESSING AND RESPONSE GENERATION:**

        # 1.  **DIRECT EXECUTION OF DOCUMENT TOOLS**: When you receive a query, you MUST determine which of your document tools are relevant and then IMMEDIATELY CALL those relevant document tools with the query (or a refined version of it if necessary). This is not about planning to call; it's about *actually calling* them.

        # 2.  **ABSOLUTE FIDELITY TO DOCUMENT TOOL OUTPUTS DURING SYNTHESIS**:
        #     *   After your document tools return their textual responses, you MUST synthesize these responses.
        #     *   Your synthesized answer MUST be based *exclusively* on the literal text content returned by the document tools you called.
        #     *   **If ALL document tools you called for the query explicitly state "information not found," "document does not contain," or any similar negative response, your *only* permissible synthesized answer is a clear statement that the information was not found in the consulted documents. For example: "The information regarding '[original query topic]' was not found in the documents within this folder."**
        #     *   **DO NOT FABRICATE OR HALLUCINATE**: Under NO circumstances should you invent information or provide details that were not *explicitly and literally* present in the text returned by your document tools. If the tools provide no relevant details for the query, your synthesized answer MUST state that clearly.

        # 3.  **FINAL OUTPUT FORMAT (Synthesized Answer Only)**:
        #     *   Your *entire output* MUST consist ONLY of the final, synthesized textual answer.
        #     *   It should be a direct, concise answer based on the synthesis from step 2.
        #     *   **Example of a GOOD final output:** "Document tool_doc_A states X. Document tool_doc_B indicates Y was not found. Based on these, the answer is X, and Y is not available."
        #     *   **Example of a BAD final output:** "My plan is to call tool_doc_A. [FUNC tool_doc_A(...)=...] Result: X." (This includes internal details/planning).

        # 4.  **CITATIONS WITHIN THE SYNTHESIZED ANSWER**:
        #     *   If a document tool provides information, your synthesized answer should reflect that, citing the document tool's name or the document reference it represents.
        #     *   If a tool states information is not found, DO NOT cite that tool or its document as a source for *found* (fabricated) information.

        # 5.  **NO EXTERNAL KNOWLEDGE OR SELF-REFERENCING PLANS**: Base your synthesized answer ONLY on the explicit text returned by your document tools. Do not use your own prior knowledge. Do not refer to your own plans or reasoning steps in your final output.

        # **To reiterate, your final deliverable is the *synthesized textual result* of your document tool calls, strictly adhering to the information (or lack thereof) provided by those tools.**
        # """
  

    def generate_tool_name(self, base_name):
        global global_id_map
    
        complete_id = base_name
        tool_number = complete_id.split("_")[-1]
        # tool_number = self.tool_counter
        # self.tool_counter += 1

        # complete_id = f"{original_doc_id}_{tool_number}"


        with helper._file_read_lock:
            if complete_id in global_id_map:
                doc_id = global_id_map[complete_id]
                final_tool_name = f"tool_{doc_id}"
                self.shortened_names[complete_id] = doc_id
                self.tool_numbers[final_tool_name] = tool_number
                return final_tool_name


        # if len(complete_id) > 35:
        folder_hash = generate_folder_hash(complete_id, context="document")
        
        doc_id = f"{folder_hash}_{tool_number}"
        tool_name = f"tool_{doc_id}"
        
        self.shortened_names[complete_id] = doc_id
        with helper._file_read_lock:
            global_id_map[complete_id] = doc_id   
    
        print(f"Shortened tool name: {doc_id} for original: {complete_id}")
        print(f"Tool name: {tool_name}")
        # else:
        #     doc_id = complete_id
        #     tool_name = f"tool_{doc_id}"
        #     self.shortened_names[complete_id] = doc_id
        #     with helper._file_read_lock:
        #         global_id_map[complete_id] = doc_id
        self.tool_numbers[tool_name] = tool_number
        print(f"Generated tool name: {tool_name} from base name: {complete_id} and tool number: {tool_number}")
        
        return tool_name

    def _build_verbatim_response_from_plan(self) -> Optional[str]:
        """Construct a deterministic, verbatim response from the last executed CoA plan."""
        plan_text = getattr(helper, "last_plan_text", None)
        if not plan_text:
            return None

        fact_lines: List[str] = []
        for raw_line in plan_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("* Fact "):
                fact_lines.append(stripped)
            elif stripped.startswith("* Limitation"):
                fact_lines.append(stripped)

        if fact_lines:
            helper.last_plan_text = None
            return "\n".join(fact_lines)

        if "No related information found." in plan_text:
            helper.last_plan_text = None
            return "No related information found."

        helper.last_plan_text = None
        return None

    def chat_repl(self, agent, user_input):


        # clean_user_input, trace_id, parent_task_id = parse_ids_from_query(user_input)

        # task_id = f"{getattr(agent, 'name', 'FolderAgent')}_{uuid.uuid4().hex[:8]}"
        # with id_stamped_print(trace_id=trace_id, task_id=task_id, parent_task_id=parent_task_id):
        try:
            global global_stop
            helper.last_plan_text = None
            if not global_stop:
                print("🔄 Still working on it... 🧠⏳ \n")
                task = agent.create_task(user_input)
                # task = agent.create_task(clean_user_input)
                step_output = agent.run_step(task.task_id)
                while not global_stop and not step_output.is_last:
                    step_output = agent.run_step(task.task_id)
                if not global_stop:
                    response = agent.finalize_response(task.task_id)
                    enforced = self._build_verbatim_response_from_plan()
                    if enforced and hasattr(response, "response"):
                        response.response = enforced
                else:
                    response = "Agent was stopped."
            else:
                response = "Agent was stopped."

            if hasattr(response, "response"):
                helper.raw_responses.append(response.response)
        except Exception as e:
            print(e)
            response = "Error, Agent was stopped."
        # print("Intermidiate responses \n")
        if hasattr(response, 'response'):
            print(response.response)
        else:
            print(response)
        return response

    def _extract_verified_facts(self, text: str) -> List[Dict[str, str]]:
        facts: List[Dict[str, str]] = []
        if not text:
            return facts

        lines = text.splitlines()
        in_section = False
        previous_facts: Dict[str, str] = {}
        document_tool_outputs: Dict[str, str] = {}  # Store original document tool outputs

        # First pass: extract document tool outputs
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Tool: "):
                # Extract tool name and output
                tool_match = re.match(r"Tool:\s*(\w+)", stripped)
                if tool_match:
                    tool_name = tool_match.group(1)
                    # Find the next line with document output
                    continue
            elif stripped.startswith("- [Document "):
                # This is a document output line
                doc_match = re.match(r"- \[Document\s+(\w+)\]:\s*\"(.*)\"", stripped)
                if doc_match:
                    doc_id = doc_match.group(1)
                    doc_content = doc_match.group(2)
                    document_tool_outputs[doc_id] = doc_content

        def _safe_load_json(raw_text: str) -> Any:
            if not raw_text:
                return None
            snippet = raw_text.strip()
            try:
                return json.loads(snippet)
            except Exception:
                match = re.search(r"\{.*\}", snippet, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        return None
            return None

        def _rewrite_fact_with_document_context(doc_id: str, fact_text: str, document_content: str) -> str:
            """Rewrite fact to resolve English pronouns using the full document context."""
            if not doc_id or not fact_text or not document_content:
                return fact_text

            prompt = (
                "Given the full document content, rewrite the fact sentence to resolve any unclear pronouns or ambiguous references "
                "by replacing them with the specific entity names mentioned in the document context. "
                "Pronouns include: he/she/it/they and their possessive forms (his/her/its/their), as well as any other ambiguous references. "
                "If the sentence already explicitly names all subjects without unclear pronouns, return it unchanged. "
                "The rewritten fact must remain faithful to the original wording and only replace pronouns with explicit names from the document. "
                "Do NOT add information not present in the fact - only resolve pronouns to eliminate ambiguity. "
                "Return JSON only as {\"fact\": \"<rewritten sentence>\"} with no extra text.\n\n"
                f'Document content:\n"""{document_content}"""\n\n'
                f'Fact to rewrite:\n"""{fact_text}"""\n\n'
                "Output:"
            )
            try:
                response = self.validator_extractor_llm.complete(prompt=prompt)
            except Exception:
                return fact_text
            payload = _safe_load_json(response.text if response else "")
            if isinstance(payload, dict):
                rewritten = payload.get("fact")
                if isinstance(rewritten, str) and rewritten.strip():
                    return rewritten.strip()
            return fact_text

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("## Verified Facts List"):
                in_section = True
                continue
            if in_section:
                if (stripped.startswith("## ") and not stripped.lower().startswith("## verified facts list")) or stripped.startswith("### ") or stripped.startswith("---"):
                    in_section = False
                    continue
                if not stripped.startswith("-"):
                    continue

                raw = stripped[1:].strip()
                document_id = ""
                fact_text = raw
                evidence_text = ""

                if raw.startswith("["):
                    closing = raw.find("]")
                    if closing != -1:
                        doc_token = raw[1:closing]
                        if doc_token.lower().startswith("document "):
                            document_id = doc_token.split(None, 1)[1]
                        else:
                            document_id = doc_token
                        fact_text = raw[closing + 1 :].lstrip(":").strip()

                if "(Evidence:" in fact_text:
                    fact_part, evidence_part = fact_text.split("(Evidence:", 1)
                    fact_text = fact_part.strip()
                    evidence_text = evidence_part.rstrip(")").strip()

                # If evidence is not provided, use the document tool output directly
                if not evidence_text and document_id and document_id in document_tool_outputs:
                    evidence_text = f'"{document_tool_outputs[document_id]}"'

                fact_with_doc = fact_text
                if document_id:
                    if getattr(self, "enable_fact_subject_rewrite", False):
                        # Use full document context for pronoun resolution
                        doc_content = document_tool_outputs.get(document_id, "")
                        rewritten = _rewrite_fact_with_document_context(document_id, fact_text, doc_content)
                        fact_text = rewritten
                    fact_with_doc = f"[Document {document_id}]: {fact_text}"
                    previous_facts[document_id] = fact_text
                else:
                    fact_with_doc = fact_text

                facts.append(
                    {
                        "document_id": document_id,
                        "fact": fact_with_doc,
                        "fact_body": fact_text,
                        "evidence": evidence_text,
                        "raw": raw,
                        "fact_with_doc": fact_with_doc,
                        "canonical": fact_with_doc.lower(),
                    }
                )

        return facts

    def _rebuild_verified_facts_section(self, response_text: str, facts: List[Dict[str, Any]]) -> str:
        if not response_text:
            return response_text

        lines: List[str] = ["## Verified Facts List"]
        for fact in facts:
            fact_line = fact.get("fact", "")
            if not fact_line:
                continue
            evidence_text = fact.get("evidence", "")
            entry = f"- {fact_line.strip()}"
            if evidence_text:
                entry += f' (Evidence: "{evidence_text}")'
            lines.append(entry)

        rebuilt_block = "\n".join(lines)
        pattern = re.compile(r"\n?##\s*Verified Facts List.*?(?=\n## |\n### |\n---|\Z)", re.DOTALL)
        response_text = pattern.sub("", response_text)
        if response_text and not response_text.endswith("\n"):
            response_text += "\n"
        if response_text:
            response_text += "\n" + rebuilt_block + "\n"
        else:
            response_text = rebuilt_block + "\n"
        return response_text

    def handle_query(self, query: str) -> str:
        """Handle query using the general folder agent."""

        
        response: str = f"Information regarding '{query}' was not found in the documents within folder or no relevant tools passed the filter."
        intersection = set([t.metadata.name for t in self.doc_tools]).intersection(set(helper.tool_list))
        has_intersection = bool(intersection)

        if not has_intersection:
            folder_filter_instructions = f"""Your **FILTER SET*** are NONE. Skip any further processing, just return "{response}"."""
        else:
            folder_filter_instructions = f"""
            *CRITICAL DIRECTIVE FOR THIS SPECIFIC TASK: ADHERENCE TO THE FILTER SET: {intersection}.*
            - Your operational scope for THIS TASK is STRICTLY LIMITED to the tools within this FILTER SET: {intersection}.
            - **You MUST IGNORE all other tools listed in "Available functions" that are not in this set.**
            - The plan you create MUST call ALL tools from this filter set.
            """

        def _apply_tool_filter(agent_runner, allowed_tools):
            """Temporarily restrict an agent's tools to the allowed subset."""
            state_stack = []

            def _capture(obj, attr, value):
                if hasattr(obj, attr):
                    state_stack.append((obj, attr, getattr(obj, attr)))
                    setattr(obj, attr, value)

            allowed_map = {tool.metadata.name: tool for tool in allowed_tools}

            worker = getattr(agent_runner, "agent_worker", None)

            for target in filter(None, [agent_runner, worker]):
                _capture(target, "tools", allowed_tools)
                _capture(target, "_tools", allowed_tools)
                _capture(target, "available_tools", allowed_tools)
                _capture(target, "_available_tools", allowed_tools)
                _capture(target, "_tool_dict", allowed_map)
                _capture(target, "_tool_mapping", allowed_map)
                _capture(target, "_tool_name_to_tool", allowed_map)

            retriever = None
            if worker and hasattr(worker, "_tool_retriever"):
                retriever = worker._tool_retriever
            for target in filter(None, [retriever]):
                _capture(target, "tools", allowed_tools)
                _capture(target, "_tools", allowed_tools)
                _capture(target, "_tool_dict", allowed_map)
                _capture(target, "_tool_mapping", allowed_map)
                _capture(target, "_tool_name_to_tool", allowed_map)

            return state_stack

        def _restore_tool_filter(state_stack):
            for obj, attr, value in reversed(state_stack):
                setattr(obj, attr, value)
        
        main_query_display, focus_items = _split_query_and_focus(query)
        display_source = main_query_display if main_query_display else query
        remove_brackets_query = display_source.replace("(", "").replace(")", "")
        focus_block = ""
        if focus_items:
            normalized_items = []
            for item in focus_items:
                cleaned = item.lstrip("- ").strip()
                normalized_items.append(cleaned if cleaned else item.strip())
            focus_lines = "\n".join(f"- {value}" for value in normalized_items)
            focus_block = f"\n\n**Additional Focus (MANDATORY):**\n{focus_lines}"

        prompt = f"""
        {folder_filter_instructions}

        **Original User Query:** {remove_brackets_query}{focus_block}
        """

        """Below is the end of the original folder agent internal coa system prompt"""
        # if not has_intersection:
        #     prompt = f"""Skip any further processing, just return "{response}"."""
        # else:
        #     prompt = f"""\
        #     You are an EXPERT FOLDER AGENT, identified as '{self.folder_agent_name}'.

        #         Your task is to answer the given query by meticulously **using ONLY INTERNAL DOCUMENT TOOLS in the FILTER set {intersection}**. 
        #         **NEVER call yourself or other folder agents, ONLY USE your available document-specific tools in the FILTER set.**
            
        #     **MANDATORY OPERATING PROTOCOL:**
        #     1.  **DOCUMENT FIDELITY & TOOL USAGE:**
        #         *   Your SOLE source of information is the EXPLICIT TEXT from your DOCUMENT TOOLS.
        #             *   **FILTER CHECK:** ***You are provided a specific FILTER set: {intersection}***.
        #                 *   ***NEVER USE ANY TOOLS or CALL ANY FUNCTIONS NOT IN the FILTER set.***
        #             *   For your document tools that ARE in the FILTER set, **you MUST call ALL of them in the format of `[FUNC tool_name("query string")]`, the tool name MUST be replaced by the actual tool name in your available document-specific tools and query string MUST be replaced by the actual query string.**
        #             *   ***NEVER use any PLACEHOLDER TOOL NAME in your plan or tool calls.***
        #         *   **ALWAYS CALL WITH ACTUAL TOOL NAMES, NEVER CALL WITH PLACEHOLDER TOOL NAMES.**
        #         *   ALWAYS CALL WITH ACTUAL QUERY STRING WITH DOUBLE QUOTES, NEVER CALL WITH PLACEHOLDER QUERY.
        #             *   Your MUST call all of the document tools in the FILTER set and use their output to answer the query.
        #         *   When you call (execute) any document tool, you MUST actually execute the tool, lead to the document tool to use its one of query engines to answer the query.
        #         *   If you call (execute) any tools, you MUST carry on until get responses from the document tool's one of query engines, OTHERWISE, it is NOT an execution or call.
        #         *   Remove any brackets in the query text string.
        #         2.  **ANTI-HALLUCINATION & VERIFICATION (APPLIES TO EVERY PART OF YOUR RESPONSE)):**
        #         *   NEVER use external/prior knowledge. Base answers ONLY on text returned by CALLED document tools.
        #             *   **CRITICAL PRINCIPLE FOR ALL STATEMENTS YOU MAKE:**
        #                 -   The user's original query (`{query}`) is used to guide which document tools to call and to understand the user's area of interest.
        #                 -   HOWEVER, you MUST NOT QUOTE, PARAPHRASE, OR REFERENCE the original query (`{query}`) in your response in a way that implies unconfirmed details are true. This is especially important when discussing missing or unconfirmable information, UNLESS that exact phrasing from the query is EXPLICITLY CONFIRMED as a fact by your document tools.
        #                 -   You MUST NOT assume information or phrasing from the input query (`{query}`) is true or should be used to describe entities/events unless that information or exact phrasing is EXPLICITLY CONFIRMED by the textual output of your called document tools.
        #                 -   **APPLICATION TO MISSING/UNCONFIRMED INFORMATION:** When stating that a detail *from the user's query* (`{query}`) cannot be found or confirmed, describe that missing detail using NEUTRAL language. DO NOT repeat unconfirmed descriptive phrases from the user's query (`{query}`) in your statement about what is missing.
        #                 -   **EXAMPLE:**
        #                     *   Query: "Information about **green elephants** sighted in **Paris**."
        #                     *   Tool Output (from Document Agent Tools): "Document A mentions elephants. Document A does not mention their color or sightings in Paris."
        #                     *   **INCORRECT RESPONSE by Folder Agent:** "Details about **green elephants** sighted in **Paris** could not be confirmed." (Contaminated by "green" and "Paris" which were not confirmed attributes of the elephants in the document).
        #                     *   **CORRECT RESPONSE by Folder Agent:** "Document A mentions elephants. However, details regarding their color or specific sightings in Paris are not provided in Document A." (Neutral description of missing info).
        #             *   **REPORTING PARTIAL BUT RELEVANT INFORMATION (VERY IMPORTANT):**.
        #                 *   If your document tools provide a VERIFIED but general answer to the query (`{query}`), you MUST report this factual information, even the answer is not specific enough.
        #             *   Discard any information that fails this strict verification.    
        #     3.  **CITATION & OUTPUT (When returning to the calling agent):**
        #             *   Your final output CANNOT be a plan without the outputs from the selected document tools.
        #         *   Your response MUST be the synthesized textual answer. NO execution traces (e.g., "==EXECUTING PLAN==").
        #             *   You MUST present any verified factual information found by your document tools that can be used to answer the query ({query}), even if it's a general answer.
        #         *   Cite ONLY document tools that provided VERIFIED information, using their unique tool name/reference (e.g., `tool_0_1_2_X`).
        #         *   NEVER cite yourself (the Folder Agent) or other Folder Agents as a document source.

        #     **QUERY TO PROCESS:** {query}
        #     """
        """Above is the end of the original folder agent internal coa system prompt"""

        if not has_intersection:
            print(f"==== Executing GeneralFolderAgent.handle_query for query: {query} ====")
            print("No document tools matched the current filter set; returning default response.")
            return response
        # pdb.set_trace()
        allowed_tools = [tool for tool in self.doc_tools if tool.metadata.name in intersection]
        if not allowed_tools:
            print(f"==== Executing GeneralFolderAgent.handle_query for query: {query} ====")
            print("Filter set computed but no matching FunctionTool objects found; returning default response.")
            return response

        allowed_doc_ids = {
            tool.metadata.name
            for tool in allowed_tools
            if getattr(tool, "metadata", None) and not tool.metadata.name.startswith("tool_")
        }

        tool_state_snapshot = _apply_tool_filter(self.general_folder_agent, allowed_tools)
        response = ""

        try:
            print(f"==== Executing GeneralFolderAgent.handle_query for query: {query} ====")
            print(f"Available document tools: {[t.metadata.name for t in self.doc_tools]}")
            print(f"Applied filter set: {[t.metadata.name for t in allowed_tools]}")
            print("Starting call to self.general_folder_agent")
            #    pdb.set_trace()
            response = self.chat_repl(self.general_folder_agent, prompt)
            print(f"Actual response content from self.general_folder_agent: '{response}'")
        except Exception as e:
            print(f"Exception during query: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            _restore_tool_filter(tool_state_snapshot)

        # return response
        if not response or str(response).strip() == "":

            print(f"GeneralFolderAgent received an empty or whitespace-only response for query: {query}")
            # Decide how to handle this, maybe return the standard "not found"
            return AgentChatResponse(response="Related information was not found in the documents within the folder.")

        response_text = response.response if isinstance(response, AgentChatResponse) else str(response)
        extracted_facts = self._extract_verified_facts(response_text)
        if extracted_facts:
            response_text = self._rebuild_verified_facts_section(response_text, extracted_facts)
            helper.folder_agent_verified_facts[self.folder_agent_name] = extracted_facts
            if "Structured Facts (machine-readable):" not in response_text:
                structured_payload = json.dumps({"verified_facts": extracted_facts}, ensure_ascii=False, indent=2)
                response_text = (
                    response_text.rstrip()
                    + "\n\nStructured Facts (machine-readable):\n```json\n"
                    + structured_payload
                    + "\n```\n"
                )

        if isinstance(response, AgentChatResponse):
            response.response = response_text
        else:
            response = AgentChatResponse(response=response_text)

        fact_doc_ids = {fact["document_id"] for fact in extracted_facts or [] if fact.get("document_id")}
        missing_doc_ids = sorted(
            doc_id for doc_id in allowed_doc_ids if doc_id and doc_id not in fact_doc_ids
        )
        if missing_doc_ids:
            missing_lines = [
                f"- Document {doc_id}: no relevant facts returned." for doc_id in missing_doc_ids
            ]
            missing_section = "\n\nDocuments with no relevant facts:\n" + "\n".join(missing_lines)
            response_text = response_text.rstrip() + missing_section + "\n"
            if isinstance(response, AgentChatResponse):
                response.response = response_text
            else:
                response = AgentChatResponse(response=response_text)

        # If the framework expects a specific format (e.g. JSON) and gets plain text,
        # or if certain string responses are considered "errors" by the calling framework.
        # The 'raw_output: null, is_error: true' suggests the calling framework didn't like 'response'.

        if isinstance(response, AgentChatResponse):
            return response

        return AgentChatResponse(response=str(response))

    def setup_tools(self):
        print("Setting up tools for GeneralFolderAgent\n")
        self.call_general_folder_agent_tool = FunctionTool.from_defaults(fn=self.handle_query)
        self.call_check_stop_tool = FunctionTool.from_defaults(fn=check_stop)


        self.tools = [
            self.call_general_folder_agent_tool,
            self.call_check_stop_tool

        ]

    def setup_advanced_agents(self):


        self.react_agent = ReActAgent.from_tools(
            self.tools,
            # memory=self.memory,
            llm=self.llm,
            verbose=True,
            # max_iterations=22
        )

        self.agent_worker = LATSAgentWorker.from_tools(
            tools=self.tools,
            # memory=self.memory,
            llm=self.llm,
            num_expansions=2,  # how many new branches to make from each node
            max_rollouts=3,  # how far down each branch to go; using -1 for unlimited rollouts
            verbose=True,
        )

        self.ToT_agent = AgentRunner(self.agent_worker)

        self.pack = CoAAgentPack(tools=self.tools, llm=self.llm)
        self.CoA_worker = CoAAgentWorker.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            output_parser=SafeChainOfAbstractionParser(verbose=True),
            # max_iterations=22
        )
        self.CoA_agent = self.CoA_worker.as_agent()
        print("Advanced agents setup completed")

    def save_shortened_names(self):
        if self.shortened_names:
            mapping_path = f'{self.folder_paths["summary_db_folder"]}/shortened_names.json'
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, 'w', encoding="utf-8") as f:
                json.dump(self.shortened_names, f, indent=2)
            print(f"Saved {len(self.shortened_names)} shortened name mappings")


class TopDocAgent(BaseAgent):
    def __init__(self, doc_agents_tools):
        super().__init__()
        # self.chat_memory = chat_memory
        self.document_agent_tools = doc_agents_tools
        self.do_google_search_tool = FunctionTool.from_defaults(fn=do_google_search)
        self.read_webpage_content_tool = FunctionTool.from_defaults(fn=read_webpage_content)
        # self.read_pre_five_messages_tool = FunctionTool.from_defaults(fn=read_pre_five_messages)
        # self.top_doc_agent = self.create_top_doc_agent()
        self.sub_docs_agents = self.create_top_sub_doc_agents()
        self.web_search_agent = self.create_web_search_agent()
        # self.setup_tools()



    def partition_doc_tools(self, doc_tools, max_per_group=128):
        groups = []
        total = len(doc_tools)
        num_groups = math.ceil(total / max_per_group)
        for i in range(num_groups):
            group_tools = doc_tools[i * max_per_group: (i + 1) * max_per_group]
            groups.append(group_tools)
        return groups

    def create_sub_docs_agent(self, group_id, sub_doc_agent_tools):
        doc_info = "\n".join([f"{tool.metadata.name}" for tool in sub_doc_agent_tools])
        sub_doc_agent = OpenAIAgent.from_tools(
            tools=sub_doc_agent_tools,
            llm=self.llm,
            verbose=True,
            system_prompt=f"""\
    You work as a sub document agent for group {group_id} and have access to the following document tools:
    {doc_info}
    When presented with a query, your task is to retrieve the precise information from your available tools.
    Please always use the tools provided to answer the question. Do not rely on prior knowledge.
    If no useful information can be used to answer the query, just tell the users and you do NOT cite the documents.
    If you can answer the query from documents, ALWAYS AND ONLY state the source with its DOCUMENT TOOL NUMBER in your final answer.
    In any case, if any information in the response not come from the document, you MUST let users be aware of it.
    """
        )
        return sub_doc_agent

    def create_top_sub_doc_agents(self):
        sub_tool_groups = self.partition_doc_tools(self.document_agent_tools, max_per_group=128)
        sub_docs_agents = []
        for idx, group in enumerate(sub_tool_groups):
            sub_agent = self.create_sub_docs_agent(group_id=idx, sub_doc_agent_tools=group)
            sub_docs_agents.append((sub_agent, idx))

        return sub_docs_agents


    def create_web_search_agent(self):
        # self.read_all_doc_tool_info_tool = FunctionTool.from_defaults(fn=read_all_doc_tool_info)
        web_search_agent = OpenAIAgent.from_tools(
            # memory=self.chat_memory,
            tools=[self.do_google_search_tool, self.read_webpage_content_tool],  # self.read_pre_five_messages_tool],
            llm=self.llm,
            verbose=True,
            system_prompt=f"""\
            You are an AI expert in climate change mitigation named MiloNet. You work as an agent which has web search capabilities, as shown below:
                1. Always start by calling do_google_search_tool to find the most relevant URL.
                2. Then call read_webpage_content_tool on that URL.
                    - If it returns None, move on to the next most relevant URL and repeat until you get non-empty content.
                3. Once you have that content, stop calling tools and provide your final answer, clearly stating it came from an online source, and list all sources.
            You MUST make users aware that the answer is found online (state the source) and state the sources in your final answer.
            """

        )
        return web_search_agent



class MultiFolderAgent(BaseAgent):
    def __init__(self, processed_documents_folder):
        super().__init__()
        print("Initializing MultiFolderAgent...")
        self.coa_agents = []
        self.processed_documents_folder = processed_documents_folder
        self.folder_agents_lock = threading.RLock()
        
        # Create folder agents
        self.folder_agents, self.top_sub_docs_agents, self.web_search_agent = self.create_folder_agents(
            processed_documents_folder
        )
        print("Creating folder agents...")
        self.setup_tools()
        print("Tools set up.")
        self.setup_advanced_agents()
        print("Advanced agents set up.")

    def reset_all_agents(self):
        """Resets the state of all underlying agents to prevent context bleeding."""
        print("--- RESETTING ALL AGENT STATES ---")
        
        # Reset all folder agents, which in turn reset their document agents
        with self.folder_agents_lock:
            for folder_name, folder_agent in self.folder_agents.items():
                if hasattr(folder_agent, 'reset'):
                    folder_agent.reset()

        # Reset the top-level orchestrating agents (SubCoAs)
        if hasattr(self, 'coa_agents'):
            for agent in self.coa_agents:
                if hasattr(agent, 'reset'):
                    # agent_name = getattr(agent, 'name', 'Unnamed Agent')
                    # print(f"Resetting top-level agent: {agent_name}")
                    agent.reset()
        
        # Reset other top-level agents if they exist and are stateful
        if hasattr(self, 'web_search_agent') and hasattr(self.web_search_agent, 'reset'):
            print("Resetting web search agent.")
            self.web_search_agent.reset()

        print("--- AGENT STATE RESET COMPLETE ---")

    def convert_ids_in_text(self, text, to_shortened=True):
       
        if not text:
            return text
            
       
        doc_pattern = r'(Document|Tool)\s+([0-9_]+|[a-z0-9]{8}_\d+|tool_[a-zA-Z0-9_]+)'
        
        def replace_id(match):
            prefix, id_val = match.groups()
            if prefix == 'Tool':
                if not id_val.startswith('tool_'):
                    shortened_id = get_shortened_id(id_val) if to_shortened else id_val
                    if shortened_id:
                        return f"tool_{shortened_id}"
                    return f"tool_{id_val}"
                return id_val
            elif prefix == 'Document':
                if id_val.startswith('tool_'):
                    id_val = id_val[5:]
                
                shortened_id = get_shortened_id(id_val) if to_shortened else id_val
                if shortened_id:
                    return f"{prefix} {shortened_id}"
                return f"{prefix} {id_val}"        
            return match.group(0)
        return re.sub(doc_pattern, replace_id, text)

    def set_stop(self, stop):
        global global_stop
        global_stop = stop
        print("---------------------------------")
        if global_stop:
            print("stop: ", global_stop, " it will take about one min to stop ⛔.\n")
        else:
            print("stop: ", global_stop)
        print("---------------------------------")

    def create_folder_agents(self, processed_documents_folder):
        folder_agents = {}
        doc_agents = []
        dict_summary_tool_total = dict()
        dict_long_summary_tool_total = dict()
        print(f"Creating folder agents for {processed_documents_folder}")
        self.create_folder_agents_recursive(processed_documents_folder, folder_agents, doc_agents,
                                            dict_summary_tool_total, dict_long_summary_tool_total)
        # pdb.set_trace()
        # try:
        #     for key in dict_summary_tool_total.keys():
        #         helper.add_doc_summary_to_store(dict_summary_tool_total[key], key)
        # except Exception as error:
        #     print(error)

        # Store long summaries
        try:
            print(f"dict_long_summary_tool_total keys: {list(dict_long_summary_tool_total.keys())}")
            print(f"dict_long_summary_tool_total size: {len(dict_long_summary_tool_total)}")

            for key in dict_long_summary_tool_total.keys():
                long_summary_value = dict_long_summary_tool_total[key]
                print(f"Processing key: {key}, value length: {len(long_summary_value) if long_summary_value else 0}")
                if long_summary_value:
                    helper.add_doc_summary_to_store(long_summary_value, key)
                    print(f"Stored long summary for {key} as {key}")
                else:
                    print(f"Skipping empty long summary for key: {key}")
        except Exception as error:
            print(f"Error storing long summaries: {error}")
            import traceback
            traceback.print_exc()

        global_top_doc_agent = TopDocAgent(doc_agents)
        # top_doc_agent = global_top_doc_agent.top_doc_agent
        top_sub_docs_agents = global_top_doc_agent.create_top_sub_doc_agents()
        web_search_agent = global_top_doc_agent.web_search_agent
        # return folder_agents, top_doc_agent, web_search_agent
        return folder_agents, top_sub_docs_agents, web_search_agent


    def create_folder_agents_recursive(self, current_folder, folder_agents, doc_agents, dict_summary_tool_total, dict_long_summary_tool_total):
        print(f"Processing folder: {current_folder}")
        # if "0_1_2_1_1_3_3_4_2_1_10_1_19" in str(current_folder):
        #     pdb.set_trace()
        sub_folders = [d for d in os.listdir(current_folder) if os.path.isdir(os.path.join(current_folder, d))]

        # Check if the current folder or any of its subfolders contain files
        def folder_contains_files(folder):
            for _, _, files in os.walk(folder):
                # files = [f for f in files if "ds_store" not in f.lower()]
                if files:
                    return True
            return False

        # for sub_folder in sub_folders:
        # def get_safe_path(current_path, target_folder):
        #     """
        #     Returns the current_path if its basename is already target_folder,
        #     otherwise returns os.path.join(current_path, target_folder).
        #     """
        #     if os.path.basename(current_path) == target_folder:
        #         return current_path
        #     return os.path.join(current_path, target_folder)

        def create_folder_agent_recursive(sub_folder):
            global generate_folder_hash
            sub_folder_path = os.path.join(current_folder, sub_folder)

            if folder_contains_files(sub_folder_path):
                # sub_folder_paths = {
                #     "document_store_folder": get_safe_path(sub_folder_path, "document_store"),
                #     "nodes_storage_folder": get_safe_path(sub_folder_path, "nodes_storage"),
                #     "subdoc_db_folder": get_safe_path(sub_folder_path, "subdoc_db"),
                #     "summary_db_folder": get_safe_path(sub_folder_path, "summary_db"),
                #     "vector_db_folder": get_safe_path(sub_folder_path, "vector_db"),
                # }
                sub_folder_paths = {
                                "document_store_folder": f"{sub_folder_path}/document_store",
                                "nodes_storage_folder": f"{sub_folder_path}/nodes_storage",
                                "subdoc_db_folder": f"{sub_folder_path}/subdoc_db",
                                "summary_db_folder": f"{sub_folder_path}/summary_db",
                                "vector_db_folder": f"{sub_folder_path}/vector_db",
                            }

                if os.path.exists(sub_folder_paths["document_store_folder"]):
                    print(f"Creating GeneralFolderAgent for {sub_folder_path}")
                    agent = GeneralFolderAgent(sub_folder_paths, current_folder)
                    return (agent, sub_folder, sub_folder_path)
            return None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_sub_folder = {
                executor.submit(create_folder_agent_recursive, sub_folder): sub_folder
                for sub_folder in sub_folders
            }
            initialization_errors = []

            for future, submitted_sub_folder in future_to_sub_folder.items():
                try:
                    result = future.result()
                    if result:
                        general_folder_agent, sub_folder, sub_folder_path = result  # Get the result from the future
                        with self.folder_agents_lock:
                            if sub_folder not in folder_agents:
                                folder_agents[sub_folder] = general_folder_agent
                                doc_agents.extend(general_folder_agent.doc_tools)
                                dict_summary_tool_total.update(general_folder_agent.dict_summary_tool)
                                dict_long_summary_tool_total.update(general_folder_agent.dict_long_summary_tool)
                                print(f"Updated dict_long_summary_tool_total with {len(general_folder_agent.dict_long_summary_tool)} items from {sub_folder}")
                                print(f"Current dict_long_summary_tool_total size: {len(dict_long_summary_tool_total)}")
                        self.create_folder_agents_recursive(sub_folder_path, folder_agents, doc_agents,
                                                            dict_summary_tool_total, dict_long_summary_tool_total)
                except Exception as e:
                    initialization_errors.append((submitted_sub_folder, e))
                    print(f"An error occurred while processing folder {submitted_sub_folder}: {e}")

            if initialization_errors and os.environ.get("MILONET_STRICT_INITIALIZATION") == "1":
                failed_folders = ", ".join(folder for folder, _ in initialization_errors)
                raise RuntimeError(f"Folder-agent initialization failed for: {failed_folders}")

        print(f"Creating GeneralFolderAgents for {current_folder} using multi-threads\n")



    def setup_tools(self):

        self.read_webpage_content_tool = FunctionTool.from_defaults(fn=read_webpage_content)
        self.do_google_search_tool = FunctionTool.from_defaults(fn=do_google_search)


        # Create a tool for each folder agent
        self.folder_tools = []

        def setup_folder_tool(item):
            print("Setting up folder tool")
            global global_id_map
            
            folder_name, agent = item

            # if len(folder_name) > 35:
            folder_hash = generate_folder_hash(folder_name, context="folder")
            tool_name = f"tool_{folder_hash}"

            with helper._file_read_lock:
                global_id_map[folder_name] = folder_hash
            # else:
            #     tool_name = f"tool_{folder_name}"
            #     with helper._file_read_lock:
            #         global_id_map[folder_name] = folder_name

            print(f"Processing folder ID: {folder_name} → {folder_hash}")
            # print(f"Processing folder ID: {folder_name} → {folder_hash if len(folder_name) > 35 else folder_name}")
        
            folder_description = f"Information about: {agent.folder_summary}"
            tool = FunctionTool.from_defaults(
                fn=agent.handle_query,
                name=tool_name,
                description=folder_description,
            )
            
            print(f"tool {tool_name} description: {len(folder_description)} characters")
            if len(folder_description) > 1024:
                print(f"Warning: folder tool {tool_name} description is too long!")
                with open(tool_long_tool_description_file, "a") as f:
                    f.write(f"{tool_name}: {folder_description}\n\n")
            return tool, folder_name
        

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(setup_folder_tool, item) for item in self.folder_agents.items()]
            for future in futures:
                try:
                    tool, folder_name = future.result()
                    self.folder_tools.append(tool)
                except Exception as e:
                    print(f"An error occurred while processing folder_tool {folder_name}: {e}")

        self.tools = [

                     ] + self.folder_tools

        self.expert_hierarchy = read_file_structure()
        print("Setup tools using multiple threads")

    def setup_advanced_agents(self):
        # self.memory = ChatMemoryBuffer.from_defaults(token_limit=2500)

        def chunk_list(lst, chunk_size=AGENT_CHUNK_SIZE):
            for i in range(0, len(lst), chunk_size):
                yield lst[i: i + chunk_size]

        subtools_chunk_size = AGENT_CHUNK_SIZE

        all_tools = self.tools
        # pdb.set_trace()
        for idx, subtools in enumerate(chunk_list(all_tools, subtools_chunk_size)):
            available_tool_names_for_subcoa = [t.metadata.name for t in subtools]
            # sub_coa_system_prompt = f"""\
            # You are an expert orchestrating agent named SubCoA_{idx}.
            # Your SOLE TASK is to answer the given query by meticulously using ***ONLY your Available Functions (or Available Parsed Functions), which are Folder Agent tools: {', '.join(available_tool_names_for_subcoa)}.***
            # **NEVER USE ANY OTHER TOOLS NOT LISTED IN YOUR AVAILABLE TOOLS.**
            # **NEVER CALL DOCUMENT TOOLS, ONLY CALL FOLDER AGENT TOOLS.**
            # These tools are Folder Agents, and you can call them to get information from documents they manage.
            
            # **CRITICAL PRE-PROCESSING RULES (APPLY THESE FIRST):**
            # 1.  **ULTRA-STRICT TOOL RELEVANCE ASSESSMENT:** Before even considering calling ANY tool, you MUST perform a deep semantic analysis of the user's query to identify its **absolute core subject matter and key entities** (e.g., specific people, events, concepts, locations, precise time periods).
            #     Then, for EACH of your available tools, you MUST meticulously compare its description against this **core subject matter and key entities**.
            #     A tool is considered relevant **ONLY IF its description's content demonstrates a STRONG, DIRECT, and UNAMBIGUOUS semantic overlap with the query's core subject and key entities.**
            #     *   A general thematic similarity is NOT enough. For example, if the query is about "Socialist conferences during World War I opposing Central Powers," a tool described as "Contains historical documents" or "Details 20th-century political figures" is LIKELY NOT SUFFICIENTLY RELEVANT unless its description *also explicitly mentions* terms like "World War I," "socialist conferences," "Central Powers," or highly specific related entities.
            #     *   If a tool's description is about 'biographies of American businessmen' and the query is about 'World War I military strategy,' that tool is CLEARLY NOT RELEVANT.
            # 2.  **PRIORITIZE "NO RELEVANT TOOLS FOUND":** If, after applying the ULTRA-STRICT TOOL RELEVANCE ASSESSMENT to ALL your available tools, you determine that **NONE of them meet the criteria for strong, direct, and unambiguous relevance** OR if no such relevant tools are present in any applicable filter list, you MUST immediately conclude that you cannot answer the query with the available tools.
            #     Your ONLY output in this case MUST be a clear statement like: "After careful assessment, no tools directly relevant to the core subject of '[briefly restate query's core subject]' were found among the available resources." or "No relevant, filtered tools are available to address the query regarding '[briefly restate query's core subject]'."
            #     **DO NOT attempt to select a 'best guess' or 'closest match' if no tool is truly, directly relevant. It is better to report no relevant tools than to use an inappropriate one.** DO NOT proceed to call any tools or synthesize an answer if this condition is met.

            # **YOUR ROLE WHEN CALLED AS A TOOL (Only if one or more HIGHLY relevant tools ARE identified, filtered, and called as per rules above):**
            # When an external agent calls you, and you have successfully called relevant internal Folder Agent tools, your *entire final output* MUST be a single, coherent, synthesized textual answer to the query you received.
            # It should NOT include your internal planning steps, execution traces (like "==EXECUTING PLAN==" or "[FUNC tool_name(...)]"), or any metadata unless explicitly part of the synthesized answer itself.

            # **CRITICAL INSTRUCTIONS FOR PROCESSING AND RESPONSE GENERATION (Only if HIGHLY relevant tools were identified, filtered, and called):**

            # 1.  **PLAN AND EXECUTE INTERNALLY**: You MUST formulate a plan to call your identified relevant Folder Agent tools AND THEN EXECUTE THAT PLAN by making the actual function calls. Ensure you are calling the tools you identified as highly relevant.

            # 2.  **ABSOLUTE FIDELITY TO TOOL OUTPUTS DURING SYNTHESIS**:
            #     *   After your Folder Agent tools return their textual responses, you MUST synthesize these responses.
            #     *   Your synthesized answer MUST be based *exclusively* on the literal text content returned by the Folder Agent tools you actually called.
            #     *   ***ULTRA-CRITICAL HANDLING OF NEGATIVE RESPONSES FROM CALLED TOOLS: If a Folder Agent tool, after being called by you, returns an EXACT textual response like "Information regarding... was not found..." or any similar explicit statement that information was not found, then for that specific tool, you can ONLY state that it did not find the information. If EVERY SINGLE Folder Agent tool you actually called for the query returns such explicit negative responses, then your ONLY permissible final output MUST be a SINGLE, CLEAR statement reflecting this. Example: "Based on the consulted expert sources, the requested information regarding '[core query subject]' was not found." DO NOT list any countries or invent any details in this scenario.***
            #     *   **DO NOT FABRICATE OR HALLUCINATE**: Under NO circumstances should you invent information, list countries, cite facts, or provide details that were not *explicitly and literally* present in the text returned by your Folder Agent tools. If the tools provide no relevant details for the query, your synthesized answer MUST state that clearly and factually.
            #     *   If some tools provide information and others return negative responses, synthesize accurately, citing only the tools that provided positive information, and clearly stating when other tools found nothing.

            # 3.  **FINAL OUTPUT AS A TOOL (Synthesized Answer Only - NO TRACES)**:
            #     *   Your *entire output*, when you are called as a tool by another agent, MUST consist ONLY of the final, synthesized textual answer derived from step 2.
            #     *   **ABSOLUTELY NO internal planning steps, NO "==EXECUTING PLAN==", NO "==EXECUTION COMPLETE==", and NO "[FUNC tool_name(...)]" traces are allowed in the final output string you return.**

            # 4.  **CORRECT CITATION WITHIN THE SYNTHESIZED ANSWER**:
            #     *   The tools you call are **Folder Agents**. They internally manage specific documents. If a Folder Agent's response indicates that information came from a *specific document ID* (e.g., "Document 0_1_2_X states..."), your synthesized answer should cite that **specific document ID** as the source.
            #     *   **NEVER cite the Folder Agent's name (e.g., 'tool_0_1_2_folder_id') as a document source.** You cite the *document* the Folder Agent refers to.
            #     *   If a Folder Agent tool states information is not found, DO NOT cite that tool or its non-existent documents as a source for *found* (fabricated) information.

            # 5.  **NO EXTERNAL KNOWLEDGE**: Base your synthesized answer ONLY on the explicit text returned by your Folder Agent tools.

            # **INTERNAL PRE-RESPONSE CHECKLIST (Mentally verify before outputting):**
            # *   Did I apply the ULTRA-STRICT TOOL RELEVANCE ASSESSMENT correctly? Did I only select Folder Agents whose descriptions showed STRONG, DIRECT, and UNAMBIGUOUS semantic overlap with the query's CORE subject and KEY ENTITIES?
            # *   If I concluded no tools were relevant, is my output ONLY the mandated 'no relevant tools found' statement?
            # *   Is every piece of information in my answer EXPLICITLY and LITERALLY present in the actual text returned by a CALLED Folder Agent?
            # *   If a Folder Agent returned "information not found," did I AVOID claiming it found something?
            # *   If ALL called Folder Agents returned "information not found," is my ENTIRE answer a simple statement of that fact?
            # *   Are my citations pointing to specific DOCUMENT IDs (if provided by the Folder Agent) and NOT to Folder Agent names themselves?

            # **To reiterate, your final deliverable when called as a tool is either (a) a statement that no relevant tools were found (if pre-processing rule 2 was met), OR (b) the *synthesized textual result* of your internal tool calls, strictly adhering to the information (or lack thereof) provided by those tools and the citation rules.**
            # """

            # Your task is to answer queries by meticulously using ***ONLY your ***AVAILABLE FOLDER AGENT TOOLS***: {', '.join(available_tool_names_for_subcoa)}.***    
            # 1.  **STRICT TOOL RELEVANCE ASSESSMENT (MANDATORY PRE-PROCESSING):**
            #     *   THE FOLDER AGENT TOOLS MUST IN YOUR **AVAILABLE FOLDER AGENT TOOLS**.
            #     *   ONLY USE OR CALL FOLDER AGENT TOOLS IN YOUR PLAN FROM YOUR AVAILABLE FOLDER AGENT TOOLS (your Available Parsed Functions).
            #     *   ***NEVER USE OR CALL DOCUMENT-LEVEL TOOLS DIRECTLY IN YOUR PLAN, ONLY CALL YOUR AVAILABLE FOLDER AGENT TOOLS.***
            #     *   Deeply analyze the user's query for its core subject and key entities.
            #     *   For EACH available Folder Agent tool from YOUR AVAILABLE FOLDER AGENT TOOLS, compare its EXACT description to the query's core.
            #     *   Each folder agent contains up to 3 documents in the folder description, you need to compare the query's core and key entities with each document in the folder description to determine if the folder tool is relevant.
            #     *   For EACH available Folder Agent tool, Make sure you use exact tool name, otherwise, it is a serious violation.
            #     *   You need to query all the relevant folder agents (up to {AGENT_CHUNK_SIZE}), do not miss any relevant folder agent.
            #     *   Do not force a connection between concepts in the tool description and the query if the link is ambiguous in the tool's exact description.
            #     *   Never repeatly identify the same tools, if you have already identified a folder agent tool as non-relevant, just ignore it.
            #     *   **PRIORITIZE "NO RELEVANT TOOLS":** If NO Folder Agent tools meet this strict relevance criteria, YOU MUST do NOT proceed the following steps and IMMEDIATELY RETURN: "After careful assessment, no tools directly relevant to '[core query subject]' were found."


# The most important rules are:
#              **TOOL RELEVECH ASSESMENT RULES (MANDATORY)**: To identify all relevant folder agent tools, you MUST consider any possible relationships between the query and the folder agent tools in the FILTER SET, even the query are not related to the folder agent tool:
#                 - Examples: 
#                 "who is the president of the United States", then you need to consider any folder in the filter list describing a person as a relevant folder.
#                 "What is A's sister 's weight", then you need to consider all folders in the filter list, which mentions a person's information (including any occupations, like sport player, singer) as a relevant folder, evenyou don't think they are the same person or relevant.
#                 - REMEMBER: You are only one of orchestrating agents, don't have the full picture, so always consider AS MORE AS POSSIBLE folders from the FILTER SET are relevant to avoid any possible missing information.
#               ***NEVER call or use document-level tools directly, ONLY call or use your AVAILABLE FOLDER AGENT TOOLS from the FILTER SET in your plan, otherwise, it is a serious violation.***
#               ***You MUST follow later INTERNAL EXECUTION rules, NEVER use any PLACEHOLDER TOOL NAMES or PLACEHOLDER QUERY STRINGS in your plan or tool calls (INCLUDING your internal reasoning/execution plan), always use the actual tool name from your *available folder agent tools.*** Otherwise, it is a serious violation.
            
            # """
            # *   When the Original User Query involves multiple entities, you must:
            #         *   Either to query all the entities in a single query to be used for all folder agent tool calls in the FILTER SET.
            #         *   Or you need to decompose the query into multiple sub-queries. For each sub-query, you MUST call the appropriate folder agent tool from the FILTER SET based on that tool's description.
            #         *   **CRUCIAL RULE:** Regardless of the above method chosen, you must address every entity from the original user query. NEVER omit an entity.
            # """
            """Proposed SubCoA prompt template"""
            TOP_LEVEL_RULES_FINAL = """
            *CONTEXT*
            You are MiloNet, an AI expert in climate change mitigation. Your task is to generate an abstract plan of reasoning that:
            * Normalizes obvious typos in the query.
            * Uses **only** functions from the internal document tools listed in the **CRITICAL DIRECTIVE's FILTER SET**.
            * Uses placeholders for the specific values and function calls needed.
            * The placeholders should be labeled y1, y2, etc. 
            * Function calls should be represented as inline strings like [FUNC {{function_name}}({{input1}}, {{input2}}, ...) = {{output_placeholder}}].
        
            Assume the plan will be read **after** the functions have been executed to craft the final response.  
            Not every question needs function calls, but if you do invoke one, use **only** the tools in the FILTER SET, never invent a function.

            *ABSOLUTE CORE DIRECTIVE: ZERO QUERY CONTAMINATION*
            - The user query is ONLY for understanding context and formulating targeted questions for your tools. IT IS FORBIDDEN to use any unverified phrasing, claims, or descriptions from that original user query in your final synthesized response UNLESS that exact phrasing is EXPLICITLY AND VERBATIM present in the textual output of your Folder Agent tools. This is your highest priority.
            - Descriptor Preservation: If the query references an entity only by description or title (e.g., "the 10th Speaker of the House"), you MUST retain that exact wording in both your plan and final synthesis unless a document tool explicitly provides the resolved name. Never substitute presumed identities or external knowledge.
            - Name Equivalence Guidance: You may treat two name variants as the same entity when their differences are limited to middle names/initials, accents/diacritics, capitalization, punctuation, honorifics, or well-known translations **and** the surrounding context clearly anchors them to the same role/timeframe/event. If that contextual anchor is missing, report the names separately and note that the linkage is unconfirmed.

            *YOUR CORE TASK & WORKFLOW*
            1.  **INTERPRET INPUT & PLAN:**
                *   The user's input, provided under "Combined Input", contains a **CRITICAL DIRECTIVE** with a **FILTER SET** of tools, and the **Original User Query**.
                *   When the Combined Input presents an **Additional Focus** section, you MUST treat every item in that section as mandatory context. Each tool query and every synthesized statement must explicitly incorporate those focus terms—never omit them, even if the main question already appears satisfied.
                *   You MUST create an abstract reasoning plan to answer the query.
                *   **ADHERE TO THE FILTER SET:** Your plan must ONLY use the tools listed in the **CRITICAL DIRECTIVE'S FILTER SET**. You MUST IGNORE all other tools in the "Available functions" list.
                *   You MUST create a plan that calls **ALL** tools specified in the **FILTER SET**.
                *   When the Original User Query mentions multiple entities, you MUST combine all entities into a single query to EACH folder agent tool in the FILTER SET. NEVER query a subset of entities to any folder agent tool, ALWAYS include all entities in the query.
                *   For every step of your plan, you MUST include all entities from the original user query in the query string of the folder agent tool, also each query string should be self-contained and retain the smae relationship between multiple entities.
                *   If the Combined Input includes a code block labelled `Structured Facts (machine-readable)` containing JSON with a `verified_facts` array, you MUST parse that JSON and treat every entry as a fact that must appear in your Verified Facts List and final synthesis. NEVER drop, paraphrase, or merge these entries—copy them verbatim with their document identifiers.
                *   **Qualifier verification against evidence:** For every extracted fact, if any temporal/scope limitation or other qualifier is not present verbatim in the accompanying Evidence snippet, you MUST replace the entire fact sentence with the exact wording from that Evidence. Do not keep any qualifier unless it is explicitly supported by the Evidence text.
                *   **Ambiguous scope handling:** If the source spans qualifiers across multiple sentences or the scope is ambiguous, copy the exact sentence from the evidence verbatim rather than attempting to split or reconstruct it. Never invent or merge qualifiers when the original wording is unclear; quote the evidence sentence as-is.
            2.  **MANDATORY FUNCTION CALL SYNTAX (CRITICAL for planning):**
                *   Your plan MUST be a sequence of single line and inline string function calls using a strict format.
                *   The format is: `[FUNC tool_folder_agent_name("actual query string") = yX]`, where `yX` is a placeholder like `y1`, `y2`, etc. Your actual query string MUST include all entities from the original user query.
                *   **KEY PRINCIPLES FOR SYNTAX:**
                    *   The `tool_folder_agent_name` MUST be the EXACT name from the "Available functions" list (and your FILTER SET). NEVER use a placeholder tool name.
                    *   The `"actual query string"` MUST be a self-contained and all entities (retaining the same relationship between entities) included question for the agent, enclosed in DOUBLE QUOTES.
                    *   **CRITICAL:** You MUST use the `= yX` placeholder at the end of each function call. This is how the system tracks results.
                *   **EXAMPLES of CORRECT and INCORRECT syntax:**
                    *   **GOOD:** `[FUNC tool_exact_name("actual query string") = y1]`
                    *   **BAD (causes critical failure):** `y1 = "some string", [FUNC tool_exact_name(y1) = y2]` -> FORBIDDEN. Do not assign query strings to variables first.
                    *   **BAD (causes critical failure):** `[FUNC tool_exact_name("actual query string")]` -> FORBIDDEN. You MUST include `= yX` at the end.
                    *   **BAD (causes critical failure):** `[FUNC tool_exact_name(actual query string) = y1]` -> FORBIDDEN. The query string MUST be in double quotes.
                    *   **BAD (causes critical failure):** `[FUNC tool_exact_name("actual query string"] = y1]` -> FORBIDDEN. The double quoted query string MUST be enclosed in parentheses `("actual query string")`.
                    *   **BAD (causes critical failure):** `[FUNC tool_exact_name("actual query string")] = y1` -> FORBIDDEN. `]` MUST be in the end of the function call to enclose the result placeholder NOT after the query string or parentheses.

            3.  **SYNTHESIZE ANSWER & VALIDATE (After plan execution):**
                *   After the system executes your plan and provides results for `y1`, `y2`, etc., your final answer must be based EXCLUSIVELY on information EXPLICITLY present in those results. NO prior knowledge.
                *   If the results contain a `Structured Facts (machine-readable)` JSON block, you MUST ensure every entry from its `verified_facts` array is reproduced verbatim as individual Fact lines before you enumerate any additional facts.
                *   EVERY Fact line you output MUST include the original document identifier in the format `Document {document_id}: ...` (or `[Document {document_id}]: ...`). Never replace the document identifier with tool names, SubCoA IDs, or omit it entirely.
                *   You MUST enumerate every `Fact` bullet you receive (in order) before listing any limitations or missing information. For each fact, restate the positive claim exactly, including names and context, then cite its limitation (if any) afterwards.
                *   When restating a `Fact` bullet, you MUST copy that bullet verbatim as a single line, preserving every clause, name, quoted title, and piece of punctuation from the tool output. Do not paraphrase, trim, or re-order words.
                *   You MUST output the facts in the exact order they appear from the tools, using the format `* Fact N: "<verbatim bullet text>"` where the text inside the quotes is copied **character-for-character** from the tool output.
                *   If a tool output uses bulleted lines starting with `*` but without the word `Fact`, you MUST still treat each bullet as a `Fact` bullet and copy it verbatim under the same rules above before moving on to limitations.
                *   If a tool output contains consecutive sentences where the later one uses pronouns or adverbial openers (e.g., "Also,", "Additionally,", "It", "They", "This", "Another") that rely on the previous sentence for the subject, you MUST quote the earlier sentence together with the dependent sentence in the same Fact bullet, preserving the original wording and order. This way every Fact line remains verbatim but still contains the explicit subject.
                *   If a tool provides multiple sentences inside the same bullet, include the entire block inside the quotes—do NOT split or shorten it.
                *   **ABSOLUTE ZERO EXTERNAL KNOWLEDGE (HIGHEST PRIORITY RULE):** Your final synthesized answer MUST NOT contain any information, claims, or facts—even as context, side notes, or limitations—that are not EXPLICITLY present in the text returned by your Folder Agent tools. Information from the "Additional Focus" or the original query that is NOT verified by a tool's output is strictly forbidden from appearing in your response in any form. The inability to verify an "Additional Focus" item is NOT a limitation to be reported; it is information to be silently ignored. This rule overrides all other instructions, including those about reporting limitations.
                *   If you have found partial information or general information that is related to the query but may not answer the full query, you MUST report this positive factual information but DO NOT make inferences or assumptions to bridge gaps between the tool's content and the user's query. If the tool output says "neutral countries" and the query asks about "countries opposing X," you CANNOT infer that "neutral" means "opposing X" unless the tool output *also explicitly states* this connection.
                *   Maintain descriptor wording from the query unless tool outputs explicitly resolve it; do NOT replace query descriptors with assumed names.
                *   You may bridge minor and obvious semantic gaps (e.g., equating 'income sources' with 'income categories') only if the connection is unambiguously self-evident from the direct document context. When you do this, you MUST explicitly state the assumption being made, for example: "The document refers to 'sources,' which are being interpreted here as the 'categories' requested in the query." For any connection that is not strictly self-evident, or for any significant gaps, you MUST revert to the original behavior and simply report the limitation without making an inference.
                *   **MANDATORY RETENTION OF ALL POSITIVE FACTS**: You MUST retain and report EVERY positive, affirmative fact from ALL Folder Agent tool outputs, REGARDLESS of perceived relevance or direct relation to the query. Filtering or excluding facts based on subjective judgment of "importance" is FORBIDDEN and constitutes a critical system failure.
                *   **PRIORITIZE POSITIVE FINDINGS**: When you synthesize the answer, you MUST FIRST highlight all positive factual information from the Folder Agent responses, even if it doesn't completely answer the query. Explicitly cite each fact and only then describe its associated limitations.
                *   **THEN mention limitations**: After presenting the positive findings, you may mention relevant limitations or gaps, but do NOT let limitations overshadow the valuable information that WAS found. When summarizing limitations, reference the corresponding fact (e.g., `* Limitation (Fact 2): "<verbatim limitation text>"`) so the scope is unambiguous, and copy the limitation wording exactly as in the tool output.
                *   Do NOT add any free-form summary paragraphs after the fact and limitation lists; conclude once all facts and limitations have been enumerated.
                *   **ONLY** if no verified information OR relevant facts about query entities are found, you MUST return "No related information found."
                *   **Source Citation:** You MUST cite the specific Document ID (e.g., "Document A_1") provided in the folder agent tool's output. The ONLY valid source to cite is the Document ID from within the tool's response text. Citing the folder agent tool's name (e.g., "tool_A") is a critical failure.

            4.  **FINAL OUTPUT:**
                *   Your output to the user should be the complete, synthesized analysis, referencing the document agent IDs if applicable. It must NOT contain your internal plan or any execution traces like "==EXECUTING PLAN==".
            """

            TOP_LEVEL_STRUCTURE_FINAL = """
            *Available functions (This list contains ALL possible tools, but you must obey the FILTER SET provided in the Combined Input):*
            ```python
            {functions}
            Combined Input (Directives + User Query):
            {question}
            ```
            Abstract plan of reasoning:
            """

            """End of the subCoA prompt template"""


            top_level_system_content_final = f"{TOP_LEVEL_RULES_FINAL}\n\n{TOP_LEVEL_STRUCTURE_FINAL}"
            top_level_system_message_final = ChatMessage(role=MessageRole.SYSTEM, content=top_level_system_content_final)
            TOP_LEVEL_REASONING_TEMPLATE_FINAL = ChatPromptTemplate(message_templates=[top_level_system_message_final])
            top_level_refine_prompt_final = """
            Generate a response to the question by using the previous abstract plan of reasoning.  
            Use that previous reasoning as context.

            RULES  
            0. **HARD FORMAT REQUIREMENT (NON-NEGOTIABLE)**  
                - Your final response MUST consist exclusively of the Fact and Limitation lines that appear in the "Previous reasoning" section. Copy each line character-for-character, including punctuation, quotation marks, spacing, document identifiers, and wording.  
                - You MUST NOT generate any new wording, summaries, explanations, or paraphrases. If a line in the previous reasoning reads `* Fact 1: "Document X: ..."` then your output MUST contain that exact text.  
                - If you are unable to reproduce the exact text for ANY reason, respond with `No related information found.` instead of attempting to paraphrase or fill in details.  
                - If the "Previous reasoning" includes a `Structured Facts (machine-readable)` JSON block containing a `verified_facts` array, you MUST parse it and reproduce every entry verbatim as Fact lines, prefixing each with its original document identifier (e.g., `Document 3dfd5c37_2: ...`).
                - If the "Previous reasoning" includes a "## Verified Facts List" section with bullet points (e.g., "- [Document X]: fact text"), you MUST treat EACH bullet point as a mandatory fact to reproduce verbatim in your output.
            *  If the "Previous reasoning" section includes a `Structured Facts (machine-readable)` JSON block with a `verified_facts` array, you MUST parse it and reproduce every entry as individual Fact lines before listing any other content. NEVER omit or merge these entries; copy them verbatim with their document identifiers.
            *  If the "Previous reasoning" section includes a "## Verified Facts List" section, you MUST reproduce EVERY bullet point from that list verbatim in your output. Do NOT selectively filter or omit facts based on perceived relevance - include ALL facts from ALL documents listed.
            1. **CRITICAL: PRESERVE ALL CONCRETE DETAILS (The PRIME DIRECTIVE OF YOUR RESPONSE)** 
                - Your output MUST PRESERVE ALL named entities, specific numbers, data, dates, names, measurements, percentages, and factual details found in your folder agent tool outputs. NEVER summarize away or omit concrete information. For example, if the original text says "other events (A, B, C)", your response MUST include the full list, DO NOT just report "other events". KEEP More details are ALWAYS better than less details.
                - **NO PARAPHRASING OF ENTITY NAMES:** When the folder agent output provides a specific entity name or description, you MUST copy it EXACTLY. Do NOT substitute with generic terms or simplified versions. Preserve the complete description including any aliases, clarifications, or additional details. For example, if the output says "Entity X (also known as Entity Y)", keep the entire phrase; do not shorten to just "Entity X" or replace with other generic terms.
                - If the folder agent tool output contains any personal names, you MUST use the full name as it appears in the source text, including all middle names and initials, every time the person is mentioned. Do not shorten names.
                - Enumerate ALL Positive Facts Verbatim First - Before any synthesis or conclusion, list EVERY positive fact from the tool outputs verbatim. Include all details, even if indirectly related.
                - Even when two facts appear redundant or closely related, you MUST still reproduce each tool-provided fact as its own entry, copying the wording exactly as given (including leading articles like "The"). Never merge, condense, or drop a fact because it seems repetitive.
            2. **STRICT FACT SEPARATION**: When facts from different tools or sentences describe similar but not identical scopes (e.g., "Allied forces of Romania and France" vs. "Allies or Entente Powers"), report them SEPARATELY. Do NOT merge or imply equivalence unless the source explicitly states they are the same. Use phrasing like "One fact states [A]; separately, another states [B]".
                - This prevents misleading synthesis.
            3. **PRESERVE DOCUMENT IDENTIFIERS**: Ensure facts include their document source identifiers when available.
            4. **QUOTE EXACT DATA** - When reporting numerical or factual information, preserve the exact values and context from your previous reasoning.
            5. If the retrieved text in your previous reasoning contain contradictory information, you MUST report the contradiction.  
            6. NEVER miss any positive (i.e., relevant and supportive) information related to the query.  
            7. If you cannot determine which parts of the retrieved text in your previous reasoning can answer the query, you MUST quote **all** of those parts. For attribute-related queries (e.g., involving entities and their details like songs, roles, or achievements), always quote all details verbatim. You MAY note explicit chains only if entity names match exactly and context directly supports without any assumption (e.g., "Noted explicit chain: entity [name] from Fact X to attribute in Fact Y via exact name match in quotes '[quote1]' and '[quote2]'"). Do not invent links or use notes for inference; if any doubt, report separately and state "No confirmed chain—facts reported independently".
            8. When you report retrieved texts, DO NOT make inferences or assumptions to bridge gaps between the tool's content and the user's query, just report exact texts and state its limitations.
            9. **ABSOLUTELY FORBIDDEN: DO NOT merge time qualifiers from different sentences.** If a folder agent output contains multiple sentences and one sentence has a time qualifier (e.g., "as of 2017", "in 1990") while another does not, you MUST preserve them as separate facts. NEVER attach a time qualifier from one sentence to a different sentence. **CRITICAL: Every qualifier in your fact MUST exactly match the qualifier in the source quote. If the source does not have a qualifier, your fact MUST NOT have it either.**
                - **Example of VIOLATION:**
                    *   Folder agent output: "Entity A has attribute X. Entity B has attribute Y as of 2020."
                    *   WRONG: `* Fact 1: "Entity A has attribute X as of 2020."`
                    *   CORRECT: `* Fact 1: "Entity A has attribute X."` and `* Fact 2: "Entity B has attribute Y as of 2020."`
            10. Descriptor Integrity: If the query uses a descriptive label instead of a proper noun, you MUST keep that exact label in your response unless tools explicitly provide the resolved name. Do not substitute presumed identities.
            11. **Source Citation:** You MUST cite the specific Document ID (e.g., "Document A_1") provided in the folder agent tool's output. The ONLY valid source to cite is the Document ID from within the tool's response text. Citing the folder agent tool's name (e.g., "tool_A") is a critical failure.
            Example 1 (Simple):
            -----------
            Question:  
            Sally has 3 apples and buys 2 more. Then magically, a wizard casts a spell that multiplies  
            the number of apples by 3. How many apples does Sally have now?

            Previous reasoning:  
            After buying the apples, Sally has [FUNC add(3, 2) = 5] apples.  
            Then, the wizard casts a spell to multiply the number of apples by 3,  
            resulting in [FUNC multiply(5, 3) = 15] apples.

            Response:  
            After the wizard casts the spell, Sally has 15 apples.

            Example 2 (Complex - Handling Partial/Related Information):
            -----------
            Question:
            Was scientist Marie Curie or politician Winston Churchill a fellow of the "Royal Society"?

            Previous reasoning:
            [FUNC tool_A("Was Marie Curie or Winston Churchill a fellow of the Royal Society?") = "Document A_1 states that Winston Churchill was elected a Fellow of the Royal Society (FRS) in 1941. It does not mention Marie Curie."]
            [FUNC tool_B("Was Marie Curie or Winston Churchill a fellow of the Royal Society?") = "Document B_2 focuses on Marie Curie's Nobel Prizes in Physics (1903) and Chemistry (1911) and does not contain information about her fellowship in the Royal Society. The document does not mention Winston Churchill."]

            Response:
            Based on the information retrieved:
            * Fact 1: "Document A_1 states that Winston Churchill was elected a Fellow of the Royal Society (FRS) in 1941. It does not mention Marie Curie."
            * Fact 2: "Document B_2 focuses on Marie Curie's Nobel Prizes in Physics (1903) and Chemistry (1911) and does not contain information about her fellowship in the Royal Society. The document does not mention Winston Churchill."

            Your Turn
            -----------
            Question:  
            {question}

            Previous reasoning:  
            {prev_reasoning}

            Response:
            
            """
            # 'Generate a response to a question by using a previous abstract plan of reasoning. Use the previous reasoning as context to write a response to the question. If responses contains contradictory informarion, you MUST report this contradictory information. NEVER miss any positive information as long as it is related to the query.\n If the retrieved texts from one of your document tools answer only part of a multi-part question, you MUST report the texts for the solved part as long as it is related to the query. but DO NOT make inferences or assumptions to bridge gaps between the tool\'s content and the user\'s query. Just report \n\nExample:\n-----------\nQuestion:\nSally has 3 apples and buys 2 more. Then magically, a wizard casts a spell that multiplies the number of apples by 3. How many apples does Sally have now?\n\nPrevious reasoning:\nAfter buying the apples, Sally has [FUNC add(3, 2) = 5] apples. Then, the wizard casts a spell to multiply the number of apples by 3, resulting in [FUNC multiply(5, 3) = 15] apples.\n\nResponse:\nAfter the wizard casts the spell, Sally has 15 apples.\n\nYour Turn:\n-----------\nQuestion:\n{question}\n\nPrevious reasoning:\n{prev_reasoning}\n\nResponse:\n'
            # top_level_refine_message_final = ChatMessage(role=MessageRole.USER, content=top_level_refine_prompt_final)
            TOP_LEVEL_REFINE_PROMPT_TEMPLATE_FINAL = PromptTemplate(top_level_refine_prompt_final)




            """Original SubCoA system prompt suspected not using"""
            # sub_coa_system_prompt = f"""\
            # You are one of expert orchestrating agents, SubCoA_{idx}. Your goai is to gather all verified information from the later PROVIDED folder agent FILTER SET for later top-level synthesis.
            # Your task is to answer queries by meticulously using ***ONLY your ***AVAILABLE FOLDER AGENT TOOLS IN THE LATER PROVIDED FILTER SET when a query is received.***


            # **ABSOLUTE CORE DIRECTIVE: ZERO QUERY CONTAMINATION:**
            # The originaluser's query (see in your context) and any query string inside a function call (e.g., [FUNC tool_exact_name("query string") = var1]) is ONLY for understanding context and selecting tools. IT IS FORBIDDEN to use any unverified phrasing, claims, or descriptions from that original user query in your final synthesized response UNLESS that exact phrasing, claim, or description is EXPLICITLY AND VERBATIM present in the textual output of your Folder Agent tools. Your response MUST reflect ONLY what is EXPLICITLY AND VERBATIM stated by your Folder Agent tools. This is your highest priority during response synthesis. Violation of this is a critical failure.
            # This Zero Query Contamination applies to every part of your response including the conclusion. NEVER assume anything not retrived from your tools.


            # **CORE OPERATING PROTOCOL:**
            # 1.  **TOOLS SELECTION:**
            #     *   THE FOLDER AGENT TOOLS MUST IN YOUR **AVAILABLE FOLDER AGENT TOOLS**.
            #     *   You will be provided a folder agent FILTER set, ONLY USE OR CALL FOLDER AGENT TOOLS IN YOUR PLAN FROM THE FILTER SET.
            #     *   ***NEVER USE OR CALL DOCUMENT-LEVEL TOOLS DIRECTLY IN YOUR PLAN, ONLY CALL YOUR AVAILABLE FOLDER AGENT TOOLS in the FILTER SET.***
            #     *   You MUST CALL ALL the folder agent tools in the FILTER SET one by one.
            # 2.  **INTERNAL EXECUTION (If any relevant Folder Agents in the FILTER SET):**
            #     *   **CRITICAL: NEVER use PLACEHOLDER tool names in your plan or tool calls, always use the actual tool name from your *available folder agent tools**.
            #     *   **CRITICAL: MUST and ALWAYS use the actual query string with the double quotes in your plan or tool calls, NEVER USE ANYTHING ELSE such as PLACEHOLDERS OR VARIABLES OR ellipsis(...) in query strings.
            #             *   Good example: [FUNC tool_exact_name("actual query string") = var1].
            #             *   BAD examples (critical failure): 
            #                     - uses placeholders for tool names: [FUNC tool_xyz("actual query string") = var1],
            #                     - uses placeholders for query strings: [FUNC tool_exact_name(y2) = var1],
            #                     - uses variables in query strings: [FUNC tool_exact_name("...{{var2}}...") = var1],
            #                     - use ellipsis(...) in query strings: [FUNC tool_exact_name(...) = var1],
            #                     - MISSING double quotes around the query string: [FUNC tool_exact_name(actual query string) = var1],
            #                     - put FUNC inside the quotation marks: "...[FUNC tool_exact_name("actual query string") = var1]...", `...[FUNC tool_exact_name("actual query string") = var1]...`, '...[FUNC tool_exact_name("actual query string") = var1]...' ```...[FUNC tool_exact_name("actual query string") = var1]...```, etc.
            #     *   if you have any BAD examples in your plan, your MUST redo your plan again.
            #     *   Formulate a plan and EXECUTE calls to ALL Folder Agent tools in your FILTER SET, never miss any.
            #     *   When you execute plan, you MUST execute each tool using the above requested format in good example.
            #     *   Your synthesized answer MUST be based EXCLUSIVELY on the literal text returned by these Folder Agents.
            #     *   **Handling Negative Responses:** If ALL called Folder Agents explicitly state "information not found," your final output MUST be a single, clear statement reflecting this (e.g., "Based on consulted sources, information on '[core query subject]' was not found."). Do NOT invent details.
            #     *   NO fabrication or hallucination.
            # 3.  **RESPONSE SYNTHESIS - STRICT ADHERENCE TO ANTI-CONTAMINATION:**
            #     *   The user's original query, any query strings in a function call AND ANY PREVIOUS ASSISTANT RESPONSES IN THE INPUT HISTORY guide your understanding but **ARE NOT SOURCES OF TRUTH OR VERIFIED INFORMATION for your final synthesized response in this turn.**
            #     *   Your final synthesized response MUST NOT use any phrasing, terminology, claims, entities, or relationships from the user's original query OR any query string inside a function call OR FROM PREVIOUS ASSISTANT MESSAGES IN THE INPUT HISTOR, UNLESS that exact phrasing, terminology, claim, entity, or relationship is also **EXPLICITLY AND VERBATIM present in the textual content returned by your AVAILABLE FOLDER AGENT TOOLS that you have called in this current reasoning process.**
            #     *   **Regarding Instructions to Use "Previous Reasoning":** When instructed to generate a response based on "previous reasoning" or a provided "Question", you must still ensure your FINAL synthesized answer to the user adheres to the ZERO QUERY CONTAMINATION directive. The phrasing of the "Question" within such instructions should NOT be copied into your final response unless verified by your tool outputs in THIS turn. Synthesize your answer based on the *results* of the execution plan (the tool outputs), not based on the phrasing of the input "Question".
            #     *   **VERIFY THIS:** Before finalizing your response, compare your drafted synthesized answer against the *literal text* of the Folder Agent outputs. If any part of your synthesized answer uses terms from the original user query that are NOT found verbatim in the Folder Agent outputs, you MUST remove or rephrase that part to reflect *only* what the Folder Agents stated.
            #     *   **CRITICAL - FORWARDING PARTIAL INFORMATION:** If a Folder Agent tool you called returns specific, verified facts relevant to *any part* of the query, even if that Folder Agent states it cannot answer the *entire* query given to it, YOU MUST INCLUDE these verified facts in your synthesized output. Your role is to gather all such factual nuggets for later top-level synthesis. Do not discard a fact just because the Folder Agent tool didn't have the full picture.
            #         However, you MUST return its limitations (e.g., "However, the information is not found in the folder") from the tool's output in your synthesized output, otherwise, it is a critical failure.
            #     *   **CRITICAL - You need to recognise the same entity in your tools' outputs and the query: **
            #         - If the query is about someone by the first and last name, and the tool's output is about the same person by the first name, or first, middle and last name, you need to include the tool's output in your synthesized output.
            # 4.  **FINAL OUTPUT (When you are called as a tool):**
            #     *   Your ENTIRE output MUST be the synthesized textual answer ONLY.
            #     *   You MUST FULLY and CORRECTLY understand the outputs from your tools and then synthesized textual answer SOLELY based on your tools' outputs with the STRICT ADHERENCE TO ANTI-CONTAMINATION.
            #     *   ABSOLUTELY NO internal planning steps or execution traces (e.g., "==EXECUTING PLAN==" or "[FUNC tool_name(...)]").
            #     *   If no verified and relevant information is found in any your available Folder Agent output, you MUST return "No related information found." Otherwise, report what was found.
            #     *   DO NOT DROP your limitations from the tool's output in your synthesized output, otherwise, it is a critical failure.
            # 5.  **CITATION:**
            #     *   Your answer should cite SPECIFIC DOCUMENT IDs (e.g., "Document 0_1_2_X states...") if provided in the Folder Agent's response.
            #     *   NEVER cite the Folder Agent's name (e.g., 'tool_0_1_2_folder_id') as a document source. Cite the document.
            #     *   If no verified and relevant information is found in any your available Folder Agent output, no citation needed.
            # 6.  **NO EXTERNAL KNOWLEDGE.**

            # **Pre-Output Mental Checklist:**
            # *   DO I follow the Zero Query Contamination rule for the response?
            # *   Is every fact in my answer EXPLICITLY from a CALLED Folder Agent's output?
            # *   **Have I *avoided* using any terms from any query string inside a function call or the original user query that were NOT in the Folder Agent outputs?**
            # *   If "not found" was the consensus, is that my sole output?
            # *   Are citations to DOCUMENT IDs, not Folder Agents?
            # *   If the conclusion appeared in the Folder Agent outputs too?
            # *   Have I avoid adding any new information in conclusion that is not in the Folder Agent outputs?
            # """
            """END of original subCoA system prompt"""
            # sub_coa_system_prompt = """Testing system prompt, only return "test"."""
            worker = CoAAgentWorker.from_tools(
                tools=subtools,
                llm=self.llm,
                verbose=True,
                reasoning_prompt_template = TOP_LEVEL_REASONING_TEMPLATE_FINAL, # update the template 
                refine_reasoning_prompt_template = TOP_LEVEL_REFINE_PROMPT_TEMPLATE_FINAL,
                output_parser=SafeChainOfAbstractionParser(verbose=True, auto_fix_duplicates=True),
                # system_prompt=sub_coa_system_prompt,
            )

            sub_agent = worker.as_agent(memory=None)
            sub_agent.name = f"SubCoA_{idx}"
            print("add " + sub_agent.name)
            self.coa_agents.append(sub_agent)

        self.react_agent = ReActAgent.from_tools(
            # self.tools,
            # max_iterations=22,
            # memory=self.chat_memory,
            llm=self.llm,
            verbose=True,
        )

        self.agent_worker = LATSAgentWorker.from_tools(
            tools=self.tools,
            # memory=self.chat_memory,
            llm=self.llm,
            num_expansions=3,  # how many new branches to make from each node
            max_rollouts=-1,  # how far down each branch to go; using -1 for unlimited rollouts
            verbose=True,
        )
        self.ToT_agent = AgentRunner(self.agent_worker)



        callback_manager = self.llm.callback_manager
        self.agent_worker = LLMCompilerAgentWorker.from_tools(
            tools=self.tools,
            # memory=self.chat_memory,
            llm=self.llm,
            verbose=True,
            callback_manager=callback_manager
        )
        self.LLMC_agent = AgentRunner(self.agent_worker, callback_manager=callback_manager)



    # def save_all_shortened_names(self):
    #     """Collect and save all shortened name mappings from all folder agents."""
    #     global global_id_map, global_id_map_lock
    #     all_mappings = {}

    #     try:
    #         if os.path.exists("global_id_map.json"):
    #             with open("global_id_map.json", 'r') as f:
    #                 all_mappings = json.load(f)
    #     except Exception as e:
    #         print(f"Error loading existing global map: {e}")



    #     with global_id_map_lock:
    #         all_mappings.update(global_id_map)
    #         global_id_map.update(all_mappings)
    #         final_mappings = global_id_map.copy()
        
    #     mappings_path = os.path.join(os.path.dirname(self.processed_documents_folder), "all_shortened_names.json")
    #     global_path = "global_id_map.json"
        
    #     try:
    #         with open(mappings_path, 'w') as f:
    #             json.dump(final_mappings, f, indent=2)
            
    #         print("DEBUG: Final mappings to be saved to global_id_map.json:")
    #         for k, v in final_mappings.items():
    #             print(f"  '{k}' -> '{v}'")
    #         with open(global_path, 'w', encoding="utf-8") as f:
    #             json.dump(final_mappings, f, indent=2)
    #         print(f"Saved {len(final_mappings)} total shortened name mappings to {mappings_path} and {global_path}")
    #         helper.update_tool_list()
    #     except Exception as e:
    #         print(f"Error saving all shortened names: {e}")


multi_folder_agent = None
agent_initilised = False


def build_gradio_interface():
    file_name_map = None

    time_start = time.time()


    # print("Pre-warming shared OllamaEmbedding model...")
    # try:
    #     # make sure get_shared_ollama_embedding is imported
    #     # from helper import get_shared_ollama_embedding (if main is in a different file, or ensure helper is imported)
    #     helper.get_shared_ollama_embedding(model_name="nomic-embed-text") # use your default model name
    #     print("Shared OllamaEmbedding model pre-warmed successfully.")
    # except Exception as e:
    #     print(f"CRITICAL ERROR during OllamaEmbedding pre-warming: {e}")
    #     print("Application might not function correctly with embeddings.")


    mode_loader = ModeLoader('./Modes')
    print("Start loading modes")
    modes_dict = mode_loader.load_modes()
    print("Finished loading modes")

    initialization_event = threading.Event()

    def initilise():
        nonlocal file_name_map
        global multi_folder_agent, agent_initilised, global_id_map

        try:
            if os.path.exists("global_id_map.json"):
                with helper._file_read_lock:
                    with open("global_id_map.json", 'r', encoding="utf-8") as f:
                        global_id_map.update(json.load(f))
                print(f"Loaded {len(global_id_map)} mappings from global_id_map.json")
        except Exception as e:
            print(f"Error loading global ID map: {e}")
        

        sanitize_paths_in_directory(original_documents_folder)

        # Generate the file/folder structure with unique indices
        file_paths_with_positions = list_files_recursive_with_numbering(original_documents_folder, max_depth=None)
        index_map = create_indexed_path_map(file_paths_with_positions)

        print("\nIndex map:")
        for path, index in index_map.items():
            print(f"{path} -> {index}")

        with open("file_structure.txt", "w", encoding='utf-8') as f:
            for path, file in file_paths_with_positions:
                f.write(f"{path} {file}\n")

        processed_files_map_path = os.path.join(processed_documents_folder, "processed_files_map.json")
        processed_files_map = load_processed_files_map(processed_files_map_path)
        mirror_folder_structure_with_indexing(original_documents_folder, processed_documents_folder, index_map,
                                              processed_files_map, processed_files_map_path)
        
        
        multi_folder_agent = MultiFolderAgent(processed_documents_folder)
        print("MultiFolderAgent created.")

        # multi_folder_agent.save_all_shortened_names()
        # helper.debug_id_mappings()
        
        create_tool_access_file(file_paths_with_positions, output_file="tool_access_info.txt",
                                all_doc_tools_file="all_doc_tool.txt")
        

        # helper.update_tool_list()
        # print(f"Initial tool list created with {len(helper.tool_list)} tools")
        # helper.ensure_consistent_tool_list_format()
        
        
        initialization_event.set()
        file_name_map = index_map
        print("Initialization complete.")
        return

    threading.Thread(target=initilise).start()
    initialization_event.wait()
    
    time_end = time.time()
    total_time_estabilished = (time_end - time_start) / 60
    print('total time of initilisation ' + str(total_time_estabilished) + ' minutes.')
    print("Starting UI initialization...")
    gradio_interface = GradioInterface(multi_folder_agent, modes_dict, file_name_map)
    print("UI objects created.")
    return gradio_interface


def main():
    gradio_interface = build_gradio_interface()
    print("Starting server...")
    gradio_interface.run()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from arize.otel import register

if __name__ == "__main__":
    # Optional observability must be configured with environment variables.
    main()

# def check_tool_mappings():
#     """Utility function to analyze the shortened name mappings file."""
#     mappings_path = os.path.join(os.path.dirname(processed_documents_folder), "all_shortened_names.json")
    
#     if not os.path.exists(mappings_path):
#         print(f"No shortened name mappings file found at {mappings_path}")
#         return
    
#     try:
#         with open(mappings_path, 'r') as f:
#             mappings = json.load(f)
        
#         total_mappings = len(mappings)
#         print(f"Total shortened name mappings: {total_mappings}")
        
#         # Sample of mappings
#         print("\nSample of mappings:")
#         sample = list(mappings.items())[:10]  # Show first 10
#         for key, value in sample:
#             print(f"  {key} -> {value}")
#     except Exception as e:
#         print(f"Error analyzing shortened name mappings: {e}")
