import io, sys, os, threading, json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# apply enhanced OpenAI model support patch
try:
    from enhanced_model_patch import comprehensive_patch
    comprehensive_patch()
    print("🚀 helper.py use enhanced model patch")
except ImportError:
    try:
        from openai_model_patch import patch_openai_model_support
        patch_openai_model_support()
        print("🔄 helper.py use basic model patch")
    except ImportError:
        print("⚠️  helper.py cannot find model patch file")
except Exception as e:
    print(f"⚠️  helper.py model patch failed: {e}")

from exceptiongroup import catch
# import core_prompts
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import uuid
from collections import OrderedDict
from llama_index.core import VectorStoreIndex
import yake
# from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from datetime import datetime
import json
import threading
from queue import Queue
import re
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

from typing import List, Dict, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, NodeRelationship, MetadataMode
from pydantic import PrivateAttr
        
class TraceNodeTap(BaseNodePostprocessor):
    prefix: str = "Got ctx: "
    def postprocess_nodes(self, nodes, query_str=None, **kwargs):
        self._dump(nodes)
        return nodes

    def _postprocess_nodes(self, nodes, **kwargs):
        self._dump(nodes)
        return nodes

    def _dump(self, nodes):
        for i, n in enumerate(nodes):
            node_obj = getattr(n, "node", n)

            # 1) get text field first
            txt = getattr(node_obj, "text", None)

            # 2) if not, use pure TEXT mode
            if not txt:
                try:
                    txt = node_obj.get_content(metadata_mode=MetadataMode.TEXT)
                except Exception:
                    txt = str(node_obj)

            # 3) simple fallback: remove the context_summary garbage string
            if "context_summary:" in txt:
                # Remove the metadata label before logging the node text.
                txt = txt.split("context_summary:", 1)[-1].strip()

            safe_txt = txt.replace("\n", "\\n ")
            print(f"{self.prefix}[{i}] {safe_txt}")


_filename_mapping_path = Path(__file__).resolve().with_name("hashing_name_mapping.json")
if _filename_mapping_path.is_file():
    with _filename_mapping_path.open("r", encoding="utf-8") as f:
        filename_map = json.load(f)
else:
    filename_map = {}

import logging
raw_responses = []
last_plan_text = None
_tool_list_lock = threading.Lock()
# reset_raw_responses = False
# doc_summary_dict = {}

# Mapping from folder agent tool name -> list of (canonical_lower, original_fact)
folder_agent_verified_facts: dict[str, list[tuple[str, str]]] = {}

class DisplayNWrite:
    _KEY_RULES = (
        # "Got output: ",
        "Got ctx: ",               
        "THE CURRENT USER INPUT: ",
        "Response: ",
    )

    # _NEG_PHRASES = [
    #     "empty response",
    #     "context does not",
    #     "information does not",
    #     "does not contain",
    #     "does not include",
    #     "does not mention",
    #     "no information",
    # ]

    def __init__(self, console, local_output_logger=None, file=None):
        self.console = console
        self.output_logger = local_output_logger  # For GUI display
        self.file = file  # File output, only used when enabled
        self.file_enabled = False  # Control file output range

        self.trace_file = None
        self.trace_file_enabled = False


        self.log_enabled = False
        self.str_output = ""
        self.print_buffer = ""
        self._line_buffer = ""
        self.lock = threading.Lock()

        self._user_input_buf = []
        self._ctx_buf = []
        self._response_buf = []
        self._capturing = None  # "user_input" / "response" / None


    def set_file(self, file):
        self.file = file

    def set_trace_file(self, trace_file):
        with self.lock:
            self.trace_file = trace_file

    def write(self, message):
        """Handle output based on settings."""
        with self.lock:
            self.console.write(message)
            self.console.flush()

            # Log to output_logger if enabled
            if self.log_enabled and self.output_logger:
                    self.output_logger.write(message)
            else:
                if "🔄 Still working on it... 🧠⏳" in message or "it will take about one min to stop" in message:
                    self.print_buffer += message

            if self.file_enabled and self.file:
                if "Error" not in message and "Exception" not in message:
                    self.file.write(message)
                    self.file.flush()
                    self.str_output += message

            if self.trace_file_enabled and self.trace_file:
                self._process_trace(message)

    def _process_trace(self, chunk: str):
        self._line_buffer += chunk
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_line(line.strip())

    
    def _handle_line(self, line: str):
        # 1. new user input, flush the previous round
        if ("THE CURRENT USER INPUT: ") in line:
            self._flush_trace_round()  # flush the previous round
            self._user_input_buf = [line]
            self._ctx_buf = []
            self._response_buf = []
            self._capturing = None
            return
            

        # 2. Got ctx: each line is stored
        elif "Got ctx:" in line:
            self._ctx_buf.append(line)
            self._capturing = None
            return 


        # 3. Response: new response block, overwrite the old one
        elif "Direct Answer:" in line:
            self._response_buf = [line]
            self._capturing = "final_response"
            return

        elif "Processing question:" in line:
            self._capturing = None  # stop any capture
            return                  # and ignore this line itself
      
        # 4. capture multi-line response
        elif self._capturing == "final_response":
            self._response_buf.append(line)
            return


    # def _filter_positive_sentences(self, text: str) -> str:
    #     txt_lower = text.lower()
    #     if any(neg in txt_lower for neg in self._NEG_PHRASES):
    #         return ""          
    #     return text.strip()

    def _safe_trace_write(self, text: str):
        try:
            self.trace_file.write(text)
            self.trace_file.flush()
        except Exception as e:
            self.console.write(f"[TRACE-LOG ERROR] {e}\n")

    def _flush_trace_round(self):

        if  self._user_input_buf:
            self._safe_trace_write("\n".join(self._user_input_buf).strip() + "\n\n")

        if self._ctx_buf:
            self._safe_trace_write("\n".join(self._ctx_buf) + "\n\n")

        if self._response_buf:
            self._safe_trace_write("\n".join(self._response_buf).strip() + "\n\n")

        # after all content is successfully written, clear all buffers at once, prepare for the next round.
        self._user_input_buf = []
        self._ctx_buf = []
        self._response_buf = []
        self._capturing = None

    def flush(self):
        # Flush console, OutputLogger, and file (if enabled)
        with self.lock:
            self._flush_trace_round()
            self.console.flush()
            if self.output_logger:
                self.output_logger.flush()
            if self.file_enabled and self.file:
                self.file.flush()
            if self.trace_file and self.trace_file_enabled:
                self.trace_file.flush()

    # Enable file output for selected code range
    def enable_file_output(self):
        with self.lock:
            self.file_enabled = True

    def enable_trace_log(self):
        with self.lock:
            self.trace_file_enabled = True
    # Disable file output to stop writing to the file
    def disable_file_output(self):
        with self.lock:
            self.file_enabled = False

    def disable_trace_log(self):
        with self.lock:
            self.trace_file_enabled = False

    def isatty(self):
        return self.console.isatty() if hasattr(self.console, 'isatty') else False

    def get_print_output(self):
        """Get only `print()` outputs for the UI."""
        with self.lock:
            return self.print_buffer

    def get_general_output(self):
        with self.lock:
            return self.output_logger.get_output() if self.output_logger else ""

    def add_to_print_buffer(self, message):
        with self.lock:
            self.print_buffer += message


class LimitedDict(OrderedDict):
    def __init__(self, max_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

        items = list(self.items())
        self.clear()

        for key, value in items:
            self[key] = value

    def __setitem__(self, key, value):
        # If the dictionary already contains 5 items, remove the oldest one
        if len(self) >= self.max_size:
            # Pop the first item (oldest) in the dictionary
            self.popitem(last=False)
        # Add the new key-value pair
        super().__setitem__(key, value)

    @classmethod
    def from_dict(cls, regular_dict, max_size=5):
        limited_dict = cls(max_size=max_size)
        for key, value in regular_dict.items():
            limited_dict[key] = value

        return limited_dict


def read_tool_access_info() -> str:
    """This tool obtains a list of the document tools, which documents each tool has access to, and the translation between the original document name and the unique reference number
    assigned to that file which is used by the tools."""
    with open("tool_access_info.txt", "r", encoding='utf-8') as f:
        text = f.read()
        #print(">>>>>>>>>>>>>>>>>>>>>>> file content: ", text)
        return text

def read_all_doc_tool_info() -> str:
    """This tool obtains a list of the document tools, which documents each tool has access to, and the translation between the original document name and the unique reference number
    assigned to that file which is used by the tools."""
    with open("all_doc_tool.txt", "r", encoding='utf-8') as f:
        text = f.read()
        #print(">>>>>>>>>>>>>>>>>>>>>>> file content: ", text)
        return text

class OutputLogger(io.StringIO):
    def __init__(self):
        super().__init__()
        self.output = []
        self.lock = threading.Lock()

    def write(self, s):
        try:
            with self.lock:
                self.output.append(s)
                super().write(s)
                # with stdout_lock:
                #     original_stdout.write(s)
                #     original_stdout.flush()
        except Exception as e:
            original_stdout.write(f"\nLogging Error: {e}\n")

    def flush(self):
        # original_stdout.flush()
        pass

    def get_output(self):
        with self.lock:
            return ''.join(self.output)

    def clear_output(self):
        with self.lock:
            self.output = []
            self.truncate(0)
            self.seek(0)

class ModeLoader:
    def __init__(self, directory):
        self.directory = (base_dir / directory).resolve()

    def load_modes(self):
        modes = {}
        try:
            if not os.path.exists(self.directory):
                print(f"Directory does not exist: {self.directory}")
                return modes
            # List all files in the directory
            for filename in os.listdir(self.directory):
                if filename.endswith('.txt'):
                    filepath = os.path.join(self.directory, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        mode_name = os.path.splitext(filename)[0]
                        modes[mode_name] = content
        except Exception as e:
            print(f"An error occurred: {e}")
        return modes

class TopAgentHelper:
    def __init__(self, agent):
        self.agent = agent

    def query_single_expert(self, expert_tool, modified_user_input):
        return expert_tool.chat(modified_user_input)

    def query_experts_concurrently(self, expert_tools, modified_user_input):
        results = []
        with ThreadPoolExecutor(max_workers=len(expert_tools)) as executor:
            futures = {executor.submit(self.query_single_expert, tool, modified_user_input): tool for tool in expert_tools}
            for future in concurrent.futures.as_completed(futures):
                tool = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error querying expert tool {tool}: {e}")
        combined_results = "\n".join(results)
        print("\n Generating response by using multiple threads for "+str(expert_tools) +" \n")
        return combined_results

lock = threading.Lock()

# def check_memory(memory_file_name):
#     with lock:
#         with open(memory_file_name, "r", encoding='utf-8') as f:
#             regular_dict = json.load(f)
#             memory_dict = LimitedDict.from_dict(regular_dict, max_size=5)
#     return memory_dict
# def reset_memory(memory_file_name):
#     with lock:
#         memory_dict = LimitedDict()
#         with open(memory_file_name, "w",  encoding='utf-8') as f:
#             json.dump(memory_dict, f)


# from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
# from sentence_transformers import SentenceTransformer
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaIndexOllamaEmbedding
# import time
# from pydantic import PrivateAttr
# from typing import Any, List
# from requests.exceptions import ConnectionError, ChunkedEncodingError
# from openai import APIConnectionError

# class RateLimitedOpenAIEmbedding(BaseEmbedding):

#     _wrapped_model: OpenAIEmbedding = PrivateAttr()
#     _min_time_between_calls: float     = PrivateAttr()
#     _last_call_time: float             = PrivateAttr(default=0.0)
#     _lock: threading.Lock              = PrivateAttr()

#     _failure_count: int = PrivateAttr(default=0)
#     _failure_threshold: int = PrivateAttr(default=1)

#     """
#     OpenAI embedding rate limit wrapper
#     """
#     def __init__( self,
#         wrapped_model: OpenAIEmbedding,
#         min_time_between_calls: float = 0.5,
#         **kwargs: Any,
#     ):
#         super().__init__(
#             model_name=wrapped_model.model_name,
#             embed_batch_size=wrapped_model.embed_batch_size,
#             **wrapped_model.dict(exclude={"model_name", "embed_batch_size"}),
#         )

#         object.__setattr__(self, "_wrapped_model", wrapped_model)
#         object.__setattr__(self, "_min_time_between_calls", min_time_between_calls)
#         object.__setattr__(self, "_last_call_time", 0.0)
#         object.__setattr__(self, "_lock", threading.Lock())

#     def _rate_limit(self) -> None:
#         print("[RateLimiter] ping")  
#         with self._lock:
#             now = time.time()
#             if self._last_call_time > 0:
#                 elapsed = now - self._last_call_time
#                 if elapsed < self._min_time_between_calls:
#                     time.sleep(self._min_time_between_calls - elapsed)
#             object.__setattr__(self, "_last_call_time", time.time())
        
#     def _reset_client(self) -> None:
#         """try to reset the underlying client"""
#         try:
#             print("[OpenAI Embedding] detected multiple failures, trying to recreate the client...")
#             # create a new base model instance
#             from llama_index.embeddings.openai import OpenAIEmbedding
#             new_base_model = OpenAIEmbedding(
#                 model=self._wrapped_model.model_name,
#                 timeout=600, 
#                 embed_batch_size=self._wrapped_model.embed_batch_size,
#                 max_retries=0
#             )
#             # replace the old wrapped model
#             object.__setattr__(self, "_wrapped_model", new_base_model)
#             object.__setattr__(self, "_failure_count", 0)
#             print("[OpenAI Embedding] client has been reset")
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             print(f"[OpenAI Embedding] failed to reset the client: {e}")


#     def _get_text_embedding(self, text: str) -> List[float]:
#         self._rate_limit()
#         try:
#             return self._wrapped_model._get_text_embedding(text)
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             # retry once
#             result = self._wrapped_model._get_text_embedding(text)
#             object.__setattr__(self, "_failure_count", 0)
#             return result


#     def _get_query_embedding(self, query: str) -> List[float]:
#         self._rate_limit()
#         try:
#             return self._wrapped_model._get_query_embedding(query)
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             # retry once
#             result = self._wrapped_model._get_query_embedding(query)
#             object.__setattr__(self, "_failure_count", 0)
#             return result
        
#     def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         self._rate_limit()
#         try:
#             return self._wrapped_model._get_text_embeddings(texts)
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             # retry once
#             results = self._wrapped_model._get_text_embeddings(texts)
#             object.__setattr__(self, "_failure_count", 0)
#             return results
        
#     async def _aget_text_embedding(self, text: str) -> List[float]:
#         self._rate_limit()
#         try:
#             return await self._wrapped_model._aget_text_embedding(text)
#         except (ConnectionError, ChunkedEncodingError, openai.APIConnectionError) as e:
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             result = await self._wrapped_model._aget_text_embedding(text)
#             object.__setattr__(self, "_failure_count", 0)
#             return result

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         self._rate_limit()
#         try:
#             # first attempt
#             return await self._wrapped_model._aget_query_embedding(query)
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             # count + maybe reset
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             print(f"[Embedding] async query failed ({cnt}/{self._failure_threshold}): {e}")
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             # **retry once** on the (possibly new) client
#             result = await self._wrapped_model._aget_query_embedding(query)
#             # clear failures on success
#             object.__setattr__(self, "_failure_count", 0)
#             return result
    
    
#     async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         self._rate_limit()
#         try:
#             return await self._wrapped_model._aget_text_embeddings(texts)
#         except (ConnectionError, ChunkedEncodingError, APIConnectionError) as e:
#             cnt = self._failure_count + 1
#             object.__setattr__(self, "_failure_count", cnt)
#             if cnt >= self._failure_threshold:
#                 self._reset_client()
#             # retry once
#             results = await self._wrapped_model._aget_text_embeddings(texts)
#             object.__setattr__(self, "_failure_count", 0)
#             return results

#     # everything else (batch methods, other attrs) is inherited or forwarded
#     def __getattr__(self, name):
#         return getattr(self._wrapped_model, name)
import threading, time, httpx, openai
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
import random, time, asyncio
from openai import OpenAI
from llama_index.llms.openai import OpenAI as llama_OpenAI
import traceback
import httpx, inspect
MAX_RETRIES = 4       # extra 4 times → total 5 times
BASE_DELAY  = 2.0     # first retry sleep 2 s
JITTER      = 0.25    # ±25% jitter

class RateLimitedOpenAIEmbedding(OpenAIEmbedding):
    """
    directly inherit OpenAIEmbedding, with internal rate limiting + failure reconnection.
    """
    @property
    def client(self) -> OpenAI:
        return self._get_client()
    
    def __init__(self, *args, min_time_between_calls: float = 2, **kw):
        kw["max_retries"] = 0                      # completely disable SDK internal 20× retries
        super().__init__(*args, **kw)


        # self._client = openai.OpenAI(api_key=kw.get("api_key"), timeout=600, max_retries=0)
        
        self._saved_api_key = kw.get("api_key") or os.getenv("OPENAI_API_KEY")
        self._min_interval = min_time_between_calls
        self._last_call    = 0.0
        self._lock         = threading.Lock()

        self._tls = threading.local()
        # self._is_async = is_async

    def _ensure_runtime_state(self):
        """Restore process-local objects that serializers intentionally omit."""
        try:
            object.__getattribute__(self, "_tls")
        except AttributeError:
            object.__setattr__(self, "_tls", threading.local())
        try:
            object.__getattribute__(self, "_lock")
        except AttributeError:
            object.__setattr__(self, "_lock", threading.Lock())
        if not hasattr(self, "_last_call"):
            object.__setattr__(self, "_last_call", 0.0)
        if not hasattr(self, "_min_interval"):
            object.__setattr__(self, "_min_interval", 2.0)


    def _get_client(self):
        """
        Each thread creates a new OpenAI()/AsyncOpenAI() instance when it is first called,
        or when the client is closed.
        """
        self._ensure_runtime_state()
        client = getattr(self._tls, "client", None)
        thread_id = threading.get_ident()
        print(f"[Thread {thread_id}] _get_client() → {id(client)}")
        closed = getattr(client, "is_closed", False) or client is None
        if client is None or closed:
            client = OpenAI(
                api_key     = self._saved_api_key,
                max_retries = 0,
                timeout     = 600,
            )
            self._tls.client = client
        return client

    # ---------- shared rate limiting / failure reconnection ----------
    def _rate_limit(self):
        # self._get_client()
        self._ensure_runtime_state()
        print("[RateLimiter] ping")
        with self._lock:
            now  = time.time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = now


    def _call(self, fn, *args, **kwargs):
        for attempt in range(MAX_RETRIES + 1):          # 0 … 4
            self._rate_limit()
            try:
                return fn(*args, **kwargs)                      # return on success
            # 1) HTTP status errors from OpenAI SDK (includes 5xx like 520)
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", None)
                is_internal = hasattr(openai, "InternalServerError") and isinstance(e, openai.InternalServerError)
                retriable = (
                    is_internal or
                    (status is not None and (500 <= status < 600 or status in (408, 409, 429)))
                )
                if not retriable or attempt == MAX_RETRIES:
                    raise
                print(f"[Embedding] server-err {status or 'unknown'} (sync) {attempt+1}/{MAX_RETRIES}: {e}")
                # Recreate client to clear any bad connections
                if hasattr(self._tls, "client"):
                    delattr(self._tls, "client")
                backoff = BASE_DELAY * (2 ** attempt)
                backoff *= 1 + random.uniform(-JITTER, JITTER)
                time.sleep(min(backoff, 30))
            # 2) Network/connection issues
            except (openai.APIConnectionError, httpx.NetworkError, httpx.TimeoutException, RuntimeError) as e:
                root = e.__cause__ or e
                print("⤷ root:", repr(root))
                if hasattr(root, "__cause__") and root.__cause__:
                    print("⤷  inner:", repr(root.__cause__))
                if attempt == MAX_RETRIES:
                    raise    # raise on 5th failure
                print(f"[Embedding] conn-err (sync) {attempt+1}/{MAX_RETRIES}: {e}")
                if hasattr(self._tls, "client"):
                    delattr(self._tls, "client")
                backoff = BASE_DELAY * (2 ** attempt)
                backoff *= 1 + random.uniform(-JITTER, JITTER)
                time.sleep(min(backoff, 30))

    async def _acall(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._call(fn, *args, **kwargs))

    # ---------- override four abstract hooks ----------
    def _get_text_embedding(self, t: str) -> List[float]:
        return self._call(super()._get_text_embedding, t)

    def _get_query_embedding(self, q: str) -> List[float]:
        return self._call(super()._get_query_embedding, q)

    async def _aget_text_embedding(self, t: str) -> List[float]:
        return await self._acall(super()._get_text_embedding, t)

    async def _aget_query_embedding(self, q: str) -> List[float]:
        vec = await self._acall(super()._get_query_embedding, q)
        print("[Embedding] returned vector len =", len(vec))
        return vec


_ollama_embed_model_instance = None
# _openai_embed_model_instance = None
_embed_model_lock = threading.Lock()


def get_shared_ollama_embedding(model_name="nomic-embed-text"):
        """
        Get a shared OllamaEmbedding instance.
        This can significantly reduce the number of interactions with the Ollama service.
        """
        global _ollama_embed_model_instance
        # First check (no lock), if instance exists, return quickly
        if _ollama_embed_model_instance is not None:
            return _ollama_embed_model_instance
        
        # if instance does not exist, get lock and create instance
        with _embed_model_lock:
            # second check (with lock), prevent other threads from creating instance
            if _ollama_embed_model_instance is None:
                print(f"[Embedding Manager] Creating shared OllamaEmbedding instance for model: {model_name}")
                try:
                    # Use the aliased class imported above.
                    _ollama_embed_model_instance = LlamaIndexOllamaEmbedding(model_name=model_name, request_timeout=600)
                    print(f"[Embedding Manager] Shared OllamaEmbedding instance created successfully.")
                except Exception as e:
                    print(f"[Embedding Manager] CRITICAL ERROR: Failed to create shared OllamaEmbedding instance: {e}")
                    raise
        return _ollama_embed_model_instance


# def get_shared_openai_embedding(model_name="text-embedding-ada-002"):
#         """
#         Get a shared OpenAI instance.
#         This can significantly reduce the number of interactions with the OpenAI service.
#         """
#         global _openai_embed_model_instance
#         # First check (no lock), if instance exists, return quickly
#         if _openai_embed_model_instance is not None:
#             return _openai_embed_model_instance
        
#         # if instance does not exist, get lock and create instance
#         with _embed_model_lock:
#             # second check (with lock), prevent other threads from creating instance
#             if _openai_embed_model_instance is None:
#                 print(f"[Embedding Manager] Creating shared OpenAI instance for model: {model_name}")
#                 try:
#                     # Use the aliased class imported above.
#                     # _openai_embed_model_instance = OpenAIEmbedding(model=model_name, timeout=600, embed_batch_size=8, max_retries=20)
#                     base_model = OpenAIEmbedding(model=model_name, timeout=600, embed_batch_size=1, max_retries=0)
#                     _openai_embed_model_instance = RateLimitedOpenAIEmbedding(wrapped_model=base_model,
#                     min_time_between_calls=2)
#                     print(f"[Embedding Manager] Shared OpenAI instance created successfully.")
#                 except Exception as e:
#                     print(f"[Embedding Manager] CRITICAL ERROR: Failed to create shared OpenAI embedding instance: {e}")
#                     raise
#         return _openai_embed_model_instance

_tls = threading.local()

def get_shared_openai_embedding(model_name="text-embedding-ada-002"):
    if getattr(_tls, "embed", None) is None:
        _tls.embed = RateLimitedOpenAIEmbedding(
            model=model_name,
            embed_batch_size=1,          # avoid large batch size
            min_time_between_calls=1,  # mild rate limiting
        )
    return _tls.embed


#
def compare_query_similarity(input, reference_input):
    # embedding_model = OpenAIEmbedding()
    # embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    input_embed = embed_model.get_text_embedding(input)
    reference_embed = embed_model.get_text_embedding(reference_input)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    score = cosine_similarity(np.array(input_embed), np.array(reference_embed))

    return score





def add_query_response_to_store(query_text, response_text, processed_response, Coa_text, web_search, complex, user_shared, mode):
    # embedding = embedding_model.encode(query_text)
    embedding_list =  embed_model.get_text_embedding(query_text)
    # embedding_list = embedding.tolist()
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(response_text.split("\n")[0])
    # topic_model = BERTopic()
    # chunks = str(response['response']).split(". ")  # Split by sentence
    # chunkssu = [chunk.strip() for chunk in chunks if chunk]
    # topics, probs = topic_model.fit_transform(chunks)
    # highest_prob_topic = probs[0].argmax()  # Since it's a single document, use index 0
    # topics = topic_model.get_topic(highest_prob_topic)
    # topic = topics[0][0] if topics else None

    topic = keywords[0][0] if keywords else ""
    time_stamp = datetime.utcnow().isoformat()
    metadata = {
        "topic": topic,
        "timestamp": time_stamp,
        "user_shared": user_shared,
        "web_search_required": web_search,
        "complex": complex,
        "COA_text": Coa_text,
        "mode": mode,
        "question": query_text,
    }
    if complex:
        try:
            vector_store._collection.add(
                ids=[str(uuid.uuid4())],
                documents=[query_text],
                embeddings=[embedding_list],
                # metadatas=[{'response': str(response['response']), **metadata}],
                metadatas=[{'response':processed_response, **metadata}],
            )
        except Exception as e:
            print(f"Error adding to collection: {e}")

    else:
        metadata["COA_text"] = ""
        vector_store._collection.add(
            ids=[str(uuid.uuid4())],
            documents=[query_text],
            embeddings=[embedding_list],
            # metadatas=[{'response': str(response['response']), **metadata}],
            metadatas=[{'response': response_text, **metadata}],
        )


def add_doc_summary_to_store(summary_text, doc_tool_id):
    existing_entries = vector_store_summaries._collection.get(
        where={"response": doc_tool_id}
    )

    if existing_entries and existing_entries.get("ids", []):
        print(f"doc_tool_id {doc_tool_id} already exists")
        return

    # embedding_list =  embed_model.get_text_embedding(summary_text)

    content = re.sub(rf'^\s*{re.escape(doc_tool_id)}\s*:\s*', '', summary_text, count=1).strip()
    if not content:
        # Fallback in the unlikely event the summary is just the id.
        content = summary_text.strip()
    embedding_list = embed_model.get_text_embedding(content)

    vector_store_summaries._collection.add(
        ids=[str(uuid.uuid4())],
        documents=[content],
        embeddings=[embedding_list],
        metadatas=[{'response':doc_tool_id}],
    )
    # doc_summary_dict["tool_"+str(doc_tool_id)] = summary_text

def get_similar_queries(new_query_text, top_k=1, similarity_threshold=0.85):
    # new_query_embedding = embedding_model.encode(new_query_text)
    new_query_embedding =  embed_model.get_text_embedding(new_query_text)
    # index = VectorStoreIndex.from_vector_store(vector_store)
    # query_engine = index.as_query_engine(similarity_top_k= 3)
    # response = query_engine.query(new_query_text)

    results = vector_store._collection.query(
        query_embeddings=[new_query_embedding], #new_query_embedding.tolist()
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    similar_queries = []
    for doc, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = 1 - distance #for cosine distance
        # similarity = np.exp(-distance) #for L2 distance by default in chroma
        if similarity > similarity_threshold:
            similar_queries.append({
                'query': doc,
                'response_metadata': metadata,
                'similarity': similarity
            })

    return similar_queries
class QueryTracker:
    def __init__(self):
        self.query_log = {}

    def log_query(self, tool_name, query, result):
        self.query_log[(tool_name, query)] = result

    def has_query_been_made(self, tool_name, query):
        return (tool_name, query) in self.query_log

stdout_lock = threading.Lock()
stderr_lock = threading.Lock()
original_stdout = sys.stdout
original_stderr = sys.stderr
output_logger = OutputLogger()
display_writer = DisplayNWrite(original_stdout, output_logger)



base_dir = Path(__file__).parent
from chromadb.config import Settings
runtime_directory = os.environ.get("MILONET_RUNTIME_DIR")
if runtime_directory:
    runtime_directory_path = Path(runtime_directory).expanduser().resolve()
    runtime_directory_path.mkdir(parents=True, exist_ok=True)
    persist_directory = str(runtime_directory_path / "chroma_db")
    persist_directory_summary = str(runtime_directory_path / "chroma_db_summary")
else:
    persist_directory = "chroma_db"
    persist_directory_summary = "chroma_db_summary"

chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="query_responses", metadata={'hnsw:space': 'cosine'}) #for cosine distance, L2 distance by default in chroma
vector_store = ChromaVectorStore(chroma_collection=collection)

chroma_client_summary = chromadb.PersistentClient(path=persist_directory_summary)
collection_summaries = chroma_client_summary.get_or_create_collection(name="query_doc_summaries", metadata={'hnsw:space': 'cosine'}) #for cosine distance, L2 distance by default in chroma
vector_store_summaries = ChromaVectorStore(chroma_collection=collection_summaries)
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #sentence-transformers/all-MiniLM-L6-v2
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2") sentence-transformers/all-mpnet-base-v2
# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )

lc_embed_model = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_name="intfloat/multilingual-e5-large"
)

embed_model = LangchainEmbedding(lc_embed_model)
tool_list = []
SHORT_ID_PATTERN = re.compile(r"^[a-f0-9]{8}(?:_\d+)?$")


_file_read_lock = threading.Lock()

def get_shortened_id_from_file(original_id):
    try:
        result = None
        with _file_read_lock:
            if os.path.exists("global_id_map.json"):
                with open("global_id_map.json", 'r', encoding='utf-8') as f:
                    global_map = json.load(f)
                result = global_map.get(original_id)    
        return result
    except Exception as e:
        print(f"Error getting shortened ID: {e}")
    return None

def parse_tool_access_file(file_path="tool_access_info.txt"):
    if not os.path.exists(file_path):
        print(f"Warning: Tool access file does not exist: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        tool_matches = re.findall(r'(?i)(?:tool|Tool)_([a-zA-Z0-9_]+|[a-z0-9]{8}_\d+) has access to:', content)
        return tool_matches
    except Exception as e:
        print(f"Error parsing tool access file: {e}")
        return []

def update_tool_list():
    global tool_list
    with _tool_list_lock:
        current_tools = tool_list.copy()
        
        if not current_tools:
            print("tool_list is empty, but skipping parse_tool_access_file as requested")
            return tool_list
        
        updated_tool_list = []
        for tool_id in current_tools:
            original_id = tool_id
            if isinstance(tool_id, str) and tool_id.startswith('tool_'):
                tool_id = tool_id.split("tool_", 1)[1]

            if SHORT_ID_PATTERN.match(tool_id):
                shortened_id = tool_id
            else:
                shortened_id = get_shortened_id_from_file(tool_id)

            if shortened_id:
                updated_tool_list.append(shortened_id)
            else:
                print(f"Warning: Could not map tool id {original_id}, keeping original value")
                updated_tool_list.append(original_id)
        
        tool_list = updated_tool_list
        print(f"Updated tool list: {(tool_list)}")
        # return tool_list

# def ensure_consistent_tool_list_format():
#        global tool_list
       
#        if not tool_list:
#            return update_tool_list()
       
#        for i, item in enumerate(tool_list):
#            if not isinstance(item, str):
#                print(f"Warning: Non-string item in tool_list: {item}")
#                tool_list[i] = str(item)
           
#            if not tool_list[i].startswith("tool_"):
#                tool_list[i] = f"tool_{tool_list[i]}"
       
#        tool_list = list(set(tool_list))
#        print(f"Normalized tool list with {len(tool_list)} tools")
#        return tool_list

# def debug_id_mappings():
      
#        if os.path.exists("global_id_map.json"):
#            try:
#                with _file_read_lock: 
#                 with open("global_id_map.json", 'r', encoding="utf-8") as f:
#                     file_map = json.load(f)
#                     print(f"File mapping has {len(file_map)} entries")
                    
#                     examples = list(file_map.items())[:3]
#                     print(f"File mapping examples: {examples}")
                    
#                     if tool_list:
#                         matches = 0
#                         for t in tool_list:
#                                 if t.startswith('tool_'):
#                                     tool_id_without_prefix = t[5:]
#                                     if tool_id_without_prefix in file_map.values():
#                                         matches += 1
#                                 else:
#                                     if t in file_map.values():
#                                         matches += 1
#                         print(f"Found {matches}/{len(tool_list)} tools using shortened format")
#            except Exception as e:
#                print(f"Error reading mapping file: {e}")

# from threading import Lock

# def log_structured_trace_safely(
#     trace_data: dict, 
#     log_file_path: str, 
#     lock: Lock
# ):
 
#     try:
#         json_string = json.dumps(trace_data)
        
#         with lock:
#             with open(log_file_path, 'a', encoding='utf-8') as f:
#                 f.write(json_string + '\n')
        
#         print(f"✅ Trace successfully logged to {log_file_path}")
        
#     except Exception as e:
#         print(f"❌ Failed to log trace to file: {e}")
#         traceback.print_exc()

from contextlib import contextmanager


ID_SEPARATOR = "||"
TRACE_ID_KEY = "TRACE_ID"
PARENT_TASK_ID_KEY = "PARENT_TASK_ID"


@contextmanager
def id_stamped_print(trace_id: str, task_id: str, parent_task_id: str):
    """one-time print with trace_id, task_id, parent_task_id"""
    original_stdout_write = sys.stdout.write
    prefix = f"[{TRACE_ID_KEY}={trace_id}][{PARENT_TASK_ID_KEY}={parent_task_id}][TASK_ID={task_id}] "

    def new_write(text: str):
        # only add prefix to non-empty, non-whitespace lines
        stripped_text = text.strip()
        if stripped_text:
            # add prefix to each line of multi-line text
            lines = stripped_text.split('\n')
            prefixed_lines = [prefix + line for line in lines]
            original_stdout_write('\n'.join(prefixed_lines) + '\n')
        elif text == '\n':
             # print empty lines
             original_stdout_write(text)
        # ignore pure whitespace or empty strings

    sys.stdout.write = new_write
    try:
        yield
    finally:
        # restore original sys.stdout.write
        sys.stdout.write = original_stdout_write

# --- ID processing functions ---
def encode_ids_in_query(query: str, trace_id: str, parent_task_id: str) -> str:
    """encode IDs to the end of the query string"""
    return f"{query}{ID_SEPARATOR}{TRACE_ID_KEY}={trace_id}{ID_SEPARATOR}{PARENT_TASK_ID_KEY}={parent_task_id}{ID_SEPARATOR}"

def parse_ids_from_query(query: str) -> tuple[str, str, str]:
    """Return the clean query, trace ID, and parent task ID."""
    trace_id = "UNKNOWN_TRACE"
    parent_task_id = "UNKNOWN_PARENT"
    # Match the encoded trace identifiers appended to the query.
    pattern = re.compile(f"\\{ID_SEPARATOR}{TRACE_ID_KEY}=(.*?)\\{ID_SEPARATOR}{PARENT_TASK_ID_KEY}=(.*?)\\{ID_SEPARATOR}$", re.DOTALL)
    
    match = pattern.search(query)
    
    if match:
        trace_id = match.group(1).strip()
        parent_task_id = match.group(2).strip()
        # Remove the encoded identifiers before forwarding the query.
        clean_query = query[:match.start()].strip()
        return clean_query, trace_id, parent_task_id
    else:
        # Return the input unchanged when no identifiers are present.
        return query, trace_id, parent_task_id

from typing import Any
from llama_index.core.tools import BaseTool, ToolMetadata, FunctionTool
class IDInjectingWrapperTool(BaseTool):
    """Inject trace identifiers before invoking a wrapped tool."""
    def __init__(self, original_tool: BaseTool, trace_id: str, parent_task_id: str):
        self._original_tool = original_tool
        self._trace_id = trace_id
        self._parent_task_id = parent_task_id
        # Preserve the original tool metadata on the wrapper.
        self._metadata = original_tool.metadata

    @property
    def metadata(self) -> ToolMetadata:
        # Expose the original tool name and description to the agent.
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> Any:
        # Extract the query from common positional or keyword inputs.
        original_query = None
        input_key = None  # Keyword containing the query, when present.

        # Check common keyword arguments first.
        if "input" in kwargs:
            original_query = kwargs["input"]
            input_key = "input"
        elif "query" in kwargs:
            original_query = kwargs["query"]
            input_key = "query"
        elif args:
            # Fall back to the first positional argument.
            original_query = args[0]
        
        # Inject trace identifiers when the query is a string.
        if original_query is not None:
            # Non-string inputs are forwarded without modification.
            if not isinstance(original_query, str):
                print(f"Warning: Tool input for {self.metadata.name} is not a string. Bypassing ID injection.")
                return self._original_tool.call(*args, **kwargs)

            # Append the trace identifiers to the query.
            query_with_ids = encode_ids_in_query(
                original_query,
                self._trace_id,
                self._parent_task_id
            )

            # Restore the encoded query to its original argument position.
            if input_key:
                kwargs[input_key] = query_with_ids
                return self._original_tool.call(*args, **kwargs)
            else:  # The query came from positional arguments.
                new_args = (query_with_ids,) + args[1:]
                return self._original_tool.call(*new_args, **kwargs)

        # Forward unrecognized inputs without modification.
        else:
            print(f"Warning: Could not determine a query string from the input for tool {self.metadata.name}. Calling original tool directly.")
            return self._original_tool.call(*args, **kwargs)

    # Apply the same identifier injection for asynchronous calls.
    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped tool asynchronously with trace identifiers."""
        original_query = None
        input_key = None

        # Extract the original query.
        if "input" in kwargs:
            original_query = kwargs["input"]
            input_key = "input"
        elif "query" in kwargs:
            original_query = kwargs["query"]
            input_key = "query"
        elif args:
            original_query = args[0]
        
        # Inject identifiers when the query is a string.
        if original_query is not None and isinstance(original_query, str):
            query_with_ids = encode_ids_in_query(
                original_query,
                self._trace_id,
                self._parent_task_id
            )
            
            # Restore the encoded query to its original argument position.
            if input_key:
                new_kwargs = kwargs.copy()
                new_kwargs[input_key] = query_with_ids
                # Invoke and await the original tool's acall method.
                return await self._original_tool.acall(*args, **new_kwargs)
            else:
                new_args = (query_with_ids,) + args[1:]
                # Invoke and await the original tool's acall method.
                return await self._original_tool.acall(*new_args, **kwargs)
        else:
            # Forward unrecognized inputs without modification.
            return await self._original_tool.acall(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

def parse_retrieval_traces_updated(file_path: str) -> List[Dict]:
    """
    parse a retrieval trace file, extract the question, retrieved texts and response.
    a session starts with "THE CURRENT USER INPUT:" and ends with the next "THE CURRENT USER INPUT:" or file end.
    The response section starts with "Response:" and can be multi-line until the next session or file end.
    :param file_path: the path to the retrieval_traces.txt file.
    :return:
        a list of dictionaries, each dictionary represents a session.
    """
    all_sessions = []
    current_session = None
    in_response = False
    response_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                if stripped_line.startswith("THE CURRENT USER INPUT:"):
                    # Save previous session if exists
                    if current_session:
                        if in_response:
                            current_session["response"] = "\n".join(response_lines).strip()
                            in_response = False
                            response_lines = []
                        all_sessions.append(current_session)
                    # Start new session
                    current_session = {
                        "question": stripped_line.split(":", 1)[1].strip(),
                        "retrieved_texts": [],
                        "response": None
                    }
                    continue

                if current_session:
                    if in_response:
                        # Check if this line is a new session start (shouldn't happen here, but for safety)
                        if stripped_line.startswith("THE CURRENT USER INPUT:"):
                            current_session["response"] = "\n".join(response_lines).strip()
                            all_sessions.append(current_session)
                            # Start new session
                            current_session = {
                                "question": stripped_line.split(":", 1)[1].strip(),
                                "retrieved_texts": [],
                                "response": None
                            }
                            in_response = False
                            response_lines = []
                        else:
                            response_lines.append(stripped_line)
                    elif stripped_line.startswith("Response: Direct Answer:"):
                        # Start collecting response lines
                        response_lines = [stripped_line.split(":", 1)[1].strip()]
                        in_response = True
                    else:
                        match = re.match(r"Got ctx: \[\d+\] (.+)", stripped_line)
                        if match:
                            current_session["retrieved_texts"].append(match.group(1))

        # After file ends, save the last session
        if current_session:
            if in_response:
                current_session["response"] = "\n".join(response_lines).strip()
            all_sessions.append(current_session)
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error parsing the file: {e}")
        return []

    return all_sessions

def parse_retrieval_traces(file_path: str) -> List[Dict]:
        """
        parse a retrieval trace file, extract the question, retrieved texts and response.
        a session starts with "THE CURRENT USER INPUT:" and ends with "Response:".
        
        :param file_path: the path to the retrieval_traces.txt file.
            
        :return:
            a list of dictionaries, each dictionary represents a session.
        """
        all_sessions = []
        current_session = None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue

                    if stripped_line.startswith("THE CURRENT USER INPUT:"):
                   
                        current_session = {
                            "question": stripped_line.split(":", 1)[1].strip(),
                            "retrieved_texts": [],
                            "response": None
                        }
                        continue

                    if current_session:
                        if stripped_line.startswith("Response:"):
                            current_session["response"] = stripped_line.split(":", 1)[1].strip()
                            all_sessions.append(current_session)
                            current_session = None
                        else:
                            current_session["retrieved_texts"].append(stripped_line)


        except FileNotFoundError:
            print(f"Error: file '{file_path}' not found.")
            return []
        except Exception as e:
            print(f"Error parsing the file: {e}")
            return []

        if current_session:
            all_sessions.append(current_session)
        return all_sessions

class Filter:
    """
    use LLM model to filter the used texts.
    """
    def __init__(self, model_name='gpt-4.1-nano', temperature=0.0):
        """
        initialize the filter.
        :param chat_function: the function you use to interact with DeepSeek API.
        :param model_name: the name of the DeepSeek model to use.
        """
        # self.chat = chat_function
        # self.model = model_name
        self.llm = llama_OpenAI(model=model_name, temperature=temperature, timeout=600, max_retries=5)
    

    def filter_used_texts(self, question: str, response: str, retrieved_texts: List[str]) -> List[str]:
        """
        filter the used texts from all retrieved texts.

        :param question: the original question from the user.
        :param response: the final system response.
        :param retrieved_texts: a list of all retrieved texts.
        :return: a list of the used texts.
        """
        # Step 1: remove duplicate retrieved texts while keeping the original order
        unique_texts = list(dict.fromkeys(set(retrieved_texts)))

        # Step 2: format the texts, so the model can clearly refer to them (by index)
        formatted_texts = "\n\n---\n\n".join(
            [f"Text {i}:\n{text}" for i, text in enumerate(unique_texts)]
        )

        try:
            print("-" * 25, "DEBUGGING: filter_used_texts INPUTS", "-" * 25)
            print(f"QUESTION (type: {type(question)}):\n---\n{question}\n---")
            print(f"RESPONSE (type: {type(response)}):\n---\n{response}\n---")
            print("-" * 75)
            # response: ChatResponse = self.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            # response_judge: ChatResponse = chat(model=self.model, messages=[
            #         {
            #         'role': 'user',
            #         'content': f"""I have a response: "{response}"

            # I need to find which of these texts contain information from that response:

            # {formatted_texts}

            # Return only a JSON list of text numbers that contain information mentioned in the response.
            # For example: [5, 13]

            # JSON list:""",
            #                     },
            #                 ]
            #             )
            # content = response_judge.message.content.split("</think>")[1].strip()
            prompt = f"""You are a strict text attribution analyzer. Your task is to identify texts that were actually used to generate the response.

            QUESTION: {question}

            RESPONSE: {response}

            RETRIEVED TEXTS:
            {formatted_texts}

            ANALYSIS RULES:
            
            **CASE 1: Positive Responses (response provides factual information)**
            Include a text ONLY if:
            - Specific facts, entities, dates, or statements from it appear explicitly in the response
            - The response directly cites, quotes, or paraphrases content from that text
            
            **CASE 2: Negative Responses (response says "not found", "does not contain", "cannot be determined")**
            For these responses, identify texts that:
            - Contain closely related terms or concepts that were checked but found insufficient
            - Are topically relevant to the question's domain and entities
            - Were likely consulted to determine the information is absent
            - Exclude: Texts that are completely unrelated to the question's topic or only share tangential keywords
            
            **STRICT EXCLUSIONS (for both cases):**
            - Texts that only share generic keywords without substantive relevance
            - Texts providing unrelated background context
            - Texts about different topics that happen to share a word with the question
            
            **EXAMPLES:**
            - Q: "Where was X born?" + R: "X was born in Paris, France in 1985" → Include texts stating X's birthplace and year
            - Q: "What do A and B have in common?" + R: "Both are physicists" → Include texts with A's profession and B's profession
            - Q: "Which city hosted Y event?" + R: "Documents do not mention which city hosted Y" → Include texts about Y event; Exclude unrelated city/event texts
            - Q: "What is Z's occupation?" + R: "Documents do not specify Z's occupation" → Include texts mentioning Z by name; Exclude texts only sharing generic keywords

            Return ONLY a JSON list of text numbers that were genuinely used (positively or checked negatively).

            JSON list:"""
            llm_response = self.llm.complete(prompt=prompt)
            content = llm_response.text
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                used_indices = json.loads(json_str)
                
                # Step 6: extract the actual texts based on the indices
                used_texts = [unique_texts[i] for i in used_indices if i < len(unique_texts)]
                return used_texts
            else:
                return [] # if no valid JSON is found, return an empty list

        except Exception as e:
            print(f"[Filter Error] Failed to filter texts: {e}")
            return [] # if any error occurs, return an empty list


from collections import defaultdict
import unicodedata 
def normalize_question_key(text: Any) -> str:
    """
    a powerful function to normalize the question string to a standard format to ensure reliable matching.
    """
    if not isinstance(text, str):
        return ""
    
    # step 1: NFC normalization, to solve the problem of multiple representations of 'é' etc.
    text = unicodedata.normalize('NFC', text)
    
    # step 2: replace all types of whitespace characters (including non-breaking spaces \xa0) with a single normal space
    text = re.sub(r'\s+', ' ', text)
    
    # step 3: convert to lowercase
    text = text.lower()
    
    # step 4: remove leading and trailing whitespace
    text = text.strip()
    
    return text

def get_ids_from_structured_contexts(structured_contexts):
    """Extracts all sentence_ids from a structured context list."""
    return {sentence_id for doc in structured_contexts for sentence_id, _ in doc}

def get_texts_from_structured_contexts(structured_contexts):
    """Extracts all sentence texts from a structured context list for RAGAS."""
    return [sentence_text for doc in structured_contexts for _, sentence_text in doc]

def restructure_contexts_like_documents_sentences(context_list: list, documents_sentences: list) -> list:
    """
    restructure a list of context blocks into a structure similar to documents_sentences.
    
    Args:
        context_list: a list of context blocks.
        documents_sentences: a nested list structure of the standard answer.

    Returns:
        a nested list, the structure is the same as documents_sentences, but only contains the sentences found in context_list.
    """
    # step 1: create an efficient lookup dictionary
    all_sentences_map = {}
    sentence_id_to_doc_id_map = {}
    for doc_group in documents_sentences:
        for sentence_id, sentence_text in doc_group:
            all_sentences_map[sentence_id] = sentence_text
            # use regex to extract the number part of the sentence ID as the document ID
            doc_id_match = re.match(r'(\d+)', sentence_id)
            if doc_id_match:
                sentence_id_to_doc_id_map[sentence_id] = doc_id_match.group(1)

    # use defaultdict(dict) to automatically handle new document IDs and ensure sentence uniqueness
    found_sentences_by_doc = defaultdict(dict)

    for context_block in context_list:
        for sentence_id, sentence_text in all_sentences_map.items():
            if sentence_text in context_block:
                doc_id = sentence_id_to_doc_id_map.get(sentence_id)
                if doc_id:
                    # store the found [id, text] in the corresponding document's group
                    found_sentences_by_doc[doc_id][sentence_id] = [sentence_id, sentence_text]


    final_structured_list = []
    # sort by document ID to maintain the stability of the output order
    for doc_id in sorted(found_sentences_by_doc.keys()):
        # .values() extracts all [id, text] pairs
        sentences_in_doc = list(found_sentences_by_doc[doc_id].values())
        final_structured_list.append(sentences_in_doc)
        
    return final_structured_list

def create_benchmark_lookup(benchmark_file_path: str) -> dict:
    """
    load ragbench_test.jsonl and create a lookup dictionary from question to documents_sentences.
    """
    print(f"building lookup dictionary from {benchmark_file_path}...")
    lookup_table = {}
    try:
        with open(benchmark_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    required_keys = ['question', 'documents_sentences', 'all_relevant_sentence_keys', 'all_utilized_sentence_keys', 'response']
                    if all(key in data for key in required_keys):
                        normalized_key = normalize_question_key(data['question'])
                        lookup_table[normalized_key] = {
                            'documents_sentences': data['documents_sentences'],
                            'documents': data['documents'],
                            'all_relevant_sentence_keys': data['all_relevant_sentence_keys'],
                            'all_utilized_sentence_keys': data['all_utilized_sentence_keys'],
                            'response': data['response']
                        }
                except (json.JSONDecodeError, KeyError):
                    continue  # skip lines with incorrect format
    except FileNotFoundError:
        print(f"error: could not find benchmark file {benchmark_file_path}")
        return None
    print(f"✅ lookup dictionary built, containing {len(lookup_table)} unique questions.")
    return lookup_table

class ResponseQualityEvaluator:
    def __init__(self, input_file, output_file):
        self.llm = llama_OpenAI(model="o4-mini")
        self.input_file = input_file
        self.output_file = output_file
        
    def evaluate_response_quality(self):
        quality_data = []
        with open(self.input_file, 'r', encoding='utf-8') as f_in:
            entries = [json.loads(line) for line in f_in if line.strip()]

        total = len(entries)
        progress_step = max(1, total // 20) if total else 1

        for idx, data in enumerate(entries, start=1):
                question = data['question']
                original_response = data['response']
                milo_response = data['milo_response']
                documents = data['documents']
                original_label = data.get("original_label", "original response")
                comparison_label = data.get("comparison_label", "RAG system's response")

                prompt = f"""
                You are an objective evaluator comparing two responses based ONLY on how well they align with the provided documents.

                Given:
                - Question: {question}
                - {original_label}: {original_response}
                - {comparison_label}: {milo_response}
                - Supporting documents: {documents}

                GENERAL RULES (Fair, document-only, independent grading)
                - Use ONLY the documents to judge correctness. No outside knowledge.
                - Evaluate each response on an ABSOLUTE scale, INDEPENDENT of the other response. If both meet Score 3, both get 3.
                - Treat the entire content of each response—including any “supporting details”, quotes, bullets, or explanations—as part of that response.
                - Silence in a document is NOT a contradiction. A claim is wrong only if a document contradicts it or it fabricates facts absent from the documents.
                - If a response claims a document says/doesn't say X, verify precisely. Penalize misquoting or mischaracterizing documents.
                - Use documents as evidence only when they explicitly connect to the subject or claim. General background passages do NOT support entity-specific claims unless the link is stated.
                - Focus grading on statements that actually resolve the question. Treat tangential commentary (e.g., extra trivia, meta-observations about sources) as neutral unless it contradicts the documents or changes the answer.
                - Authoritative sets or categories named in the documents (e.g., “Entente/Allied powers”, “member states”) count as directly identifying the parties; do not penalize a response for citing the set name instead of enumerating each member unless the question explicitly requires a list of individual countries.
                - Respect temporal qualifiers ("as of 2016", etc.). Do NOT transfer facts across years unless the documents explicitly allow it.
                - **CRITICAL: TENSE/TEMPORAL ACCURACY & CEASED OPERATIONS**: Pay close attention to verb tenses. If documents state an activity "ceased", "ended", "stopped", or "closed", then present-tense claims are FACTUALLY INCORRECT. When evaluating:
                  * Query uses present tense + documents show past/ceased status → response correctly uses past tense = Score 2-3 (accurate premise correction).
                  * Response uses present tense when documents indicate past/ceased = temporal error, reduce score.
                  * Response states "documents do not say [entity] [present verb] [action]" then provides correct past-tense version = Score 3 (proper correction, NOT mischaracterization).
                  * Terms like "ceased", "stopped", "closed", "ended" mean NO LONGER ONGOING. Present tense incompatible with these.
                - Respect geographic/scope qualifiers exactly as written (e.g., "in Europe", "within the company"). If the question has no qualifier but the documents contrast an unrestricted claim with a qualified one, treat the unrestricted claim as the default reference unless the response clearly states the qualifier.
                - When documents describe a chronological chain (e.g., "invented by X" followed by "first used in Europe by Y"), treat the earliest matching event as essential context. Answers that skip an earlier unrestricted claim in favor of a later qualified claim should lose credit.
                - **REASONABLE SEMANTIC INFERENCE**: Accept reasonable semantic inferences when finding commonalities or shared attributes between entities. This includes:
                  * Using document-supported synonyms or related terms within the same domain (e.g., "minister" and "cleric" as religious leaders)
                  * Using appropriate hypernyms (broader category terms) when documents support the hierarchical relationship (e.g., "clergy" as a category encompassing both "minister" and "cleric")
                  * Making logical connections between related concepts that are explicitly mentioned in documents
                  * **INSTITUTION NAMING EVOLUTION**: When documents show an institution underwent name changes (e.g., "Company A incorporated 1975" later became "Company A International", or "Institute B established 1920" renamed to "University B"), BOTH naming forms are acceptable when identifying the entity. Responses using the modern name with historical context (e.g., "currently known as University B, established as Institute B in 1920") or using only the founding name are equally valid. DO NOT penalize either approach as long as the entity is correctly identified and founding/historical information is accurate.
                  * Penalize ONLY when inferences are clearly unsupported, contradictory, or stretch beyond reasonable semantic bounds
                - **SHARED ATTRIBUTE INFERENCE**: When questions ask about shared attributes or commonalities, allow reasonable synthesis of information across documents. If documents provide related but not identical terms for similar roles/attributes, responses may use appropriate unifying terms that accurately capture the commonality. For example: If one document calls person A a "minister" and another calls person B a "cleric" (both in religious contexts), it is reasonable to infer they share the occupation of "religious leaders" or "clergy" when asked what they have in common.
                - **NO INDUSTRY/DOMAIN INFERENCE**: Do NOT infer industry/domain from roles under ANY circumstances. If the documents do not explicitly state the industry (e.g., "film industry", "entertainment industry"), you cannot claim any shared industry.
                - **CONTEXTUAL WORD MATCH**: For occupational comparisons, prefer identical terms but allow clearly synonymous terms and appropriate hypernyms within the same domain. Accept reasonable semantic categorization when documents support the relationship.
                - **ATTRIBUTE DETERMINATION PRINCIPLE**: For questions asking about shared attributes, categories, or memberships (industry, domain, field, etc.), require explicit naming or clear evidence in documents. Shared superficial characteristics or roles do NOT automatically indicate shared attributes unless explicitly stated. If documents lack explicit statements about the attribute for all parties involved, the correct response is that the commonality cannot be determined from the available information.
                - **ADMINISTRATIVE LABELS**: Treat administrative labels as distinct unless documents explicitly state equivalence. "City" ≠ "municipality" unless explicitly connected.
                - If the query explicitly demands a specific administrative level (e.g., county, province) and the documents only provide a related but non-identical label (e.g., borough, city), count that attribute as missing unless the document explicitly states the equivalence.
                - When documents DO supply the specific attribute asked (e.g., the “concept” of a business) and a response leaves it out, treat that as incomplete—no Score 3. If the documents lack that attribute, responses must explicitly note the gap to earn full credit.

                ⚠️ CRITICAL: QUESTION TYPE DISTINCTION (READ THIS FIRST)
                Before scoring, identify the question type:
                
                **TYPE 1: COMMONALITY-ONLY questions** (asking ONLY for shared attributes)
                - Examples: "What profession do they share?", "What do X and Y have in common?", "What occupation do both have?"
                - **SCORING REQUIREMENT**: Response ONLY needs to accurately identify the shared attribute(s). 
                - **ADDING DIFFERENCES IS OPTIONAL BONUS, NOT REQUIRED.**
                - If response correctly identifies commonality → Score 3 (even without mentioning differences)
                
                **TYPE 2: SAMENESS/COMPARISON questions** (asking if entities are identical or same on specific attributes)
                - Examples: "Do they share the SAME profession?", "Did they hail from the SAME country and career?"
                - **SCORING REQUIREMENT**: Response should clarify BOTH what matches AND what differs (when both are documented).
                - If response only says "Yes" or "No" without clarifying which parts match/differ → incomplete (Score 1-2)

                SCORING SCALE (0–3), rewarding honesty and information value
                - Score 0 — Entirely incorrect: contradicts the docs, fabricates key facts, or asserts "not found" when the information IS present.
                - Score 1 — Mixed or incomplete: contains errors, unsupported claims, or significant omissions of document-backed information needed to answer the question.
                - Score 2 — Mostly correct with minor issues: all major claims are document-supported, but may have small inaccuracies, incomplete coverage, or minor unsupported inferences. The response provides genuine value and correctly identifies what can be determined from the documents.
                - Score 3 — Excellent: provides accurate, complete information based on the documents, clearly explains reasoning, acknowledges limitations, and adds value without hallucination. Also includes responses that correctly state when information cannot be determined AND provide all available related information from the documents. **IMPORTANT: When entities underwent name changes over time, using either the historical or current name (with proper context) qualifies for Score 3 if the entity is correctly identified.**

                CRITICAL "NOT FOUND" HANDLING & ASSOCIATIVE INFERENCE RULES
                - If the information IS present in the documents but the response claims it is not: Score 0–1 (depending on severity).
                - If the information is NOT present and the response correctly says it cannot be determined from the documents: Score 3.
                - **TENSE-SPECIFIC "NOT FOUND" EVALUATION (CRITICAL)**: When a response states "documents do not say [entity] [present verb] [action]", check if documents use PAST tense or indicate ceased/ended status. If so, the response is CORRECT—the present-tense version is not supported even if past-tense IS documented. This is proper premise correction (Score 3), not false claim. Conversely, using present tense when documents show past/ceased = temporal mismatch (Score 1-2).
                - **ASSOCIATIVE INFERENCE SCORING**: When documents explicitly state that multiple entities share a common attribute or action (e.g., "A, B, and C all developed X together"), you MAY reasonably infer that ALL mentioned entities share that attribute. However, when entities are only listed as contemporaries or comparators without explicit shared action (e.g., "A developed X with rigor comparable to B, C, D"), this does NOT automatically mean B, C, D share the attribute or action with A.
                - **Example**: Document says "A developed therapy with rigor comparable to contemporaries including B" → does NOT mean "B was rigorous about A's therapy" or co-developed it → Such inference = Score 0-1.
                - Stating a document's limitation (e.g., "documents do not specify X") is acceptable when the limitation is genuine.

                CONSISTENCY CHECK (before finalizing scores)
                - Ensure neither score was influenced by the other response's content or by a "more natural intent" assumption. Grade only against the documents and the rubric.
                - ⚠️ **VERIFY QUESTION TYPE**: Re-check if this is a COMMONALITY-ONLY question ("What do they share?") or SAMENESS question ("Do they share the SAME X?"). For COMMONALITY-ONLY, do NOT penalize responses that omit differences—they are answering the question correctly by identifying commonalities alone.
                - ⚠️ **ENTITY NAME EVOLUTION CHECK**: When questions ask about entities using founding/establishment dates and documents reveal subsequent name changes, accept both naming conventions. Example: If documents state "ABC Corporation founded 1980, renamed ABC Global 2005" and question asks "which company founded in 1980?", BOTH "ABC Corporation" and "ABC Global (founded as ABC Corporation in 1980)" merit Score 3. The key is correct entity identification with accurate temporal information, not rigid adherence to one naming form.
                - **GEOGRAPHIC COMPARISON RULE**: When comparing distances or locations between entities, you MUST have explicit distance measurements, coordinates, or clear relative positioning information in the documents. General geographic knowledge cannot be used - only document-provided facts about relative positioning or distances.
                
                CRITICAL: COMPLEX SENTENCE STRUCTURE ANALYSIS (MULTI-PART COMPARISONS)
                - **LAYERED STATEMENTS**: When a response has a complex structure like "X and Y are both [core attribute], but they differ in [specific detail]", you MUST parse this as TWO separate claims:
                  1. CLAIM 1 (commonality): "X and Y are both [core attribute]" → They DO share the core attribute
                  2. CLAIM 2 (difference): "they differ in [specific detail]" → Their full profiles differ in some way
                - **DO NOT MISREAD**: A statement acknowledging a shared attribute in the main clause while noting differences in a subordinate clause is NOT denying the commonality. The main clause establishes what IS shared; the subordinate clause adds necessary nuance about what differs.
                - **STRUCTURAL PATTERNS TO RECOGNIZE**:
                  * "Both X and Y are [attribute], but..." → Acknowledges commonality + explains difference
                  * "While X and Y share [attribute], they differ in..." → Acknowledges commonality + explains difference  
                  * "X and Y are [attribute], although..." → Acknowledges commonality + adds qualifier
                  * "No, they are different" (with no acknowledgment of documented commonality) → Incomplete/misleading
                - **EVALUATION PRINCIPLE - DISTINGUISH QUESTION TYPES**:
                  * **COMMONALITY-ONLY questions** ("What do X and Y have in common?", "What profession do they share?"): These ask ONLY for shared attributes. A response accurately identifying the commonality = Score 3. Adding differences is optional, not required.
                  * **SAMENESS/COMPARISON questions** ("Do X and Y share the **same** [attribute]?", "Did they hail from the **same** [attribute A] and [attribute B]?"): These ask whether entities are identical or whether they match on specific attributes. A response must acknowledge BOTH commonalities AND differences when both are documented = Score 3. For conjunctive sameness questions (asking about "same A AND B"), clarify which parts match and which differ. Stating only "Yes" or only "No" without clarifying what matches and what differs = incomplete (Score 1-2).
                - **GENERAL PRINCIPLE**: Match evaluation criteria to the question type. For commonality-only queries, providing accurate shared attributes suffices. For sameness/comparison queries, clarifying both matches and mismatches is essential.
                - **PARSING INSTRUCTION**: Before grading, explicitly identify ALL factual claims in each response, including those in:
                  * Main clauses
                  * Subordinate clauses after "but", "however", "although", "while", "though", etc.
                  * Conditional phrases and qualifiers
                  Grade each claim independently, then evaluate whether the response as a whole provides complete, accurate information.

                OUTPUT FORMAT (exactly)
                Score for {original_label}: <0-3>
                Score for {comparison_label}: <0-3>
                Reason: <concise explanation grounded in the documents and the rules above>
                """




                # prompt = f"""
                # SYSTEM:
                # You are an impartial grader. Follow the instructions exactly.

                # USER:
                # You will compare TWO answers to ONE question using ONLY the provided DOCUMENTS.

                # Definitions:
                # - SUPPORTED: The claim is fully supported by at least one document sentence/segment.
                # - UNSUPPORTED: The documents do not provide enough information to support the claim.
                # - CONTRADICTED: The documents explicitly conflict with the claim.

                # Scoring rules (single 0-3 score per answer):
                # 0 = All factual claims are UNSUPPORTED or CONTRADICTED. No correct supported info.
                # 1 = Mixture: at least one SUPPORTED claim, but there exists any UNSUPPORTED or CONTRADICTED claim.
                # 2 = All claims are SUPPORTED, but the answer misses relevant and valuable details present in the documents.
                # 3 = All claims are SUPPORTED and the answer includes the relevant and valuable details.

                # If the response states something that cannot be found in the documents, treat it as an error (UNSUPPORTED or CONTRADICTED). Apply the 0/1 rule above.

                # Task:
                # 1. Extract factual claims from each answer (ignore greetings/common fluff).
                # 2. For each claim, mark verdict and list supporting doc_ids (empty if none).
                # 3. Compute the score using the rules.
                # 4. Provide a brief reason (<120 words) explaining the score.

                # Return STRICT JSON with this schema (no extra keys, no extra text):

                # Output EXACTLY the following 3 lines (no extra text, no JSON):
                # Score for original response: <0-3>
                # Score for Milo's response: <0-3>
                # Reason: <short explanation within 120 words>

                # CONSTRAINTS:
                # - Use ONLY the DOCUMENTS for judging support.
                # - If unsure, default to UNSUPPORTED.
                # - Keep JSON valid. No trailing commas.

                # <<<QUESTION>>>
                # {question}
                # <<<Original Response>>>
                # {original_response}
                # <<<Milo's Response>>>
                # {milo_response}
                # <<<DOCUMENTS>>>
                # {documents}

                # """
                quality = self.llm.complete(prompt=prompt)
                quality_data.append({
                    "question": question,
                    "original_response": original_response,
                    "milo_response": milo_response,
                    "documents": documents,
                    "quality": quality.text
                })

                if idx % progress_step == 0 or idx == total:
                    pct = idx / total * 100 if total else 100
                    print(f"\rQuality evaluation progress: {idx}/{total} ({pct:5.1f}%)", end="", flush=True)

        if total:
            print()
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            for quality_item in quality_data:
                f_out.write(json.dumps(quality_item, ensure_ascii=False) + '\n')
        print(f"✅ response quality evaluation completed, {len(quality_data)} entries written to {self.output_file}")


# """BELOW IS THE CODE FOR RAGAS EVALUATION"""
# from ollama import chat
# from ollama import ChatResponse
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_correctness
# from ragas.run_config import RunConfig
# import pandas as pd

# if __name__ == "__main__":
#     TIME_TAG = "181125"
#     USER_DATA_FILE = 'evaluation_data_'+TIME_TAG+'.json'
#     BENCHMARK_FILE = 'ragbench_test.jsonl' 
#     OUTPUT_FILE = 'retrieval_analysis_data_'+TIME_TAG+'.jsonl'
#     traces = parse_retrieval_traces_updated("retrieval_traces_test_milo_N-0-FP_131125.txt")
#     # filter_agent = Filter(model_name="gpt-4.1-mini")
#     filter_agent = Filter(model_name="gpt-5-mini", temperature=1)
#     evaluation_data = []
#     for i, trace in enumerate(traces):
#         question = trace["question"]
#         milo_response = trace["response"]
#         retrieved_texts = trace["retrieved_texts"]
#         used_texts = filter_agent.filter_used_texts(question, milo_response, retrieved_texts)
#         print(used_texts)
#         eval_item = {
#                 "id": str(i),
#                 "question": question,
#                 "milo_response": milo_response,
#                 "used_texts": used_texts,
#                 "retrieved_texts": retrieved_texts,
#             }
#         evaluation_data.append(eval_item)
#     with open(USER_DATA_FILE, "w", encoding='utf-8') as f:
#         json.dump(evaluation_data, f, indent=4)

#     """BELOW IS THE CODE FOR RAGAS EVALUATION"""
#     try:
#         with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
#             user_data = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"❌ error: could not load or parse your data file: {e}")
#         exit() # if could not load user data, exit
#     benchmark_lookup = create_benchmark_lookup(BENCHMARK_FILE)
#     if benchmark_lookup:
#         # --- step 3: process data and generate the final evaluation file ---
#         processed_count = 0
#         all_analysis_entries = []
#         all_ragas_eval_data = []
#         print(f"\n--- step 3: start processing data and writing to '{OUTPUT_FILE}' ---")
        
#         with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
#             for entry in user_data:
#                 question = entry.get("question")
#                 retrieved_texts = entry.get("retrieved_texts")
#                 milo_response = entry.get("milo_response")
#                 used_texts = entry.get("used_texts")
#                 if not question or not retrieved_texts or not milo_response or not used_texts:
#                     continue

#                 # find the standard answer in the benchmark dictionary by the question
#                 normalized_question = normalize_question_key(question)
#                 documents_sentences = benchmark_lookup.get(normalized_question).get("documents_sentences")
#                 documents = benchmark_lookup.get(normalized_question).get("documents")
#                 response = benchmark_lookup.get(normalized_question).get("response")
#                 all_relevant_sentence_keys = benchmark_lookup.get(normalized_question).get("all_relevant_sentence_keys")
#                 all_utilized_sentence_keys = benchmark_lookup.get(normalized_question).get("all_utilized_sentence_keys")

#                 if documents_sentences:
#                     # call the core function to restructure the contexts
#                     restructured_contexts = restructure_contexts_like_documents_sentences(
#                         retrieved_texts,
#                         documents_sentences
#                     )
#                     all_retrieved_ids = get_ids_from_structured_contexts(restructured_contexts)


#                     used_texts_structured = restructure_contexts_like_documents_sentences(
#                         used_texts,
#                         documents_sentences
#                     )
#                     used_ids = get_ids_from_structured_contexts(used_texts_structured)
#                     used_texts_texts = get_texts_from_structured_contexts(used_texts_structured)
#                     # prepare the final, informative output entry

                        
                    
#                     retrieval_intersection = len(set(all_relevant_sentence_keys).intersection(set(all_retrieved_ids)))
#                     retrieval_recall = retrieval_intersection / len(all_relevant_sentence_keys) if len(all_relevant_sentence_keys) > 0 else 0
#                     # write the result in JSON line format to the output file

#                     generator_intersection = len(set(all_utilized_sentence_keys).intersection(set(used_ids)))
#                     generator_precision = generator_intersection / len(used_ids) if len(used_ids) > 0 else 0
#                     generator_recall = generator_intersection / len(all_utilized_sentence_keys) if len(all_utilized_sentence_keys) > 0 else 0

#                     # Clean milo_response for RAGAS: remove Supporting Details (citations)
#                     answer_for_ragas = milo_response
#                     if "Supporting Details:" in milo_response:
#                         answer_for_ragas = milo_response.split("Supporting Details:")[0].strip()
#                     if answer_for_ragas.startswith("Direct Answer:"):
#                         answer_for_ragas = answer_for_ragas.replace("Direct Answer:", "").strip()

#                     all_ragas_eval_data.append({
#                         "question": question,
#                         "answer": answer_for_ragas,  # Use cleaned answer without citations
#                         "contexts": used_texts, # We'll use the raw used_texts for faithfulness check
#                         "ground_truth": response # The ground truth answer
#                     })

#                     analysis_entry = {
#                         "id": entry.get("id"),
#                         "question": question,
#                         "documents": documents,
#                         "milo_response": milo_response,
#                         "retrieved_contexts_keys": list(all_retrieved_ids),
#                         "used_texts_keys": list(used_ids),
#                         "used_texts": used_texts_texts,
#                         "response": response,
#                         "all_relevant_sentence_keys": list(all_relevant_sentence_keys),
#                         "all_utilized_sentence_keys": list(all_utilized_sentence_keys),
#                         "retrieval_recall": retrieval_recall,
#                         "generator_precision": generator_precision,
#                         "generator_recall": generator_recall
#                     }

#                     all_analysis_entries.append(analysis_entry)
                    
#                     # f_out.write(json.dumps(analysis_entry, ensure_ascii=False) + '\n')
#                     processed_count += 1
#                 else:
#                     print(question)
#                     print(f"⚠️  warning: question not found in benchmark file: '{BENCHMARK_FILE}'")
#         ragas_run_config = RunConfig(timeout=600, max_retries=5, max_workers=2)
#         if all_ragas_eval_data:
#             ragas_dataset = Dataset.from_pandas(pd.DataFrame(all_ragas_eval_data))
#             ragas_results = evaluate(
#                 ragas_dataset,
#                 metrics=[faithfulness],
#                 run_config=ragas_run_config,
#             )
#             ragas_df = ragas_results.to_pandas()
#             ragas_df = ragas_df.rename(columns={'user_input': 'question'})

#             eval_df = pd.DataFrame(all_analysis_entries)
#             # Merge custom metrics with RAGAS metrics
#             final_eval_df = pd.merge(eval_df, ragas_df[['question', 'faithfulness']], on='question', how='left')
    
#             for i, row in final_eval_df.iterrows():
#                 all_analysis_entries[i]['faithfulness'] = row['faithfulness']
        
#         with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
#             for analysis_entry in all_analysis_entries:
#                 f_out.write(json.dumps(analysis_entry, ensure_ascii=False) + '\n')
#         print(f"\n✅ processing completed, {processed_count} entries written to {OUTPUT_FILE}")

#     """BELOW IS THE CODE FOR RESPONSE QUALITY EVALUATION"""
#     import os
#     # print("API KEY:", os.getenv("OPENAI_API_KEY"))
#     print("Quality evaluation started...")
#     test = ResponseQualityEvaluator(input_file=OUTPUT_FILE, output_file='response_quality_evaluation_'+TIME_TAG+'.json')
#     test.evaluate_response_quality()
#     print("Quality evaluation completed.")
