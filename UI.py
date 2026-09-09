import copy
import pdb
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
import builtins
from llama_index.core.postprocessor import SimilarityPostprocessor
from snippet_postprocessor import RelevantSnippetPostprocessor
import re, uuid
from typing import Optional, Iterable, List, Tuple, Dict, Any
import helper, threading, os, json, math
from helper import output_logger, original_stdout, original_stderr
import core_prompts as core_prompts
import sys
from gradio.themes import Base
import gradio as gr
import time
import traceback
# from helper import TopAgentHelper, check_memory, reset_memory
from helper import add_query_response_to_store, vector_store, persist_directory, compare_query_similarity, \
    persist_directory_summary
# from helper import evaluator
# from helper_131124 import compare_query_similarity, add_query_response_to_store,get_similar_queries, vector_store, persist_directory
from llama_index.core import global_handler, set_global_handler
from helper import LimitedDict
from llama_index.core.agent import ReActAgent
# from llama_index.core.agent.react import ReActAgent

# from llama_index.core.agent.workflow import ReActAgent

from llama_index.core.agent.react.formatter import ReActChatFormatter
# from llama_index.core.agent.lats.formatter import LATSFormatter
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from cost_logging import install_llama_index_cost_logging_from_env

install_llama_index_cost_logging_from_env()
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event, Context
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import FixedRecencyPostprocessor
from llama_index.core.prompts.prompts import PromptTemplate
from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent
)
# import asyncio
from opentelemetry.context import detach, attach, set_value

# Configure LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY via environment variables.
# os.environ[
#     "LANGFUSE_HOST"
# ] = "https://cloud.langfuse.com"  # 🇪🇺 EU region, 🇺🇸 US region: "https://us.cloud.langfuse.com"
# set_global_handler("langfuse")
# langfuse_callback_handler = global_handler
# from crawl4ai import WebCrawler
# from googlesearch import search

# crawler = WebCrawler()
# crawler.warmup()
import datetime
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
import asyncio
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from collections import OrderedDict
from llama_index.core.schema import MetadataMode

config = configparser.ConfigParser()
recall_similarity = 0.85
# config.read('parameters.ini')
# ComplexAgentMaxNum = int(config['CORE_PROMPT']['ComplexAgentMaxNum'])
# SIMILARITY_CUTOFF = float(config['RETRIEVER']['SimilarityCutOff'])

global_original_print = print

config.read('parameters.ini')
# ComplexAgentMaxNum = int(config['CORE_PROMPT']['ComplexAgentMaxNum'])
RelevanceFilterMaxNum = int(
    os.getenv(
        "MILONET_RELEVANCE_FILTER_MAX_NUM",
        config['RELEVANCE_FILTER']['RelevanceFilterMaxNum'],
    )
)
MAX_FLUSH_TOOL_IDS = int(os.getenv("MILONET_MAX_FLUSH_TOOL_IDS", "10"))


class TrackingRetrieverTool(RetrieverTool):
    """RetrieverTool that records document ids back into helper.tool_list."""

    _DOC_ID_PATTERN = re.compile(r"response\s*=\s*([0-9A-Za-z_]+)")
    _ID_KEYS = ("response", "doc_tool_id", "tool_id", "doc_id")
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "with",
        "from", "about", "by", "of", "is", "was", "were", "be", "been", "are",
        "do", "does", "did", "has", "have", "had", "who", "what", "where",
        "when", "which", "whose", "that", "this", "these", "those"
    }

    def __init__(
        self,
        retriever,
        metadata: ToolMetadata,
        node_postprocessors: Optional[list] = None,
    ) -> None:
        super().__init__(
            retriever=retriever,
            metadata=metadata,
            node_postprocessors=node_postprocessors,
        )
        self._pending_ids: list[tuple[str, float, int, int, bool]] = []
        self._call_counter: int = 0
        self._candidate_cache: dict[str, dict[str, Any]] = {}
        self._base_query: str = ""
        self._last_tool_query: str = ""

    @staticmethod
    def _normalize_values(value):
        collected = set()
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                collected.add(trimmed)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                collected.update(TrackingRetrieverTool._normalize_values(item))
        return collected

    @staticmethod
    def _get_node_text(node) -> str:
        txt = getattr(node, "text", None)
        if not txt:
            try:
                txt = node.get_content(metadata_mode=MetadataMode.TEXT)
            except Exception:
                txt = ""
        return txt or ""

    def _collect_from_nodes(self, nodes, keyword_terms: Optional[List[str]] = None):
        keyword_terms = [term.lower() for term in (keyword_terms or []) if term]
        scored_ids: dict[str, tuple[float, bool]] = {}
        for entry in nodes or []:
            node = getattr(entry, "node", None)
            if node is None:
                continue
            score = getattr(entry, "score", None)
            if score is None:
                score = getattr(node, "score", None)
            try:
                score_val = float(score) if score is not None else float("-inf")
            except (TypeError, ValueError):
                score_val = float("-inf")

            node_text = self._get_node_text(node).lower()
            has_keyword = bool(keyword_terms and any(term in node_text for term in keyword_terms))

            for container in (getattr(node, "metadata", None), getattr(node, "extra_info", None)):
                if isinstance(container, dict):
                    for key in self._ID_KEYS:
                        for value in self._normalize_values(container.get(key)):
                            prev = scored_ids.get(value)
                            if prev is None or score_val > prev[0]:
                                scored_ids[value] = (score_val, has_keyword or (prev[1] if prev else False))
                            elif has_keyword and prev is not None and not prev[1]:
                                scored_ids[value] = (prev[0], True)

        if not scored_ids:
            return []

        ordered = sorted(scored_ids.items(), key=lambda item: item[1][0], reverse=True)
        results: list[tuple[str, float, bool]] = []
        for doc_id, (score_val, has_kw) in ordered:
            results.append((doc_id, score_val, has_kw))
        return results

    def _collect_from_content(self, content):
        if not content:
            return []
        seen = set()
        ordered = []
        for match in self._DOC_ID_PATTERN.findall(content):
            value = match.strip()
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    @staticmethod
    def _ensure_prefixed(doc_id):
        if not doc_id:
            return None
        return doc_id if doc_id.startswith("tool_") else f"tool_{doc_id}"

    def _update_helper_tool_list(self, doc_entries: Iterable[tuple[str, float, int, int, bool]]):
        best_scores: dict[str, tuple[float, int, bool]] = {}
        for doc_id, score, call_idx, _rank, keyword_hit in doc_entries:
            prefixed = self._ensure_prefixed(doc_id)
            if not prefixed:
                continue

            raw_id = prefixed.split("tool_", 1)[1] if prefixed.startswith("tool_") else prefixed
            if helper.SHORT_ID_PATTERN.match(raw_id):
                short_id = raw_id
            else:
                short_id = helper.get_shortened_id_from_file(raw_id)

            if short_id:
                canonical = short_id
            else:
                canonical = raw_id

            prev_tuple = best_scores.get(canonical)
            if prev_tuple is None or score > prev_tuple[0]:
                best_scores[canonical] = (score, call_idx, keyword_hit)
            else:
                # accumulate keyword hits even if score lower
                stored_score, _, stored_kw = prev_tuple
                best_scores[canonical] = (stored_score, max(call_idx, prev_tuple[1]), keyword_hit or stored_kw)

        if not best_scores:
            return

        now_call = self._call_counter
        for canonical, (score, call_idx, keyword_hit) in best_scores.items():
            cache_entry = self._candidate_cache.get(canonical, {
                "best_score": float("-inf"),
                "last_call": call_idx,
                "hits": 0,
                "keyword_hits": 0,
            })
            if score > cache_entry["best_score"]:
                cache_entry["best_score"] = score
            cache_entry["last_call"] = max(cache_entry["last_call"], call_idx)
            cache_entry["hits"] += 1
            if keyword_hit:
                cache_entry["keyword_hits"] += 1
            cache_entry["last_score"] = score
            self._candidate_cache[canonical] = cache_entry

        priorities: list[tuple[str, float]] = []
        for canonical, stats in self._candidate_cache.items():
            best_score = stats.get("best_score", float("-inf"))
            if best_score == float("-inf"):
                continue
            hits = stats.get("hits", 0)
            keyword_hits = stats.get("keyword_hits", 0)
            last_call = stats.get("last_call", now_call)
            age = max(0, now_call - last_call)
           
            base_weight = 1.0  
            keyword_boost = 0.1  
            recency_boost = 0.01

            if keyword_hit:
                base_weight *= 1.5

            priority = (
                base_weight * best_score
                + 0.05 * math.log1p(hits)
                + keyword_boost * keyword_hits
                - recency_boost * age
            )
            priorities.append((canonical, priority))

        if not priorities:
            return

        priorities.sort(key=lambda item: item[1], reverse=True)
        top_candidates = [item[0] for item in priorities[:MAX_FLUSH_TOOL_IDS]]

        with helper._tool_list_lock:
            current_list = helper.tool_list if isinstance(helper.tool_list, list) else list(helper.tool_list)
            existing = [item for item in current_list if isinstance(item, str)]
            remaining = [item for item in existing if item not in top_candidates]
            helper.tool_list = (top_candidates + remaining)[:RelevanceFilterMaxNum]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def set_base_query(self, query: str) -> None:
        self._base_query = query or ""

    def _build_keyword_terms(self) -> List[str]:
        terms: set[str] = set()
        base_tokens = self._tokenize(self._base_query)
        if base_tokens:
            terms.update(filter(lambda t: len(t) > 2, base_tokens))
            if len(base_tokens) >= 2:
                terms.add(" ".join(base_tokens[:2]))

        last_tokens = self._tokenize(self._last_tool_query)
        if last_tokens:
            terms.update(filter(lambda t: len(t) > 2, last_tokens))
            if len(last_tokens) >= 2:
                terms.add(" ".join(last_tokens[:2]))

        def _filtered(tokens: List[str]) -> List[str]:
            return [t for t in tokens if t not in self._STOPWORDS and len(t) > 2]

        filtered_base = _filtered(base_tokens)
        filtered_last = _filtered(last_tokens)

        def _add_ngrams(token_list: List[str], max_n: int = 3):
            for n in range(2, max_n + 1):
                if len(token_list) < n:
                    continue
                for i in range(len(token_list) - n + 1):
                    ngram = " ".join(token_list[i:i + n])
                    terms.add(ngram)

        _add_ngrams(filtered_base)
        _add_ngrams(filtered_last)

        # deduplicate while preserving manageable size
        limited_terms = list(terms)
        if len(limited_terms) > 25:
            limited_terms = limited_terms[:25]
        return limited_terms

    def _record_tool_usage(self, tool_output):
        if tool_output is None:
            return
        raw_docs = getattr(tool_output, "raw_output", []) or []
        content = getattr(tool_output, "content", "") or ""
        keyword_terms = self._build_keyword_terms()
        node_entries = self._collect_from_nodes(raw_docs, keyword_terms)
        scored_ids: dict[str, tuple[float, bool]] = {doc_id: (score, flag) for doc_id, score, flag in node_entries}
        for cid in self._collect_from_content(content):
            if cid not in scored_ids:
                scored_ids[cid] = (float("-inf"), False)
        if scored_ids:
            ordered = sorted(scored_ids.items(), key=lambda entry: entry[1][0], reverse=True)
            self._call_counter += 1
            call_index = self._call_counter
            neg_inf = float("-inf")
            debug_entries = [
                doc_id if score[0] == neg_inf else f"{doc_id}@{score[0]:.4f}"
                for doc_id, score in ordered
            ]
            print(f"TrackingRetrieverTool captured doc ids: {debug_entries}")
            for rank, (doc_id, (score_val, keyword_hit)) in enumerate(ordered):
                self._pending_ids.append((doc_id, score_val, call_index, rank, keyword_hit))

    def flush_pending_ids(self, commit: bool = False):
        if not self._pending_ids:
            return
        if commit:
            self._update_helper_tool_list(self._pending_ids)
        self._pending_ids.clear()

    def call(self, *args, **kwargs):  # noqa: D401
        self._capture_query(args, kwargs)
        output = super().call(*args, **kwargs)
        self._record_tool_usage(output)
        return output

    async def acall(self, *args, **kwargs):  # noqa: D401
        self._capture_query(args, kwargs)
        output = await super().acall(*args, **kwargs)
        self._record_tool_usage(output)
        return output

    def _capture_query(self, args, kwargs):
        query_candidate = None
        if args:
            first = args[0]
            if isinstance(first, str):
                query_candidate = first
            elif isinstance(first, dict):
                query_candidate = first.get("input") or first.get("query")
        if query_candidate is None:
            query_candidate = kwargs.get("input") or kwargs.get("query")
        if isinstance(query_candidate, dict):
            query_candidate = query_candidate.get("input") or query_candidate.get("query")
        if isinstance(query_candidate, str):
            self._last_tool_query = query_candidate


class IntermediateEventOne(Event):
    result: list[str]


class IntermediateEventTwo(Event):
    result: list[str]


# class ProcessCombinedResponseLLM:
#     def __init__(self, user_input, combined_response):
#         self.llm = OpenAI(model="o1-mini")
#         # self.react_agent = OpenAIAgent.from_tools(
#         #     [FunctionTool.from_defaults(fn=self.process_raw_CoA)],
#         #     llm=self.llm,
#         #     verbose=True)
#         self.user_input = user_input
#         self.combined_response = combined_response

#     def process_subCoAResponses(self, current_mode_content):
#         prompt = f"""
#         Your task is to provide a clear, direct answer to the user's question based on the information provided by various sub-agents.

#         *CRITICAL INSTRUCTIONS - MANDATORY STEPS*
        
#         1. INFORMATION SYNTHESIS (DO THIS FIRST):
#            - Create a list of all named entities mentioned across ALL sub-agent responses
#            - For each entity, collect ALL attributes and relationships mentioned by ANY sub-agent
#            - When the same name appears in multiple responses, treat it as the same entity and COMBINE all information
#            - Pay special attention to dates, titles, roles, and organizational affiliations
        
#         2. ANSWER DETERMINATION:
#            - Compare your synthesized entity information against the original query components
#            - Look specifically for entities that match BOTH:
#              a) The role/relationship requested (e.g., "founder of X")
#              b) The attribute requested (e.g., "born on date Y")
#            - If you find an entity that satisfies ALL query components across different sub-agent responses, that is your answer
        
#         3. HALLUCINATION PREVENTION:
#            - Only state information explicitly mentioned in at least one sub-agent response
#            - Never add details or connections that aren't clearly stated
#            - For any answer, you must be able to point to specific statements in the sub-agent responses
        
#         4. RESPONSE FORMAT:
#            - Provide ONLY the factual answer - nothing more, nothing less
#            - DO NOT mention sources, references, or agents
#            - DO NOT explain your reasoning or process
#            - If no entity satisfies ALL query components, state "No information is available about [specific topic]"

#         *Your OPERATION MODE*
#         {current_mode_content}

#         *Original QUERY* 
#         {self.user_input}

#         *Information Available* 
#         {self.combined_response}
#         """
#         response = self.llm.complete(prompt=prompt)
#         return response.text
from ollama import chat
from ollama import ChatResponse

class ProcessCombinedResponseLLM:
    def __init__(
        self,
        user_input,
        combined_response,
        synthesis_mode: str = "full",
        enable_final_checks: bool | None = None,
    ):
        synthesis_mode = synthesis_mode.strip().lower()
        if synthesis_mode not in {"core", "full"}:
            raise ValueError(f"Unsupported synthesis_mode={synthesis_mode!r}; expected 'core' or 'full'.")
        self.llm = OpenAI(model="o4-mini", timeout=500, max_retries=2)
        self.validator_llm = OpenAI(model="gpt-5-mini", timeout=120, max_retries=1)
        self.validator_extractor_llm = OpenAI(model="gpt-5-nano", timeout=120, max_retries=1)
        self.user_input = user_input
        self.combined_response = combined_response
        self.synthesis_mode = synthesis_mode
        self.enable_final_checks = synthesis_mode == "full" if enable_final_checks is None else enable_final_checks
        self._doc_entries: List[Dict[str, str | int]] = []
        self._doc_entry_map: "OrderedDict[str, List[Dict[str, str | int]]]" = OrderedDict()
        self._last_attribute_keyword: Optional[str] = None
        self._fact_sentence_map: Dict[str, List[str]] = {}
        self._scope_validation_applied: bool = False  # Track if scope validator has rewritten the answer

    _DOC_ID_PATTERN = re.compile(r"[0-9a-f]{4,}_[0-9]+", re.IGNORECASE)
    
    def _save_extracted_facts(self, extracted_facts: str):
        """Save extracted facts to a JSON file for later comparison."""
        from datetime import datetime
        import json
        import os
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%d%m%y")
        filename = f"extracted_facts_{timestamp}.jsonl"
        
        # Prepare data entry
        entry = {
            "question": self.user_input,
            "extracted_facts": extracted_facts,
            "combined_response": self.combined_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to JSONL file (one JSON object per line)
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"✅ Extracted facts saved to {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save extracted facts: {e}")

    def _normalize_output_line(self, raw_line: str) -> str:
        stripped = raw_line.strip()
        if not stripped:
            return ""
        stripped = re.sub(r"\s+", " ", stripped)
        if stripped.startswith("-"):
            return "-" + stripped[1:].lstrip()
        return f"- {stripped.lstrip('-• ')}"

    def _build_doc_entries(self, combined_response: str):
        entries: List[Dict[str, str | int]] = []
        doc_map: "OrderedDict[str, List[Dict[str, str | int]]]" = OrderedDict()
        if not combined_response:
            self._doc_entries = entries
            self._doc_entry_map = doc_map
            return

        active_doc: Optional[str] = None
        doc_line_counter: Dict[str, int] = {}

        subcoa_pattern = re.compile(r"^\[SubCoA_[^]]+\]\s*=>\s*(.+)$")
        bullet_subcoa_pattern = re.compile(r"^-+\s*\[SubCoA_[^]]+\]\s*=>\s*(.+)$")

        for raw_line in combined_response.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                continue

            match = subcoa_pattern.match(stripped)
            bullet_match = bullet_subcoa_pattern.match(stripped)
            if bullet_match:
                stripped = bullet_match.group(1).strip()
                raw_line = f"- {stripped}"
            elif match:
                stripped = match.group(1).strip()
                raw_line = stripped

            doc_matches = self._DOC_ID_PATTERN.findall(stripped)
            if doc_matches:
                active_doc = doc_matches[0]
            if not active_doc:
                continue

            doc_line_counter.setdefault(active_doc, 0)
            doc_line_counter[active_doc] += 1
            entry_id = f"{active_doc}#{doc_line_counter[active_doc]}"

            prompt_text = stripped if doc_matches else f"{active_doc}: {stripped}"
            normalized_line = self._normalize_output_line(raw_line)
            if not normalized_line:
                continue

            entry = {
                "uid": entry_id,
                "doc_id": active_doc,
                "index": doc_line_counter[active_doc],
                "raw_line": raw_line,
                "prompt_text": prompt_text,
                "normalized_line": normalized_line,
            }
            entries.append(entry)
            doc_map.setdefault(active_doc, []).append(entry)

        self._doc_entries = entries
        self._doc_entry_map = doc_map

    def extract_positive_facts(self, combined_response, user_input):
        """
        Extract positive factual statements by asking the LLM which sentences to discard.
        """
        self._build_doc_entries(combined_response)
        entries = self._doc_entries
        doc_map = self._doc_entry_map

        if not entries:
            return ""

        batch_payload = [
            {"id": entry["uid"], "doc_id": entry["doc_id"], "text": entry["prompt_text"]}
            for entry in entries
        ]

        instructions = {
            "objective": (
                "You will receive a list of sentences extracted from document analysis. "
                "Each item has an id, the document id, and the sentence text. "
                "Your job is ONLY to identify sentences that are purely negative disclosures (for example, sentences that ONLY say 'no information was found' or 'this document does not contain relevant details'). "
                "If a sentence contains ANY affirmative fact, contextual detail, or possible relevance to the query, you must keep it. "
                "CRITICAL: Sentences describing disease/condition presentations across the COMPLETE severity spectrum MUST ALWAYS BE KEPT: (a) minimal/absent presentations, (b) typical presentations, and (c) rare but severe outcomes. All severity levels are affirmative medical facts, not absences of information. "
                "Err on the side of keeping sentences; only flag the ones that explicitly state the absence of information."
            ),
            "output": (
                "Return ONLY a JSON array listing the ids of sentences that should be removed. "
                "Example: ['doc1#2', 'doc3#5']. If every sentence should be kept, return an empty array []. "
                "Do not include any extra fields or explanations."
            ),
            "items": batch_payload,
        }

        prompt = json.dumps(instructions, ensure_ascii=False, indent=2)
        try:
            response = self.llm.complete(prompt=prompt)
            raw = (response.text or "").strip()
        except Exception as e:
            print(f"Sentence filtering failed, keeping all lines: {e}")
            raw = ""

        remove_ids: set = set()
        try:
            payload = self._extract_json_block(raw) or raw
            # pdb.set_trace()
            data = json.loads(payload)
            if isinstance(data, dict):
                # Accept keys like {"remove":["id1","id2"]}
                data = data.get("remove") or data.get("ids") or data.get("items") or []
            if not isinstance(data, list):
                raise ValueError("Sentence filtering response must be a list.")
            for element in data:
                if isinstance(element, str):
                    remove_ids.add(element.strip())
                elif isinstance(element, dict):
                    entry_id = element.get("id")
                    if entry_id:
                        remove_ids.add(str(entry_id).strip())
        except Exception as e:
            print(f"Error parsing sentence removal response: {e}; keeping all lines.")
            remove_ids = set()

        output_lines: List[str] = []
        seen: set = set()
        for entry in entries:
            if entry["uid"] in remove_ids:
                continue
            normalized_line = entry["normalized_line"]
            if normalized_line and normalized_line not in seen:
                output_lines.append(normalized_line)
                seen.add(normalized_line)

        return "\n".join(output_lines)



    def _llm_extract_positive_facts(self, combined_response: str, user_input: str) -> str:
        """Legacy helper retained for backward compatibility."""
        return combined_response

    def _fallback_extract_positive_facts(
        self,
        combined_response: str,
        user_input: str,
        allow_raw_return: bool = True,
    ) -> str:
        """Secondary pass that asks the LLM to recover positive facts from mixed sentences."""

        prompt = f"""
        Re-read the bullet-style notes under *Information Available*. For each bullet that contains any affirmative factual claim relevant to any parts of any entity in the *Original Query* (even it is not the main subject of the query), copy the full original sentence(s) that express that fact. Also copy any explicit limitations that appear in the same bullet or immediately attached sentence. Do not paraphrase or shorten; keep tool references, names, titles, quoted works, locations, and qualifiers exactly as written. When a sentence uses pronouns (e.g., "she", "he", "they", "it", "this", "that", "these", "those") to refer to any entity named earlier in the bullet, include the nearest preceding text that states the entity's full name so the fact is self-contained.
        - Always extract every positive sentence verbatim; if the bullet also includes a limitation, put that text into "limitation".
        - If you are unsure whether a sentence is relevant, KEEP IT.
        - Never drop a fact just because it says "no connection to the query" afterward.
        - If the sentence names or describes any work, event, film, episode, organisation, or other entity that is linked anywhere in the bullet set to the query subject, you MUST keep it even if that sentence does not repeat the subject's name. Such contextual sentences are always relevant.
        - **CRITICAL FOR MEDICAL/CLINICAL QUERIES:** Sentences describing disease/condition presentations across the COMPLETE severity spectrum are affirmative medical facts and MUST ALWAYS BE EXTRACTED: (a) minimal/absent presentation patterns, (b) typical/common presentation patterns, and (c) rare but severe outcome patterns. All three severity levels are medically critical - they describe the complete clinical spectrum from mildest to most severe.

        **COMPLETE ANSWER RULE:** If a sentence provides any related information about the Original Query (e.g., confirming identity, roles, relationships, locations, timelines, etc.), you MUST include it along with its limitation. Do NOT remove a fact simply because it does not answer every component of the query.

        Output requirements:
        - Return ONLY valid JSON representing a list of objects.
        - Each object must contain the keys:
          * "fact": the exact text of the sentence(s) containing the affirmative fact, including leading context such as "According to [Document ID]" if present, and any preceding wording needed to replace pronouns with the full name of the referenced entity.
          * "limitation": the exact text of any limitation/uncertainty phrases tied to that fact within the same bullet; use "" if none.
          * "source_reference": the citation identifier extracted from the fact text, following this priority: (1) COMPLETE Document ID (e.g., "e11a1fa0_3", "d5f68839_4") - ALWAYS PREFER THIS; (2) Tool ID (e.g., "tool_e11a1fa0") - ONLY if no Document ID is available. If the fact shows "e11a1fa0_3:", use "e11a1fa0_3" as the source_reference. If multiple identifiers apply, list them comma-separated; if none appear, use "".
        - When a Document ID exists in the fact text, ensure the `fact` string itself begins with the same complete Document ID followed by a colon so downstream consumers can cite it directly.
        - Preserve the original wording; do not invent, summarize, or drop details.
        - Ignore bullets that contain zero affirmative facts.
        - Discard any entry whose affirmative portion depends solely on "Additional Focus".
        - Retain all partial facts from tool outputs about queried entities or descriptors (identity, roles, relationships, locations, timelines, etc.) exactly as written, even if they only answer part of the question.
        - If nothing affirmative is present, return an empty JSON list `[]`.

        *Original Query*
        {user_input}

        *Information Available*
        {combined_response}
        """

        try:
            response = self.llm.complete(prompt=prompt)
            raw_text = response.text.strip()
            if not raw_text:
                return ""

            structured_output = self._parse_structured_fact_output(raw_text)
            if structured_output:
                return structured_output

            return combined_response if allow_raw_return else ""
        except Exception as e:
            print(f"Error during fallback extraction: {e}")
            return ""

    def _parse_structured_fact_output(self, raw_text: str) -> str:
        """Parse JSON output of fact/limitation pairs and render bullet text."""

        json_payload = self._extract_json_block(raw_text)
        if json_payload is None:
            return ""

        try:
            data = json.loads(json_payload)
        except Exception:
            return ""

        if isinstance(data, dict):
            if "facts" in data and isinstance(data["facts"], list):
                data = data["facts"]
            else:
                return ""

        if not isinstance(data, list):
            return ""

        lines: list[str] = []
        fact_counter = 1
        for entry in data:
            if not isinstance(entry, dict):
                continue

            fact_text = entry.get("fact", "") if entry.get("fact") is not None else ""
            limitation = entry.get("limitation", "") if entry.get("limitation") is not None else ""

            fact_text = fact_text.strip()
            limitation = limitation.strip()

            if not fact_text:
                continue

            lines.append(f"- Fact {fact_counter}: {fact_text}")
            if limitation:
                lines.append(f"  Limitation: {limitation}")
            fact_counter += 1

        return "\n".join(lines)

    def _build_fact_sentence_map(self, facts_text: str) -> None:
        """Cache fact sentences by document id for downstream coverage checks."""

        self._fact_sentence_map = {}
        if not facts_text:
            return

        fact_prefix = re.compile(r"^-+\s*Fact\s+\d+\s*:\s*", re.IGNORECASE)

        for raw_line in facts_text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.lower().startswith("limitation:"):
                continue

            doc_ids = self._DOC_ID_PATTERN.findall(stripped)
            if not doc_ids:
                continue

            cleaned = fact_prefix.sub("", stripped).lstrip("- ").strip()
            if not cleaned:
                continue

            for doc_id in doc_ids:
                sentence_text = cleaned
                prefix = f"{doc_id}:"
                if sentence_text.lower().startswith(prefix.lower()):
                    sentence_text = sentence_text[len(prefix):].strip()

                if not sentence_text:
                    continue

                sentence_list = self._fact_sentence_map.setdefault(doc_id, [])
                if sentence_text not in sentence_list:
                    sentence_list.append(sentence_text)

    def _extract_json_block(self, raw_text: str) -> Optional[str]:
        """Attempt to isolate a JSON array/dict from the raw model output."""

        text = raw_text.strip()
        if not text:
            return None

        if text.startswith("```"):
            fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if fence_match:
                return fence_match.group(1).strip()

        if text[0] in "[{":
            return text

        json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1).strip()
            if candidate and candidate[0] in "[{" and candidate[-1] in "}]":
                return candidate

        return None

    def process_subCoAResponses(self, current_mode_content):

        extracted_facts = self.extract_positive_facts(self.combined_response, self.user_input)
        print("self.user_input")
        print("initial extracted:", extracted_facts)
        print(self.user_input)
        if extracted_facts and extracted_facts.strip():
            refined = self._fallback_extract_positive_facts(
                extracted_facts, self.user_input, allow_raw_return=False
            )
            if refined and refined.strip() in {"[]", "[ ]", "{}", "{ }"}:
                refined = ""
            if refined and refined.strip():
                if extracted_facts and extracted_facts.strip():
                    extracted_facts = f"{extracted_facts.strip()}\n{refined.strip()}"
                else:
                    extracted_facts = refined.strip()
            else:
                extracted_facts = extracted_facts.strip()

        if extracted_facts and extracted_facts.strip():
            baseline_facts = extracted_facts.strip()
            refined = self._fallback_extract_positive_facts(
                baseline_facts, self.user_input, allow_raw_return=False
            )
            if refined and refined.strip() in {"[]", "[ ]", "{}", "{ }"}:
                refined = ""

            if refined and refined.strip():
                refined_stripped = refined.strip()
                refined_lines = [line.strip() for line in refined_stripped.splitlines() if line.strip()]
                baseline_lines = [line.strip() for line in baseline_facts.splitlines() if line.strip()]

                refined_doc_ids = set(self._DOC_ID_PATTERN.findall("\n".join(refined_lines)))
                missing_lines: list[str] = []
                for line in baseline_lines:
                    doc_ids = self._DOC_ID_PATTERN.findall(line)
                    if not doc_ids:
                        continue
                    if any(doc_id not in refined_doc_ids for doc_id in doc_ids):
                        missing_lines.append(line)

                if missing_lines:
                    merged_lines = refined_lines[:]
                    fact_pattern = re.compile(r"-\s*Fact\s+(\d+)\s*:", re.IGNORECASE)
                    max_fact_num = 0
                    for line in merged_lines:
                        match = fact_pattern.match(line)
                        if match:
                            try:
                                max_fact_num = max(max_fact_num, int(match.group(1)))
                            except ValueError:
                                continue

                    next_fact_num = max_fact_num + 1 if max_fact_num else 1

                    for line in missing_lines:
                        if line in merged_lines:
                            continue
                        cleaned = line.lstrip("- ").strip()
                        if not cleaned:
                            continue
                        if fact_pattern.match(line):
                            merged_lines.append(line)
                        else:
                            merged_lines.append(f"- Fact {next_fact_num}: {cleaned}")
                            next_fact_num += 1
                    extracted_facts = "\n".join(merged_lines)
                else:
                    extracted_facts = "\n".join(refined_lines)
            else:
                extracted_facts = baseline_facts

        if not extracted_facts or not extracted_facts.strip():
            fallback_facts = self._fallback_extract_positive_facts(
                self.combined_response, self.user_input, allow_raw_return=True
            )
            if fallback_facts:
                print("Positive fact extraction fallback engaged")
                extracted_facts = fallback_facts
            else:
                print("Fallback could not isolate positive facts; using combined response")
                extracted_facts = self.combined_response

        self._build_fact_sentence_map(extracted_facts)
        
        # Save extracted_facts to file for later comparison (MiloNet vs MiloNet-no-post-FP)
        self._save_extracted_facts(extracted_facts)
        
        # user_data_and_query = f"""
        # **Core Mission: You are a Factual Synthesiser. Your sole purpose is to answer the *Original QUERY* with extreme precision, using ONLY the facts provided in the *Information Available*.**
        
        # ## The "Helpful Alternative" Principle (Overrides Conflicting Rules)
        #  - **This is your primary directive for handling near-misses.** When a direct, perfect answer is not found because of a semantic mismatch (e.g., "business" vs. police officer") or an entity name mismatch (e.g., "Main High School" vs. "Main High School South"), you **MUST NOT** simply state that the information is unavailable.
        #  - Instead, your `Direct Answer` **MUST** follow this exact two-part structure:
        #     1.  **State the Limitation:** Begin by truthfully stating that the specific term was not found (e.g., "The available information does not specify a business...").
        #     2.  **Provide the Alternative:** Immediately follow with "However," and present the highly relevant fact, clarifying the mismatch (e.g., "...however, his confirmed subsequent role was as an police officer.").
        #  - This structure allows you to be both precise (by acknowledging the limitation) and helpful (by providing the user's intended answer).
        #  - **CRITICAL EXCEPTION:** This principle is completely overridden by the MANDATORY LOGICAL SYNTHESIS rule. If a direct answer can be constructed through logical synthesis, you are strictly forbidden from using the "Helpful Alternative" structure. Synthesis takes absolute precedence over stating a limitation.
        user_data_and_query_vanilla =f"""
            You are a concise factual answering assistant.

            You will be given:
            - a user question
            - a list of VERIFIED FACTS that were extracted verbatim from source documents.

            Your rules:
            - Use ONLY the VERIFIED FACTS to answer the question.
            - Prefer to copy short spans directly from the facts rather than paraphrasing.
            - Do NOT introduce any new entities, dates, numbers, locations, or qualitative relations
            (such as "near", "earlier than", "more common", "in the same area") unless those
            exact words or relations already appear in the facts.
            - If the VERIFIED FACTS do not contain enough information to answer the question,
            answer exactly: "The documents do not specify this."

            Question:
            {self.user_input.strip()}

            VERIFIED FACTS:
            {extracted_facts}

            Direct Answer:"""
        # user_data_and_query_vanilla =f"""
        # You are a concise factual answering assistant.

        # You will be given:
        # - a user question
        # - some information extracted from documents.

        # Rules:
        # - Use only this information to answer the question.
        # - Do not rely on outside knowledge.
        # - If the information is insufficient, reply exactly:
        # "The documents do not specify this."

        # Question:
        # {self.user_input.strip()}

        # Information:
        # {extracted_facts}

        # Direct Answer:
        # """
        user_data_and_query = f"""
        **Core Mission: You are a Factual Synthesiser. Your sole purpose is to answer the *Original QUERY* with extreme precision, using ONLY the facts provided in the *Information Available*.**

        **CRITICAL RULE 1: MANDATORY DEFINITIONAL LOGIC FOR COMPARISONS**
        - When comparing entities where facts describe inherent capability/design differences (armed vs unarmed, combat vs non-combat role, operational vs non-operational), you MUST apply definitional logic.
        - VIOLATION: Saying "cannot be determined" when entity definitions themselves answer the comparison.
        - If facts describe Entity A as designed/equipped for an activity and Entity B as explicitly not designed/equipped for that activity, the answer is determined by definition.
        
        **CRITICAL RULE 2: NEVER INFER CATEGORICAL MEMBERSHIP**
        - FORBIDDEN: Stating entities share a category (industry/field) unless documents EXPLICITLY name that category.
        - Role descriptors alone do NOT establish shared category membership.

        ## PRIMARY DIRECTIVE: BEST EVIDENCE SYNTHESIS (Absolute Highest Priority)
         - **This is your absolute highest priority rule, overriding all others.** Your primary task is to construct the answer by first identifying the single best piece of evidence for each logical component of the query, and then synthesizing them.
         - **CRITICAL SEMANTIC INTERPRETATION FOR CREATIVE PROFESSIONALS (Apply FIRST):**
            - **When the query asks about "creative titles" for creative professionals (film directors, writers, authors, musicians, actors), this means SPECIFIC WORKS they created (films, books, albums, songs, plays), NOT job titles or professional roles.**
            - **FORBIDDEN**: Counting "director" and "film director" as two separate "creative titles" - these are job titles, not creative works.
            - **REQUIRED**: Count specific named works like "The Great Gatsby", "Abbey Road", "Good Morning", etc.
            - **Action**: When you see "creative titles" + creative professional, immediately search for specific work names (film titles, book titles, album names) in the facts, NOT occupation descriptions.
         - **Workflow:**
            1.  **Deconstruct Query:** First, mentally break the user's query into its logical parts (e.g., Part 1: "Who is the author of book A?", Part 2: "What city did that author of book A live in?"). For queries with embedded constraints (e.g., "which X located in Y" or "what attribute of entity in location Z"), recognize this as multi-hop: identify the entity satisfying the constraint first, then find its requested attributes.
            2.  **Find Best Evidence for Each Part:** For each logical part, scan ALL available facts and identify the ONE fact that provides the most **direct, explicit, and forceful** statement. A direct statement ("B lived in London") is fundamentally superior to an indirect or inferred one ("B had some activities in London, which indicates B lived there").
            3.  **Synthesize the Best:** Construct your final answer by combining these "best facts," citing each one. This is the only acceptable form of synthesis.
            4.  **COMPLETENESS CHECK (CRITICAL):** After finding the best evidence for each query part, scan the facts again to identify any **additional directly relevant attributes** that provide important context or specificity to fully answer the query. Ensure these complementary facts are included in your Supporting Details to provide a complete picture.
               - **Example:** If you've identified an intermediate entity that satisfies the query's constraints, you MUST ensure all available facts about that entity's attributes that directly answer what the query is asking for are included in your Supporting Details, even if not all are used in the Direct Answer.
         - **CRITICAL:** You are **strictly forbidden** from choosing a weaker, indirect fact just because it comes from the same source as another piece of evidence. Your duty is to assemble the strongest possible answer, regardless of how many sources you need to combine. The weaker one only can be considered as supplementary details when there are NO direct statements found.

        ## FALLBACK DIRECTIVE: The "Helpful Alternative" Principle
         - **Use this directive ONLY if the PRIMARY DIRECTIVE (Best Evidence Synthesis) is NOT possible.** 
         - When a direct, perfect answer cannot be synthesized or found in a single fact because of a semantic mismatch (e.g., "business" vs. "police officer") or an entity name mismatch, you **MUST NOT** simply state that the information is unavailable.
         - Do **NOT** invoke this fallback if any fact already states the exact attribute requested about the subject; a disclaimer or limitation in the same fact does **not** erase the affirmative answer.
         - Instead, your `Direct Answer` must open with the strongest relevant affirmative fact that can guide the user, immediately followed by a clause that truthfully states the limitation.
         - Deliver the limitation in the same sentence or the next sentence, but never before the affirmative fact—keep the user-focused information upfront while still signalling the gap.

        ## CRITICAL ENFORCEMENT (Highest Priority Rules):
         - **BEST EVIDENCE PRECEDENCE RULE (Overrides Synthesis Convenience):** When synthesizing an answer, you MUST prioritize the most **direct and explicit** factual statement for each logical component of the query, even if it requires combining facts from different sources.
            - **Direct Fact:** A statement like "A lived in place B."
            - **Indirect Fact:** A statement like "...which indicates that A chose to live in place B."
            - **RULE:** If one fact directly establishes a subject-to-context link (e.g., author of a journal), and a *separate* fact from another source provides a more direct statement for the final attribute (e.g., the author's location), you **MUST** combine them. Do not discard the more direct fact in favor of a less direct one simply because the less direct one is part of a self-contained chain from a single source.
            - **HANDLING LIMITATIONS (Revised Rule):** You must distinguish between two types of limitations:
               - **1. Resolvable Linking Limitations:** These state that a connection between facts is missing (e.g., "but there is no connection established between this person and the event in question").
               - **2. Intrinsic Factual Limitations:** These qualify the fact itself by providing context about scope, time, or certainty (e.g., "this was true only in 1989," "this applied only to the London branch," or "sources suggest this is likely").
               - **RULE FOR RESOLVABLE LIMITATIONS:** If, and only if, you use another fact to create a synthesis that **explicitly resolves** a Linking Limitation, you should ignore that now-obsolete limitation in your final answer.
               - **RULE FOR INTRINSIC LIMITATIONS:** You **MUST ALWAYS PRESERVE** Intrinsic Factual Limitations in your final answer. They are a critical part of the fact and must not be discarded.
         - **NO SELF-CONTRADICTION RULE (PRIME DIRECTIVE):** You are strictly forbidden from making statements that contradict facts you have already presented in the same thought process. If you list a set of facts, you CANNOT follow it with a summary statement that denies or ignores those very facts. Detailed factual lists always take precedence.
            - **MASTER RULE OF SYNTHESIS:** Your primary logical task is to connect facts through shared named entities (the 'context'). If Fact A links a Subject ('S') to a Context ('C'), and Fact B describes an Attribute ('A') of that same Context ('C'), you MUST conclude that the Subject ('S') is associated with that Attribute ('A').
                - **Logical Structure:** (S → C) + (C → A) ==> (S → A)
                - **This applies universally to all domains:**
                    - **People & Works:** (Actor → Movie) + (Movie → Genre) ==> (Actor → Genre)
                    - **People & Groups:** (Musician → Band) + (Band → Location) ==> (Musician → Location)
                    - **Products & Companies:** (Car Model → Brand) + (Brand → Country of Origin) ==> (Car Model → Country of Origin)
            - **TIME SCOPE REQUIREMENT:** Only apply this chain when the fact describing Attribute ('A') explicitly covers the same timeframe (or clearly states an always-true status) that is relevant to the Subject's involvement. If the attribute fact lacks timeframe information or references a period that might not overlap with the Subject's tenure, treat the Subject → Attribute link as **UNCONFIRMED** and state the limitation.
            - **CRITICAL COUNTER-EXCEPTION:** Do NOT apply the above chain when the Attribute ('A') describes the ongoing status or classification of the Context ('C') but the query asks about the Subject's own participation in that status (e.g., person's league, competition, award). Unless a single fact explicitly states that the Subject themselves held that status (with matching timeframe), you MUST report it as unspecified. Example: "Player X → Club Y" + "Club Y plays in League Z" does **NOT** permit "Player X plays in League Z" unless a fact explicitly links Player X to League Z or confirms Club Y's participation in League Z during Player X's tenure.
            - **STRICT PROHIBITION:** You are forbidden from claiming a link is missing if this S → C → A logical chain exists. The shared context 'C' IS the explicit link.
         - **CITATION RULE (CRITICAL):** Every supporting fact you reference (including inside "Supporting Details") must cite its originating identifier exactly as shown in *Information Available*, following this strict priority:
            - **PRIORITY 1 (ALWAYS PREFER):** Complete Document ID with suffixes (e.g., `e11a1fa0_3`, `d5f68839_4`, `ac27fcab_2`)
            - **PRIORITY 2 (ONLY IF NO DOCUMENT ID):** Tool reference (e.g., `tool_e11a1fa0`) - use this ONLY when no Document ID is available
            - **FORBIDDEN:** Never refer to evidence generically as "Fact 1", "Fact 2", etc.
         - **DIRECT ANSWER STYLE RULE:** Deliver the Direct Answer as clean prose without embedding tool/folder/doc identifiers; reserve explicit citations for the Supporting Details section.
         - **DIRECT ANSWER NEVER CONTAIN or USE UNCITED FACTS:** Your final answer (including supporting details) NEVER contain any uncited facts or information from external sources. Even if you think the fact is common knowledge, you MUST NOT use it, ONLY use the cited facts supported by the provided documents.
        - **SUPPORTING DETAILS VERBATIM REQUIREMENT (CRITICAL):** Supporting details MUST be reproduced verbatim from the *Information Available* section. You MUST NOT rephrase, reword, add qualifiers, or modify the original document text in any way. Copy the exact text as provided, including all original terminology and phrasing.
        - **FACTS VERBATIM REQUIREMENT (CRITICAL):** When generating or processing facts, you MUST preserve the exact original wording from documents. You MUST NOT add qualifiers like "film" to terms like "director" unless the original document explicitly uses those qualifiers.
         - **NO UNAUTHORIZED TERM NARROWING IN COMPARISONS (CRITICAL):** When stating what entities share in common, you MUST NOT add limiting qualifiers that narrow the scope of terms. VIOLATION: If Entity A is described with "[qualifier] [base term]" and Entity B is described with only "[base term]" (without the qualifier), you CANNOT say "both are [qualifier] [base term]" - this inappropriately narrows Entity B's term by adding a qualifier that wasn't in the original. CORRECT: Either use the broader unqualified term that appears in both ("both are [base term]"), use a valid generalization that encompasses both original terms, or state them separately. Note: Using a broader category term that encompasses different specific roles (e.g., "[role A]" + "[role B]" → "[common category]") is ALLOWED and appropriate.
         - **BRIDGE DISCLOSURE RULE:** When the query identifies the target indirectly (through a role, work, affiliation, location, descriptor, or relationship such as "author of", "company that operates", "city where"), you MUST explicitly restate the fact that ties that descriptor to the concrete entity before relying on any downstream attributes. This bridge fact must appear in the Direct Answer or Supporting Details with its proper citation (Document ID preferred, or tool ID if no Document ID available) so the reasoning chain remains visible.
         - **STRICT SCOPE SEPARATION (CRITICAL COUNTER-RULE): This rule prevents faulty generalization.** If the available facts describe multiple distinct groups, events, or entities that share a general term (e.g., "Communist meetings") but have different and **mutually exclusive qualifiers** (e.g., "from Allied countries" or "countries opposing the Central Powers" vs. "from neutral countries"), you **MUST treat them as separate and distinct entities.**
            - **RULE:** You are **strictly forbidden** from merging attributes across these distinct scopes. Do not take a specific detail (like a country name) linked to one group and apply it to the other group or to the general term as a whole.
            - **EXAMPLE OF FORBIDDEN ACTION:**
                - Fact A: "Communist meetings of **Allied** powers, including Russia and Cuba, opposed the Central Powers."
                - Fact B: "Communist meetings of **neutral** countries were held, but no countries are named."
                - **INCORRECT SYNTHESIS (FORBIDDEN):** "Communist meetings were held with Russia and Cuba." (This incorrectly applies Allied country names to all meetings).
            - **CORRECT HANDLING:** You MUST present these two groups distinctly in your answer, acknowledging their different compositions and stated goals.
        - If one or more extracted facts explicitly confirm the names of countries involved, clearly state those country names in your "Direct Answer".
        - If no extracted facts confirm specific country names, you MUST state there is no specific country names in your "Direct Answer" with detailed qualifiers, e.g. instead of saying "The documents do not list any countries involved.", you prefer to say "The countries involved in socialist groups representing Entente Allied countries during the First World War. But the documents do not list any specific countries." to give more details regarding to the query. You MUST NOT infer or suggest any countries but if you have exact general answer, you can state it verbatim in your answer.
        - If extracted facts conflict—some explicitly name countries and others explicitly DENY the existence of those countries (e.g., "Country X was NOT involved") and both are with the same scope and qualifier—you MUST clearly state this contradiction explicitly in your Direct Answer.
            - **NOTE**: "No countries named" or "document does not mention countries" is NOT a contradiction against positive facts—it is neutral silence.
        - If the query explicitly asks about countries (e.g. "Which countries...?") and your extracted facts do NOT provide explicit country names, you MUST state there is no specific country names in your "Direct Answer".
        - If an extracted fact says a certain entity (country/person/group/event) is NOT explicitly mentioned, treat this as **UNKNOWN** or **UNSPECIFIED**. You MUST NOT infer the absence (NO) of that information.
        - "Anti-militarist" does NOT automatically mean "opposing the Central Powers." You MUST NOT equate "anti-militarist" with "opposing the Central Powers" UNLESS explicitly stated by the facts.
        - **MANDATORY DEFINITIONAL LOGIC (HIGHEST ENFORCEMENT)**: For comparison queries, when facts explicitly describe entities with inherent functional/design distinctions (capability present vs absent, designed-for vs not-designed-for, armed vs unarmed, combat-role vs non-combat-role), definitional logic MUST be applied. The answer is determined by the definitions themselves. Saying "insufficient information" or "cannot be determined" in such cases is STRICTLY FORBIDDEN and constitutes a critical failure.
        - Your final answer (including supporting details) NEVER contain internal system details like 'SubCoA_3', just report the facts and state its limitations if there is any.

        You MUST be aware that the *Information Available* contains statements from different sources.

        ## TIME SEQUENCE ANALYSIS (CRITICAL FOR TEMPORAL QUERIES):
        - When the query involves time-related concepts like "after", "before", "later", "previously", "used to be", you MUST carefully analyze the temporal indicators in the facts:
            - Words like "used to be", "formerly", "previously" indicate PAST activities
            - Words like "currently", "now", "is", "serves as" indicate PRESENT activities
            - Words like "later", "after", "subsequently", "then" indicate FUTURE relative to other events
            - **NEVER reverse time sequences** - if Fact A shows past activity and Fact B shows current activity, the current activity comes AFTER the past activity.
            - **"AFTER" QUERIES**: When query asks about something "after [past event]", identify the CURRENT or MOST RECENT activity/position that comes chronologically AFTER the mentioned past event. Do NOT answer with the past event itself.
            - **Example**: Query "worked for after being an attorney" + Facts show "used to be attorney" (past) and "is judge" (current) → Answer with the judge position, NOT the attorney position.
        - **TEMPORAL REASONING SYNTHESIS (MANDATORY)**: When facts provide specific dates/years and the query asks about temporal relationships (before/after), you MUST perform logical inference to identify the correct answer.
            - **RULE**: If Fact A states "Event X happened in year Y" and Fact B states "Event Z began in year W", and the query asks for events that happened "before event Z began", you MUST conclude that Event X happened before event Z began if year Y < year W.
            - **FORBIDDEN RESPONSE**: "The documents do not specify" - this is WRONG when dates allow clear temporal reasoning. You MUST perform the logical inference and provide the answer.
            - **CHRONOLOGICAL LOGIC**: Year A comes before year B if A < B. This is basic mathematical reasoning you MUST apply when dates are provided.

        ## SEMANTIC PRECISION (CRITICAL FOR ALL QUERIES):
        - **PRESERVE DISTINCT CONCEPTS**: When the query uses specific terms that have distinct semantic meanings, you MUST preserve these distinctions and NOT conflate different concepts.
            - **Query-to-Fact Alignment**: Only use facts that contain EXACTLY the same semantic concept as the query term. Do NOT substitute synonyms or related concepts unless the fact explicitly supports the connection.
            - **Example Problem**: Query asks about "first used" but you answer with "invented" - this is WRONG unless the fact explicitly states the inventor was also the first user.
            - **Example Problem**: Query asks about "created" but you answer with "popularized" - this is WRONG unless the fact explicitly states the popularizer also created it.
            - **General Rule**: If the query uses term X and the fact uses term Y, you can only connect them if X and Y are truly synonymous in context OR the fact explicitly bridges the concepts.
        - **CONTEXT-AWARE INTERPRETATION**: Consider the most reasonable interpretation of ambiguous terms based on context and common usage patterns.
            - **Avoid Overly Literal Interpretations**: When a term could have multiple meanings, prefer the interpretation that makes the query meaningful rather than trivial.
            - **Utilize All Relevant Information**: If documents contain information that could answer a reasonable interpretation of the query, use that information rather than defaulting to a literal interpretation that ignores available facts.
            - **CREATIVE WORKS PRIORITY**: When comparing "titles" in creative contexts (film directors, writers), prioritize DOCUMENTED SPECIFIC WORKS over general professional roles. If documents mention concrete film titles for one person but only general roles for another, the person with documented specific works has more "creative titles". Look for phrases indicating specific works like "directed by [person] [film title]", "[film title] directed by [person]", "remade as [film title]", etc.
        - **SOFTWARE VS HARDWARE DISTINCTION (CRITICAL)**: When comparing entities, you MUST strictly distinguish between software and hardware, and never conflate them as sharing the same "media type" or category.
        - **TERM SPECIFICITY PRESERVATION**: When facts use specific qualifiers (temporal, geographical, conditional), you MUST preserve these in your interpretation and not generalize them.
        - **ADMINISTRATIVE HIERARCHY AWARENESS**: When dealing with administrative divisions (countries, states, counties, municipalities, districts, etc.), you MUST understand the hierarchical relationships and NOT confuse different levels.
            - **General Principle**: If asking about "subdivisions of [entity name]", identify what administrative level comes directly BELOW that entity in the hierarchy, NOT the entity itself or higher levels.
            - **Avoid Self-Reference**: Never answer that the subdivisions of "X" are called "X" - this is always incorrect.
            - **Use Hierarchy Context**: Look for facts that describe the administrative structure and identify the next lower level in the chain.
        - **CONJUNCTIVE QUESTION LOGIC (A AND B queries)**: For queries asking if entities share "X AND Y" (e.g., "same country AND professional career"), answer structure MUST avoid logical contradiction:
            - **If BOTH conditions met**: "Yes, [entities] share both [X] and [Y]: [details]"
            - **If ONLY SOME conditions met**: "No, while [entities] share [X] ([shared details]), they differ in [Y] ([different details])"
            - **If NO conditions met**: "No, [entities] differ in both [X] and [Y]: [details]"
            - **FORBIDDEN**: Starting with shared attribute acknowledgment then immediately saying "they do not share" - this creates logical self-contradiction. Use "while...but" or "although...however" structure to properly connect partial matches.
        - **OCCUPATION COMPARISON LOGIC**: When comparing single attributes (not conjunctive), prioritize highlighting SHARED CORE OCCUPATIONS over minor distinctions.
            - **FORBIDDEN RESPONSE**: Saying careers are "not identical" without acknowledging shared core occupations when they exist.
            - **Document-Supported Synonym Recognition**: Only recognize occupational terms as equivalent when the documents themselves support the equivalence through explicit context or parallel usage. For example, if documents describe both people using terms like "minister" and "cleric" in religious leadership contexts, you may recognize them as functionally equivalent religious leaders. Do NOT use pre-defined synonym lists - base equivalence judgments on what's actually present in the documents.
            - **QUERY-DRIVEN EQUIVALENCE**: When the query asks about shared/common occupations or attributes, you MAY use reasonable functional equivalence within the same domain (religious, academic, military, etc.), even if exact terms differ. For example, "minister" and "cleric" may be considered equivalent religious leadership roles when the query asks "what occupation do they have in common."
            - **INDUSTRY INFERENCE RESTRICTIONS**: You are FORBIDDEN from inferring or stating that people work in the same industry unless the documents EXPLICITLY use that exact industry name for BOTH individuals. However, if documents use industry-specific terms for some individuals (e.g., "film director" clearly indicates film industry), you may acknowledge this for those individuals but cannot extend it to others without explicit documentation. If the question asks about shared industry, your response should note that industry information is not consistently specified across all individuals.
        - **STRICT ATTRIBUTE VALIDATION**: Do NOT infer or state attributes, concepts, or terms that do NOT appear anywhere in the provided documents. However, allow reasonable logical connections when documents explicitly place individuals in comparative contexts. For example, if a document states that Person X developed something "with as much rigor as his contemporaries, including Person Y", you may conclude that Person Y is also rigorous. Only use terms and concepts that are explicitly present or clearly implied through such comparative contexts in the document content.
            - **Distinction Preservation**: Always note meaningful distinctions (e.g., Anglican vs. Wesleyan, Catholic vs. Protestant) but don't let minor differences erase major similarities.
            - **Category Hierarchy**: Use the most appropriate shared occupational level - prefer broad categories like "cleric" over specific roles like "bishop" when the specific roles differ.
        - **SPECIFIC QUANTITY REQUESTS**: When the query asks for a specific number of items (e.g., "what 4 divisions", "which 3 reasons", "how many 5 categories"), handle with precision and avoid over-inference.
            - **Exact Match Priority**: If facts list EXACTLY the requested number of specific, named items, use them directly without disclaimers.
            - **Over-Count Handling**: If facts list MORE than requested, do NOT select a subset - state the actual number available and list all of them, noting that more exist than requested.
            - **Under-Count Handling**: If facts list FEWER than requested, state how many are actually documented and do not speculate about missing items.
            - **No Match Rejection**: Only say information is unavailable if facts provide no items at all, or if the request cannot be satisfied by the available facts.
            - **Avoid Selection Bias**: Never select or prioritize items based on external knowledge - stick strictly to what's documented.
            - **Example 1**: Query asks "what 4 divisions" and facts list exactly 4 specific divisions → Answer with those 4 specific divisions.
            - **Example 2**: Query asks "what 4 divisions" but facts list 6 specific divisions → State "The documents list 6 divisions: [all 6], though you asked for 4."
            - **Example 3**: Query asks "what 4 reasons" but facts only provide 2 specific reasons → State "The documents provide 2 reasons: [reasons listed], with no information about additional reasons."

        ## BUSINESS CONCEPT INTERPRETATION (CONTEXT-AWARE):
        - When asked about "the concept of the business" or similar phrases, carefully analyze the context from the facts:
            - **If the facts describe a government/non-profit institution** (courts, agencies, educational bodies, regulatory bodies): Interpret "business" as the core operational concept or institutional framework of that organization
            - **If the facts describe a commercial entity** (companies, corporations, firms with profit motives): Interpret "business" as their business model and core operations
            - **Use fact evidence, not assumptions**: Only apply institutional interpretation when facts explicitly show the entity is a government/non-profit institution
            - **Example**: If facts show someone works for "Circuit court Judge" and "law firm", "business" for the court means institutional framework, for the law firm means business operations

        ## Quick Clarifications (Global)
        - Concept/Definition: For concept/nature/definition queries resolved via multi-hop, state the definition from the facts; job titles alone aren't sufficient when a definition exists.
        - Shared-Context Lift: If X is linked to named entity Y (e.g., "officer of the London Metropolitan Police"), and Y has a stated definition/function/mandate, use that to answer about the entity/organization X worked for/served.
        - Temporal: Normalize timeline ("used to be" < "is"); for "after A" queries, return the earliest confirmed subsequent role/entity after A; if none, say unknown.
        - Identity & Terms: Treat name variants (middle names, initials, casing, punctuation, transliterations) as the same unless core attributes conflict; map generic workplace terms (business/organization/institution/office/agency/body) to the explicitly named entity in the facts and note this mapping in Supporting Details.
        - Treat obvious name variants (middle names, initials, capitalization, diacritics) as the same entity unless the text clearly separates them.
        - **Extension for Spelling Variations:** For common spelling variations (e.g., "Ishqbaaz" vs. "Ishqbaaaz", or "color" vs. "colour"), treat them as the same entity **only if** they are highly similar (differ by 1-2 characters) AND the context (e.g., same topic, description, or attributes) strongly supports equivalence. If linked, explicitly note in the fact: "(Linked via spelling variant and matching context)". If uncertain, list separately and mark as "possible variant, unconfirmed". NEVER assume equivalence based on external knowledge—rely solely on *Information Available*.
        - Status (Present-Tense): Phrases like "remains/is/continues to be a [publication/journal/magazine]" are affirmative current status → treat as "still in publication/in print" unless a same-scope fact explicitly says it ceased; if no date, qualify as "as of the source's timeframe" in Supporting Details.
        - Silence: "no explicit link"/"not mentioned"/"no info found" are source disclaimers; ignore them when affirmative facts/chains exist. Only explicit denials contradict.
        - **Exhaustive List Interpretation:** When a source uses a definitive phrasing like **"Its members are [A] and [B]"** or **"The committee consists of [X], [Y], and [Z]"**, you **MUST** treat this as a complete and exhaustive list for the context provided. You are **strictly forbidden** from inventing uncertainty by suggesting there might be other, unmentioned members. Do not state that the "total membership is not specified" in such cases.

        ## Semantic Alignment (Definitions & Modifiers)
        - Leverage explicit definitions and taxonomies: when a fact states that concept/genre/group **C** is a subset of **D**, and another fact assigns the subject to **C** (including clear modifier variants like "C-style" or "C-influenced"), you **MUST** synthesize the chain `subject → C → D`.
        - Treat descriptive modifiers as belonging to their base concept when the evidence provided makes the connection explicit; surface the higher-level category in the Direct Answer and explain the bridge in Supporting Details.
        - When two facts reference the **same uniquely titled entity** (e.g., a particular work, institution, or named program) and no conflicting scope is given, you **MUST** treat it as the same item across sources, even if one fact only lists its descriptive details. Use this to connect participants or events tied to that title with definitions of the entity.
        - Any clause saying "does not explicitly confirm" or similar is a **source disclaimer**. If the positive portion of the fact plus another fact supplies the missing link, IGNORE the disclaimer and deliver the synthesized conclusion. **EXCEPTION:** If the disclaimer explicitly and unambiguously states that the final answer itself (the specific who/what being asked) is "not specified/not identified/unknown" AND no other facts provide this information, preserve this limitation.
        - Do NOT fabricate alignments; only apply this rule when the available facts supply the definition, synonym, or modifier relationship that justifies the mapping.

        # 'SAFE INFERENCE' RULE
        Use this capability with caution. You may bridge minor and obvious semantic gaps (e.g., equating 'income sources' with 'income categories') only if the connection is unambiguously self-evident from the extracted facts. When you do this, you MUST explicitly state the assumption being made, for example: 'The document refers to 'sources,' which are being interpreted here as the 'categories' requested in the query.' For any connection that is not strictly self-evident, or for any significant gaps, you MUST revert to the original behavior and simply report the limitation without making an inference.

        **1. When a fact from one source affirms (confirms) that A is true, and another source merely states that A is not mentioned, not found, or is missing, you MUST only keep the affirmative statement. Do NOT treat the absence of information as evidence against the affirmative fact.**

            - Example:  
            - Fact A: "X is a member of Y but is not mentioned as being part of Z."  
            - Fact B: "X is a member of Z."  
            - **Correct synthesis:** "X is a member of Y and Z." (Do not treat the 'not mentioned' in A as a contradiction of B.)

        **2. However, if a statement contains a limitation of scope (for example, specifies a time, place, event, or other restriction), you MUST explicitly carry this limitation forward into your synthesis.**

            - If the scope/limitation in a fact is different from that in the user query or another fact, you MUST NOT merge or generalize these as if they are the same universal entity or claim, unless the scopes are truly compatible or mergeable.
            - Example 1:
            - Fact A: "X attended the Paris conference, but not the London conference."
            - Fact B: "X attended the London conference."
            - **Correct synthesis:** "X attended both the Paris and London conferences."
            - But, if Fact B only says "X attended a conference," you CANNOT assume it's the same conference as A without explicit mention.

            - Example 2:
            - Fact A: "X is confirmed as a member of Y during 1915-1917."
            - Fact B: "X is a member of Y."
            - If the query is about 1915-1917, only combine these if there's no contradiction in time scope.
            
            - Example 3:
            - Fact A: "X had 3 members in 1915."
            - Fact B: "X had 4 members in 2014."
            - If the query is X's membership in 2000, you should combine them like "The documents do not provide X's membership information in 2000. While in 1915, X had 3 members, in 2014, X had 4 members, this cannot be used to determine the 2000 membership count.".

        **Summary Rule:**
        - Absence or silence about a fact does NOT contradict affirmative evidence.
        - Always respect the stated scope or limitations. Never merge facts across incompatible scopes.
        - Your synthesis must clearly preserve or note any limitations given in the original facts.

        ---


        **NON-NEGOTIABLE CORE RULES ON SYNTHESIS:**
        0.  **THE PRIME DIRECTIVE - PRESERVE ALL SPECIFIC DETAILS:**
            - **YOU MUST RETAIN AND REPEAT ALL** specific named entities (e.g., people's names, group names), numbers, dates, and other concrete data points found in the *Information Available*.
            - **ENTITY NAME COMPLETENESS:** When copying entity names, preserve the COMPLETE name including any aliases, alternate names, or clarifications provided (e.g., if the fact says "Entity A (also known as Entity B)", you MUST use the full phrase, NOT just "Entity A" or just "Entity B"). This applies regardless of the format: parentheses, commas, "aka", "or", etc.
            - **NEVER** summarize these specific details away. If a fact says "A is in B", your output MUST include the name "B". Do NOT replace it with a generic phrase like "another group" or "other projects". This is your highest priority.
       
        0.5. **CONTEXTUAL RELEVANCE:**
            - Before synthesizing facts, determine the **domain context** of the query
            - If multiple entities share the same name but belong to different domains, prioritize the entity that matches the query's domain context
            - **Example:**Query about pigs mentioning "B" + facts about both "B (pig)" and "B (a person)" → prioritize the pig information
            - Only treat as contradictory if facts about the SAME entity in the SAME domain conflict
            - **Clarification approach:** When domain ambiguity exists, acknowledge it: "In the context of [domain], [entity] is confirmed as..."

        1.  **PERMITTED SYNTHESIS (Direct Attributes) - HIGHEST SYNTHESIS PRIORITY:**
            - **This rule takes precedence over the 'Structure by Source/Group' rule.**
            - You **MUST** synthesize facts that describe the **same entity**. To do this, you will apply the following logic in order:

                - **A. FACT PRECEDENCE FILTER (Apply First):** Before any synthesis, you must filter the information.
                    - **Primary Facts:** Statements about the world (e.g., "John Smith is a biologist," "X designed the platform").
                    - **Source Disclaimers:** Statements about a document's limitations. This includes phrases like "this document does not mention," "no information was found," and crucially, "there is no direct documented confirmation."
                    - **RULE:** If a Primary Fact or a chain of Primary Facts (via MANDATORY LOGICAL SYNTHESIS) can answer the query, you **MUST** use it. Any conflicting Source Disclaimers **MUST BE COMPLETELY IGNORED AND DISCARDED.** This situation is **NEVER** to be reported as a contradiction or limitation. An affirmative logical chain always overrides a statement about the absence of explicit confirmation.

                - **B. ENTITY IDENTITY & DISAMBIGUATION:**
                    - Assume identity by default for orthographic variants when context aligns (middle names, initials, transliterations, spacing/case). Example: "John William Smith" = "John Smith" unless core attributes conflict.
                    - This mandatory rule includes treating common orthographic (spelling) variations of names (e.g., 'Faisal' vs. 'Faysal', 'Mohammad' vs. 'Muhammed', 'Jon' vs. 'John') as the same entity, especially when the surrounding context (such as profession, nationality, or topic) is consistent.
                    - **Expanded Matching for Name Variants:** Also match names where middle names are omitted (e.g., "James Barnerd Elen" = "James Elen" if context like role/company matches), or common abbreviations (e.g., "J.B. Elen" = "James Barnerd Elen" if attributes align). Do NOT merge if conflicts exist in dates, roles, or locations.
                    - **Do NOT merge, and instead treat as separate entities, if** you find conflicts in:
                        - **Different and non-overlapping date ranges** (e.g., birth/death years like 1910-1940 vs. 1830-1910).
                        - **Mutually exclusive core relationships** (e.g., being the 'father of X' vs. the 'son of X').
                        - **Distinct and incompatible primary roles in different historical eras** (e.g., a US Civil War officer vs. a World War II officer).
                        - **Different and non-overlapping primary roles** (e.g., Fact A: "John Smith, the biologist from London," vs. Fact B: "John Smith, the baker from Manchester").
                    - **Note:** Neutral silence ("the document does not mention X") is a Source Disclaimer, not a conflict.

                - **C. MANDATORY HIERARCHICAL & BRAND SYNTHESIS (CRITICAL LOGIC RULE):**
                   You **MUST** synthesize facts across different levels of corporate or brand hierarchy. This is a mandatory logical deduction, not a forbidden inference.
                    - **RULE 1: BRAND-ENTITY EQUIVALENCE:** If a product (e.g., a film) is described with a general brand name (e.g., "a Disney film", "a Ford car"), you **MUST** treat this as a factual statement that the product was produced by the main corporate entity associated with that brand (e.g., "The Walt Disney Company", "Ford Motor Company").
                    - **Example:** Fact A: "'The Strongest Man in the World' is a Disney film." -> **Mandatory Deduction:** "The Walt Disney Company produced 'The Strongest Man in the World'."
                    - **RULE 2: SUBSIDIARY-PARENT LINKING:** If a fact names a known subsidiary (e.g., "Walt Disney Animation Studios", "Pixar"), you **MUST** link it to its parent company (e.g., "The Walt Disney Company").
                    - **Example:** Fact B: "'The Little Mermaid' was produced by Walt Disney Feature Animation." -> **Mandatory Deduction:** "'The Little Mermaid' was produced by The Walt Disney Company."
                    - **FINAL SYNTHESIS:** After applying these deductions, you then compare the resulting unified entities.  
        
        2.  **FORBIDDEN SYNTHESIS (Inferential Linking):**
            - You should avoid creating relationships between different subjects EXCEPT when:
                - 1. The connection is self-evident as defined in the 'SAFE INFERENCE' rule, OR  
                - 2. Multiple facts clearly describe attributes of the same uniquely identifiable entity (as per PERMITTED SYNTHESIS rules), OR
                - 3. **(CRITICAL EXCEPTION FOR SHARED CONTEXT) Facts connect different entities (e.g., a person, a role) through a single, shared, uniquely named context (e.g., a specific film, a documented event, a company). In this case, you MUST synthesize the relationship.**
                    - **Example of Correct Application:**
                    - Fact 1: "Actor A appeared in the film B."
                    - Fact 2: "Director D directed the film B."
                    - **Correct Synthesis:** "Actor A was directed by Director D in the film B." This is considered a direct logical connection, not a forbidden inference.
                    - **Example of Forbidden Inference (Still Applies):** Fact 1: "Entity X was a member of Group Y." Fact 2: "Members of Group Y held activities." -> **Incorrect Inference:** "Entity X participated in the activities." (This remains forbidden because "activities" is general, unlike a specific, named film).
                    - Never merge facts across incompatible scopes (time/place/affiliation/side, etc.), e.g. Fact 1: "meetings of party A from group B occurred" and Fact 2: "meetings X, Y, Z are held by party A from group C" -> **Incorrect Synthesis:** "The meetings X,Y,Z are from group B" UNLESS if the source explicitly states that group B is identical to (or a subset of) group C.
                    - NEVER conjoin scopes (B union C) of different extracted facts. Use only the scope required by the query, UNLESS the source explicitly states B = C or B ⊆ C for the specific items.
                - **4. (CRITICAL EXCEPTION FOR LOGICAL TRANSITIVITY): When a chain of Primary Facts establishes a clear, non-speculative hierarchical relationship (such as an entity's location), you MUST synthesize the final conclusion.** This is considered a mandatory logical deduction, not a forbidden inference.
                    - **Example of Correct Application:**
                    - Fact A: "The band A is based in the city of B."
                    - Fact B: "The city of B is in the county of C."
                    - **Correct Synthesis:** "The band A is based in the county of C."

        **Mandatory Process:**

        **Step 1: Identify the Query's Core Subjects**
        - Analyze the *Original QUERY* to determine its main subjects (e.g., people, events, concepts).

        **Step 2: Synthesize a Comprehensive and Faithful Answer**
        - Review ALL facts from *Information Available*. Your primary goal is to find information that directly addresses the query's core subjects.
        - If you cannot find the direct answer in the *Information Available* for the Original Query, you MUST state it and report what you found in the Supporting Details section in details. Never enforcedly to answer the query by making assumptions or inferences to bridge gaps between the avaliable information and the user's query, just report the exact texts and state with its limitations (qualifiers).
        
        **CRITICAL: EXPLICIT FALSE PREMISE CORRECTION**
        - **Detect and EXPLICITLY correct false assumptions.** Cross-check query assertions against facts.
        - **EXPLICIT CORRECTION REQUIRED**: When facts contradict or differ from query premise, Direct Answer MUST start with: "The documents do not state/show that [false premise]" OR "[Entity] did not [incorrect action]" OR "The documents show [correct relationship], not [query premise]".
        - **THEN provide answer**: After explicit correction, give actual information.
        - **FORBIDDEN**: Implicit correction through careful wording alone. State discrepancies directly so users clearly understand the error.
        
        - **CRITICAL: General vs. Specific Answers**: If the available information provides both a general answer (e.g., "Entente powers") and specific examples (e.g., "Romania, France"), prioritize the general answer if it directly and completely answers the query. Use specific examples as supporting evidence, not as the primary answer.
        - **CRITICAL DISTINCTION - Direct Fact Comparison vs. Inference**: 
          - **ALLOWED and MANDATORY (Direct Fact Comparison)**: When comparing entities, if both are explicitly described with identical terms in the source material (e.g., both called "popular music band", both from "UK"), you MUST identify these shared explicit descriptors as commonalities. This is direct textual comparison, not inference.
          - **IGNORE META-CONCLUSIONS:** Statements from sources about their own limitations (e.g., "no commonalities were found," "the document does not link X and Y") are to be TREATED AS IRRELEVANT METADATA and COMPLETELY IGNORED. Your own direct comparison of the facts is the only valid method for finding the answer.
          - **REQUIRED (Basic Logical Deduction)**: When facts provide clear functional or definitional information that directly answers the query, you MUST draw the obvious conclusion. The following types of reasoning are MANDATORY:
            - **✅ MANDATORY Definitional Logic (CRITICAL - NEVER VIOLATE)**: For comparison queries where facts describe entities with opposing functional characteristics (one designed/equipped for activity X, other explicitly not designed/equipped for X), the comparison is resolved by definition. You are ABSOLUTELY FORBIDDEN from claiming "insufficient information" or "cannot be determined". Apply the definitive conclusion that the definitions require.
            - **✅ Mathematical Operations**: If facts provide numerical values (e.g., "A has 100 units", "B has 50 units"), you MAY perform basic arithmetic to answer queries like "what is the total" → 150 units.
            - **✅ Numerical Comparisons**: You MAY directly compare numerical values, approximate values, and numerical ranges to answer comparative queries. When comparing values where one is clearly greater than the maximum of another range, you MUST draw the obvious conclusion without requiring exact precision.
            - **✅ Unit Conversions**: You MAY use standard unit conversions (e.g., "5 feet" vs "2 meters" → 5 feet ≈ 1.52m, so 2 meters is taller; "100°C" vs "200°F" → boiling point comparison).
            - **✅ Self-Evident Naming**: Information inherent in the **names themselves** (e.g., "World War I" vs "World War II" → II came after I; "1990s" vs "2000s" → 2000s is later; "First Battle of..." vs "Second Battle of..." → chronological order).
            - **✅ Nationality → Birthplace Inference (Common Sense)**: When a query asks about birthplace/birth location and documents state a person is "[nationality] [profession]" (e.g., "French director"), you MAY reasonably infer they were born in that country UNLESS documents explicitly state otherwise (e.g., "currently lives in" different location does NOT contradict birth in origin country). This is a standard, reasonable inference about origin.
            - **❌ FORBIDDEN - External Factual Knowledge**: You CANNOT use factual knowledge not stated in the documents:
              - Geographic facts: "Massachusetts" vs "Texas" → which is further north (requires external geographic knowledge)
              - Historical dates not evident from names: "Battle of Waterloo" vs "Battle of Hastings" → which happened first (requires historical knowledge unless dates are provided)
              - Entity-specific facts: "Company A" vs "Company B" → which is older/larger/richer (unless explicitly stated in the facts)
              - Any other external knowledge not derivable from the text, names, or basic mathematical/logical reasoning
          - **FORBIDDEN (Inference)**: Do NOT infer connections that are not explicitly stated (e.g., if A is "anti-war" and B "opposes central power", do NOT assume they share the same position unless explicitly connected).

          - **Direct Answer with Comprehensive Detail (Highest Priority):**
            - Normalizes obvious typos in the query.
            - You MUST provide a direct, clear answer to the user's specific question format in the Direct Answer section.
            - **Prioritize the Core Question:** First, identify the central subject of the Original Query (e.g., the city, the date, the main person).
            - If you cannot find the direct answer in the *Information Available* for the Original Query, but you have some related facts, then you MUST state these most relevant facts you found to briefly clarify the mismatch in the Direct Answer.
              - **If the central question CAN be answered, your `Direct Answer` MUST explicitly state which entity satisfies any descriptor in the query (e.g., "the author of X is Y") before delivering the requested outcome.** Do NOT begin by stating limitations about secondary, missing details, and do not omit descriptor facts from the Direct Answer or Supporting Details.
              - **Explicit Subject Naming (MANDATORY):** Restate every subject in the Direct Answer by its explicit name or title; do not rely on pronouns such as "it", "they", or "this event" to identify the subject.
                - *Illustrative template:* `Direct Answer: The Siege of Vicksburg began on May 18, 1863, whereas the Battle of Shiloh was fought on April 6–7, 1862.`
              - When a fact contains both the answer and a limitation, open the Direct Answer with the affirmative portion and move the limitation into Supporting Details to preserve nuance without obscuring the answer. (If you are invoking the "Helpful Alternative" fallback, you may append the limitation immediately after the affirmative fact instead.)
              - **Only if the central question CANNOT be answered** should you begin your `Direct Answer` by stating the specific limitation, followed by the most relevant related facts.
            - If the user asks "What countries...", start with the countries you can identify or state clearly if none can be identified in the Direct Answer section.
            - **Numerical Comparison Queries:** For queries asking "which has more/higher/greater" or similar comparative questions with numerical values, you MUST directly compare the provided numbers and give a clear answer. Approximate values and ranges can be directly compared when the difference is mathematically unambiguous. Do NOT invoke limitations about "exact numbers" when the comparison result is mathematically obvious.
            - **Descriptor Mapping Requirement:** When the query describes the subject indirectly (e.g., "the author of X", "the company that built Y", "the city where Z"), explicitly state in the Direct Answer which entity satisfies that descriptor before giving the requested attribute. Example template: "The leader of X is Y, and after moving to England, Y chose to live in Z." Avoid leaving the descriptor implicit.
            - **CRITICAL: COMPLETE CLINICAL SPECTRUM FOR SYMPTOMS QUERIES:** When the query asks about symptoms, manifestations, or clinical features of a disease/virus/condition, you MUST provide the COMPLETE severity spectrum from mildest to most severe. Include all three critical components if present in facts: (1) minimal/absent presentation patterns (which populations typically show minimal/no manifestations); (2) typical/common presentation patterns; (3) rare but severe outcome patterns (critical for medical decision-making even if infrequent). Omitting ANY severity level creates dangerous medical misinformation. Pattern structure: "[Typical manifestations include X], though [portion/demographics] show minimal or absent manifestations. Rarely, [severe outcomes] may occur [with risk factors if documented]."
            - **ATTRIBUTE COVERAGE REQUIREMENT:** Only state an attribute value in the Direct Answer when a fact explicitly links that value to the queried subject within the same scope (time, place, role). If the available facts describe the attribute for a related entity but do not confirm it for the subject or timeframe of the query, surface the limitation in the Direct Answer instead of asserting the attribute.
        - **ATTRIBUTE LINK AUDIT (MANDATORY):**
            1. Before drafting the final answer, enumerate every attribute-value pair that could satisfy the query (e.g., {{subject}} → {{league}}, {{subject}} → {{location}}) and locate the exact fact that ties them together.
            2. For each pair, check three items: (a) same named subject, (b) same scope qualifiers (timeframe, role, location), and (c) the fact explicitly states the linkage.
            3. If ANY of these checks fail, you MUST treat the attribute as *unconfirmed*. In the Direct Answer, state that the requested attribute is not confirmed and briefly mention what the documents do say instead.
               - **Concrete reminder:** Facts like "Player X played for Club Y" combined with "Club Y plays in League Z" do **NOT** satisfy this requirement unless a single fact explicitly states that Player X played in League Z (and the timeframes match). In such cases you MUST answer that the player's league is unspecified and list the two supporting facts separately.
            4. In Supporting Details, include the contextual fact but annotate the limitation in neutral terms (e.g., "Document describes {{related entity}} having {{attribute}}; the timeframe for the queried subject is not specified.").
            5. Only attributes that pass all checks may appear as definitive answers.

        - **SCOPE VALIDATION:** Scope qualifier alignment is handled by the validator LLM. Do not override validator decisions.
            - MUST ALSO provide detailed, comprehensive coverage of ALL relevant facts to user query from the available information in the Supporting Details section.
            - Present the complete factual picture: include all supporting details, qualifications, and context from the source material in the Supporting Details section.
            - Be definitive about what the facts do and do not show, while preserving all nuances and details from the original sources in the Supporting Details section.
          - **Characterizing Source Quality**: When referencing information from sources, distinguish between specific detailed statements and general/vague statements. Use phrases like "according to one general statement..." for broad claims that lack specifics, versus "according to detailed information..." for concrete facts.
          - **Focus on Query-Relevant Entities**: Only discuss entities that are directly relevant to answering the user's query. If an entity has no connection to the query's core elements, omit it entirely from your direct answer response, if it has any information related to the query, put it in supporting details.
          - **Accurate Connection Recognition**: When the source material explicitly states a connection between entities, acknowledge this connection. Do not deny connections that are clearly stated in the facts.
          - **CRITICAL: True Contradiction Reporting**: Only report contradictions when documents contain CONFLICTING AFFIRMATIVE FACTS (e.g., Document A: "X is tall", Document B: "X is short") **AND these conflicting facts explicitly refer to the SAME identified entity or event**. Do NOT treat "no information found" as contradictory evidence. Only report contradictions when facts explicitly contradict each other or contradict the query's premises with opposing claims, and are confirmed to be about the *exact same* subject.
          - **Supporting Details - What to Include/Exclude**:
            - **INCLUDE (Always):**
              - ALL facts that answer ANY component of the query (primary or secondary)
              - Facts providing different attributes of the same entity (e.g., location AND tier AND founding date)
              - Multiple facts stating the same conclusion IF they cite different sources or add different details
              - Named entities, numbers, dates that appear in query-relevant facts
              - **ATTRIBUTE-SPECIFIC INFORMATION:** If query asks about a specific attribute, include ALL facts that mention that attribute type, even if they seem overlapping or if you used a general term in Direct Answer
              - **CRITICAL: COMPLETE CLINICAL SPECTRUM (Medical/Clinical Context):** When the query asks about symptoms, manifestations, or clinical presentations of a disease/virus/condition, you MUST include ALL facts describing the complete severity spectrum: (1) minimal/absent presentation patterns, (2) typical/common presentation patterns, and (3) rare but severe outcome patterns. This is NOT optional - all three severity levels are medically critical. Omitting minimal/absent presentation information falsely implies all cases show manifestations. Omitting rare but severe outcomes fails to warn about potential serious consequences. The complete spectrum must be preserved for responsible medical information provision.
            - **EXCLUDE (Only these):**
              - Pure meta-commentary (e.g., "Source A does not mention Entity B" when there's no positive info)
              - Truly irrelevant details (e.g., birth date when query only asks about current job)
              - Facts about entities with zero connection to any query term
            - **CRITICAL: When in doubt, INCLUDE rather than exclude.** Err on the side of completeness for query-relevant information.
        - **NO TRUNCATION RULE:** In the Supporting Details section, reproduce each cited fact exactly as it appears in *Information Available*—no paraphrasing, no partial quotes, and absolutely no ellipsis characters (`...` or `…`). If a sentence is long, copy the entire sentence instead of shortening it.
        - **ATTRIBUTE COVERAGE RULE:** Whenever an extracted fact reports an explicit value for a property that the query asks about (for example, the league, tier, category, location, date, or other descriptor), include that entire sentence verbatim in Supporting Details **only if** the fact explicitly applies the property to the same subject and scope required by the query. If the fact describes a related entity (e.g., the club’s current league) without confirming it for the queried subject or timeframe, include it as contextual evidence but explicitly note the lack of confirmation instead of presenting it as the answer.
        


        - **CRITICAL: Silence Is Neutral**  
          - If a fact set simply does NOT mention an entity or claim, treat this as **neutral** (neither confirmation nor contradiction).  
          - **NEVER treat "no information found" or "document does not mention X" as contradictory evidence.**
          - Only statements that **explicitly deny** a claim (e.g., "X is NOT Y" or "X never did Z") are treated as contradictory evidence.
          - **Example**: Source A: "John is tall." Source B: "No information about John's height." → This is NOT a contradiction, use Source A's information.
          - **Example**: Source A: "A was British." Source B: "No nationality information provided." → This is NOT a contradiction, use Source A's information.

        - **Handling Partial Matches and Contradictions (High Priority):**
          - First, look for facts that **explicitly negate** or conflict with another fact. Treat these as contradictions.
          - **CRITICAL**: Mere absence of a statement about an entity (silence) is **NOT** a contradiction and must be classified as neutral. Statements like "no information provided," "document does not mention," or "no details available" are NEUTRAL, not contradictory.
          - **MANDATORY: Query-Fact Contradictions**: If the available information contains facts that directly contradict the premise of the query (e.g., query asks about  "X belongs to A" but facts show "X belongs to B"), you MUST highlight this contradiction prominently in your Direct Answer, even if you can provide a partial answer to the query.
            - **ASSOCIATIVE INFERENCE GUIDANCE**: When facts explicitly state that multiple entities share a common attribute or action (e.g., "A, B, and C all developed X together"), you MAY reasonably infer that ALL mentioned entities share that attribute. However, when entities are only listed as contemporaries or comparators without explicit shared action (e.g., "A developed X with rigor comparable to B, C, D"), this does NOT automatically mean B, C, D share the attribute or action with A.
        - **SCOPE ALIGNMENT CHECK (CRITICAL):** Before presenting an attribute as answering the query, confirm that the fact names the same subject and shares the relevant qualifiers (timeframe, role, event, location). If the attribute fact lacks the necessary qualifier or only describes a related entity in a different scope, treat it as contextual information and clearly state that the requested linkage is unconfirmed.

        - **Comprehensive Synthesis:**
          - After providing your direct answer, present ALL relevant supporting facts in detail, IGNORE the facts that are NOT relevant to the query.
          - **Structure by Source/Group:** You should generally structure your answer around distinct groups, events, or sources. **HOWEVER, THIS RULE IS OVERRIDDEN by the PERMITTED SYNTHESIS rule.** If multiple facts provide different attributes for the **exact same uniquely identifiable entity** (e.g., a person with a specific name), you MUST synthesize them into a single coherent description of that entity, rather than describing what each source says separately.
            - **Example:** If one fact discusses Group A's activities and another discusses Group B's, present them separately: "Regarding Group A, the documents state...". Then, "Regarding Group B, the documents state...". Do NOT start with a sentence that implies they acted together.
          - **No Unjustified Summaries**: Avoid summary statements (like "Thus, the event involved both A and B") that create a link where none was stated. Your final sentence should reflect the separateness of the information if that's what the facts show.
          - **Complete Coverage of Query Parts (CRITICAL)**: Identify every explicit sub-question or information need expressed in the Original Query. For each one, ensure Supporting Details include **EVERY affirmative fact** you have that answers any component of the query—even if you didn't use it in the Direct Answer. This includes:
            - Facts answering the primary question
            - Facts answering secondary attributes or constraints mentioned in the query  
            - Facts that only partially answer parts of a multi-component query
            Keep qualifiers/limitations attached to each fact.
          - **MANDATORY NO-OMISSION RULE (CRITICAL)**: Every fact from *Information Available* that **directly answers any component of the query** or describes attributes of entities mentioned in the query MUST be restated verbatim in Supporting Details with the correct source citation (Document ID preferred; tool ID if no Document ID available). This includes:
            - Facts that establish the bridge entity (e.g., "Person X worked for Organization Y")
            - Facts that describe the requested attributes of that entity (e.g., "Organization Y is in Industry Z" or "Organization Y operates in Tier W")
            - **QUERY KEYWORD MATCHING (STRICT):** If the query contains attribute-asking words (e.g., "which", "what", "where"), scan Extracted Facts for ALL facts mentioning the query entity AND containing related descriptive information (names, classifications, tiers, categories, locations, etc.). Include ALL such facts in Supporting Details, even if some seem more specific or more general than others. For example, if query asks "which league" and you have both "Cricket League" and "Championship, second tier", include BOTH facts—do not judge one as redundant.
            - Even if you only use some facts in the Direct Answer, ALL facts that answer different aspects of the query MUST appear in Supporting Details
            - Even if multiple facts repeat the same conclusion, list each one separately. Skipping or consolidating these facts is strictly forbidden.
          - **REFINED RULE: INCLUDE ALL PARTIAL AND LIMITED FACTS (High Priority)**: In addition to the no-omission rule, you MUST include every fact from *Information Available* that contributes at least one affirmative detail about any part of the query (e.g., confirming occupation even if location is missing) and keep any limitations attached to that affirmative detail. List them separately in Supporting Details under a subheader like 'Additional Supporting Evidence:', restating them verbatim (including full text and limitations) and citing the original source identifier (PREFER Document ID like "0e544b5a_3"; if unavailable, use tool ID like "tool_0e544b5a"). NEVER use generic Fact IDs like "Fact 1" or "Fact 2". Do **NOT** include bullets whose only content is that information is absent or unspecified.
          - If different facts appear to conflict, present this variance neutrally in details without trying to resolve it.
          - **NEVER use data from a different scope to answer questions about a specific scope, such as temporal, geographic, organizational, demographic, etc., UNLESS:
            1. The facts explicitly state that the data is applicable to the specific scope, OR
            2. The facts explicitly establish that the specific scope is a subset of the larger scope AND the data applies to ALL members of the larger scope**
          - You MUST KEEP the source citations (PREFER Document ID like "e11a1fa0_3"; fallback to tool ID like "tool_e11a1fa0" only if no Document ID; NEVER use generic "Fact 1", "Fact 2", etc.) for every fact. 
        - ** Verify the final answer
            1. The direct answer is ONLY based on the information available. 
            2. NEVER use any external information or uncited facts to support the direct answer if it is not provided in the information available.
            3. You didn't synthesize facts with different qualifiers to answer the query (e.g. anti-militarist is different from opposing the central power).
            4. You didn't conjoin scopes of different extracted facts.
            5. You direct answer is in the scope of the query.
            6. You explicitly mentioned any contradictions between the available facts and the query's premises.
            7. **Comparison Query Check**: For queries asking about commonalities, you correctly identified shared explicit terms/descriptors when present (e.g., if both entities are described as "popular music band", this commonality should be stated in your Direct Answer). For numerical comparison queries, you directly compared the provided numerical values without being blocked by approximate qualifiers when the comparison is mathematically unambiguous.
            8. **Software vs Hardware Comparison Check**: When comparing software and hardware entities (e.g., video games vs gaming consoles), you explicitly stated that they are fundamentally different types of entities and cannot share the same media type, even if they are related in the same ecosystem. You did NOT conflate software (games/apps) with hardware (consoles/devices) as sharing the same "media type" or category.
            9. **Temporal Reasoning Check**: When facts provided specific dates/years and the query asked about temporal relationships (before/after), you performed logical inference using chronological logic (year A comes before year B if A < B). You did NOT say "documents do not specify" when dates clearly allowed inference of the correct answer.
            10. **Silence vs. Contradiction Check**: You did NOT treat "no information found" as contradictory evidence against positive facts from other sources.
            11. **General vs. Specific Answer Check**: You prioritized complete general answers over incomplete specific examples when the general answer fully addresses the query.
            12. **Allowed Reasoning Check**: You verified that any reasoning you applied falls into these ALLOWED categories:
                - ✅ Definitional logic (e.g., "armed fighter" → engages in combat by definition)
                - ✅ Mathematical operations on numbers provided in the facts
                - ✅ Numerical comparisons between values and ranges when differences are unambiguous
                - ✅ Standard unit conversions (e.g., feet to meters, Celsius to Fahrenheit)
                - ✅ Self-evident information from names themselves (e.g., "WWI" vs "WWII" → chronological order; "1990s" vs "2000s" → temporal sequence)
                - ✅ Temporal reasoning synthesis (year A comes before year B if A < B)
            13. **Forbidden External Knowledge Check (CRITICAL)**: You verified that your answer does NOT rely on external factual knowledge such as:
                - ❌ Geographic locations (which city/state is north/south/east/west)
                - ❌ Historical dates not evident from the names (e.g., when "Battle of Waterloo" vs "Battle of Hastings" occurred)
                - ❌ Entity-specific facts (e.g., which company is older/larger without explicit data in the facts)
                - If such external knowledge is needed but not provided, you stated "insufficient information"
            14. **Entity Recognition Check**: You recognized when different names clearly refer to the same organization/entity based on explicit context (e.g., "Disney film" + "Walt Disney Pictures" context), but avoided over-broad matching.
            15. **Attribute Link Audit Check (CRITICAL):** You completed the Attribute Link Audit. Every attribute in the Direct Answer has a fact that explicitly ties it to the query subject with matching scope. Any attribute lacking that linkage was marked as unconfirmed in the Direct Answer and noted as contextual (with its limitation) in Supporting Details.

        - Your final answer (including supporting details) NEVER contain internal system details like 'SubCoA_3'.

        - Your Output MUST ALWAYS start exactly with "Direct Answer: "
       
        ---
        *Original QUERY*
        {self.user_input}

        *Information Available*
        {extracted_facts}
        """


        try:
            print("Extracted Facts:")
            print(extracted_facts)
            self._last_attribute_keyword = None
            synthesis_prompt = (
                user_data_and_query_vanilla
                if self.synthesis_mode == "core"
                else user_data_and_query
            )
            print(f"Final synthesis mode: {self.synthesis_mode}")
            response = self.llm.complete(prompt=synthesis_prompt)
            final_response = response.text.strip()

            # Ensure downstream logging always sees the mandated prefix
            stripped = final_response.lstrip()
            if stripped.startswith("Direct Answer:"):
                final_response = stripped
            else:
                final_response = f"Direct Answer: {stripped}"

            if self.synthesis_mode == "core" and re.search(
                r"\n\s*Supporting Details\s*:",
                final_response,
                flags=re.IGNORECASE,
            ):
                print("⚠️ Core synthesis emitted Supporting Details; check core prompt/output alignment.")

            if self.enable_final_checks:
                final_response = self._ensure_supporting_citations(final_response)
                final_response = self._annotate_direct_answer_uncited(final_response)
                final_response = self._reinforce_verified_facts(final_response)
                final_response = self._apply_scope_validator(final_response)
                final_response = self._wrap_bare_citations(final_response)
                final_response = self._dedupe_supporting_doc_citations(final_response)
                final_response = self._enforce_attribute_answer(final_response)
                final_response = self._smooth_direct_answer(final_response)
            

            return final_response
        except Exception as e:
            print(f"Error during Stage 2 LLM call: {e}")
            return "Error processing response."

    def _ensure_supporting_citations(self, response_text: str) -> str:
        """Append a disclaimer to any supporting detail lacking an explicit citation."""

        lines = response_text.splitlines()
        supporting_section = False
        amended_lines: list[str] = []
        doc_pattern = re.compile(r"[A-Za-z0-9-]+_[0-9]+")

        for idx, line in enumerate(lines):
            amended_line = line
            stripped = line.strip()
            if not stripped:
                amended_lines.append(amended_line)
                continue
            if stripped.lower().startswith("supporting details"):
                supporting_section = True
                amended_lines.append(amended_line)
                continue
            if supporting_section and stripped.startswith("-"):
                if not doc_pattern.search(stripped):
                    # Check nested bullet lines until the next top-level item
                    has_nested_citation = False
                    for nested_line in lines[idx + 1 :]:
                        nested_stripped = nested_line.strip()
                        if not nested_stripped:
                            continue
                        if nested_stripped.startswith("-"):
                            break
                        if doc_pattern.search(nested_stripped):
                            has_nested_citation = True
                            break
                    if not has_nested_citation:
                        amended_line += " (This information is not cited in the provided documents.)"
            amended_lines.append(amended_line)

        return "\n".join(amended_lines)

    def _smooth_direct_answer(self, response_text: str) -> str:
        """Use a lightweight LLM to polish the direct answer while preserving all facts."""

        lines = response_text.splitlines()
        direct_idx = next(
            (idx for idx, line in enumerate(lines)
             if line.strip().lower().startswith("direct answer:")),
            None,
        )
        if direct_idx is None:
            return response_text

        # Check if scope validator has already rewritten the answer
        if self._scope_validation_applied:
            print("DEBUG: Scope validator has rewritten the answer, skipping smoothing")
            return response_text

        prefix = "Direct Answer:"
        direct_line = lines[direct_idx]
        body = direct_line[len(prefix):].strip() if direct_line.startswith(prefix) else direct_line.strip()
        if not body:
            return response_text

        def _strip_refs(text: str) -> str:
            cleaned = re.sub(r"\(Document [^)]*\)", "", text)
            cleaned = re.sub(r"\(tool_[^)]*\)", "", cleaned)
            return re.sub(r"\s+", " ", cleaned).strip()

        body_clean = _strip_refs(body)

        supporting_lines = self._collect_supporting_lines(lines)
        supporting_block = "\n".join(
            f"{idx + 1}. {self._clean_fact_for_direct_answer(line)}"
            for idx, line in enumerate(supporting_lines)
        )

        prompt = (
            "You rewrite direct answers for clarity and completeness without adding new information.\n"
            f"Original question: {self.user_input or ''}\n"
            f"Original direct answer: {body_clean}\n"
            "Supporting sentences:\n"
            f"{supporting_block or 'None provided.'}\n\n"
            "Rewrite the direct answer so that:\n"
            "- The very first sentence explicitly answers the question without restating the question; phrase the answer naturally.\n"
            "- All key subject/attribute terms from the question remain present in the answer.\n"
            "- Every fact, entity, number, date, and relationship from the original direct answer is preserved exactly as stated.\n"
            "- Redundant clauses are removed and the sentence flow is natural, grammatical English.\n"
            "- NEVER remove any factual clarifications or explanatory clauses (e.g., sentences explaining what XXX stands for XXX).\n"
            "- For interrogative or comparative questions (for example, 'Which company...' or 'Which university...'), restate the answer as a declarative sentence such as 'The company is Standard Oil...' while keeping all necessary terms.\n"
            "- Keep the rewritten direct answer concise: use no more than two sentences and ensure the second sentence adds new supporting detail rather than repeating the first; explanations like 'XXX stands for YYY' may be placed in Supporting sentences.\n"
            "- If the provided sentences do not contain the requested attribute, explicitly state that the information is not available and briefly note what the sources do describe.\n"
            "- All content must come from the existing direct answer or supporting sentences; do NOT add new facts, interpretations, or citations, and do not mention document identifiers.\n"
            "- IMPORTANT: Ensure all quotation marks are properly balanced and closed. If you include quoted text, make sure every opening quote has a matching closing quote.\n\n"
            "Return only the rewritten direct answer text."
        )

        try:
            response = self.validator_extractor_llm.complete(prompt=prompt)
            rewritten = (response.text or "").strip()
            print(f"DEBUG: LLM returned: {repr(rewritten)}")
        except Exception as exc:
            print(f"Direct answer smoothing failed: {exc}")
            rewritten = ""

        if rewritten.lower().startswith("direct answer:"):
            rewritten = rewritten[len(prefix):].strip()

        rewritten_line = rewritten.splitlines()[0].strip() if rewritten else ""
        if not rewritten_line:
            rewritten_line = body_clean

        reference_text = " ".join([body_clean] + supporting_lines)
        original_tokens = set(re.findall(r"\w+", reference_text.lower()))
        rewritten_tokens = set(re.findall(r"\w+", rewritten_line.lower()))
        if original_tokens and len(rewritten_tokens & original_tokens) < max(1, len(original_tokens) // 4):
            return response_text

        rewritten_line = re.sub(r"\s*\(Document [^)]*\)", "", rewritten_line)
        rewritten_line = re.sub(r"\s*\(tool_[^)]*\)", "", rewritten_line)
        rewritten_line = re.sub(r"\s*\[\s*\]\s*", " ", rewritten_line).strip()
        rewritten_line = re.sub(r"\s+", " ", rewritten_line).strip()
        # Apply quote balancing BEFORE removing quotes
        def _balance_quotes_early(text: str) -> str:
            if not text:
                return text
            # Don't balance quotes if text already ends with punctuation
            if text.rstrip().endswith(('.', '!', '?', ';', ':')):
                return text
            single_count = text.count("'")
            double_count = text.count('"')
            smart_single_left = text.count('\u2018')
            smart_single_right = text.count('\u2019')
            smart_double_left = text.count('\u201c')
            smart_double_right = text.count('\u201d')
            if single_count % 2 == 1:
                text += "'"
            if double_count % 2 == 1:
                text += '"'
            if smart_single_left > smart_single_right:
                text += '\u2019'
            if smart_double_left > smart_double_right:
                text += '\u201d'
            return text

        print(f"DEBUG: Before early balance: {repr(rewritten_line)}")
        rewritten_line = _balance_quotes_early(rewritten_line)
        print(f"DEBUG: After early balance: {repr(rewritten_line)}")

        # Smart quote removal: only remove quotes that don't have proper pairs
        # Instead of blindly removing all quotes, check if they're properly paired

        # For the beginning: if it starts with a quote, check if there's a matching closing quote
        if rewritten_line and rewritten_line[0] in ['"', "'", '"', "'"]:
            start_quote = rewritten_line[0]
            # Find matching closing quote
            if start_quote == '"':
                close_quote = '"'
            elif start_quote == "'":
                close_quote = "'"
            elif start_quote == '"':
                close_quote = '"'
            elif start_quote == "'":
                close_quote = "'"

            # Check if there's a matching closing quote somewhere in the text
            close_idx = rewritten_line.find(close_quote, 1)
            if close_idx == -1:
                # No matching closing quote, remove the opening quote
                rewritten_line = rewritten_line[1:].lstrip()

        # For the end: if it ends with a quote, check if it's truly unmatched
        if rewritten_line and rewritten_line[-1] in ['"', "'", '"', "'"]:
            end_quote = rewritten_line[-1]
            # For double quotes: check if there's a matching opening quote
            if end_quote == '"':
                open_count = rewritten_line[:-1].count('"')
                if open_count % 2 == 0:  # Even number means the trailing quote is unmatched
                    rewritten_line = rewritten_line[:-1].rstrip()
            # For single quotes: check if total count (including trailing) is odd
            elif end_quote == "'":
                total_single_quotes = rewritten_line.count("'")
                if total_single_quotes % 2 == 1:  # Odd number means the trailing quote is unmatched
                    rewritten_line = rewritten_line[:-1].rstrip()

        # Skip the problematic quote removal logic that was breaking valid quote pairs
        # This logic was incorrectly removing valid closing quotes followed by punctuation

        # Re-balance after quote removal
        print(f"DEBUG: After quote removal: {repr(rewritten_line)}")
        rewritten_line = _balance_quotes_early(rewritten_line)
        print(f"DEBUG: After re-balance: {repr(rewritten_line)}")
        rewritten_line = re.sub(r'([.!?])\1+', r'\1', rewritten_line)
        rewritten_line = rewritten_line.strip()

        attr_keyword_local = self._last_attribute_keyword or self._infer_attribute_keyword()
        question_starts = re.compile(r"^(which|what|who|where|when|why|how|whom|whose)\b", re.IGNORECASE)
        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", rewritten_line) if segment.strip()]
        normalized_sentences: list[tuple[str, str]] = []
        for segment in segments:
            cleaned = segment.strip('"“”').rstrip('.!?').lower()
            normalized_sentences.append((segment, cleaned))

        if normalized_sentences and question_starts.match(normalized_sentences[0][1]) and len(normalized_sentences) > 1:
            normalized_sentences = normalized_sentences[1:]
            rewritten_line = " ".join(seg for seg, _ in normalized_sentences)
        if question_starts.match(rewritten_line.split()[0].strip('"“”').lower()) and attr_keyword_local:
            rewritten_line = rewritten_line.lstrip('"“”')
            rewritten_line = f"The {attr_keyword_local.strip()} is {rewritten_line}".strip()
            normalized_sentences = [(rewritten_line, rewritten_line.strip('"“”').rstrip('.!?').lower())]

        deduped_sentences: list[str] = []
        seen_norms: set[str] = set()
        for original, normalized in normalized_sentences:
            if normalized not in seen_norms:
                deduped_sentences.append(original)
                seen_norms.add(normalized)
        if not deduped_sentences and rewritten_line:
            deduped_sentences = [rewritten_line]
        rewritten_line = " ".join(deduped_sentences)

        # Only add period if text doesn't already end with punctuation
        if rewritten_line and rewritten_line.rstrip(' \t\n\r\f\v')[-1] not in '.!?':
            stripped = rewritten_line.rstrip()
            if len(stripped) >= 2 and stripped[-1] in {'"', '”', "'", '’'} and stripped[-2] in '.!?':
                pass  # already has punctuation before ending quote
            else:
                rewritten_line += "."

        # Remove period if it's followed by a quote (cleanup for cases like .'")
        rewritten_line = re.sub(r"""\.\s*(["'’])""", r"\1", rewritten_line)

        # If there's only one sentence, insert a period between the first closing quote and the second clause.
        if rewritten_line.count('.') <= 1:
            idx = rewritten_line.find('" The ')
            if idx != -1:
                rewritten_line = rewritten_line[:idx+1] + "." + rewritten_line[idx+1:].lstrip()

        def _trim_lonely_trailing_quote(text: str) -> str:
            quote_rules = [
                ("'", {"'", "’", "‘"}),
                ("’", {"'", "’", "‘"}),
                ('"', {'"', '“', '”'}),
                ('”', {'"', '“', '”'}),
            ]
            for trailing, partners in quote_rules:
                if text.endswith(trailing):
                    body = text[:-1]
                    if not any(partner in body for partner in partners):
                        return body.rstrip()
            return text

        rewritten_line = _trim_lonely_trailing_quote(rewritten_line)
        # Add period if text ends with quote
        if (rewritten_line.endswith('"') or rewritten_line.endswith("'") or
            rewritten_line.endswith('\u201d') or rewritten_line.endswith('\u2019')):
            rewritten_line += '.'

        # Clean up punctuation inside quotes at the end of text
        # Fix cases like "text"." -> "text".
        if len(rewritten_line) >= 2:
            # Find the last quote
            last_quote_pos = max(
                rewritten_line.rfind('"'), rewritten_line.rfind("'"),
                rewritten_line.rfind('\u201d'), rewritten_line.rfind('\u2019')
            )
            if last_quote_pos > 0 and rewritten_line[last_quote_pos - 1] in '.!?':
                # Move punctuation outside the quote
                punctuation = rewritten_line[last_quote_pos - 1]
                rewritten_line = (rewritten_line[:last_quote_pos - 1] +
                                rewritten_line[last_quote_pos] + punctuation)

        # Final cleanup: fix malformed quote patterns at the end
        if rewritten_line.endswith((".'", '."')):
            # Check if this quote has a matching opening quote in the entire text
            quote_char = rewritten_line[-2]  # The quote character
            total_quotes = rewritten_line.count(quote_char)
            if total_quotes % 2 == 1:  # Odd number means the trailing quote is unmatched
                # Remove the trailing quote, keep the punctuation
                rewritten_line = rewritten_line[:-2] + rewritten_line[-1]

        print(f"DEBUG: _smooth_direct_answer final result: {repr(rewritten_line)}")

        lines[direct_idx] = f"{prefix} {rewritten_line}"
        return "\n".join(lines)

    def _annotate_direct_answer_uncited(self, response_text: str) -> str:
        """If Supporting Details include uncited assumptions, disclose that in Direct Answer."""

        lines = response_text.splitlines()
        supporting_section = False
        needs_annotation = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("supporting details"):
                supporting_section = True
                continue
            if supporting_section and stripped.startswith("-"):
                lowered = stripped.lower()
                if "not cited in the provided documents" in lowered:
                    needs_annotation = True
                    break

        if not needs_annotation:
            return response_text

        annotated_lines: list[str] = []
        annotated = False

        for line in lines:
            stripped = line.strip()
            if not annotated and stripped.lower().startswith("direct answer:"):
                if "not cited in the provided documents" not in stripped.lower():
                    suffix = " (This conclusion relies on information that is not cited in the provided documents.)"
                    if line.endswith("."):
                        line = line + suffix
                    else:
                        line = line + suffix
                annotated = True
            annotated_lines.append(line)

        return "\n".join(annotated_lines)

    def _extract_verified_facts_from_combined(self) -> list[tuple[str, str]]:
        """Extract facts from combined response, prioritizing verbatim document content over rephrased facts."""
        if not self.combined_response:
            return []

        match = re.search(
            r"##\s*Verified Facts List(?P<section>.*?)(?:\n##\s|$)",
            self.combined_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return []

        section = match.group("section")
        facts: list[tuple[str, str]] = []
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fact_match = re.match(r"-\s+(.*)", stripped)
            if fact_match:
                original = fact_match.group(1).strip()
                canonical = original
                if "(Evidence:" in canonical:
                    canonical = canonical.split("(Evidence:", 1)[0].strip()
                facts.append((canonical, original))
        return facts

    def _reinforce_verified_facts(self, response_text: str) -> str:
        facts = self._extract_verified_facts_from_combined()
        if not facts:
            return response_text

        response_lower = response_text.lower()
        missing: list[str] = []
        for canonical, original in facts:
            if canonical and canonical.lower() not in response_lower:
                missing.append(original)

        if not missing:
            return response_text

        print(f"Guardrail: reinstating {len(missing)} verified facts in subCoA output.")
        return self._append_fact_echo(response_text, missing)

    def _append_fact_echo(self, response_text: str, missing: list[str]) -> str:
        if not missing:
            return response_text

        lines = response_text.rstrip().splitlines()
        doc_pattern = re.compile(r"Document\s+([0-9a-z_]+)", re.IGNORECASE)

        def _canonical_entry(text: str) -> Optional[tuple[str, str]]:
            match = doc_pattern.search(text)
            if not match:
                return None
            doc_id = match.group(1).lower()
            cleaned = text
            # Remove leading bullet markers and surrounding brackets for normalization.
            cleaned = cleaned.lstrip("-• ").strip()
            if cleaned.lower().startswith(f"document {doc_id}:"):
                cleaned = cleaned.split(":", 1)[1].strip()
            elif cleaned.lower().startswith(f"[document {doc_id}]:"):
                cleaned = cleaned.split("]:", 1)[1].strip()
            cleaned = re.sub(r"\s+", " ", cleaned).lower()
            return doc_id, cleaned

        existing_keys: set[tuple[str, str]] = set()
        for line in lines:
            canonical = _canonical_entry(line)
            if canonical:
                existing_keys.add(canonical)

        filtered_missing: list[str] = []
        for fact in missing:
            canonical = _canonical_entry(fact)
            if canonical and canonical not in existing_keys:
                existing_keys.add(canonical)
                filtered_missing.append(fact)

        if not filtered_missing:
            return response_text

        if lines and lines[-1].strip():
            lines.append("")
        lines.append("Facts (auto-complete):")
        lines.extend(f"- {fact}" for fact in filtered_missing)
        return "\n".join(lines) + "\n"

    def _dedupe_supporting_doc_citations(self, response_text: str) -> str:
        """Collapse duplicate document identifiers within citation groups while preserving order."""

        pattern = re.compile(
            r"\((?P<body>(?:\s*[A-Za-z0-9-]+_[0-9]+\s*(?:[;,]\s*)?)+)\)"
        )

        def _replacer(match: re.Match[str]) -> str:
            body = match.group("body")
            tokens = re.split(r"[;,]", body)
            seen: set[str] = set()
            deduped: list[str] = []
            for token in tokens:
                cleaned = token.strip()
                if not cleaned:
                    continue
                if cleaned not in seen:
                    deduped.append(cleaned)
                    seen.add(cleaned)
            if not deduped:
                return match.group(0)
            return "(" + "; ".join(deduped) + ")"

        return pattern.sub(_replacer, response_text)

    def _wrap_bare_citations(self, response_text: str) -> str:
        """Ensure every bare document id is wrapped in parentheses."""

        citation_pattern = re.compile(r"(?P<id>[A-Za-z0-9-]+_[0-9]+)")

        def _wrap(match: re.Match[str]) -> str:
            start = match.start()
            end = match.end()
            source = match.string

            if start > 0 and source[start - 1] == "(":
                return match.group("id")
            if end < len(source) and source[end] == ")":
                return match.group("id")
            return f"({match.group('id')})"

        return citation_pattern.sub(_wrap, response_text)

    def _inject_supporting_sentences(
        self, lines: list[str], doc_ids: Iterable[str]
    ) -> tuple[list[str], bool]:
        """Append supporting sentences for the specified document ids if available."""

        if not doc_ids or not self._fact_sentence_map:
            return lines, False

        updated = lines[:]
        added = False

        existing_doc_ids = set(self._DOC_ID_PATTERN.findall("\n".join(updated)))

        support_idx = next(
            (idx for idx, line in enumerate(updated)
             if line.strip().lower().startswith("supporting details")),
            None,
        )
        if support_idx is None:
            updated.append("Supporting Details:")
            support_idx = len(updated) - 1

        insert_pos = support_idx + 1
        existing_bullets = {line.strip() for line in updated if line.strip().startswith("-")}

        for raw_doc_id in doc_ids:
            doc_id = (raw_doc_id or "").strip()
            if not doc_id or doc_id in existing_doc_ids:
                continue

            sentences = self._fact_sentence_map.get(doc_id)
            if not sentences:
                continue

            sentence = sentences[0].strip()
            if not sentence:
                continue

            bullet_text = sentence if sentence.startswith("-") else f"- {sentence}"
            if doc_id not in bullet_text:
                if bullet_text.endswith("."):
                    bullet_text = f"{bullet_text[:-1]} ({doc_id})."
                else:
                    bullet_text = f"{bullet_text} ({doc_id})"

            normalized = bullet_text.strip()
            if normalized in existing_bullets:
                continue

            updated.insert(insert_pos, bullet_text)
            insert_pos += 1
            existing_bullets.add(normalized)
            existing_doc_ids.add(doc_id)
            added = True

        return updated, added

    def _extract_scope_qualifiers(self, text: str) -> Dict[str, List[str]]:
        """Extract scope qualifiers from text (temporal, geographic, conditional, etc.)"""
        scope_qualifiers = {
            'temporal': [],
            'geographic': [],
            'conditional': [],
            'role_based': []
        }

        text_lower = text.lower()

        # Extract temporal qualifiers (years, dates, time periods)
        import re
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text_lower)
        scope_qualifiers['temporal'].extend(year_matches)

        # Extract temporal phrases
        temporal_phrases = [
            r'\b(as of \d{4})\b',
            r'\b(since \d{4})\b',
            r'\b(until \d{4})\b',
            r'\b(during \w+)\b',
            r'\b(in \d{4})\b',
            r'\b(\d{4}[-/]\d{4})\b'  # date ranges
        ]

        for pattern in temporal_phrases:
            matches = re.findall(pattern, text_lower)
            scope_qualifiers['temporal'].extend(matches)

        # Extract geographic qualifiers
        geo_phrases = [
            r'\b(in \w+)\b',
            r'\b(within \w+)\b',
            r'\b(at \w+)\b',
            r'\b(\w+ region)\b',
            r'\b(\w+ area)\b'
        ]

        for pattern in geo_phrases:
            matches = re.findall(pattern, text_lower)
            # Filter out common non-geographic words
            filtered_matches = [m for m in matches if not any(word in m for word in ['the', 'a', 'an', 'this', 'that'])]
            scope_qualifiers['geographic'].extend(filtered_matches)

        # Extract conditional qualifiers
        conditional_phrases = [
            r'\b(under \w+ conditions?)\b',
            r'\b(during \w+)\b',
            r'\b(under \w+)\b',
            r'\b(with \w+)\b',
            r'\b(without \w+)\b'
        ]

        for pattern in conditional_phrases:
            matches = re.findall(pattern, text_lower)
            scope_qualifiers['conditional'].extend(matches)

        # Remove duplicates and empty lists
        for key in scope_qualifiers:
            scope_qualifiers[key] = list(set(scope_qualifiers[key]))
            if not scope_qualifiers[key]:
                del scope_qualifiers[key]

        return scope_qualifiers

    def _enforce_attribute_answer(self, response_text: str) -> str:
        """Ensure the direct answer explicitly states the requested attribute when evidence exists."""

        print(f"DEBUG: _enforce_attribute_answer called with response_text length: {len(response_text)}")

        # Check if scope validator has already rewritten the answer
        if self._scope_validation_applied:
            print("DEBUG: Scope validator has rewritten the answer, skipping enforcement")
            return response_text

        attr_keyword = self._last_attribute_keyword or self._infer_attribute_keyword()
        print(f"DEBUG: attr_keyword = {repr(attr_keyword)}")
        if not attr_keyword:
            print("DEBUG: No attr_keyword, returning early")
            return response_text

        lines = response_text.splitlines()
        print(f"DEBUG: Found {len(lines)} lines")
        for i, line in enumerate(lines):
            if "direct answer" in line.lower():
                print(f"DEBUG: Line {i}: {repr(line[:100])}")

        direct_idx = next(
            (idx for idx, line in enumerate(lines)
             if line.strip().lower().startswith("direct answer:")),
            None,
        )
        print(f"DEBUG: direct_idx = {direct_idx}")
        if direct_idx is None:
            print("DEBUG: No direct answer line found, returning early")
            return response_text

        direct_body = lines[direct_idx].split(":", 1)[-1].strip()
        print(f"DEBUG: direct_body = {repr(direct_body)}")
        print(f"DEBUG: checking if '{attr_keyword.lower()}' in '{direct_body.lower()}' = {attr_keyword.lower() in direct_body.lower()}")

        # Check if we need to reconstruct the answer
        needs_reconstruction = attr_keyword.lower() not in direct_body.lower()
        print(f"DEBUG: needs_reconstruction = {needs_reconstruction} (keyword: {attr_keyword})")

        new_sentence = direct_body  # Start with current direct_body

        if needs_reconstruction:
            print("DEBUG: Starting answer reconstruction...")
            supporting_lines = self._collect_supporting_lines(lines)
            attr_line = next(
                (line for line in supporting_lines
                 if attr_keyword.lower() in line.lower()),
                None,
            )
            if not attr_line and supporting_lines:
                attr_line = supporting_lines[0]
            if not attr_line:
                print("DEBUG: No attr_line found, skipping reconstruction")
            else:
                print(f"DEBUG: attr_line before cleaning: {repr(attr_line)}")
                cleaned_attr = self._clean_fact_for_direct_answer(attr_line)
                print(f"DEBUG: attr_line after cleaning: {repr(cleaned_attr)}")
                if cleaned_attr:
                    # Reconstruct the answer using the cleaned fact
                    # This is a simplified reconstruction - just use the cleaned fact directly
                    new_sentence = cleaned_attr.rstrip('.')
                else:
                    print("DEBUG: cleaned_attr is empty, keeping original")
        else:
            print("DEBUG: Skipping reconstruction, using existing direct_body")

        print(f"DEBUG: new_sentence before quote processing: {repr(new_sentence)}")

        # Apply comprehensive quote balancing
        def _strip_orphan_quotes(text: str) -> str:
            if not text:
                return text

            # Define all quote types
            smart_lsq = '\u2018'  # Left single smart quote '
            smart_rsq = '\u2019'  # Right single smart quote '
            smart_ldq = '\u201c'  # Left double smart quote "
            smart_rdq = '\u201d'  # Right double smart quote "
            ascii_sq = "'"        # ASCII single quote '
            ascii_dq = '"'        # ASCII double quote "

            # Opening quotes and their corresponding closing quotes
            quote_pairs = {
                smart_lsq: smart_rsq,    # ' → '
                smart_ldq: smart_rdq,    # " → "
                ascii_sq: ascii_sq,      # ' → '
                ascii_dq: ascii_dq,      # " → "
            }

            result = []
            i = 0
            while i < len(text):
                char = text[i]

                # Check if this is an opening quote
                if char in quote_pairs:
                    # Treat apostrophes embedded within words as literal characters, not quotes
                    if char in {ascii_sq, smart_rsq}:
                        prev_char = text[i - 1] if i > 0 else ""
                        next_char = text[i + 1] if i + 1 < len(text) else ""
                        if prev_char.isalnum() and next_char.isalnum():
                            result.append(char)
                            i += 1
                            continue

                    opening_quote = char
                    closing_quote = quote_pairs[opening_quote]

                    result.append(char)
                    found_closing = False

                    # Look for the matching closing quote
                    for j in range(i + 1, len(text)):
                        if text[j] == closing_quote:
                            # Found matching closing quote
                            result.extend(text[i+1:j+1])
                            i = j
                            found_closing = True
                            break
                        elif text[j] in '.!?;,':  # Hit punctuation - quote should close here
                            break

                    if not found_closing:
                        # No closing quote found - add one before the next punctuation or at end
                        content_after = text[i+1:]
                        punctuation_found = False
                        for k, c in enumerate(content_after):
                            if c in '.!?;,':
                                # Check if there's already a closing quote before this punctuation
                                text_before_punct = content_after[:k]
                                if closing_quote not in text_before_punct:
                                    # Insert closing quote before punctuation only if not already there
                                    result.extend(content_after[:k])
                                    result.append(closing_quote)
                                    result.append(c)
                                    result.extend(content_after[k+1:])
                                else:
                                    # Closing quote already exists before punctuation
                                    result.extend(content_after)
                                punctuation_found = True
                                break
                        if not punctuation_found:
                            # No punctuation found, append at end
                            result.extend(content_after)
                            result.append(closing_quote)
                        i = len(text)  # Skip to end
                    else:
                        i += 1
                else:
                    result.append(char)
                    i += 1

            text = ''.join(result)

            # Clean up any remaining unmatched opening quotes at the start
            opening_quotes = set(quote_pairs.keys())
            while text and text[0] in opening_quotes:
                char = text[0]
                close_char = quote_pairs[char]
                close_idx = text.find(close_char, 1)
                if close_idx == -1:
                    text = text[1:].lstrip()
                else:
                    break

            return text

        new_sentence = _strip_orphan_quotes(new_sentence)
        print(f"DEBUG: _enforce_attribute_answer after comprehensive quote balancing: {repr(new_sentence)}")

        # Final quote closure check
        single_count = new_sentence.count("'")
        double_count = new_sentence.count('"')
        smart_single_left = new_sentence.count('\u2018')
        smart_single_right = new_sentence.count('\u2019')
        smart_double_left = new_sentence.count('\u201c')
        smart_double_right = new_sentence.count('\u201d')

        # If we have odd counts, add closing quotes
        if single_count % 2 == 1:
            new_sentence += "'"
        if double_count % 2 == 1:
            new_sentence += '"'
        if smart_single_left > smart_single_right:
            new_sentence += '\u2019'
        if smart_double_left > smart_double_right:
            new_sentence += '\u201d'

        print(f"DEBUG: _enforce_attribute_answer after final quote closure: {repr(new_sentence)}")

        lines[direct_idx] = f"Direct Answer: {new_sentence}"
        return "\n".join(lines)

    def _apply_scope_validator(self, response_text: str) -> str:
        """Use a lightweight LLM to detect scope or temporal mismatches and rewrite only the Direct Answer when required."""

        lines = response_text.splitlines()
        if not lines:
            return response_text

        def _strip_orphan_quotes(text: str) -> str:
            if not text:
                return text

            # Define all quote types
            smart_lsq = '\u2018'  # Left single smart quote '
            smart_rsq = '\u2019'  # Right single smart quote '
            smart_ldq = '\u201c'  # Left double smart quote "
            smart_rdq = '\u201d'  # Right double smart quote "
            ascii_sq = "'"        # ASCII single quote '
            ascii_dq = '"'        # ASCII double quote "

            # Opening quotes and their corresponding closing quotes
            quote_pairs = {
                smart_lsq: smart_rsq,    # ' → '
                smart_ldq: smart_rdq,    # " → "
                ascii_sq: ascii_sq,      # ' → '
                ascii_dq: ascii_dq,      # " → "
            }

            result = []
            i = 0
            while i < len(text):
                char = text[i]

                # Check if this is an opening quote
                if char in quote_pairs:
                    # Treat apostrophes embedded within words as literal characters, not quotes
                    if char in {ascii_sq, smart_rsq}:
                        prev_char = text[i - 1] if i > 0 else ""
                        next_char = text[i + 1] if i + 1 < len(text) else ""
                        if prev_char.isalnum() and next_char.isalnum():
                            result.append(char)
                            i += 1
                            continue

                    opening_quote = char
                    closing_quote = quote_pairs[opening_quote]

                    result.append(char)
                    found_closing = False

                    # Look for the matching closing quote
                    for j in range(i + 1, len(text)):
                        if text[j] == closing_quote:
                            # Found matching closing quote
                            result.extend(text[i+1:j+1])
                            i = j
                            found_closing = True
                            break
                        elif text[j] in '.!?;,':  # Hit punctuation - quote should close here
                            break

                    if not found_closing:
                        # No closing quote found - add one before the next punctuation or at end
                        content_after = text[i+1:]
                        punctuation_found = False
                        for k, c in enumerate(content_after):
                            if c in '.!?;,':
                                # Check if there's already a closing quote before this punctuation
                                text_before_punct = content_after[:k]
                                if closing_quote not in text_before_punct:
                                    # Insert closing quote before punctuation only if not already there
                                    result.extend(content_after[:k])
                                    result.append(closing_quote)
                                    result.append(c)
                                    result.extend(content_after[k+1:])
                                else:
                                    # Closing quote already exists before punctuation
                                    result.extend(content_after)
                                punctuation_found = True
                                break
                        if not punctuation_found:
                            # No punctuation found, append at end
                            result.extend(content_after)
                            result.append(closing_quote)
                        i = len(text)  # Skip to end
                    else:
                        i += 1
                else:
                    result.append(char)
                    i += 1

            text = ''.join(result)

            # Clean up any remaining unmatched opening quotes at the start
            opening_quotes = set(quote_pairs.keys())
            while text and text[0] in opening_quotes:
                char = text[0]
                close_char = quote_pairs[char]
                close_idx = text.find(close_char, 1)
                if close_idx == -1:
                    text = text[1:].lstrip()
                else:
                    break

            return text

        direct_idx = next(
            (idx for idx, line in enumerate(lines)
             if line.strip().lower().startswith("direct answer:")),
            None,
        )
        if direct_idx is None:
            return response_text

        direct_line = lines[direct_idx]
        direct_body = direct_line.split(":", 1)[-1].strip() if ":" in direct_line else direct_line.strip()
        if not direct_body:
            return response_text

        support_idx = next(
            (idx for idx, line in enumerate(lines)
             if line.strip().lower().startswith("supporting details")),
            None,
        )
        if support_idx is None:
            return response_text

        supporting_section = "\n".join(lines[support_idx:]).strip()
        if not supporting_section:
            return response_text

        attribute_keyword = self._infer_attribute_keyword()
        self._last_attribute_keyword = attribute_keyword
        attempts = 0
        current_body = direct_body

        while attempts < 2:
            validator_prompt = self._build_validator_prompt(
                current_body,
                supporting_section,
                attribute_keyword,
                attempts > 0 and attribute_keyword is not None,
            )

            try:
                validation = self.validator_llm.complete(prompt=validator_prompt)
            except Exception as exc:
                print(f"Scope validator call failed: {exc}")
                return response_text

            payload = self._extract_validator_json(validation.text)
            if not payload:
                fallback_body = validation.text.strip() if validation.text else ""
                if fallback_body:
                    cleaned_fallback = self._strip_direct_answer(fallback_body)
                    if cleaned_fallback:
                        lines[direct_idx] = f"Direct Answer: {cleaned_fallback}"
                        return "\n".join(lines)
                return response_text

            status = str(payload.get("status", "")).lower()
            body = str(payload.get("body", "")).strip()
            reason = str(payload.get("reason", "")).strip()
            note_value = payload.get("normalization_note", "")
            if isinstance(note_value, str):
                normalization_note = note_value.strip()
            else:
                normalization_note = ""

            if status not in {"valid", "rewrite"} or not body:
                return response_text


            if status == "rewrite":
                missing_doc_ids = self._DOC_ID_PATTERN.findall(reason) if reason else []
                if missing_doc_ids:
                    updated_lines, added = self._inject_supporting_sentences(lines, missing_doc_ids)
                    if added:
                        lines = updated_lines
                        direct_idx = next(
                            (idx for idx, line in enumerate(lines)
                             if line.strip().lower().startswith("direct answer:")),
                            None,
                        )
                        support_idx = next(
                            (idx for idx, line in enumerate(lines)
                             if line.strip().lower().startswith("supporting details")),
                            None,
                        )
                        if direct_idx is not None:
                            current_body = lines[direct_idx].split(":", 1)[-1].strip()
                        if support_idx is not None:
                            supporting_section = "\n".join(lines[support_idx:]).strip()
                        else:
                            supporting_section = ""
                        print(f"Scope validator: added supporting details for {', '.join(missing_doc_ids)}")
                        continue
                if reason:
                    print(f"Scope validator rewrite applied: {reason}")
                    print(f"DEBUG: Rewritten body: {repr(body)}")

            if (
                attribute_keyword
                and attribute_keyword.lower() not in body.lower()
                and attempts == 0
            ):
                print(
                    f"Scope validator missing attribute '{attribute_keyword}'; requesting rewrite."
                )
                current_body = body
                attempts += 1
                continue

            if normalization_note:
                if body.endswith("."):
                    body = f"{body} {normalization_note}"
                else:
                    body = f"{body}. {normalization_note}"

            raw_body = body
            print(f"DEBUG: Before scope validator processing - supporting_section: {repr(supporting_section[:200])}...")

            body = self._strip_direct_answer(body)
            body = _strip_orphan_quotes(body)
            body = re.sub(r"\s*([.!?])\s*(?=\1)", r"\1", body)
            body = re.sub(r"([.!?]){2,}", r"\1", body)
            if not body.strip():
                body = raw_body.strip()

            print(f"DEBUG: After scope validator processing - supporting_section: {repr(supporting_section[:200])}...")

            lines[direct_idx] = f"Direct Answer: {body}"
            self._scope_validation_applied = True  # Mark that scope validation has rewritten the answer
            return "\n".join(lines)

        return response_text

    @staticmethod
    def _extract_validator_json(raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    def _build_validator_prompt(
        self,
        direct_body: str,
        supporting_section: str,
        attribute_keyword: Optional[str],
        force_rewrite: bool,
    ) -> str:
        attribute_line = (
            f"Requested attribute keyword: {attribute_keyword}"
            if attribute_keyword
            else "Requested attribute keyword: None"
        )
        comparison_guideline = (
            "If the request implies a comparison (e.g., contains words such as 'higher', 'greater', 'more', 'earlier', 'later', 'north'), you may declare which option satisfies the comparison when the supporting details either (a) explicitly describe the ordering, (b) provide concrete values that make the ordering self-evident (such as numeric totals or measurements), or (c) **CRITICAL: DEFINITIONAL LOGIC** - describe entities with inherent functional distinctions that determine the outcome (e.g., one designed/equipped for the compared activity, the other explicitly not designed/equipped for it). For case (c), the comparison is resolved by definition and you MUST NOT say 'not confirmed'. Otherwise, the direct answer must state that the comparison is not confirmed."
        )
        sentence_requirement = (
            "The direct answer body must form complete sentence(s) that explicitly name the subject referenced in the question and state the attribute value or its limitation. The wording must be declarative and MUST NOT repeat the original question text or pose a new question. If any portion of the draft remains phrased as a question, remove the question wording entirely and restate the answer as a standalone declarative fact. Do not use meta-answer constructions such as 'is answered by', 'the answer is', or 'this question is addressed by'; state the fact directly. **CRITICAL: NEVER copy interrogative words (what, which, who, when, where, how, whose, whom) from the question into the answer. Replace interrogative phrases with the actual answer values.** For example, transform '[interrogative word] [attribute]' patterns into direct statements containing only the answer value for that attribute, without including the interrogative word itself."
        )
        explicit_naming = (
            "Every sentence in the direct answer must restate each subject by its explicit name or descriptor; do not rely on pronouns such as 'it', 'they', or 'this event' to stand in for the compared subjects."
        )
        attribute_requirement = (
            f"If the attribute keyword is provided, the direct answer body should include that keyword or clearly address the attribute concept. **CRITICAL: If the attribute keyword contains interrogative words (what, which, who, when, where, how, whose, whom), you MUST replace them with the actual answer value, NOT copy them verbatim into the answer. ** Extract the attribute type being asked about (e.g., if the keyword contains '[interrogative] [attribute type]', identify the attribute type), then provide the answer value for that attribute type in a natural declarative sentence. INTEGRATE the keyword naturally into a complete, grammatical sentence without adding extra quotes or quotation marks around supplementary information. For example, if adding 'a X county', write 'a X county' not ''a X county''. For comparative contexts in documents (e.g., 'Person X developed something with as much rigor as his contemporaries, including Person Y'), you may reasonably conclude that Person Y shares the attribute. However, when asserting that multiple subjects share the *same* occupation or descriptor, you may only reuse the identical wording if the supporting details explicitly supply that wording for each subject; otherwise collapse the shared attribute to the overlapping phrase (e.g., both are 'directors') and keep any extra modifiers (such as 'film') attached only to the subject whose evidence includes them. Only declare the attribute as 'not confirmed' if the answer genuinely lacks any relevant information about the requested attribute. {comparison_guideline} {sentence_requirement} {explicit_naming}"
            if attribute_keyword
            else f"If the documents cannot confirm the requested attribute, the direct answer must state that limitation. {comparison_guideline} {sentence_requirement} {explicit_naming}"
        )
        force_instruction = (
            f"The previous attempt failed to include the attribute keyword. You must output status \"rewrite\" with a corrected body that satisfies all requirements. **REMINDER: If the attribute keyword contains interrogative words, extract the attribute type being asked about and provide the answer value for that attribute type. Do NOT copy interrogative words into the answer.**"
            if force_rewrite and attribute_keyword
            else ""
        )

        prompt = f"""
You are a response validator focused on scope and timeframe alignment.

CRITICAL VALIDATION RULES (HIGHEST PRIORITY):

MANDATORY DEFINITIONAL LOGIC FOR COMPARISONS (ABSOLUTE PRIORITY):
- **CRITICAL**: For comparison queries, when supporting details describe entities with inherent functional/design distinctions that determine the comparison outcome, the direct answer MUST apply definitional logic and provide the conclusion.
- **FUNCTIONAL DISTINCTIONS**: When supporting details show one entity is designed/equipped/capable for the compared activity and the other is explicitly not designed/equipped/capable for it, this resolves the comparison by definition.
- **ABSOLUTELY FORBIDDEN**: Validating a direct answer that says "cannot be determined" or "insufficient information" when supporting details provide definitional characteristics that resolve the comparison (e.g., armed vs unarmed for combat comparison, combat-role vs non-combat-role, designed-for vs not-designed-for).
- **REWRITE REQUIREMENT**: If direct answer fails to apply definitional logic when supporting details provide functional distinctions, you MUST rewrite to state the definitive conclusion that the definitions require.
- **VALIDATION CHECK**: ALWAYS check for definitional logic violations FIRST, before all other validations.

REASONABLE COMMON-SENSE INFERENCE (CRITICAL):
- **NATIONALITY → BIRTHPLACE INFERENCE (ALLOWED)**: When the query asks about birthplace/birth location and the supporting details state a person is "[nationality] [profession]" (e.g., "French director", "American writer"), you SHOULD infer they were born in that country UNLESS supporting details explicitly contradict this (e.g., "moved from Country X" or "born in different location").
- **RATIONALE**: Nationality descriptors in biographical contexts typically indicate origin/birthplace. This is a standard, reasonable inference similar to other basic logical deductions.
- **VALIDATION CHECK**: If direct answer says "documents do not specify birthplace" when supporting details contain nationality information, this is overly conservative. Return status "rewrite" with body stating the reasonable inference: "[Person] is [nationality], indicating they were born in [country]."
- **EXCEPTION**: If supporting details explicitly mention current residence different from nationality (e.g., "Irish comedian living in London"), this does NOT contradict birthplace inference - people can move. Only contradict if documents explicitly state "born in [different location]" or similar.

EXACT FORM REQUIREMENT (CRITICAL):
- When query asks about a SPECIFIC term/word/phrase, the EXACT form must appear in supporting details. Similar or related forms are NOT sufficient.
- EXCEPTION: Only if supporting details explicitly state applicable rules that cover the queried term.

SCOPE VALIDATION:
- If the query contains EXPLICIT scope qualifiers (specific years: "in 1993", locations: "in Europe", conditions: "under normal conditions"), you MUST verify that the direct answer only uses facts with EXACTLY matching scope qualifiers.
- **CRITICAL DISTINCTION - Stable vs Time-Sensitive Attributes:**
  - **STABLE ATTRIBUTES (Geography, Ownership, Identity)**: Geographic location, headquarters, country of origin, parent company, administrative divisions - these attributes typically remain stable over time. For these, you MAY answer using historical information even if query uses present tense, as long as no evidence suggests the attribute changed.
    - Example: "Where does Company X operate?" + Document: "Company X operated in City Y (ceased 1985), City Y is in County Z" → Answer: "County Z" (with note about ceased operations) ✓
  - **TIME-SENSITIVE ATTRIBUTES (Counts, Statuses, Memberships)**: Employee counts, current leagues, active memberships, ongoing statuses - these change over time. For these, you MUST NOT mix timeframes.
    - Example: "How many employees does Company X have?" + Document: "14 employees as of 2010" → CANNOT answer for 2016 ✗
- FORBIDDEN: Using time-sensitive attribute information from different timeframes (e.g., "14 employees as of 2010" cannot answer "how many employees in 2016")
- VIOLATION PENALTY: If scope qualifiers don't match AND the attribute is time-sensitive, you MUST rewrite the answer using this EXACT format: "The documents do not specify the [attribute] [scope qualifier from query], but they do mention [available scoped information]."

SOFTWARE VS HARDWARE DISTINCTION (CRITICAL):
- When comparing entities, you MUST strictly distinguish between software and hardware, and never conflate them as sharing the same "media type" or category.
- **Software Entities**: Video games, applications, programs, digital content - these are intangible products that run on hardware platforms.
- **Hardware Entities**: Consoles, computers, devices, platforms - these are physical devices that run software.
- **FORBIDDEN CONFLATION**: A video game (software) and a gaming console (hardware) do NOT share "video games" as their common media type. The console is the platform/device, while the game is the content/software that runs on it.
- **Example Problem**: A specific video game and a gaming console are fundamentally different types of entities - one is software content, the other is hardware platform. They cannot share the same media type.
- **RULE**: If comparing software and hardware entities, explicitly state that they are fundamentally different types of entities (software vs hardware) and cannot share the same media type.
- **VALIDATION CHECK**: Always check for software/hardware conflation violations BEFORE evaluating scope alignment.

QUOTE FORMATTING REQUIREMENTS (CRITICAL):
- **FORBIDDEN PATTERNS**: Never create these quote-punctuation patterns: `","` or `"."` or `";"` - these indicate extra quotes after closing a quoted phrase. Correct format: `"text", next part` or `"text". Next sentence` (only ONE quote mark before punctuation).
- **FORBIDDEN**: Ending sentences with orphaned quotes that are not part of the sentence structure.
- **FORBIDDEN**: Adding extra quotes around supplementary information when integrating keywords into sentences.
- **REQUIRED**: All quotes must be properly paired and serve a grammatical purpose (e.g., quoting text, indicating titles).
- **VALIDATION CHECK**: After rewriting, scan for `","` or `"."` or `";"` patterns - if found, remove the extra quote mark.

INDUSTRY INFERENCE RESTRICTIONS:
- You are FORBIDDEN from inferring or stating that people work in the same industry unless the documents EXPLICITLY use that exact industry name for BOTH individuals.
- However, if documents use industry-specific terms for some individuals (e.g., "film director" clearly indicates film industry), you may acknowledge this for those individuals but cannot extend it to others without explicit documentation.
- VIOLATION EXAMPLES: "both in the film industry" = FORBIDDEN (unless both explicitly described as working in film industry). "both are filmmakers" = FORBIDDEN (unless both explicitly called filmmakers).
- If the question asks about shared industry, your response should note that industry information is not consistently specified across all individuals.
- VALIDATION CHECK: Always check for industry inference violations BEFORE evaluating scope alignment.

NO UNAUTHORIZED TERM NARROWING IN COMPARISONS (CRITICAL):
- When stating what entities share in common, you MUST NOT add limiting qualifiers that inappropriately narrow the scope of terms.
- **VIOLATION PATTERN (NARROWING)**: If Entity A description contains "[qualifier] [base term]" and Entity B description contains only "[base term]" (without the qualifier), saying "both are [qualifier] [base term]" is FORBIDDEN - this narrows Entity B's term by adding a qualifier.
- **ALLOWED PATTERN (GENERALIZATION)**: Using a broader category that encompasses different specific terms is ALLOWED (e.g., "[role A]" + "[role B]" → "[common category]").
- **VALIDATION CHECK**: For "both are [X]" statements, if [X] contains a limiting qualifier, verify that qualifier appears in BOTH entities' original descriptions. If not, this is narrowing and requires rewrite.

CONJUNCTIVE QUESTION LOGIC (A AND B queries):
- For queries asking if entities share "X AND Y", verify answer structure avoids logical contradiction:
  - If BOTH met: "Yes, both [X] and [Y]"
  - If PARTIAL: "No, while [shared X], they differ in [Y]" (use "while/although" to connect)
  - If NEITHER: "No, differ in both [X] and [Y]"
- **FORBIDDEN**: "They are both [X]. They do not share [X] and [Y]" - this is self-contradictory. Rewrite using proper connective structure.

OCCUPATION COMPARISON LOGIC:
- For single-attribute comparison (not conjunctive), prioritize shared core occupations.
- **FORBIDDEN**: Saying careers differ without acknowledging shared core occupations when they exist.

SEMANTIC INTERPRETATION IN CREATIVE CONTEXTS (CRITICAL FOR ARTISTIC QUERIES):
- When comparing creative professionals (film directors, actors, writers, producers), "titles" typically refers to WORK TITLES (film/movie names, book titles, etc.) rather than professional job titles, unless the query explicitly specifies "job titles" or "professional titles".
- **MISLEADING INTERPRETATION FORBIDDEN**: Never count professional roles as "titles" when comparing creative output. "Director", "writer", "producer" are job titles, not creative titles.
- **VALID INTERPRETATION**: For film directors, count their film/movie credits, not their job roles. For writers, count their written works, not their professional designations.
- **DOCUMENT-BASED COMPARISON**: When comparing "creative titles", use ONLY the specific works mentioned in the documents. If documents list concrete film titles for one person but not the other, that person has more creative titles.
- **SPECIFIC WORKS RECOGNITION**: Look for phrases like "directed by [person] [film title]", "[film title] directed by [person]", "remade as [film title]", etc. These indicate concrete creative works.
- **Example Problem**: Query "who has more creative titles?" about film directors → Count FILMS DIRECTED with SPECIFIC TITLES mentioned in documents, not job titles.
- **VALIDATION CHECK**: Always check for semantic misinterpretation of "titles" in creative contexts BEFORE evaluating other aspects.

CLINICAL SPECTRUM COMPLETENESS (MEDICAL QUERIES):
- For symptom/presentation queries: if supporting details contain multiple severity levels (minimal/absent, typical, rare-severe), direct answer MUST include ALL documented levels. Omitting any level = medical misinformation. Rewrite to include complete spectrum.

EXPLICIT PREMISE CORRECTION (CRITICAL):
- When query contains factual assertions that supporting details contradict or show differently, direct answer MUST explicitly state the discrepancy using clear language: "The documents do not state that [query premise]" OR "[Entity] did not [query action], but rather [actual action]". Implicit correction through careful wording alone is insufficient. If premise error exists but answer doesn't explicitly correct it, rewrite with explicit correction first.

Original question:
{self.user_input}

        {attribute_line}

Current direct answer body (without the "Direct Answer:" prefix):
{direct_body}

Supporting details (verbatim; do NOT rewrite them):
{supporting_section}

Task:
1. **DEFINITIONAL LOGIC VALIDATION FIRST (ABSOLUTE PRIORITY)**: For comparison queries, check if the supporting details describe entities with inherent functional distinctions (armed vs unarmed, designed-for vs not-designed-for, combat-role vs non-combat-role, operational vs non-operational). If they do, and the direct answer says "cannot be determined" or "insufficient information" or similar evasive language, this is a CRITICAL FAILURE. Return status "rewrite" with a corrected body that applies definitional logic and states the definitive conclusion. This check takes absolute priority over all others.
2. **NATIONALITY → BIRTHPLACE VALIDATION**: If the query asks about birthplace/birth location and supporting details contain nationality information (e.g., "Irish comedian", "French director"), check if direct answer says "documents do not specify birthplace". If so, this is overly conservative - return status "rewrite" with body: "[Person] is [nationality], indicating they were born in [country]." ONLY skip this inference if supporting details explicitly state a different birthplace.
3. **EXACT FORM VALIDATION**: If query asks about a specific term/word and direct answer makes a claim about it, verify the EXACT form appears in supporting details (not just similar/related forms). If violation found, rewrite: "The documents do not contain '[exact term]'. Related forms appear, but cannot verify this specific form."
4. **CLINICAL SPECTRUM VALIDATION**: For symptom queries, if supporting details show multiple severity levels, verify direct answer includes all. If incomplete, rewrite with complete spectrum.
5. **EXPLICIT PREMISE CORRECTION VALIDATION**: Check if query contains factual assertions (about who did what, relationships, roles) that supporting details contradict or show differently. If discrepancy exists, verify direct answer EXPLICITLY states it ("The documents do not state that...", "[Entity] did not [X], but rather [Y]"). If correction is only implicit through careful wording, return status "rewrite" with explicit correction.
6. SOFTWARE/HARDWARE VALIDATION: Check if the direct answer violates any software/hardware distinction rules above. If it does, return status "rewrite" with a corrected body that properly distinguishes between software and hardware entities.
7. QUOTE FORMATTING VALIDATION: Check if the direct answer contains malformed quote-punctuation patterns: `","` or `"."` or `";"` (double quote followed by another quote and punctuation). These indicate extra quotes after closing a phrase. Also check for unnecessary trailing quotes. If found, return status "rewrite" with corrected body that has only ONE quote mark before each punctuation.
8. TERM NARROWING VALIDATION: For comparison queries with "both are [X]", check if [X] contains a limiting qualifier. If so, verify that qualifier appears in BOTH entities' original descriptions in supporting details. VIOLATION PATTERN: If Entity A has "[qualifier] [base term]" but Entity B only has "[base term]" (without the qualifier), saying "both are [qualifier] [base term]" inappropriately narrows Entity B's scope. ALLOWED: Using a broader generalization that encompasses both (e.g., "[role A]" + "[role B]" → "[common category]"). If narrowing violation found, return status "rewrite" with corrected body using unqualified common term, valid generalization, or separate statements.
9. SEMANTIC INTERPRETATION VALIDATION: Check if the direct answer violates any semantic interpretation rules above, especially in creative contexts where "creative titles" is misinterpreted as job titles instead of work titles. Look for specific works mentioned in documents (e.g., film titles). If the answer claims no works are mentioned when they actually are in Supporting Details, return status "rewrite" with a corrected body that counts the documented specific works.
9. CONJUNCTIVE QUESTION & OCCUPATION VALIDATION: Check for logical contradictions in answer structure. For "X AND Y" queries, if answer says "they are X" then "they don't share X and Y", this is self-contradictory - rewrite using "No, while they share X, they differ in Y" structure. For single-attribute occupation comparisons, verify shared core occupations are acknowledged before differences.
10. SCOPE VALIDATION: Check if the direct answer violates any scope qualifier rules above. **CRITICAL**: Apply the Stable vs Time-Sensitive distinction. If the query asks about a STABLE attribute (geographic location, ownership, identity), allow answering with historical information even if there's a tense mismatch, as long as the attribute hasn't changed. ONLY flag violations for time-sensitive attributes (counts, current statuses, active memberships) when timeframes don't match.
11. Determine whether the direct answer actually resolves the user's request. {attribute_requirement}
12. TIMEFRAME VALIDATION: Check whether any conclusion relies on a timeframe or tense that is not supported by the supporting details. **EXCEPTION**: For STABLE attributes (geography, ownership, identity), if the Direct Answer correctly synthesizes historical information (e.g., "operated in City X" + "City X is in County Y" → "in County Y"), this is valid even if there's a tense mismatch. Focus on time-sensitive attributes where historical data cannot answer present-tense queries.
13. Detect if the user's query contains variant spellings or obvious typos for entity names that appear in the supporting details. If so, prepare a short explanatory note that maps the query term to the documented term (e.g., "Query term 'thanKing' refers to 'Hungry' as documented."). If no such normalization is needed, leave the note empty.
14. If all requirements are satisfied, reply with JSON only: {{"status":"valid","body":"<repeat the direct answer body>","reason":"<very short note>","normalization_note":"<note or empty string>"}}.
15. If any requirement fails, craft a revised direct answer body that supplies the correct attribute using only the cited facts and explains any required bridge (for example, by noting the entity that satisfies the location or role constraint). Avoid unnecessary negative statements; prefer concise affirmative linkage that states the subject, the requested attribute, and the supporting context. Return JSON only: {{"status":"rewrite","body":"<revised direct answer body>","reason":"<very short note>","normalization_note":"<note or empty string>"}}.
16. The body must not introduce new facts, must not cite new sources, and must stand alone without the "Direct Answer:" prefix.
{force_instruction}

Examples of acceptable rewrites (copy the pattern, not the text):
- Question: "When did the tour begin for Taylor Swift's album Red?"
  Correct direct answer: "Taylor Swift's Red Tour began on March 13, 2013, in Omaha, Nebraska."
- Question: "What do Victor Salva and Emilio Fernández have in common?"
  Correct direct answer: "Victor Salva and Emilio Fernández are both film directors and screenwriters."
- Question: "How many disciplines are combined in Hector Janse van Rensburg's degree?"
  Correct direct answer: "Hector Janse van Rensburg's Philosophy, Politics and Economics degree combines three disciplines."
- Question: "When was the colony William Bradford was governor of founded?"
  Correct direct answer: "The colony governed by William Bradford, Plymouth Colony, was founded in 1620."
- Question: "What type of media do Game A and Console B have in common?"
  Correct direct answer: "Game A is software (a video game), while Console B is hardware (a gaming console). They are fundamentally different types of entities and cannot share the same media type."
"""
        return prompt

    def _strip_direct_answer(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        if lowered.startswith("direct answer:"):
            cleaned = cleaned.split(":", 1)[-1].strip()

        if "?" in cleaned:
            parts = cleaned.split("?")
            tail = parts[-1].strip()
            cleaned = tail if tail else cleaned.replace("?", "").strip()

        patterns = [
            r"^(which|what|when|where|why|how|who)\b.*?:\s*(.+)$",
            r"^(which|what|when|where|why|how|who)\b[^—–\-]*[—–\-]\s*(.+)$",
            r"^(which|what|when|where|why|how|who)\b.*?(?:is|are)\s+answered\s+by\s+(.+)$",
            r"^(?:regarding|about|concerning)\b[^,]*,\s*(.+)$",
            r"^what\b[^,.]*?\bis that\s+(.+)$",
            r"^(which|what|when|where|why|how|who)\b[^,.!?]*?\b(?:is|are)\s+(.+)$",
            r"^(when|where|why|how)[^:]*:\s*(.+)$",
            r"^(?:noting|stating|observing|mentioning|clarifying|explaining|adding|remarking|indicating)\s+that[,:\-]?\s*(.+)$",
        ]

        while True:
            updated = False
            for pattern in patterns:
                match = re.match(pattern, cleaned, re.IGNORECASE | re.DOTALL)
                if match:
                    remainder = match.group(match.lastindex or 0).strip()
                    if remainder and remainder != cleaned:
                        cleaned = remainder
                        updated = True
                        break
            if not updated:
                break

        cleaned = re.sub(
            r"([,;]\s*)(which|what|when|where|why|how|who)\b[^:]*:\s*",
            ". ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.lstrip(",;:—–- ")

        return cleaned.strip()

    def _infer_attribute_keyword(self) -> Optional[str]:
        question = self.user_input
        if not question:
            return None

        llm_attribute = self._llm_extract_attribute(question)
        if llm_attribute:
            return llm_attribute

        # Simplified fallback regex - only used if LLM fails
        lowered = question.lower()
        # Look for any noun phrase after question words
        match = re.search(r"\b(?:what|which|who|when|where|why|how)\b[^?]*?\b([a-z0-9'\-]+(?:\s+[a-z0-9'\-]+)*)", lowered, re.IGNORECASE)
        if not match:
            return None

        phrase = match.group(1)
        phrase = re.split(
            r"\b(?:is|are|was|were|does|do|did|has|have|had|located|in|at|of|for|to|with|,|\?|who|that|when|where|why|how|which|what)\b",
            phrase,
            maxsplit=1,
        )[0]

        tokens = re.findall(r"[a-z0-9'-]+", phrase)
        if not tokens:
            return None

        stop_words = {
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "any",
            "some",
            "many",
            "other",
            "others",
            "each",
            "every",
            "all",
            "which",
            "what",
            "whose",
            "whom",
            "who",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "from",
            "with",
            "about",
            "into",
            "over",
            "after",
            "before",
            "during",
            "between",
            "through",
            "up",
            "down",
            "out",
            "off",
            "above",
            "below",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        attribute_tokens: list[str] = []
        for token in tokens:
            if token in stop_words:
                if attribute_tokens:
                    break
                continue
            attribute_tokens.append(token)
            if len(attribute_tokens) >= 2:
                break

        if not attribute_tokens:
            return None

        return " ".join(attribute_tokens)

    def _llm_extract_attribute(self, question: str) -> Optional[str]:
        prompt = f"""
        You are given a user question. Identify the key attribute, property, or concept that the question is asking about.

        Return JSON only in the format: {{"attribute": "<attribute phrase>"}} with no extra text.

        CRITICAL RULES:
        - ONLY extract words and phrases that appear DIRECTLY in the question text itself.
        - DO NOT use external knowledge, inference, or assumptions about what the question might be asking.
        - DO NOT add words that are not present in the question.
        - DO NOT infer specific types or categories unless they are explicitly named in the question.
        - For questions like "what other type of X", extract "type of X" or similar phrase from the question text.

        Rules:
        - The attribute phrase should be as concise as possible while preserving the key noun(s) or concept.
        - Look for what the question is fundamentally asking: the main thing, property, concept, or relationship being inquired about.
        - ONLY use exact wording from the question - no synonyms, no rephrasing unless absolutely necessary for clarity.
        - If the question asks "what other type of [something]", the attribute is "type of [something]".
        - If no specific attribute/concept can be extracted using ONLY words from the question, return {{"attribute": null}}.
        - Do not include any explanations or additional text.

        Examples:
        - Question: "What color is the car?" → {{"attribute": "color"}}
        - Question: "Which university did she attend?" → {{"attribute": "university"}}
        - Question: "What was needed to build the house?" → {{"attribute": "requirement"}}
        - Question: "How does photosynthesis work?" → {{"attribute": "process"}}
        - Question: "Why did the event happen?" → {{"attribute": "cause"}}
        - Question: "What other type of therapy?" → {{"attribute": "type of therapy"}}

        Question: {question}
        """
        try:
            response = self.validator_extractor_llm.complete(prompt=prompt)
        except Exception as exc:
            print(f"Attribute extractor call failed: {exc}")
            return None

        payload = self._extract_validator_json(response.text)
        if not payload:
            return None

        attribute = payload.get("attribute")
        if isinstance(attribute, str):
            cleaned = attribute.strip()
            if cleaned and cleaned.lower() != "null":
                return cleaned
        return None

    def _clean_fact_for_direct_answer(self, fact_line: str) -> str:
        if not fact_line:
            return ""
        text = fact_line.strip()
        if text.startswith("-"):
            text = text.lstrip("-").strip()

        # Remove document citations like (Document abc123_1)
        text = re.sub(r"\s*\((?:Document\s+)?[0-9a-fA-F]{4,}_[0-9]+\)\s*", "", text)

        # Smart quote handling: only remove truly orphaned quotes at start/end
        # Don't blindly strip all quotes - preserve properly paired ones
        if text:
            start_char = text[0]
            end_char = text[-1]

            # Check for matching quote pairs
            quote_pairs = {
                '"': '"',  # ASCII double quotes
                '"': '"',  # Smart double quotes
                "'": "'",  # ASCII single quotes
                "'": "'",  # Smart single quotes
            }

            if start_char in quote_pairs and end_char == quote_pairs.get(start_char):
                # Text starts and ends with matching quotes - keep them
                pass
            elif start_char in ['"', "'", '"', "'"]:
                # Starts with a quote but doesn't end with matching one - remove the opening quote
                text = text[1:].lstrip()
            elif end_char in ['"', "'", '"', "'"]:
                # Ends with a quote but doesn't start with matching one - remove the trailing quote
                text = text[:-1].rstrip()

        text = re.sub(r"\s+", " ", text).strip()
        if text and not text.endswith("."):
            text += "."
        return text

    def _collect_supporting_lines(self, lines: list[str]) -> list[str]:
        collected: list[str] = []
        capture = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("supporting details"):
                capture = True
                continue
            if lowered.startswith("facts (auto-complete"):
                break
            if capture and stripped.startswith("-"):
                collected.append(stripped)
        return collected

    def _extract_location_phrase(self) -> Optional[str]:
        if not self.user_input:
            return None
        match = re.search(r"located in ([^?]+)", self.user_input, flags=re.IGNORECASE)
        if not match:
            return None
        phrase = match.group(1).strip()
        return phrase.rstrip(".?")

class ProcessFinalResponseLLM:
    def __init__(self, CoA_messages, user_input):
        self.llm = OpenAI(model="o4-mini")
        # self.react_agent = OpenAIAgent.from_tools(
        #     [FunctionTool.from_defaults(fn=self.process_raw_CoA)],
        #     llm=self.llm,
        #     verbose=True)
        self.CoA_messages = CoA_messages
        self.user_input = user_input

    # def do_a_combination_search(self, query: str) -> str:
    #     """YOU DO HAVE THE ABILITY TO DO WEB SEARCHING USING Google. Use this tool to perform a Google search and return the content of the top 5 search link.
    #     MUST state that the information is retrieved using Google search and not from provided documents."""
    #
    #     max_retries = 3
    #     attempt = 0
    #     search_results = []
    #
    #     while attempt < max_retries:
    #         try:
    #             search_results = list(search(query, num_results=1))
    #             if search_results:  # Check if search_results is not empty
    #                 break
    #         except Exception as e:
    #             print(f"Attempt {attempt + 1} failed ☹️: {e}")
    #
    #         attempt += 1
    #         time.sleep(1)  # Wait for 1 second before retrying
    #
    #     if not search_results:
    #         return "Failed to retrieve search results after 3 attempts ☹️."
    #
    #     # Initialize an empty string to hold the concatenated results
    #     search_results_str = ""
    #
    #     # Iterate through the search results and append each result to the string
    #     for result in search_results[:1]:
    #         if not result.startswith("https://"):
    #             result = "https://" + result
    #         content = crawler.run(url=result)
    #         search_results_str += result + ":\n" + str(content.markdown) + "\n"
    #     print("search_results_str:" + search_results_str)
    #     return search_results_str

    def process_raw_CoA(self, current_mode_content):
        # self.do_a_combination_search(self.user_input)
        # prompt = f"""
        #             Your task is to help answer a complex question based on information retrieved by an AI assistant which had access to multiple documents.
        #             If the Original Query is recalling the content (or part content) of the previous questions or questions and their responses, provide the response using the previous question and responses directly. DO NOT need to go through the following process for a detailed answer.
        #             The agent uses a Chain of Abstraction approach to process the query, then calls on multiple other AI agents who have document reading tools, some of whom also use Chain of Abstraction for their reasoning.
        #             The result is along TRACE TEXT of the tool use and conversations/requests of all these agents as they find information to answer the question.
        #             You will notice that the agents return the sources of the information they find. It is ESSENTIAL that these sources (document_tools with their tool numbers) are retained and clearly stated so they can be used as citations in the final response.
        #             **DO NOT CITE ANY FOLDERS AS SOURCES**, if there are folder sources, you should recognize that are external sources such as base knowledge.
        #
        #             *Your TASK*
        #             Produce a long detailed answer to the original question using the information you find in the TRACE TEXT, using your best judgement on the relevance of the information.
        #             If you found there is no document tool providing useful information in TRACE TEXT, then you MUST retrieve useful information from web sources by calling web searching function {self.do_a_combination_search(self.user_input)} with the user query from the searched web sources to supplement the final answer and cite the web sources.
        #             **All web source citations MUST be retrieved by the web searching function, accessible ,accurate and real, do NOT make up link citations. If any link url are not started with `https://`, then add this into the url make it accessible to users.**
        #
        #             *Your OPERATION MODE*
        #             {current_mode_content}
        #
        #             *IMPORTANT: only use the information from the TRACE TEXT, DO NOT use the web searching function unless there is no document tool providing useful information in TRACE TEXT. But make it very clear with citations/clarification for users if any base knowledge or web sources used in the final response.
        #             IF YOU ONLY USE BASE KNOWLEDGE AND/OR WEB SOURCES, DO NOT MENTION ANY EXPERT TOOLS OR FOLDERS IN THE RESPONSE TO CONFUSE USERS. OTHERWISE, CITE APPROPRIATE DOCUMENT TOOLS' REFERENCE NUMBERS (DO NOT CITE ANY OTHER EXPERT TOOLS OR FOLDERS).
        #             **If you provide web links in citations, MUST make sure they are accessible and valid, DO NOT MAKE UP LINKS.**
        #
        #             *Original QUERY*
        #             {self.user_input}
        #
        #             *TRACE TEXT*
        #             {self.CoA_messages}
        #             """
        prompt = f"""
                            Your task is to help answer a complex question based on information retrieved by an AI assistant which had access to multiple documents. 
                            If the Original Query is recalling the content (or part content) of the previous questions or questions and their responses, provide the response using the previous question and responses directly. DO NOT need to go through the following process for a detailed answer.
                            The agent uses a Chain of Abstraction approach to process the query, then calls on multiple other AI agents who have document reading tools, some of whom also use Chain of Abstraction for their reasoning. 
                            The result is along TRACE TEXT of the tool use and conversations/requests of all these agents as they find information to answer the question. **YOU MUST USE ALL USEFUL INFORMATION PROVIDED BY TOOLS AND CITE THESE TOOLS IN THE RESPONSE.**
                            You will notice that the agents (document tools) output the information they find from documents. **It is ESSENTIAL that all of these sources (document_tools with their TOOL NUMBERS) are retained and clearly stated so they can be used as citations (ALWAYS cited with their TOOL NUMBERS) in the final response**.
                            **DO NOT CITE ANY FOLDERS AS SOURCES**, if there are folder sources, you should recognize that are external sources such as base knowledge.

                            *Your TASK* 
                            If you can find useful information in the TRACE TEXT, you should produce a comprehensive and detailed answer that:
                                1. Provides extensive background and context for each point
                                2. Includes specific examples, data points, and quantitative evidence where available
                                3. Explains relationships and interconnections between concepts
                                4. Discusses implications and practical applications
                                5. Uses clear section headers to organize information
                                6. Ensures each major point is supported by multiple sources where possible
                                7. Aims for exhaustive coverage rather than brevity
                                8. Anticipates and addresses potential questions
                                9. Synthesizes information from multiple sources to form cohesive arguments
                                10. Provides both theoretical understanding and practical insights

                            However, if you found there is no document tool providing useful information in TRACE TEXT, then you MUST STOP further processing ignore the operation mode, just return `WEB SEARCH REQUIRED`.
                            *Your OPERATION MODE*
                            {current_mode_content}

                            *IMPORTANT: only use the information from the TRACE TEXT, **DO NOT use your base knowledge**.
                            IF you found there is no document tool providing useful information in TRACE TEXT, then DO NOT MENTION ANY EXPERT TOOLS OR FOLDERS IN THE RESPONSE TO CONFUSE USERS. OTHERWISE, CITE APPROPRIATE DOCUMENT TOOLS' REFERENCE NUMBERS (DO NOT CITE ANY OTHER EXPERT TOOLS OR FOLDERS).

                            *Original QUERY* 
                            {self.user_input}

                            *TRACE TEXT* 
                            {self.CoA_messages}
                            """
        response = self.llm.complete(prompt=prompt)
        return response.text

    def process_complex_web(self, current_mode_content, processed_response):
        prompt = f"""
                   Your task is to help give a detailed answer following the output format of YOUR OPERATION MODE to an Original QUERY based on information in Original Answer containing multiple web sources.
                   It is ESSENTIAL that these web sources in the Original Answer are retained and clearly stated so they can be used as citations in the final response.

                   *Your TASK* 
                   If you can find useful information you in the Original Answer, you should produce a long detailed answer to the original question, using your best judgement on the relevance of the information, that:
                       1. Synthesizes information from all available web sources
                       2. Provides detailed examples and real-world case studies
                       3. Includes relevant statistics, data points, and metrics
                       4. Explains underlying concepts thoroughly with clear definitions
                       5. Discusses practical applications and real-world implications
                       6. Uses clear section headers for logical organization
                       7. Ensures each point is supported by multiple sources where possible
                       8. Provides both historical context and future implications
                       9. Addresses potential counterarguments or limitations
                       10. Connects ideas across different sources to form cohesive arguments 
                   *Your OPERATION MODE*
                   {current_mode_content}

                   *IMPORTANT: only use the information from the Original Answer, DO NOT use your base knowledge.

                   *Original QUERY* 
                   {self.user_input}

                   *Original Answer* 
                   {processed_response}
                                               """
        response = self.llm.complete(prompt=prompt)
        return response.text

    def process_mix_query(self, current_mode_content, processed_response_CoA, processed_response_web):
        prompt = f"""
                   Your task is to help give a detailed answer following the output format of YOUR OPERATION MODE to an Original QUERY by combining the information from Original CoA Answer containing multiple document sources and Original Web Answer containing multiple web sources.
                   It is ESSENTIAL that these document and web sources in the original answers are retained and clearly stated so they can be used as citations in the final response.

                   *Your TASK* 
                   If you can find useful information you in the Original CoA Answer and Original Web Answer, you should produce a long detailed answer to the original question, using your best judgement on the relevance of the information, that:
                       1. Integrates and cross-references information from both document and web sources
                       2. Provides extensive examples from both source types
                       3. Includes detailed data points and quantitative evidence
                       4. Explains concepts thoroughly with clear relationships
                       5. Discusses real-world applications and implications
                       6. Uses clear section headers for logical organization
                       7. Ensures comprehensive coverage of all relevant aspects
                       8. Highlights where sources complement or contrast each other
                       9. Provides both theoretical understanding and practical insights
                       10. Synthesizes information to form well-supported conclusions

                   *Your OPERATION MODE*
                   {current_mode_content}

                   *IMPORTANT: only use the information from the original answers, DO NOT use your base knowledge.

                   *Original QUERY* 
                   {self.user_input}

                   *Original CoA Answer* 
                   {processed_response_CoA}

                    *Original Web Answer* 
                   {processed_response_web}
                                               """
        response = self.llm.complete(prompt=prompt)
        return response.text


class RelatedQueryReActAgent:
    def __init__(self):
        self.llm = OpenAI(model="gpt-4o-mini")
        # self.memory_file_name = "memory_store.json"
        # Initialize the ReActAgent (secondary agent)

        self.react_agent = OpenAIAgent.from_tools(
            [
             FunctionTool.from_defaults(fn=helper.read_tool_access_info)],
            # memory=self.memory,
            llm=self.llm,
            verbose=True)

    # def read_pre_five_messages(self) -> str:
    #     if os.path.exists(self.memory_file_name):
    #         memory_dict = check_memory(self.memory_file_name)
    #     else:
    #         memory_dict = LimitedDict()
    #     return memory_dict

    def check_related_query(self, user_input: str) -> str:
        # Create a prompt that asks the ReActAgent to find the most related query
        # prompt = f"""
        #         The user query is: {user_input}.
        #         You can use the tool `read_pre_five_messages` to retrieve a dictionary of recent queries (in keys) and responses (in values).
        #         If the user query is related to previous user-shared information:
        #             Your task is to return two values of True or False:
        #             - Use `read_pre_five_messages` to analyze recent messages and detect any significant user-shared information, such as names, preferences, facts, or personal statements.
        #             - If the current user query appears to be asking for something previously shared (e.g., "What did I say about my...?" or "Do you remember..."), search the recent messages for relevant details.
        #             - If relevant information is found, respond with first value:"Related query found: (state the related query), and the updated query is: (state the related query)", the second value is True.
        #             - If no relevant information is found in previous messages, respond with the first value: "No related query found.", the second value is True.
        #         else:
        #            Your task is to return two values of True or False:
        #             - Use `read_pre_five_messages` to detect any related information to the user query.
        #             - If relevant information is found, update the user query based on the relevant information, and respond with the first value:"Related query found: (state the related query), and the updated query is: (state the updated query)", the second value is False.
        #             - If no relevant information is found in recent messages, respond with the first value: "No related query found.", the second value is False.
        #      """
        #

        # prompt = f"""
        #                 The user query is: {user_input}.
        #                 You can use the tool `read_pre_five_messages` to retrieve a dictionary of recent queries (in keys) and responses (in values).
        #
        #                 Ignore any processing instructions such as formatting, summarizing, or other transformations. Follow only these rules:
        #
        #                 1. **First Value Logic: Check if the query relates to previously shared information**
        #                    - Use `read_pre_five_messages` to analyze recent messages for any significant information, such as names, preferences, facts, or statements.
        #                    - If the current user query directly references previously shared information (e.g., "What did I say about my...?" or "Do you remember...?"):
        #                      - If relevant information is found, output:
        #                        - **First value**: `"Related query found: (state the related query), and the updated query is: (state the related query)"`
        #                      - If no relevant information is found, output:
        #                        - **First value**: `"No related query found."`
        #                    - If the query is not explicitly referencing previously shared information, look for any related information that may assist in updating the user query:
        #                      - If relevant information is found, output:
        #                        - **First value**: `"Related query found: (state the related query), and the updated query is: (state the updated query)"`
        #                      - If no relevant information is found, output:
        #                        - **First value**: `"No related query found."`
        #
        #                 2. **Second Value Logic: Check if the user query is sharing new information**
        #                    - Check if the current query shares new information about the user (e.g., statements like "My name is ...?" or "I live in ...?").
        #                    - If the query contains new information, output:
        #                      - **Second value**: `"True"`
        #                    - If the query is not an information-sharing statement (e.g., it's a question or request for information), output:
        #                      - **Second value**: `"False"`
        #
        #                 **Note**: Always respond with both values in this format: **`first value $$ second value`**, even the user input requires other format.
        #              """

        prompt = f"""
                                The user query is: {user_input}.
                                Ignore any processing instructions such as formatting, summarizing, or other transformations. Follow only these rules:
                                Check if the user query is sharing new information**
                                   - Check if the current query shares new information about the user or ask some information shared by the user (Examples include statements like "My name is ...", "I live in ...", "Who am I ...", "I like ...", etc.).
                                   - If the query contains new information, return True:
                                   - If the query is not an information-sharing statement (e.g., it's a question or request for information not related to the user), return False:

                                **Note**: Always respond with True or False
                             """

        # def check_related_query(self, user_input: str) -> str:
        #     # Create a prompt that asks the ReActAgent to find the most related query
        #     prompt = f"""
        #         The user query is: {user_input}.
        #         You can use the tool `read_pre_five_messages` to get the dictionary of previous queries and corresponding responses.
        #
        #         Your task is to:
        #         - Track and retain any user-shared information and meaningful questions within the last five interactions, such as names, preferences, locations, or other significant details. Use this information intelligently if the user asks about it again.
        #
        #         - When the current user query is a meta-query asking for a specific recent question by its position among the last few questions (e.g., "What was my last question?", "What was my second-to-last question?", "What did I ask three questions ago?"):
        #             - Look through the last five interactions to identify recent meaningful questions.
        #             - **Ignore any previous meta-queries** within these last five interactions (e.g., any prior instances of "What was my last question?") to focus only on substantive questions.
        #             - Identify the specific question based on its position among the filtered meaningful questions:
        #                 - If the user asks for the "last question," provide the most recent meaningful question.
        #                 - If the user asks for the "second-to-last question," provide the second most recent meaningful question, and so on.
        #             - Respond in the following format:
        #               "Related query found: '(state the specific question based on position),' and the updated query is: '(restate the content of that question as the answer).'"
        #
        #               - Example: If the last five meaningful questions are "How can I reduce emissions?" followed by "What are the main causes of climate change?" (after ignoring previous meta-queries), and the user asks, "What was my second-to-last question?", respond with:
        #                 "Related query found: 'How can I reduce emissions?' and the updated query is: 'How can I reduce emissions?'"
        #
        #         - For other types of related queries, such as specific information recall (e.g., "What did I say about my favorite color?"), respond as follows:
        #             - If relevant information is found, respond with:
        #               "Related query found: '(state the related information),' and the updated query is: '(provide a natural response based on the original information in context)'."
        #
        #         - If no related query or information is found within the last five messages, respond with:
        #           "No related query found."
        #
        #             """
        #     # Use the ReActAgent to check for the most related query
        response = self.react_agent.chat(message=prompt)
        return response.response


"""Workflow trail"""
# Define the workflow for modifying the query
# class ModifyQueryWorkflow(Workflow):
#     @step(pass_context=True)
#     async def check_related_query(self, ctx:Context, ev: StartEvent) -> InputRequiredEvent:
#         related_query_agent = RelatedQueryReActAgent()
#         # Use the secondary agent to check for related queries
#         related_query_response = await related_query_agent.check_related_query(ev.input)
#         await ctx.set("original_query", ev.input)
#         # Check if a related query was found
#         if "Related query found" in related_query_response[0]:
#             related_query = related_query_response[0].replace("Related query found: ", "")
#             await ctx.set("related_query", related_query)
#             return InputRequiredEvent(prefix = f"""A related query was found: '{related_query}'. Do you want to update your query? (yes/no)""")
#         else:
#             await ctx.set("related_query", "")
#             return InputRequiredEvent(
#                 prefix=f"no")
#
#
#     @step(pass_context=True)
#     async def return_final_query(self, ev: HumanResponseEvent) -> StopEvent:
#         response = ev.response
#         return StopEvent(result=f"{response}")
"""Workflow ending"""

from helper import IDInjectingWrapperTool, id_stamped_print
from llama_index.agent.coa import CoAAgentWorker
from safe_coa_parser import SafeChainOfAbstractionParser
# from llama_index.core.agent import ChainOfAbstractionAgent

class GradioInterface:
    def __init__(self, agent, modes, index_map):
        helper.tool_list = []
        self.llm = OpenAI(model="gpt-5-mini")
        self.user_send = False
        self.index_map = {value: os.path.split(key)[-1] for key, value in index_map.items()}
        # self.fileNameMappingAgent = RelatedQueryReActAgent()
        # self.memory_file_name = "memory_store.json"
        # self.regenerate_initialised = False
        # self.monitor_thread = threading.Thread(target=self.monitor_stop, daemon=True)
        # self.monitor_thread.start()
        self.pre_user_input = ""
        self.pre_mode = "Chat"
        self.lock = threading.Lock()  # Lock for thread safety
        # self.async_lock = asyncio.Lock()
        self.agent = agent
        self.tools = agent.tools
        self.modes = modes
        self.current_mode_content = self.modes.get(self.pre_mode, "")
        self.log_enabled = False
        self.db_memory_enabled = True
        self.log_output = """\n
        Click [Send] or type [Shift + Enter] to send your query.\n
        Click [Reset] to clear the conversation log and reset the memory.\n
        Enable [Observe] to view internal process logs.\n
        Mode selection controls the style of the output."""
        self.log_window_update_interval = 1
        self.cleared = False
        self.related_query = None
        self.updated_query = None
        self.confirm = None
        self.confirm_future = None
        self.stop = False
        self.file = None
        self.trace_file = None
        self.user_shared = False
        self.CoA_raw_output = ""
        self.truncate_phase2_subcoa = True
        # if self.log_enabled:
        #     file.write(self.log_output)
        # else:
        self.display_writer = helper.DisplayNWrite(original_stdout, output_logger)
        sys.stdout = self.display_writer
        sys.stderr = self.display_writer

        def custom_print(*args, **kwargs):
            message = " ".join(map(str, args)) + "\n"

            sys.stdout.add_to_print_buffer(message)

            global_original_print(*args, **kwargs)

        global print
        print = custom_print

        self.res_coa_dict = dict()
        if os.path.exists(persist_directory_summary) and os.listdir(persist_directory_summary):
            self.vector_store_summaries = self.load_summary_db()
        self.recall_check_response = "False"
        self.check_if_web_searched_text = 'false'
        self.tool_retriever = None
        # self.useful_tools = []
        # self.workflow = ModifyQueryWorkflow()
        # self.current_context = None
        # self.tool_access_info = tool_access_info

        # print("\nPersisting chat store in chat_store.json\n")

    # def check_complexity(self, user_input):
    #     is_complex = self.agent.react_agent.chat(
    #         f"""
    #         Your task is to determine the COMPLEXITY of the user query below.
    #         SIMPLE: The query is straightforward and can be answered directly. For example, recalling past conversations and questions, formatting, or providing simple information.
    #         COMPLEX: The query requires deeper reasoning, multiple steps, or complex analysis. For example, recalling from a document, summarizing a document, planning a task, or providing detailed explanations.
    #
    #         Follow this logic:
    #         if (query == "COMPLEX"):
    #             return "TRUE"
    #         else:
    #             return "FALSE"
    #         Do NOT include the quotes in your response, or provide any context. Simply state TRUE or FALSE.
    #
    #         This is the user query: {user_input}
    #         """
    #     )
    #     return is_complex

    def load_summary_db(self, db="chroma_db_summary"):
        persist_directory = db
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection_summaries = chroma_client.get_or_create_collection(name="query_doc_summaries", metadata={
            'hnsw:space': 'cosine'})  # for cosine distance, L2 distance by default in chroma
        vector_store_summaries = ChromaVectorStore(chroma_collection=collection_summaries)
        return vector_store_summaries

    def reset_agent_state(self):
        self.related_query = ""
        self.updated_query = ""
        self.CoA_raw_output = ""
        self.user_shared = False
        self.recall_check_response = "False"
        self.check_if_web_searched_text = "false"

    def _extract_verified_facts_from_prompt(self, prompt_text: str) -> list[tuple[str, str]]:
        if not prompt_text:
            return []

        pattern = re.compile(
            r"##\s*Verified Facts List(?P<section>.*?)(?:\n##\s|\n###\s|$)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        facts: list[tuple[str, str]] = []
        for match in pattern.finditer(prompt_text):
            section = match.group("section")
            for line in section.splitlines():
                stripped = line.strip()
                if not stripped or not stripped.startswith("-"):
                    continue
                raw = stripped[1:].strip()
                canonical = raw
                if "(Evidence:" in canonical:
                    canonical = canonical.split("(Evidence:", 1)[0].strip()
                facts.append((canonical.lower(), raw))
        return facts

    def _reinforce_subcoa_response(self, prompt_text: str, response):
        facts = self._extract_verified_facts_from_prompt(prompt_text)
        if not facts:
            return response

        if hasattr(response, "response"):
            response_text = response.response or ""
        else:
            response_text = str(response or "")

        response_lower = response_text.lower()
        missing: list[str] = []
        for canonical, original in facts:
            if canonical and canonical not in response_lower:
                missing.append(original)

        if not missing:
            return response

        if "Facts (auto-restored):" not in response_text:
            addition = "\n\nFacts (auto-restored):\n" + "\n".join(f"- {fact}" for fact in missing)
        else:
            addition = "\n".join(f"- {fact}" for fact in missing)
            addition = "\n" + addition

        updated_text = response_text.rstrip() + addition + "\n"

        if hasattr(response, "response"):
            response.response = updated_text
            return response
        return updated_text

    def chat_repl_all_coa(self, user_input, test=True):
        """
        Run self.chat_repl(...) on each sub-agent in self.coa_agents concurrently.
        If self.stop is set to True at any point,
        all sub-agents will stop the next time they check the while loop condition.
        """

        if "(" in user_input or ")" in user_input:
            user_input = user_input.replace("(", "").replace(")", "")
        
        if self.stop:
            print("⚠️ Already in STOP mode. Skipping parallel sub-agents.")
            return "Agents were stopped."
        
        # time.sleep(21)
        for i, sub_agent in enumerate(self.agent.coa_agents):
            if hasattr(sub_agent, "embed_model"):
                sub_agent.embed_model._reset_http_client()
                print(f"reset embed_model for subCoA agent {i}")

        # we create a thread for each sub-agent
        # with ThreadPoolExecutor(max_workers=len(self.agent.coa_agents)) as executor:



        # with ThreadPoolExecutor(max_workers=len(self.agent.coa_agents)) as executor:
        #     # submit each agent to run chat_repl
        #     future_to_agent = {}
        #     for sub_agent in self.agent.coa_agents:
        #         original_user_input = copy.deepcopy(user_input)
        #         self.reset_agent_state()
        #         # self.related_query = ""
        #         # self.updated_query = ""
        #         # self.CoA_raw_output = ""
        #         # self.user_shared = False
        #         # self.recall_check_response = "False"
        #         # self.check_if_web_searched_text = "false"
        #         future = executor.submit(self.chat_repl, sub_agent, original_user_input)
        #         future_to_agent[future] = sub_agent

        #     # gather results as they complete
        #     for future in as_completed(future_to_agent):
        #         agent = future_to_agent[future]
        #         try:
        #             result = future.result()  # This calls chat_repl(...)
        #             if not result:
        #                 print(f"[{agent.name}] returned an empty response.")
        #             else:
        #                 print(f"[{agent.name}] response: {result}")
        #             partial_responses.append(f"[{agent.name}] => {result}")
        #         except Exception as e:
        #             # if there's an exception, we can set self.stop = True
        #             with self.lock:
        #                 self.stop = True
        #             partial_responses.append(f"[{agent.name}] error => {e}")
        partial_responses = []
        sub_coa_agent_filtered_list = ["tool_"+str(tool.split("_")[0]) for tool in helper.tool_list]
        LOG_FILE_PATH = "evaluation_traces.jsonl"
        # LOG_LOCK = threading.Lock()
      

        with self.lock:
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write('\n')
        main_trace_id = str(uuid.uuid4())
        print(f"--- Starting new query with Trace ID: {main_trace_id} ---")
        
        # print(f"--- Log separator written for new query: '{short_user_input}' ---")
        
        if test:
            print("===== RUNNING IN SEQUENTIAL TEST MODE =====")
            for i, sub_agent in enumerate(self.agent.coa_agents):
                subcoa_task_id = f"{sub_agent.name}_{uuid.uuid4().hex[:8]}"
                try:
                    print(f"===== EXECUTING SubCoA_{i} =====")
                    original_user_input = copy.deepcopy(user_input)
                    original_user_input = self._filter_subcoa_input(original_user_input)
                    sub_agent_specific_tools = sub_agent.agent_worker.tools
                    self.reset_agent_state()
                    print(f"helper.tool_list: {helper.tool_list}")
                    result = None
                    if len(helper.tool_list) != 0:
                        current_sub_coa_agent_filtered_list = set([t.metadata.name for t in sub_agent_specific_tools]).intersection(set(sub_coa_agent_filtered_list))
                        if current_sub_coa_agent_filtered_list and i == 0:
                            print(f"current_sub_coa_agent_filtered_list: {current_sub_coa_agent_filtered_list}")
                            original_user_input = original_user_input.replace("FILTER SET", f"FILTER SET: {current_sub_coa_agent_filtered_list})")
                            result = self.chat_repl(sub_agent, original_user_input)
                    if result:
                        result = self._reinforce_subcoa_response(original_user_input, result)
                        display_text = result.response if hasattr(result, "response") else result
                        print(f"[SubCoA_{i}] response: {display_text}")
                        partial_responses.append(f"[SubCoA_{i}] => {display_text}")
                    else:
                        print(f"[SubCoA_{i}] returned an empty response.")
                    print(f"===== COMPLETED SubCoA_{i} =====")
                except Exception as e:
                    with self.lock:
                        self.stop = True
                    partial_responses.append(f"[SubCoA_{i}] error => {e}")
                    print(f"===== ERROR IN SubCoA_{i}: {e} =====")
        else:
            with ThreadPoolExecutor(max_workers=len(self.agent.coa_agents)) as executor:
                print(f"helper.tool_list: {helper.tool_list}")
                futures = []
                self.reset_agent_state()
                def _apply_tool_filter(agent_runner, allowed_tools):
                    """Limit an agent's accessible tools to the allowed list and return state for restoration."""
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

                for sub_agent in self.agent.coa_agents:
                    subcoa_task_id = f"{sub_agent.name}_{uuid.uuid4().hex[:8]}"
                    original_user_input = copy.deepcopy(user_input)
                    original_user_input = self._filter_subcoa_input(original_user_input)
                    sub_agent_specific_tools = sub_agent.agent_worker.tools


                    
                    # wrapped_tools = [
                    #     IDInjectingWrapperTool(
                    #         original_tool=tool,
                    #         trace_id=main_trace_id,
                    #         parent_task_id=subcoa_task_id 
                    #     ) for tool in sub_agent_specific_tools
                    # ]
                    
                
                    # temp_worker = CoAAgentWorker.from_tools(
                    #     tools=wrapped_tools,
                    #     llm=sub_agent.agent_worker.llm,     
                    #     verbose=True,
                    #     # system_prompt=sub_agent,
                    # )

                    # temp_agent_for_this_thread = temp_worker.as_agent(memory=None)
                    # temp_agent_for_this_thread.name = sub_agent.name



                    if len(helper.tool_list) != 0:
                        current_sub_coa_agent_filtered_list = set([t.metadata.name for t in sub_agent_specific_tools]).intersection(set(sub_coa_agent_filtered_list))
                        if current_sub_coa_agent_filtered_list:
                            print(f"current_sub_coa_agent_filtered_list: {current_sub_coa_agent_filtered_list}")
                            # original_user_input = original_user_input.replace("your EXPERT COLLEAGUE TOOLS (Folder Agents) FILTER SET", f"your EXPERT COLLEAGUE TOOLS (Folder Agents) FILTER SET: {current_sub_coa_agent_filtered_list})")
                            original_user_input = original_user_input.replace("FILTER SET", f"FILTER SET: {current_sub_coa_agent_filtered_list}")
                            # pdb.set_trace()
                            allowed_tools = [tool for tool in sub_agent_specific_tools if tool.metadata.name in current_sub_coa_agent_filtered_list]
                            if not allowed_tools:
                                print(f"[{sub_agent.name}] No tools remain after applying filter; skipping execution.")
                                continue
                            tool_state = _apply_tool_filter(sub_agent, allowed_tools)
                            def _run_with_restoration(agent_ref, prompt, state_snapshot):
                                try:
                                    return self.chat_repl(agent_ref, prompt)
                                finally:
                                    _restore_tool_filter(state_snapshot)

                            future = executor.submit(_run_with_restoration, sub_agent, original_user_input, tool_state)
                            futures.append((sub_agent, future))

                for sub_agent, future in futures:
                    try:
                        result = future.result()
                        if not result:
                            print(f"[{sub_agent.name}] returned an empty response.")
                        else:
                            print(f"[{sub_agent.name}] response: {result}")
                        partial_responses.append(f"[{sub_agent.name}] => {result}")
                    except Exception as e:
                        with self.lock:
                            self.stop = True
                        partial_responses.append(f"[{sub_agent.name}] error => {e}")

        # If one agent triggered self.stop, others check it in their loop
        # and exit. So partial_responses might have "Agent was stopped."
        # for each sub-agent that didn't finish.

        final_answer = "\n\n".join(partial_responses)
        # pdb.set_trace()
        return final_answer


    def chat_repl_all_doc(self, user_input):
        """
        Run self.chat_repl(...) on each sub-agent in self.doc_agents concurrently.
        If self.stop is set to True at any point,
        all sub-agents will stop the next time they check the while loop condition.
        """
        if self.stop:
            print("⚠️ Already in STOP mode. Skipping parallel sub-agents.")
            return "Agents were stopped."

        partial_responses = []

        # we create a thread for each sub-agent
        # with ThreadPoolExecutor(max_workers=len(self.agent.top_sub_docs_agents)) as executor: 
        with ThreadPoolExecutor(max_workers=4) as executor:
            # submit each agent to run chat_repl
            future_to_agent = {}
            for sub_doc_agent, idx in self.agent.top_sub_docs_agents:
                original_user_input = copy.deepcopy(user_input)
                self.reset_agent_state()
                # self.related_query = ""
                # self.updated_query = ""
                # self.CoA_raw_output = ""
                # self.user_shared = False
                # self.recall_check_response = "False"
                # self.check_if_web_searched_text = "false"
                future = executor.submit(self.chat_repl, sub_doc_agent, original_user_input)
                future_to_agent[future] = sub_doc_agent, idx

            # gather results as they complete
            for future in as_completed(future_to_agent):
                agent, idx = future_to_agent[future]
                try:
                    result = future.result()  # This calls chat_repl(...)
                    partial_responses.append(f"[subdoc agent {idx}] => {result}")
                except Exception as e:
                    # if there's an exception, we can set self.stop = True
                    with self.lock:
                        self.stop = True
                    partial_responses.append(f"[subdoc agent {idx}] error => {e}")

        # If one agent triggered self.stop, others check it in their loop
        # and exit. So partial_responses might have "Agent was stopped."
        # for each sub-agent that didn't finish.

        final_answer = "\n\n".join(partial_responses)
        # pdb.set_trace()
        return final_answer

    # def chat_repl(self, agent, user_input):
    #     count = 0
    #     if not self.stop:
    #         print("🤔 Thinking...🧠⏳...")
    #         self.reset_stop()
    #         task = agent.create_task(user_input)
    #         step_output = agent.run_step(task.task_id)
    #         while not self.stop and not step_output.is_last:
    #             print(f"🔄 Still working on it... (Iteration {count}): {step_output}")
    #             step_output = agent.run_step(task.task_id)
    #             count += 1
    #         if not self.stop:
    #             print("✅ Expert consultation complete! Finalizing the response 📄...")
    #             response = agent.finalize_response(task.task_id)
    #         else:
    #             print("⚠️ Process interrupted! Stopping the expert consultation ❌")
    #             response = "Agent was stopped."
    #     else:
    #         print("⚠️ Process already stopped! Unable to consult the expert ❌")
    #         response = "Agent was stopped."
    #     print("📝 Generating your response... Please hold on ⏳")
    #     return response
    async def chat_repl_coa(self, agent, user_input, max_retries=3):
        """Retry CoA planning without mutating agent memory.

        We keep the original prompt intact and append corrective guidance
        directly into the next call if a parsing error occurs.
        """
        retries = 0
        prompt_with_feedback = user_input
        last_error: ValueError | None = None

        while retries < max_retries:
            try:
                response = await agent.achat(prompt_with_feedback)
                return response
            except ValueError as err:
                retries += 1
                last_error = err
                error_feedback = (
                    "Error in generated plan: "
                    f"{err}. Regenerate a new valid plan with UNIQUE placeholders "
                    "for each function call (e.g., y1, y2, etc.; do not reuse or skip)."
                )
                prompt_with_feedback = (
                    f"{user_input}\n\n[RETRY INSTRUCTION]\n{error_feedback}"
                )
                print(f"Retry {retries}/{max_retries} for CoA agent due to: {err}")
            except Exception:
                # Let unexpected errors propagate to the caller.
                raise

        raise ValueError(
            f"Max retries ({max_retries}) exceeded for CoA agent. "
            f"Last error: {last_error}"
        )


    def chat_repl(self, agent, user_input):
        def _run_agent_loop(task):
            MAX_STEP_RETRIES = 3
            step_output = None
            last_error_feedback = None
            while True:
                with self.lock:
                    if self.stop:
                        print(f"🛑 [{agent.name}] Stop signal received during execution. Halting agent.")
                        return "Agent was stopped by a global signal."
                step_succeeded = False
                for step_attempt in range(MAX_STEP_RETRIES):
                    try:
                        with self.lock:
                            if self.stop:
                                print(f"🛑 [{agent.name}]Stop signal received during execution. Halting agent.")
                                step_output = None
                                break
                        print(f"🔄 [{agent.name}] Executing step, attempt {step_attempt + 1}...")
                        

                        if last_error_feedback:
                            print(f" 🔧 [{agent.name}] Applying correction feedback...")
                            step_output = agent.run_step(task.task_id, input=last_error_feedback)
                            last_error_feedback = None
                        else:
                            step_output = agent.run_step(task.task_id)


                        print(f"✅ [{agent.name}] Step successful.")
                        step_succeeded = True
                        break
                    except Exception as step_e:
                        print(f"❌ [{agent.name}] Step attempt {step_attempt + 1} failed: {step_e}")
                        if step_attempt + 1 >= MAX_STEP_RETRIES:
                            print(f"💀 [{agent.name}] Step failed after {MAX_STEP_RETRIES} attempts. Aborting task.")
                            raise step_e 
                        
                        last_error_feedback = f"""
                        The previous step failed with an error. 
                        Error message: {str(step_e)}.
                        This error likely occurred because the tool call was malformed (e.g., using something else such as a placeholder like 'y1' or a variable like '{{y}}' in the query string instead of a double quoated exact query string), or a placeholder tool name was used in the plan instead of the actual tool name.
                        Please analyze the error and your previous thought process, then provide a corrected plan for the next step.
                        DO NOT repeat the same mistake.
                        """
                        
                if not step_succeeded:
                     return "Agent failed to execute a step after multiple retries."
                
                if step_output and step_output.is_last:
                    print(f"🏁 [{agent.name}] Task naturally completed.")
                    break 
                time.sleep(21)
                

            print(f"✅ [{agent.name}] Expert consultation complete! Finalizing the response 📄...")
            return agent.finalize_response(task.task_id)
        
        try:
            with self.lock:
                if self.stop:
                    print(f"⚠️ [{agent.name}] Process already stopped! Unable to consult the expert ❌")
                    return "Agent was stopped before it could start."

            print(f"🤔 [{agent.name}] Thinking...🧠⏳...")

            self.reset_stop()

            task = agent.create_task(user_input)

            response = _run_agent_loop(task)

        except Exception as e:
            print(f"💥 [{agent.name}] An unrecoverable error occurred: {e}")
            traceback.print_exc()
            response = f"Agent failed with an unrecoverable error: {e}"
        
        print(f"📝 [{agent.name}] Final response for this agent: {response}")
        return response

    def check_complexity(self, user_input):
        # is_complex = self.agent.react_agent.chat(
        #     f"""
        #     Your task is to determine the COMPLEXITY of the user query below.

        #     SIMPLE: The query is straightforward and can be answered directly. For example:
        #     - Sharing the user's personal information, e.g., name, hobbies, favorites, etc.
        #     - Recalling  the user's personal information, e.g., name, hobbies, favorites, etc.
        #     - Recalling to previous information such as past conversations, numbers, questions, responses, or questions and responses, etc.
        #     - Formatting information.
        #     - Providing simple information that was previously discussed.
        #     - Recalling information from a document.
        #     - Looking for a specific number or information that can be answered directly by a document or a web search,
        #         - e.g., **Check for Exact Detail Requests: If the query specifically asks for exact text locations or detailed document excerpts.**
        #     - Summarizing a document.
        #     - Queries explicitly or implicitly involves using web search(es) as the **SOLE** method to obtain information (IMPORTANT: if web search(es) is NOT the ONLY method such as "using web search(es) and ..." etc, then it is a COMPLEX query):
        #         - Queries containing phrases like:
        #                                     - "Using web searches to tell ..."
        #                                     - "Use web searches, ..."
        #                                     - "Search online to find solutions to ..."
        #                                     - Queries implying reliance on external online searches as the ONLY method to retrieve information.

        #     COMPLEX: 
        #     MIX queries involve both web searches and documents to gather information, such as "using web search(es) and ..." etc.
        #     IMPORTANT: NEVER classify queries that explicitly or implicitly involve using web search(es) as the **SOLE** method to obtain information as complex, **UNLESS** the queries also requires the documents. 
        #     For non-web related queries which require deeper reasoning, multiple steps, or complex analysis. For example:
        #     - Recalling previous information, such as conversations, numbers, or questions, and then performing further actions based on the recalled information.
        #     - The query requires deeper reasoning, multiple steps, or complex analysis.
        #     - Planning a task or providing detailed explanations.

        #     Follow this logic:
        #     - If the query requests a specific number or fact, return FALSE, even if the data is large, complex, or ecological in nature, as long as it can be sourced directly from a document or a web search.
        #     - If the query is asking only to **recall** a number, fact, or concept already answered, return FALSE.
        #     - If the query requires deeper reasoning or analysis, or if further actions are needed based on the recalled information, return TRUE.
        #     - Do NOT include the quotes in your response, or provide any context. Simply state TRUE or FALSE.

        #     This is the user query: {user_input}
        #     """
        # )
        is_complex = self.agent.react_agent.chat(
            f"""
            simplely return TRUE for testing purposes"""
        )
        return is_complex

    def check_if_web_searched(self, user_input):
        if_web_searched = self.agent.react_agent.chat(
            f"""
            Your task is to determine if the user query below explicitly requests using web search(es).
            - If the user query explicitly requests using web search(es), return TRUE. Otherwise, return FALSE.
            - Do NOT include the quotes in your response, or provide any context. Simply state TRUE or FALSE.
            This is the user query: {user_input}
            """
        )
        return if_web_searched

    # def modify_query(self, user_input):
    #     updated_user_input = self.agent.react_agent.chat(
    #         f"""
    #         Your task is to use a dictionary {self.read_pre_ten_messages()} containing previous queries and corresponding responses to check if the current user query needs to be updated:
    #
    #         1. If there are previous queries, ensure that the current user query is linked to the most related query from the previous queries and update the user query using the context from the prior query.
    #
    #         2. This is the user query: {user_input}, and the dictionary {self.read_pre_ten_messages()} contains previous queries and corresponding responses.
    #
    #         3. If {self.read_pre_ten_messages()} is not None or empty and contains a query most related to the current query:
    #            - Update the user query based on the context from the selected previous query.
    #            - Tell user a related query (state the related query) was found in the previous conversation. Does the user want to update the current query to the updated user query (state the update query)?
    #               If the user confirms:
    #                Return the updated user query.
    #               Else:
    #                Return the original user query: {user_input}.
    #
    #         4. If there are no related queries found, return the original user query: {user_input}.
    #         """
    #     )
    #     return updated_user_input

    def confirm_user_update(self):
        if self.related_query:
            if self.related_query != "":
                llm_response = f"""A related query was found: '{self.related_query}'. Do you want to update your query to '{self.updated_query}'? (yes/no)"""
                return gr.update(value=llm_response, visible=True)
        return gr.update(value="")

    def check_related_query(self, user_input):  # Check if the query is user shared information
        related_query_agent = RelatedQueryReActAgent()
        # Use the secondary agent to check for related queries
        user_shared = related_query_agent.check_related_query(user_input)
        # Check if a related query was found
        # user_shared = related_query_response
        if user_shared.lower() in ['true']:
            self.user_shared = True
        else:
            self.user_shared = False
        """Comment  the below for removing the confirmation section"""
        # if "Related query found" in related_query_response:
        #     queries = related_query_response.split(",")
        #     self.related_query = queries[0].replace("Related query found: ", "")
        #     self.updated_query = queries[1].replace("and the updated query is: ", "")
        # else:
        #     self.related_query = ""
        #     self.updated_query = user_input

    """NOT USED CURRENTLY"""

    # def modify_query(self, user_input):
    #     self.check_related_query(user_input)
    #     if self.related_query != "":
    #         while self.confirm is None:
    #             time.sleep(1)  # Sleep for 1 second and check confirm clicking
    #         if self.confirm == "yes":
    #             response = self.updated_query
    #         else:
    #             response = user_input
    #     else:
    #         response = user_input
    #     self.confirm = None
    #     self.related_query = ""
    #     return response

    # def monitor_stop(self):
    #     """Thread that monitors the 'stop' flag and restarts the program when stop is True."""
    #     while True:
    #         if self.stop:
    #             print("Stop flag detected, restarting the program...")
    #             self.restart_program()
    #         time.sleep(0.5)  # Check every 0.5 seconds
    #
    # def restart_program(self):
    #     """Terminate and restart the program."""
    #     print("Terminating and restarting the program...")
    #     python = sys.executable  # Get the Python interpreter executable path
    #     script = sys.argv[0]  # Get the script name
    #     script_path = os.path.abspath(script)
    #
    #     # Enclose in quotes to handle spaces in paths
    #     os.execl(python, f'"{python}"', f'"{script_path}"', *sys.argv[1:])

    # def check_complexity(self, user_input):
    #     response = self.agent.react_agent.chat(
    #         f"""
    #         Your task is to determine the COMPLEXITY of the user query below, and check your memory if there is a previous question, ensure that the current user query is linked to the previous one by using the context from the prior query and update the user query accordingly.
    #
    #         SIMPLE: The query is straightforward and can be answered directly. For example, recalling past conversations, formatting, or providing simple information.
    #         COMPLEX: The query requires deeper reasoning, multiple steps, or complex analysis. For example, recalling from a document, summarizing a document, planning a task, or providing detailed explanations.
    #
    #         Follow this logic:
    #         1. If the current query is COMPLEX, return "TRUE" as the first output.
    #         2. Otherwise, return "FALSE" as the first output.
    #
    #         After determining the complexity, return the updated user query as the second output.
    #
    #         *Important*: When assessing complexity, rely solely on the context of the current query without using any prior memory.
    #
    #         Current user query: {user_input}
    #         """
    #     )
    #
    #     # Parse the response from the agent (assuming it returns TRUE/FALSE and the updated query)
    #     is_complex, updated_user_input = response.response.split("\n", 1)  # Split into two parts: complexity and updated query
    #
    #     return is_complex.strip(), updated_user_input.strip()

    def get_tool_access_info(self):
        text = helper.read_tool_access_info()
        return text
    
    def generate_response_from_scratch(self, user_input, is_complex):
        # Reset scope validation flag for new query
        self._scope_validation_applied = False

        if hasattr(self.agent, 'reset_all_agents'):
            self.agent.reset_all_agents()
        # global reset_raw_responses
        # amend user query
        # prelude = f"""
        #                 If the user asks about a specific named document,  you need to determine which available EXPERT TOOL to consult using the information below.
        #                 Note that all documents accessible to the TOOLS have been renamed with unique codes. If the user asks about a specific named document,
        #                 you must always look up the unique code to identify and use the appropriate EXPERT TOOL, *NOT* the DOCUMENT TOOL.
        #
        #                 {self.tool_access_info}
        #
        #                 *IMPORTANT: ALWAYS state the SOURCES (UNIQUE CODE AND FILE NAME) of information you are using in your response.
        #                 Remember, you may need to consult multiple tools to provide a comprehensive answer.
        #                 """
        # prelude = "" # reassign the value for testing tool_access_info whether it affects the top agent function calls (ignore hierarchy).
        def _dedupe_recent(items: list[str]) -> list[str]:
            return list(dict.fromkeys(reversed(items)))[::-1]

        def _append_tool_candidates(raw_ids: Iterable[str]) -> None:
            if not raw_ids:
                return
            with helper._tool_list_lock:
                helper.tool_list = [(item, float('-inf')) if isinstance(item, str) else item for item in helper.tool_list]
                existing = set([item[0] for item in helper.tool_list])
                for raw in raw_ids:
                    canonical = self._canonical_tool_id(raw)
                    if canonical and canonical not in existing:
                        helper.tool_list.append((canonical, float('-inf')))
                        existing.add(canonical)

            sorted_list = sorted(helper.tool_list, key=lambda x: x[1], reverse=True)[:RelevanceFilterMaxNum]
            helper.tool_list = [item[0] for item in sorted_list]
        print("Generating response 🖊️⏳....")
        if os.path.exists(persist_directory_summary) and os.listdir(persist_directory_summary):
            summary_index = VectorStoreIndex.from_vector_store(self.vector_store_summaries,
                                                               embed_model=helper.embed_model)
            processor = SimilarityPostprocessor(similarity_cutoff=0.5) # Increased from 0.01 to 0.5 for better quality retrieval in CoA
           
            
            # prompt_res = PromptTemplate(f"""According to the user query: {user_input},
            #                 provide **all tool ids** from metadata of the identified nodes, which are related to the user input. 
            #                 If multiple nodes found in your metadata, **ALWAYS provide the tool_ids from ALL of them and split them with commas**.
            #                 """)
            summary_query_engine = summary_index.as_query_engine(include_metadata=True,
                                                                 similarity_top_k=int(RelevanceFilterMaxNum/3)-2 if int(RelevanceFilterMaxNum/3)-2 > 1 else 1,
                                                                 processor=[processor], 
                                                                #  system_prompt=prompt_res
                                                                 )
            recalled_res = summary_query_engine.query(user_input)

            if "Empty Response" not in recalled_res.response:
                summary_candidates = [value["response"] for value in recalled_res.metadata.values()]
                _append_tool_candidates(summary_candidates)
            retriever = summary_index.as_retriever(
                similarity_top_k=int(RelevanceFilterMaxNum), # for testing the react agent
                # similarity_top_k = 5,
                search_type="similarity",  # Changed from "mmr" to ensure deterministic retrieval
                # search_type="mmr",  # MMR provides diversity but may cause non-deterministic results in multi-hop queries

            )
            snippet_postprocessor = RelevantSnippetPostprocessor(
                max_chars=1200,
                sentence_window=2,
            )
            tool_retriever = TrackingRetrieverTool(
                retriever=retriever,
                metadata=ToolMetadata(
                    name="document_retriever_tool",
                    description=(
                        "Searches and retrieves relevant text chunks from the documents to answer a question. "
                        "Use this to get the raw information needed for reasoning."
                    ),
                ),
                node_postprocessors=[snippet_postprocessor],
            )
            self.tool_retriever = tool_retriever
            self.tool_retriever.set_base_query(user_input)
        
            # for i in range(1, 50):
                # print(f"DEBUG: current iteration: {i}")
                # max_iterations = 30

                # new_system_prompt = f"""
                # You are a **Retrieval Only QA Agent**.

                # ---------------------------------------------------
                # PERMITTED ACTION
                # ---------------------------------------------------
                # Whenever you need information, run:

                # Thought: <one line reason for the query>
                # Action: document_retriever_tool
                # Action Input: {{{{"input": "<query string>"}}}}

                # ---------------------------------------------------
                # MUST FOLLOW RULES
                # ---------------------------------------------------
                # • **Never** use outside knowledge or inference.  
                # • **Only** speak when you can paste a sentence (or two adjacent sentences) copied verbatim from `document_retriever_tool` that explicitly shows the link between  
                #     - **Term A** = the entity the user asks about  
                #     - **Term B** = the attribute they want (year, style, founder, etc.)

                # • When you have such a sentence, reply **exactly**:

                # Answer: <attribute in ≤ 1 line>  
                # Evidence: "<exact quote>" (DocID)

                # • If you exhaust MAX_ITER queries without finding a sentence that contains *both* Term A and Term B, reply once:

                # Sorry - no document shows Term A together with Term B.  
                # Tried queries: [q1 ; q2 ; ...]

                # ---------------------------------------------------
                # SEARCH STRATEGY (loop until success or MAX_ITER)
                # ---------------------------------------------------
                # 1 Identify candidate phrases for **Term A** from the user's question  
                #  (e.g., by querying "<core clue>").

                # 2 Create a small keyword list for **Term B**.  
                # Example lists:  
                # - style → ["style","architectural style","architecture", "Gothic","Romanesque"]  
                # - year → ["year","built","completed","opened","construction"]  
                # - height → ["height","tall","meters","feet","stories"]

                # 3 For each candidate entity E and each keyword K in Term B list:  
                #  Thought: try "E K"  
                #  Action: document_retriever_tool with that query.  
                #  If any returned sentence contains both E **and** a Term B keyword, output the
                #  Answer/Evidence block and STOP.

                # 4 Every 3 failed attempts, you should change the Term A and try again. 
                #  If no quote after MAX_ITER, output the "Sorry - no document ..." message.

                # ---------------------------------------------------
                # NEVER DO
                # ---------------------------------------------------
                # • Never-answer without a matching Evidence quote.  
                # • Never mention "common knowledge" or "I can answer without tools".  
                # • Never fabricate or paraphrase evidence.

                # """
            for i in range(1):
                print(f"DEBUG: current iteration: {i}")
                
                reasoning_rule_prompt =""" 
                Generate a multi-step search plan for complex queries that may require information from multiple documents.

                QUERY ANALYSIS - First, identify the query type:
                - Single entity info: Requires finding attributes of one subject.
                - Multi-hop reasoning: Requires finding an intermediate entity (e.g., a university) to connect the subject (e.g., a coach) to the final answer (e.g., a city).

                **CRITICAL STRATEGY: DECOMPOSITION FOR MULTI-HOP QUESTIONS**
                For questions that require bridging information (e.g., "Where did Person X do Action Y?"), you MUST decompose the problem into a two-step logical chain within a single plan.

                **Step 1: Find the Intermediate Entity.** Generate searches to find the name of the connecting place, team, or organization.
                **Step 2: Find the Attribute of the Intermediate Entity.** Immediately generate searches for the attribute (e.g., location, date) of the entity you *expect* to find in Step 1.

                You must generate queries for **both steps in the same plan**. Do not wait for results. Your plan must be proactive and anticipate the necessary connections.

                **PRINCIPLE: Plan for the information you EXPECT to find.** If a query asks for a location where someone coached, you know you're looking for a *university* or a *team*. Your plan must first find the *name* of that university/team, and then find its *location*.

                **NAME VARIANT HANDLING:** Follow the Name Equivalence Guidance at all times. You may treat two name variants as the same entity only when differences are limited to middle names/initials, accents/diacritics, capitalization, punctuation, honorifics, or well-known translations **and** the surrounding context clearly anchors them to the same role/timeframe/event. If that contextual anchor is missing, keep the names separate and explicitly note that the linkage is unconfirmed.

                **CRITICAL EXAMPLE FOR MULTI-HOP QUESTIONS (like "Where did John Smith coach from 1970 to 1975?"):**

                **CORRECT LOGIC:**
                1.  The question asks for a **city**.
                2.  The **city** is the location of the **university** where he coached.
                3.  Therefore, my plan must first find the **university's name**, and then find the **university's location**.

                **CORRECT PLAN IMPLEMENTATION:**
                -- *Part 1: Find the intermediate entity (the university's name)*
                [FUNC document_retriever_tool("John Smith coaching 1970-1975") = y1]
                [FUNC document_retriever_tool("what university did John Smith coach at") = y2]

                -- *Part 2: Proactively find the attribute (location) of the entity found in Part 1*
                [FUNC document_retriever_tool("Fordham University location") = y3]
                [FUNC document_retriever_tool("Where is Fordham University?") = y4]
                [FUNC document_retriever_tool("New Jersey City University location") = y5]  <-- *It's good practice to add other likely candidates if context suggests them, but focusing on the most probable one is key.*

                **SEARCH TERM GUIDELINES:**
                - Keep searches SHORT and focused (4-6 keywords).
                - Use a mix of keyword-based searches (`"Fordham University location"`) and natural language questions (`"Where is Fordham University?"`).
                - Generate 4-6 total searches that cover both steps of the multi-hop logic.
                - When the attribute you need concerns a person, first run the exact-name query `"<name> born"`. Only if this fails to return relevant evidence should you consult the nickname map {"Bill": "William", "Bob": "Robert", "Gene": "Eugene", "Jack": "John", "Jim": "James", "Joe": "Joseph", "Liz": "Elizabeth", "Maggie": "Margaret", "Peggy": "Margaret", "Rick": "Richard", "Sue": "Susan"} and rerun the `"born"` query using the mapped full form when available.

                **MANDATORY OUTPUT FORMAT SYNTAX (CRITICAL):**
                *   Your plan MUST be a sequence of single line and inline string function calls using a strict format.
                *   The format is: `[FUNC document_retriever_tool("terms") = yX]`, where `yX` is a placeholder like `y1`, `y2`, etc.
                *   **KEY PRINCIPLES FOR SYNTAX:**
                    *   The `"terms"` MUST be enclosed in DOUBLE QUOTES.
                    *   **CRITICAL:** You MUST use the `= yX` placeholder at the end of each function call. This is how the system tracks results.
                    *   **USING PLACEHOLDERS IN ARGUMENTS:** When using a placeholder (e.g., `y1`) from a previous step, you MUST embed it directly inside the query string using `{{ }}` notation (e.g., `"county named after {{y1}}"`). Passing multiple comma-separated strings or concatenating with `+` is FORBIDDEN.
                    *   **EXAMPLE:** To find a county related to `y1` (which contains a person's name):
                    *   **CORRECT:** `[FUNC document_retriever_tool("county named after {{y1}}") = y2]`
                    *   **INCORRECT (CRITICAL FAILURE):** `[FUNC document_retriever_tool("county named after", "y1") = y2]`
                *   **EXAMPLES of CORRECT and INCORRECT syntax:**
                    *   **GOOD:** `[FUNC document_retriever_tool("terms") = y1]`
                    *   **BAD (causes critical failure):** `[FUNC document_retriever_tool("terms")]` -> FORBIDDEN. You MUST include `= yX` at the end.
                    *   **BAD (causes critical failure):** `[FUNC document_retriever_tool(terms) = y1]` -> FORBIDDEN. The terms MUST be in double quotes.

                CRITICAL: 
                This is ONLY the planning stage. Do NOT provide final answers here.
                Only output the function call plan with the EXACT format specified.
                """


                reasoning_structure_prompt ="""
                
                *Available functions:*
                ```python
                {functions}
                ```
                Input:
                {question}

                Abstract plan of reasoning:
                """

                reasoning_prompt = f"{reasoning_rule_prompt}\n\n{reasoning_structure_prompt}"
                reasoning_prompt = ChatMessage(role=MessageRole.SYSTEM, content=reasoning_prompt)
                reasoning_prompt = ChatPromptTemplate(message_templates=[reasoning_prompt])

                
                # # refine_reasoning_prompt ="""
                # # Synthesize the answer using ONLY the actual tool execution results from multiple searches.

                # # MULTI-DOCUMENT REASONING:
                # # 1. For the first step of entity identification, you MUST really carefully to find the exact and accurate entity the query is asking for, you can NOT use different connection to answer the question;
                # #     - For example, if the query is asking for the entity connecting "Oxford Parkway" and "Southern Newberry Parkway", you can not use the entity connecting "Oxford Parkway" and "Northern Newberry Parkway" to answer the question.
                # #     - For example, if the query is asking for events involving the countries that "opposed the Central Powers" in the First World War, you can not use the event involving "neutral countries" in the First World War to answer the question. Self-inference is NOT allowed.
                # # 2. Examine all tool results (y1, y2, y3, etc.) CAREFULLY
                # # 3. Look for PARTIAL information that can be combined
                # # 4. Find connections between different tool outputs (even if indirect).
                # # 4. Build complete picture from scattered pieces
                # # 5. If one result mentions entity and another mentions attribute, connect them
                # # 6. Only make connections that are logically supported by the tool outputs
                # # 7. Every tool result should only rely on the information from the retrieved texts from documents.
                # # 8. When you build the connection between entities and attributes, you MUST verify the connection is valid by then retrieved texts. Do NOT make any assumptions or inferences to bridge gaps between the document tool's output and the user's query, just report exact texts and state its limitations.

                # # CRITICAL RULES:
                # # - Use ONLY what the tools actually returned
                # # - **You NEVER use any other information that not in the doucments to support or inference your asnwer including common knowledge, external knowledge, or any other information.**
                # # - When linking info from different results, be explicit: "y1 shows X, y2 shows Y, therefore X+Y"
                # # - If tool outputs don't connect clearly, say "Results from y1 and y2 do not clearly link"

                # # SYNTHESIS PATTERNS:
                # # - Entity identification: "From y1: [entity found]" (If multiple entities found, use all of them for following steps)
                # # - Attribute linking: "From y2: [attribute of entity from y1]"  
                # # - Cross-verification: "y1 and y2 both confirm [fact]"
                # # - Gap identification: "y1 shows X but no tool result shows Y needed to answer fully"


                # # FORMAT:
                # # "Based on y1: [result], y2: [result], y3: [result] ... yn: [result] → [synthesized conclusion]"
                # # OR
                # # "Tool results y1: [output], y2: [output] ... yn: [output] do not provide sufficient information to answer the question"

                # # Output only the final synthesized conclusion.  

                # # Example
                # # -----------
                # # Question:  
                # # Which city did John Smith coach from 1970 to 1975?

                # # Previous reasoning:
                # # [FUNC document_retriever_tool("John Smith coaching 1970-1975") = "John Smith was head coach at Northwood University from 1970-1975."]
                # # After finding the Northwood University, we need to find the location of the Northwood University.
                # # [FUNC document_retriever_tool("Northwood University location") = "Northwood University is located in Midland, Michigan."]
                # # Response:  
                # # John Smith was head coach in Midland, Michigan from 1970-1975.

                # # Your Turn
                # # -----------
                # # Question:  
                # # {question}

                # # Previous reasoning:  
                # # {prev_reasoning}

                # # Response:
                
                # # """
                refine_reasoning_prompt ="""
                Your primary goal is to synthesize information from different tool outputs to construct a complete answer, especially for multi-hop questions. You must act as a strict fact-checker before combining information.

                **CRITICAL TASK: MULTI-HOP SYNTHESIS**
                You are expected to connect pieces of information across different tool results. This is not forbidden inference; it is the core requirement of the task.

                **SYNTHESIS PROCESS - Follow these steps with extreme precision:**

                1.  **Step 1: Identify the Intermediate Entity and its EXACT context.**
                    - Scan the first set of tool results (e.g., y1, y2).
                    - **CRITICAL: Pay strict attention to grammar.** Ensure the fact you extract is grammatically correct and unambiguous. For example, in the sentence "She appeared in Film A and had a supporting role in Film B," the supporting role is **ONLY for Film B**. Do not misattribute facts across different clauses.
                    - Quote the exact text that supports your finding.
                    - Apply the Name Equivalence Guidance: only treat name variants as the same entity when their differences are limited to middle names/initials, accents/diacritics, capitalization, punctuation, honorifics, or well-known translations **and** the surrounding context clearly ties them to the same role/timeframe/event. Otherwise, list them separately and state that the linkage is unconfirmed.
                    - **INSUFFICIENT INFORMATION CHECK:** 
                    - If you cannot complete the full answer but have found a key intermediate entity, your task is to request the missing information.
                    - **Example Output**: "I have found that the institution is 'Fordham University', but the current documents do not specify the city. I need to perform a new search for: 'Fordham University city location'."

                2.  **Step 2: Find the Final Attribute of the EXACT Intermediate Entity.**
                    - Look at the second set of tool results (e.g., y3, y4).
                    - **CRITICAL: Verify EXACT ENTITY MATCHING.** The name of the intermediate entity found in Step 1 (e.g., the movie title) MUST be an **exact, case-sensitive match** to the entity described in Step 2. 
                    - A partial match is a **failed match**. For instance, if the entity from Step 1 is **"Saw II"**, you MUST find a document that provides an attribute for **"Saw II"**, not "Saw".
                    - For person-centric attributes, verify that you at least executed `"<exact name> born"`. Only if that search fails and the first name appears in the nickname map {Bill→William, Bob→Robert, Gene→Eugene, Jack→John, Jim→James, Joe→Joseph, Liz→Elizabeth, Maggie→Margaret, Peggy→Margaret, Rick→Richard, Sue→Susan} should you require an additional `"<mapped name> born"` query. If the mapped query is skipped when applicable, treat Step 2 as incomplete and request it explicitly before synthesizing.
                    - Quote the exact text that provides the attribute for the correctly matched entity.

                2.5 **Step 2.5: Confirm Supporting Documents Before Synthesizing.**
                    - Collect the doc ids you intend to cite in the final answer and list them explicitly (e.g., `Docs considered: XXX_1, XXX_2, ...`).
                    - Verify that at least one document simultaneously contains the bridge entity from Step 1 **and** the target attribute/keyword from Step 2 (use domain-appropriate cues such as "number one"/"hit single"/"Billboard" for music, "born"/"birthplace" for people, "score"/"points"/"record" for performance, etc.).
                    - If no such document exists, do **not** synthesize yet; state the missing evidence and request a new targeted search (specify the query you need, e.g., `"Need search: '{{y1}} hit single'"`).

                3.  **Step 3: Combine and Conclude (ONLY if verification passes).**
                    - If and only if the grammatical link in Step 1 is clear AND the entity match in Step 2 is exact, logically combine the findings to form a direct answer.
                    - If either check fails, you must state that the information cannot be reliably combined.

                4.  **MANDATORY RESPONSE FORMAT (NO EXCEPTIONS):**
                    - Your final response **must** explicitly mention the subject from the query, the intermediate entity (Step 1), and the final attribute (Step 2) in a single coherent sentence or two short sentences.
                    - Example template: "[Subject] is connected to [Intermediate Entity], and [Intermediate Entity] has/occurs/is [Attribute]. Therefore, [Direct Answer]."
                    - Do **not** collapse to only the final attribute; always restate the bridge entity so downstream components retain the full reasoning chain.

                **RULES FOR SYNTHESIS:**
                - **GROUNDING IS KEY:** Every piece of your synthesized answer must be directly supported by a specific tool result.
                - **ALLOWED CONNECTION:** Connecting "Fact A" from `y1` with "Fact B" from `y3` is REQUIRED, **but only after passing the grammar and exact match verifications.**
                - **FORBIDDEN INFERENCE:** Do not assume connections. A mismatch between "Saw" and "Saw II" is a gap you are forbidden to bridge.
                - **INSUFFICIENT INFORMATION CHECK:** Conclude that you have insufficient information if:
                    - You cannot clearly complete Step 1 due to ambiguous grammar.
                    - OR, you complete Step 1, but cannot complete Step 2 because of an entity mismatch.

                ---
                **Example Walkthrough (Correct Logic)**

                Question:
                Which city did John Smith coach from 1970 to 1975?

                Previous reasoning:
                [FUNC document_retriever_tool("John Smith coaching 1970-1975") = y1] -> Result: "John Smith was head coach at Northwood University from 1970-1975."
                [FUNC document_retriever_tool("Northwood University location") = y2] -> Result: "Northwood University is located in Midland, Michigan."

                Response:
                Based on y1, the intermediate entity is "Northwood University". Based on y2, the attribute for the exactly matching entity "Northwood University" is the location Midland, Michigan. Therefore, John Smith coached in Midland, Michigan from 1970-1975.

                **Example Walkthrough (Handling Failure)**

                Question:
                What movie genre did Emmanuelle Vaugier play a supporting role in?

                Previous reasoning:
                [FUNC document_retriever_tool("...") = y1] -> Result: "She appeared as Addison Corday in 'Saw II' and had a supporting role in the film '40 Days and 40 Nights'."
                [FUNC document_retriever_tool("...") = y2] -> Result: "'Saw' is a horror film."

                Response:
                Based on y1, the document states Emmanuelle Vaugier had a supporting role in "40 Days and 40 Nights", not "Saw II". The document y2 provides the genre for "Saw", not "Saw II" or "40 Days and 40 Nights". The available information is insufficient to determine the genre of the film where she had a supporting role.

                ---
                **Your Turn**

                Question:
                {question}

                Previous reasoning:
                {prev_reasoning}

                Response:
                """
                refine_reasoning_prompt = PromptTemplate(refine_reasoning_prompt)
                coa_agent_worker = CoAAgentWorker.from_tools(
                tools=[tool_retriever],  
                llm=OpenAI(model="o4-mini"),
                reasoning_prompt_template=reasoning_prompt, 
                refine_reasoning_prompt_template=refine_reasoning_prompt, 
                output_parser=SafeChainOfAbstractionParser(verbose=True, auto_fix_duplicates=False),
                verbose=True, 
                max_iterations=5
                )
                coa_agent = coa_agent_worker.as_agent(memory=None)




                react_system_prompt = """
                You are a highly intelligent research assistant operating under extreme difficulty. The information from your tools comes as a list of noisy, often irrelevant document summaries. Your mission is to navigate this noise to find the answer.

                **Core Principles of Factual Grounding & Anti-Hallucination:**
                1.  **Source of Truth:** Your ONLY source of information is the content of the summaries returned by your tool.
                2.  **No External Knowledge:** You are STRICTLY FORBIDDEN from using external knowledge.
                3.  **Report Gaps, Do Not Invent:** If a connection is not explicitly stated in a summary, you MUST report it as missing. Hallucination is forbidden.

                **Your Overall Strategy for Multi-Hop Questions in a Noisy, Summary-Based Environment:**

                1.  **Step 1: Broad, Exploratory Search.**
                    - Your FIRST action MUST be a broad, simple search focused only on the main subject and context. DO NOT include the final desired attribute (like "city") in the first search.
                    - **Good first query example:** "John Smith coaching history" or "John Smith 1970-1975"
                    - **Bad first query example:** "John Smith coach 1970-1975 city"

                2.  **Step 2: Sift Through Noise to Find the Intermediate Entity.**
                    - After the first broad search, you will receive many summaries. Your task is to meticulously read ALL of them and find the MOST LIKELY intermediate entity (e.g., the name of the university).
                    - State your finding clearly in your thought process. Example: "Thought: The first summary is the most relevant. It states he coached at 'Northwood University' from 1970-1975. This is my key intermediate entity."
                    - Use the Name Equivalence Guidance: treat name variants as the same entity only when the differences are limited to middle names/initials, accents/diacritics, capitalization, punctuation, honorifics, or well-known translations **and** the surrounding context clearly anchors them to the same role/timeframe/event. Otherwise, keep variants separate and mark the linkage as unconfirmed.

                3.  **Step 3: Targeted, Verification Search.**
                    - Now that you have a SPECIFIC entity name, your SECOND action MUST be a targeted search for the final attribute of THAT entity.
                    - **Good second query example:** "Northwood University location city"
                    - This focused search has a much higher chance of success than the initial broad search.
                    - When the attribute is about a person, you MUST run `"<exact name> born"` before concluding failure. If that search returns no relevant summaries and the first name appears in the nickname map {Bill→William, Bob→Robert, Gene→Eugene, Jack→John, Jim→James, Joe→Joseph, Liz→Elizabeth, Maggie→Margaret, Peggy→Margaret, Rick→Richard, Sue→Susan}, run `"<mapped name> born"` as well. Only after the applicable searches fail may you conclude that the information is unavailable.

                4.  **Synthesize Final Answer:**
                    - Use the results from your targeted search to answer the question, if the question is a comparison question, you should include the attribute for each entity for the comparison result. If the targeted search also fails, then conclude that the information cannot be found.
                    - Answer the question strictly using information from the documents; do not add external knowledge or rely on common assumptions. Always name the specific entities involved. For example: If the question is "Where is the home country of the author of Harry Potter?", and the documents state "J. K. Rowling's home country is Britain," you must answer "J. K. Rowling's home country is Britain," rather than just "Britain."
                    - **Final Response Requirement:** Explicitly restate the subject, the intermediate entity you identified, and the final attribute in the concluding sentence. Do not skip the bridge entity even if the answer feels obvious.


                **Self-Correction / Anti-Loop:**
                - If your broad search in Step 1 yields NO relevant summaries at all, try rephrasing the broad search ONCE. If it still fails, stop and report that the subject was not found.
                - If your targeted search in Step 3 fails, stop and report that you found the intermediate entity but could not find its specific attribute. DO NOT go back to broad searches.
                
                """

                
                react_agent = OpenAIAgent.from_tools(
                    tools=[tool_retriever],  
                    llm=OpenAI(model="o4-mini"),
                    system_prompt=react_system_prompt,
                    verbose=True,
                    max_iterations=10
                )
                # new_system_prompt = """
                # Return ONLY the final answer in **one concise sentence**, immediately followed by `"Source: …"` that either (a) quotes the exact sentence (or fragment) from a retrieved document **or** (b) names the retrieved doc ID that states the fact.
                # **Core Directives:**
                # 1.  **Grounding**: Every claim you make must be directly traceable to a quote from the documents.
                # 2.  **No Fabrication**: You must never invent information or assume connections that are not explicitly stated.
                # 3.  **Fidelity**: Your final answer must be a synthesis of verified facts only.
                # """

                # candidate_expansion_prompt = PromptTemplate(
                # template="""
                # You are at a step in a reasoning process. You have the original query and the history of your previous steps (the "evidence chain"). Your task is to propose the NEXT logical step(s) to continue the investigation.

                # **Original Query**: {query_str}
                # **Evidence Chain So Far**: {chain_of_thought}

                # **Your Task**:
                # Based on the evidence so far, generate a list of 2-3 brief, diverse, and actionable next steps (hypotheses) to pursue.

                # **Consider these strategies:**
                # -   **Deepen**: If you've found a promising entity, propose a step to find a specific attribute of it.
                # -   **Connect**: If you have multiple pieces of information, propose a step to verify the link between them.
                # -   **Broaden**: If the current path seems like a dead end, propose an alternative starting point from the original query.
                # -   **Verify**: If you have a potential final answer, propose a step to double-check it against all original query constraints.

                # **Output Format**: A simple list of proposed next actions.
                # """)

                # reflection_prompt = PromptTemplate(
                #     template="""
                # **Original Query**: {query_str}
                # **Full Reasoning Chain**: {chain_of_thought}
                # **Last Step's Action**: {last_step_action}
                # **Last Step's Outcome**: {last_step_outcome}

                # Return *only* this JSON (no extra text):
                # {{
                # "score": <integer 1-10>,
                # "is_done": <true | false>,
                # "reasoning": "<≤50 tokens explanation>"
                # }}

                # Guidelines (STRICT - VIOLATE = FAILURE):
                # - score=10 ONLY if outcome FULLY answers query only use the explicit information from the documents without any external information including common knowledge.
                # - score 1-5 if partial match or external inference needed.
                # - Set is_done=true ONLY if score=10 AND no contradictions in chain. If contradictions, score<=5 and is_done=false.
                # - If ANY key term missing or not explicitly linked in docs, score<=5 and is_done=false.
                # - Reasoning MUST explain exact doc matches/mismatches.
                # """
                # )
                

                # lats_agent_worker = LATSAgentWorker.from_tools(
                #     tools=[tool_retriever],
                #     llm=OpenAI(model="o3-mini", temperature=0),
                #     # system_prompt=new_system_prompt,
                #     candidate_expansion_prompt=candidate_expansion_prompt,
                #     reflection_prompt=reflection_prompt,
                #     # chat_history=[ChatMessage(role=MessageRole.SYSTEM, content=new_system_prompt)] ,
                #     num_expansions=int(3),  
                #     max_rollouts=int(5),
                #     verbose=True
                # )

                
                # lats_agent = AgentRunner(lats_agent_worker, memory=None)

                # lats_agent.reset()
                # react_agent.agent_worker.max_iterations = max_iterations
            
                # react_agent.update_prompts({
                # "agent_worker:system_prompt": react_system_prompt 
                # })

                # print(f"DEBUG: max_iterations: {max_iterations}")
                # print(f"DEBUG: agent_worker.max_iterations: {react_agent.agent_worker.max_iterations}")
                # answer_synthesis_prompt = PromptTemplate(
                #     template="""
                # **Original Query**: {query_str}
                # **All Verified Evidence**:
                # {evidence_bullets}

                # Synthesize the FINAL ANSWER in ONE concise sentence, using ONLY what is explicitly supported by the evidence. If no direct answer, say "Not found in documents".
                # """
                # )
                
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError: 
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # evidence = loop.run_until_complete(lats_agent.achat(user_input))
                # evidence_bullets = evidence.response.replace("\n", "\n- ") if evidence else "- (no evidence)"
                # synthesized = lats_agent_worker.llm.complete(
                #     answer_synthesis_prompt.format(
                #         query_str=user_input,
                #         evidence_bullets=evidence_bullets
                #     )
                # ).text.strip()
                # reasoning_chain = tree.get_chat_history() 
                # recalled_res_additional_original = loop.run_until_complete(lats_answer_runner(user_input))
                # synthesized = coa_agent.chat(user_input)
                synthesized_coa= loop.run_until_complete(self.chat_repl_coa(coa_agent, user_input))
                synthesized_react= loop.run_until_complete(react_agent.achat(user_input))

                print(f"synthesized_coa: {synthesized_coa.response}")
                print(f"synthesized_react: {synthesized_react.response}")

                initial_response_prompt = f"""
                You are given two responses from two different agents for the user query: {user_input}, coa_agent: {synthesized_coa.response}, react_agent: {synthesized_react.response}
                
                1. If coa_agent's response can answer the query, you should use ALWAYS use coa_agent's response.
                2. Otherwise, if coa_agent's response cannot answer the query, and react_agent's response can answer the query, you should use react_agent's response.
                3. If neither response can answer the query, you should return the user query as the output.
                """

                initial_response = self.llm.complete(prompt=initial_response_prompt).text


                conclusion_prompt = f"""
                You are a helpful assistant that creates a final answer based STRICTLY on the initial response.
                
                **CRITICAL**: Only use information that is EXPLICITLY mentioned in the initial response. Do NOT add external knowledge, assumptions, or details not present in the response.
                
                INCLUDE all meaningful content from the ENTIRE reasoning process:
                - Names, organizations, countries, cities, dates
                - Key concepts, relationships, and connections  
                - Important descriptive terms and attributes
                - Intermediate concepts mentioned during reasoning
                - Related entities that were discussed
                
                EXCLUDE from your output:
                - Placeholder variables (y1, y2, y3, etc.)
                - System tool references (tool_xxx, SubCoA_x, etc.)
                - External knowledge not mentioned in the response
                - Assumptions or inferences beyond what is stated
                
                ------------------------------------------------------------
                Initial response: {initial_response}
                ------------------------------------------------------------
                Create a search sentence using ONLY information explicitly mentioned in the response:
                ------------------------------------------------------------
                """
                response = self.llm.complete(prompt=conclusion_prompt).text
                print(f"Initial response: {initial_response}")
                print(f"Conclusion: {response}")

                # lats_agent.reset()

            # recalled_res_additional_original = lats_agent.chat(user_input) 

            # print(recalled_res_additional_original.response)


        if "Empty Response" not in synthesized_react.response or "Empty Response" not in synthesized_coa.response:
            # prompt_res_additional = (f"""According to the user query: {recalled_res_additional.response},
            #             provide **all tool ids** from metadata of the identified nodes, which are related to the user input. 
            #             If multiple nodes found in your metadata, **ALWAYS provide the tool_ids from ALL of them and split them with commas**.
            #             """)
            summary_query_engine_additional = summary_index.as_query_engine(include_metadata=True,
                                                                similarity_top_k=RelevanceFilterMaxNum+2,
                                                                processor=[processor], 
                                                                # system_prompt=prompt_res_additional
                                                                )
            # recalled_res_additional = summary_query_engine_additional.query(recalled_res_additional_original.response)
            recalled_res_additional = summary_query_engine_additional.query(response)

            if "Empty Response" not in recalled_res_additional.response:
                supplemental_candidates = [value["response"] for value in recalled_res_additional.metadata.values()]
                _append_tool_candidates(supplemental_candidates)
            

            
        
        commit_retrieved_ids = False
        if "initial_response" in locals():
            if initial_response.strip() == user_input.strip():
                commit_retrieved_ids = True

        def _flush_tool_retriever(commit_flag: bool):
            tool_retriever_attr = getattr(self, "tool_retriever", None)
            if isinstance(tool_retriever_attr, TrackingRetrieverTool):
                print(f"Pending ids before flush (commit={commit_flag}): {sorted(tool_retriever_attr._pending_ids)}")
                tool_retriever_attr.flush_pending_ids(commit=commit_flag)

        # is_complex = self.check_complexity(user_input)
        # user_input = self.modify_query(user_input)  #self.modify_query(user_input).response
        # self.check_related_query(user_input)
        question_type_complex = False
        if str(is_complex.response).lower() == "true":
            agent_type = 'CoA'
            question_type_complex = True

        else:
            agent_type = 'Doc'  # 'ReAct'

        # enhanced_query_prompt = f"""
        # You are a helpful assistant that enhances or returns the original user queries. 

        # You are given:
        # - Original user's query: {user_input}
        # - Initial response: {recalled_res_additional_original.response}

        # First, you need to analyze the initial response carefully:

        # 1. If it does NOT provide a direct answer to the user's query (including cases where it contains QUALIFYING WORDS like "while," "although," "but," "however," "not detailed," "not available," "cannot be found," or similar expressions that indicate limitations or lack of complete information), but only provides related information:
        # Return the original user's query "{user_input}" without any changes immediately, do not process further steps.

        # 2. If it provide a direct answer to the user's query (does NOT contain any QUALIFYING WORDS and expressions that indicate limitations or lack of complete information), then:
        # Enhance the query by adding more keywords and details based on the initial response but you CANNOT add any words that are not present in the initial response. The purpose of this enhancement is to transform queries that require multi-step reasoning into queries that can be answered with a single-step search.
        # """

        enhanced_query_prompt = f"""
        You are a helpful assistant that enhances or returns the original user queries. 

        You are given:
        - Original user's query: {user_input}
        - Initial response: {response}

        STEP 1 - RESPONSE COMPLETENESS ANALYSIS:
        Carefully analyze if the initial response provides a COMPLETE and DIRECT answer to the user's query:

        Ask yourself:
        - Does the initial response answer ALL aspects of the user's query?
        - Does the initial response provide specific, definitive information without hedging?
        - Are there any statements indicating missing, unavailable, or incomplete information in the initial response?
        - Does the initial response use any language that suggests uncertainty or limitations?

        STEP 2 - DECISION:
        Based on your analysis:

        - If the initial response is INCOMPLETE, UNCERTAIN, or indicates ANY limitations or missing information:
          Return exactly the original user's query "{user_input}".

        - If the initial response is COMPLETE, CERTAIN, and fully answers the query:
          Return two lines in the following format:
            {user_input}
            Additional focus: <extra tokens drawn ONLY from the initial response>

          The "Additional focus" clause must reuse wording from the initial response (names, dates, places, numeric facts, etc.) and should highlight key details that would help a retrieval system repeat the answer. You cannot use any external knowledge or assumptions.

        Remember: When in doubt, do NOT enhance; just return the original query.

        FORMAT:
        - If not enhancing, output exactly the original query as a single line.
        - If enhancing, output the original query on the first line and the "Additional focus: ..." line on the second line.
        - Do NOT add quotation marks, brackets, or explanations beyond the specified structure.
        """

        if os.getenv("MILONET_DISABLE_QUERY_ENHANCEMENT", "").lower() in {"1", "true", "yes"}:
            enhanced_query = user_input.strip()
        else:
            enhanced_query = self.llm.complete(prompt=enhanced_query_prompt).text.strip()

        original_query_clean = user_input.strip()
        if enhanced_query and enhanced_query != original_query_clean:
            user_input = enhanced_query
        else:
            user_input = original_query_clean
        print(f"Enhanced query: {user_input}")

        _flush_tool_retriever(True)


        modified_user_input = core_prompts.create_top_prompt(

            # prelude=prelude,
            # expert_hierarchy=self.agent.expert_hierarchy,
            summary_list=[item[0] if isinstance(item, tuple) else item for item in helper.tool_list],
            # tools=self.tools,
            user_query=user_input,
            complex=question_type_complex,
            mode_content=self.current_mode_content,
        )
        # pdb.set_trace()
        # print("MODIFIED INPUT:\n")
        # print(modified_user_input)

        # agent_type = 'CoA'
        # helper.reset_raw_responses = False
        # if helper.reset_raw_responses:
        helper.raw_responses = []
        start_time = time.time()

        if agent_type == 'ReAct':
            try:
                llm_response = self.agent.react_agent.chat(modified_user_input)
            except Exception as e:
                print(e, traceback.print_exc())
        elif agent_type == 'ToT':
            llm_response = self.agent.ToT_agent.chat(modified_user_input)
        elif agent_type == 'LLMC':
            llm_response = self.agent.LLMC_agent.chat(modified_user_input)
        elif agent_type == 'CoA':
            try:
                # llm_response = process_user_query(self.agent.CoA_agent, modified_user_input)
                #   llm_response = self.agent.CoA_agent.chat(modified_user_input)
                print("This is a complex question, let me consult my expert teams ⏳....")
                # llm_response = self.chat_repl(self.agent.CoA_agent, modified_user_input)
                # print(modified_user_input)
                helper.update_tool_list()
                if helper.tool_list:
                    helper.tool_list = _dedupe_recent(helper.tool_list)
                llm_response = self.chat_repl_all_coa(modified_user_input, test=False)

            except Exception as e:
                print(e, traceback.print_exc())
                llm_response = "Invalid functions were called internally or agent automatically parsed query incorrectly"
        elif agent_type == 'Doc':
            try:
                # llm_response = process_user_query(self.agent.CoA_agent, modified_user_input)
                #     llm_response = self.agent.top_doc_agent.query(user_input)
                print("This is a good question, let me consult an expert ⏳....")
                if self.check_if_web_searched_text.lower() == 'true':
                    print("Starting web searches 🔍")
                if self.user_shared or self.recall_check_response.lower() == "true":
                    # llm_response = self.chat_repl(self.agent.top_doc_agent, user_input)
                    llm_response = self.chat_repl_all_doc(user_input)
                else:
                    # llm_response = self.chat_repl(self.agent.top_doc_agent, modified_user_input)
                    llm_response = self.chat_repl_all_doc(modified_user_input)
            except Exception as e:
                print(e, traceback.print_exc())
        else:
            llm_response = "Invalid agent type selected."
        _flush_tool_retriever(True)

        helper.tool_list.clear()
            # helper.update_tool_list()

            # helper.tool_list = [f"tool_{tool}" for tool in helper.tool_list]

        end_time = time.time()
        print("The agent response time: " + str((end_time - start_time) / 60) + " minutes ⏱️")
        # write output to log file
        with open("raw_responses.txt", "w", encoding='utf-8') as file:
            for response in helper.raw_responses:
                file.write(response + "\n")
        print("File written successfully. 😊")

        # helper.reset_raw_responses = True
        return llm_response, question_type_complex

    def serialize_llama_response(self, response):
        if not isinstance(response, str):
            return {
                'response': response.response,  # Assuming response has a 'response' field
                # 'sources': response.sources if hasattr(response, 'sources') else None,
                'metadata': response.metadata if hasattr(response, 'metadata') else None
            }
        else:
            return {
                'response': response,
                # 'sources': None,
                'metadata': None
            }

    def post_process_output(self, if_complex, user_input,
                            serial_response):  # Post-processing the initial answer for making it more detailed and adding web searches.
        res_llm = ProcessFinalResponseLLM(self.CoA_raw_output, user_input)
        if if_complex:  # Post-processing for complex queries that require web searches or supplementary information
            print('Starting processing final output for the complex query ⏳')
            processed_response = res_llm.process_raw_CoA(self.current_mode_content)
            if processed_response == "WEB SEARCH REQUIRED" or self.check_if_web_searched_text.lower() == 'true':
                modified_user_input = f"""
                *USER QUERY*
                The user has asked the following question: {user_input}
                Use your web search tools to answer the question.
                Always cite the web sources with links you searched in the response, and make the USER AWARE that this information is not from the web sources.
                Do NOT use your base knowledge.
                Once you have enough information to answer the question, DO NOT seek further information, DO NOT describe your process, just provide a FULL ANSWER to the user.
                The final answer including the source citations should NOT be over 1000 words.
                """
                print("Starting web searches 🔍")
                # try:
                processed_response_web = self.agent.web_search_agent.chat(modified_user_input)

                if processed_response == "WEB SEARCH REQUIRED":
                    processed_response = res_llm.process_complex_web(self.current_mode_content,
                                                                     processed_response_web.response)
                else:
                    processed_response = res_llm.process_mix_query(self.current_mode_content, processed_response,
                                                                   processed_response_web.response)  # Combine web search and documents resulted answers
                # except Exception as e:
                #     processed_response= "Invalid web content returned.❌"
                #     print(e, traceback.print_exc())
                #     print(processed_response)

            serial_response["response"] = processed_response
            print('Finishing processing final output ✅')
        else:  # Post-processing for simple queries that only require web searches
            if self.check_if_web_searched_text.lower() == 'true':
                print('Starting processing final output for the simple web search required query 🔍')
                prompt = f""" The user has asked the following question: {user_input}, and you can access the initial output: {serial_response["response"]}, check if the initial output contains any useful information from cited web sources.
                                If it contains, return True, otherwise return False.
                                Only return True or False.
                                """
                simple_web_search_already = res_llm.llm.complete(prompt=prompt).text
                if simple_web_search_already.lower() == 'false':
                    modified_user_input = f"""
                                   **IMPORTANT: Here ONLY Use your do_google_search_tool and read_webpage_content_tool to gather information from the top 5 search results to provide a detailed response to the user's query. Do NOT use any document-related tools.**
                                   *USER QUERY*
                                   The user has asked the following question: {user_input}
                                   Always cite the web sources with links you searched in the response, and make the USER AWARE that this information is not from the web sources.
                                   Do NOT use your base knowledge.
                                   Once you have enough information to answer the question, DO NOT seek further information, DO NOT describe your process, just provide a FULL ANSWER to the user.
                                   The final answer including the source citations should NOT be over 1000 words.
                                   """
                    processed_response_web = self.agent.web_search_agent.chat(modified_user_input)
                    processed_response = res_llm.process_complex_web(self.current_mode_content,
                                                                     processed_response_web.response)
                    serial_response["response"] = processed_response
                    print('Finishing processing final output ✅')

        return serial_response



    def post_process_output_subCoAs(self, user_input,
                            serial_response):  # Post-processing the initial answer for making it more detailed and adding web searches.
        milonet_variant = os.getenv("MILONET_VARIANT", "full").strip().lower()
        if milonet_variant not in {"core", "full"}:
            raise ValueError(f"Unsupported MILONET_VARIANT={milonet_variant!r}; expected 'core' or 'full'.")
        enable_final_checks = milonet_variant == "full"
        print(f"MiloNet variant: {milonet_variant}; final synthesis checks enabled: {enable_final_checks}")
        res_llm = ProcessCombinedResponseLLM(
            user_input,
            serial_response["response"],
            synthesis_mode=milonet_variant,
            enable_final_checks=enable_final_checks,
        )
        print('Starting processing final output for the combined response ⏳')
        processed_response = res_llm.process_subCoAResponses(self.current_mode_content)
        serial_response["response"] = processed_response
        print('Finishing processing final output ✅')
        return serial_response

    def update_CoA_texts(self):  # Update the CoA chain text for post-processing
        start_index = self.display_writer.str_output.find("==== Generated Chain of Abstraction ====")
        end_index = self.display_writer.str_output.rfind("The agent response time: ")
        if start_index != -1 and end_index != -1:
            self.CoA_raw_output = self.display_writer.str_output[
                                  start_index:end_index]

    def _canonical_tool_id(self, tool_id: str | None) -> str | None:
        if not tool_id:
            return None
        raw_id = tool_id.split("tool_", 1)[1] if tool_id.startswith("tool_") else tool_id
        if helper.SHORT_ID_PATTERN.match(raw_id):
            return raw_id
        shortened = helper.get_shortened_id_from_file(raw_id)
        return shortened or raw_id

    def process_input(self, user_input, selected_mode):  # Processing the user query
        self.check_if_web_searched_text = 'false'

        # def process_user_query(top_agent, top_modified_user_input):
        #     top_helper = TopAgentHelper(top_agent)
        #     expert_tools = tools
        #     final_response = top_helper.query_experts_concurrently(expert_tools, top_modified_user_input)
        #     return final_response
        self.stop = False
        self.pre_user_input = user_input
        self.pre_mode = selected_mode
        self.cleared = False  # Reset the cleared flag on new input

        # Reset tool filter cache so each query rebuilds its own candidate set.
        helper.tool_list = []

        # write user query to log file
        # with self.lock:
        #     # async with self.async_lock:
        #     with open("conversation_log.txt", "a") as f:
        #         f.write(f"User: {user_input}\n\n")

        # Retrieve the mode content based on the selected mode
        # mode_content = self.modes.get(selected_mode, "")  # MODE NOT UESED YET
        """START Commented out the related query check and is_complex check for ragbench testing"""
        # self.check_related_query(user_input)  # Check if the query is user shared information
        # is_complex = self.check_complexity(user_input)  # Check if the query is complex
        # if str(is_complex.response).lower() == "true":
        #     if_complex = True
        # else:
        #     if_complex = False
        """END Commented out the related query check and is_complex check for ragbench testing"""

        self.user_shared = False
        from types import SimpleNamespace

        is_complex = SimpleNamespace()
        is_complex.response = "TRUE"
        print("THE CURRENT USER INPUT: ", user_input)
 
        
        if_complex= True
        """"Implement the memory as a json file on the disk"""
        # memory_file_name = "memory_store.json"
        # previous_input = ""
        # # serial_response = ""
        # if_complex = False
        # if os.path.exists(memory_file_name):
        #     memory_dict = check_memory(memory_file_name)
        #     max_score = 0.9
        #     for input in memory_dict.keys():
        #         score = compare_query_similarity(user_input, input)
        #         if score > max_score:
        #             serial_response, if_complex = memory_dict[input]
        #             max_score = score
        #             previous_input = input
        #
        #     if max_score == 0.9:
        #         llm_response, if_complex = self.generate_response_from_scratch(user_input)
        #         serial_response = self.serialize_llama_response(llm_response)
        #         if if_complex:
        #             memory_dict[user_input] = (serial_response, if_complex)
        #     elif if_complex:
        #         print("The query is complex and similar to the previous answered question: "+ str(previous_input)+ ". Recall the previous answer.")
        # else:
        #     llm_response, if_complex = self.generate_response_from_scratch(user_input)
        #     serial_response = self.serialize_llama_response(llm_response)
        #     memory_dict = dict()
        #     if if_complex:
        #         memory_dict[user_input] = (serial_response, if_complex)
        # with open(memory_file_name, "w", encoding='utf-8') as f:
        #     json.dump(memory_dict, f, ensure_ascii=False, indent=4)
        """End of json memory implementation"""

        # Check if the query is referring to the previous questions
        llm = OpenAI(model="gpt-5-mini")
        prompt = (f"""
                      Your task is to determine if the user query refers directly or indirectly to previous questions, or to previous questions and their responses.
                        Examples of direct references:
                        - "What is the last question?"
                        - "What is the second last question I asked?"
                        - "What are the questions I asked and their responses?"
                        - "List all questions I asked?"

                        Examples of indirect references:
                        - "Format the response into a table." (implicitly refers to the last response).
                        - If the user is asking about specific personal information they previously shared, such as "What is my name?" or "Do you remember my favorite color?". (implicitly refers to a previous response).
                        - "What did I say about emissions?" (implicitly refers to a previous response).
                        - "Can it help with global warming mitigation?" ('it' implicitly refers to the last response).

                        If the query refers to previous questions or responses, return True. Otherwise, return False.

                        ***Always only return True or False based on the query: "{user_input}".***

                        """)
        recall_check_response = llm.complete(prompt=prompt)
        self.recall_check_response = recall_check_response.text


        new_recall_check_response_text = None
        if self.recall_check_response.lower() == "true":
            prompt_refer = f"""
                Your task is to determine if the user query requests new information based on the context of previous questions and their responses.

                Examples of queries that do NOT request new information (they only recall or manipulate previously shared details):
                    - "What is the last question?"
                    - "What is the second last question I asked?"
                    - "What are the questions I asked and their responses?"
                    - "List all questions I asked?"
                    - Asking about previously shared personal information, e.g., "What is my name?" or "Do you remember my favorite color?"
                    - "Format the response into a table."
                    - "What did I say about emissions?"
                    - "Compare these three places." (where 'these' implicitly refers to the last three questions and responses)

                Examples of queries requesting NEW information based on previous questions/responses:
                    - "Can it help with global warming mitigation?" 
                      (Based on the previous context, now asking about global warming mitigation.)
                    - "Compare it with a carbon sink?" 
                      (Based on the previous context, now asking to compare with a new subject: carbon sink.)

                If the query asks for new information based on previous context, return True.
                Otherwise, return False.

                ***Always only return True or False based on the query: "{user_input}".***
            """
            new_recall_check_response = llm.complete(prompt=prompt_refer)
            new_recall_check_response_text = new_recall_check_response.text

        """Commented check_if_web_searched for ragbench evaluations"""
        # check_if_web_searched = self.check_if_web_searched(user_input)
        check_if_web_searched = SimpleNamespace()
        check_if_web_searched.response = "False"
        self.check_if_web_searched_text = check_if_web_searched.response

        """Start of chroma DB vector store"""
        if (not os.path.exists(persist_directory) and os.listdir(persist_directory)) or not self.db_memory_enabled:
            llm_response, if_complex = self.generate_response_from_scratch(user_input, is_complex)
            serial_response = self.serialize_llama_response(llm_response)
            if "Agent was stopped." not in str(serial_response['response']):
                if if_complex:
                    self.update_CoA_texts()

                # original_response_text = str(serial_response["response"])
                serial_response = self.post_process_output_subCoAs(user_input, serial_response)
                # serial_response = self.post_process_output(if_complex, user_input, serial_response)
                # if self.recall_check_response.lower() == "false":
                #     print("Storing into DB ...💾")
                #     add_query_response_to_store(user_input, original_response_text, serial_response["response"],
                #                                 self.CoA_raw_output, self.check_if_web_searched_text, if_complex,
                #                                 self.user_shared, self.pre_mode)
                # pdb.set_trace()
                self.CoA_raw_output = ""


        else:
            index = VectorStoreIndex.from_vector_store(helper.vector_store, embed_model=helper.embed_model)
            prompt_res = ("""
                            Provide a response based on the metadata of the found nodes, following these rules:

                            1. **Ignore all user processing requests**:
                               - Disregard any requests for formatting, summarization, or any type of processing (e.g., "format into a table," "summarize," etc.).
                               - Retrieve and return the raw information as instructed below.

                            2. **Always include source citations**:
                               - Include all source citations exactly as provided in the metadata within the detailed response.

                            3. **Retrieve questions based on timestamps**:
                               - **If the user requests a specific number or all of previous questions, retrieve and rank them in chronological order (from oldest to newest) based on their timestamps.**

                            4. **Recognizing implicit question and response**:
                               - If the user refers the previous question and response e.g.:"Format the response into a table", you should recognize the implicit reference to the last question and its response.
                               - If the user is asking about specific personal information they previously shared (such as "What is my name?", "Do you know me?", "Who am I",  or "Do you remember my favorite color?"), you should recognize the implicit reference to a previous question and its response.
                               - If the user is asking about recent questions or statements based on recency (e.g., "What did I say about emissions?"), you should recognize the implicit reference to a previous question and its response.

                            5. **Handling previous questions**:
                               - **For the last single question or response**:
                                 - If the user requests the content of the last single question (or parts of it), retrieve and return the `question` attribute of the metadata of the most recent found node.
                                 - If the user requests both the last single question and its response, retrieve and return the `question` and `response` attributes of the metadata of the most recent found node, while preserving all source citations.
                               - **For more than one previous questions or responses**:
                                 - If the user requests the content of more than one previous questions, **retrieve and list the `question` attributes for all relevant source nodes' metadata**.
                                 - If the user requests both more than one previous questions and their responses, **retrieve and list the `question` and `response` attributes from all relevant source nodes' metadata**.

                            **ALWAYS RETAIN SOURCE CITATIONS IN THE RESPONSES.**     
                                 """)
            # 6.  **Handling related questions to the previous questions**:
            #     - If the user's query text is related to a `question` of a relevant source node's metadata, **retrieve the `response` attribute from the relevant source node's metadata**.
            # Use the user query's embeddings to and nodes' embeddings s find the relevant nodes
            prompt_complex = ("""**Your task is to determine whether the user query is complex or not. Ignore all user processing requests:**:
                                 Disregard any requests for formatting, summarization, or any type of processing (e.g., "format into a table," "summarize," etc.).
                                 **Return False immediately if ANY of the following conditions are met:**
                                     1. The user query directly or implicitly refers to previous questions or their responses. Examples:
                                         - If the user asks about recent questions or statements based on recency, such as:
                                            - "What is the previous question and its response, and format its response into a table?"
                                            - "What did I say about emissions?"
                                         - If the user requests processing based on previous responses, such as:
                                            - "Format the response into a table", you should recognize this is an implicit reference to the last response.
                                         - If the user is asking about specific personal information they previously shared.
                                     2. The user query explicitly or implicitly involves using web search(es) as the **SOLE** method to obtain information (IMPORTANT: if web search(es) is NOT the SOLE method e.g. queries like "using web search(es) and ..." to gather information, it should RETURN the complex of True or False from metadata of the found node). Examples:
                                        - Queries containing phrases like:
                                            - "Using web searches to tell ..."
                                            - "Use web searches, ..."
                                            - "Search online to find solutions to ..."
                                        - Queries implying reliance on external online searches as the only method to retrieve information.
                                 **If none of the above conditions are met:**
                                    - Provide the complex of True or False from metadata of the found node. 
                                                     """)
            # check_if_web_searched = self.check_if_web_searched(user_input)
            # self.check_if_web_searched_text = check_if_web_searched.response
            if recall_check_response.text.lower() == "true":

                postprocessor = FixedRecencyPostprocessor(top_k=10,
                                                          date_key="timestamp")  # For the query recalling previous questions.

                query_engine = index.as_query_engine(include_metadata=True, verbose=True, similarity_top_k=1000,
                                                     node_postprocessors=[postprocessor],

                                                     )
            else:
                # if check_if_web_searched.response.lower() == "true":
                #     processor = SimilarityPostprocessor(
                #         similarity_cutoff=0.6)  # For the query explicitly using web searches
                # else:
                processor = SimilarityPostprocessor(
                        similarity_cutoff=0.6)  # For other queries asking normal questions for information
                # relativity_prompt = PromptTemplate(template=prompt_res)
                query_engine = index.as_query_engine(include_metadata=True, verbose=True,
                                                     node_postprocessors=[processor]
                                                     )

            recalled_res = query_engine.query(
                "According to the user query: " + user_input + '\n' + prompt_res)  # Get the similar node considering the average similarity

            recalled_complex = query_engine.query(
                "According to the user query: " + user_input + '\n' + prompt_complex)  # Get the similar node's complex if found

            if "Empty Response" not in recalled_res.response.strip('.'):
                print("Recalled the response from the DB 💾")
                questions_list = [
                    node.metadata["question"]
                    for node in recalled_res.source_nodes
                ]
                responses_list = [
                    node.metadata["response"]
                    for node in recalled_res.source_nodes
                ]
                time_stamps = [
                    node.metadata["timestamp"]
                    for node in recalled_res.source_nodes
                ]
                serial_response = {
                    'metadata': recalled_res.metadata
                }
                similar_node_found = False
                recalled_similar_res = recalled_res
                if "Empty Response" not in recalled_res.response.strip('.'):
                    if recall_check_response.text.lower() != "true":  # Second check for the similar node with the similar query text only.
                        recalled_similar_res = query_engine.query(user_input + '\n')
                        if "Empty Response" not in recalled_similar_res.response.strip('.'):
                            score = recalled_similar_res.source_nodes[0].score
                            if score > recall_similarity and recalled_similar_res.source_nodes[0].node.metadata["response"] and not (
                                    recalled_similar_res.source_nodes[0].node.metadata["complex"] == False and
                                    recalled_similar_res.source_nodes[0].node.metadata[
                                        "web_search_required"].lower() == "true" and check_if_web_searched.response.lower() == "false"):
                                if if_complex==True:
                                    similar_node_found = True
                                else:
                                    print(
                                        "A similar query was found, but since it is a simple query, this current query will be regenerated for a better answer.✖️")
                            if score > recall_similarity and recalled_similar_res.source_nodes[0].node.metadata["response"] and (
                                    recalled_similar_res.source_nodes[0].node.metadata["complex"] == False and
                                    recalled_similar_res.source_nodes[0].node.metadata[
                                        "web_search_required"].lower() == "true" and check_if_web_searched.response.lower() == "false"):
                                print(
                                    "A similar query was found, but since it was generated involving a web search and the current query does not explicitly require a web search, it will not be considered for generating a new answer.🔍✖️")
                        else:
                            recalled_similar_res = recalled_res

                # Get the most similar node for the similar question, get all save complex and user shared information by using last 10 saved questions in DB (put in the lists).
                if recall_check_response.text.lower() == "true":
                    prompt = (f"""
                    Based on the user query: {user_input}, 
                        You have access to three corresponding lists: the previous question list: {questions_list}, the response list: {responses_list} and the timestamps list: {time_stamps}.
    
                        IMPORTANT: The provided lists are already sorted by timestamp from newest to oldest, i.e., index 0 represents the most recent question and response.
    
                        Instructions:
                        1. **If the user requests a specific number of `N` or all of previous (last) questions, retrieve `N` questions from the question list based on their timestamps.**
                        2. If the user query refers to the previous questions or their responses (either explicitly or implicitly), fulfill the query using the provided previous questions and responses from the question and response lists based on their timestamps. Examples:
                             - If the user asks the previous question(s) and/or response(s) such as: "What is/are the previous question(s) and its/their response(s)", then return the FULL and EXACT CONTENTS from the question and response lists. 
                             - **NEVER return brief placeholders in responses such as "response content as provided in the response list", "Full detailed response as provided in the response list". YOU MUST ALWAYS return **FULL and EXACT response texts, even it is reposting the user providing contents.**.
                             - If the user asks the the previous question(s) and/or response(s) and a further processing, such as: "What is the previous question and its response, and format its response into a table" or "what is/are the previous question(s), do an analysis on it", then do the further processing on the retrieved question and response, such as formatting or analysing as requested.
                             - If the user asks: "Format the response into a table" you should recognize the implicit reference to the last response and act accordingly.
                             - If the user asks for new information that relies on previous questions and/or responses, ensure that you include the relevant context from those previous interactions when answering.
                             - Always retrieve the previous question(s) based on their timestamps.
    
                    Respond with your output based on the user query.
                    - If the user refers the previous responses, **ALWAYS return FULL and EXACT response texts**, even some of all of them are similar to each other. NEVER return brief placeholders in responses, even it is reposting the user providing contents.
                    - If the user refers the previous questions, also include the recalled questions' texts in your response. 
                    - If there is no useful information found in previous responses, you should use your base knowledge, but ensure that citations are provided for any base knowledge used in the final response.
                    - *IMPORTANT*:  **ALWAYS keep the source citations in the responses. Always retrieve the previous question(s) based on their timestamps. If the sources are not from the documents (tools), you MUST inform the user.
                    YOU MUST ANSWER THE USER QUERY IN A MEANINGFUL WAY, NOT JUST RECALL THE PREVIOUS RESPONSES AND QUESTIONS, ALSO, ensure that citations including documents with their reference numbers, base knowledge, and web sources (containing valid links) for every piece of information in the final response, UNLESS, the response only recalls previous questions.** 
                    """)
                else:
                    prompt = (f"""
                               Based on the user query: {user_input}, answer the query by the full and exact texts of {recalled_similar_res.source_nodes[0].node.metadata["response"]}.
                               - *IMPORTANT*:  **Ensure that citations including documents with their reference numbers, base knowledge, and web sources (containing valid links) for every piece of information in the final response if possible. 
                                If the sources are not from the documents (tools), you MUST inform the user.** 
                               """)
                if not self.stop:
                    recall_response = llm.complete(prompt=prompt, timeout=1000)
                    # Check if the response need to be regenerated
                    if not self.user_shared and ((
                                                         str(is_complex.response).lower() == 'true' and recalled_complex.response.lower() == "false" and recall_check_response.text.lower() != "true") or (
                                                         not similar_node_found and recall_check_response.text.lower() != "true")):
                        if str(is_complex.response).lower() == 'true' and recalled_complex.response.lower() == "false" and recall_check_response.text.lower() != "true":
                            print(
                                "The current query is not directly referring, but related to the previous queries and the current query is more complex, it needs to be regenerated. ⏳")
                        if not similar_node_found:
                            print(
                                "The current query is not directly referring, and a similar query found but it is not strongly related to the current query, it needs to be regenerated for a new answer. ⏳")
                        llm_response, if_complex = self.generate_response_from_scratch(user_input, is_complex)
                        serial_response = self.serialize_llama_response(llm_response)

                        if "Agent was stopped." not in str(serial_response['response']):
                            if if_complex:
                                self.update_CoA_texts()
                            original_response_text = str(serial_response["response"])
                            serial_response = self.post_process_output_subCoAs(user_input, serial_response)
                            # if not self.stop:
                            #     # serial_response = self.post_process_output(if_complex, user_input, serial_response)
                            #     if recall_check_response.text.lower() != "true":
                            #         print("Storing into DB ...💾")
                            #         add_query_response_to_store(user_input, original_response_text,
                            #                                     serial_response["response"],
                            #                                     self.CoA_raw_output, self.check_if_web_searched_text,
                            #                                     if_complex,
                            #                                     self.user_shared, self.pre_mode)
                            # else:
                            if self.stop:
                                serial_response["response"] = "Agent was stopped."
                            self.CoA_raw_output = ""
                    else:
                        if_complex = True if (recalled_complex.response.lower() == "true" or str(
                            is_complex.response).lower() == "true") else False
                        if recall_check_response.text.lower() == "false":
                            if if_complex:
                                self.CoA_raw_output = recalled_similar_res.source_nodes[0].node.metadata["COA_text"]
                                node_num = len(recalled_similar_res.source_nodes)
                                if node_num > 1 and self.CoA_raw_output == "":
                                    for i in range(1, node_num):
                                        coA_text = recalled_similar_res.source_nodes[i].node.metadata["COA_text"]
                                        if coA_text != "":
                                            self.CoA_raw_output = coA_text
                                            break
                                if self.CoA_raw_output == "":
                                    llm_response, if_complex = self.generate_response_from_scratch(user_input,
                                                                                                   is_complex)
                                    serial_response = self.serialize_llama_response(llm_response)
                                    self.update_CoA_texts()
                                serial_response = self.post_process_output_subCoAs(user_input, serial_response)
                                # if not self.stop:
                                #     # original_response_text = str(recalled_res.response)
                                #     serial_response = self.post_process_output(if_complex, user_input,
                                #                                                serial_response)
                                #
                                # else:
                                #     serial_response["response"] = "Agent was stopped."
                                if self.stop:
                                    serial_response["response"] = "Agent was stopped."


                            else:

                                serial_response["response"] = recall_response.text
                                previous_mode = recalled_similar_res.source_nodes[0].node.metadata["mode"]
                                # previous_web_search_required = recalled_similar_res.source_nodes[0].node.metadata["web_search_required"]
                                if previous_mode != self.pre_mode or self.check_if_web_searched_text.lower() == "true":
                                    llm_response, if_complex = self.generate_response_from_scratch(user_input,
                                                                                                   is_complex)
                                    serial_response["response"] = llm_response.response
                                    serial_response = self.post_process_output_subCoAs(user_input, serial_response)
                                    # if not self.stop:
                                    #     serial_response = self.post_process_output(if_complex, user_input,
                                    #                                                serial_response)
                            # if not self.stop:
                            #     # if recall_check_response.text.lower() != "true" and (if_complex or self.user_shared):
                            #     if recall_check_response.text.lower() != "true":
                            #         print("Storing into DB ...💾")
                            #         add_query_response_to_store(user_input, recalled_similar_res.response,
                            #                                     serial_response["response"], self.CoA_raw_output,
                            #                                     self.check_if_web_searched_text,
                            #                                     if_complex,
                            #                                     self.user_shared, self.pre_mode)
                            if self.stop:
                                self.CoA_raw_output = ""
                        else:
                            serial_response["response"] = recall_response.text
                            # if new_recall_check_response_text.lower() == "true" and new_recall_check_response_text:
                            #     print("Storing into DB ...💾")
                            #     add_query_response_to_store(user_input, recall_response.text,
                            #                                 serial_response["response"],
                            #                                 self.CoA_raw_output, self.check_if_web_searched_text,
                            #                                 False,
                            #                                 self.user_shared, self.pre_mode)
                else:
                    serial_response["response"] = "Agent was stopped."
                    self.CoA_raw_output = ""
            else:
                print("No related queries found 🤔")
                llm_response, if_complex = self.generate_response_from_scratch(user_input, is_complex)
                serial_response = self.serialize_llama_response(llm_response)
                # if if_complex:
                if "Agent was stopped." not in str(serial_response['response']):
                    if if_complex:
                        self.update_CoA_texts()
                    original_response_text = str(serial_response["response"])
                    serial_response = self.post_process_output_subCoAs(user_input, serial_response)
                    # serial_response = self.post_process_output(if_complex, user_input, serial_response)
                    # if recall_check_response.text.lower() != "true" and (if_complex or self.user_shared):

                    # if recall_check_response.text.lower() != "true":
                    #     print("Storing into DB ...💾")
                    #     add_query_response_to_store(user_input, original_response_text, serial_response["response"],
                    #                                 self.CoA_raw_output, self.check_if_web_searched_text, if_complex,
                    #                                 self.user_shared, self.pre_mode)
                    self.CoA_raw_output = ""

        """Implement a json dictionary for last 10 queries shared between different agents"""
        # if os.path.exists(self.memory_file_name):
        #     memory_dict = check_memory(self.memory_file_name)
        # else:
        #     memory_dict = LimitedDict()
        # memory_dict[user_input] = serial_response['response']
        # with self.lock:
        # # async with self.async_lock:
        #     with open(self.memory_file_name , "w", encoding='utf-8') as f:
        #         json.dump(memory_dict, f, ensure_ascii=False, indent=4)
        """End of chroma DB vector store"""
        # with self.lock:
        #     # async with self.async_lock:
        #     with open("conversation_log.txt", "a", encoding='utf-8') as f:
        #         f.write(f"LLM: {serial_response['response']}\n\n\n")
        # if not os.path.exists(file_name):
        # self.agent.chat_store.persist(persist_path=file_name)
        # else:
        #     with open(file_name, 'r') as file:
        #         store =  json.load(file)
        # print("\n Persisting chat store in chat_store.json\n")
        # helper.tool_list = []
        return serial_response, if_complex

    # def process_input(self, user_input, selected_mode):
    #     self.cleared = False  # Reset the cleared flag on new input
    #
    #     # write user query to log file
    #     with open("conversation_log.txt", "a") as f:
    #         f.write(f"User: {user_input}\n\n")
    #
    #     # Retrieve the mode content based on the selected mode
    #     mode_content = self.modes.get(selected_mode, "")
    #
    #     # amend user query
    #     prelude = f"""
    #     If the user asks about a specific named document, you need to work out which tool to consult using the information below.
    #     Note that all documents available to the TOOLS have been renamed to give them unique reference numbers. If the user asks about a specific named document,
    #     you need to look up the reference number for the relevant document so you know which tool to use. When querying the tool, always refer to the document by
    #     the reference number (not the original file name).
    #
    #     {self.tool_access_info}
    #
    #     *IMPORTANT: ALWAYS provide REFERENCES to the SOURCES of information you are using in your response.
    #     Remember, you may need to consult multiple tools to provide a comprehensive answer.
    #     """
    #
    #     modified_user_input = core_prompts.create_top_prompt(
    #         prelude=prelude,
    #         expert_hierarchy=self.agent.expert_hierarchy,
    #         user_query=user_input
    #     )
    #     print(modified_user_input)
    #     is_complex = self.check_complexity(user_input)
    #
    #     if str(is_complex).lower() == "true":
    #         agent_type = 'CoA'
    #     else:
    #         agent_type = 'ReAct'
    #
    #     if agent_type == 'ReAct':
    #         llm_response = self.agent.react_agent.chat(modified_user_input)
    #     elif agent_type == 'ToT':
    #         llm_response = self.agent.ToT_agent.chat(modified_user_input)
    #     elif agent_type == 'LLMC':
    #         llm_response = self.agent.LLMC_agent.chat(modified_user_input)
    #     elif agent_type == 'CoA':
    #         llm_response = self.agent.CoA_agent.chat(modified_user_input)
    #     else:
    #         llm_response = "Invalid agent type selected."
    #
    #     # write output to log file
    #     with open("conversation_log.txt", "a") as f:
    #         f.write(f"LLM: {llm_response}\n\n\n")
    #
    #     return llm_response, ""

    def clear_conversation(self):
        with self.lock:  # Lock the operation
            # with open("conversation_log.txt", "w") as f:
            #     f.write("")
            self.agent.react_agent.memory.reset()
            self.agent.ToT_agent.memory.reset()
            self.agent.CoA_agent.memory.reset()
            self.agent.LLMC_agent.memory.reset()
            # reset_memory(self.memory_file_name)
            self.log_output = "All cleared."
            self.cleared = True
            output_logger.output.clear()
            llm_response = ""
            return llm_response, self.log_output

    def stop_chat(self):
        with self.lock:  # Lock the operation
            self.stop = True
            self.agent.set_stop(True)

    def reset_stop(self):
        with self.lock:
            self.stop = False
            self.agent.set_stop(False)

    def clear_log(self):
        with self.lock:  # Lock the operation
            self.log_output = ""

    def update_log(self):
        with self.lock:  # Lock the operation
            if self.cleared:
                return self.log_output
            if self.user_send:
                if self.log_enabled:
                    self.log_output = self.display_writer.get_general_output()
                else:
                    self.log_output = self.display_writer.get_print_output()
            return self.log_output

    def set_subcoa_phase2_truncation(self, enabled: bool):
        with self.lock:
            self.truncate_phase2_subcoa = bool(enabled)
            state = "enabled" if self.truncate_phase2_subcoa else "disabled"
            print(f"SubCoA Phase 2 truncation {state}")

    def _filter_subcoa_input(self, prompt: str) -> str:
        if not self.truncate_phase2_subcoa or not isinstance(prompt, str):
            return prompt
        markers = ("## Phase 2", "**--- PHASE 2")
        cutoff = None
        for marker in markers:
            index = prompt.find(marker)
            if index != -1:
                cutoff = index if cutoff is None else min(cutoff, index)
        if cutoff is None:
            return prompt
        return prompt[:cutoff].rstrip()

    def toggle_logging(self, logging_enabled):
        with self.lock:  # Lock the operation to prevent simultaneous toggling
            print(f"Toggling logging: {logging_enabled}")  # Debug statement
            self.log_enabled = logging_enabled
            if logging_enabled:
                # sys.stdout = output_logger
                # sys.stderr = output_logger
                self.display_writer.log_enabled = logging_enabled
            else:
                # sys.stdout = original_stdout
                # sys.stderr = original_stderr
                self.display_writer.log_enabled = False
                print("Logging disabled")  # Debug statement

    def selected_mode(self, mode):
        print("Updated the output mode to " + mode + " 🔧")
        self.pre_mode = mode
        self.current_mode_content = self.modes.get(mode, "")

    def toggle_db_memory(self, db_memory_enabled):
        with self.lock:  # Lock the operation to prevent simultaneous toggling
            print(f"DB memory: {db_memory_enabled}")  # Debug statement
            self.db_memory_enabled = db_memory_enabled
            # if db_memory_enabled:
            #     sys.stdout = output_logger
            #     sys.stderr = output_logger
            # else:
            #     sys.stdout = original_stdout
            #     sys.stderr = original_stderr
            #     print("Logging disabled")  # Debug statement

    # def process_input_wrapper(self, *args):
    #     response = self.process_input(*args)
    #     if isinstance(response, tuple) and len(response) == 2:
    #         agent_response, user_input = response
    #         if hasattr(agent_response, 'response'):
    #             markdown_output = f"{agent_response.response}\n\n"
    #             if hasattr(agent_response, 'sources') and agent_response.sources:
    #                 markdown_output += "**Sources:**\n"
    #                 for source in agent_response.sources:
    #                     markdown_output += f"- {source}\n"
    #             return markdown_output, user_input
    #         elif isinstance(agent_response, str):
    #             return agent_response, user_input
    #     return response
    def process_input_wrapper(self, *args):
        print("Starting analysing...🧠⏳")
        self.user_send = True
        # CoAFailed = False
        self.file = open('console_output.txt', 'w', encoding='utf-8')
        self.display_writer.str_output = ""
        self.display_writer.set_file(self.file)
        self.display_writer.enable_file_output()

        self.trace_file = open('retrieval_traces_milo-full_041225.txt', 'a', encoding='utf-8')
        self.trace_file.write(f"\n{'='*50}\n")
        self.trace_file.write(f"NEW SESSION: {datetime.datetime.now()}\n")
        self.trace_file.write(f"{'='*50}\n")

        self.display_writer.set_trace_file(self.trace_file)
        self.display_writer.enable_trace_log()

        # def custom_replacement(match):
        #     # Get the prefix (e.g., "Document" or "Tool")
        #     prefix = match.group(1) if match.group(1) else ""
        #     # Get the captured ID part
        #     id_value = match.group(2)
            
        #     if prefix == "Tool":
        #         if id_value.startswith("tool_"):
        #             tool_id = id_value 
        #         else:
        #             tool_id = f"tool_{id_value}" 
        #         description = self.index_map.get(id_value.replace("tool_", ""))
              
        #         return f"{tool_id}: {description}" if description else tool_id
        #     elif prefix == "Document":
              
        #         if id_value.startswith("tool_"):
        #             id_value = id_value[5:] 
        #         description = self.index_map.get(id_value)
        #         return f"{prefix} {id_value}: {description}" if description else f"{prefix} {id_value}"
        #     else:
        #         description = self.index_map.get(id_value)
        #         return f"{id_value}: {description}" if description else id_value
        
        """Testing the process_input function with a JSONL batch file."""
        input_file = os.getenv("MILONET_UI_BATCH_INPUT", "ragbench_5_judgement_v2.jsonl")
        output_file = os.getenv("MILONET_UI_BATCH_OUTPUT", "my_rag_output_test_good_4.1_mini_080725_4.jsonl")
        batch_limit_text = os.getenv("MILONET_UI_BATCH_LIMIT", "").strip()
        batch_limit = int(batch_limit_text) if batch_limit_text else None
        results = [] # Initialize results list outside the loop

        # Open the output file in append mode BEFORE the loop
        with open(output_file, "a", encoding="utf-8") as outfile: 
            with open(input_file, "r", encoding="utf-8") as infile:
                count = 0
                for line in infile: # This loop iterates through every question
                    if batch_limit is not None and count >= batch_limit:
                        break
                    data = json.loads(line.strip())
                    # if data["judged_quality"].lower() == "poor":
                    #     continue
                    question = data.get("original_question") or data.get("question")
                    if not question:
                        raise ValueError(f"Missing question field in batch row {count}: {data}")
                    case_id = str(data.get("id") or count)
                    os.environ["MILONET_COST_LOG_CASE_ID"] = case_id
                    os.environ["MILONET_COST_LOG_STAGE"] = "milonet_ui_wrapper_query"
                    print(f"Processing question: {question}") 
                    
                    response_data = self.process_input(question, "Chat")
                    # Basic error handling/check if response is as expected
                    if isinstance(response_data, tuple) and len(response_data) > 0 and isinstance(response_data[0], dict) and "response" in response_data[0]:
                        response = response_data[0]["response"]
                    else:
                        response = "Error: Unexpected response format from process_input"
                        print(f"Unexpected response format for question '{question}': {response_data}")

                    # Prepare the result dictionary
                    result = {
                        "case_id": case_id,
                        "question": question,
                        "response": response
                    }
                    
                    # Write the result immediately to the output file
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()


                    print(f"Finished processing question {question}")
                    print(f"Response: {response}")
                    helper.tool_list = []
                    count += 1

        # The final writing loop is no longer needed as we write inside the main loop
        # with open(output_file, "w", encoding="utf-8") as outfile:
        #     for result in results:
        #         outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

        print("Done processing the configured JSONL batch.")

        # response = self.process_input(*args)
        
        # pdb.set_trace()
            # try:
            #     response = self.process_input(*args)
            # except Exception as e:
            #     print(e, traceback.print_exc())
            #     response = "", ""
            #     self.file.close()
            # """Rerunning for the occasionally fail to execute the CoA plan """
            # if response == ("", "") or not (
            #         "Abstract plan of reasoning:" in response[0]["response"] or ("= y1" or "=y1") in response[0][
            #     "response"] or "Identify Relevant Expert Colleague Tools:" in response[0]["response"]):
            #     break
        
        self.display_writer.flush()
        self.display_writer.disable_file_output()
        self.file.close()
        self.display_writer.disable_trace_log()
        self.trace_file.close()
        helper.tool_list = []
        exit(0)
        # user_query = ""

        # Complex = False
        # if isinstance(response[1], str):
        #     if response[1].lower() == "true":
        #         Complex = True
        #     else:
        #         Complex = False
        # elif isinstance(response[1], bool):
        #         Complex = response[1]
        #
        # if self.display_writer.str_output != "":
        #     if "Answer: FALSE\n" in self.display_writer.str_output:
        #         Complex = False
        #     elif "Answer: TRUE\n" in self.display_writer.str_output:
        #         Complex = True
        #

        # query_start = self.display_writer.str_output.find("This is the user query: ")
        # if query_start != -1:
        #     query_start += len("This is the user query: ")
        #     query_end = self.display_writer.str_output.find("\n", query_start)
        #     if query_end == -1:
        #         query_end = len(self.display_writer.str_output)
        #     user_query = self.display_writer.str_output[query_start:query_end].strip()
        #
        # start_index = self.display_writer.str_output.find("==== Generated Chain of Abstraction ====")
        # end_index =  self.display_writer.str_output.rfind("========================")
        # if start_index != -1 and end_index != -1:
        #     self.CoA_raw_output = self.display_writer.str_output[start_index:end_index + len("========================")]

        # res_coa_file = 'response_COAchain_pairs.json'
        # if Complex:
        #     if self.CoA_raw_output == "":
        #         with open(res_coa_file, 'r', encoding='utf-8') as f:
        #             self.res_coa_dict = json.load(f)
        #             self.CoA_raw_output = self.res_coa_dict[user_query]
        #     else:
        #         if os.path.exists("res_coa_file.json"):
        #             with open(res_coa_file, 'r', encoding='utf-8') as f:
        #                 self.res_coa_dict = json.load(f)
        #         self.res_coa_dict[user_query] = self.CoA_raw_output
        #         with open(res_coa_file, 'w', encoding='utf-8') as f:
        #                 json.dump(self.res_coa_dict, f, indent=4)
        # self.CoA_raw_output = ""
        print('Closing output file ✅')
        print("\n")
        print("\n")

        # if Complex:
        #     print('Starting processing final output')
        #     # if self.CoA_raw_output =="":
        #     #     self.CoA_raw_output =
        #     res_llm = ProcessFinalResponseLLM(self.CoA_raw_output, user_query)
        #     processed_response = res_llm.process_raw_CoA()
        #     response = processed_response, ""
        #     print('Finishing processing final output')

        # sys.stdout = original_stdout
        # sys.stderr = original_stderr

        # THE BELOW BLOCK IS FOR REPLACE THE FILE NAME WITH TOOL NUMBER, BUT IT IS NO NEEDED FOR THE RGABENCH TEST

        # pattern = re.compile(r"(tool|Tool)?[\s_]*(\d+(_\d+)+)")
        # if isinstance(response, tuple) and len(response) == 2:
        #     agent_response, if_complex = response
        #     if isinstance(agent_response, dict) and 'response' in agent_response:
        #         markdown_output = f"{agent_response['response']}\n\n"
        #         source_lst = agent_response.get('sources')
        #         if isinstance(source_lst, str):
        #             source_lst = source_lst.replace("'", "").strip('[').strip(']').strip().split(",")
        #             source_lst = [s.strip() for s in source_lst if s.strip()]
        #         # else:
        #         #     print(source_lst)
        #
        #         """Comment the source list for the update the prompt and serialised response for DB store"""
        #         # if len(source_lst) > 0:
        #         #     markdown_output += "**Sources:**\n"
        #         #     for source in source_lst:
        #         #         markdown_output += f"- {source}\n"
        #
        #         # self.agent.chat_store.persist(persist_path="chat_store.json")
        #         # print("\n Persisting chat store in chat_store.json\n")
        #
        #         # pattern = r"\d+(_\d+)+"
        #
        #         # def custom_replacement(match):
        #         #     found_pattern = match.group()  # Get the matched pattern (e.g., "123_456")
        #         #     # Generate a replacement for each pattern (e.g., prefix with "Pattern_" and add found pattern)
        #         #     replacement = f"{self.index_map.get(found_pattern)}: {found_pattern}"
        #         #     return replacement
        #
        #         updated_text = re.sub(pattern, custom_replacement, markdown_output)
        #         return updated_text, ""
        #     elif isinstance(agent_response, str):
        #         updated_text = re.sub(pattern, custom_replacement, agent_response)
        #         # self.agent.chat_store.persist(persist_path="chat_store.json")
        #         # print("\n Persisting chat store in chat_store.json\n")
        #         return updated_text, ""
        #

        # THE ABOVE BLOCK IS FOR REPLACE THE FILE NAME WITH TOOL NUMBER, BUT IT IS NO NEEDED FOR THE RGABENCH TEST


        # self.agent.chat_store.persist(persist_path="chat_store.json")
        # print("\n Persisting chat store in chat_store.json\n" )
        # self.regenerate_initialised = True
        self.reset_stop()
        return response[0]["response"], ""

    def regenerate(self):
        if self.db_memory_enabled:
            if self.recall_check_response.lower() == 'false': #and self.check_if_web_searched_text.lower() == 'false':
                self.db_memory_enabled = False
            response = self.process_input_wrapper(self.pre_user_input, self.pre_mode)
            self.db_memory_enabled = True
        else:
            response = self.process_input_wrapper(self.pre_user_input, self.pre_mode)

        return response

    def run(self):

        js_func = """
        function refresh() {
            const url = new URL(window.location);
            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
        """

        # self.clear_log()

        class CustomTheme(Base):
            def __init__(self):
                super().__init__()
                self.font_size = "18px"
                self.spacing_size = "4px"
                self.radius_size = "10px"
                self.text_color = "#FFFFFF"
                self.background_fill = "#2B3035"
                self.block_background_fill = "#343A40"
                self.input_background_fill = "#495057"
                self.button_primary_background_fill = "#0056B3"
                self.button_secondary_background_fill = "#6C757D"

        custom_theme = CustomTheme()

        with gr.Blocks(theme=custom_theme, css="""
           #left-textbox {
                width: 700px;
                margin-top: 50px;
            }

            #yes-button, #no-button {
                width: 80px;
            }

            #button-container {
                align-items: left;
                width: 80px;
                margin-top: 50px;
            }
            #unique-llm-output {
                font-size: 18px !important;
                max-height: 68vh !important;
                overflow-y: auto !important;
                padding: 15px !important;
                /* Ensure only one scrollable area */
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }

            #unique-llm-output pre {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }
            .large-button {
            height: 84px !important;
            }
            """, js=js_func) as miloNet:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# MiloNet [v0.13]")
                    log_output = gr.Textbox(label="Internal processes", lines=25, interactive=False, value="")

                    with gr.Row():
                        with gr.Column(scale=3, min_width=160):
                            mode_options = list(self.modes.keys())
                            mode_dropdown = gr.Dropdown(label="Select Mode", choices=mode_options, value="Chat")
                        with gr.Column(scale=1, min_width=60):
                            clear_button = gr.Button("Reset", min_width=60)
                        with gr.Column(scale=1, min_width=100):
                            logging_enabled = gr.Checkbox(label="Observe", value=False)
                        # with gr.Column(scale=1, min_width=100):
                        #     db_memory_enabled = gr.Checkbox(label="DB Memory", value=False)
                        with gr.Column(scale=1, min_width=80):
                            regen_button = gr.Button("Regenerate", min_width=60, interactive=False)
                        with gr.Column(scale=1, min_width=60):
                            stop_button = gr.Button("Stop", min_width=60)
                    with gr.Row():
                        with gr.Column(scale=6, min_width=600):
                            user_input_text = gr.Textbox(label="Textbox", lines=3, container=False, value="",
                                                         interactive=True)
                        with gr.Column(scale=1, min_width=60):
                            send_button = gr.Button("Send", elem_classes="large-button")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=60):
                            db_memory_enabled = gr.Checkbox(label="DB Memory", value=True, visible=False)

                with gr.Column(scale=1):
                    # Adding a unique ID and updating the CSS targeting.
                    # with gr.Row():
                    #     user_prompt = gr.Markdown("", elem_id="left-textbox")
                    #     with gr.Column(elem_id="button-container"):
                    #         yes_button = gr.Button("Yes", elem_id="yes-button")
                    #         no_button = gr.Button("No", elem_id="no-button")
                    llm_output = gr.Markdown(elem_classes="output-markdown", elem_id="unique-llm-output")

            # send_button.click(
            #     fn=self.process_input_wrapper,
            #     inputs=[user_input, mode_dropdown],
            #     outputs=[llm_output, user_input],
            # )

            gr.on(
                triggers=[user_input_text.submit, send_button.click],
                fn=self.process_input_wrapper,
                inputs=[user_input_text, mode_dropdown],
                outputs=[llm_output, user_input_text],
                queue=False
            ).then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=regen_button)

            stop_button.click(fn=self.stop_chat,
                              inputs=[],
                              outputs=[],
                              queue=False)

            clear_button.click(
                fn=self.clear_conversation,
                inputs=[],
                outputs=[llm_output, log_output],
                queue=False
            )

            def set_yes():
                if self.related_query:
                    self.confirm = "yes"
                    # self.confirm_future.set_result(True)

            def set_no():
                if self.related_query:
                    self.confirm = "no"
                    # self.confirm_future.set_result(True)

            # yes_button.click(fn=set_yes, outputs=None)
            # no_button.click(fn=set_no, outputs=None)

            regen_button.click(fn=lambda: gr.update(interactive=False), inputs=None, outputs=regen_button).then(
                fn=self.regenerate, inputs=[],
                outputs=[llm_output, log_output],
                queue=False).then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=regen_button)

            # fn=self.regenerate,
            # inputs=[],
            # outputs=[llm_output, log_output],
            # queue=False
            # )
            mode_dropdown.change(
                fn=self.selected_mode,
                inputs=mode_dropdown,
                outputs=[],

            )
            logging_enabled.change(
                fn=self.toggle_logging,
                inputs=logging_enabled,
                outputs=[],
                # queue=False
            )
            db_memory_enabled.change(
                fn=self.toggle_db_memory,
                inputs=db_memory_enabled,
                outputs=[],
            )
            # miloNet.load(fn=self.confirm_user_update, inputs=None, outputs=user_prompt, every=0.1)
            # self.initialize()
            miloNet.load(fn=self.update_log, inputs=[], outputs=log_output, every=self.log_window_update_interval)

        # helper.print_output_logger.clear_output()
        miloNet.launch(share=True)
