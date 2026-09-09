# --- BASELINE SCORING VERSION ---
import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig


load_dotenv()


def normalize_question_key(text: Any) -> str:
    """
    Normalize a question string to a canonical format so we can perform reliable lookups.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def deduplicate_preserve_order(values: Optional[List[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values or []:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def get_ids_from_structured_contexts(structured_contexts: List[List[List[str]]]) -> set[str]:
    """Extract all sentence IDs from a structured context list."""
    return {sentence_id for doc in structured_contexts for sentence_id, _ in doc}


def get_texts_from_structured_contexts(structured_contexts: List[List[List[str]]]) -> List[str]:
    """Extract the sentence texts from a structured context list for RAGAS contexts."""
    return [sentence_text for doc in structured_contexts for _, sentence_text in doc]


def restructure_contexts_like_documents_sentences(
    context_list: List[str],
    documents_sentences: List[List[List[str]]],
) -> List[List[List[str]]]:
    """
    Restructure arbitrary context blocks into the same nested shape as documents_sentences.
    """
    all_sentences_map: Dict[str, str] = {}
    sentence_id_to_doc_id_map: Dict[str, str] = {}
    for doc_group in documents_sentences or []:
        for sentence_id, sentence_text in doc_group:
            all_sentences_map[sentence_id] = sentence_text
            doc_id_match = re.match(r"(\d+)", sentence_id)
            if doc_id_match:
                sentence_id_to_doc_id_map[sentence_id] = doc_id_match.group(1)

    found_sentences_by_doc: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for context_block in context_list:
        if not context_block:
            continue
        for sentence_id, sentence_text in all_sentences_map.items():
            if sentence_text and sentence_text in context_block:
                doc_id = sentence_id_to_doc_id_map.get(sentence_id)
                if doc_id:
                    found_sentences_by_doc[doc_id][sentence_id] = [sentence_id, sentence_text]

    final_structured_list: List[List[List[str]]] = []
    for doc_id in sorted(found_sentences_by_doc.keys()):
        sentences_in_doc = list(found_sentences_by_doc[doc_id].values())
        if sentences_in_doc:
            final_structured_list.append(sentences_in_doc)
    return final_structured_list


def sentence_lookup_from_documents_sentences(
    documents_sentences: List[List[List[str]]],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for doc in documents_sentences or []:
        for sentence_id, sentence_text in doc:
            lookup[sentence_id] = sentence_text
    return lookup


def sentence_texts_from_ids(sentence_ids: List[str], sentence_lookup: Dict[str, str]) -> List[str]:
    return [sentence_lookup[sentence_id] for sentence_id in sentence_ids if sentence_id in sentence_lookup]


def collect_used_sentence_ids(entry: Dict[str, Any], fallback: List[str]) -> List[str]:
    support_info = entry.get("sentence_support_information") or []
    ordered_ids: List[str] = []
    seen = set()
    for support in support_info:
        for key in support.get("supporting_sentence_keys") or []:
            if key and key not in seen:
                seen.add(key)
                ordered_ids.append(key)
    if ordered_ids:
        return ordered_ids
    return deduplicate_preserve_order(fallback)


def load_user_dataset(file_path: str) -> List[Dict[str, Any]]:
    _, ext = os.path.splitext(file_path.lower())
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data

        try:
            payload = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data

    if isinstance(payload, dict):
        return [payload]
    return list(payload or [])


def create_benchmark_lookup(benchmark_file_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load a benchmark file (JSONL) into a lookup dictionary keyed by normalized question.
    """
    if not benchmark_file_path:
        print("ℹ️  no benchmark file provided; relying on inline fields only.")
        return {}

    print(f"building lookup dictionary from {benchmark_file_path}...")
    lookup_table: Dict[str, Dict[str, Any]] = {}
    try:
        with open(benchmark_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                required_keys = [
                    "question",
                    "documents_sentences",
                    "all_relevant_sentence_keys",
                    "all_utilized_sentence_keys",
                    "response",
                    "documents",
                ]
                if all(key in data for key in required_keys):
                    normalized_key = normalize_question_key(data["question"])
                    lookup_table[normalized_key] = {
                        "documents_sentences": data["documents_sentences"],
                        "documents": data["documents"],
                        "all_relevant_sentence_keys": data["all_relevant_sentence_keys"],
                        "all_utilized_sentence_keys": data["all_utilized_sentence_keys"],
                        "response": data["response"],
                    }
    except FileNotFoundError:
        print(f"error: could not find benchmark file {benchmark_file_path}")
        return {}
    print(f"✅ lookup dictionary built, containing {len(lookup_table)} unique questions.")
    return lookup_table


def evaluate_baseline_entries(
    user_data: List[Dict[str, Any]],
    benchmark_lookup: Dict[str, Dict[str, Any]],
    method_label: Optional[str] = None,
) -> tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    processed_count = 0
    all_analysis_entries: List[Dict[str, Any]] = []
    ragas_eval_data: List[Dict[str, Any]] = []
    label_prefix = method_label.lower() if method_label else None

    for entry in user_data:
        question = entry.get("question")
        if not question:
            continue
        normalized_question = normalize_question_key(question)
        # Baseline answer text can come from pipelines with different field names:
        # - For adapter/runner baselines (RAPTOR, RankGPT, Vanilla, Milo-core/full):
        #   prefer the method-specific "<label_prefix>_response" when available
        #   (e.g., "raptor_response", "vanilla_response", "milo-core_response").
        # - For generic baselines (Oracle / ragbench, early experiments):
        #   fall back to "response" / "baseline_response".
        # - For older MiloNet evaluation_data files:
        #   accept "milo_response" as a final fallback.
        baseline_response = None
        if label_prefix:
            baseline_response = entry.get(f"{label_prefix}_response")
        if not baseline_response:
            baseline_response = entry.get("response") or entry.get("baseline_response")
        if not baseline_response:
            baseline_response = entry.get("milo_response")

        benchmark_entry = benchmark_lookup.get(normalized_question, {})
        documents_sentences = entry.get("documents_sentences") or benchmark_entry.get("documents_sentences")
        documents = entry.get("documents") or benchmark_entry.get("documents")
        response_ground_truth = benchmark_entry.get("response") or entry.get("response")
        all_relevant_sentence_keys = deduplicate_preserve_order(
            entry.get("all_relevant_sentence_keys") or benchmark_entry.get("all_relevant_sentence_keys")
        )
        all_utilized_sentence_keys = deduplicate_preserve_order(
            entry.get("all_utilized_sentence_keys") or benchmark_entry.get("all_utilized_sentence_keys")
        )

        if not (documents_sentences and documents and baseline_response):
            continue

        sentence_lookup = sentence_lookup_from_documents_sentences(documents_sentences)

        # SPECIAL CASE: Oracle (ragbench) early return - use gold keys directly
        # Oracle should have perfect retrieval recall and generator recall by definition.
        is_oracle = (label_prefix == "ragbench")
        
        if is_oracle:
            # Oracle: use gold keys directly, map to texts for consistency
            # Handle empty sets gracefully (will result in rr=0 or gr=0 as expected)
            all_retrieved_ids = deduplicate_preserve_order(all_relevant_sentence_keys or [])
            retrieved_texts_flat = sentence_texts_from_ids(all_retrieved_ids, sentence_lookup)
            
            used_ids_list = deduplicate_preserve_order(all_utilized_sentence_keys or [])
            used_set = set(used_ids_list)
            used_texts_for_ragas = sentence_texts_from_ids(used_ids_list, sentence_lookup)
            
            # Calculate metrics
            relevant_set = set(all_relevant_sentence_keys or [])
            retrieved_intersection = relevant_set.intersection(all_retrieved_ids)
            retrieval_recall = len(retrieved_intersection) / len(relevant_set) if relevant_set else 0.0
            
            utilized_gold_set = set(all_utilized_sentence_keys or [])
            generator_intersection = utilized_gold_set.intersection(used_set)
            generator_precision = len(generator_intersection) / len(used_set) if used_set else 0.0
            generator_recall = len(generator_intersection) / len(utilized_gold_set) if utilized_gold_set else 0.0
            
            # Build analysis entry
            analysis_entry = {
                "id": entry.get("id"),
                "question": question,
                "documents": documents,
                "baseline_response": baseline_response,
                "retrieved_contexts_keys": all_retrieved_ids,
                "used_texts_keys": used_ids_list,
                "used_texts": used_texts_for_ragas,
                "response": response_ground_truth,
                "all_relevant_sentence_keys": all_relevant_sentence_keys,
                "all_utilized_sentence_keys": all_utilized_sentence_keys,
                "retrieval_recall": retrieval_recall,
                "generator_precision": generator_precision,
                "generator_recall": generator_recall,
            }
            
            if label_prefix:
                analysis_entry[f"{label_prefix}_response"] = baseline_response
                analysis_entry[f"{label_prefix}_retrieved_contexts_keys"] = all_retrieved_ids
                analysis_entry[f"{label_prefix}_retrieved_texts"] = retrieved_texts_flat
                analysis_entry[f"{label_prefix}_used_texts_keys"] = used_ids_list
                analysis_entry[f"{label_prefix}_used_texts"] = used_texts_for_ragas
                analysis_entry[f"{label_prefix}_retrieval_recall"] = retrieval_recall
                analysis_entry[f"{label_prefix}_generator_precision"] = generator_precision
                analysis_entry[f"{label_prefix}_generator_recall"] = generator_recall
            
            all_analysis_entries.append(analysis_entry)
            
            # For the RAGBench reference baseline, evaluate faithfulness against the full D_q.
            ragas_contexts = documents if documents else used_texts_for_ragas

            ragas_eval_data.append(
                {
                    "question": question,
                    "answer": baseline_response,
                    "contexts": ragas_contexts,
                    "ground_truth": response_ground_truth,
                }
            )
            processed_count += 1
            continue

        # Prefer prefixed retrieved_texts when available, otherwise fall back to base field.
        label_retrieved_texts = entry.get(f"{label_prefix}_retrieved_texts") if label_prefix else None
        base_retrieved_texts = entry.get("retrieved_texts")

        # 1) Prefer existing retrieved_contexts_keys (generic or prefixed) to avoid
        #    recomputing via substring matching when the system already logged them.
        retrieved_contexts_keys = entry.get("retrieved_contexts_keys")
        if retrieved_contexts_keys is None and label_prefix:
            retrieved_contexts_keys = entry.get(f"{label_prefix}_retrieved_contexts_keys")

        # 2) Then look for any explicitly logged sentence IDs (keep order, de-dup).
        raw_label_ids = entry.get(f"{label_prefix}_retrieved_sentence_ids") if label_prefix else None
        label_retrieved_ids = deduplicate_preserve_order(raw_label_ids or [])

        # Detect text blocks available for fallback matching.
        has_text_blocks = bool(label_retrieved_texts or base_retrieved_texts)
        has_keys = bool(retrieved_contexts_keys) if retrieved_contexts_keys is not None else False

        if retrieved_contexts_keys is not None and has_keys:
            # Use the provided keys directly, preserving order and removing duplicates.
            all_retrieved_ids = deduplicate_preserve_order(retrieved_contexts_keys)
            retrieved_texts_flat = (
                label_retrieved_texts
                or base_retrieved_texts
                or sentence_texts_from_ids(all_retrieved_ids, sentence_lookup)
            )
        elif retrieved_contexts_keys is not None and not has_keys and has_text_blocks:
            # Empty key lists are treated as missing when text blocks are available;
            # fall back to substring matching to recover them.
            context_list = label_retrieved_texts or base_retrieved_texts or []
            restructured_contexts = restructure_contexts_like_documents_sentences(
                context_list, documents_sentences
            )
            ids_set = get_ids_from_structured_contexts(restructured_contexts)
            
            # Order IDs by documents_sentences sequence for stable, reproducible output
            ordered_ids = []
            seen = set()
            for doc_group in documents_sentences:
                for pair in doc_group:
                    if not pair or not pair[0]:
                        continue
                    sent_id = pair[0]
                    if sent_id in ids_set and sent_id not in seen:
                        ordered_ids.append(sent_id)
                        seen.add(sent_id)
            
            all_retrieved_ids = ordered_ids
            retrieved_texts_flat = get_texts_from_structured_contexts(restructured_contexts)
        elif label_retrieved_ids:
            # Use explicit sentence IDs when context keys are absent.
            all_retrieved_ids = label_retrieved_ids
            retrieved_texts_flat = (
                label_retrieved_texts
                or base_retrieved_texts
                or sentence_texts_from_ids(all_retrieved_ids, sentence_lookup)
            )
        else:
            # 3) Only as a last resort, fall back to substring matching from raw texts.
            # Do NOT use documents as fallback - it would incorrectly inflate rr/rp.
            context_list = label_retrieved_texts or base_retrieved_texts or []
            restructured_contexts = restructure_contexts_like_documents_sentences(
                context_list, documents_sentences
            )
            ids_set = get_ids_from_structured_contexts(restructured_contexts)
            
            # Order IDs by documents_sentences sequence for stable, reproducible output
            ordered_ids = []
            seen = set()
            for doc_group in documents_sentences:
                for pair in doc_group:
                    if not pair or not pair[0]:
                        continue
                    sent_id = pair[0]
                    if sent_id in ids_set and sent_id not in seen:
                        ordered_ids.append(sent_id)
                        seen.add(sent_id)
            
            all_retrieved_ids = ordered_ids
            retrieved_texts_flat = get_texts_from_structured_contexts(restructured_contexts)

        label_used_ids_raw = entry.get(f"{label_prefix}_used_sentence_ids") if label_prefix else None
        # RankGPT and RAPTOR store selected sentence IDs in the generic field.
        generic_used_ids_raw = entry.get("used_texts_keys")
        
        label_used_texts = entry.get(f"{label_prefix}_used_texts") if label_prefix else None
        base_used_texts = entry.get("used_texts")
        
        # For MiloNet: prefer original document context blocks (used_texts) over sentence-level texts,
        # to match paper definition: "match benchmark sentences in retrieved document context blocks"
        # Detect document context blocks, which typically exceed 200 characters.
        is_document_context_blocks = False
        candidate_used_texts = None
        
        if label_used_texts and isinstance(label_used_texts, list) and len(label_used_texts) > 0:
            # Check if these are document context blocks vs sentence-level texts
            sample_size = min(3, len(label_used_texts))
            avg_length = sum(len(str(t)) for t in label_used_texts[:sample_size]) / sample_size
            if avg_length > 200:  # Likely document context blocks
                is_document_context_blocks = True
                candidate_used_texts = label_used_texts
        elif base_used_texts and isinstance(base_used_texts, list) and len(base_used_texts) > 0:
            sample_size = min(3, len(base_used_texts))
            avg_length = sum(len(str(t)) for t in base_used_texts[:sample_size]) / sample_size
            if avg_length > 200:  # Likely document context blocks
                is_document_context_blocks = True
                candidate_used_texts = base_used_texts
        
        if is_document_context_blocks and candidate_used_texts:
            # Use document context blocks directly for RAGAS (matches paper definition)
            used_texts_for_ragas = candidate_used_texts
            # Still need to compute used_ids_list for metrics (rr, gr, etc.)
            # For MiloNet: extract actual used sentence keys from document context blocks via substring matching
            if label_used_ids_raw is not None:
                used_ids_list = deduplicate_preserve_order(label_used_ids_raw)
                used_set = set(used_ids_list)
            else:
                # Extract sentence keys from document context blocks via substring matching
                # This matches the paper definition: "match benchmark sentences in retrieved document context blocks"
                used_texts_structured = restructure_contexts_like_documents_sentences(
                    candidate_used_texts, documents_sentences
                )
                structured_ids = get_ids_from_structured_contexts(used_texts_structured)
                # Also check if there's sentence_support_information as a fallback
                support_ids = collect_used_sentence_ids(entry, [])
                if support_ids:
                    # Union with support_ids if available, but prioritize substring matching results
                    used_set = structured_ids.union(set(support_ids))
                else:
                    used_set = structured_ids
                used_ids_list = sorted(used_set)
        elif label_used_ids_raw is not None:
            # Keep original list order for serialization, convert to set only for calculations
            used_ids_list = deduplicate_preserve_order(label_used_ids_raw)
            used_set = set(used_ids_list)
            used_texts_for_ragas = label_used_texts or sentence_texts_from_ids(used_ids_list, sentence_lookup)
        elif generic_used_ids_raw is not None:
            # Fallback to generic used_texts_keys (RankGPT/RAPTOR)
            used_ids_list = deduplicate_preserve_order(generic_used_ids_raw)
            used_set = set(used_ids_list)
            used_texts_for_ragas = label_used_texts or base_used_texts or sentence_texts_from_ids(used_ids_list, sentence_lookup)
        else:
            used_sentence_ids = collect_used_sentence_ids(entry, all_utilized_sentence_keys)
            used_sentence_texts = sentence_texts_from_ids(used_sentence_ids, sentence_lookup)
            used_texts_structured = (
                restructure_contexts_like_documents_sentences(used_sentence_texts, documents_sentences)
                if used_sentence_texts
                else []
            )
            # Preserve explicitly selected IDs if text matching misses them.
            structured_ids = get_ids_from_structured_contexts(used_texts_structured)
            used_set = structured_ids.union(set(used_sentence_ids))
            
            used_texts_for_ragas = (
                get_texts_from_structured_contexts(used_texts_structured) or used_sentence_texts or []
            )
            
            # Convert set to sorted list for JSON serialization (stable, reproducible order)
            used_ids_list = sorted(used_set)

        relevant_set = set(all_relevant_sentence_keys)
        retrieved_intersection = relevant_set.intersection(all_retrieved_ids)
        retrieval_recall = len(retrieved_intersection) / len(relevant_set) if relevant_set else 0.0

        utilized_gold_set = set(all_utilized_sentence_keys)
        generator_intersection = utilized_gold_set.intersection(used_set)
        generator_precision = len(generator_intersection) / len(used_set) if used_set else 0.0
        generator_recall = len(generator_intersection) / len(utilized_gold_set) if utilized_gold_set else 0.0

        analysis_entry = {
            "id": entry.get("id"),
            "question": question,
            "documents": documents,
            "baseline_response": baseline_response,
            "retrieved_contexts_keys": all_retrieved_ids,
            "used_texts_keys": used_ids_list,
            "used_texts": used_texts_for_ragas,
            "response": response_ground_truth,
            "all_relevant_sentence_keys": all_relevant_sentence_keys,
            "all_utilized_sentence_keys": all_utilized_sentence_keys,
            "retrieval_recall": retrieval_recall,
            "generator_precision": generator_precision,
            "generator_recall": generator_recall,
        }

        if label_prefix:
            analysis_entry[f"{label_prefix}_response"] = baseline_response
            analysis_entry[f"{label_prefix}_retrieved_contexts_keys"] = all_retrieved_ids
            analysis_entry[f"{label_prefix}_retrieved_texts"] = retrieved_texts_flat
            analysis_entry[f"{label_prefix}_used_texts_keys"] = used_ids_list
            analysis_entry[f"{label_prefix}_used_texts"] = used_texts_for_ragas
            analysis_entry[f"{label_prefix}_retrieval_recall"] = retrieval_recall
            analysis_entry[f"{label_prefix}_generator_precision"] = generator_precision
            analysis_entry[f"{label_prefix}_generator_recall"] = generator_recall

        all_analysis_entries.append(analysis_entry)
        ragas_eval_data.append(
            {
                "question": question,
                "answer": baseline_response,
                "contexts": used_texts_for_ragas,
                "ground_truth": response_ground_truth,
            }
        )
        processed_count += 1

    return processed_count, all_analysis_entries, ragas_eval_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score baseline responses with Milo metrics.")
    parser.add_argument(
        "--baseline-file",
        "-i",
        default="ragbench_test.jsonl",
        help="Path to the baseline JSON/JSONL file (default: ragbench_test.jsonl).",
    )
    parser.add_argument(
        "--benchmark-file",
        "-b",
        default=None,
        help="Optional benchmark JSONL file used to fill missing metadata.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="baseline_metrics.jsonl",
        help="Path for the output JSONL metrics file (default: baseline_metrics.jsonl).",
    )
    parser.add_argument(
        "--method-label",
        default=None,
        help="Optional label (e.g., 'vanilla', 'milo') to namespace response/context fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        user_data = load_user_dataset(args.baseline_file)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"❌ error: could not load or parse your data file ({args.baseline_file}): {exc}")
        return

    if not user_data:
        print(f"⚠️  warning: no entries found in {args.baseline_file}")
        return

    benchmark_lookup = create_benchmark_lookup(args.benchmark_file)
    processed_count, analysis_entries, ragas_eval_data = evaluate_baseline_entries(
        user_data,
        benchmark_lookup,
        method_label=args.method_label,
    )

    if not analysis_entries:
        print("⚠️  no valid entries to score.")
        return

    label_prefix = args.method_label.lower() if args.method_label else None
    ragas_run_config = RunConfig(timeout=600, max_retries=5, max_workers=2)
    if ragas_eval_data:
        ragas_dataset = Dataset.from_pandas(pd.DataFrame(ragas_eval_data))
        ragas_results = evaluate(
            ragas_dataset,
            metrics=[faithfulness],
            run_config=ragas_run_config,
        )
        ragas_df = ragas_results.to_pandas().rename(columns={"user_input": "question"})
        eval_df = pd.DataFrame(analysis_entries)
        final_eval_df = pd.merge(eval_df, ragas_df[["question", "faithfulness"]], on="question", how="left")
        for idx, row in final_eval_df.iterrows():
            analysis_entries[idx]["faithfulness"] = row["faithfulness"]
            if label_prefix:
                analysis_entries[idx][f"{label_prefix}_faithfulness"] = row["faithfulness"]

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for analysis_entry in analysis_entries:
            f_out.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")

    print(f"\n✅ baseline scoring completed, {processed_count} entries written to {args.output_file}")


if __name__ == "__main__":
    main()
