"""Convert RAPTOR summary selections into sentence-key evaluation records."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_question(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return " ".join(text.split()).strip().lower()


def build_benchmark_lookup(path: Optional[Path]) -> Dict[str, dict]:
    if not path:
        return {}
    lookup: Dict[str, dict] = {}
    for row in load_jsonl(path):
        key = normalize_question(row.get("question"))
        if not key:
            continue
        lookup[key] = row
    return lookup


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def expand_sentence_ids(
    doc_indices: List[int],
    documents_sentences: List[List[List[str]]],
) -> List[str]:
    sentence_ids: List[str] = []
    for idx in doc_indices:
        if idx < 0 or idx >= len(documents_sentences):
            continue
        for sentence_pair in documents_sentences[idx]:
            if sentence_pair and sentence_pair[0]:
                sentence_ids.append(sentence_pair[0])
    return deduplicate_preserve_order(sentence_ids)


def prepare_entry(
    raptor_row: dict,
    benchmark_fallback: dict,
) -> dict:
    question = raptor_row.get("question") or benchmark_fallback.get("question")
    if not question:
        raise ValueError("Missing question in RAPTOR row; cannot prepare entry.")

    documents = raptor_row.get("documents") or benchmark_fallback.get("documents", [])
    documents_sentences = raptor_row.get("documents_sentences") or benchmark_fallback.get(
        "documents_sentences", []
    )
    relevant = raptor_row.get("all_relevant_sentence_keys") or benchmark_fallback.get(
        "all_relevant_sentence_keys", []
    )
    utilized = raptor_row.get("all_utilized_sentence_keys") or benchmark_fallback.get(
        "all_utilized_sentence_keys", []
    )

    clusters: List[dict] = raptor_row.get("selected_clusters") or []
    if not clusters:
        raise ValueError(f"No selected_clusters provided for question: {question}")

    contexts: List[str] = []
    expanded_sentence_ids: List[str] = []
    summary_ids: List[str] = []

    for cluster in clusters:
        summary_text = cluster.get("summary_text", "").strip()
        if summary_text:
            contexts.append(summary_text)
        summary_id = cluster.get("summary_id")
        if summary_id:
            summary_ids.append(summary_id)

        doc_indices = cluster.get("doc_indices") or []
        # Older RAPTOR outputs may store zero-based document indices as strings.
        if not doc_indices and "doc_ids" in cluster:
            try:
                doc_indices = [int(val) for val in cluster["doc_ids"]]
            except (TypeError, ValueError):
                doc_indices = []

        for idx in doc_indices:
            if 0 <= idx < len(documents):
                contexts.append(documents[idx])

        expanded_sentence_ids.extend(expand_sentence_ids(doc_indices, documents_sentences))

    expanded_sentence_ids = deduplicate_preserve_order(expanded_sentence_ids)
    answer = (
        raptor_row.get("raptor_response")
        or raptor_row.get("baseline_response")
        or raptor_row.get("answer")
        or raptor_row.get("response")
    )
    if answer is None:
        raise ValueError(f"No response found for question: {question}")

    return {
        "id": raptor_row.get("id"),
        "question": question,
        "documents": documents,
        "documents_sentences": documents_sentences,
        "all_relevant_sentence_keys": relevant,
        "all_utilized_sentence_keys": utilized,
        "retrieved_texts": contexts,
        "used_texts": contexts,
        "retrieved_contexts_keys": expanded_sentence_ids,
        "used_texts_keys": expanded_sentence_ids,
        "baseline_response": answer,
        "raptor_response": answer,
        "selected_summary_ids": summary_ids,
        "selected_clusters": clusters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build RAPTOR evaluation JSON by expanding selected documents "
            "to sentence IDs."
        )
    )
    parser.add_argument(
        "--raptor-output",
        required=True,
        help="Path to RAPTOR JSONL containing selected clusters and responses.",
    )
    parser.add_argument(
        "--benchmark-file",
        help="Optional benchmark JSONL used to fill missing document fields.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_data_raptor.json",
        help="Path to save evaluation JSON (default: evaluation_data_raptor.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raptor_rows = load_jsonl(Path(args.raptor_output))
    if not raptor_rows:
        raise SystemExit("No RAPTOR rows found; aborting.")

    benchmark_lookup = build_benchmark_lookup(Path(args.benchmark_file)) if args.benchmark_file else {}

    prepared_entries: List[dict] = []
    for row in raptor_rows:
        key = normalize_question(row.get("question"))
        fallback = benchmark_lookup.get(key, {})
        try:
            entry = prepare_entry(row, fallback)
        except ValueError as exc:
            print(f"Skipping row due to error: {exc}")
            continue
        prepared_entries.append(entry)

    if not prepared_entries:
        raise SystemExit("No valid entries prepared; nothing to write.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prepared_entries, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(prepared_entries)} entries: {args.output}")


if __name__ == "__main__":
    main()
