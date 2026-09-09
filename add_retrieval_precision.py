"""Add retrieval_precision fields to retrieval analysis JSONL files."""

import argparse
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject retrieval_precision into retrieval_analysis_data JSONL files."
    )
    parser.add_argument(
        "input",
        help="Path to the retrieval_analysis_data_*.jsonl file to update.",
    )
    parser.add_argument(
        "--prefix",
        help="Metric prefix, e.g. 'milo-core'. If omitted, tries to auto-detect per line.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to updating the input file in-place.",
    )
    parser.add_argument(
        "--no-plain",
        dest="keep_plain",
        action="store_false",
        help="Skip writing the plain 'retrieval_precision' alongside the prefixed field.",
    )
    parser.set_defaults(keep_plain=True)
    return parser.parse_args()


def detect_retrieved_key(data: Dict, prefix: Optional[str]) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Return (prefix, retrieved_keys) for this row.
    If prefix is provided, look up that specific prefixed key; otherwise auto-detect.
    """
    if prefix:
        key = f"{prefix}_retrieved_contexts_keys"
        return prefix, data.get(key)

    # Prefer explicit prefixes if present
    for key in data:
        if key.endswith("_retrieved_contexts_keys") and isinstance(data[key], list):
            detected_prefix = key[: -len("_retrieved_contexts_keys")]
            return detected_prefix, data[key]

    return None, data.get("retrieved_contexts_keys")


def compute_precision(retrieved: List[str], relevant: List[str]) -> float:
    retrieved_set = set(retrieved or [])
    if not retrieved_set:
        return 0.0
    relevant_set = set(relevant or [])
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output or args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    temp_fd, temp_path = tempfile.mkstemp(suffix=".jsonl", prefix="retrieval_precision_")
    os.close(temp_fd)

    updated, skipped = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin, open(temp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                fout.write(line)
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                fout.write(line)
                skipped += 1
                continue

            prefix, retrieved_keys = detect_retrieved_key(row, args.prefix)
            relevant_keys = row.get("all_relevant_sentence_keys")

            if retrieved_keys is None or relevant_keys is None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            precision = compute_precision(retrieved_keys, relevant_keys)

            key_name = "retrieval_precision"
            if prefix:
                key_name = f"{prefix}_retrieval_precision"
            row[key_name] = precision
            if args.keep_plain:
                row["retrieval_precision"] = precision
            elif "retrieval_precision" in row:
                row.pop("retrieval_precision")

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            updated += 1

    shutil.move(temp_path, output_path)

    print(f"✅ Updated {updated} rows with retrieval_precision; skipped {skipped} rows.")
    print(f"📄 Output written to {output_path}")


if __name__ == "__main__":
    main()
