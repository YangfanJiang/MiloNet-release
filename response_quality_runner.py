import argparse
import json
import os
import tempfile

from helper import ResponseQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResponseQualityEvaluator with custom inputs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file containing retrieval analysis entries.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the quality evaluation JSONL.",
    )
    parser.add_argument(
        "--original-field",
        default="response",
        help="Field name for the baseline/original response (default: response).",
    )
    parser.add_argument(
        "--comparison-field",
        default="milo_response",
        help="Field name for the system response to evaluate (default: milo_response).",
    )
    parser.add_argument(
        "--documents-field",
        default="documents",
        help="Field name for supporting documents (default: documents).",
    )
    parser.add_argument(
        "--original-label",
        default="original response",
        help="Label to display for the baseline response in the evaluator prompt/output.",
    )
    parser.add_argument(
        "--comparison-label",
        default="RAG system's response",
        help="Label to display for the system response in the evaluator prompt/output.",
    )
    return parser.parse_args()


def build_temp_input(
    source_path: str,
    temp_path: str,
    original_field: str,
    comparison_field: str,
    documents_field: str,
    original_label: str,
    comparison_label: str,
) -> int:
    count = 0
    with open(source_path, "r", encoding="utf-8") as fin, open(temp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            original_response = row.get(original_field)
            system_response = row.get(comparison_field)
            documents = row.get(documents_field) or []

            if not question or original_response is None or system_response is None:
                continue

            payload = {
                "question": question,
                "documents": documents,
                "response": original_response,
                "milo_response": system_response,
                "original_label": original_label,
                "comparison_label": comparison_label,
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    fd, temp_path = tempfile.mkstemp(suffix=".jsonl", prefix="quality_input_")
    os.close(fd)

    try:
        prepared = build_temp_input(
            source_path=args.input,
            temp_path=temp_path,
            original_field=args.original_field,
            comparison_field=args.comparison_field,
            documents_field=args.documents_field,
            original_label=args.original_label,
            comparison_label=args.comparison_label,
        )
        if not prepared:
            print("⚠️ No valid entries to evaluate.")
            return

        evaluator = ResponseQualityEvaluator(input_file=temp_path, output_file=args.output)
        evaluator.evaluate_response_quality()
        print(f"✅ Quality evaluation written to {args.output}")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
