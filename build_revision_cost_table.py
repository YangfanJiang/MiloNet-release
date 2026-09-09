import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_NUMERIC_FIELDS = [
    "calls_per_query",
    "prompt_tokens_per_query",
    "completion_tokens_per_query",
    "latency_seconds_per_query",
]


def fmt(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Cost input must be a JSON list of system rows.")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Cost row {index} is not an object.")
        if not row.get("system"):
            raise ValueError(f"Cost row {index} is missing 'system'.")
        rows.append(row)
    return rows


def validate_rows(rows: List[Dict[str, Any]], allow_missing: bool) -> None:
    missing = []
    for row in rows:
        if row.get("total_tokens_per_query") is None:
            prompt = row.get("prompt_tokens_per_query")
            completion = row.get("completion_tokens_per_query")
            if prompt is not None and completion is not None:
                row["total_tokens_per_query"] = float(prompt) + float(completion)
        for field in REQUIRED_NUMERIC_FIELDS:
            if row.get(field) is None:
                missing.append(f"{row['system']}:{field}")
    if missing and not allow_missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Missing cost values. Fill revision_cost_inputs.json or rerun with "
            f"--allow-missing for a draft table. Missing: {joined}"
        )


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("system", "System"),
        ("profiled_queries", "Profiled queries"),
        ("calls_per_query", "Calls/query"),
        ("prompt_tokens_per_query", "Prompt tokens/query"),
        ("completion_tokens_per_query", "Completion tokens/query"),
        ("total_tokens_per_query", "Total tokens/query"),
        ("latency_seconds_per_query", "Latency/query (s)"),
        ("notes", "Notes"),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header for _, header in columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(key)) for key, _ in columns) + " |\n")
        f.write(
            "\nThis table reports inference-time cost for the system run, not "
            "training cost or one-time corpus indexing/preprocessing cost.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the revision cost table from a small JSON manifest.")
    parser.add_argument("--input", default="revision_cost_inputs.json")
    parser.add_argument("--output", default="revision_cost_table.md")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Emit TBD placeholders instead of failing on missing values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    validate_rows(rows, args.allow_missing)
    write_markdown(rows, Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
