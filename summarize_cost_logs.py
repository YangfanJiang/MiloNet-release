import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_NOTES = {
    "Vanilla RAG": "flat retrieval + generation",
    "RankGPT": "reranking + generation",
    "RAPTOR c4_k12": "hierarchy + generation",
    "MiloNet-core": "routing + hierarchical evidence construction",
    "MiloNet-full": "core + verification-heavy postprocessing",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_query_count(values: List[str]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected System=N for --query-count, got {value!r}")
        system, count_text = value.split("=", 1)
        parsed[system.strip()] = int(count_text)
    return parsed


def mean(values: List[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def aggregate(rows: List[Dict[str, Any]], query_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success") is False:
            continue
        system = row.get("system")
        if system:
            by_system[str(system)].append(row)

    output: List[Dict[str, Any]] = []
    for system in sorted(by_system.keys()):
        system_rows = by_system[system]
        case_ids = {str(row.get("case_id")) for row in system_rows if row.get("case_id") not in (None, "")}
        query_count = len(case_ids) if case_ids else query_counts.get(system)
        if not query_count:
            raise ValueError(
                f"Cannot infer query count for {system}. Provide --query-count '{system}=N'."
            )

        per_case: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "calls": 0.0,
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_seconds": 0.0,
        })
        if case_ids:
            for row in system_rows:
                case_id = str(row.get("case_id"))
                bucket = per_case[case_id]
                bucket["calls"] += 1
                bucket["prompt_tokens"] += float(row.get("prompt_tokens") or 0)
                bucket["completion_tokens"] += float(row.get("completion_tokens") or 0)
                bucket["total_tokens"] += float(row.get("total_tokens") or 0)
                bucket["latency_seconds"] += float(row.get("latency_seconds") or 0)
            calls_per_query = mean([bucket["calls"] for bucket in per_case.values()])
            prompt_per_query = mean([bucket["prompt_tokens"] for bucket in per_case.values()])
            completion_per_query = mean([bucket["completion_tokens"] for bucket in per_case.values()])
            total_per_query = mean([bucket["total_tokens"] for bucket in per_case.values()])
            latency_per_query = mean([bucket["latency_seconds"] for bucket in per_case.values()])
        else:
            calls_per_query = len(system_rows) / query_count
            prompt_per_query = sum(float(row.get("prompt_tokens") or 0) for row in system_rows) / query_count
            completion_per_query = sum(float(row.get("completion_tokens") or 0) for row in system_rows) / query_count
            total_per_query = sum(float(row.get("total_tokens") or 0) for row in system_rows) / query_count
            latency_per_query = sum(float(row.get("latency_seconds") or 0) for row in system_rows) / query_count

        estimated_count = sum(1 for row in system_rows if row.get("tokens_estimated"))
        token_source = "estimated" if estimated_count == len(system_rows) else "mixed" if estimated_count else "api_usage"
        note = DEFAULT_NOTES.get(system, "")
        if token_source != "api_usage":
            note = f"{note}; token counts {token_source}".strip("; ")

        output.append(
            {
                "system": system,
                "profiled_queries": query_count,
                "calls_per_query": calls_per_query,
                "prompt_tokens_per_query": prompt_per_query,
                "completion_tokens_per_query": completion_per_query,
                "total_tokens_per_query": total_per_query,
                "latency_seconds_per_query": latency_per_query,
                "notes": note,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-call cost logs into revision_cost_inputs.json.")
    parser.add_argument("--log", action="append", required=True, help="Cost JSONL log. Can be repeated.")
    parser.add_argument("--output", default="revision_cost_inputs.json")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip log paths that do not exist yet. Useful while MiloNet-core profiling is still pending.",
    )
    parser.add_argument(
        "--query-count",
        action="append",
        default=[],
        help="Fallback query denominator when log rows lack case_id, formatted as 'System=N'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for log_path in args.log:
        path = Path(log_path)
        if not path.exists() and args.allow_missing:
            print(f"Skipping missing log: {path}")
            continue
        rows.extend(read_jsonl(path))
    output = aggregate(rows, parse_query_count(args.query_count))
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
