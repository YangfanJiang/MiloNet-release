"""Compare Milo-core and Vanilla Correctness scores per question."""

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

EPS = 1e-8


def normalize_question(text: Optional[str]) -> Optional[str]:
    """Normalize question text for matching (trim, collapse whitespace, unify quotes, lowercase)."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return cleaned.lower()


def load_scores(path, prefix):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            question_key = normalize_question(question)
            if not question_key:
                continue
            retrieval_precision = row.get(f"{prefix}_retrieval_precision") or row.get("retrieval_precision")
            retrieval_recall = row.get(f"{prefix}_retrieval_recall") or row.get("retrieval_recall")
            gen_precision = row.get(f"{prefix}_generator_precision") or row.get("generator_precision")
            gen_recall = row.get(f"{prefix}_generator_recall") or row.get("generator_recall")
            faithfulness = row.get(f"{prefix}_faithfulness") or row.get("faithfulness")
            trustworthiness = row.get(f"{prefix}_trustworthiness") or row.get("trustworthiness")

            rp = retrieval_precision or 0.0
            rr = retrieval_recall or 0.0
            gp = gen_precision or 0.0
            gr = gen_recall or 0.0
            f = faithfulness or 0.0

            r_f1 = (2 * rp * rr) / (rp + rr + EPS)
            g_f1 = (2 * gp * gr) / (gp + gr + EPS)
            correctness = (r_f1 + g_f1 + f) / 3.0
            trustworthiness = ( rr+ gr + f) / 3.0

            data[question_key] = {
                "retrieval_precision": rp,
                "retrieval_recall": rr,
                "retrieval_f1": r_f1,
                "generator_precision": gp,
                "generator_recall": gr,
                "generator_f1": g_f1,
                "faithfulness": f,
                "correctness": correctness,
                "trustworthiness": trustworthiness,
                "response": row.get(f"{prefix}_response") or row.get("response"),
            }
    return data


def load_quality_scores(path, target_label=None):
    scores = {}
    score_line_pattern = re.compile(r"^Score for\s+(.+?)\s+response:\s*([0-9.]+)", re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:  
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            question_key = normalize_question(question)
            if not question_key:
                continue
            quality_text = row.get("quality", "")
            label_to_score = {}
            for part in quality_text.splitlines():
                part = part.strip()
                match = score_line_pattern.match(part)
                if match:
                    label = match.group(1).strip().lower()
                    try:
                        value = float(match.group(2))
                        label_to_score[label] = value
                    except ValueError:
                        continue

            chosen_score = None
            if target_label:
                chosen_score = label_to_score.get(target_label.strip().lower())
            else:
                # Legacy fallback order
                for candidate in [
                    "milo",
                    "rag system's",
                    "rag system",
                    "vanilla",
                    "original",
                ]:
                    if candidate in label_to_score:
                        chosen_score = label_to_score[candidate]
                        break
                if chosen_score is None and label_to_score:
                    # pick first available
                    chosen_score = next(iter(label_to_score.values()))

            if chosen_score is not None:
                scores[question_key] = chosen_score
    return scores


def summarize_quality(scores):
    if not scores:
        return {"count": 0, "mean": None, "ratio_lt3": None}
    mean = sum(scores) / len(scores)
    lt3 = sum(1 for s in scores if s < 3) / len(scores)
    return {"count": len(scores), "mean": mean, "ratio_lt3": lt3}


def compute_averages(scores_dict):
    if not scores_dict:
        return {}

    fields = ["retrieval_precision", "retrieval_recall", "retrieval_f1",
              "generator_precision", "generator_recall", "generator_f1",
              "faithfulness", "correctness", "trustworthiness"]
    totals = defaultdict(float)
    for question, metrics in scores_dict.items():
        for field in fields:
            totals[field] += metrics.get(field, 0.0)
    count = len(scores_dict)
    return {field: totals[field] / count for field in fields}


def main():
    parser = argparse.ArgumentParser(description="Compare two systems' correctness/trustworthiness/quality metrics")
    # General arguments
    parser.add_argument("--system-a", help="Path to retrieval metrics for system A")
    parser.add_argument("--system-b", help="Path to retrieval metrics for system B")
    parser.add_argument("--prefix-a", help="Prefix for system A metric keys")
    parser.add_argument("--prefix-b", help="Prefix for system B metric keys")
    parser.add_argument("--label-a", help="Display name for system A")
    parser.add_argument("--label-b", help="Display name for system B")
    parser.add_argument("--quality-a", help="Response quality JSON for system A")
    parser.add_argument("--quality-b", help="Response quality JSON for system B")
    parser.add_argument("--quality-label-a", help="Label inside quality text for system A (e.g., 'Milo' or 'RAG system's')")
    parser.add_argument("--quality-label-b", help="Label inside quality text for system B (e.g., 'Vanilla')")
    parser.add_argument("--summary-output", help="Optional path to save the summary JSON (default: auto-generated name)")

    # Backwards compatibility aliases
    parser.add_argument("--milo", help="(Alias for --system-a) Milo metrics file")
    parser.add_argument("--vanilla", help="(Alias for --system-b) Vanilla metrics file")
    parser.add_argument("--prefix-milo", help="(Alias for --prefix-a) Milo prefix")
    parser.add_argument("--prefix-vanilla", help="(Alias for --prefix-b) Vanilla prefix")
    parser.add_argument("--quality-milo", help="(Alias for --quality-a) Milo quality file")
    parser.add_argument("--quality-vanilla", help="(Alias for --quality-b) Vanilla quality file")
    args = parser.parse_args()

    system_a_path = args.system_a or args.milo
    system_b_path = args.system_b or args.vanilla
    if not system_a_path or not system_b_path:
        parser.error("You must provide --system-a/--milo and --system-b/--vanilla paths.")

    prefix_a = args.prefix_a or args.prefix_milo or "system_a"
    prefix_b = args.prefix_b or args.prefix_vanilla or "system_b"
    label_a = args.label_a or ("Milo" if args.milo and not args.label_a else "System A")
    label_b = args.label_b or ("Vanilla" if args.vanilla and not args.label_b else "System B")

    quality_a_path = args.quality_a or args.quality_milo
    quality_b_path = args.quality_b or args.quality_vanilla

    scores_a = load_scores(system_a_path, prefix_a)
    scores_b = load_scores(system_b_path, prefix_b)
    shared_questions = sorted(set(scores_a) & set(scores_b))
    print(f"Found {len(shared_questions)} shared questions")

    subset_a = {q: scores_a[q] for q in shared_questions}
    subset_b = {q: scores_b[q] for q in shared_questions}

    avg_a = compute_averages(subset_a)
    avg_b = compute_averages(subset_b)

    print("\nAverage Correctness components (Retrieval F1, Generation F1, Faithfulness F, Correctness, Trustworthiness):")
    print(f"{label_a}:", avg_a)
    print(f"{label_b}:", avg_b)

    # 0-FP design explicitly focuses on faithfulness, so highlight it separately.
    avg_f_a = avg_a.get("faithfulness")
    avg_f_b = avg_b.get("faithfulness")
    delta_f = (avg_f_a - avg_f_b) if (avg_f_a is not None and avg_f_b is not None) else None
    faithfulness_pairs = [
        (q, scores_a[q]["faithfulness"], scores_b[q]["faithfulness"])
        for q in shared_questions
    ]
    better_f = sum(1 for _, a_f, b_f in faithfulness_pairs if a_f > b_f)
    worse_f = sum(1 for _, a_f, b_f in faithfulness_pairs if a_f < b_f)

    print("\n0-FP design focus — Faithfulness (F):")
    print(f"  {label_a} average F: {avg_f_a:.4f}")
    print(f"  {label_b} average F: {avg_f_b:.4f}")
    if delta_f is not None:
        print(f"  ΔF ({label_a} - {label_b}): {delta_f:+.4f}")
    print(f"  Questions where {label_a} has higher F: {better_f}/{len(shared_questions)}")
    print(f"  Questions where {label_a} has lower F: {worse_f}/{len(shared_questions)}")

    avg_t_a = avg_a.get("trustworthiness")
    avg_t_b = avg_b.get("trustworthiness")
    delta_t = (avg_t_a - avg_t_b) if (avg_t_a is not None and avg_t_b is not None) else None
    print("\nTrustworthiness summary:")
    print(f"  {label_a} average T: {avg_t_a:.4f}" if avg_t_a is not None else f"  {label_a} average T: N/A")
    print(f"  {label_b} average T: {avg_t_b:.4f}" if avg_t_b is not None else f"  {label_b} average T: N/A")
    if delta_t is not None:
        print(f"  ΔT ({label_a} - {label_b}): {delta_t:+.4f}")

    quality_summary_a = None
    quality_summary_b = None
    if quality_a_path and quality_b_path:
        quality_a = load_quality_scores(quality_a_path, args.quality_label_a)
        quality_b = load_quality_scores(quality_b_path, args.quality_label_b)
        quality_scores_a = [quality_a[q] for q in shared_questions if q in quality_a]
        quality_scores_b = [quality_b[q] for q in shared_questions if q in quality_b]

        quality_summary_a = summarize_quality(quality_scores_a)
        quality_summary_b = summarize_quality(quality_scores_b)

        print("\nQuality score summary:")
        print(f"{label_a}:", quality_summary_a)
        print(f"{label_b}:", quality_summary_b)

    # Prepare JSON summary
    summary = {
        "shared_question_count": len(shared_questions),
        "system_a": {
            "label": label_a,
            "averages": avg_a,
            "quality": quality_summary_a,
        },
        "system_b": {
            "label": label_b,
            "averages": avg_b,
            "quality": quality_summary_b,
        },
        "faithfulness_focus": {
            "label_a_avg_f": avg_f_a,
            "label_b_avg_f": avg_f_b,
            "delta_f": delta_f,
            "higher_count_label_a": better_f,
            "lower_count_label_a": worse_f,
        },
        "trustworthiness_focus": {
            "label_a_avg_t": avg_t_a,
            "label_b_avg_t": avg_t_b,
            "delta_t": delta_t,
        },
    }

    date_str = datetime.now().strftime("%d%m%y")

    def sanitize_label(label):
        if not label:
            return "system"
        sanitized = re.sub(r"[^A-Za-z0-9]+", "-", label.strip()).strip("-")
        return sanitized or "system"

    file_name = args.summary_output or f"{sanitize_label(label_a)}_vs_{sanitize_label(label_b)}_overall_results_{date_str}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Saved summary to {file_name}")


if __name__ == "__main__":
    main()
