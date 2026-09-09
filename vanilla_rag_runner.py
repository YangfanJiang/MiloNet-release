import argparse
import json
import math
import os
import re
import statistics
import time
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

try:
    from enhanced_model_patch import comprehensive_patch

    comprehensive_patch()
except Exception as patch_error:  # pylint: disable=broad-except
    print(f"⚠️  Enhanced model patch not applied: {patch_error}")

import baseline_milo_score as baseline_metrics
from cost_logging import CostLogger


load_dotenv()


def read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_topk_map(analysis_path: str) -> Tuple[Dict[str, int], int]:
    question_topk: Dict[str, int] = {}
    counts: List[int] = []
    for row in read_jsonl(analysis_path):
        key = baseline_metrics.normalize_question_key(row.get("question"))
        used_texts = row.get("used_texts") or []
        value = max(1, len(used_texts))
        question_topk[key] = value
        counts.append(value)
    default_k = int(round(statistics.mean(counts))) if counts else 3
    default_k = max(1, default_k)
    return question_topk, default_k


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_text(question_tokens: set, text: str) -> float:
    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0
    overlap = sum(1 for token in text_tokens if token in question_tokens)
    return overlap / math.sqrt(len(text_tokens))


def flatten_documents_sentences(documents_sentences: Sequence[Sequence[Sequence[str]]]) -> List[Tuple[str, str]]:
    flattened: List[Tuple[str, str]] = []
    for doc in documents_sentences or []:
        for sentence_id, sentence_text in doc:
            flattened.append((sentence_id, sentence_text))
    return flattened


def select_top_k_sentences(
    question: str, documents_sentences: Sequence[Sequence[Sequence[str]]], top_k: int
) -> List[Tuple[str, str]]:
    flattened = flatten_documents_sentences(documents_sentences)
    if not flattened or top_k <= 0:
        return []
    question_tokens = set(tokenize(question))
    scored = [
        (score_text(question_tokens, sentence_text), idx, sentence_id, sentence_text)
        for idx, (sentence_id, sentence_text) in enumerate(flattened)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    limit = min(top_k, len(scored))
    return [(sentence_id, sentence_text) for _, _, sentence_id, sentence_text in scored[:limit]]


def select_top_k_documents(question: str, documents: List[str], top_k: int) -> List[str]:
    question_tokens = set(tokenize(question))
    scored = [
        (score_text(question_tokens, doc), idx, doc)
        for idx, doc in enumerate(documents)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    k = min(top_k, len(scored))
    return [doc for _, _, doc in scored[:k]] if k else []


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(
        f"[{idx + 1}] {ctx.strip()}" for idx, ctx in enumerate(contexts)
    )
    return (
        "You are a concise RAG assistant. Answer the question strictly using the"
        " provided contexts. If the contexts do not contain enough information, say so.\n\n"
        f"Question:\n{question.strip()}\n\nContexts:\n{context_block}\n\nAnswer:"
    )


def generate_response(
    llm: LlamaOpenAI,
    prompt: str,
    retries: int = 3,
    cost_logger: CostLogger | None = None,
    case_id: str | None = None,
) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            if cost_logger and cost_logger.enabled:
                completion = cost_logger.complete(llm, prompt, stage="answer_generation", case_id=case_id)
            else:
                completion = llm.complete(prompt=prompt)
            return completion.text.strip()
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            sleep_for = 2 ** attempt
            time.sleep(sleep_for)
    raise RuntimeError(f"LLM generation failed after {retries} attempts: {last_error}")


def save_json(data: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(data: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_ragas_merge(analysis_entries: List[dict], ragas_eval_data: List[dict]) -> None:
    if not ragas_eval_data:
        return

    ragas_dataset = Dataset.from_pandas(pd.DataFrame(ragas_eval_data))
    ragas_results = evaluate(
        ragas_dataset,
        metrics=[faithfulness],
        run_config=RunConfig(timeout=600, max_retries=5, max_workers=2),
    )
    ragas_df = ragas_results.to_pandas().rename(columns={"user_input": "question"})
    eval_df = pd.DataFrame(analysis_entries)
    final_eval_df = pd.merge(eval_df, ragas_df[["question", "faithfulness"]], on="question", how="left")
    for idx, row in final_eval_df.iterrows():
        analysis_entries[idx]["faithfulness"] = row["faithfulness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-equalized vanilla RAG runner.")
    parser.add_argument("--benchmark-file", default="ragbench_test.jsonl", help="Baseline benchmark (JSONL).")
    parser.add_argument(
        "--analysis-file",
        default="retrieval_analysis_data_121125.jsonl",
        help="MiloNet analysis (JSONL) used to compute evidence counts.",
    )
    parser.add_argument(
        "--evaluation-output",
        default="evaluation_data_vanilla.json",
        help="Path to save vanilla RAG evaluation data (JSON).",
    )
    parser.add_argument(
        "--metrics-output",
        default="retrieval_analysis_data_vanilla.jsonl",
        help="Path to save computed metrics JSONL.",
    )
    parser.add_argument(
        "--model",
        default="o4-mini",
        help="OpenAI-compatible model identifier (default: o4-mini for final response generation).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default 0 to match MiloNet).",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Optional limit on number of benchmark questions to process.",
    )
    parser.add_argument(
        "--cost-log",
        default=None,
        help="Optional JSONL file for per-call cost logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_entries = read_jsonl(args.benchmark_file)
    if args.question_limit:
        benchmark_entries = benchmark_entries[: args.question_limit]

    topk_map, default_k = build_topk_map(args.analysis_file)

    # gpt-4.1 models reject the old max_tokens param; leave it unset to avoid 400 errors.
    llm = LlamaOpenAI(model=args.model, temperature=args.temperature, max_tokens=None)
    cost_logger = CostLogger(args.cost_log, "Vanilla RAG", args.model)

    evaluation_data: List[dict] = []
    for entry in benchmark_entries:
        question = entry.get("question", "")
        normalized_question = baseline_metrics.normalize_question_key(question)
        documents = entry.get("documents") or []
        documents_sentences = entry.get("documents_sentences") or []
        top_k = topk_map.get(normalized_question, default_k)
        sentence_choices = select_top_k_sentences(question, documents_sentences, top_k)
        sentence_ids = [sentence_id for sentence_id, _ in sentence_choices]
        if sentence_choices:
            contexts = [text for _, text in sentence_choices]
        else:
            contexts = select_top_k_documents(question, documents, top_k)
            if not contexts and documents:
                contexts = documents[: top_k]
            sentence_ids = []

        prompt = build_prompt(question, contexts)
        answer = generate_response(
            llm,
            prompt,
            cost_logger=cost_logger,
            case_id=str(entry.get("id") or normalized_question),
        )

        evaluation_data.append(
            {
                "id": entry.get("id"),
                "question": question,
                "documents": documents,
                "documents_sentences": documents_sentences,
                "all_relevant_sentence_keys": entry.get("all_relevant_sentence_keys"),
                "all_utilized_sentence_keys": entry.get("all_utilized_sentence_keys"),
                "retrieved_texts": contexts,
                "vanilla_retrieved_texts": contexts,
                "used_texts": contexts,
                "vanilla_used_texts": contexts,
                "vanilla_retrieved_sentence_ids": sentence_ids,
                "vanilla_used_sentence_ids": sentence_ids,
                "vanilla_response": answer,
                "response": entry.get("response"),
            }
        )

    save_json(evaluation_data, args.evaluation_output)
    print(f"✅ Generated {len(evaluation_data)} vanilla RAG answers → {args.evaluation_output}")

    benchmark_lookup = baseline_metrics.create_benchmark_lookup(args.benchmark_file)
    processed_count, analysis_entries, ragas_eval_data = baseline_metrics.evaluate_baseline_entries(
        evaluation_data,
        benchmark_lookup,
        method_label="vanilla",
    )

    if not analysis_entries:
        print("⚠️ No valid entries for metrics.")
        return

    run_ragas_merge(analysis_entries, ragas_eval_data)

    vanilla_answer_map = {
        baseline_metrics.normalize_question_key(item["question"]): item.get("vanilla_response", item.get("milo_response"))
        for item in evaluation_data
    }
    for entry in analysis_entries:
        key = baseline_metrics.normalize_question_key(entry["question"])
        if key in vanilla_answer_map:
            entry["vanilla_response"] = vanilla_answer_map[key]
        elif "vanilla_response" not in entry and "baseline_response" in entry:
            entry["vanilla_response"] = entry["baseline_response"]

    write_jsonl(analysis_entries, args.metrics_output)
    print(
        f"✅ Metrics computed for {processed_count} entries → {args.metrics_output}"
    )


if __name__ == "__main__":
    main()
