import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from openai_model_patch import patch_openai_model_support
from cost_logging import CostLogger


@dataclass
class RankGPTResult:
    answer: str
    selected_indices: List[int]
    retrieved_texts: List[str]
    retrieved_sentence_ids: List[str]


def _deduplicate(seq: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class RankGPTBaseline:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout: int = 600,
        max_retries: int = 3,
        sleep_between_retries: float = 2.0,
        cost_log: str | None = None,
    ):
        patch_openai_model_support()
        self.model = model
        llm_kwargs = {
            "model": model,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if model in {"o4-mini", "o4"}:
            llm_kwargs["temperature"] = 1.0
        self.llm = OpenAI(**llm_kwargs)
        self.sleep_between_retries = sleep_between_retries
        self.cost_logger = CostLogger(cost_log, "RankGPT", model)

    def _complete(self, prompt: str, *, stage: str, case_id: str | None = None) -> str:
        last_exc = None
        for attempt in range(self.llm.max_retries):
            try:
                if self.cost_logger.enabled:
                    response = self.cost_logger.complete(self.llm, prompt, stage=stage, case_id=case_id)
                else:
                    response = self.llm.complete(prompt=prompt)
                return response.text.strip()
            except Exception as exc:
                last_exc = exc
                time.sleep(self.sleep_between_retries * (attempt + 1))
        raise last_exc  # type: ignore

    def _build_rank_prompt(self, question: str, docs: List[str]) -> str:
        passages = [f"[{idx}] {doc.strip()}" for idx, doc in enumerate(docs)]
        passages_text = "\n\n".join(passages)
        return (
            "You are an intelligent relevance ranking agent.\n"
            f"Question: {question.strip()}\n\n"
            "Task:\n"
            "1. Analyze the candidate documents below. Some are irrelevant distractors.\n"
            "2. Select ONLY the documents that provide the necessary information to answer the question.\n"
            "3. Rank the selected documents in descending order of relevance.\n\n"
            f"Candidate Documents:\n{passages_text}\n\n"
            "Output the indices of the selected documents (e.g., [0, 2]).\n"
            "If no documents are relevant, output [0] as a fallback.\n\n"
            "Selected Indices:"
        )

    def _parse_indices(self, raw_text: str, doc_count: int) -> List[int]:
        nums = [int(n) for n in re.findall(r"\d+", raw_text)]
        filtered = [n for n in nums if 0 <= n < doc_count]
        unique = _deduplicate(filtered)
        return unique

    def _gather_sentence_ids(self, indices: List[int], doc_sentences: List[List[List[str]]]) -> List[str]:
        sentence_ids: List[str] = []
        for idx in indices:
            if idx >= len(doc_sentences):
                continue
            for sentence in doc_sentences[idx]:
                if sentence and sentence[0]:
                    sentence_ids.append(sentence[0])
        return _deduplicate(sentence_ids)

    def run_case(self, case: dict) -> RankGPTResult:
        question = case.get("question", "")
        case_id = str(case.get("id") or question)
        docs = case.get("documents") or []
        doc_sentences = case.get("documents_sentences") or []

        if not question or not docs:
            raise ValueError("Case is missing question/documents.")

        if len(docs) <= 1:
            selected_indices = [0]
        else:
            rank_prompt = self._build_rank_prompt(question, docs)
            response = self._complete(rank_prompt, stage="rerank", case_id=case_id)
            selected_indices = self._parse_indices(response, len(docs))
            if not selected_indices:
                selected_indices = [0]

        selected_texts = [docs[idx] for idx in selected_indices if idx < len(docs)]
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx.strip()}" for i, ctx in enumerate(selected_texts)
        )
        gen_prompt = (
            "You are a concise RAG assistant. Answer the question strictly using the"
            " provided contexts. If the contexts do not contain enough information, say so.\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Contexts:\n{context_block}\n\n"
            "Answer:"
        )
        answer = self._complete(gen_prompt, stage="answer_generation", case_id=case_id)
        sentence_ids = self._gather_sentence_ids(selected_indices, doc_sentences)

        return RankGPTResult(
            answer=answer,
            selected_indices=selected_indices,
            retrieved_texts=selected_texts,
            retrieved_sentence_ids=sentence_ids,
        )


def load_ragbench_cases(path: str) -> List[dict]:
    cases: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def write_evaluation_file(entries: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def build_evaluation_entry(case: dict, result: RankGPTResult) -> dict:
    sentence_ids = result.retrieved_sentence_ids
    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "baseline_response": result.answer,
        "retrieved_texts": result.retrieved_texts,
        "used_texts": result.retrieved_texts,
        "retrieved_contexts_keys": sentence_ids,
        "used_texts_keys": sentence_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RankGPT baseline over RAGBench cases.")
    parser.add_argument("--ragbench-file", required=True, help="Path to ragbench_test*.jsonl")
    parser.add_argument("--output", default="evaluation_data_rankgpt.json", help="Where to store evaluation entries JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for ranking/generation")
    parser.add_argument("--limit", type=int, help="Optional limit of cases to process")
    parser.add_argument("--cost-log", default=None, help="Optional JSONL file for per-call cost logging.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is not set. Please export it and retry.")
        sys.exit(1)

    args = parse_args()
    cases = load_ragbench_cases(args.ragbench_file)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases loaded; exiting.")
        return

    runner = RankGPTBaseline(model=args.model, cost_log=args.cost_log)
    evaluation_entries: List[dict] = []

    for idx, case in enumerate(cases, start=1):
        try:
            result = runner.run_case(case)
        except Exception as exc:
            print(f"⚠️ failed to process case {case.get('id')}: {exc}")
            continue
        evaluation_entries.append(build_evaluation_entry(case, result))
        if idx % 10 == 0 or idx == len(cases):
            print(f"Processed {idx}/{len(cases)} cases...")

    write_evaluation_file(evaluation_entries, args.output)
    print(f"✅ Saved {len(evaluation_entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
