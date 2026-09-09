import argparse
import json
import os
from typing import Optional

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from enhanced_model_patch import comprehensive_patch as patch_openai_model_support


def load_quality_entries(path: str) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose FP/FN/Hallucinations from existing response_quality_evaluation jsonl"
    )
    parser.add_argument("--input", required=True, help="Existing response_quality_evaluation_*.jsonl")
    parser.add_argument("--output", default="response_quality_diagnostics.json", help="Output JSON file")
    parser.add_argument("--label", default=None, help="Optional prefix for diagnostics fields (e.g., raptor)")
    parser.add_argument("--quality-label", default=None, help="Optional label text inside 'Score for ... response' lines")
    parser.add_argument("--limit", type=int, help="Optional limit on number of entries to process")
    return parser.parse_args()


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    args = parse_args()
    entries = load_quality_entries(args.input)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("⚠️  No entries found; exiting.")
        return

    patch_openai_model_support()
    llm = LlamaOpenAI(model="gpt-5-mini", timeout=600, max_retries=3)
    diagnostics = []
    total = len(entries)
    progress_step = max(1, total // 20) if total else 1

    for idx, entry in enumerate(entries, start=1):
        question = entry.get("question", "")
        quality_text = entry.get("quality", "")
        original_response = entry.get("original_response")

        # Determine which response field to use based on quality_label
        quality_label_lower = args.quality_label.lower() if args.quality_label else ""
        if "original response" in quality_label_lower:
            comparison_response = original_response
        else:
            # All other systems use the milo_response field
            comparison_response = entry.get("milo_response") 
        if not question or comparison_response is None:
            continue

        label = args.label.strip() if args.label else None
        prefix = f"{label}_" if label else ""
        documents = entry.get("documents") or []
        doc_verified = False
        if not documents:
            diag = {
                "fp": False,
                "fn": False,
                "hallucination_explanation": "Skipped: documents missing.",
                "omission_explanation": "",
                "diagnostic_failed": True,
            }
        else:
            formatted_docs = []
            for doc_idx, doc in enumerate(documents, start=1):
                doc_text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
                formatted_docs.append(f"[Document {doc_idx}]\n{doc_text}")
            documents_block = "\n\n".join(formatted_docs)
            doc_prompt = f"""
You are verifying a system response using the provided documents. Treat the documents as the final ground truth. The evaluation summary is only a hint (possibly empty).

Instructions:
- Set fp=true if the response:
    1) states a factual claim that contradicts the documents, OR
    2) asserts a concrete factual detail that is not supported by ANY of the documents (i.e., no document snippet justifies it).
- If all provided documents together do NOT contain enough information to answer the question, but the response still gives a specific factual answer, treat that as fp=true.
- Set fn=true if the documents contain the necessary answer/information but the response fails to deliver it or gives an incorrect/misleading answer.
**Logical Interpretation Rules (Apply these before marking fp=true):**
1. **Scope-Sensitive Negation:** When the response denies a claim (e.g., "documents do not state X"), evaluate this denial strictly within the **specific constraints** (location, timeframe, entity) requested by the user query. Do NOT mark fp=true just because the document mentions the keyword in a *different* context or location than what was asked.
2. **Functional Distinction:** Do not conflate entities that share a broad domain/topic but belong to different functional categories (e.g., content vs. platform, product vs. creator). If the response correctly distinguishes them as different types based on document evidence, do NOT mark this as a contradiction.
3. **Status vs. Name:** Distinguish between an entity's *name* and its *documented status*. If a document explicitly states an activity has ceased (e.g., "closed", "formerly"), the response is correct to deny that the entity *currently* performs that activity, even if the activity remains part of its proper name.
- Do NOT use any external or world knowledge (e.g., geography, distances between countries, population sizes, etc.) to judge whether the system response is supported.
- If the response references document identifiers that are not present in the provided list, ignore that metadata reference. Only mark fp when the response introduces a concrete factual detail about the task domain that lacks document support.
- If none of the fp or fn conditions apply, set fp=false and fn=false.
- Provide concise explanations referencing document evidence:
    - hallucination_explanation: why fp is true or false.
    - omission_explanation: why fn is true or false.
- Return ONLY JSON with keys: fp, fn, hallucination_explanation, omission_explanation.

Question: {question}
System Response: {comparison_response}
Evaluation Summary (hint): {quality_text}
Documents:
{documents_block}
"""
#             doc_prompt = f"""
# You are verifying a system response using the provided documents. Treat the documents as the final ground truth. The evaluation summary is only a hint (possibly empty).

# Instructions:
# - Set fp=true if the response:
#     1) states a factual claim that contradicts the documents, OR
#     2) asserts a concrete factual detail that is not supported by ANY of the documents (i.e., no document snippet justifies it).
# - If all provided documents together do NOT contain enough information to answer the question, but the response still gives a specific factual answer, treat that as fp=true.
# - Set fn=true if the documents contain the necessary answer/information but the response fails to deliver it or gives an incorrect/misleading answer.
# - Do NOT use any external or world knowledge (e.g., geography, distances between countries, population sizes, etc.) to judge whether the system response is supported.
# - If the response references document identifiers that are not present in the provided list, ignore that metadata reference. Only mark fp when the response introduces a concrete factual detail about the task domain that lacks document support.
# - If none of the fp or fn conditions apply, set fp=false and fn=false.
# - Provide concise explanations referencing document evidence:
#     - hallucination_explanation: why fp is true or false.
#     - omission_explanation: why fn is true or false.
# - Return ONLY JSON with keys: fp, fn, hallucination_explanation, omission_explanation.

# Question: {question}
# System Response: {comparison_response}
# Evaluation Summary (hint): {quality_text}
# Documents:
# {documents_block}
# """
            diag_text = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    diag_text = llm.complete(prompt=doc_prompt).text.strip()
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        print(f"⚠️  LLM call failed (attempt {attempt+1}/{max_retries}): {exc}")
                    else:
                        print(f"❌ LLM call failed after {max_retries} attempts: {exc}")
                        diag_text = None
            if diag_text is None:
                diag = {
                    "fp": False,
                    "fn": False,
                    "hallucination_explanation": "LLM diagnostic failed",
                    "omission_explanation": "LLM diagnostic failed",
                    "diagnostic_failed": True,
                }
            else:
                try:
                    doc_diag = json.loads(diag_text)
                except json.JSONDecodeError:
                    doc_diag = {
                        "fp": False,
                        "fn": False,
                        "hallucination_explanation": "",
                        "omission_explanation": "",
                        "raw": diag_text,
                        "diagnostic_failed": True,
                    }
                diag = {
                    "fp": bool(doc_diag.get("fp")),
                    "fn": bool(doc_diag.get("fn")),
                    "hallucination_explanation": doc_diag.get("hallucination_explanation", ""),
                    "omission_explanation": doc_diag.get("omission_explanation", ""),
                    "diagnostic_failed": doc_diag.get("diagnostic_failed", False),
                }
                if "raw" in doc_diag:
                    diag["raw"] = doc_diag["raw"]
                doc_verified = True

        diagnostics.append(
            {
                "question": question,
                f"{prefix}fp": bool(diag.get("fp")),
                f"{prefix}fn": bool(diag.get("fn")),
                f"{prefix}hallucination_explanation": diag.get("hallucination_explanation", ""),
                f"{prefix}omission_explanation": diag.get("omission_explanation", ""),
                f"{prefix}diagnostic_failed": diag.get("diagnostic_failed", False),
                f"{prefix}doc_verification": doc_verified,
            }
        )
        if idx % progress_step == 0 or idx == total:
            pct = idx / total * 100
            print(f"\rDiagnosing progress: {idx}/{total} ({pct:5.1f}%)", end="", flush=True)
    if total:
        print()

    with open(args.output, "w", encoding="utf-8") as f:
        for diag in diagnostics:
            f.write(json.dumps(diag, ensure_ascii=False) + "\n")
    print(f"✅ Diagnostics written to {args.output}")


if __name__ == "__main__":
    main()
