import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from openai_model_patch import patch_openai_model_support
from cost_logging import CostLogger


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_level1_summary_source(path: Path) -> Dict[str, Dict[int, str]]:
    source: Dict[str, Dict[int, str]] = {}
    for row in load_jsonl(path):
        case_id = str(row.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"Missing case ID in Level-1 summary source: {path}")
        if case_id in source:
            raise ValueError(f"Duplicate case ID {case_id!r} in Level-1 summary source")

        summaries: Dict[int, str] = {}
        for cluster in row.get("raptor_clusters") or []:
            doc_indices = cluster.get("cluster_doc_indices") or []
            level1_summaries = cluster.get("level1_summaries") or []
            if len(doc_indices) != len(level1_summaries):
                raise ValueError(
                    f"Case {case_id} has mismatched document indices and Level-1 summaries"
                )
            for raw_index, raw_summary in zip(doc_indices, level1_summaries):
                doc_index = int(raw_index)
                summary = str(raw_summary).strip()
                if doc_index < 0 or not summary:
                    raise ValueError(f"Case {case_id} contains an invalid Level-1 summary")
                if doc_index in summaries and summaries[doc_index] != summary:
                    raise ValueError(
                        f"Case {case_id} has conflicting summaries for document {doc_index}"
                    )
                summaries[doc_index] = summary

        if not summaries:
            raise ValueError(f"Case {case_id} has no Level-1 summaries")
        source[case_id] = summaries

    return source


def validate_level1_summary_coverage(
    benchmark_rows: List[dict], source: Dict[str, Dict[int, str]]
) -> None:
    errors: List[str] = []
    for row in benchmark_rows:
        case_id = str(row.get("id") or "").strip()
        documents = row.get("documents") or []
        if case_id not in source:
            errors.append(f"{case_id or '<missing ID>'}: case not found")
            continue
        expected = set(range(len(documents)))
        actual = set(source[case_id])
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            errors.append(f"{case_id}: missing={missing}, extra={extra}")

    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"Level-1 summary coverage validation failed: {preview}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SummaryStore:
    def __init__(self, db_path: str, embed_model: HuggingFaceEmbeddings):
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection(
            name="query_doc_summaries", metadata={"hnsw:space": "cosine"}
        )
        self.embed_model = embed_model

    def fetch_summary_by_id(self, doc_id: str) -> Optional[str]:
        try:
            results = self.collection.get(where={"response": doc_id})
        except Exception as exc:
            print(f"⚠️  Failed to fetch summary for {doc_id}: {exc}")
            return None
        docs = results.get("documents", [])
        if docs:
            return docs[0]
        return None

    def fetch_summary_for_document(self, doc_id: str, doc_text: str) -> Optional[str]:
        summary = self.fetch_summary_by_id(doc_id)
        if summary:
            return summary
        try:
            embedding = self.embed_model.embed_query(doc_text)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents"],
            )
            docs = results.get("documents", [[]])
            if docs and docs[0]:
                return docs[0][0]
        except Exception as exc:
            print(f"⚠️  Summary fallback query failed: {exc}")
        return None


def build_vanilla_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)
    return (
        "You are a concise RAG assistant. Answer the question strictly using the"
        " provided contexts. If the contexts do not contain enough information, say so.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Contexts:\n{context_block}\n\n"
        "Answer:"
    )


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def embed_texts(model: HuggingFaceEmbeddings, texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    embeddings = model.embed_documents(texts)
    return [np.array(vec) for vec in embeddings]


def initialize_centroids(embeddings: List[np.ndarray], k: int) -> List[np.ndarray]:
    if not embeddings:
        return []
    centroids = [embeddings[0]]
    chosen = {0}
    while len(centroids) < k and len(centroids) < len(embeddings):
        best_idx = None
        best_score = float("inf")
        for i, emb in enumerate(embeddings):
            if i in chosen:
                continue
            sims = [cosine_similarity(emb, c) for c in centroids]
            score = max(sims) if sims else 0.0
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        centroids.append(embeddings[best_idx])
        chosen.add(best_idx)
    return centroids


def kmeans_cosine(
    embeddings: List[np.ndarray],
    k: int,
    iterations: int = 6,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    if not embeddings:
        return [], []
    if len(embeddings) <= k:
        clusters = [[i] for i in range(len(embeddings))]
        return clusters, embeddings[: len(clusters)]

    centroids = initialize_centroids(embeddings, k)
    if len(centroids) < k:
        # fallback: pad with remaining embeddings
        for emb in embeddings:
            if len(centroids) >= k:
                break
            centroids.append(emb)

    for _ in range(iterations):
        cluster_assignments = [[] for _ in range(len(centroids))]
        for idx, emb in enumerate(embeddings):
            sims = [cosine_similarity(emb, c) for c in centroids]
            best_cluster = int(np.argmax(sims)) if sims else 0
            cluster_assignments[best_cluster].append(idx)

        for i, cluster in enumerate(cluster_assignments):
            if cluster:
                centroids[i] = np.mean([embeddings[idx] for idx in cluster], axis=0)
            else:
                # Reinitialize empty cluster to farthest point
                sims = [max(cosine_similarity(emb, c) for c in centroids) for emb in embeddings]
                new_idx = int(np.argmin(sims))
                centroids[i] = embeddings[new_idx].copy()
                cluster_assignments[i] = [new_idx]
        clusters = cluster_assignments

    return clusters, centroids


def compute_adaptive_cluster_count(num_docs: int) -> int:
    if num_docs <= 0:
        return 1
    k = int(np.floor(np.sqrt(num_docs)))
    k = max(5, min(50, k))
    return min(k, num_docs)


def cluster_level1_summaries(
    summary_texts: List[str],
    embed_model: HuggingFaceEmbeddings,
) -> Tuple[List[List[int]], List[np.ndarray], List[np.ndarray]]:
    summary_embeddings = embed_texts(embed_model, summary_texts)
    k = compute_adaptive_cluster_count(len(summary_embeddings))
    clusters, centroids = kmeans_cosine(summary_embeddings, k)
    return clusters, centroids, summary_embeddings


def summarize_cluster(
    level1_summaries: List[str],
    llm: LlamaOpenAI,
    max_concat_chars: int,
    cost_logger: CostLogger | None = None,
    case_id: str | None = None,
) -> str:
    concatenated = "\n\n".join(truncate(text, max_concat_chars) for text in level1_summaries)
    prompt = (
        "You are summarizing summaries. Given several bullet summaries, produce an aggregated summary"
        " that captures the shared key information. Do not hallucinate missing details.\n\n"
        f"Summaries:\n{concatenated}\n\nCluster Summary:"
    )
    if cost_logger and cost_logger.enabled:
        return cost_logger.complete(llm, prompt, stage="cluster_summary", case_id=case_id).text.strip()
    return llm.complete(prompt=prompt).text.strip()


def build_contexts_for_generation(
    parent_summary: str,
    level1_summaries: List[str],
    documents: List[str],
    doc_indices: List[int],
    max_chars_summary: int,
    max_chars_doc: int,
) -> List[str]:
    sections = [f"[Cluster Summary]\n{truncate(parent_summary, max_chars_summary)}"]
    for idx, sum_text in enumerate(level1_summaries, start=1):
        sections.append(f"[Level-1 Summary {idx}]\n{truncate(sum_text, max_chars_summary)}")
    for doc_idx in doc_indices:
        if 0 <= doc_idx < len(documents):
            sections.append(f"[Doc {doc_idx}]\n{truncate(documents[doc_idx], max_chars_doc)}")
    return sections


def build_summary_lookup(documents: List[str], summary_store: SummaryStore) -> Dict[int, str]:
    summary_map: Dict[int, str] = {}
    for idx, doc in enumerate(documents):
        summary_text = summary_store.fetch_summary_by_id(doc_id=str(idx))
        if summary_text:
            summary_map[idx] = summary_text
    return summary_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-level Light-RAPTOR baseline")
    parser.add_argument("--benchmark-file", required=True, help="Path to ragbench_test*.jsonl")
    parser.add_argument("--summary-db", default="chroma_db_summary", help="Path to summary database")
    parser.add_argument("--output", default="raptor_run.jsonl", help="JSONL output file for RAPTOR results")
    parser.add_argument("--limit", type=int, help="Optional limit on questions to process")
    parser.add_argument(
        "--cluster-select",
        type=int,
        default=2,
        help="Number of parent clusters to route generation to (collapsed mode)",
    )
    parser.add_argument("--max-summary-chars", type=int, default=1000)
    parser.add_argument("--max-doc-chars", type=int, default=1200)
    parser.add_argument(
        "--global-top-k",
        "--cluster-pool-top-k",
        dest="global_top_k",
        type=int,
        default=12,
        help="Total number of mixed nodes (cluster summaries + level-1 summaries + docs) to feed into QA",
    )
    parser.add_argument(
        "--doc-top-k",
        type=int,
        default=5,
        help="(deprecated) retained for CLI compatibility; currently no effect",
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--cost-log", default=None, help="Optional JSONL file for per-call cost logging.")
    parser.add_argument(
        "--level1-summary-source",
        help="Optional RAPTOR JSONL providing Level-1 summaries by case and document index",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is not set.")
        sys.exit(1)

    args = parse_args()
    patch_openai_model_support()

    benchmark_rows = load_jsonl(Path(args.benchmark_file))
    if args.limit is not None:
        benchmark_rows = benchmark_rows[: args.limit]
    if not benchmark_rows:
        print("No benchmark entries found.")
        return

    level1_summary_source = None
    if args.level1_summary_source:
        source_path = Path(args.level1_summary_source)
        level1_summary_source = load_level1_summary_source(source_path)
        validate_level1_summary_coverage(benchmark_rows, level1_summary_source)
        print(f"Level-1 summary source validated for {len(benchmark_rows)} cases")

    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    summary_store = (
        None if level1_summary_source is not None else SummaryStore(args.summary_db, embed_model)
    )
    llm_kwargs = {"model": args.llm_model, "timeout": 600, "max_retries": 3}
    if args.llm_model in {"o4-mini", "o4"}:
        llm_kwargs["temperature"] = 1.0
    llm = LlamaOpenAI(**llm_kwargs)
    cost_logger = CostLogger(args.cost_log, "RAPTOR c4_k12", args.llm_model)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fout:
        for entry in benchmark_rows:
            question = entry.get("question", "").strip()
            documents = entry.get("documents") or []
            documents_sentences = entry.get("documents_sentences") or []
            if not question or not documents:
                continue

            # Fetch Level-1 summaries for each document ID
            question_id = str(entry.get("id") or "Q")
            if level1_summary_source is not None:
                fixed_summaries = level1_summary_source[question_id]
                doc_indices_with_summary = list(range(len(documents)))
                lvl1_summaries = [fixed_summaries[idx] for idx in doc_indices_with_summary]
            else:
                lvl1_summaries = []
                doc_indices_with_summary = []
                for idx, doc_text in enumerate(documents):
                    summary = summary_store.fetch_summary_for_document(str(idx), doc_text)
                    if summary:
                        lvl1_summaries.append(summary)
                        doc_indices_with_summary.append(idx)

            if not lvl1_summaries:
                continue

            clusters, centroids, summary_embeddings = cluster_level1_summaries(
                lvl1_summaries,
                embed_model,
            )
            if not clusters:
                continue

            question_embedding = np.array(embed_model.embed_query(question))
            doc_embeddings = embed_texts(embed_model, documents)

            cluster_scores = []
            for centroid in centroids:
                cluster_scores.append(cosine_similarity(question_embedding, centroid))

            cluster_order = sorted(range(len(clusters)), key=lambda i: cluster_scores[i], reverse=True)
            if args.cluster_select > 0:
                selected_cluster_indices = cluster_order[: args.cluster_select]
            else:
                selected_cluster_indices = cluster_order

            cluster_records = []
            global_nodes: List[Dict[str, object]] = []
            for idx, cluster_idx in enumerate(selected_cluster_indices, start=1):
                cluster_indices = clusters[cluster_idx]
                if not cluster_indices:
                    continue
                summaries_in_cluster = [lvl1_summaries[i] for i in cluster_indices]
                doc_ids = [doc_indices_with_summary[i] for i in cluster_indices]
                cluster_label = f"{question_id}_C{idx}"

                parent_summary = summarize_cluster(
                    summaries_in_cluster,
                    llm,
                    args.max_summary_chars,
                    cost_logger=cost_logger,
                    case_id=str(question_id),
                )
                parent_embedding = np.array(embed_model.embed_query(parent_summary))
                parent_similarity = cosine_similarity(question_embedding, parent_embedding)

                cluster_nodes: List[Dict[str, object]] = []
                parent_node = {
                    "node_id": f"{cluster_label}_P",
                    "cluster_id": cluster_label,
                    "node_type": "cluster_summary",
                    "doc_id": None,
                    "text": parent_summary,
                    "similarity": parent_similarity,
                }
                cluster_nodes.append(parent_node)
                global_nodes.append(parent_node)

                for local_idx, summary_idx in enumerate(cluster_indices):
                    doc_id = doc_indices_with_summary[summary_idx]
                    summary_text = lvl1_summaries[summary_idx]
                    sum_emb = summary_embeddings[summary_idx]
                    sum_similarity = cosine_similarity(question_embedding, sum_emb)
                    summary_node = {
                        "node_id": f"{cluster_label}_S{local_idx}",
                        "cluster_id": cluster_label,
                        "node_type": "summary",
                        "doc_id": doc_id,
                        "text": summary_text,
                        "similarity": sum_similarity,
                    }
                    cluster_nodes.append(summary_node)
                    global_nodes.append(summary_node)

                    doc_similarity = 0.0
                    if 0 <= doc_id < len(doc_embeddings):
                        doc_similarity = cosine_similarity(question_embedding, doc_embeddings[doc_id])
                    doc_node = {
                        "node_id": f"{cluster_label}_D{doc_id}",
                        "cluster_id": cluster_label,
                        "node_type": "document",
                        "doc_id": doc_id,
                        "text": documents[doc_id],
                        "similarity": doc_similarity,
                    }
                    cluster_nodes.append(doc_node)
                    global_nodes.append(doc_node)

                cluster_records.append(
                    {
                        "cluster_id": cluster_label,
                        "parent_summary": parent_summary,
                        "cluster_doc_indices": doc_ids,
                        "selected_doc_indices": [],
                        "selected_candidates": [],
                        "level1_summaries": summaries_in_cluster,
                        "cluster_nodes": [
                            {
                                "node_id": node["node_id"],
                                "node_type": node["node_type"],
                                "doc_id": node.get("doc_id"),
                                "similarity": node["similarity"],
                                "text_excerpt": truncate(str(node["text"]), 200),
                            }
                            for node in cluster_nodes
                        ],
                        "cluster_response": None,
                    }
                )

            if not global_nodes:
                continue

            global_nodes.sort(key=lambda node: float(node["similarity"]), reverse=True)
            context_limit = max(1, args.global_top_k)
            selected_nodes = global_nodes[:context_limit]
            if not selected_nodes:
                continue

            contexts: List[str] = []
            selected_nodes_metadata: List[Dict[str, object]] = []
            for node in selected_nodes:
                node_type = str(node["node_type"])
                doc_id = node.get("doc_id")
                cluster_label = str(node["cluster_id"])
                if node_type == "cluster_summary":
                    label = f"[Cluster Summary {cluster_label}]"
                    max_chars = args.max_summary_chars
                elif node_type == "summary":
                    label = f"[Level-1 Summary Doc {doc_id}]"
                    max_chars = args.max_summary_chars
                else:
                    label = f"[Doc {doc_id}]"
                    max_chars = args.max_doc_chars
                context_text = truncate(str(node["text"]), max_chars)
                contexts.append(f"{label}\n{context_text}")
                selected_nodes_metadata.append(
                    {
                        "node_id": node["node_id"],
                        "node_type": node_type,
                        "cluster_id": cluster_label,
                        "doc_id": doc_id,
                        "similarity": node["similarity"],
                        "text_excerpt": truncate(str(node["text"]), 200),
                    }
                )

            if not contexts:
                continue

            prompt = build_vanilla_prompt(question, contexts)
            if cost_logger.enabled:
                final_response = cost_logger.complete(
                    llm,
                    prompt,
                    stage="answer_generation",
                    case_id=str(question_id),
                ).text.strip()
            else:
                final_response = llm.complete(prompt=prompt).text.strip()
            if not final_response:
                continue

            cluster_doc_hits: Dict[str, List[int]] = {}
            for node in selected_nodes_metadata:
                if node["node_type"] == "document" and node["doc_id"] is not None:
                    cluster_doc_hits.setdefault(str(node["cluster_id"]), []).append(int(node["doc_id"]))

            for cluster_record in cluster_records:
                c_id = cluster_record["cluster_id"]
                cluster_record["selected_doc_indices"] = cluster_doc_hits.get(c_id, [])
                cluster_record["selected_candidates"] = [
                    node for node in selected_nodes_metadata if node["cluster_id"] == c_id
                ]
                cluster_record["cluster_response"] = final_response

            selected_cluster_entries = []
            for cluster_record in cluster_records:
                c_id = cluster_record["cluster_id"]
                selected_cluster_entries.append(
                    {
                        "summary_id": c_id,
                        "summary_text": cluster_record["parent_summary"],
                        "doc_indices": cluster_doc_hits.get(c_id, []),
                    }
                )

            record = {
                "id": entry.get("id"),
                "question": question,
                "documents": documents,
                "documents_sentences": documents_sentences,
                "raptor_clusters": cluster_records,
                "selected_clusters": selected_cluster_entries,
                "selected_nodes": selected_nodes_metadata,
                "raptor_response": final_response,
                "level1_summary_source": args.level1_summary_source,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Two-level Light-RAPTOR generation completed → {args.output}")


if __name__ == "__main__":
    main()
