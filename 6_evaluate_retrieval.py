"""Retrieval evaluation for dense, lexical, and hybrid strategies.

This script measures retrieval quality over 20 hand-tagged queries using
keyword-coverage recall@k, precision@k, topic_recall@k, and latency across
three strategies (dense, lexical, hybrid).

It attempts to import retrieval functions from `3_agent_orchestration` to stay
aligned with dashboard behavior. If unavailable, it falls back to local
implementations and logs a warning.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import chromadb
import mlflow
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

EVAL_DIR = "/workspace/project/eval"
RANDOM_SEED = 42
N_SAMPLES_SENTIMENT = 200
N_QUERIES_RETRIEVAL = 20
TOP_K = 10

PERSIST_DIR = "./youtube_vector_db"
COLLECTION_NAME = "youtube_comments"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOGGER = logging.getLogger(__name__)

QUERIES: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "What are the main issues or complaints people have?",
        "expected_keywords": ["issue", "complaint", "problem", "wrong", "broken", "bad", "hate", "dislike", "error"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q2",
        "query": "Which tutorials focus on Power BI dashboards?",
        "expected_keywords": ["dashboard", "visual", "report", "chart"],
        "expected_topic_ids": [0, 1, 2],
    },
    {
        "id": "q3",
        "query": "Summarize positive reactions in these comments.",
        "expected_keywords": ["great", "helpful", "love", "awesome", "good", "excellent"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q4",
        "query": "Summarize negative sentiment around tutorial pacing.",
        "expected_keywords": ["too fast", "confusing", "hard", "unclear", "slow", "difficult"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q5",
        "query": "Which people are mentioned by name?",
        "expected_keywords": ["he", "she", "instructor", "teacher", "guy", "person"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q6",
        "query": "Which products or tools are being discussed?",
        "expected_keywords": ["power bi", "excel", "sql", "tableau", "tool", "software"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q7",
        "query": "What are comments saying about topic 0?",
        "expected_keywords": ["topic", "tutorial", "learn", "example"],
        "expected_topic_ids": [0],
    },
    {
        "id": "q8",
        "query": "What are comments saying about topic 1?",
        "expected_keywords": ["visual", "dashboard", "chart", "report"],
        "expected_topic_ids": [1],
    },
    {
        "id": "q9",
        "query": "What are comments saying about topic 2?",
        "expected_keywords": ["data", "model", "table", "relationship"],
        "expected_topic_ids": [2],
    },
    {
        "id": "q10",
        "query": "What are comments saying about topic 3?",
        "expected_keywords": ["measure", "dax", "formula", "calculation"],
        "expected_topic_ids": [3],
    },
    {
        "id": "q11",
        "query": "What are comments saying about topic 4?",
        "expected_keywords": ["filter", "slicer", "page", "interaction"],
        "expected_topic_ids": [4],
    },
    {
        "id": "q12",
        "query": "What are comments saying about topic 5?",
        "expected_keywords": ["beginner", "start", "basic", "step"],
        "expected_topic_ids": [5],
    },
    {
        "id": "q13",
        "query": "What are comments saying about topic 6?",
        "expected_keywords": ["advanced", "performance", "optimize", "complex"],
        "expected_topic_ids": [6],
    },
    {
        "id": "q14",
        "query": "What are comments saying about topic 7?",
        "expected_keywords": ["download", "file", "resource", "link"],
        "expected_topic_ids": [7],
    },
    {
        "id": "q15",
        "query": "What are comments saying about topic 8?",
        "expected_keywords": ["exam", "assignment", "project", "practice"],
        "expected_topic_ids": [8],
    },
    {
        "id": "q16",
        "query": "What are comments saying about topic 9?",
        "expected_keywords": ["thanks", "subscribed", "channel", "video"],
        "expected_topic_ids": [9],
    },
    {
        "id": "q17",
        "query": "Show comments on cYwioeHu_OU and common pain points.",
        "expected_keywords": ["cYwioeHu_OU", "pain", "problem", "error", "stuck"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q18",
        "query": "Show comments on Lfzu74XDyco and what viewers found useful.",
        "expected_keywords": ["Lfzu74XDyco", "helpful", "useful", "clear", "great"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q19",
        "query": "Show comments on TiS6vnju_mI and recurring suggestions.",
        "expected_keywords": ["TiS6vnju_mI", "should", "please", "add", "improve"],
        "expected_topic_ids": [None],
    },
    {
        "id": "q20",
        "query": "In one view, summarize sentiment, named products, and dominant topics for dashboard tutorials.",
        "expected_keywords": ["sentiment", "product", "topic", "dashboard", "tutorial", "power bi"],
        "expected_topic_ids": [0, 1, 2],
    },
]


def _fallback_dense(collection: Any, query: str, n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
    return collection.query(query_texts=[query], n_results=n_results, where=where)


def _fallback_lexical(
    query: str,
    documents: list[str],
    ids: list[str],
    n_results: int = 10,
    metadatas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    scores = np.array(bm25.get_scores(query.lower().split()))

    if scores.size == 0:
        return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]], "raw_scores": [[]]}

    if scores.max() == scores.min():
        norm = np.zeros_like(scores)
    else:
        norm = (scores - scores.min()) / (scores.max() - scores.min())

    top_idx = np.argsort(-norm)[:n_results]
    return {
        "documents": [[documents[i] for i in top_idx]],
        "ids": [[ids[i] for i in top_idx]],
        "metadatas": [[metadatas[i] if metadatas else {} for i in top_idx]],
        "distances": [[float(1.0 - norm[i]) for i in top_idx]],
        "raw_scores": [[float(scores[i]) for i in top_idx]],
    }


def _fallback_rrf(*result_lists: dict[str, Any], k: int = 60) -> dict[str, Any]:
    from collections import defaultdict

    scores: dict[Any, float] = defaultdict(float)
    doc_map: dict[Any, str] = {}
    meta_map: dict[Any, dict[str, Any]] = {}

    for result in result_lists:
        docs = result.get("documents", [[]])[0]
        ids = result.get("ids", [list(range(len(docs)))])[0]
        metas = result.get("metadatas", [[{}] * len(docs)])[0]
        for rank, (doc, doc_id, meta) in enumerate(zip(docs, ids, metas)):
            scores[doc_id] += 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc
            meta_map[doc_id] = meta

    ranked_ids = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])]
    if not ranked_ids:
        return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}

    max_s = max(scores.values())
    min_s = min(scores.values())
    denom = max_s - min_s if max_s != min_s else 1.0
    distances = [1.0 - (scores[doc_id] - min_s) / denom for doc_id in ranked_ids]
    return {
        "documents": [[doc_map[i] for i in ranked_ids]],
        "ids": [ranked_ids],
        "metadatas": [[meta_map[i] for i in ranked_ids]],
        "distances": [distances],
    }


def _load_retrievers() -> tuple[Any, Any, Any, bool]:
    try:
        orch = importlib.import_module("3_agent_orchestration")
        return orch.retrieve_dense, orch.retrieve_lexical, orch.retrieve_hybrid, True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not import 3_agent_orchestration (%s). Using fallback retrieval functions.", exc)

        def _fallback_hybrid(collection: Any, query: str, n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
            dense = _fallback_dense(collection, query, n_results=n_results * 2, where=where)
            lexical = _fallback_lexical(
                query=query,
                documents=CACHED_DOCS,
                ids=CACHED_IDS,
                n_results=n_results * 2,
                metadatas=CACHED_METADATAS,
            )
            fused = _fallback_rrf(dense, lexical)
            for key in fused:
                fused[key] = [fused[key][0][:n_results]]
            return fused

        return _fallback_dense, _fallback_lexical, _fallback_hybrid, False


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _topic_recall(metadatas: list[dict[str, Any]], expected_topic_ids: list[Any], k: int) -> float:
    if not expected_topic_ids or expected_topic_ids == [None]:
        return float("nan")

    expected = {int(topic_id) for topic_id in expected_topic_ids if topic_id is not None}
    if not expected:
        return float("nan")

    hits = 0
    for meta in metadatas[:k]:
        topic_id = meta.get("topic_id") if isinstance(meta, dict) else None
        try:
            if topic_id is not None and int(topic_id) in expected:
                hits += 1
        except (TypeError, ValueError):
            continue
    return hits / float(k)


def _load_collection() -> Any:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


CACHED_DOCS: list[str] = []
CACHED_IDS: list[str] = []
CACHED_METADATAS: list[dict[str, Any]] = []


def run_evaluation(n_queries: int = N_QUERIES_RETRIEVAL, top_k: int = TOP_K, seed: int = RANDOM_SEED) -> dict[str, Any]:
    del seed  # Seed kept for CLI contract; current query set is deterministic.

    os.makedirs(EVAL_DIR, exist_ok=True)
    eval_dir = Path(EVAL_DIR)

    collection = _load_collection()
    corpus = collection.get(include=["documents", "metadatas"])

    global CACHED_DOCS, CACHED_IDS, CACHED_METADATAS
    CACHED_DOCS = [str(doc) for doc in corpus.get("documents", [])]
    CACHED_IDS = [str(doc_id) for doc_id in corpus.get("ids", [])]
    CACHED_METADATAS = [meta if isinstance(meta, dict) else {} for meta in corpus.get("metadatas", [])]

    if not CACHED_DOCS:
        raise ValueError("Vector DB is empty. Run 2_build-database.py first.")

    retrieve_dense, retrieve_lexical, retrieve_hybrid, imported_from_file3 = _load_retrievers()

    selected_queries = QUERIES[: min(n_queries, len(QUERIES))]
    records: list[dict[str, Any]] = []

    for q in selected_queries:
        query_text = str(q["query"])
        expected_keywords = list(q["expected_keywords"])
        expected_topic_ids = list(q["expected_topic_ids"])

        for strategy in ["dense", "lexical", "hybrid"]:
            start = time.perf_counter()
            if strategy == "dense":
                result = retrieve_dense(collection, query_text, n_results=top_k)
            elif strategy == "lexical":
                result = retrieve_lexical(
                    query_text,
                    CACHED_DOCS,
                    CACHED_IDS,
                    n_results=top_k,
                    metadatas=CACHED_METADATAS,
                )
            else:
                if imported_from_file3:
                    # Keep hybrid aligned with file 3 retrieval logic while using cached corpus for lexical component.
                    dense_result = retrieve_dense(collection, query_text, n_results=top_k * 2)
                    lexical_result = retrieve_lexical(
                        query_text,
                        CACHED_DOCS,
                        CACHED_IDS,
                        n_results=top_k * 2,
                        metadatas=CACHED_METADATAS,
                    )
                    rrf = getattr(importlib.import_module("3_agent_orchestration"), "reciprocal_rank_fusion", _fallback_rrf)
                    result = rrf(dense_result, lexical_result)
                    for key in result:
                        result[key] = [result[key][0][:top_k]]
                else:
                    result = retrieve_hybrid(collection, query_text, n_results=top_k)

            latency_ms = (time.perf_counter() - start) * 1000.0

            docs = [str(doc) for doc in result.get("documents", [[]])[0][:top_k]]
            metas = result.get("metadatas", [[]])[0][:top_k] if result.get("metadatas") else [{} for _ in docs]

            matched_keywords = {
                kw for kw in expected_keywords if any(kw.lower() in doc.lower() for doc in docs)
            }
            docs_with_keyword = sum(1 for doc in docs if _contains_any_keyword(doc, expected_keywords))

            recall_k = len(matched_keywords) / float(len(expected_keywords)) if expected_keywords else 0.0
            precision_k = docs_with_keyword / float(top_k)
            topic_recall_k = _topic_recall([m if isinstance(m, dict) else {} for m in metas], expected_topic_ids, top_k)

            records.append(
                {
                    "query_id": q["id"],
                    "query": query_text,
                    "strategy": strategy,
                    "recall@10": recall_k,
                    "precision@10": precision_k,
                    "topic_recall@10": topic_recall_k,
                    "latency_ms": latency_ms,
                }
            )

    per_query_df = pd.DataFrame(records)

    summary = (
        per_query_df.groupby("strategy", as_index=False)
        .agg(
            mean_recall_at_10=("recall@10", "mean"),
            mean_precision_at_10=("precision@10", "mean"),
            mean_topic_recall_at_10=("topic_recall@10", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .sort_values("mean_recall_at_10", ascending=False)
        .reset_index(drop=True)
    )

    winner = summary.iloc[0]
    best_single = summary[summary["strategy"].isin(["dense", "lexical"])].sort_values(
        "mean_recall_at_10", ascending=False
    )
    best_single_recall = float(best_single.iloc[0]["mean_recall_at_10"]) if not best_single.empty else 0.0
    hybrid_recall = float(summary.loc[summary["strategy"] == "hybrid", "mean_recall_at_10"].iloc[0]) if "hybrid" in summary["strategy"].values else 0.0
    improvement_pct = ((hybrid_recall - best_single_recall) / best_single_recall * 100.0) if best_single_recall > 0 else 0.0
    verdict = f"Hybrid improves recall@10 by {improvement_pct:.2f}% over best single strategy."

    mlflow.set_experiment("YouTube_Eval")
    with mlflow.start_run(run_name="retrieval_eval"):
        for _, row in summary.iterrows():
            strategy = row["strategy"]
            mlflow.log_metric(f"recall@10_{strategy}", float(row["mean_recall_at_10"]))
            mlflow.log_metric(f"precision@10_{strategy}", float(row["mean_precision_at_10"]))
            mlflow.log_metric(f"latency_ms_{strategy}", float(row["mean_latency_ms"]))

    (eval_dir / "retrieval_queries.json").write_text(json.dumps(selected_queries, indent=2), encoding="utf-8")
    per_query_df.to_csv(eval_dir / "retrieval_per_query.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(eval_dir / "retrieval_strategy_summary.csv", index=False, encoding="utf-8-sig")

    winning_strategy = str(winner["strategy"])
    merged = per_query_df.merge(
        per_query_df[per_query_df["strategy"] == winning_strategy][["query_id", "recall@10"]].rename(
            columns={"recall@10": "winner_recall"}
        ),
        on="query_id",
        how="left",
    )
    merged["delta_vs_winner"] = merged["winner_recall"] - merged["recall@10"]
    failures = merged.sort_values("delta_vs_winner", ascending=False).head(3)

    report_lines = [
        "# Retrieval Evaluation Report",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Strategy Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Top-3 Failure-Mode Examples",
        "",
    ]
    for _, row in failures.iterrows():
        report_lines.append(
            f"- Query {row['query_id']} ({row['strategy']}): recall@10={row['recall@10']:.3f}, "
            f"winner_recall={row['winner_recall']:.3f}, delta={row['delta_vs_winner']:.3f}"
        )

    (eval_dir / "retrieval_evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Retrieval evaluation summary:")
    print(summary.to_string(index=False))
    print(verdict)

    return {
        "summary": summary,
        "verdict": verdict,
        "imported_from_file3": imported_from_file3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense/lexical/hybrid retrieval quality.")
    parser.add_argument("--n-queries", type=int, default=N_QUERIES_RETRIEVAL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    run_evaluation(n_queries=args.n_queries, top_k=args.top_k, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
