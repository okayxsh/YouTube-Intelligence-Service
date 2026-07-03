"""Intelligent RAG orchestration for YouTube comments.

Supports three retrieval strategies over `youtube_comments` in `./youtube_vector_db`:
- dense: Chroma embedding retrieval
- lexical: BM25 over retrieved corpus text
- hybrid: dense + lexical fused with RRF (k=60)

Routes queries into four local agent styles based on regex intent:
- sentiment_summary, entity_query, topic_query, general_qa

Filter rules:
- topic query with `topic N` applies `where={"topic_id": N}`
- sentiment queries mentioning positive/negative apply polarity thresholds

Guardrail:
- If retrieved evidence is weak (similarity proxy < 0.5) or answer contains
  hallucination-like phrases, force `Data insufficient.`

Examples:
- python3 3_agent-orchestration.py --query "topic 3 complaints" --strategy hybrid
- python3 3_agent-orchestration.py --query "Which companies are mentioned?" --strategy lexical --show-retrieved
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import chromadb
import mlflow
import ollama
from chromadb.utils import embedding_functions

# NOTE: requirements.txt must pin `onnxruntime==1.18.1`
# for the spacy optimization path to work cleanly.
# Do NOT add the mock workaround back.

# === Constants ===
PERSIST_DIR = "./youtube_vector_db"
COLLECTION_NAME = "youtube_comments"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_QUERY = "What are the main issues or complaints people have in these comments?"
DEFAULT_MODEL = "phi3"
DEFAULT_N_RESULTS = 10
EXPERIMENT_NAME = "YouTube_Intelligence_Engine"
LOGGER = logging.getLogger(__name__)


# === DB Connection ===
def load_index(collection_name: str = COLLECTION_NAME) -> tuple[Any, Any, Any]:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    return client, collection, embedding_function


# === Retrieval layer (dense / lexical / hybrid) ===
def retrieve_dense(collection: Any, query: str, n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
    """ChromaDB cosine-near retrieval over embedded documents (distance returned as L2 proxy)."""
    return collection.query(query_texts=[query], n_results=n_results, where=where)


def retrieve_lexical(
    query: str,
    documents: list[str],
    ids: list[str],
    n_results: int = 10,
    metadatas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """BM25 over clean_text with Chroma-shaped return payload."""
    from rank_bm25 import BM25Okapi
    import numpy as np

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
    distances = (1.0 - norm[top_idx]).tolist()
    return {
        "documents": [[documents[i] for i in top_idx]],
        "ids": [[ids[i] for i in top_idx]],
        "metadatas": [[metadatas[i] if metadatas else {} for i in top_idx]],
        "distances": [distances],
        "raw_scores": [[float(scores[i]) for i in top_idx]],
    }


def reciprocal_rank_fusion(*result_lists: dict[str, Any], k: int = 60) -> dict[str, Any]:
    """RRF fusion with k=60 (Cormack et al., 2009 textbook default for robust hybrid rank merging)."""
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

    if not scores:
        return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}

    ranked = sorted(scores.items(), key=lambda item: -item[1])
    top_ids = [doc_id for doc_id, _ in ranked]
    max_s = max(scores.values())
    min_s = min(scores.values())
    denom = max_s - min_s if max_s != min_s else 1.0
    top_dists = [1.0 - (scores[doc_id] - min_s) / denom for doc_id in top_ids]
    return {
        "documents": [[doc_map[i] for i in top_ids]],
        "ids": [top_ids],
        "metadatas": [[meta_map[i] for i in top_ids]],
        "distances": [top_dists],
    }


def _metadata_matches_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if where is None:
        return True
    for key, value in where.items():
        if isinstance(value, dict):
            target = meta.get(key)
            if "$lt" in value and not (isinstance(target, (int, float)) and target < value["$lt"]):
                return False
            if "$gt" in value and not (isinstance(target, (int, float)) and target > value["$gt"]):
                return False
        elif meta.get(key) != value:
            return False
    return True


def retrieve_hybrid(collection: Any, query: str, n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dense + lexical fusion over same corpus using RRF(k=60)."""
    dense_result = retrieve_dense(collection, query, n_results=n_results * 2, where=where)
    all_docs = collection.get(include=["documents", "metadatas"])

    corpus_docs = all_docs.get("documents", [])
    corpus_ids = all_docs.get("ids", [])
    corpus_meta = all_docs.get("metadatas", [])
    filtered = [(d, i, m) for d, i, m in zip(corpus_docs, corpus_ids, corpus_meta) if _metadata_matches_where(m or {}, where)]

    if not filtered:
        return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}

    docs, ids, metas = map(list, zip(*filtered))
    lex_result = retrieve_lexical(query, docs, ids, n_results=n_results * 2, metadatas=metas)
    fused = reciprocal_rank_fusion(dense_result, lex_result)
    for key in fused:
        fused[key] = [fused[key][0][:n_results]]
    return fused


# === Agent specs & query classifier ===
FilterFn = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class AgentSpec:
    persona: str
    role: str
    task: str
    output_format: str
    max_cites: int
    filter_fn: FilterFn


def _sentiment_filter(query: str) -> dict[str, Any] | None:
    lowered = query.lower()
    if "negative" in lowered:
        return {"sentiment_polarity": {"$lt": -0.1}}
    if "positive" in lowered:
        return {"sentiment_polarity": {"$gt": 0.1}}
    return None


def _topic_filter(query: str) -> dict[str, Any] | None:
    match = re.search(r"\btopic\s*(\d+)\b", query, flags=re.IGNORECASE)
    if match:
        return {"topic_id": int(match.group(1))}
    return None


AGENTS: dict[str, AgentSpec] = {
    "sentiment_summary": AgentSpec(
        persona="an audience-analyst at a content studio",
        role="sentiment specialist grounded only in retrieved comments",
        task="Summarize audience sentiment for the query.",
        output_format="a bullet list grouped by positive / neutral / negative",
        max_cites=8,
        filter_fn=_sentiment_filter,
    ),
    "entity_query": AgentSpec(
        persona="an entity-recognition analyst",
        role="entity-focused analyst extracting named mentions",
        task="Answer with grounded entities and evidence from comments.",
        output_format="a bullet list grouped by entity label (PEOPLE, ORGS, PRODUCTS, PLACES)",
        max_cites=8,
        filter_fn=lambda _q: None,
    ),
    "topic_query": AgentSpec(
        persona="a topic-modeling analyst",
        role="topic interpreter for clustered comment themes",
        task="Explain dominant topics and their frequency hints from context.",
        output_format="a bullet list of dominant topics with comment counts",
        max_cites=8,
        filter_fn=_topic_filter,
    ),
    "general_qa": AgentSpec(
        persona="a precise YouTube-comment analyst",
        role="general analyst answering with strict grounding",
        task="Answer the question directly from retrieved comments.",
        output_format="a 2-3 sentence paragraph",
        max_cites=6,
        filter_fn=lambda _q: None,
    ),
}


def classify_query(query: str) -> str:
    """Local regex router. Priority: sentiment > entity > topic > general_qa."""
    if re.search(r"\b(sentiment|feeling|positive|negative|mood|happy|angry|sad)\b", query, flags=re.IGNORECASE):
        return "sentiment_summary"
    if re.search(r"\b(which|who|what|where).*\b(brand|company|person|product|place|mention|named)\b", query, flags=re.IGNORECASE):
        return "entity_query"
    if re.search(r"\b(topic|themes?|about|discuss)\b", query, flags=re.IGNORECASE):
        return "topic_query"
    return "general_qa"


def _build_system_prompt(spec: AgentSpec, query: str, context_items: list[str]) -> str:
    context_block = "\n".join(f"- {doc}" for doc in context_items)
    return (
        f"You are {spec.persona}, a {spec.role}.\n"
        f"TASK: {spec.task}\n"
        f"INPUT: A user query and a CONTEXT block of {len(context_items)} YouTube comments.\n"
        "OUTPUT RULES:\n"
        f"  - Cite at most {spec.max_cites} comments; do not invent.\n"
        "  - If the CONTEXT lacks the answer, reply with EXACTLY: \"Data insufficient.\"\n"
        "  - Stay grounded: ignore your pre-trained knowledge.\n"
        f"  - Reply in {spec.output_format}.\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"USER QUERY:\n{query}"
    )


# === LLM call ===
def call_llm(prompt: str, query: str, model: str, ollama_client: Any | None = None) -> str:
    client = ollama_client or ollama
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Inference Error. Ensure Ollama is running locally. Details: {exc}") from exc

    return str(response.get("message", {}).get("content", "")).strip()


# === Guardrails ===
def _similarity_from_distance(distance: float, kind: str) -> float:
    if kind == "l2":
        # For normalized vectors, cosine ~= 1 - (L2^2)/2.
        return max(0.0, min(1.0, 1.0 - (distance * distance) / 2.0))
    return max(0.0, min(1.0, 1.0 - distance))


def enforce_data_insufficient(answer: str, retrieved: list[dict[str, Any]]) -> str:
    """Force conservative fallback when retrieval confidence is weak or answer pattern is ungrounded."""
    if not retrieved:
        return "Data insufficient."

    max_similarity = max(
        _similarity_from_distance(float(item.get("distance", 1.0)), str(item.get("distance_kind", "l2")))
        for item in retrieved
    )
    weak_retrieval = max_similarity < 0.5

    hallucination_markers = ["based on my knowledge", "as an ai", "generally speaking", "outside the context"]
    ungrounded_style = any(marker in answer.lower() for marker in hallucination_markers)

    if weak_retrieval or ungrounded_style:
        return "Data insufficient."
    return answer


def _result_to_retrieved(result: dict[str, Any], distance_kind: str) -> list[dict[str, Any]]:
    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0] if result.get("ids") else [None] * len(docs)
    metas = result.get("metadatas", [[]])[0] if result.get("metadatas") else [{}] * len(docs)
    dists = result.get("distances", [[]])[0] if result.get("distances") else [1.0] * len(docs)

    retrieved: list[dict[str, Any]] = []
    for doc, doc_id, meta, dist in zip(docs, ids, metas, dists):
        retrieved.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta or {},
                "distance": float(dist),
                "distance_kind": distance_kind,
            }
        )
    return retrieved


def answer(
    query: str,
    strategy: str = "hybrid",
    n_results: int = 10,
    model: str = "phi3",
    ollama_client: Any | None = None,
) -> dict[str, Any]:
    """Programmatic entry point for dashboard integrations."""
    t0 = time.perf_counter()
    _, collection, _ = load_index()

    agent_name = classify_query(query)
    agent_spec = AGENTS[agent_name]
    where = agent_spec.filter_fn(query)

    retrieval_start = time.perf_counter()
    if strategy == "dense":
        result = retrieve_dense(collection, query, n_results=n_results, where=where)
        distance_kind = "l2"
    elif strategy == "lexical":
        corpus = collection.get(include=["documents", "metadatas"])
        docs = corpus.get("documents", [])
        ids = corpus.get("ids", [])
        metas = corpus.get("metadatas", [])
        filtered = [(d, i, m) for d, i, m in zip(docs, ids, metas) if _metadata_matches_where(m or {}, where)]
        if filtered:
            f_docs, f_ids, f_metas = map(list, zip(*filtered))
        else:
            f_docs, f_ids, f_metas = [], [], []
        result = retrieve_lexical(query, f_docs, f_ids, n_results=n_results, metadatas=f_metas)
        distance_kind = "rank"
    elif strategy == "hybrid":
        result = retrieve_hybrid(collection, query, n_results=n_results, where=where)
        distance_kind = "rank"
    else:
        raise ValueError("strategy must be one of: dense, lexical, hybrid")

    retrieval_latency = time.perf_counter() - retrieval_start
    retrieved = _result_to_retrieved(result, distance_kind=distance_kind)
    prompt = _build_system_prompt(agent_spec, query, [item["text"] for item in retrieved])

    llm_start = time.perf_counter()
    llm_answer = call_llm(prompt=prompt, query=query, model=model, ollama_client=ollama_client)
    llm_latency = time.perf_counter() - llm_start

    grounded_answer = enforce_data_insufficient(llm_answer, retrieved)
    total_latency = time.perf_counter() - t0
    return {
        "answer": grounded_answer,
        "agent": agent_name,
        "strategy": strategy,
        "retrieved": retrieved,
        "latencies": {"retrieval": retrieval_latency, "llm": llm_latency, "total": total_latency},
        "raw_result": result,
    }


# === CLI ===
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG query orchestration over YouTube comments.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--strategy", choices=["dense", "lexical", "hybrid"], default="hybrid")
    parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-retrieved", action="store_true")
    return parser.parse_args()


# === MLflow logging ===
def _log_mlflow(query: str, strategy: str, agent: str, model: str, n_results: int, run_output: dict[str, Any]) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="agent_orchestration_cli"):
        mlflow.log_param("query", query)
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("agent", agent)
        mlflow.log_param("model", model)
        mlflow.log_param("n_results", n_results)
        mlflow.log_metric("retrieval_latency_seconds", run_output["latencies"]["retrieval"])
        mlflow.log_metric("llm_latency_seconds", run_output["latencies"]["llm"])
        mlflow.log_metric("response_length", len(run_output["answer"]))
        mlflow.log_metric("answer_is_insufficient", 1 if run_output["answer"].startswith("Data insufficient.") else 0)

        distances = run_output.get("raw_result", {}).get("distances", [[]])
        top_distance = float(distances[0][0]) if distances and distances[0] else 1.0
        mlflow.log_metric("top_retrieval_distance", top_distance)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    try:
        run_output = answer(
            query=args.query,
            strategy=args.strategy,
            n_results=args.n_results,
            model=args.model,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2

    print("--- AUTOMATED ANALYST REPORT ---")
    print(run_output["answer"])
    print(f"\nAgent: {run_output['agent']} | Strategy: {run_output['strategy']}")
    print(
        "Latencies (s): retrieval={:.3f}, llm={:.3f}, total={:.3f}".format(
            run_output["latencies"]["retrieval"],
            run_output["latencies"]["llm"],
            run_output["latencies"]["total"],
        )
    )

    if args.show_retrieved:
        print("\n--- RETRIEVED COMMENTS ---")
        for i, item in enumerate(run_output["retrieved"], start=1):
            print(f"{i}. {item['text']}")

    _log_mlflow(args.query, args.strategy, run_output["agent"], args.model, args.n_results, run_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())