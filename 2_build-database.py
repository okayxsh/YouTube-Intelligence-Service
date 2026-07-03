"""Build or incrementally update the ChromaDB index for processed YouTube comments.

Input:
- Reads `youtube_comments_PROCESSED.csv` and indexes all rows using `clean_text`
  as Chroma documents.

Output:
- Persistent ChromaDB directory at `./youtube_vector_db`.
- Collection name remains `youtube_comments` for dashboard compatibility.

Behavior:
- Incremental by default: re-runs skip already indexed IDs.
- `--reset` wipes `./youtube_vector_db` before rebuilding.
- Metadata includes filter fields (video_id, comment_id, sentiment fields,
  entities, topic_id). Entities are normalized JSON and capped at 800 chars
  to stay within practical metadata limits.

Failure modes:
- Missing input CSV raises FileNotFoundError.
- Missing schema fields are logged; rows lacking required indexing fields are
  skipped gracefully instead of crashing.
- Embedding memory pressure is mitigated via batched encoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# NOTE: requirements.txt must include `onnxruntime==1.18.1`
# (see separate task). Do NOT add the mock workaround back.

INPUT_CSV = "youtube_comments_PROCESSED.csv"
PERSIST_DIR = "./youtube_vector_db"
COLLECTION_NAME = "youtube_comments"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 500
ENCODE_BATCH_SIZE = 64

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["video_id", "comment_id", "clean_text"]
OPTIONAL_COLUMNS = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "entities",
    "topic_id",
    "keywords",
]


def load_index(collection_name: str = COLLECTION_NAME) -> tuple[Any, Any, Any]:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )
    return client, collection, embedding_function


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_entities(raw_entities: Any) -> tuple[str, bool]:
    try:
        parsed = json.loads(raw_entities if isinstance(raw_entities, str) else "[]")
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    serialized = json.dumps(parsed, ensure_ascii=False)
    if len(serialized) > 800:
        return serialized[:800], True
    return serialized, False


def _log_missing_columns(df: pd.DataFrame) -> None:
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]

    for col in missing_required:
        LOGGER.warning("Missing required column: %s (rows needing this will be skipped)", col)
    for col in missing_optional:
        LOGGER.warning("Missing optional column: %s (default value will be used)", col)


def _prepare_index_rows(
    df: pd.DataFrame, existing_ids: set[str]
) -> tuple[list[dict[str, Any]], int, int]:
    rows_to_index: list[dict[str, Any]] = []
    truncated_entities_count = 0
    skipped_existing_count = 0

    for row_index, row in df.iterrows():
        video_id = str(row.get("video_id", "")).strip()
        clean_text = str(row.get("clean_text", "")).strip()
        comment_id = str(row.get("comment_id", "")).strip()

        if not video_id or not clean_text or not comment_id:
            continue

        doc_id = f"comment_{video_id}_{row_index}"
        if doc_id in existing_ids:
            skipped_existing_count += 1
            continue

        entities_str, was_truncated = _sanitize_entities(row.get("entities", "[]"))
        if was_truncated:
            truncated_entities_count += 1

        metadata = {
            "video_id": video_id,
            "comment_id": comment_id,
            "sentiment_polarity": _safe_float(row.get("sentiment_polarity", 0.0), 0.0),
            "sentiment_subjectivity": _safe_float(row.get("sentiment_subjectivity", 0.0), 0.0),
            "entities": entities_str,
            "topic_id": _safe_int(row.get("topic_id", -1), -1),
        }

        # We intentionally exclude `keywords` from metadata to reduce metadata size
        # pressure while keeping retrieval filters focused and robust.
        rows_to_index.append({"id": doc_id, "document": clean_text, "metadata": metadata})

    return rows_to_index, truncated_entities_count, skipped_existing_count


def build_vector_database(reset: bool = False, collection_name: str = COLLECTION_NAME) -> None:
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {input_path}")

    if reset and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    LOGGER.info("Loading processed CSV: %s", INPUT_CSV)
    df = pd.read_csv(input_path)
    _log_missing_columns(df)

    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("").astype(str).str.strip()
        df = df[df["clean_text"] != ""].copy()
    else:
        df = df.iloc[0:0].copy()

    LOGGER.info("Total rows after cleaning: %s", len(df))

    _, collection, _ = load_index(collection_name=collection_name)

    LOGGER.info("Loading embedding model all-MiniLM-L6-v2...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    existing = set(collection.get(include=[]).get("ids", []))
    LOGGER.info("Collection has %s existing docs (from prior run)", len(existing))

    rows_to_index, truncated_entities_count, skipped_already_indexed = _prepare_index_rows(df, existing)
    total_candidates = len(rows_to_index)

    total_batches = (total_candidates + BATCH_SIZE - 1) // BATCH_SIZE if total_candidates else 0
    added_count = 0

    for batch_i in range(total_batches):
        start = batch_i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_candidates)
        batch = rows_to_index[start:end]

        LOGGER.info("Embedding batch %s/%s (size=%s)...", batch_i + 1, total_batches, len(batch))
        batch_docs = [item["document"] for item in batch]
        batch_embeddings = model.encode(
            batch_docs,
            show_progress_bar=True,
            batch_size=ENCODE_BATCH_SIZE,
        )

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings.tolist(),
            metadatas=[item["metadata"] for item in batch],
            ids=[item["id"] for item in batch],
        )
        added_count += len(batch)

    LOGGER.info("Added %s docs in this run, skipped %s already indexed", added_count, skipped_already_indexed)
    if truncated_entities_count > 0:
        LOGGER.warning("warning: %s entities strings truncated", truncated_entities_count)
    LOGGER.info("Vector database build complete: ./youtube_vector_db")

    LOGGER.info("Executing Test Query: 'What are the main complaints people have about the tutorial?'")
    results = collection.query(
        query_texts=["What are the main complaints people have about the tutorial?"],
        n_results=2,
    )

    LOGGER.info("Top Matches Retrieved:")
    for idx, doc in enumerate(results.get("documents", [[]])[0], start=1):
        LOGGER.info("%s. %s", idx, doc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or incrementally update the YouTube ChromaDB index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ./youtube_vector_db before rebuilding.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    build_vector_database(reset=args.reset)


if __name__ == "__main__":
    main()