"""Preprocess YouTube comments into a feature-enriched dataset for RAG ingestion.

This module reads raw comments from INPUT_CSV, cleans text, computes sentiment
features, extracts named entities, derives TF-IDF keywords, and assigns LDA
topic IDs. It writes OUTPUT_CSV plus topic sidecar artifacts used by reporting.

Inputs:
- INPUT_CSV with canonical `comment_text` from the extractor.

Outputs:
- youtube_comments_PROCESSED.csv
- topic_terms.json
- topic_summary.csv

Topic modeling choice:
- Uses sklearn LDA (instead of BERTopic) to avoid new dependencies while still
  providing stable 10-topic assignments and interpretable topic-term summaries
  required for project reporting.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from textblob import TextBlob

INPUT_CSV = "youtube_comments_10k_v2.csv"
OUTPUT_CSV = "youtube_comments_PROCESSED.csv"
TOPIC_TERMS_JSON = "topic_terms.json"
TOPIC_SUMMARY_CSV = "topic_summary.csv"
NUM_TOPICS = 10

ENTITY_LABELS = {"ORG", "PERSON", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART", "FAC"}

LOGGER = logging.getLogger(__name__)


def clean_text(text: Any) -> str:
    """Strip URLs and unwanted characters while preserving quotes/apostrophes/hyphens."""
    normalized = str(text)
    normalized = re.sub(r"http\S+", "", normalized)
    normalized = re.sub(r"[^\w\s.,!?'\"\-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def sentiment_scores(text: str) -> tuple[float, float]:
    sentiment = TextBlob(text).sentiment
    return float(sentiment.polarity), float(sentiment.subjectivity)


def extract_entities(text: str, nlp: Any) -> list[tuple[str, str]]:
    """Return unique `(text, label)` entities for required labels, preserving order."""
    doc = nlp(text)
    seen: set[tuple[str, str]] = set()
    entities: list[tuple[str, str]] = []

    for ent in doc.ents:
        pair = (ent.text.strip(), ent.label_)
        if ent.label_ in ENTITY_LABELS and pair[0] and pair not in seen:
            seen.add(pair)
            entities.append(pair)

    return entities


def format_entities(entities: list[tuple[str, str]]) -> str:
    payload = [{"text": text, "label": label} for text, label in entities]
    return json.dumps(payload, ensure_ascii=False)


def extract_keywords_tfidf(corpus: list[str], top_k: int = 5) -> list[list[str]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        LOGGER.warning(
            "scikit-learn is not installed; skipping TF-IDF keywords. Install with: pip install scikit-learn"
        )
        return [[] for _ in corpus]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    keywords_per_doc: list[list[str]] = []
    for row_idx in range(matrix.shape[0]):
        row = matrix.getrow(row_idx)
        if row.nnz == 0:
            keywords_per_doc.append([])
            continue

        scored = sorted(zip(row.indices, row.data), key=lambda pair: pair[1], reverse=True)
        top_terms = [terms[col_idx] for col_idx, _ in scored[:top_k]]
        keywords_per_doc.append(top_terms)

    return keywords_per_doc


def extract_topic_ids(
    corpus: list[str],
    num_topics: int = NUM_TOPICS,
) -> tuple[list[int], dict[int, list[str]], pd.DataFrame]:
    """Train corpus-level LDA and return topic IDs plus interpretable topic summaries.

    LDA is used over BERTopic to avoid adding dependencies while still producing
    stable topic clusters and top-term interpretations suitable for reporting.
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        LOGGER.warning(
            "scikit-learn is not installed; skipping topic modeling. Install with: pip install scikit-learn"
        )
        topic_ids = [-1 for _ in corpus]
        empty_summary = pd.DataFrame(
            columns=["topic_id", "doc_count", "top_10_terms", "interpretation_label"]
        )
        return topic_ids, {}, empty_summary

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=5000,
    )
    doc_term_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=20,
    )
    doc_topic = lda.fit_transform(doc_term_matrix)
    topic_ids = doc_topic.argmax(axis=1).astype(int).tolist()

    topic_terms: dict[int, list[str]] = {}
    for topic_id, topic_vector in enumerate(lda.components_):
        top_indices = topic_vector.argsort()[-10:][::-1]
        topic_terms[topic_id] = [feature_names[idx] for idx in top_indices]

    summary_records: list[dict[str, Any]] = []
    counts = pd.Series(topic_ids).value_counts().to_dict()
    for topic_id in range(num_topics):
        terms = topic_terms.get(topic_id, [])
        interpretation_label = "_".join(terms[:3]) if terms else "unassigned_topic"
        summary_records.append(
            {
                "topic_id": topic_id,
                "doc_count": int(counts.get(topic_id, 0)),
                "top_10_terms": json.dumps(terms, ensure_ascii=False),
                "interpretation_label": interpretation_label,
            }
        )

    topic_summary = pd.DataFrame(summary_records)
    return topic_ids, topic_terms, topic_summary


def _load_spacy_model() -> Any | None:
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except OSError:
        LOGGER.warning(
            "spaCy model en_core_web_sm is not installed. Install with: python -m spacy download en_core_web_sm. Proceeding without NER."
        )
        return None
    except ImportError:
        LOGGER.warning("spaCy is not installed. Install with: pip install spacy. Proceeding without NER.")
        return None


def run(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    topic_terms_json: str = TOPIC_TERMS_JSON,
    topic_summary_csv: str = TOPIC_SUMMARY_CSV,
) -> None:
    LOGGER.info("Loading input")
    df = pd.read_csv(input_csv)
    if "comment_text" not in df.columns:
        raise ValueError("Input CSV must contain 'comment_text' column.")

    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    LOGGER.info("Cleaning text")
    df["clean_text"] = df["comment_text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""].copy()

    LOGGER.info("Sentiment scoring")
    sentiment_df = df["clean_text"].apply(lambda text: pd.Series(sentiment_scores(text)))
    sentiment_df.columns = ["sentiment_polarity", "sentiment_subjectivity"]
    df[["sentiment_polarity", "sentiment_subjectivity"]] = sentiment_df

    LOGGER.info("NER")
    nlp = _load_spacy_model()
    if nlp is None:
        df["entities"] = "[]"
    else:
        df["entities"] = df["clean_text"].apply(lambda text: format_entities(extract_entities(text, nlp)))

    LOGGER.info("Keyword extraction (TF-IDF)")
    keywords = extract_keywords_tfidf(df["clean_text"].tolist(), top_k=5)
    df["keywords"] = [json.dumps(items, ensure_ascii=False) for items in keywords]

    LOGGER.info("Topic modeling (LDA, k=10)")
    topic_ids, topic_terms, topic_summary = extract_topic_ids(df["clean_text"].tolist(), num_topics=NUM_TOPICS)
    df["topic_id"] = topic_ids

    LOGGER.info("Saving outputs")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    topic_terms_path = Path(topic_terms_json)
    topic_terms_path.write_text(
        json.dumps({str(k): v for k, v in topic_terms.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    topic_summary.to_csv(topic_summary_csv, index=False, encoding="utf-8-sig")

    LOGGER.info("Done")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    run()


if __name__ == "__main__":
    main()