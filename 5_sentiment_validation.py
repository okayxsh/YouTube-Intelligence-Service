"""Sentiment validation for the YouTube Intelligence Engine.

This script evaluates sentiment labeling consistency on a held-out sample from
`youtube_comments_PROCESSED.csv` using two strategies:
1) TextBlob threshold labels (weak labels from existing pipeline polarity)
2) VADER labels as an independent baseline

It reports inter-model agreement, Cohen's kappa, and confusion matrix. This is
not ground-truth accuracy, but an inter-model agreement audit that is suitable
for evaluation monitoring in coursework settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

EVAL_DIR = "/workspace/project/eval"
RANDOM_SEED = 42
N_SAMPLES_SENTIMENT = 200
N_QUERIES_RETRIEVAL = 20
TOP_K = 10

INPUT_CSV = "youtube_comments_PROCESSED.csv"
LOGGER = logging.getLogger(__name__)


def _label_from_polarity(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def _vader_label(compound_score: float) -> str:
    if compound_score > 0.1:
        return "positive"
    if compound_score < -0.1:
        return "negative"
    return "neutral"


def _load_vader_analyzer() -> Any:
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment baseline unavailable. Install nltk and ensure vader_lexicon is available."
        ) from exc

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unable to load VADER lexicon. Run: python -m nltk.downloader vader_lexicon"
            ) from exc


def run_validation(n: int = N_SAMPLES_SENTIMENT, seed: int = RANDOM_SEED) -> dict[str, Any]:
    os.makedirs(EVAL_DIR, exist_ok=True)
    eval_dir = Path(EVAL_DIR)

    LOGGER.info("Loading processed CSV: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    required_cols = ["clean_text", "sentiment_polarity", "comment_text"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["clean_text", "sentiment_polarity"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    if df.empty:
        raise ValueError("No rows available after cleaning for sentiment validation.")

    sample_n = min(n, len(df))
    sampled = df.sample(n=sample_n, random_state=seed).copy()

    sampled["textblob_label"] = sampled["sentiment_polarity"].astype(float).apply(_label_from_polarity)

    analyzer = _load_vader_analyzer()
    sampled["vader_compound"] = sampled["clean_text"].apply(
        lambda text: float(analyzer.polarity_scores(text).get("compound", 0.0))
    )
    sampled["vader_label"] = sampled["vader_compound"].apply(_vader_label)

    agreement_rate = float((sampled["textblob_label"] == sampled["vader_label"]).mean())

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for Cohen's kappa computation.") from exc

    kappa = float(cohen_kappa_score(sampled["textblob_label"], sampled["vader_label"]))

    labels_order = ["positive", "neutral", "negative"]
    confusion = pd.crosstab(
        sampled["textblob_label"],
        sampled["vader_label"],
        rownames=["textblob_label"],
        colnames=["vader_label"],
        dropna=False,
    ).reindex(index=labels_order, columns=labels_order, fill_value=0)

    dist = sampled["textblob_label"].value_counts()
    metrics = {
        "agreement_rate": agreement_rate,
        "kappa": kappa,
        "n_sampled": int(sample_n),
        "positive_pct": float(dist.get("positive", 0) / sample_n),
        "neutral_pct": float(dist.get("neutral", 0) / sample_n),
        "negative_pct": float(dist.get("negative", 0) / sample_n),
    }

    metrics_path = eval_dir / "sentiment_validation_metrics.json"
    confusion_path = eval_dir / "sentiment_confusion_matrix.csv"
    sample_path = eval_dir / "sentiment_sample_20.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion.to_csv(confusion_path, encoding="utf-8-sig")

    sample_20 = sampled.sample(n=min(20, len(sampled)), random_state=seed)[
        ["comment_text", "textblob_label", "vader_label", "sentiment_polarity", "vader_compound"]
    ]
    sample_20.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print("Sentiment validation summary:")
    print(f"N sampled: {sample_n}")
    print(f"Agreement rate (VADER vs TextBlob): {agreement_rate:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print("Confusion matrix (rows=TextBlob, cols=VADER):")
    print(confusion.to_string())
    print(
        "Distribution of predicted polarities (TextBlob): "
        f"pos={int(dist.get('positive', 0))}, "
        f"neu={int(dist.get('neutral', 0))}, "
        f"neg={int(dist.get('negative', 0))}"
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sentiment label agreement (TextBlob vs VADER).")
    parser.add_argument("--n", type=int, default=N_SAMPLES_SENTIMENT, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    run_validation(n=args.n, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
