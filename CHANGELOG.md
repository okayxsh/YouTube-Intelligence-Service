# Changelog
All notable changes to the YouTube Intelligence Engine are recorded here.
This file supports the project rubric's "continuous improvement" criterion.

## [v0.5] — current shipping version
### Added
- 5-stage modular pipeline (0_extract → 4_dashboard)
- Local RAG via Ollama phi3 + ChromaDB dense retrieval
- MLflow telemetry (latency, response length) per query
- Streamlit dashboard with expandable source-comment panel

### Acknowledged limitations (see Section 6 of the report)
- Only the first 500 of 12,011 comments indexed for retrieval
- Single dense retrieval (no BM25 / hybrid)
- Single prompt template (no agent routing)
- Topic modeling, keyword extraction not yet implemented

## [v1.0] — planned
- Full 12k corpus indexing via batched embedding
- Hybrid retrieval (dense + BM25 fused via Reciprocal Rank Fusion)
- 4-agent query classifier (sentiment / entity / topic / general)
- Topic modeling (LDA) on clean_text, topic_id in ChromaDB metadata
- TF-IDF keyword extraction column
- 5_sentiment_validation.py + 6_evaluate_retrieval.py
