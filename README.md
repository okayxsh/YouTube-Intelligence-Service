# YouTube Intelligence Engine

Local-first NLP and RAG pipeline for YouTube comments. The project scrapes tutorial/data-science comment threads, enriches them with lightweight NLP, stores them in ChromaDB, and exposes a retrieval-grounded Streamlit dashboard with MLflow telemetry.

## System Architecture

### 1. Extraction (`0_extract.py`)
Connects to the YouTube v3 Data API and safely paginates raw comment threads across 5 curated tutorial / data-science video IDs up to the target row count (10,000). Videos: cYwioeHu_OU, Lfzu74XDyco, TiS6vnju_mI, QOcP5OvSwlI, dQw4w9WgXcQ. Pagination uses `nextPageToken` with `maxResults=100` per page. HTTP 403 (comments disabled) is skipped gracefully. Output: `youtube_comments_10k_v2.csv` (about 12,011 rows in the current dataset build).

### 2. Preprocessing & Metadata Extraction (`1_preprocess.py`)
Lightweight text-cleaning and enrichment pipeline producing `youtube_comments_PROCESSED.csv` with columns including `clean_text`, `sentiment` (TextBlob polarity in `[-1, 1]`), and `entities` (spaCy NER over ORG, PERSON, PRODUCT, GPE). Optionally produces `topic_terms.json` and `topic_summary.csv` sidecars if the topic modeling stage is enabled.

### 3. Vector Database Indexing (`2_build-database.py`)
Encodes `clean_text` into 384-dimensional dense embeddings using `sentence-transformers/all-MiniLM-L6-v2` and persists them into a local ChromaDB collection named `youtube_comments` under `./youtube_vector_db`. Metadata stored alongside each vector includes sentiment score and extracted entities (JSON-shaped string, truncated at 800 characters to stay under ChromaDB's per-value limit). Current shipping configuration indexes the first 500 rows of the processed CSV to keep embedding time tractable on CPU. Lifting this cap to the full corpus is a planned iteration (see `CHANGELOG.md` and Section 6 of the report).

### 4. Agent Orchestration & Local Inference (`3_agent-orchestration.py`)
Implements retrieval-augmented generation (RAG):

1. The user query is embedded with `all-MiniLM-L6-v2`.
2. The top-10 most relevant comments are retrieved from ChromaDB by cosine similarity.
3. Retrieved comments are formatted into a strict context block.
4. Local inference runs through Ollama with `phi3` (Phi-3 3.8B) and a context-bounded system prompt.

If the retrieved context is too weak, the prompt instructs the model to reply with the exact string `Data insufficient.` rather than confabulate. This is a strict retrieval-first guardrail that mitigates hallucination. The current shipping version uses a single dense retrieval path; a hybrid BM25 + dense router is planned for a later iteration (Section 6.2.3 of the report).

### 5. UI & Execution Logging (`4_dashboard.py`)
Interactive Streamlit dashboard that accepts a natural language query, runs the RAG pipeline, displays the generated analytical report alongside the retrieved source comments in an expandable panel, and logs every interaction to MLflow (experiment `YouTube_Intelligence_Engine`):

- params: `model`, `query`
- metrics: `execution_time_seconds`, `response_length`

Launch locally with: `streamlit run 4_dashboard.py`.

## Monitoring & Evaluation (MLflow)

To monitor system latencies, input drift, and response length across multiple user queries, start the local MLflow UI server:

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000.

MLflow logs the following per dashboard query:

- params: `model`, `query`
- metrics: `execution_time_seconds`, `response_length`

MLflow is used to inspect query latency, response length, parameter configurations, and experiment history. The local SQLite backend (`mlflow.db`) is gitignored.

## Evaluation Constraints (Anti-Hallucination Guardrails)

The system follows a strict retrieval-first architecture. User prompts are matched against indexed documents using cosine similarity in vector space; responses are constrained exclusively to retrieved context.

Guardrail rules:

- The model must not rely on external or pre-trained world knowledge.
- Only retrieved records may be used during answer generation.
- If relevant context cannot be retrieved, no inference is attempted.
- Unsupported conclusions are prohibited.

Fallback behavior: when sufficient supporting evidence is unavailable, the engine returns exactly:

```text
Data insufficient.
```

## Installation & Setup

### 1. Environment Initialization

Ensure Python 3.12+ is installed.

```bash
git clone https://github.com/YOUR_USERNAME/youtube-intelligence-engine.git
cd youtube-intelligence-engine

python3 -m venv .project
source .project/bin/activate

# Windows
# .project\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Windows Installation Notes & Troubleshooting

- PyTorch DLL load failure (`WinError 1114`): if importing `torch` or `spacy` crashes on Windows, reinstall PyTorch using the CPU-only build.
- spaCy model downloader HTTP 404: if `spacy download` fails, install the model package directly from the published wheel.

### 2. Local Inference Engine Setup

Install and start Ollama, then pull the local reasoning model:

```bash
ollama pull phi3
```

Also install the text corpora used by the NLP steps:

```bash
python -m textblob.download_corpora
```

If you use the optional VADER cross-validation in `5_sentiment_validation.py`, also run:

```bash
python -m nltk.downloader vader_lexicon
```

### 3. API Credentials

Create a YouTube Data API v3 key from Google Cloud Console. You can export it as an environment variable or use the local `.env` fallback.

#### Linux / macOS (Terminal)

```bash
export YOUTUBE_API_KEY="AIzaSy..."
```

#### Windows (PowerShell)

```powershell
$env:YOUTUBE_API_KEY="AIzaSy..."
```

#### Local Fallback (`.env` file)

Alternatively, create a `.env` file in the root project directory and add your key:

```env
YOUTUBE_API_KEY="AIzaSy..."
```

## Execution Pipeline

Run each module sequentially to generate the required artifacts.

### Step 1: Scrape Raw Comments

```bash
python3 0_extract.py
```

### Step 2: Clean and Enrich Text

```bash
python3 1_preprocess.py
```

### Step 3: Build the Vector Brain

```bash
python3 2_build-database.py
```

Resulting database:

```text
./youtube_vector_db
```

### Step 4: Launch the Analytical Interface

```bash
streamlit run 4_dashboard.py
```

## Repository Layout

```text
.
├── 0_extract.py              # Stage 1 — scrape YouTube comments
├── 1_preprocess.py           # Stage 2 — clean, sentiment, NER
├── 2_build-database.py       # Stage 3 — ChromaDB indexing
├── 3_agent-orchestration.py  # Stage 4 — RAG pipeline
├── 4_dashboard.py            # Stage 5 — Streamlit + MLflow
├── 5_sentiment_validation.py # Eval — TextBlob vs VADER
├── 6_evaluate_retrieval.py   # Eval — recall@10 across strategies
├── report_stats.py           # Print run summary for the report
├── mlflow.db                 # MLflow SQLite backend (gitignored)
├── youtube_vector_db/        # ChromaDB persistence (gitignored)
├── requirements.txt
├── setup.sh                  # One-shot environment bootstrapper
├── .env.example              # YOUTUBE_API_KEY template
├── architecture.png          # System architecture diagram
├── CHANGELOG.md              # Iteration history
└── README.md
```

Note: `5_sentiment_validation.py` and `6_evaluate_retrieval.py` are expected to be added by the next iteration; the report already acknowledges them as planned work.

## Technology Stack

| Layer | Technology |
| --- | --- |
| Data Collection | YouTube Data API v3 |
| NLP Processing | TextBlob, spaCy |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Inference Engine | Ollama |
| LLM | Phi-3 (3.8B) |
| Dashboard | Streamlit |
| Observability | MLflow |

## Design Goals

- Fully local execution
- No external LLM APIs
- Low-resource hardware compatibility
- Retrieval-grounded responses
- Persistent semantic memory
- Reproducible experimentation
- Explainable and auditable inference workflow