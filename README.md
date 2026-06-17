# YouTube Intelligence Engine: NLP & RAG Pipeline

An automated, local-first NLP pipeline designed to scrape YouTube comment threads, extract structured metadata, index observations into a vector database, and provide a context-constrained analytical interface via an LLM agent.

The entire stack is optimized to run locally on low-compute/constrained edge hardware (such as standard x86 or ARM CPUs/NPUs) without exposing data to external cloud APIs.

---

## System Architecture

### 1. Extraction (`0_extract.py`)
Connects to the YouTube v3 Data API and safely paginates through raw comment threads up to a specified target limit (e.g., 10,000 rows).

### 2. Preprocessing & Metadata Extraction (`1_preprocess.py`)
A lightweight text-cleaning and enrichment pipeline utilizing:

- **TextBlob** for sentiment analysis scoring (`-1.0` to `1.0`)
- **spaCy (`en_core_web_sm`)** for Named Entity Recognition (NER), extracting:
  - Organizations
  - Products
  - Locations
  - Individuals

### 3. Vector Database Indexing (`2_build-database.py`)
Encodes cleaned comments into 384-dimensional dense embeddings using the local **SentenceTransformer** model:

```text
all-MiniLM-L6-v2
```

The embeddings and metadata are persisted inside a local **ChromaDB** vector database.

### 4. Agent Orchestration & Local Inference (`3_agent-orchestration.py`)
Handles Retrieval-Augmented Generation (RAG) by:

1. Accepting user queries
2. Retrieving the top 10 semantically relevant records from ChromaDB
3. Constructing a strict context boundary
4. Performing local inference through **Ollama** using:

```text
phi3 (3.8B)
```

### 5. UI & Execution Logging (`4_dashboard.py`)
Provides an interactive **Streamlit** dashboard while automatically logging:

- Response latency
- Token usage
- Runtime parameters
- Execution metadata

All metrics are stored in **MLflow** for observability and experimentation tracking.

---

## Installation & Setup

### 1. Environment Initialization

Ensure **Python 3.12+** is installed.

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
- **PyTorch DLL Load Failure (`WinError 1114`):** If you run into a DLL initialization crash when importing `torch` or `spacy` on Windows, reinstall PyTorch using the CPU-only build:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
  ```
- **spaCy Model Downloader HTTP 404:** If the automatic `spacy download` command fails on Windows, install the model package directly:
  ```powershell
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
  ```


### 2. Local Inference Engine Setup

Install and start Ollama, then pull the local reasoning model:

```bash
ollama pull phi3
```

### 3. API Credentials

Create a YouTube Data API v3 key from Google Cloud Console. You can export it as an environment variable or use the local `.env` fallback.

#### Linux / macOS (Terminal):
```bash
export YOUTUBE_API_KEY="AIzaSy..."
```

#### Windows (PowerShell):
```powershell
$env:YOUTUBE_API_KEY="AIzaSy..."
```

#### Local Fallback (.env file):
Alternatively, create a `.env` file in the root project directory and add your key:
```env
YOUTUBE_API_KEY="AIzaSy..."
```

---

## Execution Pipeline

Run each module sequentially to generate the required artifacts.

### Step 1: Scrape Raw Comments

Extract comment threads from target YouTube videos.

```bash
python3 0_extract.py
```

### Step 2: Clean and Enrich Text

Generate structured metadata and NLP features.

```bash
python3 1_preprocess.py
```

### Step 3: Build the Vector Brain

Create embeddings and populate the ChromaDB persistence directory.

```bash
python3 2_build-database.py
```

Resulting database:

```text
./youtube_vector_db
```

### Step 4: Launch the Analytical Interface

Start the Streamlit dashboard.

```bash
streamlit run 4_dashboard.py
```

---

## Monitoring & Evaluation (MLflow)

To monitor system latencies, input drift, and response length metrics across multiple user queries, start the local MLflow UI server:

```bash
mlflow ui --port 5000
```

Then navigate to:

```text
http://localhost:5000
```

MLflow can be used to inspect:

- Query latency
- Retrieval performance
- Response length
- Parameter configurations
- Experiment history

---

## Evaluation Constraints (Anti-Hallucination Guardrails)

The system follows a strict retrieval-first architecture.

Incoming user prompts are matched against indexed documents using cosine similarity in vector space. Responses are constrained exclusively to retrieved context.

### Guardrail Rules

- The model must not rely on external or pre-trained world knowledge.
- Only retrieved records may be used during answer generation.
- If relevant context cannot be retrieved, no inference is attempted.
- Unsupported conclusions are prohibited.

### Fallback Behavior

When sufficient supporting evidence is unavailable, the engine returns:

```text
Data insufficient.
```

---

## Technology Stack

| Layer | Technology |
|---------|------------|
| Data Collection | YouTube Data API v3 |
| NLP Processing | TextBlob, spaCy |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Inference Engine | Ollama |
| LLM | Phi-3 (3.8B) |
| Dashboard | Streamlit |
| Observability | MLflow |

---

## Design Goals

- Fully local execution
- No external LLM APIs
- Low-resource hardware compatibility
- Retrieval-grounded responses
- Persistent semantic memory
- Reproducible experimentation
- Explainable and auditable inference workflow