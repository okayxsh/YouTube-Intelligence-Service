#!/usr/bin/env bash
# One-shot environment bootstrapper for the YouTube Intelligence Engine.
# Run on a fresh machine after cloning: bash setup.sh
set -euo pipefail

echo "==> Creating virtual environment in .project ..."
python3 -m venv .project
# shellcheck disable=SC1091
source .project/bin/activate

echo "==> Installing Python dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading spaCy model ..."
python -m spacy download en_core_web_sm

echo "==> Downloading TextBlob corpora ..."
python -m textblob.download_corpora

echo "==> (Optional) Downloading NLTK VADER lexicon ..."
python -m nltk.downloader vader_lexicon || echo "  skipped — install when needed"

echo "==> Reminder: pull Ollama model ..."
echo "    ollama pull phi3"

echo "==> Reminder: set your YouTube API key ..."
echo "    cp .env.example .env  &&  edit .env"

echo "==> Setup complete."
