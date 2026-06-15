import pandas as pd
import spacy
from textblob import TextBlob
import re

# 1. Load the lightweight NLP model
print("Loading spaCy NLP model...")
nlp = spacy.load("en_core_web_sm")

# 2. Load your scraped data
print("Loading raw comments...")
# Replace with your actual filename if different
df = pd.read_csv("youtube_comments_10k.csv") 

# Drop empty comments to prevent errors
df = df.dropna(subset=['comment_text'])

# --- CORE FUNCTIONS ---

def clean_text(text):
    """Strips URLs, special characters, and extra spaces."""
    text = str(text)
    text = re.sub(r"http\S+", "", text) # Remove links
    text = re.sub(r"[^\w\s.,!?]", "", text) # Remove emojis and weird symbols
    return text.strip()

def get_sentiment(text):
    """Returns a score from -1.0 (Negative) to 1.0 (Positive)."""
    return TextBlob(str(text)).sentiment.polarity

def extract_entities(text):
    """Finds proper nouns, organizations, and products (NER)."""
    doc = nlp(str(text))
    # Extract entities and their labels (e.g., 'Apple': 'ORG')
    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'GPE']]
    return ", ".join(entities)

# --- EXECUTION PIPELINE ---
print("Cleaning text...")
df['clean_text'] = df['comment_text'].apply(clean_text)

print("Calculating sentiment scores...")
df['sentiment'] = df['clean_text'].apply(get_sentiment)

print("Extracting named entities (NER)...")
df['entities'] = df['clean_text'].apply(extract_entities)

# 3. Save the processed data
output_file = "youtube_comments_PROCESSED.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\nPhase 1 Complete. Processed data saved to: {output_file}")
print("\nPreview of new structured data:")
print(df[['clean_text', 'sentiment', 'entities']].head())