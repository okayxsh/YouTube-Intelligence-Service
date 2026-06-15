import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize Local Database
print("Initializing ChromaDB storage...")
# This creates a folder called 'youtube_vector_db' to save your database persistently
chroma_client = chromadb.PersistentClient(path="./youtube_vector_db")

# 2. Configure the Embedding Model
# This downloads a lightweight, highly accurate model to your local machine
print("Loading embedding model (all-MiniLM-L6-v2)...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Create a Database Collection (like a SQL table)
# We use get_or_create so we don't crash if it already exists
collection = chroma_client.get_or_create_collection(
    name="youtube_comments",
    embedding_function=sentence_transformer_ef
)

# 4. Load the Processed Data
print("Loading processed CSV...")
df = pd.read_csv("youtube_comments_PROCESSED.csv")

# Ensure there are no empty values in clean_text
df = df.dropna(subset=['clean_text'])

# For the initial test, we will limit it to the first 500 rows. 
# Once verified, you can remove .head(500) to process all 10,000.
df_sample = df.head(500).copy()

# 5. Prepare Data for Insertion
# ChromaDB requires lists of IDs, Documents (text), and Metadatas (dictionaries)
documents = df_sample['clean_text'].astype(str).tolist()
ids = [f"comment_{i}" for i in df_sample.index]

# Package the sentiment and entities as metadata so we can filter by them later
metadatas = []
for _, row in df_sample.iterrows():
    metadatas.append({
        "sentiment": float(row['sentiment']),
        "entities": str(row['entities'])
    })

# 6. Execute Vectorization and Storage
print(f"Embedding and storing {len(documents)} comments. This will utilize local CPU/NPU resources...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("\nPhase 2 Complete. Database successfully built and saved to ./youtube_vector_db")

# --- VERIFICATION TEST ---
print("\nExecuting Test Query: 'Where can I find the data files?'")
results = collection.query(
    query_texts=["Where can I find the data files?"],
    n_results=2 # Return top 2 closest matches
)

print("\nTop Matches Retrieved:")
for idx, doc in enumerate(results['documents'][0]):
    print(f"{idx + 1}. {doc}")