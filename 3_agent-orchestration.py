import sys
import types
import importlib.machinery

mock_ort = types.ModuleType("onnxruntime")
mock_ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
sys.modules['onnxruntime'] = mock_ort

import chromadb
from chromadb.utils import embedding_functions
import ollama

# 1. Connect to the Existing Database
print("Mounting ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./youtube_vector_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="youtube_comments", embedding_function=sentence_transformer_ef)

# 2. Define the User Query
user_query = "What are the main issues or complaints people have in these comments?"

# 3. Retrieve Context (The 'R' in RAG)
print("Retrieving semantic matches from database...")
results = collection.query(
    query_texts=[user_query],
    n_results=10 # Pull the top 10 most relevant comments
)

retrieved_comments = results['documents'][0]
context_block = "\n".join([f"- {comment}" for comment in retrieved_comments])

# 4. Construct the Strict Prompt
# This formatting dictates the model's behavior. Do not soften the constraints.
system_prompt = f"""You are a highly precise analytical engine. Your task is to answer the user's query based STRICTLY on the provided context.

RULES:
1. Do not use external knowledge or pre-trained memory.
2. If the answer is not contained within the CONTEXT, you must output exactly: "Data insufficient."
3. Synthesize the comments into a clear, direct summary.

CONTEXT:
{context_block}
"""

print("Routing context and query to local LLM inference engine...\n")

# 5. Execute Local LLM via Ollama
try:
    response = ollama.chat(model='phi3', messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': user_query}
    ])
    
    print("--- AUTOMATED ANALYST REPORT ---")
    print(response['message']['content'])
    
except Exception as e:
    print(f"Inference Error. Ensure Ollama is running locally. Details: {e}")