import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama
import mlflow
import time

# --- MLFLOW SETUP ---
# Create an experiment tracker
mlflow.set_experiment("YouTube_Intelligence_Engine")

# --- UI SETUP ---
st.set_page_config(page_title="YouTube AI Analyst", layout="centered")
st.title("YouTube Intelligence Engine")
st.markdown("Ask questions about the scraped YouTube data. The AI will retrieve relevant comments and summarize them.")

# --- DATABASE CONNECTION ---
@st.cache_resource
def load_database():
    chroma_client = chromadb.PersistentClient(path="./youtube_vector_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return chroma_client.get_collection(name="youtube_comments", embedding_function=sentence_transformer_ef)

collection = load_database()

# --- USER INPUT ---
user_query = st.text_input("Enter your query:", placeholder="E.g., What are the main positive highlights?")

if st.button("Generate Insight"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving context and generating report..."):
            
            # Start tracking this specific run
            with mlflow.start_run():
                start_time = time.time()
                
                # 1. Retrieval
                results = collection.query(query_texts=[user_query], n_results=10)
                retrieved_comments = results['documents'][0]
                context_block = "\n".join([f"- {comment}" for comment in retrieved_comments])
                
                # 2. Prompt Formatting
                system_prompt = f"""You are a highly precise analytical engine. Answer the user's query STRICTLY based on the provided context.
                RULES:
                1. No external knowledge.
                2. If context lacks the answer, state: "Data insufficient."
                CONTEXT:
                {context_block}"""
                
                # 3. LLM Inference
                try:
                    response = ollama.chat(model='phi3', messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_query}
                    ])
                    answer = response['message']['content']
                    
                    # 4. Display Results
                    st.subheader("Automated Analyst Report")
                    st.write(answer)
                    
                    with st.expander("View Retrieved Source Comments"):
                        for comment in retrieved_comments:
                            st.markdown(f"- {comment}")
                            
                    # 5. Log Metrics to MLflow
                    execution_time = time.time() - start_time
                    mlflow.log_param("model", "phi3")
                    mlflow.log_param("query", user_query)
                    mlflow.log_metric("execution_time_seconds", execution_time)
                    mlflow.log_metric("response_length", len(answer))
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")