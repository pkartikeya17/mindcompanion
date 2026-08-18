import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load these globally so they don't reload on every single API request
CHUNKS_FILE = "Knowledge/metadata/knowledge_chunks.json"
INDEX_FILE = "Knowledge/embeddings/vector_db.index"

print("Loading AI Embedding Model and Vector Database...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    CHUNKS = json.load(f)

def get_relevant_context(query_text, top_k=3):
    """Searches the vector database and returns the text of the best chunks."""
    query_vector = MODEL.encode([query_text]).astype('float32')
    distances, indices = INDEX.search(query_vector, top_k)
    
    retrieved_texts = []
    for i in range(top_k):
        chunk_idx = indices[0][i]
        matched_chunk = CHUNKS[chunk_idx]
        # Format it nicely for the LLM to read
        retrieved_texts.append(f"Source: {matched_chunk['document']}\nText: {matched_chunk['text']}")
        
    # Combine all chunks into one giant string
    return "\n\n---\n\n".join(retrieved_texts)

if __name__ == "__main__":
    # Quick test to make sure it works
    test_context = get_relevant_context("I feel overwhelmed.")
    print(test_context)