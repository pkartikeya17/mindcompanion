import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_vector_db():
    # Paths based on your current setup
    chunks_file = "Knowledge/metadata/knowledge_chunks.json"
    index_dir = "Knowledge/embeddings"
    index_file = os.path.join(index_dir, "vector_db.index")
    
    os.makedirs(index_dir, exist_ok=True)
    
    # 1. Load the text chunks
    print("Loading chunks from JSON...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    texts = [chunk['text'] for chunk in chunks]
    print(f"Loaded {len(texts)} chunks.")
    
    # 2. Load the Embedding Model
    print("Loading embedding model (this will download ~80MB the first time)...")
    # We use a highly efficient open-weight model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Create Embeddings
    print("Generating embeddings... This may take a minute or two.")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 4. Create the FAISS Vector Database
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]  # The mathematical size of the vectors
    index = faiss.IndexFlatL2(dimension)
    
    # FAISS requires float32 data types
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    
    # 5. Save the database to disk
    faiss.write_index(index, index_file)
    print(f"Successfully saved vector database to {index_file}!")

if __name__ == "__main__":
    create_vector_db()