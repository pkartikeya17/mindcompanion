import pymupdf as fitz  # Updated import to fix the warning
import re
import json
import os

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_pdf(file_path, chunk_size=800, overlap=150):
    doc = fitz.open(file_path)
    document_title = os.path.basename(file_path).replace('.pdf', '')
    
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text") + " "
        
    cleaned_text = clean_text(full_text)
    
    words = cleaned_text.split(' ')
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        metadata = {
            "document": document_title,
            "source_type": "clinical_guideline",
            "text": chunk_text
        }
        chunks.append(metadata)
    return chunks

if __name__ == "__main__":
    # Updated paths to match your exact folder structure
    ingestion_folder = "Knowledge/"
    output_file = "Knowledge/metadata/knowledge_chunks.json"
    
    all_chunks = []
    
    if os.path.exists(ingestion_folder):
        for filename in os.listdir(ingestion_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(ingestion_folder, filename)
                print(f"Processing: {filename}")
                chunks = chunk_pdf(file_path)
                all_chunks.extend(chunks)
                
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)
            
        print(f"Successfully processed {len(all_chunks)} total chunks.")
    else:
        print(f"Could not find folder: {ingestion_folder}")