import pandas as pd
import pdfplumber
import faiss
import numpy as np
import openai
import os
from groq import Groq

EMBED_DIM = 1536  # For OpenAI ada-002 embeddings

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string(index=False)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500):
    # Simple chunking by words
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    # Use Hugging Face embedding model via Groq API
    resp = client.embeddings.create(
        input=[text],
        model="nomic-ai/nomic-embed-text-v1"
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_faiss_index(chunks):
    index = faiss.IndexFlatL2(EMBED_DIM)
    embeddings = np.vstack([get_embedding(chunk) for chunk in chunks])
    index.add(embeddings)
    return index, embeddings

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)