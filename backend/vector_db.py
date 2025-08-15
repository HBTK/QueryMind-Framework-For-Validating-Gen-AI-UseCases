# File: vector_db.py

"""
vector_db.py

This module manages the storage and retrieval of conversation context embeddings
using Chroma DB. It uses a transformer model (GPT-2 in this example) to generate embeddings.
These embeddings are stored in a Chroma collection, and given a new query, the most similar
context (or contexts) are retrieved to be used as reference for context similarity validation.

Dependencies:
  - chromadb (pip install chromadb)
  - torch (pip install torch)
  - transformers (pip install transformers)
  - scikit-learn (pip install scikit-learn)  # For cosine similarity
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load GPT-2 model for embeddings
MODEL_NAME = "gpt2"

def load_model_and_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def get_embedding(text: str) -> list:
    """
    Generates an embedding for the given text using GPT-2.
    This function tokenizes the text, passes it through the model, and averages
    the last hidden states over the sequence length.
    Returns a list of floats representing the embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    embedding = last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy().tolist()

# -------------------------------
# Chroma DB Integration
# -------------------------------
import chromadb

# Create a Chroma DB client using the new configuration (no Settings)
chroma_client = chromadb.Client()

# Create or get a collection for conversation embeddings
COLLECTION_NAME = "conversation_history"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def add_context_to_db(doc_id: str, text: str):
    """
    Computes the embedding for the provided text and adds it to the Chroma collection.
    The doc_id should be unique (e.g., a timestamp or message ID).
    """
    embedding = get_embedding(text)
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )

def search_context(query: str, top_k: int = 3) -> str:
    """
    Computes an embedding for the query and searches the collection for the top_k most similar documents.
    Returns the concatenated text of the retrieved documents (if any).
    """
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    if results.get("documents"):
        docs = results["documents"][0]
        return " ".join(docs)
    return ""
