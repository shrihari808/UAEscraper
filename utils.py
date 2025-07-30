# utils.py

"""
Utility functions for data handling, API clients, and the LangChain vector store.
"""

import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import config

# --- Data Loading ---
def load_and_clean_companies(csv_path):
    """Loads and cleans company data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        df['Cleaned Name'] = df['Institution Name'].str.replace(
            r'\s*\b(P\.J\.S\.C|PJSC|P\.S\.C|PSC|L\.L\.C|LLC|FZ|DMCC|F\.Z|PLC|Limited)\b.*', '', regex=True
        )
        df['Cleaned Name'] = df['Cleaned Name'].str.replace(r'[.&]', '', regex=True).str.strip()
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None

def load_enriched_data(csv_path):
    """Loads the CSV file containing verified LinkedIn URLs."""
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['linkedin_url'])
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Please run Step 1 first.")
        return None

# --- API Clients & Models ---
def get_openai_client():
    """Initializes and returns the OpenAI client for non-LangChain use (e.g., URL finder)."""
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found in .env file.")
        return None
    # This is for direct openai library usage, not langchain
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY)

def get_llm():
    """Initializes and returns the LangChain ChatOpenAI model."""
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found in .env file.")
        return None
    return ChatOpenAI(model=config.LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

# --- Vector Store Management (LangChain Implementation) ---
def get_vector_store():
    """
    Initializes and returns a LangChain FAISS vector store.
    It will load from disk if it exists, otherwise it will create it.
    """
    print("🧠 Initializing LangChain vector store...")
    # Using a local, open-source model for embeddings is cost-effective.
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Create the directory if it doesn't exist
    directory = os.path.dirname(config.FAISS_INDEX_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"   -> Created directory: {directory}")

    if os.path.exists(config.FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Knowledge base loaded from disk. Contains {vector_store.index.ntotal} vectors.")
    else:
        # This case is for initialization; the index is created when documents are first added.
        print("   -> No existing knowledge base found. A new one will be created upon adding documents.")
        # We return an empty placeholder that will be populated by the scraper.
        # A dummy text and embedding is needed to initialize the FAISS index.
        vector_store = FAISS.from_texts(["init"], embeddings)

    return vector_store
