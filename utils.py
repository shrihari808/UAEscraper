# utils.py

"""
Utility functions for data handling, API clients, and the LangChain vector store.
"""

import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
    """Loads the CSV file containing all company data and URLs."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Please run Step 1 (--find-urls) first.")
        return None

# --- API Clients & Models ---
def get_openai_client():
    """Initializes and returns the OpenAI client for non-LangChain use (e.g., URL finder)."""
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found in .env file.")
        return None
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY)

def get_llm():
    """Initializes and returns the LangChain ChatOpenAI model."""
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found in .env file.")
        return None
    return ChatOpenAI(model=config.LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

# --- Vector Store Management (LangChain Implementation) ---
def get_vector_store(index_name: str):
    """
    Initializes and returns a specific LangChain FAISS vector store by name.
    It will load from disk if it exists, otherwise it will create a new one.
    """
    if index_name not in config.FAISS_INDEX_PATHS:
        print(f"❌ Error: Index name '{index_name}' not found in config.")
        return None

    index_path = config.FAISS_INDEX_PATHS[index_name]
    print(f"🧠 Initializing vector store '{index_name}' from path: {index_path}")
    
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    index_file_path = os.path.join(index_path, "index.faiss")

    if os.path.exists(index_file_path):
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Knowledge base for '{index_name}' loaded from disk. Contains {vector_store.index.ntotal} vectors.")
    else:
        print(f"   -> No existing knowledge base found for '{index_name}'. A new one will be created.")
        os.makedirs(index_path, exist_ok=True)
        # Create an empty store to be populated later
        vector_store = FAISS.from_texts(["init"], embeddings)
        vector_store.delete([vector_store.index_to_docstore_id[0]])

    return vector_store

def load_all_vector_stores():
    """
    Loads all existing FAISS vector stores defined in the config.
    Used by the analysis engine to query across all data sources.
    """
    print("🧠 Loading all available vector stores for analysis...")
    stores = {}
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    for index_name, index_path in config.FAISS_INDEX_PATHS.items():
        index_file_path = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file_path):
            try:
                stores[index_name] = FAISS.load_local(
                    index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"  -> Successfully loaded index '{index_name}'")
            except Exception as e:
                print(f"  -> ⚠️  Could not load index '{index_name}': {e}")
        else:
            print(f"  -> ℹ️  Index for '{index_name}' not found, skipping.")
    
    if not stores:
        print("❌ No vector stores were loaded. Analysis will not have any context.")
    
    return stores
