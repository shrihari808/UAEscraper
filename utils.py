# utils.py

"""
Utility functions for data handling, API clients, and the LangChain vector store.
"""

import pandas as pd
import os
import re # Import re module
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import config

# --- REVISED: Centralized Cleaning Function ---
def clean_company_name(name):
    """
    Applies a standard, more robust cleaning process to a company name
    to ensure consistency across scraping and analysis.
    """
    if not isinstance(name, str):
        return ""
    # ADDED 'Company' to the list of suffixes to be removed by the regex.
    # This is the key fix.
    suffixes_to_remove = r'\s*\b(P\.J\.S\.C|PJSC|P\.S\.C|PSC|L\.L\.C|LLC|FZ|DMCC|F\.Z|PLC|Limited|Company)\b.*'
    name = re.sub(suffixes_to_remove, '', name, flags=re.IGNORECASE)
    # Remove special characters and extra whitespace
    name = re.sub(r'[.&,]', '', name)
    return name.strip()

# --- Data Loading ---
def load_and_clean_companies(csv_path):
    """Loads and cleans company data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Use the revised centralized cleaning function
        df['Cleaned Name'] = df['Institution Name'].apply(clean_company_name)
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
def get_vector_store():
    """
    Initializes and returns a LangChain FAISS vector store.
    It will load from disk if it exists, otherwise it will create a new one.
    """
    print("🧠 Initializing LangChain vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    index_file_path = os.path.join(config.FAISS_INDEX_PATH, "index.faiss")

    if os.path.exists(index_file_path):
        vector_store = FAISS.load_local(
            config.FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Knowledge base loaded from disk. Contains {vector_store.index.ntotal} vectors.")
    else:
        print("   -> No existing knowledge base found. A new one will be created upon adding documents.")
        os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
        vector_store = FAISS.from_texts(["init"], embeddings)
        vector_store.delete([vector_store.index_to_docstore_id[0]])

    return vector_store
