# config.py

"""
Central configuration file for the Fintech Founder Finder project.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- File Paths ---
# Input files
INPUT_CSV_ORIGINAL = 'combined_institutions.csv'
COOKIES_FILE = 'cookies.json'

# Output/processed files
OUTPUT_CSV_LINKEDIN = 'institutions_linkedin.csv'
FAISS_INDEX_PATH = "vectorstorage/linkedin_data.index"
METADATA_PATH = "vectorstorage/linkedin_metadata.json"
ANALYSIS_OUTPUT_DIR = "analysisJsons" # New directory for JSON outputs

# --- Scraping Parameters ---
# Number of LinkedIn posts to scrape per company
NO_OF_POSTS_TO_SCRAPE = 10
# Number of news articles to scrape per company
NO_OF_NEWS_ARTICLES_TO_SCRAPE = 2
# Number of website pages to scrape per company
NO_OF_WEBSITE_PAGES_TO_SCRAPE = 5
# Number of companies to sample for processing in each run
# Set to None to process all companies
SAMPLE_SIZE = 5

# --- Vectorization ---
# Updated to use FinBERT for more accurate financial text embeddings
EMBEDDING_MODEL = 'ProsusAI/finbert'

# --- LLM Models ---
# Model for URL finding and analysis
LLM_MODEL = "gpt-4o-mini"

# --- News Scraping ---
# List of credible news sources for the Brave Search API query
CREDIBLE_NEWS_SITES = [
    "difc.ae", "fintechfutures.com", "fintechnews.ae", "mea-finance.com",
    "thefinanceworld.com", "zawya.com", "dubaiinvestments.com",
    "emirates247.com", "khaleejtimes.com"
]
