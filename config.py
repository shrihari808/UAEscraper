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

# --- Rate Limiting & Timeouts ---
# Delay in seconds between Brave API calls to avoid hitting rate limits.
BRAVE_API_RATE_LIMIT = 1 
# Timeout in seconds for Selenium to wait for a page to load.
SELENIUM_PAGE_LOAD_TIMEOUT = 15

# --- File Paths ---
# Input files
INPUT_CSV_ORIGINAL = 'combined_institutions.csv'
COOKIES_FILE = 'cookies.json'

# Output/processed files
OUTPUT_CSV_LINKEDIN = 'institutions_linkedin.csv'
FAISS_INDEX_PATH = "vectorstorage/linkedin_data.index"
ANALYSIS_OUTPUT_DIR = "analysisJsons"

# --- Scraping Parameters ---
NO_OF_POSTS_TO_SCRAPE = 10
NO_OF_NEWS_ARTICLES_TO_SCRAPE = 2
NO_OF_WEBSITE_PAGES_TO_SCRAPE = 5
NO_OF_APPS_TO_SCRAPE = 1
SAMPLE_SIZE = 10 
RANDOM_STATE = 99

# --- Concurrency / Parallelism ---
# Max workers for Selenium-based tasks (resource-intensive)
SELENIUM_MAX_WORKERS = 2
# Max workers for network/API-based tasks (less intensive)
NETWORK_MAX_WORKERS = 10


# --- Vectorization ---
# The model will automatically use the GPU if torch with CUDA is installed
EMBEDDING_MODEL = 'ProsusAI/finbert'

# --- LLM Models ---
LLM_MODEL = "gpt-4o-mini"

# --- News Scraping ---
CREDIBLE_NEWS_SITES = [
    "difc.ae", "fintechfutures.com", "fintechnews.ae", "mea-finance.com",
    "thefinanceworld.com", "zawya.com", "dubaiinvestments.com",
    "emirates247.com", "khaleejtimes.com", "gulfnews.com", "arabianbusiness.com"
]

# --- Deep Search Scraper Configuration ---
REGIONAL_TECH_EVENTS = [
    "GITEX Global", "Dubai Fintech Summit", "Seamless Middle East",
    "Step Conference Dubai", "LEAP Riyadh"
]

INDUSTRY_PUBLICATIONS = [
    "fintechmagazine.com", "forbesmiddleeast.com", "wamda.com",
    "magnitt.com", "menabytes.com"
]
