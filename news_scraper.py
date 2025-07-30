import pandas as pd
import json
import re
import os
import time
import random
from dotenv import load_dotenv
import numpy as np
import requests

# --- Selenium Imports ---
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Vectorization Imports ---
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
FAISS_INDEX_PATH = "linkedin_data.index" # We will add to this existing index
METADATA_PATH = "linkedin_metadata.json" # And this metadata file

# --- MODULE 1: Data Loading ---
def load_enriched_data(csv_path):
    """Loads the CSV file containing company names."""
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['linkedin_url']) # Ensure we only process companies with a LinkedIn presence
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Please run Step 1 first.")
        return None

# --- MODULE 2: News Searching & Scraping ---
def search_brave(query):
    """Performs a targeted web search using the Brave Search API."""
    if not BRAVE_API_KEY:
        print("⚠️  Brave API key is missing. Skipping real search.")
        return []
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "country": "US", "search_lang": "en"}
    try:
        print(f"  -> Sending API request for query: {query}")
        response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('web', {}).get('results', [])
    except Exception as e:
        print(f"❌ An error occurred during Brave Search: {e}")
        return []

def search_and_scrape_news(company_name, driver):
    """Searches for news on credible sites and scrapes the top results."""
    print(f"  -> Searching for external news for '{company_name}'...")
    
    credible_sites = [
        "difc.ae", "fintechfutures.com", "fintechnews.ae", "mea-finance.com",
        "thefinanceworld.com", "zawya.com", "dubaiinvestments.com", 
        "emirates247.com", "khaleejtimes.com"
    ]
    site_query = " OR ".join([f"site:{site}" for site in credible_sites])
    query = f'"{company_name}" ({site_query})'
    
    search_results = search_brave(query)
    scraped_news = []
    
    if not search_results:
        print("    -> No news found on credible sites.")
        return []
        
    for result in search_results[:2]: # Scrape top 2 articles
        url = result.get("url")
        if url:
            try:
                print(f"    -> Scraping news article: {url}")
                driver.get(url)
                time.sleep(random.uniform(2, 4)) # Wait for page to load
                article_text = driver.find_element(By.TAG_NAME, 'body').text
                scraped_news.append({"text": article_text, "source": url})
            except Exception as e:
                print(f"    ⚠️ Could not scrape article {url}: {e}")
                
    return scraped_news

# --- MODULE 3: Vectorization and Storage ---
def append_embeddings_to_storage(scraped_news, company_name):
    """Generates embeddings for news and appends them to the existing FAISS index."""
    print("  -> Generating and storing news embeddings...")
    
    if not scraped_news:
        print("    -> No news content to vectorize.")
        return

    all_texts = [item['text'] for item in scraped_news]
    metadata = [{"company": company_name, "source": item['source'], "type": "news"} for item in scraped_news]

    # Generate embeddings
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)
    
    # Load existing FAISS index and metadata
    if not os.path.exists(FAISS_INDEX_PATH):
        print("    ❌ FAISS index not found. Please run Step 2 (LinkedIn Scraper) first.")
        return
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        existing_metadata = json.load(f)

    # Add new data to index and metadata
    index.add(embeddings)
    existing_metadata.extend(metadata)
    
    # Save updated index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(existing_metadata, f, indent=4)
        
    print(f"    ✅ Stored {len(all_texts)} new news vector embeddings.")

# --- Main Execution ---
if __name__ == "__main__":
    INPUT_CSV = 'institutions_linkedin.csv'
    
    print("--- Step 3: News Scraper and Vectorizer ---")
    
    companies_df = load_enriched_data(INPUT_CSV)
    if companies_df is not None:
        print(f"\n✅ Found {len(companies_df)} companies to process.")
        
        # --- NEW: Use a fixed random_state for reproducible sampling ---
        if len(companies_df) >= 5:
            # This ensures the same "random" 5 companies are picked every time.
            companies_to_process_df = companies_df.sample(n=5, random_state=42)
        else:
            companies_to_process_df = companies_df
        
        print(f"\nWill now process a sample of {len(companies_to_process_df)} companies for news:")
        for name in companies_to_process_df['Cleaned Name']:
            print(f"  - {name}")

        # Initialize a simple driver, no login needed for public news sites
        driver = uc.Chrome(use_subprocess=True)
        
        if driver:
            try:
                for index, row in companies_to_process_df.iterrows():
                    company_name = row['Cleaned Name']
                    
                    print(f"\nProcessing: {company_name}")
                    
                    scraped_news = search_and_scrape_news(company_name, driver)
                    
                    if scraped_news:
                        append_embeddings_to_storage(scraped_news, company_name)
                    
                    time.sleep(1) # Pause between companies

            except Exception as e:
                print(f"❌ An unexpected error occurred in the main loop: {e}")
            finally:
                if driver:
                    driver.quit()
                print(f"\n\n🎉 --- Process Complete --- 🎉")
                print(f"✅ News vector knowledge base appended to '{FAISS_INDEX_PATH}' and '{METADATA_PATH}'")
