import pandas as pd
import json
import re
import os
import time
import random
from dotenv import load_dotenv
import numpy as np

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
FAISS_INDEX_PATH = "linkedin_data.index"
METADATA_PATH = "linkedin_metadata.json"

# --- MODULE 1: Data Loading ---
def load_enriched_data(csv_path):
    """Loads the CSV file containing verified LinkedIn URLs."""
    try:
        df = pd.read_csv(csv_path)
        # Filter out companies where no valid URL was found
        df = df.dropna(subset=['linkedin_url'])
        return df
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Please run Step 1 first.")
        return None

# --- MODULE 2: Selenium Scraping ---
def setup_driver_with_cookies():
    """Initializes a browser and logs in using a cookie file."""
    print("🚀 Initializing WebDriver and loading cookies...")
    try:
        with open("cookies.json", "r") as f:
            cookies = json.load(f)
    except FileNotFoundError:
        print("\n❌ FATAL ERROR: cookies.json not found.")
        return None
    driver = None
    try:
        driver = uc.Chrome(use_subprocess=True)
        driver.get("https://www.linkedin.com/")
        for cookie in cookies:
            if 'expiry' in cookie:
                cookie['expiry'] = int(cookie['expiry'])
            driver.add_cookie(cookie)
        print("  -> Cookies loaded into browser.")
        driver.get("https://www.linkedin.com/feed/")
        WebDriverWait(driver, 15).until(lambda d: "Feed" in d.title or "Sign In" in d.title)
        if "Feed" not in driver.title:
            print("❌ LOGIN FAILED. Cookies might be invalid.")
            driver.quit()
            return None
        print(f"✅ Successfully logged into LinkedIn.")
        return driver
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during WebDriver setup: {e}")
        if driver: driver.quit()
        return None

def human_like_delay():
    time.sleep(random.uniform(2.5, 4.5))

def scrape_linkedin_page(driver, company_url):
    """Scrapes the 'About', 'Posts', and 'Jobs' sections of a LinkedIn page."""
    scraped_data = {"about": "", "posts": [], "jobs": []}
    wait = WebDriverWait(driver, 10)
    
    # Scrape 'About' page
    try:
        about_url = company_url.rstrip('/') + '/about/'
        driver.get(about_url)
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
        scraped_data["about"] = driver.find_element(By.TAG_NAME, 'body').text
        print("    ✅ Scraped 'About' page.")
    except Exception:
        print("    ⚠️ Could not scrape 'About' page.")
    
    # Scrape 'Posts' page
    try:
        posts_url = company_url.rstrip('/') + '/posts/'
        driver.get(posts_url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "scaffold-finite-scroll__content")))
        for _ in range(4): 
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            human_like_delay()
        post_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'update-components-text')]")
        for post in post_elements:
            try:
                more_button = post.find_element(By.CSS_SELECTOR, ".update-components-text__see-more")
                driver.execute_script("arguments[0].click();", more_button)
                time.sleep(0.5)
            except NoSuchElementException: pass
        for post in post_elements: scraped_data["posts"].append(post.text)
        print(f"    ✅ Scraped {len(scraped_data['posts'])} posts.")
    except Exception:
        print("    -> No posts section found or error during scraping.")

    # Scrape 'Jobs' page (simplified for this step)
    try:
        jobs_url = company_url.rstrip('/') + '/jobs/'
        driver.get(jobs_url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'jobs-search-results-list')))
        job_elements = driver.find_elements(By.CSS_SELECTOR, '.job-card-list__title')
        for job in job_elements: scraped_data["jobs"].append(f"Hiring for: {job.text.strip()}")
        print(f"    ✅ Scraped {len(scraped_data['jobs'])} jobs.")
    except Exception:
        print("    -> No jobs section found or error during scraping.")
        
    return scraped_data

# --- MODULE 3: Vectorization and Storage ---
def create_and_store_embeddings(scraped_data, company_name, linkedin_url):
    """Generates embeddings and saves them to a FAISS index and metadata file."""
    print("  -> Generating and storing embeddings...")
    all_texts = []
    metadata = []

    # Add 'About' text
    if scraped_data.get("about"):
        all_texts.append(scraped_data["about"])
        metadata.append({"company": company_name, "source": linkedin_url, "type": "about"})

    # Add 'Posts' texts
    for post in scraped_data.get("posts", []):
        all_texts.append(post)
        metadata.append({"company": company_name, "source": linkedin_url, "type": "post"})
        
    # Add 'Jobs' texts
    for job in scraped_data.get("jobs", []):
        all_texts.append(job)
        metadata.append({"company": company_name, "source": linkedin_url, "type": "job"})

    if not all_texts:
        print("    -> No text content to vectorize.")
        return

    # Generate embeddings
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)
    
    # Load existing FAISS index and metadata or create new ones
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            existing_metadata = json.load(f)
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        existing_metadata = []

    # Add new data to index and metadata
    index.add(embeddings)
    existing_metadata.extend(metadata)
    
    # Save updated index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(existing_metadata, f, indent=4)
        
    print(f"    ✅ Stored {len(all_texts)} new vector embeddings.")

# --- Main Execution ---
if __name__ == "__main__":
    INPUT_CSV = 'institutions_linkedin.csv'
    
    print("--- Step 2: LinkedIn Scraper and Vectorizer ---")
    
    companies_df = load_enriched_data(INPUT_CSV)
    if companies_df is not None:
        print(f"\n✅ Found {len(companies_df)} companies with LinkedIn URLs to process.")
        
        # --- NEW: Select 5 random companies for processing ---
        if len(companies_df) >= 5:
            companies_to_process_df = companies_df.sample(n=5, random_state=42)
        else:
            companies_to_process_df = companies_df # process all if less than 5
        
        print(f"\nWill now process a random sample of {len(companies_to_process_df)} companies:")
        for name in companies_to_process_df['Cleaned Name']:
            print(f"  - {name}")

        driver = setup_driver_with_cookies()
        
        if driver:
            try:
                for index, row in companies_to_process_df.iterrows():
                    company_name = row['Cleaned Name']
                    linkedin_url = row['linkedin_url']
                    
                    print(f"\nProcessing: {company_name}")
                    
                    scraped_data = scrape_linkedin_page(driver, linkedin_url)
                    
                    if scraped_data:
                        create_and_store_embeddings(scraped_data, company_name, linkedin_url)
                    
                    time.sleep(1) # Pause between companies

            except Exception as e:
                print(f"❌ An unexpected error occurred in the main loop: {e}")
            finally:
                if driver:
                    driver.quit()
                print(f"\n\n🎉 --- Process Complete --- 🎉")
                print(f"✅ Vector knowledge base saved to '{FAISS_INDEX_PATH}' and '{METADATA_PATH}'")
