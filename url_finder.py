import pandas as pd
import json
import re
import os
import requests
import time
from dotenv import load_dotenv
import openai

# --- Selenium Imports ---
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait

# --- Configuration ---
load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- MODULE 1: Data Loading ---
def load_companies(csv_path):
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

# --- MODULE 2: URL Finding ---
def search_brave(query):
    """Performs a generic web search using the Brave Search API."""
    if not BRAVE_API_KEY:
        print("⚠️  Brave API key is missing. Skipping real search.")
        return []
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "country": "US", "search_lang": "en"}
    try:
        response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('web', {}).get('results', [])
    except Exception as e:
        print(f"❌ An error occurred during Brave Search: {e}")
        return []

def find_linkedin_url_with_llm(company_name, client):
    """Uses Brave Search and an LLM to find the most likely official LinkedIn company page."""
    print(f"  -> Searching for '{company_name}'...")
    query = f'"{company_name}" linkedin company profile'
    search_results = search_brave(query)
    if not search_results:
        print("    -> No results from Brave Search.")
        return None

    context = "".join([f"Result {i+1}:\nTitle: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('description', '')}\n\n" for i, r in enumerate(search_results[:3])])
    prompt = f"""
    You are an expert business analyst. Based on the following search results, identify the single most likely official LinkedIn company page URL for "{company_name}".

    **CRITICAL INSTRUCTIONS:**
    1. The correct URL must contain `/company/`.
    2. For a name like "Samaa Finance", the URL could be `.../samaa-finance` OR `.../samaafinance`.
    3. Prioritize the result that most closely matches the company name.

    Search Results:
    {context}

    Return a JSON object with one key: "url". The value should be the correct URL or null.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
        url = json.loads(response.choices[0].message.content).get("url")
        if url and "linkedin.com/company/" in url:
            return url
        else:
            return None
    except Exception as e:
        print(f"    -> An unexpected error occurred during LLM verification: {e}")
        return None

# --- MODULE 3: URL Verification ---
def verify_url(driver, url):
    """Visits a URL with Selenium to check if it's a valid, available page."""
    try:
        driver.get(url)
        # A short wait to allow for redirects or error page loads
        time.sleep(3)
        page_source = driver.page_source.lower()
        if "page isn’t available" in page_source or "page was not found" in page_source:
            return False
        return True
    except Exception as e:
        print(f"    -> Error while verifying URL {url}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    INPUT_CSV = 'combined_institutions.csv'
    OUTPUT_CSV = 'institutions_linkedin.csv'
    
    print("--- Step 1: LinkedIn URL Finder and Verifier ---")
    
    if not OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found.")
    else:
        companies_df = load_companies(INPUT_CSV)
        if companies_df is not None:
            print(f"\n✅ Successfully loaded and cleaned {len(companies_df)} companies.")
            
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize lists to store new data
            linkedin_urls = []
            statuses = []
            
            # Setup Selenium driver
            print("\n🚀 Initializing browser for URL verification...")
            driver = uc.Chrome(use_subprocess=True)
            
            print(f"\n🔎 Starting URL search for {len(companies_df)} companies...")
            for index, row in companies_df.iterrows():
                company_name = row['Cleaned Name']
                print(f"\nProcessing ({index+1}/{len(companies_df)}): {company_name}")
                
                url = find_linkedin_url_with_llm(company_name, openai_client)
                
                if url:
                    print(f"    -> Found potential URL: {url}")
                    print("    -> Verifying page availability...")
                    if verify_url(driver, url):
                        print("    ✅ URL is valid and page exists.")
                        linkedin_urls.append(url)
                        statuses.append("Found")
                    else:
                        print("    ❌ URL leads to an unavailable page.")
                        linkedin_urls.append(None)
                        statuses.append("Invalid Page")
                else:
                    print("    -> No confident URL found by LLM.")
                    linkedin_urls.append(None)
                    statuses.append("Not Found")
                
                time.sleep(1) # Pause between companies

            # Add new columns to the DataFrame
            companies_df['linkedin_url'] = linkedin_urls
            companies_df['status'] = statuses
            
            # Save the enriched DataFrame to a new CSV
            companies_df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"\n\n🎉 --- Process Complete --- 🎉")
            print(f"✅ Enriched data saved to '{OUTPUT_CSV}'")
            
            # Cleanly close the browser
            if driver:
                driver.quit()
