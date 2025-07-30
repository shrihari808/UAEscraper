# modules/url_finder.py

import requests
import json
import time
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait

import config

class URLFinder:
    """Finds and verifies LinkedIn company URLs."""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.driver = None

    def _search_brave(self, query):
        """Performs a web search using the Brave Search API."""
        if not config.BRAVE_API_KEY:
            print("⚠️  Brave API key is missing. Skipping real search.")
            return []
        headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
        params = {"q": query, "country": "US", "search_lang": "en"}
        try:
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('web', {}).get('results', [])
        except Exception as e:
            print(f"❌ An error occurred during Brave Search: {e}")
            return []

    def _find_url_with_llm(self, company_name):
        """Uses Brave Search and an LLM to find the most likely official LinkedIn company page."""
        print(f"  -> Searching for '{company_name}'...")
        query = f'"{company_name}" linkedin company profile'
        search_results = self._search_brave(query)
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
            response = self.openai_client.chat.completions.create(model=config.LLM_MODEL, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
            url = json.loads(response.choices[0].message.content).get("url")
            return url if url and "linkedin.com/company/" in url else None
        except Exception as e:
            print(f"    -> An unexpected error occurred during LLM verification: {e}")
            return None

    def _verify_url(self, url):
        """Visits a URL with Selenium to check if it's a valid, available page."""
        try:
            self.driver.get(url)
            time.sleep(3)
            page_source = self.driver.page_source.lower()
            return "page isn’t available" not in page_source and "page was not found" not in page_source
        except Exception as e:
            print(f"    -> Error while verifying URL {url}: {e}")
            return False

    def process_companies(self, df):
        """Iterates through a DataFrame to find and verify LinkedIn URLs."""
        print("\n🚀 Initializing browser for URL verification...")
        self.driver = uc.Chrome(use_subprocess=True)
        
        linkedin_urls = []
        statuses = []

        print(f"\n🔎 Starting URL search for {len(df)} companies...")
        for index, row in df.iterrows():
            company_name = row['Cleaned Name']
            print(f"\nProcessing ({index+1}/{len(df)}): {company_name}")
            
            url = self._find_url_with_llm(company_name)
            
            if url:
                print(f"    -> Found potential URL: {url}")
                print("    -> Verifying page availability...")
                if self._verify_url(url):
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
            
            time.sleep(1)

        if self.driver:
            self.driver.quit()

        df['linkedin_url'] = linkedin_urls
        df['status'] = statuses
        return df
