# modules/url_finder.py

import requests
import json
import time
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait

import config

class URLFinder:
    """Finds and verifies LinkedIn and official website URLs using an analytical, LLM-driven approach."""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.driver = None

    def _search_brave(self, query):
        """Performs a web search using the Brave Search API."""
        if not config.BRAVE_API_KEY:
            print("⚠️  Brave API key is missing. Skipping real search.")
            return []
        
        # Add rate limiting delay
        time.sleep(config.BRAVE_API_RATE_LIMIT)
        
        headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
        params = {"q": query, "country": "US", "search_lang": "en", "count": 15} 
        try:
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('web', {}).get('results', [])
        except Exception as e:
            print(f"❌ An error occurred during Brave Search: {e}")
            return []

    def _find_linkedin_url_with_llm(self, company_name):
        """
        Uses a hybrid approach to find the LinkedIn URL.
        First, it checks the top search result for a high-confidence match.
        If not found, it uses an analytical LLM prompt on a larger set of results.
        """
        print(f"  -> Searching for LinkedIn page for '{company_name}'...")
        query = f'"{company_name}" linkedin company profile'
        search_results = self._search_brave(query)
        if not search_results:
            print("    -> No results from Brave Search.")
            return None

        # --- OPTIMIZATION: Check if the first result is a direct match ---
        first_url = search_results[0].get('url', '')
        if "linkedin.com/company/" in first_url:
            print(f"    -> High-confidence match found in the first result: {first_url}")
            return first_url

        # --- FALLBACK: If no direct match, use LLM for deeper analysis ---
        print("    -> No high-confidence match in top result, using LLM for deeper analysis...")
        context = "".join([f"Result {i+1}:\nTitle: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('description', '')}\n\n" for i, r in enumerate(search_results)])
        
        prompt = f"""
        You are an expert business analyst. Your task is to identify the single, official LinkedIn company page URL for "{company_name}" from the search results below.
        
        **CRITICAL INSTRUCTIONS:**
        1.  **Analyze Brand vs. Formal Name:** The company's common brand name might be simpler than its formal name. For example, for "Al Mashreq Al Islami Finance Company PJSC", the correct page is likely just titled "Mashreq".
        2.  **Identify Company Pages:** The correct URL must contain `/company/`. The title or snippet often includes follower counts, employee numbers, or the industry (e.g., "Financial Services").
        3.  **Avoid Incorrect Pages:** Discard personal profiles (e.g., URLs with `/in/`), news articles, or directory listings.
        4.  **Make a Confident Choice:** Based on all evidence, select the single most probable URL.

        **Search Results:**
        {context}

        Return a JSON object with one key: "url". The value should be the correct URL or null if no confident match is found.
        """
        try:
            response = self.openai_client.chat.completions.create(model=config.LLM_MODEL, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
            url = json.loads(response.choices[0].message.content).get("url")
            return url if url and "linkedin.com/company/" in url else None
        except Exception as e:
            print(f"    -> An unexpected error occurred during LLM verification: {e}")
            return None

    def _find_website_url_with_llm(self, company_name):
        """Uses Brave Search and an analytical LLM prompt to find the official corporate website."""
        print(f"  -> Searching for official website for '{company_name}'...")
        query = f'"{company_name}" official website'
        search_results = self._search_brave(query)
        if not search_results:
            print("    -> No results from Brave Search for website.")
            return None

        context = "".join([f"Result {i+1}:\nTitle: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('description', '')}\n\n" for i, r in enumerate(search_results)])

        prompt = f"""
        You are an expert web analyst. Your task is to identify the single, official corporate website for "{company_name}" from the search results below.

        **CRITICAL INSTRUCTIONS:**
        1.  **Identify the Homepage:** The correct URL is the company's own homepage, not a third-party site.
        2.  **IGNORE Irrelevant Links:** You MUST discard links to news articles (e.g., Zawya, Reuters), business directories (e.g., Wikipedia, Bloomberg profiles), or social media.
        3.  **Analyze URL and Snippet:** The official website URL is often a clean domain (e.g., `sirajfinance.com`). The snippet will describe the company's own services (e.g., "Siraj Finance offers..."). A news snippet would say "Siraj Finance announced...".
        4.  **Example:** For "First Abu Dhabi Islamic Finance PJSC", the correct URL is `https://www.bankfab.com/en-ae/islamic-banking`, NOT a news article from zawya.com.

        **Search Results:**
        {context}

        Return a JSON object with one key: "url". The value should be the correct corporate website URL or null if no official site is found.
        """
        try:
            response = self.openai_client.chat.completions.create(model=config.LLM_MODEL, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
            url = json.loads(response.choices[0].message.content).get("url")
            print(f"    -> LLM identified potential website: {url}")
            return url
        except Exception as e:
            print(f"    -> An unexpected error occurred during LLM website identification: {e}")
            return None

    def _verify_url(self, url):
        """Visits a URL with Selenium to check if it's a valid, available page."""
        if not url or not isinstance(url, str):
            return False
        try:
            self.driver.get(url)
            time.sleep(3)
            page_source = self.driver.page_source.lower()
            return "page isn’t available" not in page_source and "page was not found" not in page_source
        except Exception as e:
            print(f"    -> Error while verifying URL {url}: {e}")
            return False

    def process_companies(self, df):
        """Iterates through a DataFrame to find and verify LinkedIn and website URLs."""
        print("\n🚀 Initializing browser for URL verification...")
        self.driver = uc.Chrome(use_subprocess=True)
        
        linkedin_urls = []
        website_urls = []
        statuses = []

        print(f"\n🔎 Starting URL search for {len(df)} companies...")
        for index, row in df.iterrows():
            company_name = row['Cleaned Name']
            print(f"\nProcessing ({index+1}/{len(df)}): {company_name}")
            
            # Find LinkedIn URL
            linkedin_url = self._find_linkedin_url_with_llm(company_name)
            
            if linkedin_url:
                print(f"    -> Found potential LinkedIn URL: {linkedin_url}")
                print("    -> Verifying page availability...")
                if self._verify_url(linkedin_url):
                    print("    ✅ LinkedIn URL is valid and page exists.")
                    linkedin_urls.append(linkedin_url)
                    statuses.append("Found")
                else:
                    print("    ❌ LinkedIn URL leads to an unavailable page.")
                    linkedin_urls.append(None)
                    statuses.append("Invalid Page")
            else:
                print("    -> No confident LinkedIn URL found by LLM.")
                linkedin_urls.append(None)
                statuses.append("Not Found")

            # Find Website URL
            website_url = self._find_website_url_with_llm(company_name)
            website_urls.append(website_url)
            
            time.sleep(1)

        if self.driver:
            self.driver.quit()

        df['linkedin_url'] = linkedin_urls
        df['website_url'] = website_urls
        df['status'] = statuses
        return df
