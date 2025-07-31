# modules/news_scraper.py

import requests
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from langchain.docstore.document import Document

import config

class NewsScraper:
    """Searches for and scrapes news articles, returning LangChain Documents."""

    def __init__(self, driver=None):
        """
        Initializes the scraper.
        If a driver is provided, it uses it. Otherwise, it creates a new one.
        """
        if driver:
            self.driver = driver
            self.owns_driver = False
        else:
            self.driver = self._setup_driver()
            self.owns_driver = True

    def _setup_driver(self):
        """Initializes a new browser instance."""
        print("  -> (NewsScraper) Creating new driver instance...")
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        return uc.Chrome(options=options, use_subprocess=True)


    def _search_brave(self, query):
        """Performs a targeted web search using the Brave Search API."""
        if not config.BRAVE_API_KEY:
            print("⚠️  Brave API key is missing. Skipping real search.")
            return []
            
        time.sleep(config.BRAVE_API_RATE_LIMIT)

        headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
        params = {"q": query, "country": "US", "search_lang": "en"}
        try:
            print(f"  -> Sending API request for query: {query}")
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('web', {}).get('results', [])
        except Exception as e:
            print(f"❌ An error occurred during Brave Search: {e}")
            return []

    def scrape_articles(self, company_name):
        """Searches for news and scrapes the top results."""
        print(f"  -> Searching for external news for '{company_name}'...")
        
        site_query = " OR ".join([f"site:{site}" for site in config.CREDIBLE_NEWS_SITES])
        query = f'"{company_name}" ({site_query})'
        
        search_results = self._search_brave(query)
        documents = []
        
        if not search_results:
            print(f"    -> No news found on credible sites for '{company_name}'.")
            return []
            
        for result in search_results[:config.NO_OF_NEWS_ARTICLES_TO_SCRAPE]:
            url = result.get("url")
            if url:
                try:
                    print(f"    -> Scraping news article for '{company_name}': {url}")
                    self.driver.get(url)
                    time.sleep(random.uniform(2, 4))
                    article_text = self.driver.find_element(By.TAG_NAME, 'body').text
                    documents.append(Document(
                        page_content=article_text,
                        metadata={"company": company_name, "source": url, "type": "news"}
                    ))
                except Exception as e:
                    print(f"    ⚠️ Could not scrape article {url}: {e}")
                    
        return documents

    def close(self):
        """Closes the Selenium driver only if this instance created it."""
        if self.driver and self.owns_driver:
            self.driver.quit()
            print("\nBrowser instance closed.")
