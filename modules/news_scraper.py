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

    def __init__(self):
        self.driver = uc.Chrome(use_subprocess=True)

    def _search_brave(self, query):
        """Performs a targeted web search using the Brave Search API."""
        if not config.BRAVE_API_KEY:
            print("⚠️  Brave API key is missing. Skipping real search.")
            return []
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
            print("    -> No news found on credible sites.")
            return []
            
        for result in search_results[:config.NO_OF_NEWS_ARTICLES_TO_SCRAPE]:
            url = result.get("url")
            if url:
                try:
                    print(f"    -> Scraping news article: {url}")
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
        """Closes the Selenium driver."""
        if self.driver:
            self.driver.quit()
            print("\nBrowser for news scraping closed.")
