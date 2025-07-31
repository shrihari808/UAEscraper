# modules/deep_search_scraper.py

import requests
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

class DeepSearchScraper:
    """
    A collection of tools for performing targeted, intent-based searches and scraping.
    The orchestration is handled by the main pipeline to manage rate limiting effectively.
    """

    def __init__(self):
        """Initializes the scraper."""
        pass

    def generate_queries(self, company_name):
        """Generates a list of targeted search queries with their associated type."""
        queries_with_type = []
        query_map = {
            "partnership_info": [
                f'"{company_name}" partners with',
                f'"{company_name}" collaboration',
                f'"{company_name}" integration with',
            ],
            "announcement_release": [
                f'"{company_name}" press release',
                f'"{company_name}" announces',
                f'"{company_name}" launches product',
                f'"{company_name}" funding round',
            ],
            "conference_mention": [f'"{company_name}" at "{event}"' for event in config.REGIONAL_TECH_EVENTS],
            "forum_discussion": [
                f'"{company_name}" review site:reddit.com',
                f'site:teamblind.com "{company_name}"',
            ],
            "magazine_feature": [f'"{company_name}" site:{pub}' for pub in config.INDUSTRY_PUBLICATIONS],
        }
        for doc_type, queries in query_map.items():
            for query in queries:
                queries_with_type.append({"doc_type": doc_type, "query": query})
        return queries_with_type

    def search_brave(self, query_info):
        """Performs a web search using the Brave Search API for a single query."""
        query = query_info["query"]
        doc_type = query_info["doc_type"]

        if not config.BRAVE_API_KEY:
            return []
        
        headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
        params = {"q": query, "country": "SA", "search_lang": "en", "count": 3}
        try:
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=15)
            response.raise_for_status()
            results = response.json().get('web', {}).get('results', [])
            for r in results:
                r['doc_type'] = doc_type
                r['company_name'] = query_info['company_name']
            return results
        except requests.RequestException as e:
            print(f"    ❌ An error occurred during Brave Search for query '{query}': {e}")
            return []

    def _get_clean_text(self, soup):
        """Extracts clean, meaningful text from a BeautifulSoup object."""
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text

    def scrape_single_url(self, task_info):
        """
        Scrapes a single URL using a driver from a shared pool.
        """
        url = task_info['url']
        doc_type = task_info['doc_type']
        company_name = task_info['company_name']
        driver_pool = task_info['driver_pool']
        
        driver = driver_pool.get() # Borrow a driver
        try:
            driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
            
            print(f"      -> Scraping URL for {company_name}: {url}")
            driver.get(url)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            page_text = self._get_clean_text(soup)

            if page_text:
                return Document(
                    page_content=page_text,
                    metadata={"company": company_name, "source": driver.current_url, "type": doc_type}
                )
        except Exception as e:
            print(f"      ⚠️ Could not scrape URL {url}: {e}")
            return None
        finally:
            driver_pool.put(driver) # Always return the driver

    def close(self):
        """A method for consistency with other scrapers."""
        pass
