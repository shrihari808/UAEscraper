# modules/deep_search_scraper.py

import requests
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

import config

class DeepSearchScraper:
    """
    Performs targeted, intent-based searches across the web for high-value intelligence
    like partnerships, announcements, and forum discussions.
    """

    def __init__(self):
        """Initializes the scraper and the selenium driver."""
        # --- NEW: Add options for stability ---
        options = uc.ChromeOptions()
        options.add_argument('--headless') # Run in the background without a visible browser window
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        # --- END NEW ---
        
        self.driver = uc.Chrome(options=options, use_subprocess=True)
        # Add a 5-second timeout for page loads
        self.driver.set_page_load_timeout(5)

    def _generate_queries(self, company_name):
        """Generates a dictionary of targeted search queries based on intent."""
        queries = {
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
        return queries

    def _search_brave(self, query):
        """Performs a web search using the Brave Search API."""
        if not config.BRAVE_API_KEY:
            print("    ⚠️ Brave API key is missing. Skipping deep search.")
            return []
        headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
        params = {"q": query, "country": "SA", "search_lang": "en", "count": 3}
        try:
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get('web', {}).get('results', [])
        except requests.RequestException as e:
            print(f"    ❌ An error occurred during Brave Search for query '{query}': {e}")
            return []

    def _get_clean_text(self, soup):
        """Extracts clean, meaningful text from a BeautifulSoup object."""
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text

    def _find_and_click_relevant_link(self, company_name):
        """
        Searches the current page for a link containing the company name and clicks it
        to navigate to a more detailed article page.
        """
        try:
            # Split company name to check for partial matches (e.g., "Emirates Money")
            name_parts = company_name.lower().split()
            # Find all links on the page
            links = self.driver.find_elements(By.TAG_NAME, 'a')
            
            for link in links:
                link_text = link.text.lower()
                # Check if at least two parts of the company name are in the link text
                if sum(part in link_text for part in name_parts) >= 2:
                    href = link.get_attribute('href')
                    if href:
                        print(f"        -> Found relevant link: '{link.text}'. Clicking...")
                        self.driver.get(href) # Navigate to the more detailed page
                        # The page load timeout set in __init__ will apply here
                        return True # Indicate that we have successfully navigated
            return False # No relevant link was found and clicked
        except WebDriverException as e:
            # This will catch the TimeoutException if the page load fails
            print(f"        ⚠️ Error while trying to find and click a link: {e}")
            return False


    def scrape_deep_web(self, company_name):
        """
        Orchestrates the deep web scrape for a single company.
        """
        print(f"  -> Starting deep web search for '{company_name}'...")
        queries_by_type = self._generate_queries(company_name)
        documents = []
        scraped_urls = set()

        for doc_type, queries in queries_by_type.items():
            for query in queries:
                print(f"    -> Querying for '{doc_type}': {query}")
                search_results = self._search_brave(query)
                time.sleep(random.uniform(1, 2))

                for result in search_results:
                    url = result.get("url")
                    if url and url not in scraped_urls:
                        scraped_urls.add(url)
                        print(f"      -> Navigating to initial URL: {url}")
                        try:
                            self.driver.get(url)
                            # The page load timeout set in __init__ will apply here

                            # --- NEW LOGIC ---
                            # Try to find a more specific article link on the page and click it
                            self._find_and_click_relevant_link(company_name)
                            # After potentially navigating, the final URL might have changed
                            final_url = self.driver.current_url
                            print(f"      -> Scraping content from final URL: {final_url}")
                            # --- END NEW LOGIC ---

                            page_source = self.driver.page_source
                            soup = BeautifulSoup(page_source, 'html.parser')
                            page_text = self._get_clean_text(soup)

                            if page_text:
                                documents.append(Document(
                                    page_content=page_text,
                                    metadata={"company": company_name, "source": final_url, "type": doc_type}
                                ))
                        except Exception as e:
                            print(f"      ⚠️ Could not scrape URL {url}: {e}")
        
        print(f"  -> Deep web search for '{company_name}' complete. Found {len(documents)} documents.")
        return documents

    def close(self):
        """Closes the Selenium driver."""
        if self.driver:
            self.driver.quit()
            print("\nDeep search browser closed.")
