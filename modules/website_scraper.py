# modules/website_scraper.py

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from langchain.docstore.document import Document

import config

class WebsiteScraper:
    """
    Asynchronously scrapes data from a company's official website 
    and returns LangChain Documents.
    """

    def __init__(self):
        """Initializes the scraper with standard headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _is_valid_url(self, url, base_domain):
        """Checks if a URL is valid and belongs to the same domain to avoid crawling external sites."""
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc) and parsed_url.netloc == base_domain

    def _get_clean_text(self, soup):
        """
        Extracts clean, meaningful text from a BeautifulSoup object.
        Removes common non-content tags like nav, footer, script, etc.
        """
        for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            script_or_style.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _fetch_url(self, session, url):
        """Asynchronously fetches a single URL."""
        try:
            # Add a polite delay
            await asyncio.sleep(random.uniform(1, 2))
            async with session.get(url, timeout=10, headers=self.headers) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            print(f"    ⚠️  Could not fetch {url}: {e}")
            return None

    async def scrape_website(self, company_name, base_url):
        """
        Asynchronously crawls a website starting from the base_url, scrapes content,
        and returns a list of LangChain Document objects.
        """
        if not base_url or not isinstance(base_url, str) or not base_url.startswith('http'):
            print(f"    -> Invalid or missing base URL for {company_name}, skipping website scrape.")
            return []

        print(f"  -> Starting ASYNC website scrape for '{company_name}' at {base_url}")
        documents = []
        urls_to_visit = {base_url}
        visited_urls = set()
        base_domain = urlparse(base_url).netloc

        async with aiohttp.ClientSession() as session:
            while urls_to_visit and len(visited_urls) < config.NO_OF_WEBSITE_PAGES_TO_SCRAPE:
                url = urls_to_visit.pop()
                if url in visited_urls:
                    continue

                visited_urls.add(url)
                html_content = await self._fetch_url(session, url)

                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    page_text = self._get_clean_text(soup)
                    if page_text:
                        documents.append(Document(
                            page_content=page_text,
                            metadata={"company": company_name, "source": url, "type": "website"}
                        ))
                        print(f"    ✅ Scraped: {url}")

                    # Find new internal links (this part remains synchronous but is fast)
                    if len(visited_urls) < config.NO_OF_WEBSITE_PAGES_TO_SCRAPE:
                        for link in soup.find_all('a', href=True):
                            absolute_link = urljoin(base_url, link['href']).split('#')[0] # Join relative URLs and remove fragments
                            if self._is_valid_url(absolute_link, base_domain) and absolute_link not in visited_urls:
                                urls_to_visit.add(absolute_link)

        print(f"  -> Finished scraping. Found {len(documents)} pages from {company_name}'s website.")
        return documents

    def close(self):
        """A method for consistency with other scrapers, no action needed for requests."""
        # This method is synchronous, so we use asyncio.run to call the async scrape_website
        pass
