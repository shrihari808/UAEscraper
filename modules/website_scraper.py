# modules/website_scraper.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import re
import os
import fitz # PyMuPDF
from langchain.docstore.document import Document

import config

class WebsiteScraper:
    """
    REVISED: Scrapes data from a company's official website in two distinct stages.
    It now chunks PDF text by page to improve analysis and bypasses SSL errors.
    """

    def __init__(self):
        """Initializes the scraper with standard headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.high_value_keywords = ['investor', 'relation', 'press', 'media', 'news', 'report', 'annual', 'financial']
        self.temp_pdf_dir = "temp_pdfs"
        if not os.path.exists(self.temp_pdf_dir):
            os.makedirs(self.temp_pdf_dir)
        
        self.excluded_domains = [
            'linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com', 
            'youtube.com', 't.me', 'tiktok.com'
        ]

    def _is_valid_url(self, url, base_domain):
        """Checks if a URL is valid and belongs to the same domain."""
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc) and parsed_url.netloc == base_domain

    def _get_clean_text(self, soup):
        """Extracts clean, meaningful text from a BeautifulSoup object."""
        for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _extract_text_chunks_from_pdf(self, pdf_path, pdf_url):
        """
        REVISED: Extracts text from a PDF and returns it as a list of Document objects,
        one for each page (chunking).
        """
        documents = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip(): # Only add pages with actual text content
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_url, 
                            "type": "pdf_report_page",
                            "page": page_num + 1
                        }
                    ))
            doc.close()
            return documents
        except Exception as e:
            print(f"      ⚠️  Could not extract text from PDF {pdf_path}: {e}")
            return []

    def _scrape_pdf(self, company_name, pdf_url):
        """
        REVISED: Downloads a PDF, extracts its text page-by-page (chunks), 
        and returns a list of Documents.
        """
        try:
            print(f"    -> Found PDF document: {pdf_url}")
            # FIX: Added verify=False to ignore SSL certificate verification errors.
            response = requests.get(pdf_url, headers=self.headers, timeout=30, stream=True, verify=False)
            response.raise_for_status()
            
            pdf_filename = os.path.join(self.temp_pdf_dir, pdf_url.split('/')[-1])
            with open(pdf_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"      -> Downloaded. Now extracting text chunks by page...")
            # Pass the original URL for metadata
            pdf_docs = self._extract_text_chunks_from_pdf(pdf_filename, pdf_url)
            os.remove(pdf_filename)

            if pdf_docs:
                # Add company name to metadata of each page document
                for doc in pdf_docs:
                    doc.metadata["company"] = company_name
                print(f"      ✅ Extracted {len(pdf_docs)} pages as text chunks from PDF: {pdf_url}")
                return pdf_docs
        except requests.RequestException as e:
            print(f"    ⚠️  Failed to download PDF {pdf_url}: {e}")
        except Exception as e:
            print(f"    ⚠️  An unexpected error occurred while processing PDF {pdf_url}: {e}")
        return []


    def _find_links(self, soup, base_url, base_domain, only_high_value=False):
        """Finds links on a page, optionally filtering for high-value ones."""
        links = set()
        for link in soup.find_all('a', href=True):
            absolute_link = urljoin(base_url, link['href']).split('#')[0]
            if not self._is_valid_url(absolute_link, base_domain):
                continue

            if only_high_value:
                link_text = link.get_text().lower()
                link_href = link['href'].lower()
                is_pdf = link_href.endswith('.pdf')
                has_keyword = any(re.search(r'\b' + keyword + r'\b', link_text) for keyword in self.high_value_keywords) or \
                              any(keyword in link_href for keyword in self.high_value_keywords)
                
                if is_pdf or has_keyword:
                    links.add(absolute_link)
            else:
                links.add(absolute_link)
        return links

    def scrape_website(self, company_name, base_url):
        """
        Crawls a website in two stages: high-value content first, then general content.
        """
        if not base_url or not isinstance(base_url, str) or not base_url.startswith('http'):
            print(f"    -> Invalid or missing base URL for {company_name}, skipping.")
            return []

        try:
            domain = urlparse(base_url).netloc.replace('www.', '')
            if domain in self.excluded_domains:
                print(f"    -> Skipping website scrape for '{company_name}' due to social media link: {base_url}")
                return []
        except Exception:
            pass

        print(f"  -> Starting 2-stage website scrape for '{company_name}' at {base_url}")
        documents = []
        visited_urls = set()
        base_domain = urlparse(base_url).netloc

        # --- Stage 1: Hunt for High-Value Documents ---
        print("\n    --- Stage 1: Searching for High-Value Documents (Reports, Press, etc.) ---")
        high_value_urls_to_visit = {base_url}
        
        while len(documents) < (config.NO_OF_WEBSITE_PAGES_TO_SCRAPE * 5) and high_value_urls_to_visit: # Allow more docs if they are PDF pages
            url = high_value_urls_to_visit.pop()
            if url in visited_urls:
                continue
            
            visited_urls.add(url)
            
            if url.lower().endswith('.pdf'):
                pdf_docs = self._scrape_pdf(company_name, url)
                documents.extend(pdf_docs) # Use extend for list of docs
                continue

            try:
                time.sleep(random.uniform(1, 2))
                # FIX: Added verify=False to ignore SSL certificate verification errors.
                response = requests.get(url, headers=self.headers, timeout=10, verify=False)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                new_links = self._find_links(soup, base_url, base_domain, only_high_value=True)
                high_value_urls_to_visit.update(new_links - visited_urls)

                page_text = self._get_clean_text(soup)
                if page_text:
                    print(f"    ✅ Scraped High-Value Page: {url}")
                    documents.append(Document(page_content=page_text, metadata={"company": company_name, "source": url, "type": "website_high_value"}))
            except Exception as e:
                print(f"    ⚠️  Could not scrape {url} in Stage 1: {e}")

        print(f"\n    --- Stage 1 Complete. Found {len(documents)} high-value document chunks. ---")

        # --- Stage 2: General Website Crawl (only if stage 1 found little) ---
        if len(documents) < 5:
            print("\n    --- Stage 2: Performing General Website Crawl ---")
            general_urls_to_visit = {base_url}
            general_docs_found = 0
            
            while general_docs_found < config.NO_OF_WEBSITE_PAGES_TO_SCRAPE and general_urls_to_visit:
                url = general_urls_to_visit.pop()
                if url in visited_urls:
                    continue
                
                visited_urls.add(url)
                
                if url.lower().endswith('.pdf'):
                    continue

                try:
                    time.sleep(random.uniform(1, 2))
                    response = requests.get(url, headers=self.headers, timeout=10, verify=False)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')

                    page_text = self._get_clean_text(soup)
                    if page_text:
                        print(f"    ✅ Scraped General Page: {url}")
                        documents.append(Document(page_content=page_text, metadata={"company": company_name, "source": url, "type": "website_general"}))
                        general_docs_found += 1
                    
                    new_links = self._find_links(soup, base_url, base_domain)
                    general_urls_to_visit.update(new_links - visited_urls)

                except Exception as e:
                    print(f"    ⚠️  Could not scrape {url} in Stage 2: {e}")

        print(f"\n  -> Finished scraping. Found {len(documents)} total document chunks from {company_name}'s website.")
        return documents

    def close(self):
        """A method for consistency with other scrapers."""
        print("  -> Website scraper session closed.")
