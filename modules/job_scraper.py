# modules/job_board_scraper.py

import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

import config

class JobBoardScraper:
    """
    Scrapes job postings from various UAE job boards for a given company,
    returning LangChain Documents.
    """

    def __init__(self):
        """Initializes the scraper with a Selenium WebDriver."""
        options = uc.ChromeOptions()
        # The next line is optional, it can help in some environments
        # options.add_argument('--headless') 
        self.driver = uc.Chrome(use_subprocess=True, options=options)
        self.wait = WebDriverWait(self.driver, 20) # Increased wait time for more reliability

    def _get_clean_text(self, soup):
        """Extracts clean, meaningful text from a BeautifulSoup object."""
        for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _handle_bayt_popups(self):
        """Handles cookie consent banners on Bayt.com."""
        try:
            # Short wait for the cookie banner
            cookie_wait = WebDriverWait(self.driver, 5)
            accept_button = cookie_wait.until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler')))
            accept_button.click()
            print("      -> Accepted Bayt cookie policy.")
            time.sleep(1) # Small delay after clicking
        except TimeoutException:
            print("      -> No Bayt cookie banner found or it was already handled.")
        except Exception as e:
            print(f"      ⚠️  Could not handle Bayt popup: {e}")

    def _scrape_bayt(self, company_name):
        """
        REVISED: Navigates to Bayt.com, handles popups, uses the search bar,
        and then scrapes the results.
        """
        documents = []
        try:
            url = "https://www.bayt.com/en/uae/jobs/"
            print(f"    -> Navigating to Bayt.com to search for '{company_name}'")
            self.driver.get(url)

            # Handle any popups like cookie consent
            self._handle_bayt_popups()

            # Find the search input, clear it, type the company name, and submit
            search_input = self.wait.until(EC.element_to_be_clickable((By.ID, 'text_search')))
            search_input.clear()
            search_input.send_keys(f'"{company_name}"')
            search_input.send_keys(Keys.RETURN)
            
            # Wait for search results to load by checking for a known element
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.list.is-basic")))
            print("      -> Search submitted. Scraping results...")

            job_links = self.driver.find_elements(By.CSS_SELECTOR, 'li[data-js-job] h2 a')
            urls = [link.get_attribute('href') for link in job_links[:config.NO_OF_JOBS_TO_SCRAPE]]

            if not urls:
                print(f"      -> No job listings found for '{company_name}' on Bayt.")
                return []

            for job_url in urls:
                try:
                    self.driver.get(job_url)
                    self.wait.until(EC.presence_of_element_located((By.ID, 'job_description_and_requirements')))
                    
                    title_element = self.driver.find_element(By.CSS_SELECTOR, 'h1.h3')
                    job_title = title_element.text.strip()
                    
                    desc_element = self.driver.find_element(By.ID, 'job_description_and_requirements')
                    job_description = self._get_clean_text(BeautifulSoup(desc_element.get_attribute('outerHTML'), 'html.parser'))

                    content = f"Job Title: {job_title}\n\nJob Description:\n{job_description}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"company": company_name, "source": "bayt.com", "type": "job_posting", "url": job_url}
                    ))
                    print(f"      ✅ Scraped job: {job_title}")
                except Exception as e:
                    print(f"      ⚠️ Could not scrape job detail from Bayt URL {job_url}: {e}")
                    
        except Exception as e:
            print(f"    ⚠️ Error scraping Bayt for '{company_name}': {e}")
        return documents

    def _scrape_naukri_gulf(self, company_name):
        """
        REVISED: Navigates to NaukriGulf.com, handles popups, uses the search bar,
        and then scrapes the results.
        """
        documents = []
        try:
            url = "https://www.naukrigulf.com/jobs-in-uae"
            print(f"    -> Navigating to NaukriGulf to search for '{company_name}'")
            self.driver.get(url)

            # Find search input, type company name, and press Enter
            search_input = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[placeholder='Skills, Designations, Companies']")))
            search_input.clear()
            search_input.send_keys(f'"{company_name}"')
            
            # Click the search button
            search_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Search']")))
            search_button.click()

            # Wait for results to load
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "job-card-container")))
            print("      -> Search submitted. Scraping results...")

            job_cards = self.driver.find_elements(By.CLASS_NAME, 'job-card-container')[:config.NO_OF_JOBS_TO_SCRAPE]
            
            if not job_cards:
                print(f"      -> No job listings found for '{company_name}' on NaukriGulf.")
                return []

            for card in job_cards:
                try:
                    soup = BeautifulSoup(card.get_attribute('outerHTML'), 'html.parser')
                    job_title_element = soup.select_one('h2.job-card-title')
                    job_title = job_title_element.text.strip() if job_title_element else "N/A"
                    
                    job_description = self._get_clean_text(soup)

                    content = f"Job Title: {job_title}\n\nJob Information:\n{job_description}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"company": company_name, "source": "naukrigulf.com", "type": "job_posting"}
                    ))
                    print(f"      ✅ Scraped job: {job_title}")
                except Exception as e:
                    print(f"      ⚠️ Could not process job card from NaukriGulf: {e}")

        except Exception as e:
            print(f"    ⚠️ Error scraping NaukriGulf for '{company_name}': {e}")
        return documents

    def scrape_jobs(self, company_name):
        """
        Iterates through configured job boards and scrapes job postings.
        """
        print(f"  -> Searching for job postings for '{company_name}'...")
        all_documents = []
        
        all_documents.extend(self._scrape_bayt(company_name))
        all_documents.extend(self._scrape_naukri_gulf(company_name))

        if not all_documents:
            print(f"  -> No job postings found for '{company_name}' on the targeted boards.")
            
        return all_documents

    def close(self):
        """Closes the Selenium driver."""
        if self.driver:
            self.driver.quit()
            print("\nBrowser for job board scraping closed.")
