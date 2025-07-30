# modules/linkedin_scraper.py

import json
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.docstore.document import Document

import config

class LinkedInScraper:
    """Scrapes data from LinkedIn company pages and returns LangChain Documents."""

    def __init__(self):
        self.driver = self._setup_driver_with_cookies()

    def _setup_driver_with_cookies(self):
        """Initializes a browser and logs in using a cookie file."""
        print("🚀 Initializing WebDriver and loading cookies for LinkedIn...")
        try:
            with open(config.COOKIES_FILE, "r") as f:
                cookies = json.load(f)
        except FileNotFoundError:
            print(f"\n❌ FATAL ERROR: {config.COOKIES_FILE} not found. Please run cookie_generator.py first.")
            return None
        
        driver = uc.Chrome(use_subprocess=True)
        driver.get("https://www.linkedin.com/")
        for cookie in cookies:
            if 'expiry' in cookie:
                cookie['expiry'] = int(cookie['expiry'])
            driver.add_cookie(cookie)
        
        print("  -> Cookies loaded into browser.")
        driver.get("https://www.linkedin.com/feed/")
        WebDriverWait(driver, 15).until(lambda d: "Feed" in d.title or "Sign In" in d.title)
        
        if "Feed" not in driver.title:
            print("❌ LOGIN FAILED. Cookies might be invalid or expired.")
            driver.quit()
            return None
            
        print("✅ Successfully logged into LinkedIn.")
        return driver

    def _human_like_delay(self):
        time.sleep(random.uniform(2.5, 4.5))

    def scrape_page(self, company_name, company_url):
        """Scrapes LinkedIn and returns a list of LangChain Document objects."""
        if not self.driver:
            print("Driver not initialized. Aborting scrape.")
            return []

        documents = []
        wait = WebDriverWait(self.driver, 10)
        
        # Scrape 'About' page
        try:
            about_url = company_url.rstrip('/') + '/about/'
            self.driver.get(about_url)
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
            about_text = self.driver.find_element(By.TAG_NAME, 'body').text
            documents.append(Document(
                page_content=about_text,
                metadata={"company": company_name, "source": company_url, "type": "about"}
            ))
            print("    ✅ Scraped 'About' page.")
        except Exception:
            print("    ⚠️ Could not scrape 'About' page.")
        
        # Scrape 'Posts' page
        try:
            posts_url = company_url.rstrip('/') + '/posts/'
            self.driver.get(posts_url)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "scaffold-finite-scroll__content")))
            for _ in range(2):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self._human_like_delay()
            post_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'update-components-text')]")[:config.NO_OF_POSTS_TO_SCRAPE]
            for post in post_elements:
                try:
                    more_button = post.find_element(By.CSS_SELECTOR, ".update-components-text__see-more")
                    self.driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.5)
                except NoSuchElementException: pass
            
            for post in post_elements:
                documents.append(Document(
                    page_content=post.text,
                    metadata={"company": company_name, "source": company_url, "type": "post"}
                ))
            print(f"    ✅ Scraped {len(post_elements)} posts.")
        except Exception:
            print("    -> No posts section found or error during scraping.")

        # Scrape 'Jobs' page
        try:
            jobs_url = company_url.rstrip('/') + '/jobs/'
            self.driver.get(jobs_url)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'jobs-search-results-list')))
            job_elements = self.driver.find_elements(By.CSS_SELECTOR, '.job-card-list__title')
            for job in job_elements:
                documents.append(Document(
                    page_content=f"Hiring for: {job.text.strip()}",
                    metadata={"company": company_name, "source": company_url, "type": "job"}
                ))
            print(f"    ✅ Scraped {len(job_elements)} jobs.")
        except Exception:
            print("    -> No jobs section found or error during scraping.")
            
        return documents

    def close(self):
        """Closes the Selenium driver."""
        if self.driver:
            self.driver.quit()
            print("\nBrowser closed.")
