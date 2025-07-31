# modules/linkedin_scraper.py

import json
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.docstore.document import Document

import config

class LinkedInScraper:
    """
    Scrapes data from a LinkedIn company's 'About' and 'Posts' pages.
    Can be initialized with a pre-existing driver or create its own.
    """

    def __init__(self, driver=None):
        """
        Initializes the scraper.
        If a driver is provided, it uses it. Otherwise, it creates a new one.
        """
        if driver:
            self.driver = driver
            self.owns_driver = False
        else:
            self.driver = self._setup_driver_with_cookies()
            self.owns_driver = True

    def _setup_driver_with_cookies(self):
        """Initializes a browser and logs in using a cookie file."""
        print("  -> (LinkedInScraper) Creating new driver instance...")
        try:
            with open(config.COOKIES_FILE, "r") as f:
                cookies = json.load(f)
        except FileNotFoundError:
            print(f"\n❌ FATAL ERROR: {config.COOKIES_FILE} not found. Please run cookie_generator.py first.")
            return None
        
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        driver = uc.Chrome(options=options, use_subprocess=True)
        
        driver.get("https://www.linkedin.com/")
        for cookie in cookies:
            if 'expiry' in cookie:
                cookie['expiry'] = int(cookie['expiry'])
            driver.add_cookie(cookie)
        
        driver.get("https://www.linkedin.com/feed/")
        try:
            WebDriverWait(driver, 15).until(lambda d: "Feed" in d.title or "Sign In" in d.title)
        except TimeoutException:
            print("❌ Timed out waiting for the feed page to load. Cookies might be invalid.")
            driver.quit()
            return None

        if "Feed" not in driver.title:
            print("❌ LOGIN FAILED. Cookies might be invalid or expired.")
            driver.quit()
            return None
            
        print("✅ (LinkedInScraper) New driver logged into LinkedIn.")
        return driver

    def _human_like_delay(self):
        """Pauses execution for a random duration to mimic human behavior."""
        time.sleep(random.uniform(2.5, 4.5))

    def scrape_page(self, company_name, company_url):
        """
        Main orchestration method. Scrapes the company's 'About' and 'Posts' pages.
        """
        if not self.driver:
            print("Driver not initialized. Aborting scrape.")
            return []

        if not company_url or not isinstance(company_url, str) or "linkedin.com/company/" not in company_url:
            print(f"    ⚠️  Invalid or missing LinkedIn URL for '{company_name}'. Skipping.")
            return []

        documents = []
        wait = WebDriverWait(self.driver, 10)
        
        # --- Scrape Company 'About' Page ---
        try:
            about_url = company_url.rstrip('/') + '/about/'
            self.driver.get(about_url)
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
            about_text = self.driver.find_element(By.TAG_NAME, 'body').text
            documents.append(Document(
                page_content=about_text,
                metadata={"company": company_name, "source": about_url, "type": "company_about"}
            ))
            print(f"    ✅ Scraped 'About' page for {company_name}.")
        except Exception as e:
            print(f"    ⚠️ Could not scrape 'About' page for {company_name}: {e}")
        
        self._human_like_delay()

        # --- Scrape Company 'Posts' Page ---
        posts_url = company_url.rstrip('/') + '/posts/'
        self.driver.get(posts_url)
        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "scaffold-finite-scroll__content")))
            
            for _ in range(2): 
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self._human_like_delay()
            
            post_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'update-components-text')]")[:config.NO_OF_POSTS_TO_SCRAPE]

            if not post_elements:
                print(f"    -> No posts found on the page for {company_name}.")
                return documents

            for post in post_elements:
                try:
                    more_button = post.find_element(By.CSS_SELECTOR, ".update-components-text__see-more")
                    self.driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.5)
                except NoSuchElementException:
                    pass 
            
            for post in post_elements:
                documents.append(Document(
                    page_content=post.text,
                    metadata={"company": company_name, "source": posts_url, "type": "company_post"}
                ))
            print(f"    ✅ Scraped {len(post_elements)} company posts for {company_name}.")

        except TimeoutException:
            print(f"    -> No 'Posts' section found for {company_name}, or it failed to load. Skipping.")
        except Exception as e:
            print(f"    -> An unexpected error occurred during post scraping for {company_name}: {e}")

        return documents

    def close(self):
        """Closes the Selenium driver only if this instance created it."""
        if self.driver and self.owns_driver:
            self.driver.quit()
            print("\nBrowser instance closed.")
