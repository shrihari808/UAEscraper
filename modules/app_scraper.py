# modules/app_scraper.py

import requests
from google_play_scraper import search as search_play_store, app as app_play_store
from langchain.docstore.document import Document
import config

class AppScraper:
    """
    Searches for and scrapes data from the Google Play Store and Apple App Store
    for a given company, returning LangChain Documents.
    """

    def __init__(self):
        """Initializes the scraper."""
        pass

    def _format_app_details(self, store_name, details):
        """Formats scraped app details into a clean string for the LLM."""
        key_details = {
            "App Name": details.get('title') or details.get('trackName'),
            "Developer": details.get('developer') or details.get('artistName'),
            "Description": details.get('description'),
            "Genre": details.get('primaryGenreName') or details.get('genre'),
            "Rating": details.get('score') or details.get('averageUserRating'),
            "Reviews": details.get('reviews') or details.get('userRatingCount'),
            "Last Updated": details.get('updated') or details.get('currentVersionReleaseDate'),
            "Release Date": details.get('released') or details.get('releaseDate'),
            "Content Rating": details.get('contentRating') or details.get('trackContentRating'),
        }
        
        text_content = f"--- {store_name} App Information ---\n"
        for key, value in key_details.items():
            if value:
                if isinstance(value, (int, float)):
                    value = f"{value:,}"
                if 'Date' in key and isinstance(value, str):
                    value = value.split('T')[0]
                text_content += f"{key}: {value}\n"
        
        return text_content.strip()

    def _search_apple_app_store(self, term, country, limit):
        """Searches the Apple App Store using the official iTunes Search API."""
        url = "https://itunes.apple.com/search"
        params = {
            "term": term,
            "country": country,
            "media": "software",
            "entity": "software,iPadSoftware",
            "limit": limit
        }
        try:
            print(f"    -> Querying iTunes API for '{term}' in country '{country}'...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.RequestException as e:
            print(f"    ⚠️ An error occurred during Apple App Store API call: {e}")
            return []

    def scrape_apps(self, company_name):
        """
        Searches both the Google Play Store and Apple App Store for a company's app,
        scrapes the top result, and returns LangChain Documents.
        """
        print(f"  -> Searching for mobile apps for '{company_name}' in the UAE region...")
        documents = []

        # --- Scrape Google Play Store ---
        try:
            # FIX: Removed the unsupported 'gl' argument. 'country' is the correct parameter.
            search_results = search_play_store(
                query=company_name,
                n_hits=config.NO_OF_APPS_TO_SCRAPE,
                country="ae" 
            )
            
            if search_results:
                app_id = search_results[0]['appId']
                print(f"    -> Found potential Google Play app: {app_id}")
                details = app_play_store(app_id, lang='en', country='ae')
                
                formatted_text = self._format_app_details("Google Play Store", details)
                documents.append(Document(
                    page_content=formatted_text,
                    metadata={"company": company_name, "source": f"google-play-store:{app_id}", "type": "mobile_app"}
                ))
                print(f"    ✅ Scraped Google Play Store for '{details.get('title')}'.")
            else:
                # FIX: Handle case where no results are returned, as no exception is thrown.
                print(f"    -> No Google Play app found for '{company_name}'.")

        except Exception as e:
            # FIX: Broadened exception handling as 'NotFound' is not a valid exception class.
            print(f"    ⚠️ An error occurred during Google Play scraping: {e}")

        # --- Scrape Apple App Store ---
        try:
            search_results = self._search_apple_app_store(
                term=company_name,
                country="ae",
                limit=config.NO_OF_APPS_TO_SCRAPE
            )

            if search_results:
                app_details = search_results[0]
                app_name = app_details.get('trackName')
                app_id = app_details.get('trackId')
                print(f"    -> Found potential Apple App Store app: {app_name} ({app_id})")

                formatted_text = self._format_app_details("Apple App Store", app_details)
                documents.append(Document(
                    page_content=formatted_text,
                    metadata={"company": company_name, "source": f"apple-app-store:{app_id}", "type": "mobile_app"}
                ))
                print(f"    ✅ Scraped Apple App Store for '{app_name}'.")

        except Exception as e:
            print(f"    ⚠️ An unexpected error occurred during Apple App Store processing: {e}")

        if not documents and len(documents) == 0: # Check if still no documents after both stores
            print(f"  -> No mobile apps found for '{company_name}' in either store.")
            
        return documents

    def close(self):
        """A method for consistency with other scrapers."""
        print("  -> App scraper session closed.")
