# main.py

"""
Main orchestrator for the Fintech Founder Finder pipeline.
Use command-line arguments to control which steps are executed.
"""

import json
import argparse
import os
import pandas as pd
import io
import asyncio
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import utils
from modules.url_finder import URLFinder
from modules.linkedin_scraper import LinkedInScraper
from modules.website_scraper import WebsiteScraper
from modules.news_scraper import NewsScraper
from modules.app_scraper import AppScraper
from modules.analysis_engine import AnalysisEngine
from modules.deep_search_scraper import DeepSearchScraper

# --- Driver Pool Management ---

def create_driver_pool(size):
    """Creates a pool of reusable Selenium driver instances."""
    print(f"🚀 Initializing a pool of {size} browser instances...")
    driver_pool = queue.Queue()
    for _ in range(size):
        # Each scraper can create a driver; we'll use LinkedInScraper as the creator
        scraper = LinkedInScraper() 
        if scraper.driver:
            driver_pool.put(scraper.driver)
        else:
            print("❌ Failed to create a driver for the pool. The pipeline may fail.")
    print("✅ Browser pool initialized.")
    return driver_pool

def shutdown_driver_pool(driver_pool):
    """Closes all drivers in the pool."""
    print("\n shutting down all browsers in the pool...")
    while not driver_pool.empty():
        driver = driver_pool.get()
        driver.quit()
    print("✅ Browser pool shut down.")

# --- Helper functions for parallel execution ---

def scrape_linkedin_for_company(args):
    """Helper function to run LinkedIn scraping in a thread using a pooled driver."""
    company_name, linkedin_url, driver_pool = args
    driver = driver_pool.get() # Borrow a driver
    try:
        scraper = LinkedInScraper(driver=driver)
        documents = scraper.scrape_page(company_name, linkedin_url)
        return documents
    finally:
        driver_pool.put(driver) # Return the driver to the pool

async def scrape_website_for_company_async(args):
    """Async helper for website scraping."""
    company_name, website_url = args
    scraper = WebsiteScraper()
    documents = await scraper.scrape_website(company_name, website_url)
    return documents

def scrape_news_for_company(args):
    """Helper function to run news scraping in a thread using a pooled driver."""
    company_name, driver_pool = args
    driver = driver_pool.get() # Borrow a driver
    try:
        scraper = NewsScraper(driver=driver)
        documents = scraper.scrape_articles(company_name)
        return documents
    finally:
        driver_pool.put(driver) # Return the driver

def scrape_apps_for_company(company_name):
    """Helper function to run app scraping in a thread."""
    scraper = AppScraper()
    documents = scraper.scrape_apps(company_name)
    return documents

# --- Main pipeline steps ---

def step_1_find_urls(df, client):
    """Finds and verifies LinkedIn and website URLs for companies."""
    print("\n--- Step 1: Finding LinkedIn & Website URLs ---")
    finder = URLFinder(openai_client=client)
    enriched_df = finder.process_companies(df)
    enriched_df.to_csv(config.OUTPUT_CSV_LINKEDIN, index=False)
    print(f"\n✅ Step 1 Complete. Enriched data saved to '{config.OUTPUT_CSV_LINKEDIN}'")

def run_scraping_in_parallel(worker_function, tasks, max_workers):
    """Generic function to run scraping tasks in a thread pool."""
    all_documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_function, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                # FIX: Ensure that only non-None results are processed
                if result:
                    # If the result is a list, extend. If it's a single item, append.
                    if isinstance(result, list):
                        all_documents.extend(result)
                    else:
                        all_documents.append(result)
            except Exception as e:
                print(f"❌ A task generated an exception: {e}")
    return all_documents

async def run_async_scraping_in_parallel(worker_function, tasks):
    """Generic function to run async scraping tasks."""
    all_documents = []
    results = await asyncio.gather(*[worker_function(task) for task in tasks])
    for result in results:
        if result:
            if isinstance(result, list):
                all_documents.extend(result)
            else:
                all_documents.append(result)
    return all_documents

def step_2_scrape_linkedin(df, vector_store, driver_pool):
    """Scrapes LinkedIn company data in parallel and adds it to the vector store."""
    print("\n--- Step 2: Scraping LinkedIn Company Profiles (Parallel) ---")
    tasks = [(row['Cleaned Name'], row['linkedin_url'], driver_pool) for _, row in df.iterrows()]
    all_documents = run_scraping_in_parallel(scrape_linkedin_for_company, tasks, config.SELENIUM_MAX_WORKERS)
    
    if all_documents:
        vector_store.add_documents(all_documents)
        print("\n✅ Step 2 Complete. LinkedIn data vectorized.")

def step_3_scrape_websites(df, vector_store):
    """Scrapes company websites asynchronously and adds them to the vector store."""
    print("\n--- Step 3: Scraping Company Websites (Async) ---")
    tasks = [(row['Cleaned Name'], row.get('website_url')) for _, row in df.iterrows()]
    all_documents = asyncio.run(run_async_scraping_in_parallel(scrape_website_for_company_async, tasks))

    if all_documents:
        vector_store.add_documents(all_documents)
        print("\n✅ Step 3 Complete. Website data vectorized.")

def step_4_scrape_news(df, vector_store, driver_pool):
    """Scrapes news articles in parallel and adds them to the vector store."""
    print("\n--- Step 4: Scraping News Articles (Parallel) ---")
    tasks = [(row['Cleaned Name'], driver_pool) for _, row in df.iterrows()]
    all_documents = run_scraping_in_parallel(scrape_news_for_company, tasks, config.SELENIUM_MAX_WORKERS)
    
    if all_documents:
        vector_store.add_documents(all_documents)
        print("\n✅ Step 4 Complete. News data vectorized.")

def step_5_scrape_apps(df, vector_store):
    """Scrapes app stores in parallel and adds findings to the vector store."""
    print("\n--- Step 5: Scraping App Stores (Parallel) ---")
    tasks = [row['Cleaned Name'] for _, row in df.iterrows()]
    all_documents = run_scraping_in_parallel(scrape_apps_for_company, tasks, config.NETWORK_MAX_WORKERS)

    if all_documents:
        vector_store.add_documents(all_documents)
        print("\n✅ Step 5 Complete. App store data vectorized.")

def step_7_scrape_deep_web(df, vector_store, driver_pool):
    """
    Performs deep web scraping using a two-stage process to respect API rate limits
    while maximizing scraping concurrency.
    """
    print("\n--- Step 7: Deep Web Intelligence Scraping (Pipelined) ---")
    scraper = DeepSearchScraper()
    
    # --- Stage 1: Sequentially fetch all search results to respect rate limits ---
    all_urls_to_scrape = []
    scraped_urls = set()

    print(f"    -> Stage 1: Sequentially fetching search results for {len(df)} companies...")
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        print(f"        -> Generating and fetching queries for: {company_name}")
        queries = scraper.generate_queries(company_name)
        for query_info in queries:
            query_info['company_name'] = company_name 
            time.sleep(config.BRAVE_API_RATE_LIMIT)
            search_results = scraper.search_brave(query_info)
            
            for result in search_results:
                url = result.get("url")
                if url and url not in scraped_urls:
                    scraped_urls.add(url)
                    all_urls_to_scrape.append({
                        "url": url,
                        "doc_type": result.get("doc_type"),
                        "company_name": company_name,
                        "driver_pool": driver_pool
                    })

    if not all_urls_to_scrape:
        print("\n- No unique URLs found from deep search to scrape.")
        return

    # --- Stage 2: Scrape all collected URLs in parallel ---
    print(f"\n    -> Stage 2: Concurrently scraping {len(all_urls_to_scrape)} unique URLs...")
    all_documents = run_scraping_in_parallel(scraper.scrape_single_url, all_urls_to_scrape, config.SELENIUM_MAX_WORKERS)
    
    if all_documents:
        print(f"\n  -> Deep web search complete. Found {len(all_documents)} documents.")
        vector_store.add_documents(all_documents)
        print("\n✅ Step 7 Complete. Deep web data vectorized.")


def step_6_analyze_company(company_name, llm, vector_store):
    """Analyzes a single company using the LangChain RAG chain."""
    print(f"\n--- Step 6: Analyzing '{company_name}' ---")
    engine = AnalysisEngine(llm=llm, vector_store=vector_store)
    
    analysis_result = engine.analyze(company_name)
    if analysis_result and "error" not in analysis_result:
        if not os.path.exists(config.ANALYSIS_OUTPUT_DIR):
            os.makedirs(config.ANALYSIS_OUTPUT_DIR)
            print(f"   -> Created directory: {config.ANALYSIS_OUTPUT_DIR}")

        output_filename = os.path.join(config.ANALYSIS_OUTPUT_DIR, f"{company_name.replace(' ', '_')}_analysis.json")
        
        with open(output_filename, "w") as f:
            json.dump(analysis_result, f, indent=4)
        print(f"\n✅ Analysis saved to '{output_filename}'")
        print("\n--- Generated Intelligence ---")
        print(json.dumps(analysis_result, indent=4))
    else:
        print(f"❌ Analysis failed for {company_name}. Result: {analysis_result}")


def main():
    """Main function to parse arguments and run the selected pipeline steps."""
    parser = argparse.ArgumentParser(description="Fintech Founder Finder Pipeline")
    parser.add_argument('--find-urls', action='store_true', help="Run Step 1: Find LinkedIn and website URLs.")
    parser.add_argument('--scrape-linkedin', action='store_true', help="Run Step 2: Scrape LinkedIn company pages.")
    parser.add_argument('--scrape-websites', action='store_true', help="Run Step 3: Scrape company websites and vectorize.")
    parser.add_argument('--scrape-news', action='store_true', help="Run Step 4: Scrape news articles and vectorize.")
    parser.add_argument('--scrape-apps', action='store_true', help="Run Step 5: Scrape App Stores and vectorize.")
    parser.add_argument('--scrape-deep-web', action='store_true', help="Run Step 7: Scrape deep web for partnerships, forums, etc.")
    parser.add_argument('--analyze', type=str, metavar='COMPANY_NAME', help="Run Step 6: Analyze a specific company.")
    parser.add_argument('--all-scrape', action='store_true', help="Run all scraping steps (2, 3, 4, 5, 7) in parallel.")
    
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    # --- START: TEMPORARY DEBUGGING BLOCK ---
    DEBUG_COMPANIES = ["Tabby", "Tamara"] 
    # --- END: TEMPORARY DEBUGGING BLOCK ---

    openai_client = None
    if args.find_urls:
        openai_client = utils.get_openai_client()
        if not openai_client: return
        
        initial_df = utils.load_and_clean_companies(config.INPUT_CSV_ORIGINAL)
        if initial_df is not None:
            step_1_find_urls(initial_df, openai_client)

    # --- Scraping Block ---
    selenium_steps = args.scrape_linkedin or args.scrape_news or args.scrape_deep_web or args.all_scrape
    other_scraping_steps = args.scrape_websites or args.scrape_apps
    
    if selenium_steps or other_scraping_steps:
        enriched_df = None
        if 'DEBUG_COMPANIES' in locals() and DEBUG_COMPANIES:
            print(f"--- ⚠️  RUNNING IN DEBUG MODE FOR: {', '.join(DEBUG_COMPANIES)} ---")
            debug_data = {'Institution Name': DEBUG_COMPANIES}
            enriched_df = utils.load_and_clean_companies(io.StringIO(pd.DataFrame(debug_data).to_csv(index=False)))
            enriched_df['linkedin_url'] = [f'https://www.linkedin.com/company/{name.lower()}' for name in DEBUG_COMPANIES]
            enriched_df['website_url'] = [f'https://www.{name.lower()}.ai' for name in DEBUG_COMPANIES]
        else:
            enriched_df = utils.load_enriched_data(config.OUTPUT_CSV_LINKEDIN)

        if enriched_df is None:
            print("Cannot run scraping. Please run with --find-urls first to generate the required input file.")
            return
        
        sample_df = enriched_df
        print(f"\nWill now process {len(sample_df)} companies for scraping:")
        for name in sample_df['Cleaned Name']: print(f"  - {name}")

        vector_store = utils.get_vector_store()
        driver_pool = None
        
        try:
            if selenium_steps:
                driver_pool = create_driver_pool(config.SELENIUM_MAX_WORKERS)

            if args.scrape_linkedin or args.all_scrape:
                step_2_scrape_linkedin(sample_df, vector_store, driver_pool)

            if args.scrape_websites or args.all_scrape:
                step_3_scrape_websites(sample_df, vector_store)
            
            if args.scrape_news or args.all_scrape:
                step_4_scrape_news(sample_df, vector_store, driver_pool)
            
            if args.scrape_apps or args.all_scrape:
                step_5_scrape_apps(sample_df, vector_store)
            
            if args.scrape_deep_web or args.all_scrape:
                step_7_scrape_deep_web(sample_df, vector_store, driver_pool)
            
            print("\n💾 Saving final knowledge base to disk...")
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("✅ Knowledge base saved.")

        finally:
            if driver_pool:
                shutdown_driver_pool(driver_pool)


    if args.analyze:
        if not os.path.exists(config.FAISS_INDEX_PATH):
             print("Knowledge base not found. Please run a scrape command first.")
             return
        
        llm = utils.get_llm()
        if not llm: return
        
        vector_store = utils.get_vector_store()
        step_6_analyze_company(args.analyze, llm, vector_store)

    print("\n\n🎉 --- Pipeline Finished --- 🎉")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
