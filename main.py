# main.py

"""
Main orchestrator for the Fintech Founder Finder pipeline.
Use command-line arguments to control which steps are executed.
"""

import json
import argparse
import os

import config
import utils
from modules.url_finder import URLFinder
from modules.linkedin_scraper import LinkedInScraper
from modules.website_scraper import WebsiteScraper
from modules.news_scraper import NewsScraper
from modules.app_scraper import AppScraper # New import
from modules.analysis_engine import AnalysisEngine

def step_1_find_urls(df, client):
    """Finds and verifies LinkedIn and website URLs for companies."""
    print("\n--- Step 1: Finding LinkedIn & Website URLs ---")
    finder = URLFinder(openai_client=client)
    enriched_df = finder.process_companies(df)
    enriched_df.to_csv(config.OUTPUT_CSV_LINKEDIN, index=False)
    print(f"\n✅ Step 1 Complete. Enriched data saved to '{config.OUTPUT_CSV_LINKEDIN}'")

def step_2_scrape_linkedin(df, vector_store):
    """Scrapes LinkedIn data and adds it to the vector store."""
    print("\n--- Step 2: Scraping LinkedIn Profiles ---")
    scraper = LinkedInScraper()
    if not scraper.driver:
        return
    
    all_documents = []
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        linkedin_url = row['linkedin_url']
        print(f"\nProcessing LinkedIn for: {company_name}")
        documents = scraper.scrape_page(company_name, linkedin_url)
        all_documents.extend(documents)
    scraper.close()

    if all_documents:
        valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
        
        if valid_documents:
            print(f"  -> Found {len(valid_documents)} valid documents to add to the knowledge base.")
            vector_store.add_documents(valid_documents)
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("\n✅ Step 2 Complete. LinkedIn data vectorized and saved.")
        else:
            print("\n- No valid documents with content found from LinkedIn to add to the knowledge base.")

def step_3_scrape_websites(df, vector_store):
    """Scrapes company websites and adds them to the vector store."""
    print("\n--- Step 3: Scraping Company Websites ---")
    scraper = WebsiteScraper()
    
    all_documents = []
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        website_url = row.get('website_url') 
        print(f"\nProcessing Website for: {company_name}")
        documents = scraper.scrape_website(company_name, website_url)
        all_documents.extend(documents)
    scraper.close()

    if all_documents:
        valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
        
        if valid_documents:
            print(f"  -> Found {len(valid_documents)} valid website pages to add to the knowledge base.")
            vector_store.add_documents(valid_documents)
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("\n✅ Step 3 Complete. Website data vectorized and saved.")
        else:
            print("\n- No valid content found from websites to add to the knowledge base.")

def step_4_scrape_news(df, vector_store):
    """Scrapes news articles and adds them to the vector store."""
    print("\n--- Step 4: Scraping News Articles ---")
    scraper = NewsScraper()
    
    all_documents = []
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        print(f"\nProcessing News for: {company_name}")
        documents = scraper.scrape_articles(company_name)
        all_documents.extend(documents)
    scraper.close()

    if all_documents:
        valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
        
        if valid_documents:
            print(f"  -> Found {len(valid_documents)} valid news articles to add to the knowledge base.")
            vector_store.add_documents(valid_documents)
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("\n✅ Step 4 Complete. News data vectorized and saved.")
        else:
            print("\n- No valid news articles with content found to add to the knowledge base.")

def step_5_scrape_apps(df, vector_store):
    """Scrapes app stores and adds findings to the vector store."""
    print("\n--- Step 5: Scraping App Stores (Google Play & Apple) ---")
    scraper = AppScraper()
    
    all_documents = []
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        print(f"\nProcessing App Stores for: {company_name}")
        documents = scraper.scrape_apps(company_name)
        all_documents.extend(documents)
    scraper.close()

    if all_documents:
        # App scraper returns pre-filtered, valid documents
        print(f"  -> Found {len(all_documents)} valid app records to add to the knowledge base.")
        vector_store.add_documents(all_documents)
        vector_store.save_local(config.FAISS_INDEX_PATH)
        print("\n✅ Step 5 Complete. App store data vectorized and saved.")
    else:
        print("\n- No app store data found to add to the knowledge base.")

def step_6_analyze_company(company_name, llm, vector_store):
    """Analyzes a single company using the LangChain RAG chain."""
    print(f"\n--- Step 6: Analyzing '{company_name}' ---")
    engine = AnalysisEngine(llm=llm, vector_store=vector_store)
    
    analysis_result = engine.analyze(company_name)
    if analysis_result:
        if not os.path.exists(config.ANALYSIS_OUTPUT_DIR):
            os.makedirs(config.ANALYSIS_OUTPUT_DIR)
            print(f"   -> Created directory: {config.ANALYSIS_OUTPUT_DIR}")

        output_filename = os.path.join(config.ANALYSIS_OUTPUT_DIR, f"{company_name.replace(' ', '_')}_analysis.json")
        
        with open(output_filename, "w") as f:
            json.dump(analysis_result, f, indent=4)
        print(f"\n✅ Analysis saved to '{output_filename}'")
        print("\n--- Generated Intelligence ---")
        print(json.dumps(analysis_result, indent=4))

def main():
    """Main function to parse arguments and run the selected pipeline steps."""
    parser = argparse.ArgumentParser(description="Fintech Founder Finder Pipeline")
    parser.add_argument('--find-urls', action='store_true', help="Run Step 1: Find LinkedIn and website URLs.")
    parser.add_argument('--scrape-linkedin', action='store_true', help="Run Step 2: Scrape LinkedIn profiles and vectorize.")
    parser.add_argument('--scrape-websites', action='store_true', help="Run Step 3: Scrape company websites and vectorize.")
    parser.add_argument('--scrape-news', action='store_true', help="Run Step 4: Scrape news articles and vectorize.")
    parser.add_argument('--scrape-apps', action='store_true', help="Run Step 5: Scrape App Stores and vectorize.") # New argument
    parser.add_argument('--analyze', type=str, metavar='COMPANY_NAME', help="Run Step 6: Analyze a specific company.")
    
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.find_urls:
        openai_client = utils.get_openai_client()
        if not openai_client: return
        initial_df = utils.load_and_clean_companies(config.INPUT_CSV_ORIGINAL)
        if initial_df is not None:
            step_1_find_urls(initial_df, openai_client)

    # Logic for scraping steps
    if args.scrape_linkedin or args.scrape_websites or args.scrape_news or args.scrape_apps:
        enriched_df = utils.load_enriched_data(config.OUTPUT_CSV_LINKEDIN)
        if enriched_df is None:
            print("Cannot run scraping. Please run with --find-urls first to generate the required input file.")
            return

        if config.SAMPLE_SIZE and len(enriched_df) > config.SAMPLE_SIZE:
            sample_df = enriched_df.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)
        else:
            sample_df = enriched_df
        
        print(f"\nWill now process a sample of {len(sample_df)} companies for scraping:")
        for name in sample_df['Cleaned Name']: print(f"  - {name}")

        vector_store = utils.get_vector_store()
        
        if args.scrape_linkedin:
            linkedin_df = sample_df.dropna(subset=['linkedin_url'])
            linkedin_df = linkedin_df[linkedin_df['linkedin_url'].str.contains('linkedin.com', na=False)]
            if not linkedin_df.empty:
                step_2_scrape_linkedin(linkedin_df, vector_store)
            else:
                print("\n- No valid LinkedIn URLs found in the sample to scrape.")

        if args.scrape_websites:
            website_df = sample_df.dropna(subset=['website_url'])
            website_df = website_df[website_df['website_url'].str.startswith('http', na=False)]
            if not website_df.empty:
                step_3_scrape_websites(website_df, vector_store)
            else:
                print("\n- No valid website URLs found in the sample to scrape.")
        
        if args.scrape_news:
            step_4_scrape_news(sample_df, vector_store)
        
        if args.scrape_apps: # New step execution
            step_5_scrape_apps(sample_df, vector_store)

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
    main()
