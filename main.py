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
from modules.news_scraper import NewsScraper
from modules.analysis_engine import AnalysisEngine

def step_1_find_urls(df, client):
    """Finds and verifies LinkedIn URLs for companies."""
    print("\n--- Step 1: Finding LinkedIn URLs ---")
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
        vector_store.add_documents(all_documents)
        vector_store.save_local(config.FAISS_INDEX_PATH)
        print("\n✅ Step 2 Complete. LinkedIn data vectorized and saved.")

def step_3_scrape_news(df, vector_store):
    """Scrapes news articles and adds them to the vector store."""
    print("\n--- Step 3: Scraping News Articles ---")
    scraper = NewsScraper()
    
    all_documents = []
    for _, row in df.iterrows():
        company_name = row['Cleaned Name']
        print(f"\nProcessing News for: {company_name}")
        documents = scraper.scrape_articles(company_name)
        all_documents.extend(documents)
    scraper.close()

    if all_documents:
        vector_store.add_documents(all_documents)
        vector_store.save_local(config.FAISS_INDEX_PATH)
        print("\n✅ Step 3 Complete. News data vectorized and saved.")


def step_4_analyze_company(company_name, llm, vector_store):
    """Analyzes a single company using the LangChain RAG chain."""
    print(f"\n--- Step 4: Analyzing '{company_name}' ---")
    engine = AnalysisEngine(llm=llm, vector_store=vector_store)
    
    analysis_result = engine.analyze(company_name)
    if analysis_result:
        # Create the output directory if it doesn't exist
        if not os.path.exists(config.ANALYSIS_OUTPUT_DIR):
            os.makedirs(config.ANALYSIS_OUTPUT_DIR)
            print(f"   -> Created directory: {config.ANALYSIS_OUTPUT_DIR}")

        # Define the output filename within the new directory
        output_filename = os.path.join(config.ANALYSIS_OUTPUT_DIR, f"{company_name.replace(' ', '_')}_analysis.json")
        
        with open(output_filename, "w") as f:
            json.dump(analysis_result, f, indent=4)
        print(f"\n✅ Analysis saved to '{output_filename}'")
        print("\n--- Generated Intelligence ---")
        print(json.dumps(analysis_result, indent=4))

def main():
    """Main function to parse arguments and run the selected pipeline steps."""
    parser = argparse.ArgumentParser(description="Fintech Founder Finder Pipeline")
    parser.add_argument('--find-urls', action='store_true', help="Run Step 1: Find and verify LinkedIn URLs.")
    parser.add_argument('--scrape-linkedin', action='store_true', help="Run Step 2: Scrape LinkedIn profiles and vectorize.")
    parser.add_argument('--scrape-news', action='store_true', help="Run Step 3: Scrape news articles and vectorize.")
    parser.add_argument('--analyze', type=str, metavar='COMPANY_NAME', help="Run Step 4: Analyze a specific company using the existing knowledge base.")
    
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

    if args.scrape_linkedin or args.scrape_news:
        enriched_df = utils.load_enriched_data(config.OUTPUT_CSV_LINKEDIN)
        if enriched_df is None:
            print("Cannot run scraping. Please run with --find-urls first to generate the required input file.")
            return

        sample_df = enriched_df.sample(n=config.SAMPLE_SIZE, random_state=42) if config.SAMPLE_SIZE and len(enriched_df) > config.SAMPLE_SIZE else enriched_df
        
        print(f"\nWill now process a sample of {len(sample_df)} companies for scraping:")
        for name in sample_df['Cleaned Name']: print(f"  - {name}")

        vector_store = utils.get_vector_store()
        
        if args.scrape_linkedin:
            step_2_scrape_linkedin(sample_df, vector_store)
        
        if args.scrape_news:
            step_3_scrape_news(sample_df, vector_store)

    if args.analyze:
        if not os.path.exists(config.FAISS_INDEX_PATH):
             print("Knowledge base not found. Please run a scrape command (--scrape-linkedin or --scrape-news) first.")
             return
        
        llm = utils.get_llm()
        if not llm: return
        
        vector_store = utils.get_vector_store()
        step_4_analyze_company(args.analyze, llm, vector_store)

    print("\n\n🎉 --- Pipeline Finished --- 🎉")


if __name__ == "__main__":
    main()

"""
 python main.py --find-urls --scrape-linkedin --scrape-news --analyze
 python main.py --help
 python main.py --scrape-linkedin --scrape-news
 python main.py --analyze "Company Name"

"""
