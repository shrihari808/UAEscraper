
Automated URL Discovery: Employs an LLM-driven approach to accurately find official corporate websites and LinkedIn company pages from a simple list of company names. [cite: modules/url_finder.py]

Comprehensive Multi-Source Scraping:

LinkedIn: Scrapes company 'About' pages, recent posts, and active job listings to understand company culture, recent activities, and growth signals. [cite: modules/linkedin_scraper.py]

Websites: Performs a sophisticated two-stage crawl, prioritizing high-value documents like annual reports, investor relations, and press releases. It also intelligently scrapes general website content and can extract text from PDF documents. [cite: modules/website_scraper.py]

News Articles: Gathers recent news from a curated list of credible financial news sites to capture market sentiment and recent developments. [cite: modules/news_scraper.py]

App Stores: Searches and scrapes application data from both the Google Play Store and Apple App Store to analyze mobile presence and user feedback. [cite: modules/app_scraper.py]

Job Boards: Scrapes job postings from popular regional platforms like Bayt.com and NaukriGulf.com to identify specific hiring signals and technology needs. [cite: modules/job_scraper.py]

AI-Powered Analysis Engine: Utilizes a Retrieval-Augmented Generation (RAG) model to analyze the aggregated data. It performs targeted vector searches to build a rich context and generates a structured JSON report with key insights. [cite: modules/analysis_engine.py]

Vectorized Knowledge Base: Creates and maintains a local FAISS vector store using financial-domain-specific embeddings (ProsusAI/finbert) for efficient and relevant similarity searches on the scraped data. [cite: utils.py]

Modular and Configurable: The project is organized into distinct, easy-to-understand modules. A central config.py file allows for easy customization of scraping parameters, models, and target sources. [cite: config.py]

How It Works: The Pipeline
The project operates as a multi-step pipeline, orchestrated by main.py. Each step can be run independently or in sequence.

URL Finding: The pipeline starts with a list of company names in combined_institutions.csv. The URLFinder module uses the Brave Search API and an LLM to find their official websites and LinkedIn URLs. The enriched data is saved to institutions_linkedin.csv.

Scraping: The various scraper modules (linkedin_scraper, website_scraper, etc.) are invoked to gather unstructured text data from the URLs found in the previous step. This data is converted into LangChain Document objects, preserving metadata like the source URL and company name.

Vectorization: The collected documents are processed using the ProsusAI/finbert embedding model, which is specialized for financial text. The resulting vectors are stored in a FAISS vector store located in the vectorstorage/ directory. This creates a searchable, local knowledge base.

Analysis: When the --analyze command is run, the AnalysisEngine performs multiple, targeted vector searches against the knowledge base to build a rich, relevant context for the specified company. This context is then fed into a powerful LLM (gpt-4o-mini by default) with a structured prompt. The LLM fills in a detailed JSON template, and the final analysis is saved in the analysisJsons/ directory.

Setup and Installation
Prerequisites
Python 3.8+

A modern web browser with its corresponding WebDriver (the script uses undetected-chromedriver, which handles this automatically for Chrome).

API keys for:

OpenAI: For analysis and URL finding.

Brave Search: For web searches.

Installation Steps
Clone the repository:

git clone [https://github.com/shrihari808/uaescraper.git](https://github.com/shrihari808/uaescraper.git)
cd uaescraper

Create and activate a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Set up your environment variables:

Create a file named .env in the root directory of the project.

Add your API keys to this file. This keeps them secure and out of the code.

OPENAI_API_KEY="your_openai_api_key"
BRAVE_API_KEY="your_brave_search_api_key"

Generate LinkedIn Authentication Cookies:

To scrape LinkedIn without getting blocked, you need to provide session cookies from a logged-in account.

Run the cookie_generator.py script:

python cookie_generator.py

A browser window will open. Log in to your LinkedIn account manually. Once you land on your main feed page, the script will automatically save the necessary cookies to cookies.json. You can then close the browser. [cite: cookie_generator.py]

Usage
The main script main.py is controlled via command-line arguments. You can run each step of the pipeline individually.

Step 1: Find Company URLs
This command processes combined_institutions.csv and creates institutions_linkedin.csv with the discovered URLs.

python main.py --find-urls

Step 2: Scrape Data Sources
You can run each scraper individually. The script will use institutions_linkedin.csv as input and save the vectorized data to the vectorstorage/ directory.

# Scrape only LinkedIn
python main.py --scrape-linkedin

# Scrape only websites
python main.py --scrape-websites

# Scrape news, apps, and jobs
python main.py --scrape-news --scrape-apps --scrape-jobs

# Run all scraping steps at once
python main.py --scrape-linkedin --scrape-websites --scrape-news --scrape-apps --scrape-jobs

Note: The SAMPLE_SIZE in config.py is set to 5 by default to avoid long run times. Set it to None to process all companies.

Step 3: Analyze a Company
After building the knowledge base with the scraping commands, you can analyze any company. The company name should match the "Cleaned Name" in the CSV files.

python main.py --analyze "Amlak Finance"

The output will be a detailed JSON file saved in the analysisJsons/ directory (e.g., Amlak_Finance_analysis.json) and printed to the console.

Configuration
The config.py file allows you to customize the pipeline's behavior without changing the source code.

API_KEYS: Loaded from the .env file.

FILE_PATHS: Define input and output file locations.

SCRAPING_PARAMETERS: Control the number of items to scrape (e.g., NO_OF_POSTS_TO_SCRAPE, NO_OF_NEWS_ARTICLES_TO_SCRAPE).

SAMPLE_SIZE: Set the number of companies to process in a single run. Useful for testing. Set to None to run on the entire list.

EMBEDDING_MODEL: The HuggingFace model to use for vectorization. Defaults to ProsusAI/finbert.

LLM_MODEL: The OpenAI model to use for analysis. Defaults to gpt-4o-mini.

CREDIBLE_NEWS_SITES: A list of trusted domains for news scraping.

Project Structure
/
├── .gitignore
├── analysisJsons/              # Output directory for JSON analysis reports
│   └── Amlak_Finance_analysis.json
├── combined_institutions.csv   # Initial input list of companies
├── config.py                   # Central configuration file
├── cookie_generator.py         # Script to generate LinkedIn cookies
├── cookies.json                # Stores LinkedIn session cookies
├── institutions_linkedin.csv   # Output of the URL finding step
├── main.py                     # Main orchestrator script for the pipeline
├── modules/                    # Core logic for each pipeline step
│   ├── analysis_engine.py
│   ├── app_scraper.py
│   ├── job_scraper.py
│   ├── linkedin_scraper.py
│   ├── news_scraper.py
│   ├── url_finder.py
│   └── website_scraper.py
├── requirements.txt            # Python dependencies
├── utils.py                    # Utility functions (data loading, LLM clients, etc.)
└── vectorstorage/              # Output directory for the FAISS vector store
    └── ...
