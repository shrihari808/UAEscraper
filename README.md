##  Overview

The Fintech Founder Finder pipeline performs multi-source data scraping and uses advanced RAG analysis to generate detailed intelligence reports about fintech companies. It's specifically designed to identify investment opportunities by analyzing company strategies, technology adoption, financial inclusion initiatives, and hiring patterns.

##  Key Features

- **Multi-Source Data Collection**: Scrapes LinkedIn, company websites, news articles, app stores, and job boards
- **AI-Powered URL Discovery**: Uses LLM-based analysis to find and verify company URLs
- **Advanced Document Processing**: Handles PDFs, web pages, and structured data with chunking for optimal analysis
- **RAG-Based Intelligence Analysis**: Generates structured intelligence reports using multi-query retrieval
- **Modular Pipeline**: Run individual steps or full pipeline based on your needs
- **Vector-Based Knowledge Base**: Uses FAISS for efficient document storage and retrieval

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │    Analysis     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • LinkedIn      │───▶│ • URL Discovery │───▶│ • RAG Analysis  │
│ • Websites      │    │ • Web Scraping  │    │ • LLM Insights  │
│ • News Articles │    │ • PDF Processing│    │ • JSON Reports  │
│ • App Stores    │    │ • Vectorization │    │ • Intelligence  │
│ • Job Boards    │    │ • FAISS Storage │    │   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Required APIs and Services
- **OpenAI API Key**: For LLM-based analysis and URL discovery
- **Brave Search API Key**: For web search capabilities
- **LinkedIn Account**: For authenticated scraping (cookies required)

### System Requirements
- Python 3.8+
- Chrome browser (for Selenium automation)
- Sufficient storage for vector database and temporary files

##  Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fintech-founder-finder
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
BRAVE_API_KEY=your_brave_search_api_key_here
```

4. **Generate LinkedIn cookies**
```bash
python cookie_generator.py
```
Follow the interactive prompts to log into LinkedIn and save session cookies.

5. **Prepare your data**
Place your company data in a CSV file named `combined_institutions.csv` with at least an `Institution Name` column.

##  Input Data Format

Your input CSV should contain:
```csv
Institution Name,Sector,Location
"Al Mashreq Al Islami Finance Company PJSC","Financial Services","Dubai, UAE"
"Emirates NBD Bank PJSC","Banking","Dubai, UAE"
```

##  Usage

The pipeline is designed with a modular approach. You can run individual steps or combine them:

### Step 1: Find Company URLs
```bash
python main.py --find-urls
```
Discovers and verifies LinkedIn and website URLs for all companies.

### Step 2: Scrape LinkedIn Data
```bash
python main.py --scrape-linkedin
```
Scrapes company profiles, posts, and job listings from LinkedIn.

### Step 3: Scrape Company Websites
```bash
python main.py --scrape-websites
```
Crawls official websites, processes PDFs, and extracts high-value content.

### Step 4: Scrape News Articles
```bash
python main.py --scrape-news
```
Searches and scrapes relevant news articles from credible sources.

### Step 5: Scrape App Stores
```bash
python main.py --scrape-apps
```
Searches Google Play and Apple App Store for company applications.

### Step 6: Scrape Job Boards
```bash
python main.py --scrape-jobs
```
Searches UAE job boards (Bayt, NaukriGulf) for company job postings.

### Step 7: Generate Intelligence Reports
```bash
python main.py --analyze "Company Name"
```
Generates a structured intelligence report for a specific company.

### Combined Operations
```bash
# Run full data collection pipeline
python main.py --scrape-linkedin --scrape-websites --scrape-news --scrape-apps --scrape-jobs

# Then analyze specific companies
python main.py --analyze "Emirates NBD"
```



## ⚙️ Configuration

Key configuration options in `config.py`:

```python
# Scraping limits
NO_OF_POSTS_TO_SCRAPE = 10
NO_OF_NEWS_ARTICLES_TO_SCRAPE = 2
NO_OF_WEBSITE_PAGES_TO_SCRAPE = 5
NO_OF_JOBS_TO_SCRAPE = 5

# Processing options
SAMPLE_SIZE = 5  # Number of companies to process (None for all)
EMBEDDING_MODEL = 'ProsusAI/finbert'  # Financial text embeddings
LLM_MODEL = "gpt-4o-mini"

# Credible news sources for UAE/MEA region
CREDIBLE_NEWS_SITES = [
    "difc.ae", "fintechfutures.com", "fintechnews.ae",
    "mea-finance.com", "zawya.com", "khaleejtimes.com"
]
```

## 📁 Project Structure

```
fintech-founder-finder/
├── main.py                 # Main pipeline orchestrator
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── cookie_generator.py    # LinkedIn authentication setup
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
│
├── modules/               # Core processing modules
│   ├── url_finder.py     # URL discovery with LLM analysis
│   ├── linkedin_scraper.py    # LinkedIn data extraction
│   ├── website_scraper.py     # Website crawling and PDF processing
│   ├── news_scraper.py        # News article collection
│   ├── app_scraper.py         # Mobile app store search
│   ├── job_scraper.py         # Job board scraping
│   └── analysis_engine.py     # RAG-based intelligence analysis
│
├── analysisJsons/         # Generated intelligence reports
├── vectorstorage/         # FAISS vector database
└── temp_pdfs/            # Temporary PDF storage
```

##  Advanced Features

### PDF Processing
- Automatic PDF download and text extraction
- Page-by-page chunking for better analysis
- Support for annual reports and financial documents

### Smart URL Discovery
- LLM-powered analysis for accurate company page identification
- Brand name vs. formal name matching
- Verification of page availability

### Multi-Query RAG Analysis
- Targeted searches for different intelligence signals
- Large-batch retrieval with company-specific filtering
- Context aggregation from multiple document types

### Robust Error Handling
- SSL certificate bypass for problematic websites
- Retry mechanisms for failed requests
- Graceful degradation when sources are unavailable


## Important Notes

### Rate Limiting and Ethics
- Built-in delays to respect website rate limits
- Targets only publicly available information
- Designed for legitimate business intelligence use

### Data Privacy
- No storage of personal information
- Focus on corporate and public business data
- Cookies stored locally and can be deleted

### Legal Compliance
- Ensure compliance with local data protection laws
- Respect robots.txt and website terms of service
- Use responsibly for legitimate business purposes

##  Troubleshooting

### Common Issues

**LinkedIn Login Issues**
```bash
# Regenerate cookies
python cookie_generator.py
```

**SSL Certificate Errors**
- The scraper automatically bypasses SSL verification for problematic sites
- If issues persist, check your internet connection

**Vector Store Issues**
```bash
# Reset vector database
rm -rf vectorstorage/
```

**Memory Issues with Large Datasets**
- Reduce `SAMPLE_SIZE` in config.py
- Process companies in smaller batches

### Debug Mode
Add verbose logging by modifying the print statements in individual modules or implement Python's logging module for detailed debugging.

##  Future Enhancements

- [ ] Support for additional job boards and regional sources
- [ ] Enhanced financial document analysis capabilities
- [ ] Real-time monitoring and alert systems
- [ ] Integration with CRM and deal flow management systems
- [ ] Advanced sentiment analysis for news and social media
