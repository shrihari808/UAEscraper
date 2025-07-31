# modules/analysis_engine.py

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import re

import config

class AnalysisEngine:
    """Analyzes company data by querying multiple vector stores using a LangChain RAG chain."""

    def __init__(self, llm, vector_stores):
        """
        Initializes the engine.
        
        Args:
            llm: The language model to use for analysis.
            vector_stores (dict): A dictionary where keys are index names (e.g., 'linkedin')
                                  and values are the loaded FAISS vector store objects.
        """
        self.llm = llm
        self.vector_stores = vector_stores
        self.chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Builds the LangChain Expression Language (LCEL) chain."""
        
        def get_docs_from_all_stores(info):
            """
            Performs searches across all available vector stores to retrieve a comprehensive
            and accurate set of documents for a specific company.
            """
            company_name = info.get("company_name")
            print(f"  -> Retrieving documents for '{company_name}' from all vector stores...")
            
            all_docs = []
            # Iterate through each loaded vector store
            for store_name, store in self.vector_stores.items():
                if store:
                    # Perform a search on the current store, filtering by company name
                    # to ensure relevance. Using a generous k to get diverse results.
                    retrieved_docs = store.similarity_search(
                        company_name, 
                        k=10, 
                        filter={"company": company_name}
                    )
                    all_docs.extend(retrieved_docs)
                    print(f"    -> Found {len(retrieved_docs)} documents in '{store_name}' index.")

            # Deduplicate the combined documents based on the source URL to avoid redundancy
            unique_docs = list({doc.metadata['source']: doc for doc in all_docs}.values())
            print(f"  -> Total unique documents after deduplication: {len(unique_docs)}")

            # Format the context for the LLM, slicing content to manage token count
            context_with_sources = []
            MAX_CHARS_PER_DOC = 4000 
            for doc in unique_docs:
                source = doc.metadata.get('source', 'N/A')
                content = doc.page_content[:MAX_CHARS_PER_DOC]
                context_with_sources.append(f"Source: {source}\nContent:\n{content}")

            return "\n---\n".join(context_with_sources)


        # --- PROMPT ---
        template = """
        You are a top-tier venture capital analyst and investigative journalist specializing in UAE Fintech.
        Your task is to synthesize the provided data for "{company_name}" to extract deep, actionable intelligence.
        You MUST cite the source for every piece of information you extract where specified in the output format.

        **CONTEXT (Retrieved from multiple specialized Knowledge Bases):**
        Each block of information is preceded by its source URL.
        {context}

        **YOUR CORE OBJECTIVES (Synthesize all data to answer):**
        1.  **Company Metadata:** Name, Company LinkedIn URL, Sector, Location/Freezone, Stage (e.g., startup, growth), Founding Year.
        2.  **Financial Offerings & Products:** What specific financial products do they offer? What is their core value proposition?
        3.  **Strategic Focus & Signals:**
            - **Digital Transformation:** Stage of digital adoption (e.g., use of cloud, AI, mobile-first).
            - **Key Focus Areas:** Target clients (retail, corporate, SME) and industry focus.
            - **Partnerships & M&A:** Any mentions of partnerships, M&A, or key collaborations.
            - **Growth Signals:** Hiring for tech/expansion roles, new product launches, market expansion.
        4.  **People Intelligence:** List ALL people mentioned in the context. Include their full name, role, and a link to their personal LinkedIn profile if available in the context.
        5.  **Competitive Landscape (Implied):** Based on their offerings, who might be their key competitors? (1-2 names).
        6.  **Mobile App Presence:**
            - Specify if they have an app on Google Play and/or Apple App Store.
            - If an app exists, what is its purpose and key features?
            - Specify ratings, reviews, and last updated date if available.
            - If no app is mentioned, state "No mobile app presence found in the provided data."
        7.  **KEY FINTECH VERTICAL ANALYSIS (CRITICAL):**
            - **Primary Focus:** Analyze involvement in: Buy Now, Pay Later (BNPL), Remittance & Wallets, Spend Management, and Wealth Management (WealthTech).
            - **Evidence Requirement:** For each area, provide direct quotes or strong evidence from the context. If no evidence, state "No signal found".
            - **Signal Strength:** Classify the signal as 'Strong', 'Weak', or 'None'.

        **OUTPUT FORMAT:**
        Return a single, clean JSON object.
        - The top-level keys must be: "CompanyMetadata", "FinancialOfferingsAndProducts", "StrategicFocusAndSignals", "PeopleIntelligence", "CompetitiveLandscape", "MobileAppPresence", "FintechVerticalAnalysis".
        - For "CompanyMetadata", return simple key-value pairs (e.g., "Name": "Example Inc."). The keys are: "Name", "LinkedInURL", "Sector", "LocationFreezone", "Stage", "FoundingYear". Do NOT include sources for these fields.
        - For the following keys, provide the value and the source URL in a nested object like this: {{"value": "The extracted information", "source": "The URL source for this specific information"}}:
            - FinancialOfferingsAndProducts
            - StrategicFocusAndSignals (and all its sub-keys like DigitalTransformation, KeyFocusAreas, etc.)
            - CompetitiveLandscape
        - For "PeopleIntelligence", the value should be a list of objects. Each object must have the keys "name", "role", "linkedin_url", and "source".
        - For "MobileAppPresence", provide nested objects for "GooglePlay" and "AppleAppStore". The value for each should be an object containing the app details and a "source" key.
        - For "FintechVerticalAnalysis", provide nested objects for each key area. Each object must have fields for "signal_strength", "evidence", and "source".
        """

        prompt = ChatPromptTemplate.from_template(template)

        class CustomJsonOutputParser(JsonOutputParser):
            def parse(self, text: str):
                """
                Parses the LLM output to extract a JSON object, even if it's
                embedded in markdown code blocks or has other surrounding text.
                """
                try:
                    cleaned_text = text.strip()
                    
                    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
                    match = re.search(code_block_pattern, cleaned_text, re.DOTALL)
                    
                    if match:
                        cleaned_text = match.group(1).strip()
                    
                    json_start = cleaned_text.find('{')
                    json_end = cleaned_text.rfind('}') + 1

                    if json_start == -1 or json_end == 0:
                        raise json.JSONDecodeError("No JSON object markers found in the cleaned output.", cleaned_text, 0)

                    json_str = cleaned_text[json_start:json_end]
                    
                    return json.loads(json_str)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from LLM output. Raw text: '{text}'. Error: {e}")
                    return {"error": "Failed to parse LLM JSON output", "raw_output": text}

        output_parser = CustomJsonOutputParser()

        chain = (
            {"context": get_docs_from_all_stores, "company_name": RunnablePassthrough()}
            | prompt
            | self.llm
            | output_parser
        )
        return chain

    def analyze(self, company_name):
        """Invokes the RAG chain to analyze a company."""
        print(f"🤖 Performing RAG-based analysis for '{company_name}' using LangChain...")
        try:
            # The invoke method now needs a dictionary for the RunnablePassthrough
            result = self.chain.invoke({"company_name": company_name})
            print("  -> ✅ Detailed analysis complete.")
            return result
        except Exception as e:
            print(f"  -> ❌ Error during LangChain analysis: {e}")
            return {"error": str(e)}
