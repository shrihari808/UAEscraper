# modules/analysis_engine.py

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import re

import config

class AnalysisEngine:
    """Analyzes company data using a LangChain RAG chain."""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Builds the LangChain Expression Language (LCEL) chain."""
        def get_company_specific_retriever(info):
            """
            Performs multiple, strictly filtered searches to retrieve a comprehensive
            and accurate set of documents for a specific company.
            """
            company_name = info.get("company_name")
            
            # --- Strict Multi-retrieval Strategy ---
            # This strategy ensures that we only retrieve documents matching the company name
            # by applying a metadata filter directly in the vector search.

            # 1. Get dedicated App information strictly for this company
            app_docs = self.vector_store.similarity_search(
                company_name, k=2, filter={"company": company_name, "type": "mobile_app"}
            )

            # 2. Get dedicated LinkedIn information strictly for this company
            linkedin_docs = self.vector_store.similarity_search(
                company_name, k=3, filter={"company": company_name, "type": "company_about"}
            )

            # 3. Get a diverse set of general information strictly for this company
            general_docs = self.vector_store.max_marginal_relevance_search(
                company_name, k=8, fetch_k=50, filter={"company": company_name}
            )

            # Combine and deduplicate the strictly filtered documents
            combined_docs = app_docs + linkedin_docs + general_docs
            unique_docs = list({doc.metadata['source']: doc for doc in combined_docs}.values())


            # Format the context, with content slicing
            context_with_sources = []
            MAX_CHARS_PER_DOC = 5000 
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

        **CONTEXT (Retrieved from a finance-optimized Knowledge Base):**
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
            {"context": get_company_specific_retriever, "company_name": RunnablePassthrough()}
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
