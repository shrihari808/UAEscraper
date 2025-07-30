# modules/analysis_engine.py

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

import config

class AnalysisEngine:
    """Analyzes company data using a LangChain RAG chain."""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Builds the LangChain Expression Language (LCEL) chain."""
        # Define a custom retriever that filters by company name
        def get_company_specific_retriever(info):
            company_name = info.get("company_name")
            # Perform a similarity search first to get relevant documents
            docs = self.vector_store.similarity_search(company_name, k=50) 
            # Post-filter to ensure absolute relevance to the target company
            filtered_docs = [doc for doc in docs if doc.metadata.get('company') == company_name]
            # Combine the content of the filtered documents
            return "\n---\n".join([doc.page_content for doc in filtered_docs])

        # Updated prompt to be more specific for financial analysis
        template = """
        You are a top-tier venture capital analyst and investigative journalist specializing in UAE Fintech, leveraging a financial-domain-specific language model (FinBERT) for information retrieval.
        Your task is to synthesize the provided data for "{company_name}" to extract deep, actionable intelligence.

        **CONTEXT (Retrieved from a finance-optimized Knowledge Base):**
        {context}

        **YOUR CORE OBJECTIVES (Synthesize all data to answer):**
        1.  **Company Metadata:** Name, Sector, Location/Freezone, Stage (e.g., startup, growth, established), Founding Year.
        2.  **Financial Offerings & Products:** What specific financial products do they offer (e.g., SME loans, real estate finance, asset financing, supply chain finance)? What is their core value proposition?
        3.  **Strategic Focus & Signals:**
            - **Digital Transformation:** What is their stage of digital adoption (e.g., legacy, transitioning, digital-native)? Mention any use of cloud, AI, or mobile-first strategies.
            - **Key Focus Areas:** Are they targeting retail, corporate, or SME clients? Any specific industry focus?
            - **Partnerships & M&A:** Are there any mentions of partnerships with other fintechs, banks, or tech companies? Any M&A activity?
            - **Growth Signals:** Are they hiring for specific tech or expansion roles? Any mention of new product launches or market expansion?
        4.  **Decision Makers:** Key people and roles (e.g., CEO, CTO, Chief Strategy Officer, Head of AI).
        5.  **Competitive Landscape (Implied):** Based on their offerings and news, who might be their key competitors in the UAE market? (Provide 1-2 names if possible).

        **OUTPUT FORMAT:**
        Return a single, clean JSON object with the keys exactly as listed above. For "Company Metadata", ensure you include the "Name" of the company which is "{company_name}".
        """

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = JsonOutputParser()

        # LCEL Chain Definition
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
            result = self.chain.invoke({"company_name": company_name})
            print("  -> ✅ Detailed analysis complete.")
            return result
        except Exception as e:
            print(f"  -> ❌ Error during LangChain analysis: {e}")
            return {"error": str(e)}
