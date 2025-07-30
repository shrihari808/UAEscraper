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
            docs = self.vector_store.similarity_search(company_name, k=50) # Broad search
            # Filter documents to only include the specified company
            filtered_docs = [doc for doc in docs if doc.metadata.get('company') == company_name]
            return "\n---\n".join([doc.page_content for doc in filtered_docs])

        template = """
        You are a top-tier venture capital analyst and investigative journalist specializing in UAE Fintech. 
        Your task is to synthesize the provided data for "{company_name}" to extract deep, actionable intelligence.

        **CONTEXT (Retrieved from Knowledge Base):**
        {context}

        **YOUR CORE OBJECTIVES (Synthesize all data to answer):**
        1.  **Company Metadata:** Name, Sector, Location/Freezone, Stage.
        2.  **Product & Platform Requirements:** What are they building? Key technologies? Development stage?
        3.  **AI & Cloud Strategy:** AI implementation? Cloud usage?
        4.  **Digital Transformation & Inclusion:** Digitization intent? Inclusion focus?
        5.  **Partnerships & Events:** Collaborations mentioned anywhere? Events attended?
        6.  **Decision Makers:** Key people and roles (CTO, CDO, Founder, etc.)?

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
