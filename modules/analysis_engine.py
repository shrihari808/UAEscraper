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
        def get_company_specific_retriever(info):
            company_name = info.get("company_name")
            docs = self.vector_store.similarity_search(company_name, k=50) 
            filtered_docs = [doc for doc in docs if doc.metadata.get('company') == company_name]
            return "\n---\n".join([doc.page_content for doc in filtered_docs])

        # --- UPDATED PROMPT ---
        template = """
        You are a top-tier venture capital analyst and investigative journalist specializing in UAE Fintech, leveraging a financial-domain-specific language model (FinBERT) for information retrieval.
        Your task is to synthesize the provided data for "{company_name}" to extract deep, actionable intelligence.

        **CONTEXT (Retrieved from a finance-optimized Knowledge Base):**
        {context}

        **YOUR CORE OBJECTIVES (Synthesize all data to answer):**
        1.  **Company Metadata:** Name, Sector, Location/Freezone, Stage (e.g., startup, growth, established), Founding Year.
        2.  **Financial Offerings & Products:** What specific financial products do they offer (e.g., SME loans, real estate finance, asset financing)? What is their core value proposition?
        3.  **Strategic Focus & Signals:**
            - **Digital Transformation:** What is their stage of digital adoption? Mention any use of cloud, AI, or mobile-first strategies.
            - **Key Focus Areas:** Are they targeting retail, corporate, or SME clients? Any specific industry focus?
            - **Partnerships & M&A:** Are there any mentions of partnerships, M&A activity, or key collaborations?
            - **Growth Signals:** Are they hiring for specific tech or expansion roles? Any mention of new product launches or market expansion?
        4.  **Decision Makers:** Key people and roles (e.g., CEO, CTO, Chief Strategy Officer, Head of AI).
        5.  **Competitive Landscape (Implied):** Based on their offerings, who might be their key competitors in the UAE market? (Provide 1-2 names if possible).
        6.  **Mobile App Presence:**
            - Specify if they have an app on Google Play and/or Apple App Store.
            - If an app exists, what is its primary purpose?
            - What are its key features as described in the context?
            - Specify its ratings and number of reviews if available. Also specify the last updated date if available.
            - If no app is mentioned, state "No mobile app presence found in the provided data."

        **OUTPUT FORMAT:**
        Return a single, clean JSON object with the keys exactly as listed above. For "Company Metadata", ensure you include the "Name" of the company.
        For "Mobile App Presence", For each app provide a nested object with keys "Google Play" and "Apple App Store", each containing:app_name(string), has_app (boolean), purpose (string), key_features (list of strings), ratings (string), reviews (string), last_updated (string).
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        # --- UPDATED OUTPUT PARSER to handle the new structure ---
        class CustomJsonOutputParser(JsonOutputParser):
            def parse(self, text: str):
                try:
                    # The LLM might return a JSON string that is not perfectly clean
                    # We can try to find the JSON object within the text
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        text = text[json_start:json_end]
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from LLM output: {e}")
                    # Fallback to returning an error structure
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
            result = self.chain.invoke({"company_name": company_name})
            print("  -> ✅ Detailed analysis complete.")
            return result
        except Exception as e:
            print(f"  -> ❌ Error during LangChain analysis: {e}")
            return {"error": str(e)}
