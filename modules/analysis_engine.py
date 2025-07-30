# modules/analysis_engine.py

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from collections import OrderedDict

import config

class AnalysisEngine:
    """
    REVISED: Implements an advanced multi-query retrieval strategy by fetching a large
    batch of documents and then filtering them to build a rich, targeted context.
    """

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.chain = self._create_rag_chain()

    def _get_multi_query_retrieved_context(self, company_name):
        """
        Performs multiple targeted vector searches, fetching a large number of results
        and then filtering them by company to ensure relevance.
        """
        print(f"  -> Performing advanced multi-query retrieval for '{company_name}'...")
        
        targeted_queries = {
            "strategy_and_vision": f"CEO or executive statements about {company_name}'s strategy, vision, and future plans.",
            "technology_and_ai": f"Information on {company_name}'s technology stack, cloud infrastructure (AWS, Azure), AI, machine learning, or digitization efforts.",
            "products_and_services": f"Details about {company_name}'s products like BNPL, digital wallets, remittance services, or loans.",
            "financial_inclusion": f"Statements or programs by {company_name} related to serving the unbanked, underbanked, SMEs, or specific community groups.",
            "hiring_and_growth": f"Job postings, hiring signals, or mentions of team expansion at {company_name}."
        }

        all_retrieved_docs = []
        # --- FIX: Revert to manual search-then-filter, but with a much larger 'k' value ---
        for key, query in targeted_queries.items():
            print(f"    -> Searching for signal: '{key}'")
            # Fetch a large number of documents to increase the chance of finding relevant ones
            retrieved_docs = self.vector_store.similarity_search(query, k=150) # Increased k significantly
            
            # Filter the large batch of results for the correct company
            filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get('company') == company_name]
            all_retrieved_docs.extend(filtered_docs)
        
        # De-duplicate the documents based on page content to avoid repetition
        unique_docs = list(OrderedDict.fromkeys(doc.page_content for doc in all_retrieved_docs))
        
        print(f"  -> Aggregated {len(unique_docs)} unique, relevant document chunks for context.")
        
        return "\n---\n".join(unique_docs)


    def _create_rag_chain(self):
        """Builds the LangChain Expression Language (LCEL) chain using the advanced retriever."""
        
        def retriever(info):
            return info["context"]

        template = """
        You are a specialist intelligence analyst for a Venture Capital firm focused on Fintech in the Middle East and Africa.
        Your task is to analyze the provided context about "{company_name}" and populate a structured JSON object.
        The context has been pre-compiled from multiple targeted searches on strategy, technology, products, and hiring.
        Adhere strictly to the requested format. If no information is found for a field, use an empty string "" or null.

        **CONTEXT (Compiled from targeted searches on annual reports, news, etc.):**
        {context}

        **JSON OUTPUT STRUCTURE (Fill this in based on the context):**
        {{
            "Institution_Name": "{company_name}",
            "BNPL_Signal": "Extract any direct mention of 'Buy Now, Pay Later' services, partnerships, or plans. Example: 'CEO mentioned plans to offer micro-credit to small traders.' or 'Launched a new BNPL product in Q2.'",
            "Remittance_App_Signal": "Find any mention of remittance services, diaspora banking, or cross-border payment partnerships. Example: 'Partnership with diaspora remittance program.'",
            "Wallet_Signal": "Look for mentions of digital wallets, e-money licenses, or prepaid card offerings. Example: 'Exploring e-money license for prepaid wallet.'",
            "Cloud_And_DevOps_Needs": "Infer technology needs from text. Look for mentions of specific cloud providers (AWS, Azure, GCP), infrastructure roles, or system migration projects. Example: 'On-prem hosting, observing paper-based loan applications' or 'Migrating to AWS.'",
            "AI_Opportunities": "Identify specific use cases for AI mentioned or implied. Look for terms like 'AI agents', 'RAG', 'document analysis', 'risk scoring', 'automation'. Example: 'AI agents: RAG use cases: loan documentation search, risk discovery.'",
            "Digitization_Signals": "Find evidence of digital transformation efforts. Look for mentions of 'digitization roadmap', 'API', 'onboarding', 'paperless'. Example: 'Received ADGM grant; Mentioned in Central Bank's 2024 digitization roadmap.'",
            "Financial_Inclusion_Statement": "Extract direct quotes or statements about serving the unbanked, underbanked, SMEs, women, or migrant workers. Example: 'Serving 80% unbanked rural population, focus on women-led MSMEs and credit-starved migrant-agri-crews.'",
            "CEO_Statement": "Find a direct quote from the CEO about the company's strategy, vision, or next steps. The quote should be verbatim. Example: 'Our next step is serving digital-mobile onboarding.'",
            "CTO_Statement": "Find a direct quote from the CTO (or equivalent technology leader) about technology, AI, or platform strategy. The quote should be verbatim. Example: 'We are assessing cloud-based micro-credit onboarding and workflow with AI.'",
            "Hiring_Signals": "List specific, high-value job titles being hired for that indicate strategic direction. Focus on tech, product, and leadership roles. Example: 'Product Manager - Wallet; Backend Engineer with AI/NLP experience.'"
        }}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        class CustomJsonOutputParser(JsonOutputParser):
            def parse(self, text: str):
                try:
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        text = text[json_start:json_end]
                    return json.loads(text)
                except Exception as e:
                    print(f"Error decoding JSON from LLM output: {e}")
                    return {"error": "Failed to parse LLM JSON output", "raw_output": text}

        output_parser = CustomJsonOutputParser()

        chain = (
            {
                "context": retriever,
                "company_name": lambda x: x["company_name"]
            }
            | prompt
            | self.llm
            | output_parser
        )
        return chain

    def analyze(self, company_name):
        """
        REVISED: Invokes the RAG chain by first building a context with multi-query
        retrieval and then passing it to the LLM.
        """
        print(f"🤖 Performing highly-structured RAG analysis for '{company_name}'...")
        
        rich_context = self._get_multi_query_retrieved_context(company_name)
        
        if not rich_context or not rich_context.strip():
            print("  -> ⚠️  Could not retrieve any relevant context from the vector store. Aborting analysis.")
            return {"error": "No relevant context found for this company."}

        try:
            result = self.chain.invoke({
                "company_name": company_name,
                "context": rich_context
            })
            print("  -> ✅ Detailed analysis complete.")
            return result
        except Exception as e:
            print(f"  -> ❌ Error during LangChain analysis: {e}")
            return {"error": str(e)}
