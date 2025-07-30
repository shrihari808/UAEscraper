import pandas as pd
import json
import re
import os
import time
import random
from dotenv import load_dotenv
import numpy as np
import openai

# --- Vectorization Imports ---
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
FAISS_INDEX_PATH = "linkedin_data.index"
METADATA_PATH = "linkedin_metadata.json"

# --- MODULE 1: Data Retrieval (RAG) ---
def load_knowledge_base():
    """Loads the FAISS index and metadata from disk."""
    print("🧠 Loading knowledge base...")
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("❌ Knowledge base files not found. Please run Step 2 and 3 scrapers first.")
        return None, None
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    print(f"✅ Knowledge base loaded with {index.ntotal} vectors.")
    return index, metadata

def retrieve_relevant_context(company_name, metadata):
    """
    Retrieves all text snippets for a specific company from the metadata.
    This function assumes the previous scraping steps have saved the 'text' of each item.
    """
    print(f"🔍 Retrieving all context for '{company_name}' from the knowledge base...")
    
    # Reconstruct the context for the LLM from the metadata
    about_context = ""
    posts_context = []
    jobs_context = []
    news_context = []

    # This requires that the metadata file contains a "text" key for each entry.
    # The previous scripts (Step 2 & 3) should be updated to save this.
    company_data = [item for item in metadata if item.get('company') == company_name]

    if not company_data:
        print(f"  -> No data found for '{company_name}' in the knowledge base.")
        return None

    for item in company_data:
        text = item.get("text", "") # Get the text, default to empty string if not found
        if item['type'] == 'about':
            about_context = text
        elif item['type'] == 'post':
            posts_context.append(text)
        elif item['type'] == 'job':
            jobs_context.append(text)
        elif item['type'] == 'news':
            news_context.append(text)
            
    print(f"    -> Found {len(posts_context)} posts, {len(jobs_context)} jobs, and {len(news_context)} news articles.")
    
    return {
        "about": about_context,
        "posts": posts_context,
        "jobs": jobs_context,
        "news": news_context
    }

# --- MODULE 2: LLM Analysis (RAG-Powered) ---
def analyze_retrieved_context(company_name, context, client):
    """Uses the LLM to analyze the retrieved context from the vector store."""
    print(f"🤖 Performing RAG-based analysis for '{company_name}'...")

    if not context:
        print("  -> No context to analyze.")
        return None

    # --- FIX: Pre-format strings before inserting into the f-string ---
    about_context_str = context.get("about", "")
    posts_context_str = "\n---\n".join(context.get("posts", []))
    jobs_context_str = ", ".join(context.get("jobs", []))
    news_context_str = "\n---\n".join(context.get("news", []))

    prompt = f"""
    You are a top-tier venture capital analyst and investigative journalist specializing in UAE Fintech. Your task is to synthesize the provided data for "{company_name}" from a knowledge base to extract deep, actionable intelligence.

    **DATA PROVIDED (Retrieved from Knowledge Base):**
    1.  **LinkedIn 'About' Section:** {about_context_str[:4000]}
    2.  **Relevant LinkedIn Posts:** {posts_context_str[:6000]}
    3.  **Open Job Titles on LinkedIn:** {jobs_context_str[:1000]}
    4.  **External News Articles:** {news_context_str[:6000]}

    **YOUR CORE OBJECTIVES (Synthesize all data to answer):**
    1.  **Company Metadata:** Sector, Location/Freezone, Stage.
    2.  **Product & Platform Requirements:** What are they building? Key technologies? Development stage?
    3.  **AI & Cloud Strategy:** AI implementation? Cloud usage?
    4.  **Digital Transformation & Inclusion:** Digitization intent? Inclusion focus?
    5.  **Partnerships & Events:** Collaborations mentioned anywhere? Events attended?
    6.  **Decision Makers:** Key people and roles (CTO, CDO, Founder, etc.)?

    **OUTPUT FORMAT:**
    Return a single, clean JSON object with the keys exactly as listed above. Do not include explanations.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
        analysis = json.loads(response.choices[0].message.content)
        print("  -> ✅ Detailed analysis complete.")
        return analysis
    except Exception as e:
        print(f"  -> ❌ Error during LLM analysis: {e}")
        return {"error": str(e)}

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Step 4: Query and Analysis Engine ---")
    
    # Get company name from user input
    company_name_input = input("Enter the company name to analyze: ")
    
    if not OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found.")
    else:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        index, metadata = load_knowledge_base()
        
        if index and metadata:
            # Retrieve all the context for the company from our knowledge base
            context = retrieve_relevant_context(company_name_input, metadata)
            
            if context:
                # Send the retrieved context to the LLM for final analysis
                analysis_result = analyze_retrieved_context(company_name_input, context, openai_client)
                
                if analysis_result:
                    # Save the JSON output to a file
                    output_filename = f"{company_name_input.replace(' ', '_')}_analysis.json"
                    with open(output_filename, "w") as f:
                        json.dump(analysis_result, f, indent=4)
                    
                    print(f"\n✅ Analysis saved to '{output_filename}'")
                    
                    # Display the JSON output
                    print("\n--- Generated Intelligence ---")
                    print(json.dumps(analysis_result, indent=4))

