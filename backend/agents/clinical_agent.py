import logging
from typing import Optional, Dict
from backend.rag.rag_chain import RAGChain
import os
import requests

logger = logging.getLogger("nephro_assistant")

class ClinicalAgent:
    """
    Clinical agent for handling medical queries using RAG and web search.
    Logs all major actions and retrieval attempts.
    """
    def __init__(self, llm, rag_chain: RAGChain):
        self.llm = llm
        self.rag_chain = rag_chain
        self.logger = logger

    def handle_medical_query(self, query: str, patient_context: Optional[Dict] = None) -> Dict:
        self.logger.info(f"ClinicalAgent: Handling medical query: {query}")
        queries = [query]
        if hasattr(self.rag_chain, 'expand_query'):
            queries = self.rag_chain.expand_query(query)
        web_intent_terms = ["latest", "search the internet", "google", "web", "recent", "new research", "clinical trials"]
        if any(term in query.lower() for term in web_intent_terms):
            self.logger.info(f"ClinicalAgent: Web intent detected for query: {query}")
            web_answer, web_sources = self.web_search(query)
            self.logger.info(f"ClinicalAgent: Web search result: {web_answer[:100]}... | Sources: {web_sources}")
            return {
                "response": f"\U0001F50E *Web Answer:* {web_answer}",
                "agent": "clinical",
                "sources": web_sources,
                "source_type": "web"
            }
        rag_response = None
        for q in queries:
            rag_response = self.rag_chain.get_answer(q, k_chunks=5, patient_context=patient_context)
            self.logger.info(f"ClinicalAgent: RAG retrieval for query '{q}' | Citations: {rag_response.get('citations', [])}")
            if not self.rag_chain.is_unknown_answer(rag_response["answer"]) and len(rag_response["answer"].strip()) > 30:
                self.logger.info(f"ClinicalAgent: RAG answer found for query '{q}'.")
                return {
                    "response": rag_response["answer"],
                    "agent": "clinical",
                    "source_type": rag_response.get("source_type"),
                    "citations": rag_response.get("citations", [])
                }
        known_terms = ["ckd", "kidney", "nephropathy", "glomerular", "proteinuria", "creatinine", "nephrotoxic"]
        if any(term in query.lower() for term in known_terms):
            self.logger.info(f"ClinicalAgent: Forcing RAG for nephrology keyword in query: {query}")
            return {
                "response": rag_response["answer"],
                "agent": "clinical",
                "source_type": rag_response.get("source_type"),
                "citations": rag_response.get("citations", [])
            }
        self.logger.info(f"ClinicalAgent: Defaulting to RAG answer for query: {query}")
        return {
            "response": rag_response["answer"],
            "agent": "clinical",
            "source_type": rag_response.get("source_type"),
            "citations": rag_response.get("citations", [])
        }

    def web_search(self, query: str):
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            self.logger.warning("ClinicalAgent: SERPAPI_API_KEY not set. Web search unavailable.")
            return ("Web search unavailable: SERPAPI_API_KEY not set.", [])
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": 3
        }
        resp = requests.get("https://serpapi.com/search", params=params)
        if resp.status_code == 200:
            data = resp.json()
            answer = "No good web answer found."
            sources = []
            if "answer_box" in data and "answer" in data["answer_box"]:
                answer = data["answer_box"]["answer"]
            elif "organic_results" in data:
                results = data["organic_results"]
                if results:
                    snippets = [r.get("snippet", "") for r in results if r.get("snippet")]
                    answer = "\n\n".join(snippets) if snippets else answer
                    sources = [
                        {"link": r.get("link", ""), "snippet": r.get("snippet", "")}
                        for r in results[:3]
                    ]
            self.logger.info(f"ClinicalAgent: Web search completed for query '{query}'.")
            return (answer, sources)
        self.logger.error(f"ClinicalAgent: Web search failed for query '{query}'.")
        return ("Web search failed.", [])

    def _is_boilerplate(self, answer: str) -> bool:
        boilerplate_keywords = [
            "copyright", "table of contents", "elsevier", "isbn", "edition", "activate the ebook", "peel off sticker", "visit http", "log in or sign up", "access code", "permissions policies"
        ]
        return any(kw in answer.lower() for kw in boilerplate_keywords) 