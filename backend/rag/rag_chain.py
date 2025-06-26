import logging
from typing import Optional, Dict, Any
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.chat_models import ChatOllama

class RAGChain:
    def __init__(self, vector_store: Any, llm=None):
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        # Use ChatOllama with Mistral for QA
        self.llm = llm or ChatOllama(model="mistral", temperature=0)
        self.chain = load_qa_with_sources_chain(self.llm, chain_type="stuff")

    def expand_query(self, query):
        synonyms = {
            "symptoms": ["signs", "manifestations", "clinical features"],
            "treatment": ["management", "therapy"],
            "cause": ["etiology", "reason", "origin"],
        }
        for key, variants in synonyms.items():
            if key in query.lower():
                return [query] + [query.lower().replace(key, v) for v in variants]
        return [query]
    
    def truncate_to_max_tokens(self, prompt_base: str, context_chunks: list, max_tokens=1024):
        context = ""
        for chunk in context_chunks:
            chunk_text = getattr(chunk, 'page_content', None) or chunk.get('page_content', chunk.get('text', ''))
            # Skip any chunk that alone exceeds 1000 tokens
            if len(self.tokenizer.encode(chunk_text)) > 1000:
                continue
            temp_context = context + "\n\n" + chunk_text
            token_count = len(self.tokenizer.encode(prompt_base + temp_context))
            if token_count >= max_tokens:
                break
            context = temp_context
        return context

    def get_answer(self, question: str, k_chunks: int = 5, patient_context: Optional[Dict] = None):
        chunks = self.vector_store.similarity_search(question, k=k_chunks)
        response = self.chain.run(input_documents=chunks, question=question)
        # Return both the chunk text and the source for each chunk
        citations = [
            {"source": doc.metadata.get("source", ""), "text": doc.page_content}
            for doc in chunks
        ]
        return {
            "answer": response,
            "source_type": "textbook",
            "citations": citations,
        }

    @staticmethod
    def is_unknown_answer(answer: str) -> bool:
        return any(
            phrase in answer.lower()
            for phrase in ["i don't know", "not found", "couldn't find", "no relevant information"]
        ) 