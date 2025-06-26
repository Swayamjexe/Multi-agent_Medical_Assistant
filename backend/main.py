import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from backend.rag.rag_chain import RAGChain
from backend.agents.receptionist_agent import ReceptionistAgent
from backend.agents.clinical_agent import ClinicalAgent
from typing import Optional
import os

# Configure logging
LOG_DIR = "backend/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "nephro_assistant.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nephro_assistant")

app = FastAPI()

# Use ChatOllama (Mistral) as the LLM for RAG
llm = ChatOllama(model="mistral", temperature=0)

# Load vector store from textbook
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("backend/data/faiss_index", embeddings, allow_dangerous_deserialization=True)

# Agents
rag_chain = RAGChain(vector_store, llm)
receptionist = ReceptionistAgent(llm)
clinical = ClinicalAgent(llm, rag_chain)

# Simple in-memory session state for POC
session_state = {}

class Message(BaseModel):
    text: str
    patient_name: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/chat")
def chat(message: Message):
    """
    Main chat endpoint for user interaction.
    Handles patient lookup, agent routing, and information retrieval.
    Logs all interactions and agent decisions.
    """
    session_id = message.session_id or "default"
    state = session_state.setdefault(session_id, {"current_patient": None})
    logger.info(f"Session {session_id} | User: {message.text} | Patient: {message.patient_name}")

    if not state["current_patient"]:
        name = message.patient_name or receptionist.extract_name(message.text)
        if name:
            logger.info(f"Session {session_id} | Receptionist: Looking up patient '{name}'")
            response = receptionist.handle_name(name)
            if receptionist.patient:
                state["current_patient"] = receptionist.patient
                logger.info(f"Session {session_id} | Receptionist: Patient '{name}' found and selected.")
            else:
                logger.info(f"Session {session_id} | Receptionist: Patient '{name}' not found or multiple matches.")
            return {"response": response, "agent": "receptionist"}
        else:
            logger.info(f"Session {session_id} | Receptionist: Greeting user.")
            return {"response": receptionist.greet(), "agent": "receptionist"}
    else:
        if receptionist.is_medical_query(message.text):
            logger.info(f"Session {session_id} | Routing to clinical agent for query: {message.text}")
            rag_response = clinical.handle_medical_query(message.text)
            if rag_response.get("source_type") == "textbook" and rag_response.get("citations"):
                logger.info(f"Session {session_id} | Clinical agent: RAG answer with {len(rag_response['citations'])} citations.")
                rag_response["citations"] = [
                    {"source": c["source"], "text": c["text"]} for c in rag_response["citations"]
                ]
            if rag_response.get("source_type") == "web" and rag_response.get("sources"):
                logger.info(f"Session {session_id} | Clinical agent: Web search answer with {len(rag_response['sources'])} sources.")
                rag_response["sources"] = [s["link"] if isinstance(s, dict) and "link" in s else s for s in rag_response["sources"]]
            return rag_response
        else:
            logger.info(f"Session {session_id} | Receptionist: Non-medical follow-up.")
            return {"response": "How else may I assist you?", "agent": "receptionist"}

@app.get("/patient/{name}")
def get_patient(name: str):
    response = receptionist.handle_name(name)
    return {"response": response} 