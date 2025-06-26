# Nephrology Multi-Agent RAG Assistant

## Overview
This project is a multi-agent, Retrieval-Augmented Generation (RAG) enabled nephrology assistant. It features a receptionist and a clinical agent, providing specialized, context-aware answers to nephrology questions using a comprehensive textbook and web search for the latest information.

## Architecture Justification
- **RAG Pipeline:** Uses a nephrology textbook as the primary knowledge source, ensuring clinical accuracy and depth. Clean text is extracted from the PDF (skipping front matter, filtering out non-clinical content) and chunked for efficient retrieval.
- **Agents:**
  - **Receptionist Agent:** Handles patient identification, session management, and routes medical queries to the clinical agent.
  - **Clinical Agent:** Answers medical questions using RAG over the textbook, and falls back to web search (via SerpAPI) for queries requiring the latest research or explicit web intent.
- **Patient Data:** Patient records are stored in a SQLite database. The receptionist agent fetches and summarizes patient data to provide personalized context for clinical answers.
- **Web Search:** If a question cannot be answered from the textbook or explicitly requests recent information, the system uses SerpAPI to fetch and cite relevant web articles, including URLs.
- **Session Handling:** Each user session tracks the current patient and conversation context for continuity.
- **Logging:** All user interactions, agent handoffs, and retrieval attempts/results are logged with timestamps for audit and debugging.

## How It Works
1. **Textbook Processing:**
   - The PDF is processed to extract clean, clinical text (skipping TOC, references, etc.).
   - Text is chunked and embedded using MiniLM and stored in a FAISS vector store.
2. **Receptionist Agent:**
   - Greets users, extracts names, and fetches patient records from the database.
   - Handles session state and routes medical queries.
3. **Clinical Agent:**
   - Uses RAG to answer nephrology questions from the textbook.
   - If the answer is not found or the user requests the latest/web info, performs a web search and cites URLs.
4. **Logging:**
   - All interactions and decisions are logged in `backend/logs/nephro_assistant.log`.

## Setup & Usage
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Prepare the textbook:**
   - Place your nephrology textbook PDF in the `textbook/` directory.
   - Run the chunking script to build the FAISS index:
     ```
     python backend/rag/chunk_and_embed_textbook.py
     ```
3. **Start the backend:**
   ```
   uvicorn backend.main:app --reload
   ```
4. **(Optional) Start the frontend:**
   - If using Streamlit or another UI, run the appropriate command.

## Requirements
- Python 3.11+
- FAISS, LangChain, ChatOllama, SerpAPI key (for web search)
- SQLite for patient data

## Notes
- All logs are saved in `backend/logs/nephro_assistant.log`.
- Sessions are managed in-memory for POC; extend as needed for production.
- Web search requires a valid SerpAPI API key in your environment variables. 