import pdfplumber
import re
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PDF_PATH = "textbook/nephrology_textbook.pdf"
INDEX_PATH = "backend/data/faiss_index"

def extract_clean_text(pdf_path, start_page=22):
    print("Extracting text from PDF with pdfplumber...")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i < start_page:
                continue
            page_text = page.extract_text()
            if not page_text:
                continue
            text += clean_page_text(page_text) + "\n"
    return text

def clean_page_text(text):
    # Remove references section
    text = re.sub(r'\bREFERENCES\b.*?(?=\n[A-Z])', '', text, flags=re.DOTALL)
    # Remove self-assessment questions
    text = re.sub(r'\bSELF[- ]?ASSESSMENT QUESTIONS\b.*?(?=\n[A-Z])', '', text, flags=re.DOTALL)
    # Remove excessive empty lines
    text = re.sub(r'\n{2,}', '\n', text)
    return text

def is_relevant_chunk(chunk):
    if len(chunk) < 150:
        return False
    if "chapter" in chunk.lower() and len(chunk) < 500:
        return False
    if re.search(r'\b(page|contents|figure|table|section|author|publisher)\b', chunk.lower()):
        return False
    if sum(chunk.count(c) for c in "·•…→") > 3:
        return False
    return True

def chunk_and_filter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    filtered = []
    for i, chunk in enumerate(chunks):
        if is_relevant_chunk(chunk.page_content):
            # Add a source field to metadata for LangChain QA compatibility
            chunk.metadata["source"] = f"nephrology_textbook.pdf:chunk_{i+1}"
            filtered.append(chunk)
    return filtered

def embed_and_save(chunks, index_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

if __name__ == "__main__":
    text = extract_clean_text(PDF_PATH)
    print(f"Total length after cleaning: {len(text)} characters")
    chunks = chunk_and_filter(text)
    print(f"Total relevant chunks: {len(chunks)}")
    embed_and_save(chunks, INDEX_PATH)
    print("✅ FAISS index built and saved successfully.") 