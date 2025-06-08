# app.py
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import get_pdf_text, chunk_text, build_vector_store, load_vector_store, make_qa_chain
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in environment")

app = FastAPI(title="PDFInsight Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_DIR = "faiss_index"

@app.post("/upload/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Upload one or more PDFs to build the index."""
    pdf_bytes = [await f.read() for f in files]
    text = get_pdf_text(pdf_bytes)
    if not text:
        raise HTTPException(400, "No text extracted from PDFs.")
    chunks = chunk_text(text)
    build_vector_store(chunks, INDEX_DIR)
    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Ask a question against the indexed PDFs."""
    if not os.path.isdir(INDEX_DIR):
        raise HTTPException(400, "Index not found. Please upload PDFs first.")
    db = load_vector_store(INDEX_DIR)
    docs = db.similarity_search(question)
    chain = make_qa_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return {"answer": result["output_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
