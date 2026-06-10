"""
DocMind RAG Backend
FastAPI application with RAG pipeline using sentence-transformers + FAISS + Groq LLM
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import json
import time

from rag_pipeline import RAGPipeline
from chunker import DocumentChunker

app = FastAPI(title="DocMind RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag = RAGPipeline()
chunker = DocumentChunker()

# ---------- Request / Response Models ----------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float

class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    doc_name: str

# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "DocMind RAG API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "index_size": rag.index_size(),
        "model": rag.embedding_model_name,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    Pipeline: parse → chunk → embed → store in FAISS index.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    content = await file.read()

    # Parse raw text from file
    raw_text = chunker.extract_text(content, file.filename)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    # Chunk the document
    chunks = chunker.chunk_text(raw_text, doc_name=file.filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no valid chunks.")

    # Embed and index
    rag.index_chunks(chunks)

    return IngestResponse(
        message="Document ingested successfully.",
        chunks_indexed=len(chunks),
        doc_name=file.filename,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Query the indexed documents.
    Pipeline: embed query → FAISS retrieval → rerank → LLM generation.
    """
    if rag.index_size() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a document first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.perf_counter()
    result = rag.query(req.question, top_k=req.top_k)
    latency = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round(latency, 2),
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Streaming version of /query — yields answer tokens as SSE.
    """
    if rag.index_size() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    def generate():
        for token in rag.query_stream(req.question, top_k=req.top_k):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/index")
def clear_index():
    """Clear the FAISS index (reset for new document set)."""
    rag.clear()
    return {"message": "Index cleared."}


@app.get("/index/stats")
def index_stats():
    return {
        "total_chunks": rag.index_size(),
        "documents": rag.get_doc_list(),
        "embedding_dim": rag.embedding_dim,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
