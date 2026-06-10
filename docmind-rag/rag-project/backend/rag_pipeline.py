"""
RAG Pipeline Core
Handles: embedding, FAISS indexing, cosine retrieval, reranking, LLM generation
"""

import os
import numpy as np
from typing import Iterator
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Embed chunks with sentence-transformers (all-MiniLM-L6-v2)
    2. Store in FAISS flat L2 index (swappable to IVF for scale)
    3. Retrieve top-k chunks by cosine similarity
    4. Generate answer with Groq (llama-3.1-8b-instant) via prompt engineering
    """

    SYSTEM_PROMPT = """You are DocMind, a precise document question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."
Be concise, accurate, and cite the relevant section when possible.

Context:
{context}"""

    def __init__(self):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # FAISS flat index — exact nearest-neighbor, no approximation
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Metadata store: maps FAISS integer ID → chunk metadata
        self.chunks: list[dict] = []

        # Groq client for LLM generation
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env file.")
        self.client = Groq(api_key=api_key)
        self.llm_model = "llama-3.1-8b-instant"

    # ---------- Indexing ----------

    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Embed a list of chunk dicts and add to FAISS index.
        Each chunk: {"text": str, "doc_name": str, "chunk_id": int, "page": int}
        """
        texts = [c["text"] for c in chunks]

        # Batch embed — normalized for cosine similarity
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def index_size(self) -> int:
        return self.index.ntotal

    def get_doc_list(self) -> list[str]:
        return list({c["doc_name"] for c in self.chunks})

    def clear(self) -> None:
        self.index.reset()
        self.chunks.clear()

    # ---------- Retrieval ----------

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Embed query, search FAISS, return top-k chunks with similarity scores.
        """
        q_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
        )
        q_embedding = np.array(q_embedding, dtype=np.float32)

        # D = distances, I = indices
        distances, indices = self.index.search(q_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            # Convert L2 distance to cosine similarity (since embeddings are normalized)
            chunk["similarity"] = float(1 - dist / 2)
            results.append(chunk)

        # Filter low-relevance results
        results = [r for r in results if r["similarity"] > 0.25]
        return results

    # ---------- Generation ----------

    def _build_context(self, retrieved_chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            parts.append(
                f"[Source {i} | {chunk['doc_name']} | Page {chunk.get('page', '?')}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full RAG query: retrieve → build prompt → generate → return answer + sources.
        """
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the indexed documents.",
                "sources": [],
            }

        context = self._build_context(retrieved)
        prompt = self.SYSTEM_PROMPT.format(context=context)

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()

        sources = [
            {
                "doc_name": c["doc_name"],
                "page": c.get("page", None),
                "chunk_preview": c["text"][:150] + "...",
                "similarity": round(c["similarity"], 3),
            }
            for c in retrieved
        ]

        return {"answer": answer, "sources": sources}

    def query_stream(self, question: str, top_k: int = 5) -> Iterator[str]:
        """
        Streaming generation — yields tokens one by one for SSE endpoint.
        """
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            yield "I couldn't find relevant information in the indexed documents."
            return

        context = self._build_context(retrieved)
        prompt = self.SYSTEM_PROMPT.format(context=context)

        stream = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
