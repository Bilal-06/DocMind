"""
Document Chunker
Handles: PDF/TXT text extraction, semantic chunking with overlap
"""

import re
from typing import Optional
import pdfplumber


class DocumentChunker:
    """
    Two-stage pipeline:
    1. Extract raw text from PDF (pdfplumber) or TXT
    2. Chunk with sliding window + sentence boundary awareness
    
    Design choices:
    - chunk_size=512 tokens (~400 words): balances context richness vs retrieval precision
    - overlap=64 tokens: prevents splitting mid-thought at chunk boundaries
    - Sentence-boundary splitting: never cuts mid-sentence
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size  # approximate word count per chunk
        self.overlap = overlap        # words shared between consecutive chunks

    # ---------- Extraction ----------

    def extract_text(self, content: bytes, filename: str) -> str:
        """Extract raw text from PDF or TXT bytes."""
        if filename.lower().endswith(".pdf"):
            return self._extract_pdf(content)
        elif filename.lower().endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def _extract_pdf(self, content: bytes) -> str:
        """
        Use pdfplumber for accurate PDF text extraction.
        Preserves page boundaries with [PAGE N] markers for citation.
        """
        import io
        pages_text = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[PAGE {i}]\n{text}")
        return "\n\n".join(pages_text)

    # ---------- Chunking ----------

    def chunk_text(self, text: str, doc_name: str) -> list[dict]:
        """
        Split text into overlapping chunks with metadata.
        Returns list of:
        {
            "text": str,
            "doc_name": str,
            "chunk_id": int,
            "page": int | None,
            "word_count": int
        }
        """
        # Split into page sections if PDF
        sections = self._split_by_page(text)

        chunks = []
        chunk_id = 0

        for page_num, section_text in sections:
            section_chunks = self._sliding_window_chunk(
                section_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
            for chunk_text in section_chunks:
                cleaned = self._clean_text(chunk_text)
                if len(cleaned.split()) < 20:  # skip tiny chunks
                    continue
                chunks.append({
                    "text": cleaned,
                    "doc_name": doc_name,
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "word_count": len(cleaned.split()),
                })
                chunk_id += 1

        return chunks

    def _split_by_page(self, text: str) -> list[tuple[Optional[int], str]]:
        """Split text by [PAGE N] markers. If no markers, treat as single section."""
        page_pattern = re.compile(r'\[PAGE (\d+)\]')
        parts = page_pattern.split(text)

        if len(parts) == 1:
            # No page markers (TXT file)
            return [(None, text)]

        sections = []
        # parts = ["", "1", "page1 text", "2", "page2 text", ...]
        i = 1
        while i < len(parts) - 1:
            page_num = int(parts[i])
            content = parts[i + 1].strip()
            if content:
                sections.append((page_num, content))
            i += 2

        return sections if sections else [(None, text)]

    def _sliding_window_chunk(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[str]:
        """
        Split text into overlapping word-level windows.
        Respects sentence boundaries by snapping split points to nearest period.
        """
        # Split into sentences first
        sentences = self._split_sentences(text)
        words_per_sentence = [s.split() for s in sentences]

        chunks = []
        flat_words = []
        # Track which sentence each word belongs to (for boundary snapping)
        sentence_boundaries = set()
        pos = 0
        for words in words_per_sentence:
            flat_words.extend(words)
            pos += len(words)
            sentence_boundaries.add(pos)

        total = len(flat_words)
        start = 0

        while start < total:
            end = min(start + chunk_size, total)

            # Snap end to nearest sentence boundary (within ±30 words)
            if end < total:
                best = end
                for boundary in sentence_boundaries:
                    if abs(boundary - end) < 30 and boundary > start:
                        best = boundary
                        break
                end = best

            chunk = " ".join(flat_words[start:end])
            chunks.append(chunk)

            if end >= total:
                break

            # Move forward by chunk_size - overlap
            start = end - overlap

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter — splits on . ! ? followed by whitespace."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace, control characters, repeated punctuation."""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # control chars
        text = re.sub(r'\s+', ' ', text)         # collapse whitespace
        text = re.sub(r'\.{3,}', '...', text)    # normalize ellipsis
        text = re.sub(r'-{3,}', '---', text)     # normalize long dashes
        return text.strip()
