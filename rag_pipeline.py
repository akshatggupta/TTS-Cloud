"""
rag_pipeline.py
Core RAG logic:
  1. Parse PDF → text chunks
  2. Embed with sentence-transformers (free, local)
  3. Store/query Pinecone vector DB
  4. Generate answer with Groq LLM
"""

import re
import fitz  # PyMuPDF
import hashlib
from typing import Tuple, List

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq

# ─── Embedding Model (runs locally, no API key needed) ──────────────────────
# all-MiniLM-L6-v2 → 384-dim, fast, great quality
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ─── Pinecone Setup ─────────────────────────────────────────────────────────
def init_pinecone(api_key: str, index_name: str):
    """Initialize Pinecone and return the index object."""
    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,          # matches all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)


# ─── PDF Processing ─────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    chunk_size: characters per chunk
    overlap: characters shared between consecutive chunks
    """
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        last_period = chunk.rfind('. ')
        if last_period > chunk_size // 2:
            chunk = chunk[:last_period + 1]

        chunks.append(chunk.strip())
        start += len(chunk) - overlap

    return [c for c in chunks if len(c) > 50]  # filter tiny chunks


def process_pdf(pdf_bytes: bytes, index) -> int:
    """
    Full pipeline: PDF → chunks → embeddings → Pinecone.
    Returns number of chunks stored.
    """
    # 1. Extract text
    text = extract_text_from_pdf(pdf_bytes)

    # 2. Chunk
    chunks = chunk_text(text)

    # 3. Embed
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # 4. Upsert to Pinecone in batches of 50
    batch_size = 50
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Unique ID based on content hash
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:16] + f"_{i}"
        vectors.append({
            "id": chunk_id,
            "values": emb,
            "metadata": {
                "text": chunk,
                "chunk_index": i
            }
        })

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    return len(chunks)


# ─── Query & Answer ─────────────────────────────────────────────────────────
def retrieve_context(question: str, index, top_k: int = 4) -> Tuple[str, str]:
    """
    Embed the question, query Pinecone, return combined context and source snippet.
    """
    embedder = get_embedder()
    q_embedding = embedder.encode([question])[0].tolist()

    results = index.query(
        vector=q_embedding,
        top_k=top_k,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n---\n\n".join(chunks)

    # Build a short source label (first 60 chars of each chunk)
    sources = " | ".join([f"Chunk {i+1}: {c[:60]}..." for i, c in enumerate(chunks)])

    return context, sources


def answer_question(question: str, index, groq_api_key: str) -> Tuple[str, str]:
    """
    Full RAG answer:
      1. Retrieve relevant context from Pinecone
      2. Send to Groq LLM with a clear prompt
      3. Return (answer, sources)
    """
    # 1. Retrieve
    context, sources = retrieve_context(question, index)

    # 2. Build prompt
    system_prompt = """You are a helpful academic assistant that answers questions about course syllabi.
Use ONLY the provided context to answer. If the answer is not in the context, say so clearly.
Be concise, accurate, and student-friendly. Format lists and key info clearly."""

    user_prompt = f"""Context from the syllabus:
{context}

Student Question: {question}

Answer based on the syllabus context above:"""

    # 3. Call Groq
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",   # Free, fast Groq model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()
    return answer, sources
