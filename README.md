# Document RAG Chatbot

A retrieval-augmented Q&A backend for your documents (PDF, DOCX, TXT, MD, CSV), built around a hybrid retrieval pipeline rather than plain vector search.

## How it works

- **Hybrid retrieval**: combines dense vector similarity (ChromaDB) with BM25 keyword search, fused via **Reciprocal Rank Fusion (RRF)** — chosen over a weighted score blend because cosine similarity and BM25 scores aren't on comparable scales.
- **Cross-encoder reranking**: top candidates from hybrid retrieval are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2` for tighter relevance, loaded lazily and degrading gracefully if it fails to load.
- **LLM layer**: Groq for generation.
- **Evaluation harness**: `rag_evaluator.py` / `run_eval.py` for scoring retrieval + answer quality (RAGAS-style metrics).
- **API**: FastAPI (`api.py`) exposing the chatbot as an HTTP service.

## Stack

FastAPI · ChromaDB · `rank_bm25` · Sentence-Transformers (bi-encoder + cross-encoder) · Groq · LangChain document loaders

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # then add your GROQ_API_KEY
uvicorn api:app --reload
```

## Project structure

```
main.py              # Core RAG pipeline: ingestion, hybrid search, reranking
api.py                # FastAPI app / endpoints
rag_evaluator.py       # Evaluation metrics
run_eval.py            # Evaluation harness entry point
uploads/                # Drop source documents here for ingestion
```

## Status

Actively evolving — recent work (not yet in this snapshot) adds Redis-backed caching, per-session collection scoping for multi-user safety, and a TanStack Start frontend.
