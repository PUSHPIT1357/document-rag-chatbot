import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from groq import Groq
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ============================================
# ENVIRONMENT SETUP
# ============================================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# ============================================
# GROQ SERVICE CLASS (unchanged)
# ============================================
class GroqService:
    """Groq API service with working model selection"""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in environment variables")

        self.client = Groq(api_key=GROQ_API_KEY)

        model_candidates = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b"
        ]

        self.model = None
        self.model_name = None

        print("🔍 Testing available Groq models...")

        for model_name in model_candidates:
            try:
                test_response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test"}
                    ],
                    model=model_name,
                    max_tokens=10
                )
                if test_response and test_response.choices:
                    self.model = model_name
                    self.model_name = model_name
                    print(f"✅ Groq initialized with: {model_name}")
                    break
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue

        if not self.model:
            raise ValueError("No working Groq model found. Please check your API key.")

    def generate_answer(self, query: str, chunks: List[str]) -> str:
        """Generate an answer using Groq based on provided context chunks"""

        if not chunks:
            return "No relevant information found in the document."

        context = "\n\n".join([
            f"[Source {i+1}]:\n{chunk}"
            for i, chunk in enumerate(chunks[:5])
        ])

        prompt = f"""You are a helpful document assistant.

Answer ONLY from the context below.
If the answer is not present in the context, say "I don't know" or "Information not found in the document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (based ONLY on the context above):"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise document analyst. Always base your answers on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1000
            )

            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "No response generated from the model."

        except Exception as e:
            print(f"🔥 Groq Error: {e}")
            return f"Error generating answer: {str(e)}"


# ============================================
# MAIN RAG BOT CLASS
# ============================================
class DocumentRAGBot:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_free",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "documents"

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print("✅ Connected to existing Chroma collection.")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new vector storage space.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.document_chunks = []
        self.is_ready = False
        self.pdf_name = None

        # ============================================
        # BM25 (keyword search) index state
        # ============================================
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_texts: List[str] = []      # chunk text, aligned by chunk_id
        self.bm25_metadatas: List[Dict] = []         # metadata, aligned by chunk_id

        # ============================================
        # Reranker (cross-encoder) — loaded lazily on first use
        # so a bot instance that never reranks doesn't pay the load cost
        # ============================================
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_load_failed = False

        # ============================================
        # Initialize Groq Service
        # ============================================
        self.groq_service = None
        try:
            self.groq_service = GroqService()
            print("✅ Groq Service ready!")
        except ValueError as e:
            print(f"⚠️ {e}")
            print("⚠️ Groq will not be available. Using fallback mode.")
        except Exception as e:
            print(f"⚠️ Failed to initialize Groq: {e}")
            print("⚠️ Using fallback mode.")

        print("🤖 Document RAG Bot core initialized.")

    # ============================================
    # DOCUMENT LOADING (unchanged)
    # ============================================
    def load_document(self, file_path: str):
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            print(f"❌ Unsupported file format: {extension}")
            print(f"   Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}")
            return None

        try:
            print(f"📄 Loading {extension} document: {file_path.name}")

            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                print(f"   ✅ Loaded {len(docs)} pages from PDF")

            elif extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                print(f"   ✅ Loaded DOCX document")

            elif extension in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                print(f"   ✅ Loaded text/markdown document")

            elif extension == ".csv":
                loader = CSVLoader(
                    file_path=str(file_path),
                    encoding='utf-8',
                    csv_args={'delimiter': ',', 'quotechar': '"'}
                )
                docs = loader.load()
                print(f"   ✅ Loaded CSV with {len(docs)} rows")

            else:
                raise ValueError(f"Unhandled file extension: {extension}")

            return docs

        except Exception as e:
            print(f"❌ Error loading {extension} document: {e}")
            return None

    

    def split_documents(self, docs):
        chunks = self.text_splitter.split_documents(docs)
        print(f"✂️ Extracted {len(chunks)} structural context blocks.")
        return chunks

    def create_embeddings(self, texts: List[str]):
        return self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32)

    # ============================================
    # Tokenizer for BM25
    # ============================================
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase word tokenizer. Good enough for BM25 term matching
        on financial docs (keeps alphanumerics, e.g. 'IFRS9', 'SEBI', figures)."""
        text = text.lower()
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text)

    def store_in_vector_db(self, chunks, pdf_name: str):
        ids = [f"{pdf_name}_{i:04d}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy()
            metadata['chunk_id'] = i
            metadata['pdf_name'] = pdf_name
            metadatas.append(metadata)

        embeddings = self.create_embeddings(texts)
        embeddings_list = embeddings.tolist()

        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )

        # ============================================
        # Build BM25 keyword index (aligned by chunk_id / list position)
        # ============================================
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_texts = texts
        self.bm25_metadatas = metadatas
        print(f"🔎 BM25 keyword index built over {len(texts)} chunks.")

        self.document_chunks = chunks
        self.is_ready = True
        self.pdf_name = pdf_name
        return chunks

    def build_pipeline(self, file_path: str) -> bool:
        docs = self.load_document(file_path)
        if not docs:
            return False

        chunks = self.split_documents(docs)
        doc_name = Path(file_path).stem
        self.store_in_vector_db(chunks, doc_name)
        return True

    # ============================================
    # VECTOR SEARCH (unchanged, still usable standalone)
    # ============================================
    def search(self, query: str, k: int = 5):
        if not self.is_ready:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        relevant_chunks = []
        if results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if results.get('distances') else 1.0
                relevant_chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': float(1.0 - distance)
                })
        return relevant_chunks

    # ============================================
    # NEW: KEYWORD (BM25) SEARCH
    # ============================================
    def keyword_search(self, query: str, k: int = 10):
        """Sparse lexical search over the same chunk set. Good for exact terms,
        codes, section numbers, and named entities that embeddings can blur."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        if len(scores) == 0:
            return []

        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            idx = int(idx)
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                'content': self.bm25_corpus_texts[idx],
                'metadata': self.bm25_metadatas[idx],
                'bm25_score': score
            })
        return results

    # ============================================
    # NEW: HYBRID SEARCH (RRF fusion of vector + BM25)
    # ============================================
    def hybrid_search(self, query: str, k: int = 5, vector_k: int = 15,
                       keyword_k: int = 15, rrf_k: int = 60):
        """
        Combines vector similarity and BM25 keyword rankings using
        Reciprocal Rank Fusion. RRF is used instead of a weighted score
        blend because cosine similarity (0-1) and BM25 (unbounded, corpus-
        dependent) scores aren't on comparable scales — rank position is.

        score(chunk) = sum over each retriever of 1 / (rrf_k + rank)

        rrf_k=60 is the standard default from the original RRF paper;
        it dampens the influence of any single very-high rank.
        """
        if not self.is_ready:
            return []

        vector_hits = self.search(query, k=vector_k)
        keyword_hits = self.keyword_search(query, k=keyword_k) if self.bm25 else []

        # Fall back to vector-only if BM25 isn't built yet (e.g. old collection loaded)
        if not self.bm25:
            return vector_hits[:k]

        fused_scores: Dict[int, float] = {}
        content_lookup: Dict[int, Dict] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit['metadata'].get('chunk_id')
            if cid is None:
                continue
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            content_lookup[cid] = {**hit, 'matched_by': ['vector']}

        for rank, hit in enumerate(keyword_hits):
            cid = hit['metadata'].get('chunk_id')
            if cid is None:
                continue
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid in content_lookup:
                content_lookup[cid]['matched_by'].append('keyword')
            else:
                content_lookup[cid] = {**hit, 'similarity': hit.get('bm25_score', 0.0),
                                        'matched_by': ['keyword']}

        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for cid, rrf_score in ranked_ids:
            hit = content_lookup[cid]
            results.append({
                'content': hit['content'],
                'metadata': hit['metadata'],
                'similarity': hit.get('similarity', 0.0),   
                'rrf_score': rrf_score,
                'matched_by': hit['matched_by']
            })
        return results

    # ============================================
    # NEW: RERANKER (cross-encoder relevance scoring)
    # ============================================
    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """Lazy-loaded cross-encoder. Returns None (and disables reranking
        permanently for this instance) if the model can't be loaded, so a
        network hiccup degrades gracefully to the un-reranked ordering
        instead of crashing the request."""
        if self._reranker is None and not self._reranker_load_failed:
            try:
                print("🧠 Loading cross-encoder reranker (first use)...")
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Reranker ready.")
            except Exception as e:
                print(f"⚠️ Reranker failed to load, skipping reranking: {e}")
                self._reranker_load_failed = True
        return self._reranker

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-scores retrieved candidates with a cross-encoder that jointly
        encodes (query, chunk) pairs — this is more accurate than the bi-encoder
        similarity used for initial retrieval, because the bi-encoder scores
        query and chunk independently and only compares their vectors afterward,
        while the cross-encoder attends over both texts together.

        Use this AFTER a wider retrieval pass (hybrid/vector/keyword), not as
        a replacement for it — cross-encoders are too slow to run over an
        entire corpus, only over a shortlist.
        """
        if not candidates:
            return []

        model = self.reranker
        if model is None:
            # Reranker unavailable — fall back to the incoming order untouched
            return candidates[:top_k]

        pairs = [(query, c['content']) for c in candidates]
        raw_scores = model.predict(pairs)

        # ms-marco-MiniLM cross-encoders output raw logits (unbounded, can be
        # negative), not a 0-1 probability like your cosine similarity. Squash
        # through a sigmoid so rerank_score is directly comparable to/displayable
        # alongside similarity (both become a 0-1 "match confidence").
        normalized_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

        reranked = []
        for c, raw, norm in zip(candidates, raw_scores, normalized_scores):
            reranked.append({
                **c,
                'rerank_score_raw': float(raw),
                'rerank_score': float(norm)
            })

        reranked.sort(key=lambda c: c['rerank_score'], reverse=True)
        return reranked[:top_k]

    def ask(self, query: str, k: int = 3, mode: str = "hybrid",
             rerank: bool = True, retrieval_k: int = 15):
        """
        mode: "hybrid" (default), "vector", or "keyword" — controls the
              initial retrieval pass.
        rerank: if True (default), retrieves `retrieval_k` candidates and
                narrows to `k` via cross-encoder reranking. If False, just
                takes the top `k` straight from retrieval.
        retrieval_k: candidate pool size handed to the reranker. Should be
                     notably larger than `k` — reranking only helps if it has
                     a wider net to pick better results out of.
        """
        if not self.is_ready:
            return {
                'answer': "State missing. Load text context using the UI upload menu.",
                'sources': [],
                'query': query,
                'chunks_found': 0
            }

        pool_size = retrieval_k if rerank else k

        if mode == "vector":
            candidates = self.search(query, pool_size)
        elif mode == "keyword":
            candidates = self.keyword_search(query, pool_size)
        else:
            candidates = self.hybrid_search(query, k=pool_size)

        chunks = self.rerank(query, candidates, top_k=k) if rerank else candidates[:k]

        if not chunks:
            return {
                'answer': "No context matches discovered in document vector arrays.",
                'sources': [],
                'query': query,
                'chunks_found': 0
            }

        # True only if reranking was requested AND the cross-encoder actually
        # loaded and ran — distinguishes "reranked, no reordering was needed"
        # from "reranking silently didn't happen" so the UI/logs never lie.
        reranked_actually = rerank and self.reranker is not None

        answer = self._format_answer(query, chunks)
        sources = []
        for c in chunks[:3]:
            retrieval_score = c.get('similarity', c.get('bm25_score', 0.0))
            rerank_score = c.get('rerank_score')

            display_score = rerank_score if (reranked_actually and rerank_score is not None) else retrieval_score
            score_type = "rerank" if (reranked_actually and rerank_score is not None) else "retrieval"

            sources.append({
                'content': c['content'][:500] + ('...' if len(c['content']) > 500 else ''),
                'metadata': c['metadata'],
                # Kept as the original 0-1 float — your existing frontend already
                # reads this key and multiplies by 100 itself to show "X% match".
                # It now reflects the rerank score when reranking ran, so the
                # displayed percentage will actually change post-rerank.
                'similarity': round(display_score, 4),
                'score_type': score_type,              # "rerank" or "retrieval" — optional label for the UI
                'retrieval_score': round(retrieval_score, 4),
                'rerank_score': round(rerank_score, 4) if rerank_score is not None else None,
                'matched_by': c.get('matched_by')
            })

        return {
            'answer': answer,
            'sources': sources,
            'query': query,
            'chunks_found': len(chunks),
            'mode': mode,
            'reranked': reranked_actually
        }

    def ask_for_eval(self, query: str, k: int = 3, mode: str = "hybrid",
                      rerank: bool = True, retrieval_k: int = 15) -> Dict[str, Any]:
        """
        Same retrieval → (rerank) → generate pipeline as ask(), but returns
        the FULL, untruncated context text instead of the 500-char preview
        `ask()` uses for UI display. Use this for evaluation — scoring
        faithfulness against a truncated ("...") chunk would silently
        misjudge whether the answer is actually grounded in the source.
        """
        if not self.is_ready:
            return {'query': query, 'answer': '', 'contexts': []}

        pool_size = retrieval_k if rerank else k

        if mode == "vector":
            candidates = self.search(query, pool_size)
        elif mode == "keyword":
            candidates = self.keyword_search(query, pool_size)
        else:
            candidates = self.hybrid_search(query, k=pool_size)

        chunks = self.rerank(query, candidates, top_k=k) if rerank else candidates[:k]

        if not chunks:
            return {'query': query, 'answer': '', 'contexts': []}

        answer = self._format_answer(query, chunks)
        return {
            'query': query,
            'answer': answer,
            'contexts': [c['content'] for c in chunks],  # full text, no truncation
        }

    def _format_answer(self, query: str, chunks: List[Dict]) -> str:
        chunk_texts = [c['content'] for c in chunks[:5]]

        if self.groq_service:
            try:
                answer = self.groq_service.generate_answer(query, chunk_texts)
                if answer and not answer.startswith("Error"):
                    return answer
            except Exception as e:
                print(f"⚠️ Groq error: {e}")

        return self._clean_fallback_answer(chunks)

    def _clean_fallback_answer(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant information found in the document."

        summary = "⚠️ Cloud LLM Engine Offline. Found highly matching context blocks:\n\n"
        for i, chunk in enumerate(chunks[:3], 1):
            content = chunk["content"].strip().replace("\n", " ")
            if len(content) > 250:
                content = content[:250].rstrip() + "..."
            sim = chunk.get('similarity', chunk.get('bm25_score', 0.0))
            summary += f"- Block {i} (Match Score: {sim:.2f}): {content}\n"
        return summary

    def get_stats(self):
        return {
            'pdf_loaded': self.is_ready,
            'pdf_name': self.pdf_name,
            'chunks_count': len(self.document_chunks),
            'database': 'ChromaDB (FREE)',
            'embedding_model': 'all-MiniLM-L6-v2 (FREE)',
            'keyword_index': 'BM25Okapi (rank_bm25)' if self.bm25 else 'Not built',
            'reranker_status': (
                'Loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)' if self._reranker is not None
                else 'Not loaded yet (lazy)' if not self._reranker_load_failed
                else 'Unavailable (load failed, falling back to retrieval order)'
            ),
            'groq_status': f'Active ({self.groq_service.model_name})' if self.groq_service else 'Fallback (No AI)'
        }


# ============================================
# TEST FUNCTION
# ============================================
def test_bot():
    bot = DocumentRAGBot()
    base_dir = Path(__file__).resolve().parent

    test_files = [
        "Financial-Reporting-FR-FAQ-Revised-Final.pdf",
        "test_document.docx",
        "data.csv",
        "notes.txt",
        "readme.md"
    ]

    test_file = None
    for file_name in test_files:
        candidate = base_dir / file_name
        if candidate.exists():
            test_file = candidate
            break

    if not test_file:
        print(f"💡 No test documents found in project directory.")
        print(f"   Supported formats: {', '.join(DocumentRAGBot.SUPPORTED_EXTENSIONS)}")
        print("   Please place a document in the project directory.")
        return

    if bot.build_pipeline(str(test_file)):
        test_questions = [
            "What are the key financial metrics mentioned?",
            "What is the reporting period covered?"
        ]

        print("\n" + "="*60)
        print("🧪 TESTING WITH DOCUMENT QUESTIONS (hybrid search + rerank)")
        print("="*60)

        for question in test_questions:
            result = bot.ask(question, mode="hybrid", rerank=True)
            print(f"\n❓ Question: {question}")
            print(f"✅ Answer: {result['answer']}")
            print(f"📚 Found {result['chunks_found']} sources via {result['mode']} "
                  f"(reranked={result['reranked']})")
            print("-"*60)


if __name__ == "__main__":
    test_bot()