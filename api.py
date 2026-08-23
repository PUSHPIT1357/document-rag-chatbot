from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
import shutil
from pathlib import Path

# Import the RAG bot
from dotenv import load_dotenv
load_dotenv()

from main import DocumentRAGBot

# ============================================
# Allowed File Extensions
# ============================================
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

# ============================================
# Lifespan Handler (Replaces deprecated on_event)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown hooks cleanly."""
    print("\n" + "="*60)
    print("🚀 Document RAG Chatbot API Starting...")
    print("="*60)

    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("chroma_db_free", exist_ok=True)

    # Global bot baseline configuration
    global bot
    bot = DocumentRAGBot()

    if bot.groq_service:
        print(f"✅ Groq client warmed up successfully! (model: {bot.groq_service.model_name})")
    else:
        print("⚠️ Running in fallback mode without active LLM orchestration.")

    print("📁 Directories created and engine verified.")
    print(f"📚 Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")

    yield  # --- Application Execution Window ---

    print("\n" + "="*60)
    print("🛑 Document RAG Chatbot API Shutting Down...")
    print("="*60)
    bot = None
    print("✅ Cleanup complete")
    print("="*60 + "\n")

# ============================================
# FastAPI Application Setup
# ============================================
app = FastAPI(
    title="Document RAG Chatbot API",
    description="RAG-powered document Q&A with Groq LLM integration. Supports PDF, DOCX, TXT, MD, and CSV files.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instance holder
bot: Optional[DocumentRAGBot] = None

# ============================================
# Pydantic Models (Updated for Pydantic V2)
# ============================================

class QuestionRequest(BaseModel):
    question: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {""}
        }
    )

class SourceInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity: float

class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceInfo]] = []
    query: str
    chunks_found: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "FIFO (First In, First Out) assumes oldest inventory items are sold first...",
                "sources": [{
                    "content": "FIFO method assumes that goods are sold in the order they were purchased...",
                    "metadata": {"page": 1, "chunk_id": 5},
                    "similarity": 0.95
                }],
                "query": "What is the difference between FIFO and Weighted Average?",
                "chunks_found": 3
            }
        }
    )

class HealthResponse(BaseModel):
    status: str
    bot_initialized: bool
    bot_ready: bool
    chunks: int
    supported_formats: List[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
    format: str
    chunks: int
    status: str
    free: bool = True
    llm_active: bool = False

class StatsResponse(BaseModel):
    pdf_loaded: bool
    pdf_name: Optional[str] = None
    chunks_count: int
    database: str
    embedding_model: str
    llm_status: str
    llm_active: bool = False
    supported_formats: List[str]
    free: bool = True
    api_keys_needed: bool = False

class RootResponse(BaseModel):
    message: str
    status: str
    version: str
    bot_ready: bool
    llm_active: bool
    api_keys_needed: bool
    supported_formats: List[str]
    endpoints: Dict[str, str]

# ============================================
# Helper Functions
# ============================================

def allowed_file(filename: str) -> bool:
    """Check if the file extension is in the allowed list"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def check_llm_status() -> bool:
    """Safely checks whether the Groq LLM service is online and ready."""
    return bool(bot and bot.groq_service)

def get_file_format(filename: str) -> str:
    """Extract clean file format from filename"""
    return Path(filename).suffix.lower()[1:]  # Remove the dot, e.g., '.pdf' -> 'pdf'

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information"""
    llm_active = check_llm_status()
    return RootResponse(
        message="Document RAG Chatbot API",
        status="running",
        version="1.0.0",
        bot_ready=bool(bot and bot.is_ready),
        llm_active=llm_active,
        api_keys_needed=not llm_active,
        supported_formats=list(ALLOWED_EXTENSIONS),
        endpoints={
            "upload": "/upload",
            "ask": "/ask",
            "health": "/health",
            "stats": "/stats",
            "reset": "/reset",
            "formats": "/formats",
            "docs": "/docs"
        }
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    Supported formats: PDF, DOCX, TXT, MD, CSV
    """
    global bot

    # Validate file extension
    if not allowed_file(file.filename):
        extension = Path(file.filename).suffix.lower()
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{extension}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Enforce that instance is alive
        if bot is None:
            bot = DocumentRAGBot()

        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        file_path = Path("uploads") / file.filename
        
        # Read and save file
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        file_format = get_file_format(file.filename)
        file_size = file_path.stat().st_size
        print(f"\n📄 File uploaded: {file_path} (Format: {file_format.upper()}, Size: {file_size} bytes)")

        # Build vector index through the shared embedding/vector-store architecture
        if not bot.build_pipeline(str(file_path)):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {file_format.upper()} document. The file might be corrupted or in an unsupported format."
            )

        print(f"✅ Pipeline built successfully with {len(bot.document_chunks)} chunks")
        
        return UploadResponse(
            message=f"{file_format.upper()} document processed successfully",
            filename=file.filename,
            format=file_format,
            chunks=len(bot.document_chunks),
            status="ready",
            free=True,
            llm_active=check_llm_status()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload pipeline exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded document.
    The bot will search through the document and provide an answer based on the content.
    """
    if not bot or not bot.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a document first using the `/upload` endpoint."
        )

    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be blank.")

        print(f"\n❓ Processing question: {request.question}")
        result = bot.ask(request.question)

        sources = [
            SourceInfo(
                content=source['content'],
                metadata=source.get('metadata', {}),
                similarity=source.get('similarity', 0.0)
            ) for source in result.get('sources', [])
        ]

        print(f"✅ Answer generated with {len(sources)} sources")
        
        return QuestionResponse(
            answer=result['answer'],
            sources=sources,
            query=result['query'],
            chunks_found=result.get('chunks_found', 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Execution crash on query handling: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal pipeline failure: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API and bot.
    Returns 200 even if no document is loaded.
    """
    return HealthResponse(
        status="healthy",
        bot_initialized=bool(bot),
        bot_ready=bool(bot and bot.is_ready),
        chunks=len(bot.document_chunks) if bot else 0,
        supported_formats=list(ALLOWED_EXTENSIONS)
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get detailed statistics about the current bot state and configuration.
    Returns 200 with empty stats if no document is loaded.
    """
    if not bot or not bot.is_ready:
        # Return 200 with empty stats instead of 404
        return StatsResponse(
            pdf_loaded=False,
            pdf_name=None,
            chunks_count=0,
            database="ChromaDB (FREE)",
            embedding_model="all-MiniLM-L6-v2 (FREE)",
            llm_status="No document loaded",
            llm_active=check_llm_status(),
            supported_formats=list(ALLOWED_EXTENSIONS),
            free=True,
            api_keys_needed=not check_llm_status()
        )

    try:
        stats = bot.get_stats()
        llm_active = check_llm_status()

        return StatsResponse(
            pdf_loaded=stats['pdf_loaded'],
            pdf_name=stats['pdf_name'],
            chunks_count=stats['chunks_count'],
            database=stats['database'],
            embedding_model=stats['embedding_model'],
            llm_status=stats['groq_status'],
            llm_active=llm_active,
            supported_formats=list(ALLOWED_EXTENSIONS),
            free=True,
            api_keys_needed=not llm_active
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats internal processing issue: {str(e)}")

@app.delete("/reset")
async def reset_bot():
    """
    Reset the bot state and clear all loaded documents.
    """
    global bot
    if bot:
        bot.is_ready = False
        bot.document_chunks = []
        bot.pdf_name = None
        try:
            bot.chroma_client.delete_collection(bot.collection_name)
        except Exception:
            pass
        print("🔄 Bot state reset successfully")
    
    return {
        "message": "Bot reset successfully. All documents cleared.",
        "status": "reset",
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }

@app.get("/formats")
async def get_supported_formats():
    """
    Get list of supported document formats with descriptions.
    """
    return {
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "descriptions": {
            ".pdf": "PDF documents",
            ".docx": "Microsoft Word documents",
            ".txt": "Plain text files",
            ".md": "Markdown files",
            ".csv": "CSV spreadsheet files"
        }
    }

# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail, 
            "status_code": exc.status_code,
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled system exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server configuration error", 
            "detail": str(exc),
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")