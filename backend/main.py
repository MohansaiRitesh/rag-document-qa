"""
FastAPI Backend for RAG System

This creates REST API endpoints for:
1. Document upload
2. Question answering
3. System statistics
4. Health checks

API Documentation will be available at: http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import uvicorn

from config import get_settings, setup_directories
from rag_engine import RAGEngine


# Initialize FastAPI app
app = FastAPI(
    title="Maester AI API",
    description="Upload documents and ask questions with cited, grounded answers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We need to specify frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings
settings = get_settings()

# Mount static files to serve extracted images
app.mount("/images", StaticFiles(directory=settings.extracted_images_dir), name="images")

# Initialize RAG Engine
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize system on startup
    """
    global rag_engine
    
    print("\n" + "="*50)
    print("[INFO] Starting RAG System")
    print("="*50)
    
    # Setup directories
    setup_directories()
    
    # Initialize RAG engine
    rag_engine = RAGEngine()
    
    print("\n[INFO] System ready!")
    print(f"[INFO] API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("="*50 + "\n")


# Pydantic models for request/response
class Message(BaseModel):
    """Model for a single message in the conversation history"""
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    """Model for question queries with history and metadata filters"""
    question: str
    history: Optional[List[Message]] = []
    filters: Optional[Dict] = None
    use_hyde: Optional[bool] = None
    use_litm_packing: Optional[bool] = None
    use_agentic_rag: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of the document?",
                "history": [
                    {"role": "user", "content": "Who founded TechVision?"},
                    {"role": "assistant", "content": "TechVision was founded by Dr. Sarah Chen and Michael Rodriguez."}
                ],
                "filters": {
                    "file_type": "pdf"
                },
                "use_agentic_rag": True
            }
        }


class QueryResponse(BaseModel):
    """Model for query responses"""
    success: bool
    question: str
    answer: Optional[str] = None
    sources: Optional[List[Dict]] = None
    context_used: Optional[bool] = None
    message: Optional[str] = None


# API ENDPOINTS
@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Maester AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        System health status
    """
    health = rag_engine.health_check()
    
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    
    Returns:
        Database statistics
    """
    try:
        stats = rag_engine.get_stats()
        return {
            "success": True,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata-values")
async def get_metadata_values():
    """
    Get all unique sources and file types in the DB for filter options
    """
    try:
        results = rag_engine.vector_store.collection.get()
        metadatas = results.get("metadatas", []) or []
        
        sources = sorted(list(set(m.get("source") for m in metadatas if m.get("source"))))
        file_types = sorted(list(set(m.get("file_type") for m in metadatas if m.get("file_type"))))
        
        return {
            "success": True,
            "sources": sources,
            "file_types": file_types
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: Optional[str] = Form("recursive"),
    semantic_threshold_alpha: Optional[float] = Form(1.0)
):
    """
    Upload a document for indexing
    
    Args:
        file: Document file (PDF, DOCX, TXT)
        chunking_strategy: Chunking algorithm to use (recursive or semantic)
        semantic_threshold_alpha: Scaling factor for standard deviation threshold
        
    Returns:
        Upload status and statistics
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
            )
        
        # Validate file size
        contents = await file.read()
        if len(contents) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_file_size / (1024*1024)}MB"
            )
        
        # Save file temporarily
        temp_file_path = Path(settings.upload_dir) / file.filename
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Process document
        result = rag_engine.upload_document(
            file_path=str(temp_file_path),
            original_filename=file.filename,
            chunking_strategy=chunking_strategy,
            semantic_threshold_alpha=semantic_threshold_alpha
        )
        
        if result["success"]:
            return JSONResponse(
                content=result,
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about uploaded documents
    
    Args:
        request: Query request with question
        
    Returns:
        Answer with sources and citations
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process query with history, filters, and toggles
        history_list = [msg.model_dump() for msg in request.history] if request.history else []
        result = rag_engine.query(
            question=request.question,
            history=history_list,
            filters=request.filters,
            use_hyde=request.use_hyde,
            use_litm_packing=request.use_litm_packing
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-stream")
async def query_documents_stream(request: QueryRequest):
    """
    Ask a question about uploaded documents and stream response tokens
    
    Args:
        request: Query request with question
        
    Returns:
        Server-Sent Events event-stream yielding sources and tokens
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
            
        history_list = [msg.model_dump() for msg in request.history] if request.history else []
        
        if request.use_agentic_rag:
            generator = rag_engine.query_agentic_stream(
                question=request.question,
                history=history_list,
                filters=request.filters,
                use_hyde=request.use_hyde,
                use_litm_packing=request.use_litm_packing
            )
        else:
            generator = rag_engine.query_stream(
                question=request.question,
                history=history_list,
                filters=request.filters,
                use_hyde=request.use_hyde,
                use_litm_packing=request.use_litm_packing
            )
        
        return StreamingResponse(generator, media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_database():
    """
    Clear all documents from the database
    
    Returns:
        Deletion status
    """
    try:
        result = rag_engine.clear_database()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    
    Returns:
        List of uploaded files
    """
    try:
        upload_dir = Path(settings.upload_dir)
        files = []
        
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix
                    })
        
        return {
            "success": True,
            "files": files,
            "total": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Run the FastAPI server
    
    Access at: http://localhost:8000
    API Docs: http://localhost:8000/docs
    """
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True  # Auto-reload on code changes
    )
