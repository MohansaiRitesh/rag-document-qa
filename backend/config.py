"""
Configuration Management for RAG System

This file centralizes all configuration settings:
- API keys
- Model configurations
- File paths
- System parameters
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for pydantic v1
    from pydantic import BaseSettings

from functools import lru_cache
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent



class Settings(BaseSettings):
    """
    Settings class using Pydantic for type validation
    Automatically loads from .env file
    """
    
    # API Keys
    groq_api_key: str = ""
    
    # Model Configuration
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Parameters
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks for context
    chunking_strategy: str = "recursive"  # "recursive", "semantic", or "hierarchical"
    semantic_threshold_alpha: float = 1.0  # Threshold calculation std dev scale
    semantic_max_chunk_size: int = 1500  # Max character fallback limit
    parent_chunk_size: int = 1500  # Parent chunk character size
    parent_chunk_overlap: int = 200  # Parent chunk overlap
    child_chunk_size: int = 300  # Child chunk character size
    child_chunk_overlap: int = 50  # Child chunk overlap
    top_k_results: int = 4  # Number of relevant chunks to retrieve
    use_hyde: bool = True  # Enable HyDE (Hypothetical Document Embeddings) Query Expansion
    use_litm_packing: bool = True  # Enable Lost-in-the-Middle prompt context packing
    
    # Re-ranking Parameters
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 15  # Candidate chunks to retrieve before re-ranking
    
    # Vector Database
    vector_db_path: str = str(BASE_DIR / "data" / "chromadb")
    collection_name: str = "documents"
    
    # File Upload
    upload_dir: str = str(BASE_DIR / "data" / "uploads")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: set = {".pdf", ".docx", ".txt"}
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Create a cached instance of settings
    lru_cache ensures we only create one instance
    """
    return Settings()


# Create directories if they don't exist
def setup_directories():
    """Initialize required directories"""
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vector_db_path, exist_ok=True)
    print("[INFO] Directories initialized")
