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
    llm_model: str = "llama-3.3-70b-versatile"  # Latest Llama 3.3 model (Dec 2024)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Parameters
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks for context
    top_k_results: int = 4  # Number of relevant chunks to retrieve
    
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
