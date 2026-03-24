"""
RAG Engine Module

This is the orchestrator that brings everything together:
1. Document Processing
2. Vector Storage
3. LLM Generation

This is the main "brain" of your RAG system.
"""

from typing import List, Dict
from pathlib import Path
import shutil

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from config import get_settings


class RAGEngine:
    """
    Main RAG Engine - orchestrates all components
    """
    
    def __init__(self):
        """
        Initialize RAG Engine with all components
        """
        self.settings = get_settings()
        
        print("🚀 Initializing RAG Engine...")
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        
        self.vector_store = VectorStore(
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.collection_name,
            embedding_model=self.settings.embedding_model
        )
        
        self.llm_handler = LLMHandler(
            api_key=self.settings.groq_api_key,
            model=self.settings.llm_model
        )
        
        print("✅ RAG Engine ready!")
    
    def upload_document(self, file_path: str, original_filename: str = None) -> Dict:
        """
        Upload and process a document
        
        This is the INDEXING phase:
        File → Load → Chunk → Embed → Store
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original name of the file
            
        Returns:
            Dictionary with upload status
        """
        try:
            print(f"\n{'='*50}")
            print(f"📄 Processing document: {original_filename or file_path}")
            print(f"{'='*50}")
            
            # Step 1: Process document (load + chunk)
            chunks = self.document_processor.process_document(file_path)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No content extracted from document"
                }
            
            # Step 2: Add to vector store (embed + store)
            self.vector_store.add_documents(chunks)
            
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            return {
                "success": True,
                "message": f"Document processed successfully",
                "chunks_created": len(chunks),
                "total_documents_in_db": stats["total_documents"],
                "filename": original_filename or Path(file_path).name
            }
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    def query(self, question: str) -> Dict:
        """
        Answer a question using RAG
        
        This is the QUERY phase:
        Question → Retrieve → Generate → Answer
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            print(f"\n{'='*50}")
            print(f"❓ Question: {question}")
            print(f"{'='*50}")
            
            # Step 1: Retrieve relevant chunks
            print(f"🔍 Searching for relevant chunks (top {self.settings.top_k_results})...")
            retrieved_chunks = self.vector_store.similarity_search(
                query=question,
                k=self.settings.top_k_results
            )
            
            if not retrieved_chunks:
                return {
                    "success": True,
                    "answer": "No relevant information found in the uploaded documents.",
                    "sources": [],
                    "question": question
                }
            
            print(f"✅ Found {len(retrieved_chunks)} relevant chunks")
            
            # Step 2: Generate response using LLM
            print("🤖 Generating response...")
            result = self.llm_handler.generate_response(question, retrieved_chunks)
            
            return {
                "success": True,
                "question": question,
                **result
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "question": question
            }
    
    def get_stats(self) -> Dict:
        """Get current system statistics"""
        return self.vector_store.get_collection_stats()
    
    def clear_database(self) -> Dict:
        """Clear all documents from the database"""
        try:
            self.vector_store.clear_collection()
            return {
                "success": True,
                "message": "Database cleared successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing database: {str(e)}"
            }
    
    def health_check(self) -> Dict:
        """
        Check if all components are working
        """
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check LLM
        try:
            llm_ok = self.llm_handler.test_connection()
            health["components"]["llm"] = "ok" if llm_ok else "error"
        except:
            health["components"]["llm"] = "error"
            health["status"] = "degraded"
        
        # Check vector store
        try:
            stats = self.vector_store.get_collection_stats()
            health["components"]["vector_store"] = "ok"
            health["total_documents"] = stats["total_documents"]
        except:
            health["components"]["vector_store"] = "error"
            health["status"] = "degraded"
        
        return health


# Example usage (for testing)
if __name__ == "__main__":
    from config import setup_directories
    
    # Setup directories
    setup_directories()
    
    # Initialize engine
    engine = RAGEngine()
    
    # Health check
    health = engine.health_check()
    print("\nHealth Check:", health)
    
    # Get stats
    stats = engine.get_stats()
    print("\nDatabase Stats:", stats)
