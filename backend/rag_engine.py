"""
RAG Engine Module

This is the orchestrator that brings everything together:
1. Document Processing
2. Vector Storage
3. LLM Generation

This is the main "brain" of RAG system.
"""

from typing import List, Dict, Generator
from pathlib import Path
import shutil
import json

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from reranker import Reranker
from agentic_engine import AgenticEngine
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
        
        print("[INFO] Initializing RAG Engine...")
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            chunking_strategy=self.settings.chunking_strategy,
            semantic_threshold_alpha=self.settings.semantic_threshold_alpha,
            semantic_max_chunk_size=self.settings.semantic_max_chunk_size,
            embedding_model_name=self.settings.embedding_model,
            api_key=self.settings.groq_api_key
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
        
        self.reranker = Reranker(
            model_name=self.settings.reranker_model
        )
        
        self.agentic_engine = AgenticEngine(
            vector_store=self.vector_store,
            llm_handler=self.llm_handler
        )
        
        print("[INFO] RAG Engine ready!")
    
    def upload_document(
        self, 
        file_path: str, 
        original_filename: str = None,
        chunking_strategy: str = None,
        semantic_threshold_alpha: float = None
    ) -> Dict:
        """
        Upload and process a document
        
        This is the INDEXING phase:
        File → Load → Chunk → Embed → Store
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original name of the file
            chunking_strategy: Optional chunking strategy override
            semantic_threshold_alpha: Optional threshold coefficient override
            
        Returns:
            Dictionary with upload status
        """
        try:
            print(f"\n{'='*50}")
            print(f"[INFO] Processing document: {original_filename or file_path}")
            print(f"[INFO] Chunking Strategy: {chunking_strategy or self.settings.chunking_strategy}")
            print(f"{'='*50}")
            
            # Step 1: Process document (load + chunk)
            chunks = self.document_processor.process_document(
                file_path,
                chunking_strategy=chunking_strategy,
                semantic_threshold_alpha=semantic_threshold_alpha
            )
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No content extracted from document"
                }
            
            # Step 2: Add to vector store (embed + store)
            if isinstance(chunks, tuple):
                child_docs, parent_mappings = chunks
                self.vector_store.add_documents(child_docs, parent_mappings=parent_mappings)
                chunks_created = len(child_docs)
            else:
                self.vector_store.add_documents(chunks)
                chunks_created = len(chunks)
            
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            return {
                "success": True,
                "message": f"Document processed successfully",
                "chunks_created": chunks_created,
                "total_documents_in_db": stats["total_documents"],
                "filename": original_filename or Path(file_path).name
            }
            
        except Exception as e:
            print(f"[ERROR] Error processing document: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    def query(
        self, 
        question: str, 
        history: List[Dict[str, str]] = None, 
        filters: Dict = None,
        use_hyde: bool = None,
        use_litm_packing: bool = None
    ) -> Dict:
        """
        Answer a question using RAG, conversation history, and metadata filters
        
        This is the QUERY phase:
        Question + History → Condense → (HyDE Generation) → Retrieve → Generate → Answer
        
        Args:
            question: User's question
            history: List of previous chat messages
            filters: Optional dictionary containing metadata query filters
            use_hyde: Optional boolean override to enable/disable HyDE query expansion.
                      If None, falls back to the default settings value.
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            print(f"\n{'='*50}")
            print(f"[INFO] Question: {question}")
            if filters:
                print(f"[INFO] Active Filters: {filters}")
            print(f"{'='*50}")
            
            # Step 1: Condense the query if there is conversation history
            condensed_query = self.llm_handler.condense_query(question, history)
            
            # Determine if HyDE should be used
            active_use_hyde = use_hyde if use_hyde is not None else self.settings.use_hyde
            
            semantic_query = None
            if active_use_hyde:
                # Generate hypothetical document for semantic similarity lookup
                semantic_query = self.llm_handler.generate_hypothetical_document(condensed_query)
            
            # Stage 1 Retrieval: Retrieve candidate chunks using Hybrid Search (BM25 + Semantic)
            print(f"[INFO] Stage 1 Retrieval: Fetching top {self.settings.rerank_top_n} candidates...")
            candidate_chunks = self.vector_store.hybrid_search(
                query=condensed_query,
                k=self.settings.rerank_top_n,
                where=filters,
                semantic_query=semantic_query
            )
            
            if not candidate_chunks:
                return {
                    "success": True,
                    "answer": "No relevant information found in the uploaded documents.",
                    "sources": [],
                    "question": question
                }
            
            # Stage 2 Re-ranking: Re-score and sort candidates using Cross-Encoder
            retrieved_chunks = self.reranker.rerank(
                query=condensed_query,
                chunks=candidate_chunks,
                top_k=self.settings.top_k_results
            )
            
            print(f"[INFO] Stage 2 Re-ranking: Selected top {len(retrieved_chunks)} chunks")
            
            # Determine if LitM context packing should be used
            active_use_litm = use_litm_packing if use_litm_packing is not None else self.settings.use_litm_packing
            
            # Step 2: Generate response using LLM (passing original query and history)
            print("[INFO] Generating response...")
            result = self.llm_handler.generate_response(
                query=question, 
                retrieved_chunks=retrieved_chunks,
                history=history,
                use_litm_packing=active_use_litm
            )
            
            return {
                "success": True,
                "question": question,
                **result
            }
            
        except Exception as e:
            print(f"[ERROR] Error processing query: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "question": question
            }
            
    def query_stream(
        self, 
        question: str, 
        history: List[Dict[str, str]] = None, 
        filters: Dict = None,
        use_hyde: bool = None,
        use_litm_packing: bool = None
    ) -> Generator[str, None, None]:
        """
        Answer a question using RAG, yielding sources first, then tokens.
        
        Yields:
            JSON strings representing sources and tokens.
        """
        try:
            print(f"\n{'='*50}")
            print(f"[INFO] Streaming Question: {question}")
            if filters:
                print(f"[INFO] Active Filters: {filters}")
            print(f"{'='*50}")
            
            # Step 1: Condense the query if there is conversation history
            condensed_query = self.llm_handler.condense_query(question, history)
            
            # Determine if HyDE should be used
            active_use_hyde = use_hyde if use_hyde is not None else self.settings.use_hyde
            
            semantic_query = None
            if active_use_hyde:
                # Generate hypothetical document for semantic similarity lookup
                semantic_query = self.llm_handler.generate_hypothetical_document(condensed_query)
            
            # Stage 1 Retrieval: Retrieve candidate chunks using Hybrid Search (BM25 + Semantic)
            print(f"[INFO] Stage 1 Retrieval: Fetching top {self.settings.rerank_top_n} candidates...")
            candidate_chunks = self.vector_store.hybrid_search(
                query=condensed_query,
                k=self.settings.rerank_top_n,
                where=filters,
                semantic_query=semantic_query
            )
            
            if not candidate_chunks:
                yield json.dumps({
                    "type": "sources",
                    "sources": []
                }) + "\n"
                yield json.dumps({
                    "type": "token",
                    "token": "No relevant information found in the uploaded documents."
                }) + "\n"
                return
            
            # Stage 2 Re-ranking: Re-score and sort candidates using Cross-Encoder
            retrieved_chunks = self.reranker.rerank(
                query=condensed_query,
                chunks=candidate_chunks,
                top_k=self.settings.top_k_results
            )
            
            print(f"[INFO] Stage 2 Re-ranking: Selected top {len(retrieved_chunks)} chunks (Stream)")
            
            # Format sources metadata and yield it first
            sources = []
            for text, metadata, distance in retrieved_chunks:
                relevance_score = 1.0 - distance
                source_name = metadata.get('source', 'Unknown')
                chunk_id = metadata.get('chunk_id', 'N/A')
                if metadata.get('is_parent'):
                    chunk_id = f"Parent {chunk_id}"
                
                sources.append({
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "preview": text[:200] + "..." if len(text) > 200 else text,
                    "relevance_score": max(0.0, min(1.0, relevance_score)),
                    "is_image": metadata.get("is_image", False),
                    "image_path": metadata.get("image_path")
                })
                
            yield json.dumps({
                "type": "sources",
                "sources": sources
            }) + "\n"
            
            # Step 2: Generate response stream using LLM
            print("[INFO] Initiating response stream...")
            active_use_litm = use_litm_packing if use_litm_packing is not None else self.settings.use_litm_packing
            
            token_stream = self.llm_handler.generate_response_stream(
                query=question,
                retrieved_chunks=retrieved_chunks,
                history=history,
                use_litm_packing=active_use_litm
            )
            
            for token in token_stream:
                yield json.dumps({
                    "type": "token",
                    "token": token
                }) + "\n"
                
        except Exception as e:
            print(f"[ERROR] Error in query stream: {str(e)}")
            yield json.dumps({
                "type": "error",
                "message": f"Error: {str(e)}"
            }) + "\n"

    def query_agentic_stream(
        self,
        question: str,
        history: List[Dict[str, str]] = None,
        filters: Dict = None,
        use_hyde: bool = None,
        use_litm_packing: bool = None
    ) -> Generator[str, None, None]:
        """
        Stream agentic query processing yielding thoughts, sources, and tokens via AgenticEngine.
        """
        return self.agentic_engine.execute_stream(
            query=question,
            filters=filters,
            use_hyde=use_hyde,
            use_litm_packing=use_litm_packing,
            history=history
        )
    
    def get_stats(self) -> Dict:
        """Get current system statistics"""
        return self.vector_store.get_collection_stats()
    
    def clear_database(self) -> Dict:
        """Clear all documents from the database and uploads directory"""
        try:
            # Clear vector store collection
            self.vector_store.clear_collection()
            
            # Clear files from uploads folder
            upload_dir_path = Path(self.settings.upload_dir)
            if upload_dir_path.exists():
                for item in upload_dir_path.iterdir():
                    if item.is_file():
                        try:
                            item.unlink()
                        except Exception as file_err:
                            print(f"[WARNING] Could not delete file {item.name}: {str(file_err)}")
                            
            return {
                "success": True,
                "message": "Database and uploaded documents cleared successfully"
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
