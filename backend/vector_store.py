"""
Vector Store Module

This module manages the vector database (ChromaDB):
1. Creating embeddings from text
2. Storing embeddings with metadata
3. Similarity search for retrieval

Key Concepts:
- Embeddings: Numerical representations of text (vectors)
- Similarity Search: Finding text chunks similar to a query
- ChromaDB: Local vector database for storing and searching embeddings
"""

from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
import uuid
from bm25 import BM25


class VectorStore:
    """
    Manages embeddings and vector database operations
    """
    
    def __init__(
        self, 
        persist_directory: str,
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection
            embedding_model: HuggingFace model for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        print(f"[INFO] Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("[INFO] Embedding model loaded")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection with Cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "RAG document collection"}
        )
        
        print(f"[INFO] Vector store initialized with {self.collection.count()} existing documents")
        
        # Initialize BM25 and fit with existing documents
        self.bm25 = BM25()
        self._fit_bm25_from_db()
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store
        
        Args:
            documents: List of LangchainDocument objects
        """
        if not documents:
            print("[WARNING] No documents to add")
            return
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Create embeddings
        print(f"[INFO] Creating embeddings for {len(texts)} chunks...")
        embeddings = self.create_embeddings(texts)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[SUCCESS] Added {len(documents)} chunks to vector store")
        
        # Refit BM25 index
        self._fit_bm25_from_db()
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (text, metadata, distance) tuples
        """
        # Create query embedding
        query_embedding = self.create_embeddings([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                formatted_results.append((text, metadata, distance))
        
        return formatted_results

    def _fit_bm25_from_db(self) -> None:
        """Fetch all documents from ChromaDB and fit the BM25 index"""
        results = self.collection.get()
        self.indexed_documents = results.get('documents', []) or []
        self.indexed_metadatas = results.get('metadatas', []) or []
        self.bm25.fit(self.indexed_documents)

    def hybrid_search(
        self, 
        query: str, 
        k: int = 4,
        rrf_k: int = 60
    ) -> List[Tuple[str, Dict, float]]:
        """
        Perform Hybrid Search combining Semantic (ChromaDB) and Lexical (BM25) search,
        merging results using Reciprocal Rank Fusion (RRF).
        """
        # Step 1: Run semantic search (get a larger set of candidates, e.g. 20)
        semantic_candidates = self.similarity_search(query, k=20)
        
        # Step 2: Run lexical search (get a larger set of candidates, e.g. 20)
        lexical_results = self.bm25.search(query, top_n=20)
        
        # Step 3: Perform Reciprocal Rank Fusion
        # A unique identifier is required for each chunk. (source, chunk_id) is perfect.
        def get_chunk_key(metadata: Dict) -> Tuple[str, int]:
            return (metadata.get('source', 'Unknown'), metadata.get('chunk_id', -1))
            
        rrf_scores: Dict[Tuple[str, int], float] = {}
        chunk_lookup: Dict[Tuple[str, int], Tuple[str, Dict, float]] = {}
        
        # Track ranks in semantic search
        for rank, (text, metadata, distance) in enumerate(semantic_candidates, 1):
            key = get_chunk_key(metadata)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            chunk_lookup[key] = (text, metadata, distance)
            
        # Track ranks in lexical search
        for rank, (doc_idx, bm25_score) in enumerate(lexical_results, 1):
            text = self.indexed_documents[doc_idx]
            metadata = self.indexed_metadatas[doc_idx]
            key = get_chunk_key(metadata)
            
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            
            # If the chunk is not in lookup, add it with a default distance of 0.5 (50% similarity)
            if key not in chunk_lookup:
                chunk_lookup[key] = (text, metadata, 0.5)
                
        # Step 4: Sort chunks by RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Step 5: Take top K results
        top_keys = sorted_keys[:k]
        
        results = [chunk_lookup[key] for key in top_keys]
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"[SUCCESS] Deleted collection: {self.collection_name}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from collection"""
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "RAG document collection"}
        )
        print("[SUCCESS] Collection cleared")
        
        # Refit BM25 index (will be empty)
        self._fit_bm25_from_db()


# Example usage (for testing)
if __name__ == "__main__":
    from langchain.schema import Document
    
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory="./test_chromadb",
        collection_name="test_collection"
    )
    
    # Create sample documents
    docs = [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "test.txt", "chunk_id": 0}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test.txt", "chunk_id": 1}
        )
    ]
    
    # Add documents
    vector_store.add_documents(docs)
    
    # Search
    results = vector_store.similarity_search("What is Python?", k=2)
    
    print("\nSearch Results:")
    for text, metadata, distance in results:
        print(f"\nText: {text}")
        print(f"Source: {metadata.get('source')}")
        print(f"Distance: {distance}")
