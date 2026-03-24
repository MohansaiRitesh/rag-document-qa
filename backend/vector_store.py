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
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✅ Embedding model loaded")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )
        
        print(f"✅ Vector store initialized with {self.collection.count()} existing documents")
    
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
            print("⚠️ No documents to add")
            return
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Create embeddings
        print(f"🔄 Creating embeddings for {len(texts)} chunks...")
        embeddings = self.create_embeddings(texts)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(documents)} chunks to vector store")
    
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
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"✅ Deleted collection: {self.collection_name}")
    
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
            metadata={"description": "RAG document collection"}
        )
        print("✅ Collection cleared")


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
