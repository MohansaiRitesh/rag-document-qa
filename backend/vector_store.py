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
from pathlib import Path
import os
import json
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
        self.parent_store_path = str(Path(persist_directory).parent / "parent_store.json")
        
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

    def _load_parent_store(self) -> Dict[str, Dict]:
        """Load parent documents from JSON file"""
        if os.path.exists(self.parent_store_path):
            try:
                with open(self.parent_store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error reading parent store: {str(e)}")
        return {}

    def _save_parent_store(self, store: Dict[str, Dict]) -> None:
        """Save parent documents to JSON file"""
        try:
            with open(self.parent_store_path, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARNING] Error writing parent store: {str(e)}")
            
    def _resolve_and_deduplicate_parents(
        self, 
        candidates: List[Tuple[str, Dict, float]], 
        k: int
    ) -> List[Tuple[str, Dict, float]]:
        """
        Resolve child chunks to parent documents using the parent store and deduplicate them.
        Keep non-hierarchical chunks as-is.
        """
        parent_store = self._load_parent_store()
        seen_parents = set()
        resolved_results = []
        
        for text, metadata, distance in candidates:
            parent_id = metadata.get("parent_id")
            if parent_id and parent_id in parent_store:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_info = parent_store[parent_id]
                    # Retain child's distance score but return parent's text and metadata
                    resolved_results.append((
                        parent_info["text"],
                        parent_info["metadata"],
                        distance
                    ))
            else:
                # Non-hierarchical chunk: create a unique key to deduplicate if any duplicates exist
                chunk_key = (metadata.get("source"), metadata.get("chunk_id"), text[:100])
                if chunk_key not in seen_parents:
                    seen_parents.add(chunk_key)
                    resolved_results.append((text, metadata, distance))
                    
        return resolved_results[:k]
    
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
    
    def add_documents(self, documents: List[Document], parent_mappings: Dict = None) -> None:
        """
        Add documents to vector store
        
        Args:
            documents: List of LangchainDocument objects
            parent_mappings: Optional dict of parent_id -> {"text": parent_text, "metadata": parent_metadata}
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
        
        # Update parent lookup store if provided
        if parent_mappings:
            print(f"[INFO] Saving {len(parent_mappings)} parent mappings...")
            store = self._load_parent_store()
            store.update(parent_mappings)
            self._save_parent_store(store)
            print("[SUCCESS] Parent mappings saved")
            
        # Refit BM25 index
        self._fit_bm25_from_db()
    
    @staticmethod
    def _match_metadata_filter(metadata: Dict, where: Dict) -> bool:
        """
        Evaluate if a document metadata dictionary matches a ChromaDB where clause.
        Supports direct matches, logical operators ($and, $or), and comparison operators ($eq, $ne, $in, $nin, $gt, $gte, $lt, $lte).
        """
        if not where:
            return True
            
        for key, value in where.items():
            if key == "$and":
                if not isinstance(value, list):
                    return False
                return all(VectorStore._match_metadata_filter(metadata, sub_filter) for sub_filter in value)
            elif key == "$or":
                if not isinstance(value, list):
                    return False
                return any(VectorStore._match_metadata_filter(metadata, sub_filter) for sub_filter in value)
            else:
                meta_val = metadata.get(key)
                if isinstance(value, dict):
                    for op, op_val in value.items():
                        if op == "$eq" and meta_val != op_val:
                            return False
                        elif op == "$ne" and meta_val == op_val:
                            return False
                        elif op == "$in" and (not isinstance(op_val, list) or meta_val not in op_val):
                            return False
                        elif op == "$nin" and (not isinstance(op_val, list) or meta_val in op_val):
                            return False
                        elif op == "$gt":
                            try:
                                if not (meta_val > op_val): return False
                            except:
                                return False
                        elif op == "$gte":
                            try:
                                if not (meta_val >= op_val): return False
                            except:
                                return False
                        elif op == "$lt":
                            try:
                                if not (meta_val < op_val): return False
                            except:
                                return False
                        elif op == "$lte":
                            try:
                                if not (meta_val <= op_val): return False
                            except:
                                return False
                else:
                    if meta_val != value:
                        return False
        return True

    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        where: Dict = None,
        resolve_parents: bool = True
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            where: Optional ChromaDB metadata filter clause
            resolve_parents: If True, resolves child chunks to parent chunks and deduplicates
            
        Returns:
            List of (text, metadata, distance) tuples
        """
        # Create query embedding
        query_embedding = self.create_embeddings([query])[0]
        
        # Search in ChromaDB
        # If resolving parents, retrieve more candidates first to account for deduplication
        query_k = max(k * 3, 20) if resolve_parents else k
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                formatted_results.append((text, metadata, distance))
        
        if resolve_parents:
            return self._resolve_and_deduplicate_parents(formatted_results, k)
            
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
        rrf_k: int = 60,
        where: Dict = None,
        semantic_query: str = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Perform Hybrid Search combining Semantic (ChromaDB) and Lexical (BM25) search,
        merging results using Reciprocal Rank Fusion (RRF).
        """
        # Step 1: Run semantic search (get a larger set of raw candidates)
        # We retrieve raw child chunks without resolving parents yet
        sim_query = semantic_query if semantic_query is not None else query
        semantic_candidates = self.similarity_search(sim_query, k=30, where=where, resolve_parents=False)
        
        # Step 2: Run lexical search (get a larger set of candidates)
        lexical_results = self.bm25.search(query, top_n=30)
        
        # Filter lexical results by metadata conditions if provided
        if where:
            filtered_lexical = []
            for doc_idx, score in lexical_results:
                meta = self.indexed_metadatas[doc_idx]
                if self._match_metadata_filter(meta, where):
                    filtered_lexical.append((doc_idx, score))
            lexical_results = filtered_lexical
        
        # Step 3: Perform Reciprocal Rank Fusion
        # A unique identifier is required for each chunk. (source, chunk_id) is perfect.
        def get_chunk_key(metadata: Dict) -> Tuple[str, int]:
            cid = metadata.get('chunk_id')
            if cid is None:
                cid = metadata.get('child_id', -1)
            return (metadata.get('source', 'Unknown'), cid)
            
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
        
        # Step 5: Retrieve a larger set of merged child candidates so we have enough after deduplication
        candidate_k = max(k * 3, 20)
        top_keys = sorted_keys[:candidate_k]
        merged_candidates = [chunk_lookup[key] for key in top_keys]
        
        # Step 6: Resolve child chunks to parents and deduplicate
        resolved_results = self._resolve_and_deduplicate_parents(merged_candidates, k)
        
        return resolved_results
    
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
        
        # Delete parent lookup store if it exists
        if os.path.exists(self.parent_store_path):
            try:
                os.remove(self.parent_store_path)
                print("[SUCCESS] Parent store cleared")
            except Exception as e:
                print(f"[WARNING] Could not clear parent store file: {str(e)}")
        
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
