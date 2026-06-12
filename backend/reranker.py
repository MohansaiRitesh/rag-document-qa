"""
Re-ranker Module using Sentence Transformers Cross-Encoder

This module performs the Stage 2 Re-ranking of retrieved candidates
using a Cross-Encoder model. It takes (query, document) pairs as input
and outputs fine-grained relevance scores mapped to the [0.0, 1.0] range.
"""

import math
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Manages loading and inference for the Cross-Encoder re-ranking model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Cross-Encoder model.
        
        Args:
            model_name: HuggingFace model name for the Cross-Encoder
        """
        print(f"[INFO] Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[INFO] Re-ranker model loaded successfully")

    def rerank(
        self, 
        query: str, 
        chunks: List[Tuple[str, Dict, float]], 
        top_k: int = 4
    ) -> List[Tuple[str, Dict, float]]:
        """
        Re-rank candidate chunks against the user query.
        
        Args:
            query: The user's query string
            chunks: A list of (text, metadata, original_distance) tuples from Stage 1
            top_k: The final number of top chunks to return
            
        Returns:
            A list of (text, metadata, pseudo_distance) sorted by relevance descending.
        """
        if not chunks:
            return []
            
        # Prepare inputs: [Query, Document] pairs
        pairs = [[query, text] for text, metadata, _ in chunks]
        
        # Run inference (predict similarity logits)
        print(f"[INFO] Re-ranking {len(chunks)} candidates using Cross-Encoder...")
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        scored_chunks = []
        for idx, score in enumerate(scores):
            text, metadata, _ = chunks[idx]
            
            # Map raw logit score to [0, 1] range using sigmoid function
            # sigmoid(x) = 1 / (1 + exp(-x))
            try:
                sigmoid_score = 1.0 / (1.0 + math.exp(-float(score)))
            except OverflowError:
                sigmoid_score = 0.0 if score < 0 else 1.0
                
            # Convert relevance score back to a pseudo-distance: distance = 1.0 - similarity
            # This aligns with the cosine distance expectation in llm_handler.py
            pseudo_distance = round(1.0 - sigmoid_score, 3)
            
            scored_chunks.append((text, metadata, pseudo_distance))
            
        # Sort chunks by distance ascending (which is similarity descending)
        scored_chunks.sort(key=lambda x: x[2])
        
        # Take the top K results
        reranked_results = scored_chunks[:top_k]
        print(f"[INFO] Re-ranking complete. Selected top {len(reranked_results)} chunks.")
        
        return reranked_results
