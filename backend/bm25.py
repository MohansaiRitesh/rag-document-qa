"""
BM25 Lexical Search Module

This module implements a custom, lightweight BM25 (Best Matching 25) indexer.
It ranks documents based on term frequency and document length normalization.
"""

import math
from typing import List, Dict, Tuple

class BM25:
    """
    Lightweight, custom BM25 indexer for text chunks.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 parameters.
        
        Args:
            k1: Term frequency saturation tuning parameter (default 1.5)
            b: Document length normalization tuning parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize raw text into cleaned, lowercase alphanumeric tokens.
        """
        tokens = []
        for word in text.lower().split():
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word:
                tokens.append(clean_word)
        return tokens

    def fit(self, texts: List[str]) -> None:
        """
        Build the BM25 index from a list of document strings.
        
        This calculates:
        1. Document lengths and average document length.
        2. Term frequencies for each document.
        3. Inverse Document Frequency (IDF) for all unique terms.
        """
        self.doc_count = len(texts)
        self.doc_lengths = []
        self.doc_freqs = []
        self.idf = {}
        
        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            return
            
        term_doc_occurrences: Dict[str, int] = {}
        total_len = 0
        
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            
            # Count terms in this document (Term Frequency)
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            
            # Count how many documents contain each term (for IDF calculation)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_occurrences[token] = term_doc_occurrences.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.doc_count
        
        # Calculate IDF for all unique terms
        # Formula: idf = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        for term, count in term_doc_occurrences.items():
            numerator = self.doc_count - count + 0.5
            denominator = count + 0.5
            self.idf[term] = math.log(1.0 + (numerator / denominator))
            
    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Search indexed documents and rank them using the BM25 formula.
        
        Args:
            query: User search query
            top_n: Number of results to return
            
        Returns:
            List of (doc_index, score) tuples sorted by score in descending order
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []
            
        scores = []
        for i in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lengths[i]
            freqs = self.doc_freqs[i]
            
            for token in query_tokens:
                if token not in freqs:
                    continue
                
                f = freqs[token]
                idf_val = self.idf.get(token, 0.0)
                
                # BM25 math formula:
                # score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (numerator / denominator)
                
            scores.append((i, score))
            
        # Sort by score descending and return results that have matching keywords (score > 0)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item in scores[:top_n] if item[1] > 0.0]
