"""
LLM Handler Module

This module manages interactions with the Language Model:
1. Initialize Groq API connection
2. Generate responses with context
3. Create citation-aware prompts

Key Concepts:
- Context Window: The amount of text the LLM can process
- System Prompt: Instructions for the LLM's behavior
- Temperature: Controls randomness (0 = deterministic, 1 = creative)
"""

from typing import List, Dict, Tuple
from groq import Groq
import json


class LLMHandler:
    """
    Handles all LLM operations using Groq API
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1
    ):
        """
        Initialize LLM handler
        
        Args:
            api_key: Groq API key
            model: Model name to use
            temperature: Response randomness (0-1)
        """
        if not api_key:
            raise ValueError("Groq API key is required")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        print(f"[INFO] LLM Handler initialized with model: {model}")
    
    def create_system_prompt(self) -> str:
        """
        Create system prompt for the LLM
        
        This defines how the LLM should behave
        """
        return """You are a helpful AI assistant that answers questions based on the provided context.

CRITICAL INSTRUCTIONS:
1. ONLY answer using information from the provided context
2. If the context doesn't contain the answer, say "I cannot answer this question based on the provided documents."
3. Always cite your sources by mentioning the document name
4. Be precise and factual - no speculation or assumptions
5. If you quote directly, use quotation marks and cite the source
6. Provide clear, well-structured answers

Remember: Your goal is to provide GROUNDED, CITED answers without hallucination."""
    
    def format_context(
        self, 
        retrieved_chunks: List[Tuple[str, Dict, float]]
    ) -> str:
        """
        Format retrieved chunks into context for the LLM
        
        Args:
            retrieved_chunks: List of (text, metadata, distance) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (text, metadata, distance) in enumerate(retrieved_chunks, 1):
            source = metadata.get('source', 'Unknown')
            chunk_id = metadata.get('chunk_id', 'N/A')
            
            # Format each chunk with metadata
            context_part = f"""
[Source {i}: {source}, Chunk {chunk_id}]
{text}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
        
    def condense_query(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        Condense a follow-up query and history into a standalone search query.
        """
        if not history:
            return query
            
        # Format chat history
        history_parts = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.get('content')}")
        history_text = "\n".join(history_parts)
        
        system_prompt = (
            "You are a helpful assistant that reformulates user follow-up questions "
            "and conversation history into a single, standalone search query. "
            "Do NOT answer the question, just return the search query and nothing else. "
            "Make it concise and suitable for a vector database search."
        )
        
        user_message = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query. Do NOT answer the question, just output the rephrased query and nothing else.

Conversation History:
{history_text}

Follow-up Question: {query}
Standalone Search Query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Highly deterministic
                max_tokens=256
            )
            condensed = response.choices[0].message.content.strip()
            print(f"[INFO] Condensed Query: '{query}' -> '{condensed}'")
            return condensed
        except Exception as e:
            print(f"[WARNING] Query condensation failed: {str(e)}. Using original query.")
            return query
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, Dict, float]],
        history: List[Dict[str, str]] = None
    ) -> Dict:
        """
        Generate response using retrieved context and conversation history
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved document chunks
            history: Optional conversation history
            
        Returns:
            Dictionary with answer and metadata
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "context_used": False
            }
        
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Create user message with context (sent as the final user message)
        user_message = f"""Context Information:
{context}

Question: {query}

Please provide a detailed answer based on the context above. Remember to cite your sources."""
        
        try:
            # Construct chat messages array
            messages = [{"role": "system", "content": self.create_system_prompt()}]
            
            # Append history if available
            if history:
                for msg in history:
                    role = msg.get("role")
                    if role in ["user", "assistant"]:
                        messages.append({"role": role, "content": msg.get("content")})
                        
            # Append the current query with context
            messages.append({"role": "user", "content": user_message})
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Extract sources
            sources = []
            for text, metadata, distance in retrieved_chunks:
                sources.append({
                    "source": metadata.get('source', 'Unknown'),
                    "chunk_id": metadata.get('chunk_id', 'N/A'),
                    "relevance_score": round(max(0.0, min(1.0, 1.0 - distance)), 3),  # Convert distance to similarity
                    "preview": text[:200] + "..." if len(text) > 200 else text
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "model": self.model
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test if the API connection works
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print("[SUCCESS] LLM connection test successful")
            return True
        except Exception as e:
            print(f"[ERROR] LLM connection test failed: {str(e)}")
            return False


# Example usage (for testing)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize handler
    handler = LLMHandler(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    # Test connection
    handler.test_connection()
    
    # Test with sample context
    sample_chunks = [
        (
            "Python is a high-level, interpreted programming language.",
            {"source": "python_intro.txt", "chunk_id": 0},
            0.2
        )
    ]
    
    result = handler.generate_response(
        "What is Python?",
        sample_chunks
    )
    
    print("\nAnswer:", result["answer"])
    print("\nSources:", json.dumps(result["sources"], indent=2))
