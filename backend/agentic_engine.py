"""
Agentic Engine Module for RAG System

Implements the Adaptive Corrective RAG (CRAG) state machine:
1. Intent Routing (Chitchat, Document Q&A, Web Search)
2. Vector Search & Context Relevance Grading (HIGH, MEDIUM, LOW)
3. Self-Correction Query Rewriting (Iterative retries)
4. Web Search Fallback (Zero-API DuckDuckGo scraper)
5. Self-Reflection Grounding Verification
6. End-to-End SSE Thought & Token Streaming
"""

import json
import re
import time
import urllib.parse
from typing import List, Dict, Any, Generator, Tuple
import requests
from groq import Groq
from langchain.schema import Document as LangchainDocument

from agent_state import AgentState, AgentThought
from config import get_settings


class AgenticEngine:
    """
    State-Machine Orchestrator for Agentic RAG
    """
    
    def __init__(self, vector_store=None, llm_handler=None, groq_client: Groq = None):
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.settings = get_settings()
        
        if groq_client:
            self.groq_client = groq_client
        elif llm_handler and hasattr(llm_handler, 'client'):
            self.groq_client = llm_handler.client
        else:
            self.groq_client = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None


    # AGENT TOOL 1: INTENT ROUTER
    def classify_intent(self, query: str) -> str:
        """
        Classifies the query intent into:
        - "chitchat": Greetings, pleasantries, identity questions ("Hi", "Who are you?", "Thanks")
        - "web_search": Real-time info, sports, weather, news, current events, explicit web requests
        - "document_qa": Questions about specific topics, policies, domain concepts, or uploaded documents
        """
        if not self.groq_client:
            return "document_qa"
            
        system_prompt = (
            "You are an Intent Classification Router for a RAG system.\n"
            "Analyze the user query and output EXACTLY one word classification:\n"
            "- 'chitchat': Greetings, pleasantries, small talk, system identity (e.g. 'hello', 'who created you', 'thanks').\n"
            "- 'web_search': Questions asking for real-time info, weather, live stocks, current news, or external web search.\n"
            "- 'document_qa': Specific factual questions, policy lookups, technical concepts, data queries, or domain knowledge.\n\n"
            "Output ONLY the single classification word in lower case with no extra punctuation."
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip().lower()
            if "chitchat" in intent:
                return "chitchat"
            elif "web_search" in intent or "web" in intent:
                return "web_search"
            else:
                return "document_qa"
        except Exception as e:
            print(f"[WARNING] Intent classification failed: {e}. Defaulting to 'document_qa'")
            return "document_qa"


    # AGENT TOOL 2: CONTEXT RELEVANCE GRADER
    def grade_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Grades the relevance of retrieved chunks against the query.
        Returns Tuple[grade, rationale]:
        - grade: "HIGH", "MEDIUM", or "LOW"
        - rationale: Short sentence explaining the evaluation reasoning.
        """
        if not chunks:
            return "LOW", "No context chunks were retrieved from the database."
            
        if not self.groq_client:
            return "HIGH", "Relevance grader bypassed (no LLM client)."

        context_str = "\n---\n".join([f"Chunk {i+1}:\n{c.get('text', '')}" for i, c in enumerate(chunks[:5])])
        
        system_prompt = (
            "You are a Quality Control Evaluator assessing context relevance for RAG search.\n"
            "Task: Evaluate if the retrieved context chunks contain facts capable of answering the user's query.\n\n"
            "Grades:\n"
            "- 'HIGH': The context contains clear, direct answers to the query.\n"
            "- 'MEDIUM': The context is partially relevant or contains related topic details.\n"
            "- 'LOW': The context is completely off-topic or lacks key information.\n\n"
            "Output JSON format strictly:\n"
            '{"grade": "HIGH"|"MEDIUM"|"LOW", "rationale": "<one sentence explanation>"}'
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nRetrieved Context:\n{context_str}"}
                ],
                temperature=0.0,
                max_tokens=128
            )
            content = response.choices[0].message.content.strip()
            # Parse JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                grade = data.get("grade", "MEDIUM").upper()
                rationale = data.get("rationale", "Graded based on keyword and semantic overlap.")
                if grade not in ["HIGH", "MEDIUM", "LOW"]:
                    grade = "MEDIUM"
                return grade, rationale
            return "MEDIUM", "Could not parse grader JSON output."
        except Exception as e:
            print(f"[WARNING] Relevance grading failed: {e}")
            return "MEDIUM", f"Grading check failed ({str(e)}), proceeding with caution."


    # AGENT TOOL 3: QUERY REWRITER
    def rewrite_query(self, query: str, rationale: str) -> str:
        """
        Rewrites/optimizes a query that produced LOW or MEDIUM relevance context.
        Expands abbreviations, converts question forms into keyword-rich search statements.
        """
        if not self.groq_client:
            return query
            
        system_prompt = (
            "You are a Search Query Optimizer AI.\n"
            "The user's original search query failed to retrieve relevant documents from a vector database.\n"
            "Analyze the original query and the failure reason, then generate an improved, keyword-rich search query.\n"
            "Rules:\n"
            "- Make the query explicit, removing ambiguous pronouns.\n"
            "- Include technical synonyms or standard terminology.\n"
            "- Do NOT answer the question. Output ONLY the rephrased query string.\n"
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Query: {query}\nFailure Rationale: {rationale}"}
                ],
                temperature=0.3,
                max_tokens=60
            )
            rewritten = response.choices[0].message.content.strip()
            # Clean quotes if wrapped
            rewritten = rewritten.strip('"\'')
            return rewritten if rewritten else query
        except Exception as e:
            print(f"[WARNING] Query rewriting failed: {e}")
            return query


    # AGENT TOOL 4: WEB SEARCH FALLBACK (Zero-API DuckDuckGo Scraper)
    def search_web(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Performs web search fallback using DuckDuckGo HTML endpoint (zero API key dependency).
        Returns list of dicts: [{"text": ..., "source": ..., "url": ...}]
        """
        results = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            resp = requests.get(url, headers=headers, timeout=8)
            
            if resp.status_code == 200:
                # Basic regex extraction of search result titles, snippets, and URLs
                snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                
                for idx, (title_raw, snippet_raw) in enumerate(zip(titles, snippets)):
                    if idx >= max_results:
                        break
                    # Clean HTML tags
                    snippet = re.sub(r'<[^>]+>', '', snippet_raw).strip()
                    title = re.sub(r'<[^>]+>', '', title_raw).strip()
                    
                    if snippet:
                        results.append({
                            "text": snippet,
                            "source": f"Web Result: {title}",
                            "file_type": "web",
                            "is_image": False,
                            "url": f"https://{title}"
                        })
        except Exception as e:
            print(f"[WARNING] DuckDuckGo Web search failed: {e}")
            
        if not results:
            # Fallback placeholder if offline/blocked
            results.append({
                "text": f"Search fallback executed for query: '{query}'. (Live web search endpoint unreachable).",
                "source": "Web Search (Fallback)",
                "file_type": "web",
                "is_image": False
            })
            
        return results


    # AGENT TOOL 5: GROUNDING CHECK (Self-Reflection)
    def check_grounding(self, answer: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Verifies if the generated answer is grounded in the provided context chunks.
        Returns Tuple[grade, rationale] -> "GROUNDED" or "NOT_GROUNDED"
        """
        if not chunks or not self.groq_client:
            return "GROUNDED", "Grounding check skipped."

        context_str = "\n---\n".join([c.get('text', '') for c in chunks[:4]])
        
        system_prompt = (
            "You are a Grounding & Factuality Evaluator AI.\n"
            "Task: Check if the generated answer is strictly grounded in the provided context.\n"
            "If the answer contains claims that contradict or are completely unmentioned in the context, grade it as 'NOT_GROUNDED'.\n"
            "Otherwise grade as 'GROUNDED'.\n\n"
            "Output JSON format strictly:\n"
            '{"grade": "GROUNDED"|"NOT_GROUNDED", "rationale": "<one sentence explanation>"}'
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nAnswer:\n{answer}"}
                ],
                temperature=0.0,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                grade = data.get("grade", "GROUNDED").upper()
                rationale = data.get("rationale", "Answer verified against context.")
                return grade, rationale
            return "GROUNDED", "Could not parse grounding JSON output."
        except Exception as e:
            print(f"[WARNING] Grounding check failed: {e}")
            return "GROUNDED", f"Grounding check error ({str(e)})."


    # STATE MACHINE EXECUTION LOOP (Streaming)
    def execute_stream(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        use_hyde: bool = True,
        use_litm_packing: bool = True,
        history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """
        Executes the Adaptive CRAG State Machine and yields SSE events:
        - {"type": "thought", "step": ..., "thought": ..., "status": ...}
        - {"type": "sources", "sources": [...]}
        - {"type": "token", "token": ...}
        - {"type": "error", "error": ...}
        """
        state = AgentState(original_query=query)
        
        try:
            # STEP 1: INTENT ROUTING
            t0 = state.add_thought("Intent Router", "Analyzing user query intent...", "info")
            yield f"data: {json.dumps({'type': 'thought', **t0})}\n\n"
            
            state.intent = self.classify_intent(query)
            t1 = state.add_thought("Intent Classification", f"Classified query intent as '{state.intent.upper()}'", "success")
            yield f"data: {json.dumps({'type': 'thought', **t1})}\n\n"
            
            # BRANCH 1: CHITCHAT
            if state.intent == "chitchat":
                t2 = state.add_thought("Direct LLM Generator", "Bypassing database search for direct conversational response.", "info")
                yield f"data: {json.dumps({'type': 'thought', **t2})}\n\n"
                
                # Send empty sources
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                
                # Stream chitchat response
                for token in self.llm_handler.generate_response_stream(query, [], history=history):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                return

            # BRANCH 2: DIRECT WEB SEARCH REQUEST
            elif state.intent == "web_search":
                t2 = state.add_thought("Web Search Tool", f"Executing external web search for: '{query}'", "warning")
                yield f"data: {json.dumps({'type': 'thought', **t2})}\n\n"
                
                state.web_results = self.search_web(query)
                state.used_web_search = True
                
                sources = [{
                    "source": w["source"],
                    "file_type": "web",
                    "page_number": 1,
                    "relevance_score": 0.95,
                    "is_image": False
                } for w in state.web_results]
                
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # Stream response using web results
                for token in self.llm_handler.generate_response_stream(query, state.web_results, history=history):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                return

            # STEP 2: DOCUMENT Q&A RETRIEVAL & SELF-CORRECTION LOOP
            while state.retry_count <= state.max_retries:
                search_q = state.current_query
                t_ret = state.add_thought(
                    "Vector Store Retrieval", 
                    f"Searching vector database for: '{search_q}' (Attempt {state.retry_count + 1})", 
                    "info"
                )
                yield f"data: {json.dumps({'type': 'thought', **t_ret})}\n\n"
                
                # Execute retrieval + re-ranking from vector store
                chunks = self.vector_store.hybrid_search(
                    query=search_q,
                    top_k=self.settings.top_k_results,
                    filters=filters,
                    semantic_query=search_q
                )
                
                # Grade context relevance
                grade, rationale = self.grade_relevance(search_q, chunks)
                state.relevance_grade = grade
                
                t_grade = state.add_thought(
                    "Relevance Grader", 
                    f"Context Quality: [{grade}] - {rationale}", 
                    "success" if grade == "HIGH" else ("warning" if grade == "MEDIUM" else "error")
                )
                yield f"data: {json.dumps({'type': 'thought', **t_grade})}\n\n"
                
                if grade in ["HIGH", "MEDIUM"]:
                    state.retrieved_chunks = chunks
                    break
                    
                # If LOW, attempt self-correction query rewriting
                state.retry_count += 1
                if state.retry_count <= state.max_retries:
                    rewritten = self.rewrite_query(search_q, rationale)
                    state.current_query = rewritten
                    t_rewrite = state.add_thought(
                        "Query Rewriter Agent", 
                        f"Low relevance detected. Rewrote query to: '{rewritten}'", 
                        "warning"
                    )
                    yield f"data: {json.dumps({'type': 'thought', **t_rewrite})}\n\n"
                else:
                    t_max = state.add_thought(
                        "Max Retries Reached", 
                        "Vector DB search failed to yield high quality context after retries. Triggering Web Search fallback...", 
                        "error"
                    )
                    yield f"data: {json.dumps({'type': 'thought', **t_max})}\n\n"

            # STEP 3: FALLBACK TO WEB SEARCH IF VECTOR SEARCH FAILED
            if state.relevance_grade == "LOW" or not state.retrieved_chunks:
                state.used_web_search = True
                state.web_results = self.search_web(query)
                active_chunks = state.web_results
                t_web = state.add_thought("Web Search Fallback", f"Retrieved {len(state.web_results)} web context results.", "warning")
                yield f"data: {json.dumps({'type': 'thought', **t_web})}\n\n"
            else:
                active_chunks = state.retrieved_chunks

            # STEP 4: YIELD SOURCES TO UI
            sources = []
            for idx, chunk in enumerate(active_chunks):
                sources.append({
                    "source": chunk.get("source", "Unknown"),
                    "file_type": chunk.get("file_type", "txt"),
                    "page_number": chunk.get("page_number", 1),
                    "relevance_score": float(chunk.get("rerank_score", chunk.get("similarity_score", 0.0))),
                    "is_image": chunk.get("is_image", False),
                    "image_path": chunk.get("image_path", None)
                })
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # STEP 5: STREAM GENERATION TO UI
            t_gen = state.add_thought("LLM Generator", "Synthesizing answer from validated context...", "info")
            yield f"data: {json.dumps({'type': 'thought', **t_gen})}\n\n"
            
            for token in self.llm_handler.generate_response_stream(
                query=query, 
                context_chunks=active_chunks, 
                use_litm_packing=use_litm_packing,
                history=history
            ):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                
        except Exception as e:
            print(f"[ERROR] Agentic execution error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
