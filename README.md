# 📚 Grounded RAG Document Q&A System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-green.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.32-red.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Groq_Llama_3.3-70B-orange.svg?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" />
</p>

A production-grade, high-performance **Retrieval-Augmented Generation (RAG)** pipeline designed for grounded, verifiable question-answering over private documents. Connect your files (PDF, DOCX, TXT) and receive context-restricted answers with precise citations, powered by dense-lexical hybrid search, cross-encoder re-ranking, and low-latency token streaming.

---

## 🌟 Key Capabilities

Most RAG repositories demonstrate a basic split-embed-query loop. This system implements production-tier methodologies to handle real-world challenges like referential ambiguity, embedding dilution, and context attention loss:

| Dimension | Basic RAG | This Production-Grade Pipeline |
|---|---|---|
| **Chunking** | Fixed-size splits only | **Multi-Strategy**: Recursive, Semantic (z-score spikes), or Hierarchical (Parent-Child) |
| **Retrieval** | Single vector search | **Hybrid Search**: Dense Semantic (MiniLM) + Lexical (Custom BM25) fused via RRF |
| **Re-ranking** | First-pass results | **2-Stage Retrieval**: Stage 1 candidate pool (15) $\rightarrow$ Stage 2 Cross-Encoder reranking (4) |
| **LLM Focus** | Raw ordered contexts | **Lost-in-the-Middle (LitM)**: Alternates chunk relevance to prompt borders |
| **Recall Boost** | Query vector similarity | **HyDE**: Generates hypothetical answer paragraph to search, bridging semantic gaps |
| **Response Latency** | Blocking full JSON | **Real-Time Token Streaming**: Server-Sent Events (SSE) yielding tokens at ~750 tok/s |
| **Conversations** | Single-turn Q&A | **Multi-Turn Chat**: Self-contained query condensation using conversation history |
| **Verification** | Unverifiable answers | **Metadata Filters & Grounded Citations**: Color-coded relevance scores + chunk source previews |
| **UI Responsiveness** | Sluggish blocking calls | **Lag-Free UI**: Caches stats, metadata, and backend health checks in Streamlit session state |

---

## 🏗️ System Architecture & Data Flow

```
                               ┌──────────────────────────────┐
                               │       Streamlit Frontend     │  ← Optimized state cache &
                               │        (frontend/app.py)     │    Server-Sent Events reader
                               └──────────────┬───────────────┘
                                              │  HTTP REST / query-stream (SSE)
                                              ▼
                               ┌──────────────────────────────┐
                               │        FastAPI Backend       │  ← Schema validation & Cors
                               │       (backend/main.py)      │    lifespan controllers
                               └──────────────┬───────────────┘
                                              │  Internal calls
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       RAG ENGINE                                       │
 │                                 (backend/rag_engine.py)                                │
 └──────┬──────────────────────┬───────────────────────┬──────────────────────────┬───────┘
        │                      │                       │                          │
        ▼                      ▼                       ▼                          ▼
┌──────────────┐       ┌───────────────┐       ┌───────────────┐          ┌──────────────┐
│  DocProcessor│       │  Vector Store │       │  LLM Handler  │          │   Reranker   │
│ ──────────── │       │ ───────────── │       │ ───────────── │          │ ──────────── │
│ Load file    │       │ ChromaDB      │       │ Groq API      │          │ Cross-Encoder│
│ (pdf/docx/txt)       │ BM25 index    │       │ (Llama 3.3)   │          │ (ms-marco)   │
│              │       │ Parent lookup │       │ Condensation  │          │ Sigmoid score│
│ Split texts: │       │ (JSON database)       │ HyDE gen      │          │ normalizer   │
│ - Recursive  │       │               │       │ LitM packing  │          │              │
│ - Semantic   │       │               │       │ SSE stream    │          │              │
│ - ParentChild│       │               │       │               │          │              │
└──────────────┘       └───────────────┘       └───────────────┘          └──────────────┘
```

### 1. Document Indexing Pipeline (Upload)
1. **File Loading**: Raw text is parsed from PDFs (with `[Page N]` citation markers), Word documents, or text files.
2. **Chunking Tiers**:
   - **Recursive**: Splits text by separator priority `["\n\n", "\n", ". ", " ", ""]`.
   - **Semantic**: Segments text dynamically at sentence transitions where embedding distances spike.
   - **Hierarchical**: Splits text into large parent chunks (1500 chars) and smaller child chunks (300 chars).
3. **Storage Strategy**: Standard and semantic chunks are saved directly in ChromaDB. Hierarchical chunks save child vectors in ChromaDB, while registering their parent structures inside `data/parent_store.json`.
4. **Lexical Synching**: Rebuilds the custom in-memory BM25 index over the entire corpus.

### 2. Retrieval & Generation Pipeline (Query)
1. **Condensation**: Rewrites follow-up questions into standalone queries using chat history.
2. **HyDE Expansion** *(Optional)*: Synthesizes a hypothetical answer via Llama 3.3 to search vector space, bridging the semantic gap between questions and documents.
3. **Stage 1 Search**: Retrieves top 20 candidates from BM25 and ChromaDB. Merges ranks using Reciprocal Rank Fusion (RRF, $k=60$). If hierarchical mode is active, child chunks are resolved to their parents via `parent_store.json` and deduplicated.
4. **Stage 2 Reranking**: Re-scores the top 15 candidates using a ms-marco cross-encoder. Scores are normalized to $[0, 1]$ via Sigmoid.
5. **LitM Packing** *(Optional)*: Re-orders the top 4 chunks (alternating high relevance to prompt edges) to bypass the transformer attention valley.
6. **Token streaming**: Feeds context to Groq and streams text tokens back to the UI at ultra-low latency.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (Get a free key at [console.groq.com](https://console.groq.com))

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/MohansaiRitesh/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the `backend/` directory:
```bash
# File: backend/.env
GROQ_API_KEY=gsk_your_actual_key_goes_here
```

### 3. Start the Backend API Server
```bash
cd backend
python main.py
```
- API server will start at: `http://localhost:8000`
- Interactive Swagger Documentation: `http://localhost:8000/docs`

### 4. Start the Frontend UI (In a new terminal)
```bash
cd frontend
streamlit run app.py
```
- Streamlit application will open automatically at: `http://localhost:8501`

---

## ⚙️ Configuration Reference

All settings can be configured in `backend/config.py` or overridden via environment variables in `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (required) | API key for Llama 3.3 Groq access |
| `llm_model` | `llama-3.3-70b-versatile` | Llama 3.3 70B model identifier |
| `embedding_model` | `all-MiniLM-L6-v2` | Dense sentence-transformer model (384-dims) |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | Cross-Encoder model for Stage 2 re-ranking |
| `chunking_strategy` | `recursive` | Default strategy: `"recursive"`, `"semantic"`, `"hierarchical"` |
| `chunk_size` | `1000` | Target character size for standard chunks |
| `chunk_overlap` | `200` | Overlapping characters between consecutive chunks |
| `parent_chunk_size` | `1500` | Parent segment size (Hierarchical strategy) |
| `child_chunk_size` | `300` | Child segment size (Hierarchical strategy) |
| `use_hyde` | `True` | Generate hypothetical answers before search |
| `use_litm_packing` | `True` | Pack context chunks to primacy/recency boundaries |
| `top_k_results` | `4` | Number of context chunks fed to the LLM |
| `rerank_top_n` | `15` | Size of Stage 1 candidate pool before reranking |

---

## 🧪 Verification & Testing

The repository contains a self-contained system diagnostic and verification script `test_system.py`. You can run this script to ensure all Python dependencies are correctly installed, directory paths are active, and API integrations (e.g., Groq client validation) are operational:

```bash
python test_system.py
```

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check backend components status (LLM connection, ChromaDB) |
| `GET` | `/stats` | Retrieve total chunk count indexed in database |
| `GET` | `/documents` | List uploaded filenames, sizes, and file types |
| `GET` | `/metadata-values` | Fetch list of unique sources and extensions in database |
| `POST` | `/upload` | Upload PDF, DOCX, or TXT file (Multipart Form Data) |
| `POST` | `/query` | Ask question with history and filters (JSON body) |
| `POST` | `/query-stream` | Ask question and stream back tokens and sources (SSE) |
| `DELETE` | `/clear` | Purge all document vectors, parent stores, and uploads |

---

## 📖 Project Structure

```
RAG Q&A System/
│
├── backend/
│   ├── main.py                 # FastAPI application, HTTP endpoints
│   ├── rag_engine.py           # Orchestrator (Upload and Query flows)
│   ├── document_processor.py   # Document loading & recursive/semantic/hierarchical splits
│   ├── vector_store.py         # ChromaDB, BM25, Hybrid RRF, Parent resolution
│   ├── bm25.py                 # Tokenizer and custom BM25 fit/search implementation
│   ├── llm_handler.py          # Groq client, prompt construction, HyDE, LitM re-ordering
│   ├── reranker.py             # MS-MARCO Cross-Encoder scoring & sigmoid normalizer
│   ├── config.py               # Pydantic Settings loaders
│   └── data/
│       ├── uploads/            # Temporary storage of parsed files
│       ├── chromadb/           # ChromaDB database files
│       └── parent_store.json   # Parent chunks metadata store (Hierarchical strategy)
│
├── frontend/
│   └── app.py                  # Optimized Streamlit UI (caching, SSE streams, glassmorphic layout)
│
├── test_system.py              # System configuration, connection, and diagnostic tests
├── requirements.txt            # System dependencies
└── README.md                   # ← Root README (You are here)
```

---

## 🙏 Credits & Libraries

- **LLM Engine**: [Groq API](https://groq.com) for LPU-accelerated Llama 3.3 inference.
- **Embeddings & Reranking**: [Sentence-Transformers](https://www.sbert.net) (`all-MiniLM-L6-v2` / `ms-marco-MiniLM-L-6-v2`).
- **Vector Storage**: [ChromaDB](https://www.trychroma.com) for persistent, localized semantic vectors.
- **REST Framework**: [FastAPI](https://fastapi.tiangolo.com) for async ASGI endpoints and Pydantic validation.
- **Web UI**: [Streamlit](https://streamlit.io) for rapid Python UI rendering.
