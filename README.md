# 📚 Grounded RAG Document Q&A System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-green.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.32-red.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Groq_Llama_3.3-70B-orange.svg?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/Multimodal_RAG-VLM-purple.svg?style=for-the-badge" alt="Multimodal" />
  <img src="https://img.shields.io/badge/Agentic_CRAG-Self--Correcting-teal.svg?style=for-the-badge" alt="Agentic" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" />
</p>

A production-grade, **Multimodal + Agentic Retrieval-Augmented Generation (RAG)** pipeline for grounded, verifiable question-answering over private documents. Upload PDFs, DOCX, or TXT files and receive cited, context-restricted answers powered by dense-lexical hybrid search, cross-encoder re-ranking, VLM image understanding, and an adaptive self-correcting agent.

---

## 🌟 Key Capabilities

Most RAG repositories demonstrate a basic split-embed-query loop. This system implements **production-tier methodologies** spanning classical retrieval, multimodal understanding, and autonomous agentic reasoning:

| Dimension | Basic RAG | This Production-Grade Pipeline |
|---|---|---|
| **Document Support** | Text only | **Multimodal**: PDF text + embedded images via VLM captioning (Llama 4 Scout) |
| **Chunking** | Fixed-size only | **3 Strategies**: Recursive, Semantic (z-score distance spikes), Hierarchical (Parent-Child) |
| **Retrieval** | Single vector search | **Hybrid Search**: Dense Semantic (MiniLM) + Lexical (Custom BM25) fused via RRF |
| **Re-ranking** | First-pass results | **2-Stage**: Stage 1 candidate pool (15) → Stage 2 Cross-Encoder reranking (4) |
| **LLM Focus** | Raw ordered contexts | **Lost-in-the-Middle (LitM)**: Distributes chunks to prompt boundaries |
| **Recall Boost** | Query vector only | **HyDE**: Synthesizes a hypothetical answer to bridge semantic search gap |
| **Response Latency** | Blocking JSON | **Real-Time SSE Streaming**: Token-by-token at ~750 tok/s via Groq LPU |
| **Conversations** | Single-turn Q&A | **Multi-Turn**: Self-contained query condensation using history |
| **Agent Mode** | Linear pipeline | **Agentic CRAG**: Intent routing → retrieval → relevance grading → self-correction → web fallback |
| **Grounding** | Unverified answers | **Self-Reflection**: Post-generation grounding check against source context |
| **Web Fallback** | No external search | **Zero-API DuckDuckGo Scraper**: Live web search when local knowledge is insufficient |
| **UI Mode** | Blocking renders | **Lag-Free UI**: Session-state caching + SSE streaming with thought step visualization |

---

## 🏗️ System Architecture & Data Flow

```
                            ┌──────────────────────────────────┐
                            │        Streamlit Frontend        │  ← Premium dark-space UI
                            │        (frontend/app.py)         │    SSE stream + thought panel
                            └──────────────┬───────────────────┘
                                           │  HTTP REST / SSE (/query-stream)
                                           ▼
                            ┌──────────────────────────────────┐
                            │         FastAPI Backend          │  ← CORS, Pydantic validation
                            │        (backend/main.py)         │    Static image file serving
                            └──────────────┬───────────────────┘
                                           │
                       ┌───────────────────┼────────────────────────┐
                       ▼                   ▼                        ▼
             ┌──────────────┐   ┌──────────────────┐    ┌──────────────────────┐
             │  RAG Engine  │   │  Agentic Engine  │    │  Document Processor  │
             │rag_engine.py │   │agentic_engine.py │    │document_processor.py │
             └──────┬───────┘   └────────┬─────────┘    └──────────┬───────────┘
                    │                    │                          │
         ┌──────────┴──────┐    ┌────────┴──────┐         ┌────────┴──────────┐
         ▼                 ▼    ▼               ▼         ▼                   ▼
  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐ ┌─────────────┐ ┌──────────────┐
  │ VectorStore │  │ LLMHandler │  │ Intent Classifier│ │ PDF/DOCX/   │ │ VLM Image    │
  │ ChromaDB    │  │ Groq API   │  │ Relevance Grader │ │ TXT Loader  │ │ Captioning   │
  │ BM25 Index  │  │ HyDE / LitM│  │ Query Rewriter   │ │             │ │ (Llama 4     │
  │ Parent Store│  │ SSE Stream │  │ Web Search       │ │ Recursive / │ │  Scout VLM)  │
  └─────────────┘  └────────────┘  │ Grounding Check  │ │ Semantic /  │ └──────────────┘
                                   └──────────────────┘ │ Hierarchical│
                                                         └─────────────┘
```

---

## ⚙️ Two Query Modes

### 🔵 Standard RAG Mode (`use_agentic_rag: false`)
A high-performance, deterministic pipeline:
```
Query → [Condense] → [HyDE] → Hybrid Search (BM25+Semantic) → Cross-Encoder Rerank → [LitM] → LLM Stream
```

### 🟣 Agentic CRAG Mode (`use_agentic_rag: true`)
A self-correcting autonomous loop with real-time thought streaming to the UI:
```
Query
  ↓ Intent Router (chitchat / document_qa / web_search)
  ↓ Vector Search + Relevance Grader (HIGH / MEDIUM / LOW)
  ↓ [LOW] → Query Rewriter → Retry (up to 2x)
  ↓ [Still LOW] → DuckDuckGo Web Fallback
  ↓ Generate Response → Grounding Verifier (GROUNDED / NOT_GROUNDED)
  ↓ Stream tokens + thought steps to UI
```

---

## 📄 Multimodal PDF Support

When a PDF is uploaded, the system **automatically extracts embedded images** (charts, graphs, tables, diagrams) using **PyMuPDF (fitz)**. Each image is:
1. Saved locally to `data/extracted_images/`
2. Sent to **Groq's Llama 4 Scout VLM** via multimodal API
3. Captioned with a rich, searchable text description
4. Indexed as a vector chunk alongside text content

This means your image-heavy PDFs — reports, presentations, technical documents — are fully searchable and citeable.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (Get a free key at [console.groq.com](https://console.groq.com))

### 1. Installation
```bash
git clone https://github.com/MohansaiRitesh/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file inside the `backend/` directory:
```bash
# File: backend/.env
GROQ_API_KEY=gsk_your_actual_key_goes_here
```

### 3. Start the Backend API Server
```bash
cd backend
python main.py
```
- API server: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`

### 4. Start the Frontend UI
```bash
cd frontend
streamlit run app.py
```
- Streamlit UI: `http://localhost:8501`

---

## ⚙️ Configuration Reference

All settings live in `backend/config.py` and can be overridden via `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (required) | API key for Groq LLM and VLM access |
| `llm_model` | `llama-3.3-70b-versatile` | Primary LLM for answers, condensation, HyDE |
| `embedding_model` | `all-MiniLM-L6-v2` | Dense sentence-transformer (384-dims) |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | Cross-Encoder for Stage 2 re-ranking |
| `chunking_strategy` | `recursive` | `"recursive"`, `"semantic"`, or `"hierarchical"` |
| `chunk_size` | `1000` | Characters per recursive chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `parent_chunk_size` | `1500` | Parent segment size (hierarchical strategy) |
| `child_chunk_size` | `300` | Child segment size (hierarchical strategy) |
| `use_hyde` | `True` | Enable HyDE hypothetical document expansion |
| `use_litm_packing` | `True` | Enable Lost-in-the-Middle context re-ordering |
| `top_k_results` | `4` | Final chunks fed to the LLM |
| `rerank_top_n` | `15` | Candidate pool size before cross-encoder reranking |
| `extracted_images_dir` | `data/extracted_images` | Storage for VLM-captioned PDF images |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend component status (LLM, ChromaDB) |
| `GET` | `/stats` | Total chunk count in vector database |
| `GET` | `/documents` | List uploaded filenames and metadata |
| `GET` | `/metadata-values` | Unique sources and file types for filter dropdowns |
| `POST` | `/upload` | Upload PDF, DOCX, or TXT (Multipart Form Data) |
| `POST` | `/query` | Blocking query with history and filters |
| `POST` | `/query-stream` | Streaming query — Standard RAG or Agentic CRAG via `use_agentic_rag` flag |
| `DELETE` | `/clear` | Purge all vectors, parent store, and uploaded files |

**Switching between modes** — set the `use_agentic_rag` boolean in the `/query-stream` request body:
```json
{
  "question": "What are the leave policies?",
  "use_agentic_rag": true
}
```

---

## 📁 Project Structure

```
RAG Q&A System/
│
├── backend/
│   ├── main.py                 # FastAPI application, all HTTP endpoints, static file serving
│   ├── rag_engine.py           # Standard RAG orchestrator (upload + query + stream)
│   ├── agentic_engine.py       # Agentic CRAG state machine (intent → grade → rewrite → web → verify)
│   ├── agent_state.py          # AgentState dataclass tracking the full agentic loop
│   ├── document_processor.py   # Document loading, all 3 chunking strategies, VLM image captioning
│   ├── vector_store.py         # ChromaDB, BM25, Hybrid RRF search, parent-child resolution
│   ├── bm25.py                 # Custom from-scratch BM25 implementation (no external library)
│   ├── llm_handler.py          # Groq client, HyDE, LitM, condense, streaming response
│   ├── reranker.py             # MS-MARCO Cross-Encoder scoring + sigmoid normalization
│   ├── config.py               # Pydantic Settings with .env loader
│   └── data/
│       ├── uploads/            # Temporary parsed file storage
│       ├── chromadb/           # Persistent ChromaDB vector store
│       ├── extracted_images/   # VLM-captioned PDF image files (served as static assets)
│       └── parent_store.json   # Parent chunk lookup for hierarchical retrieval
│
├── frontend/
│   └── app.py                  # Premium Streamlit UI with SSE streaming + agentic thought panel
│
├── test_system.py              # System diagnostic: imports, env, directories, model, Groq API
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # ← You are here
```

---

## 🧪 Verification & Testing

Run the built-in system diagnostic to verify your installation end-to-end:
```bash
python test_system.py
```

This script checks:
- ✅ All Python packages are importable
- ✅ `backend/.env` exists and `GROQ_API_KEY` is set
- ✅ Required directories (`backend/`, `frontend/`, `backend/data/`) exist
- ✅ `all-MiniLM-L6-v2` embedding model loads and produces 384-dim vectors
- ✅ Groq API connection returns a valid response

---

## 🙏 Credits & Libraries

- **LLM & VLM**: [Groq API](https://groq.com) — LPU-accelerated Llama 3.3-70B (text) and Llama 4 Scout 17B (vision)
- **Embeddings & Reranking**: [Sentence-Transformers](https://www.sbert.net) (`all-MiniLM-L6-v2` / `ms-marco-MiniLM-L-6-v2`)
- **Vector Storage**: [ChromaDB](https://www.trychroma.com) — persistent local vector database with cosine distance
- **PDF Parsing**: [PyPDF2](https://pypdf2.readthedocs.io) for text extraction, [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) for image extraction
- **REST Framework**: [FastAPI](https://fastapi.tiangolo.com) — async ASGI with Pydantic validation
- **Web UI**: [Streamlit](https://streamlit.io) — Python-native UI with session state management
- **Web Search**: [DuckDuckGo HTML endpoint](https://html.duckduckgo.com) — zero API key web search fallback
