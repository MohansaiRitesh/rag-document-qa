# 📚 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for document question-answering with cited, grounded answers. Built with FastAPI, Streamlit, and Groq's lightning-fast LLM API.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Multi-format Support**: PDF, DOCX, TXT files
- **Intelligent Chunking**: Context-aware document splitting
- **Semantic Search**: Find relevant information using embeddings
- **Grounded Answers**: LLM responses based only on your documents
- **Source Citations**: Every answer includes source references
- **Fast & Free**: Uses Groq's lightning-fast LLM API (free tier available)
- **Persistent Storage**: ChromaDB vector database saves your documents
- **Modern UI**: Clean, professional Streamlit interface

## 🎥 Demo

![RAG System Demo](demo.gif)

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  Frontend (User Interface)
│   Frontend  │
└──────┬──────┘
       │ HTTP Requests
       ▼
┌─────────────┐
│   FastAPI   │  Backend (REST API)
│   Backend   │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌─────────────┐  ┌──────────────┐
│  RAG Engine │  │  Vector DB   │
│             │  │  (ChromaDB)  │
└─────────────┘  └──────────────┘
       │
       ▼
┌──────────────┐
│  Groq API    │
│  (Llama 3.3) │
└──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (free at https://console.groq.com)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API key**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

4. **Run the backend**
```bash
cd backend
python main.py
```

5. **Run the frontend** (in a new terminal)
```bash
cd frontend
streamlit run app.py
```

6. **Open your browser**
Navigate to `http://localhost:8501`

## 📖 Usage

### Upload Documents
1. Go to the "Upload Documents" tab
2. Select a PDF, DOCX, or TXT file
3. Click "Upload & Process"
4. Wait for processing (creates ~40-50 chunks)

### Ask Questions
1. Go to the "Ask Questions" tab
2. Type your question
3. Click "Get Answer"
4. View answer with source citations

### Example
```
Upload: "company_policy.pdf"
Question: "How many vacation days do employees get?"
Answer: "Employees receive 15 days of paid vacation per year. 
         (Source: company_policy.pdf, Relevance: 94%)"
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (REST API)
- **Frontend**: Streamlit (Web UI)
- **LLM**: Groq API (Llama 3.3 70B)
- **Embeddings**: Sentence Transformers (MiniLM-L6)
- **Vector DB**: ChromaDB (local storage)
- **Document Processing**: PyPDF2, python-docx, LangChain

## 📁 Project Structure

```
rag_project/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── rag_engine.py        # Main orchestrator
│   ├── document_processor.py # Load & chunk documents
│   ├── vector_store.py      # ChromaDB interface
│   ├── llm_handler.py       # Groq LLM integration
│   └── config.py            # Configuration
├── frontend/
│   └── app.py               # Streamlit UI
├── test_documents/          # Sample documents & tests
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Configuration

Edit `backend/config.py` to customize:

```python
# Chunking
chunk_size = 1000        # Characters per chunk
chunk_overlap = 200      # Overlap for context

# Retrieval
top_k_results = 4        # Chunks to retrieve

# Model
llm_model = "llama-3.3-70b-versatile"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

## 🧪 Testing

We provide comprehensive test materials:

```bash
# Test the system
cd test_documents
# Upload techvision_employee_handbook.txt
# Run questions from TEST_QUESTIONS.md
```

**Test Categories:**
- ✅ Easy: Basic fact retrieval (5 questions)
- 🟡 Medium: Multi-step reasoning (5 questions)
- 🔴 Hard: Complex scenarios (5 questions)
- 🚫 Negative: Hallucination prevention (3 questions)

## 📊 Performance

- **Embedding Creation**: ~100 chunks/second
- **Vector Search**: <100ms for 1000 chunks
- **LLM Response**: ~1-2 seconds (via Groq)
- **Total Query Time**: ~2-3 seconds

## 🔒 Security Notes

**Current Setup** (Development):
- No authentication
- Local storage only
- API key in `.env` file

**For Production**:
- Add API authentication
- Use environment secrets management
- Implement rate limiting
- Enable HTTPS
- Add input sanitization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** - Fast, free LLM API
- **ChromaDB** - Simple vector database
- **LangChain** - Document processing utilities
- **Sentence Transformers** - Embedding models
- **FastAPI** - Modern Python framework
- **Streamlit** - Rapid UI development

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/YOUR_USERNAME/rag-document-qa](https://github.com/YOUR_USERNAME/rag-document-qa)

## 🗺️ Roadmap

- [ ] Add conversation history
- [ ] Support more file formats (Excel, Markdown)
- [ ] Implement streaming responses
- [ ] Add user authentication
- [ ] Multi-language support
- [ ] Advanced filtering and search
- [ ] Export Q&A to PDF
- [ ] Docker deployment

---

**Built with ❤️ for learning RAG systems**
