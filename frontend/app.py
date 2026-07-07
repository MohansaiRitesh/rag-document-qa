"""
Streamlit Frontend for RAG System

This creates a premium, high-performance interactive web interface for:
1. Uploading documents with strategy selectors
2. Asking questions with multi-turn chat
3. Viewing cited, grounded answers with styled source cards
4. Advanced search filters & optimizations
"""

import streamlit as st
import requests
from typing import Dict, List, Generator
import json
from pathlib import Path


import os

# Backend API URL
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# HELPER FUNCTIONS & CACHED API CLIENTS
@st.cache_data(ttl=10)
def check_backend_health() -> bool:
    """Check if backend is running (cached to avoid lagginess on input)"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        return response.status_code == 200
    except:
        return False


def init_session_state():
    """Initialize session state variables if they do not exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = None
    if "metadata_values" not in st.session_state:
        st.session_state.metadata_values = None


def refresh_system_state(force=False):
    """Refresh system statistics and metadata filter values from backend"""
    # Fetch stats
    if force or st.session_state.stats is None:
        try:
            response = requests.get(f"{API_URL}/stats", timeout=2)
            if response.status_code == 200:
                st.session_state.stats = response.json()
            else:
                st.session_state.stats = {"success": False, "total_documents": 0}
        except Exception:
            st.session_state.stats = {"success": False, "total_documents": 0}

    # Fetch metadata values (sources and file types)
    if force or st.session_state.metadata_values is None:
        try:
            response = requests.get(f"{API_URL}/metadata-values", timeout=2)
            if response.status_code == 200:
                st.session_state.metadata_values = response.json()
            else:
                st.session_state.metadata_values = {"success": False, "sources": [], "file_types": []}
        except Exception:
            st.session_state.metadata_values = {"success": False, "sources": [], "file_types": []}


def upload_document(file, chunking_strategy: str = "recursive", alpha: float = 1.0) -> Dict:
    """Upload a document to the backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {
            "chunking_strategy": chunking_strategy,
            "semantic_threshold_alpha": alpha
        }
        response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=60)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def query_documents_stream(
    question: str, 
    history: List[Dict] = None, 
    filters: Dict = None, 
    use_hyde: bool = True, 
    use_litm_packing: bool = True
) -> Generator[Dict, None, None]:
    """Ask a question and stream response events from the backend"""
    try:
        response = requests.post(
            f"{API_URL}/query-stream",
            json={
                "question": question,
                "history": history or [],
                "filters": filters,
                "use_hyde": use_hyde,
                "use_litm_packing": use_litm_packing
            },
            stream=True,
            timeout=30
        )
        if response.status_code != 200:
            yield {"type": "error", "message": f"Server returned error code {response.status_code}: {response.text}"}
            return
            
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    event = json.loads(decoded_line)
                    yield event
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        yield {"type": "error", "message": str(e)}


def clear_database() -> Dict:
    """Clear all documents"""
    try:
        response = requests.delete(f"{API_URL}/clear", timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================
# CUSTOM STYLING (PREMIUM DARK MODE)
# ============================================
st.markdown("""
<style>
/* ===== GLOBAL VARIABLES & THEME ===== */
:root {
    --bg-primary: #06090e;
    --bg-secondary: #0c0f16;
    --bg-card: #121721;
    --border-color: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(99, 102, 241, 0.2);
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
    --accent-color: #6366f1;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.35);
    --glow: 0 0 20px rgba(99, 102, 241, 0.12);
}

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ===== BASE WRAPPERS ===== */
.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-main) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Remove Streamlit default header decoration */
header {
    background-color: rgba(0,0,0,0) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Padding adjustments */
.block-container {
    max-width: 900px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* ===== APP HEADER ===== */
.app-header {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2.5rem;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.03em !important;
}
.app-header p {
    font-size: 1.05rem !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
}

/* ===== SIDEBAR STYLING ===== */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
    width: 330px !important;
}
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-main) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.75rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: 1.25rem 0 !important;
    border-color: var(--border-color) !important;
}

/* Settings Controls */
.stCheckbox > label {
    color: var(--text-muted) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}
.stCheckbox > label:hover {
    color: var(--text-main) !important;
}
.stSelectbox > label, .stSlider > label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    margin-bottom: 6px !important;
}
div[data-baseweb="select"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}
div[data-baseweb="select"]:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}
div[data-baseweb="select"] > div {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

/* Custom Metrics in Sidebar */
.metric-container {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-lbl {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(0, 0, 0, 0.15) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    margin-bottom: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border: none !important;
    background-color: transparent !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-gradient) !important;
    color: #ffffff !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: var(--text-main) !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
}
.stTabs [data-baseweb="tab-highlight-bar"] {
    display: none !important; /* Hide default line */
}

/* ===== BUTTONS ===== */
div.stButton > button {
    background: var(--accent-gradient) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 10px 20px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Secondary Button (Destructive / Info) */
div.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-main) !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(239, 68, 68, 0.06) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    color: #f87171 !important;
}

/* ===== CHAT MESSAGES ===== */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    margin-bottom: 16px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="stChatMessage"]:hover {
    border-color: var(--border-accent) !important;
    box-shadow: var(--glow) !important;
}
div[data-testid="stChatMessage"][data-testid$="user"] {
    background-color: rgba(99, 102, 241, 0.02) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    border-radius: 14px !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    padding: 6px 12px !important;
    box-shadow: var(--shadow-md) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.15) !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--text-main) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* ===== SOURCE CITATION CARDS ===== */
.source-card {
    background-color: rgba(0, 0, 0, 0.12);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-bottom: 1rem;
    overflow: hidden;
    transition: all 0.2s ease;
}
.source-card:hover {
    border-color: rgba(99, 102, 241, 0.25);
    transform: translateX(2px);
}
.source-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    background-color: rgba(255, 255, 255, 0.01);
    border-bottom: 1px solid var(--border-color);
    flex-wrap: wrap;
    gap: 0.5rem;
}
.source-file-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.source-filename {
    font-weight: 600;
    color: var(--text-main);
    font-size: 0.875rem;
}
.source-chunk-badge {
    background-color: rgba(255, 255, 255, 0.05);
    color: var(--text-muted);
    font-size: 0.725rem;
    padding: 1px 7px;
    border-radius: 20px;
    border: 1px solid var(--border-color);
}
.source-relevance {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.relevance-label {
    font-size: 0.725rem;
    color: var(--text-muted);
}
.relevance-bar-container {
    width: 50px;
    height: 5px;
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
}
.relevance-bar {
    height: 100%;
    border-radius: 3px;
}
.relevance-text {
    font-size: 0.775rem;
    font-weight: 700;
}
.source-preview-container {
    padding: 0.85rem 1.1rem;
    background-color: rgba(0, 0, 0, 0.2);
    border-left: 3px solid #6366f1;
}
.source-preview {
    margin: 0;
    font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
    font-size: 0.825rem;
    color: #cbd5e1;
    white-space: pre-wrap;
    line-height: 1.5;
}

/* Expander Overrides */
.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.02) !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-color) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.streamlit-expanderContent {
    background-color: rgba(0, 0, 0, 0.1) !important;
    border-left: 1px solid var(--border-color) !important;
    border-right: 1px solid var(--border-color) !important;
    border-bottom: 1px solid var(--border-color) !important;
    border-bottom-left-radius: 10px !important;
    border-bottom-right-radius: 10px !important;
    padding: 1.25rem !important;
}

/* ===== WELCOME SCREEN ===== */
.welcome-container {
    padding: 3rem 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.04) 0%, rgba(168, 85, 247, 0.04) 100%);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    margin-top: 1rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
}
.welcome-hero {
    text-align: center;
    margin-bottom: 2.5rem;
}
.welcome-hero h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.02em !important;
}
.welcome-hero p {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 600px;
    margin: 0 auto;
}
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2.25rem;
}
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.25rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}
.welcome-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.25);
    box-shadow: var(--glow);
}
.card-icon {
    font-size: 1.75rem;
    margin-bottom: 0.75rem;
}
.welcome-card h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text-main) !important;
    margin-bottom: 0.4rem !important;
}
.welcome-card p {
    font-size: 0.875rem !important;
    color: var(--text-muted) !important;
    line-height: 1.5 !important;
    text-align: left !important;
}
.welcome-footer {
    text-align: center;
    border-top: 1px solid var(--border-color);
    padding-top: 1.5rem;
}
.welcome-footer p {
    font-size: 0.95rem;
    color: var(--text-main);
    margin: 0 !important;
}

/* ===== FILE UPLOADER & CARDS ===== */
[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important;
    border: 1px dashed rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
.file-details-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
}
.file-details-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.file-details-icon {
    font-size: 1.5rem;
}
.file-details-meta {
    display: flex;
    flex-direction: column;
}
.file-details-name {
    font-weight: 600;
    color: var(--text-main);
    font-size: 0.9rem;
    word-break: break-all;
}
.file-details-sub {
    font-size: 0.775rem;
    color: var(--text-muted);
    margin-top: 1px;
}
.file-details-status {
    background-color: rgba(99, 102, 241, 0.1);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.2);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
}

/* Success details card */
.upload-success-card {
    background-color: rgba(16, 185, 129, 0.03);
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: var(--shadow-sm);
}
.upload-success-header {
    margin-bottom: 0.75rem;
}
.upload-success-header h4 {
    margin: 0.5rem 0 0 0 !important;
    color: var(--text-main) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
.upload-success-badge {
    background-color: rgba(16, 185, 129, 0.12);
    color: #10b981;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.upload-stats-row {
    display: flex;
    gap: 2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 0.75rem;
    margin-top: 0.75rem;
}
.upload-stat-item {
    display: flex;
    flex-direction: column;
}
.upload-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #10b981;
}
.upload-stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Offline Page Card */
.offline-card {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(239, 68, 68, 0.02);
    border: 1px solid rgba(239, 68, 68, 0.15);
    border-radius: 20px;
    max-width: 480px;
    margin: 6rem auto;
    box-shadow: var(--shadow-md);
}
.offline-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.offline-card h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #ef4444 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}
.offline-card p {
    color: var(--text-muted) !important;
    font-size: 0.925rem !important;
    line-height: 1.6 !important;
    margin-bottom: 1.5rem !important;
    text-align: center !important;
}
.offline-code-title {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


def render_sources_section(sources: List[Dict], expanded: bool = False):
    """Render the sources used for a response in a styled UI card format"""
    if not sources:
        return
        
    with st.expander("📑 Sources used for this answer", expanded=expanded):
        for i, source in enumerate(sources, 1):
            relevance = source.get('relevance_score', 0)
            # Colors: Green for high, Amber for medium, Rose for low relevance
            if relevance > 0.8:
                color = "#10b981" # Green
            elif relevance > 0.6:
                color = "#f59e0b" # Amber
            else:
                color = "#f43f5e" # Rose
                
            relevance_pct = int(relevance * 100)
            preview = source.get('preview', '').strip()
            
            source_html = f'''
            <div class="source-card">
                <div class="source-card-header">
                    <div class="source-file-info">
                        <span class="source-icon">{"🖼️" if source.get("is_image") else "📄"}</span>
                        <span class="source-filename">{source.get('source', 'Unknown')}</span>
                        <span class="source-chunk-badge">{"Visual Element" if source.get("is_image") else f"Chunk #{source.get('chunk_id', 'N/A')}"}</span>
                    </div>
                    <div class="source-relevance">
                        <span class="relevance-label">Relevance</span>
                        <div class="relevance-bar-container">
                            <div class="relevance-bar" style="width: {relevance_pct}%; background-color: {color};"></div>
                        </div>
                        <span class="relevance-text" style="color: {color};">{relevance_pct}%</span>
                    </div>
                </div>
                <div class="source-preview-container" style="border-left-color: {color};">
                    <pre class="source-preview">{preview}</pre>
                </div>
            </div>
            '''
            st.markdown(source_html, unsafe_allow_html=True)
            
            if source.get("is_image") and source.get("image_path"):
                image_url = f"{API_URL}/images/{source.get('image_path')}"
                st.image(image_url, caption=f"Visual Element from {source.get('source')} (Page {source.get('page_number', 'N/A')})", use_column_width=True)


# ============================================
# MAIN APPLICATION FLOW
# ============================================
def main():
    # Initialize session state variables
    init_session_state()
    
    # Check backend health
    if not check_backend_health():
        # Beautiful offline card layout
        offline_html = '''
        <div class="offline-card">
            <div class="offline-icon">⚠️</div>
            <h2>System Offline</h2>
            <p>The RAG backend server is currently unreachable. Make sure the API service is up and running on port 8000.</p>
            <div class="offline-code-title">Start Command</div>
        </div>
        '''
        st.markdown(offline_html, unsafe_allow_html=True)
        st.code("cd backend && python main.py", language="bash")
        st.stop()
        
    # Populate/update stats & filter values in memory
    refresh_system_state()
    
    # Styled Header
    st.markdown('''
    <div class="app-header">
        <h1>📚 RAG Document Q&A System</h1>
        <p>Analyze documents and extract grounded insights with verifiable sources</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # ============================================
    # SIDEBAR CONTROL
    # ============================================
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # System statistics
        stats = st.session_state.stats
        if stats and stats.get("success"):
            st.subheader("📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-val">{stats.get("total_documents", 0)}</div>
                    <div class="metric-lbl">Chunks</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown('''
                <div class="metric-container">
                    <div class="metric-val" style="background: linear-gradient(135deg, #34d399 0%, #059669 100%); -webkit-background-clip: text;">Active</div>
                    <div class="metric-lbl">Backend</div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
        # Action buttons
        st.subheader("🛠️ Maintenance")
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            with st.spinner("Clearing Database..."):
                result = clear_database()
                if result.get("success"):
                    st.session_state.messages = []
                    refresh_system_state(force=True)
                    st.toast("✅ Database cleared successfully!", icon="🗑️")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
                    
        if st.button("💬 Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.toast("✅ Chat history cleared!", icon="💬")
            st.rerun()
            
        st.markdown("---")
        
        # Filtering Controls
        st.subheader("🔍 Context Filter")
        meta_vals = st.session_state.metadata_values
        filter_dict = {}
        
        if meta_vals and meta_vals.get("success") and (meta_vals.get("sources") or meta_vals.get("file_types")):
            sources = meta_vals.get("sources", [])
            file_types = meta_vals.get("file_types", [])
            
            enable_filters = st.checkbox("Apply Query Filters", value=False)
            
            if enable_filters:
                selected_source = st.selectbox(
                    "Filter by Document",
                    options=["All"] + sources,
                    help="Limit search to a single document"
                )
                selected_file_type = st.selectbox(
                    "Filter by File Type",
                    options=["All"] + file_types,
                    help="Limit search to specific file formats"
                )
                
                # Construct query dictionary filter
                and_conditions = []
                if selected_source != "All":
                    and_conditions.append({"source": selected_source})
                if selected_file_type != "All":
                    and_conditions.append({"file_type": selected_file_type})
                    
                if len(and_conditions) == 1:
                    filter_dict = and_conditions[0]
                elif len(and_conditions) > 1:
                    filter_dict = {"$and": and_conditions}
                    
                st.info(f"Active Filter: `{filter_dict}`")
        else:
            st.caption("No indexed documents available for filters.")
            
        st.markdown("---")
        
        # Optimization Toggles
        st.subheader("🚀 Optimizations")
        use_hyde = st.checkbox(
            "Enable HyDE (Query Expansion)", 
            value=True,
            help="Generates a hypothetical answer before retrieval. Improves recall for semantic search at the cost of one extra LLM latency call (~300-600ms)."
        )
        use_litm_packing = st.checkbox(
            "Enable Lost-in-the-Middle Packing",
            value=True,
            help="Restructures retrieved chunks to place the highest-relevance contexts at the prompt boundaries (start and end). Counteracts transformer model 'inattention valley' in long contexts."
        )
        
        st.markdown("---")
        
        # Info Badge
        st.subheader("⚙️ System Stack")
        st.markdown("""
        **LLM:** Llama 3.3 (via Groq)  
        **Embeddings:** MiniLM-L6 (384d)  
        **Vector Database:** ChromaDB  
        **Pipeline Framework:** FastAPI / Streamlit
        """)
        
    # ============================================
    # MAIN CONTENTS (TABS)
    # ============================================
    tab1, tab2 = st.tabs(["💬 Chat Q&A Space", "📤 Document Manager"])
    
    # --------------------------------------------
    # TAB 1: CHAT & QUERY
    # --------------------------------------------
    with tab1:
        # Check database count to determine dashboard vs chat state
        total_chunks = 0
        if stats and stats.get("success"):
            total_chunks = stats.get("total_documents", 0)
            
        if total_chunks == 0:
            # Welcome Dashboard Hero for empty state
            welcome_html = '''
            <div class="welcome-container">
                <div class="welcome-hero">
                    <h2>Meet Your Grounded AI Assistant</h2>
                    <p>A premium Retrieval-Augmented Generation hub. Connect your files and start extracting cited facts instantly.</p>
                </div>
                <div class="welcome-grid">
                    <div class="welcome-card">
                        <div class="card-icon">📁</div>
                        <h3>Smart Ingestion</h3>
                        <p>Upload PDFs, Word docs, or Text. Select Semantic, Hierarchical, or Recursive chunking strategies.</p>
                    </div>
                    <div class="welcome-card">
                        <div class="card-icon">⚡</div>
                        <h3>Advanced Retrieval</h3>
                        <p>Leverages dense embeddings, HyDE query expansion, and Lost-in-the-Middle chunk re-ordering for precise recall.</p>
                    </div>
                    <div class="welcome-card">
                        <div class="card-icon">📑</div>
                        <h3>Grounded Citations</h3>
                        <p>Answers include precise source references, document names, chunk IDs, and color-coded relevance scores.</p>
                    </div>
                </div>
                <div class="welcome-footer">
                    <p>💡 <b>Quick Start:</b> Head over to the <b>Document Manager</b> tab to upload your first document!</p>
                </div>
            </div>
            '''
            st.markdown(welcome_html, unsafe_allow_html=True)
            
        else:
            # Display chat message history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        render_sources_section(message["sources"], expanded=False)
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Display user input in UI
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Append user prompt to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Stream assistant output
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    sources = []
                    
                    history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    
                    # Fetch streams
                    stream_generator = query_documents_stream(
                        question=prompt,
                        history=history_payload,
                        filters=filter_dict,
                        use_hyde=use_hyde,
                        use_litm_packing=use_litm_packing
                    )
                    
                    for event in stream_generator:
                        if event.get("type") == "sources":
                            sources = event.get("sources", [])
                        elif event.get("type") == "token":
                            full_response += event.get("token", "")
                            # Styled cursor indicator
                            message_placeholder.markdown(full_response + "▌")
                        elif event.get("type") == "error":
                            st.error(f"❌ Error: {event.get('message', 'Unknown error')}")
                            break
                            
                    # Render final response cleanly without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Custom styled sources card
                    if sources:
                        render_sources_section(sources, expanded=True)
                        
                # Save response to history and rerun cleanly
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                st.rerun()

    # --------------------------------------------
    # TAB 2: DOCUMENT UPLOADS
    # --------------------------------------------
    with tab2:
        st.header("Document Manager")
        st.info("📄 **Supported Formats:** PDF, Microsoft Word (.docx), Plain Text (.txt)")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Custom styled file details card instead of standard metrics
            file_type_str = uploaded_file.type.split('/')[-1].upper()
            file_size_str = f"{uploaded_file.size / 1024:.1f} KB"
            
            details_html = f'''
            <div class="file-details-card">
                <div class="file-details-left">
                    <span class="file-details-icon">📄</span>
                    <div class="file-details-meta">
                        <span class="file-details-name">{uploaded_file.name}</span>
                        <span class="file-details-sub">{file_type_str} &bull; {file_size_str}</span>
                    </div>
                </div>
                <div class="file-details-status">Ready to Index</div>
            </div>
            '''
            st.markdown(details_html, unsafe_allow_html=True)
            
            # Ingestion Configurations
            st.markdown("---")
            with st.expander("⚙️ Advanced Ingestion Settings", expanded=True):
                chunking_strategy = st.selectbox(
                    "Chunking Strategy",
                    options=["recursive", "semantic", "hierarchical"],
                    format_func=lambda x: {
                        "recursive": "Recursive Character Chunking",
                        "semantic": "Semantic Chunking (Vector Similarity)",
                        "hierarchical": "Hierarchical (Parent-Child) Chunking"
                    }.get(x, x),
                    help="Recursive splits by fixed sizes. Semantic splits dynamically on context shifts. Hierarchical indexes sub-chunks but references parent documents."
                )
                
                alpha = 1.0
                if chunking_strategy == "semantic":
                    alpha = st.slider(
                        "Semantic Threshold Alpha",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        help="Standard deviation multiplier. Lower alpha creates smaller, tighter chunks. Higher alpha creates larger, broader chunks."
                    )
            
            st.markdown("---")
            
            # Action button
            if st.button("📤 Upload & Process Document", use_container_width=True):
                with st.spinner("Processing & Indexing Document..."):
                    result = upload_document(
                        uploaded_file,
                        chunking_strategy=chunking_strategy,
                        alpha=alpha
                    )
                    
                if result.get("success"):
                    # Trigger cache clear and reload
                    refresh_system_state(force=True)
                    
                    # Beautiful Success display
                    success_html = f'''
                    <div class="upload-success-card">
                        <div class="upload-success-header">
                            <span class="upload-success-badge">✅ SUCCESS</span>
                            <h4>Document processed and indexed into Vector Store.</h4>
                        </div>
                        <div class="upload-stats-row">
                            <div class="upload-stat-item">
                                <span class="upload-stat-val">+{result.get("chunks_created", 0)}</span>
                                <span class="upload-stat-label">Chunks Created</span>
                            </div>
                            <div class="upload-stat-item">
                                <span class="upload-stat-val">{result.get("total_documents_in_db", 0)}</span>
                                <span class="upload-stat-label">Total DB Chunks</span>
                            </div>
                        </div>
                    </div>
                    '''
                    st.markdown(success_html, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error(f"❌ **Ingestion Failed:** {result.get('message', 'Unknown error occurred')}")
                    
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.info("💡 **Next Step:** Once uploaded, head over to the **Chat Q&A Space** tab to query your new document context!")


if __name__ == "__main__":
    main()
