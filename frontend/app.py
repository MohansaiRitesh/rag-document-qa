"""
Streamlit Frontend for RAG System - Unified Single-Page Conversational Workspace
"""

import streamlit as st
import requests
from typing import Dict, List, Generator
import json
from pathlib import Path


# Backend API URL
API_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Maester AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# HELPER FUNCTIONS & API CLIENTS
# =============================================================================
@st.cache_data(ttl=10)
def check_backend_health() -> bool:
    """Check if backend is running (cached to prevent UI lag on input)"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        return response.status_code == 200
    except:
        return False


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = None
    if "metadata_values" not in st.session_state:
        st.session_state.metadata_values = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []


def refresh_system_state(force=False):
    """Refresh system statistics, metadata filter values, and indexed files from backend"""
    if force or st.session_state.stats is None:
        try:
            response = requests.get(f"{API_URL}/stats", timeout=2)
            if response.status_code == 200:
                st.session_state.stats = response.json()
            else:
                st.session_state.stats = {"success": False, "total_documents": 0}
        except Exception:
            st.session_state.stats = {"success": False, "total_documents": 0}

    if force or st.session_state.metadata_values is None:
        try:
            response = requests.get(f"{API_URL}/metadata-values", timeout=2)
            if response.status_code == 200:
                st.session_state.metadata_values = response.json()
            else:
                st.session_state.metadata_values = {"success": False, "sources": [], "file_types": []}
        except Exception:
            st.session_state.metadata_values = {"success": False, "sources": [], "file_types": []}

    # Fetch document files list
    if force or not st.session_state.indexed_files:
        try:
            response = requests.get(f"{API_URL}/documents", timeout=2)
            if response.status_code == 200:
                st.session_state.indexed_files = response.json().get("files", [])
            else:
                st.session_state.indexed_files = []
        except Exception:
            st.session_state.indexed_files = []


def upload_document(file, chunking_strategy: str = "recursive", alpha: float = 1.0) -> Dict:
    """Upload a document to the backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {
            "chunking_strategy": chunking_strategy,
            "semantic_threshold_alpha": alpha
        }
        response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=90)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def query_documents_stream(
    question: str, 
    history: List[Dict] = None, 
    filters: Dict = None, 
    use_hyde: bool = True, 
    use_litm_packing: bool = True,
    use_agentic_rag: bool = False
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
                "use_litm_packing": use_litm_packing,
                "use_agentic_rag": use_agentic_rag
            },
            stream=True,
            timeout=60
        )
        if response.status_code != 200:
            yield {"type": "error", "message": f"Server error ({response.status_code}): {response.text}"}
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
    """Clear all uploaded documents"""
    try:
        response = requests.delete(f"{API_URL}/clear", timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


# =============================================================================
# PRODUCTION DESIGN SYSTEM & CUSTOM CSS OVERHAUL
# =============================================================================
st.markdown("""
<style>
/* Import Web Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Design Tokens */
:root {
    --bg-main: #080b11;
    --bg-surface: #0e131f;
    --bg-card: #151c2c;
    --bg-input: #1a2336;
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-accent: rgba(99, 102, 241, 0.3);
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%);
    --accent-indigo: #6366f1;
    --accent-purple: #a855f7;
    --accent-cyan: #06b6d4;
    --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.15);
}

/* Base Body Styling */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Header & Workspace Container */
header, [data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    max-width: 840px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* App Navigation Header */
.app-brand-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-status {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.775rem;
    color: #34d399;
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.2);
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
    width: 330px !important;
}
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-top: 1rem !important;
}

/* Quick Action Buttons */
div.stButton > button {
    background: var(--accent-gradient) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    color: #f87171 !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    margin-bottom: 1.25rem !important;
    box-shadow: var(--shadow-soft) !important;
    transition: all 0.2s ease;
}
div[data-testid="stChatMessage"]:hover {
    border-color: var(--border-accent) !important;
}
div[data-testid="stChatMessage"][data-testid$="user"] {
    background-color: rgba(99, 102, 241, 0.04) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
}

/* Input Area */
div[data-testid="stChatInput"] {
    border-radius: 16px !important;
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-accent) !important;
    padding: 6px 12px !important;
    box-shadow: var(--shadow-glow) !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
}

/* Active File Item Cards */
.file-item-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.file-item-name {
    font-size: 0.825rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 190px;
}
.file-item-sub {
    font-size: 0.725rem;
    color: var(--text-muted);
}

/* Empty State Starter Cards */
.hero-container {
    text-align: center;
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 580px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}

/* Perplexity Citation Cards */
.citation-card {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.citation-card:hover {
    border-color: var(--border-accent);
}
.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.citation-title {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.citation-score {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
}
.citation-snippet {
    font-family: monospace;
    font-size: 0.825rem;
    color: #cbd5e1;
    white-space: pre-wrap;
    line-height: 1.4;
    margin: 0;
}

/* Offline Banner */
.offline-card {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(239, 68, 68, 0.03);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 20px;
    max-width: 500px;
    margin: 5rem auto;
}
.offline-card h2 {
    font-family: 'Space Grotesk', sans-serif;
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CITATION & MEDIA CARDS RENDERER
# =============================================================================
def render_citations(sources: List[Dict], expanded: bool = False):
    """Render Perplexity-styled citations drawer"""
    if not sources:
        return
        
    with st.expander("📑 Grounded Sources & Citations", expanded=expanded):
        for i, src in enumerate(sources, 1):
            score = src.get('relevance_score', 0.0)
            score_pct = int(score * 100)
            
            if score > 0.8:
                badge_bg = "rgba(16, 185, 129, 0.15)"
                badge_color = "#34d399"
            elif score > 0.5:
                badge_bg = "rgba(245, 158, 11, 0.15)"
                badge_color = "#fbbf24"
            else:
                badge_bg = "rgba(244, 63, 94, 0.15)"
                badge_color = "#fb7185"
                
            icon = "🖼️" if src.get("is_image") else ("🌐" if src.get("file_type") == "web" else "📄")
            preview = src.get('preview', '').strip()
            
            card_html = f'''
            <div class="citation-card">
                <div class="citation-header">
                    <span class="citation-title">{icon} {src.get('source', 'Document')}</span>
                    <span class="citation-score" style="background: {badge_bg}; color: {badge_color};">
                        {score_pct}% Match
                    </span>
                </div>
                {f'<pre class="citation-snippet">{preview}</pre>' if preview else ''}
            </div>
            '''
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Render embedded extracted images if present
            if src.get("is_image") and src.get("image_path"):
                image_url = f"{API_URL}/images/{src.get('image_path')}"
                st.image(
                    image_url, 
                    caption=f"Extracted Visual Content from {src.get('source')} (Page {src.get('page_number', 'N/A')})", 
                    use_column_width=True
                )


# =============================================================================
# MAIN APP ENTRY POINT (UNIFIED SINGLE PAGE)
# =============================================================================
def main():
    init_session_state()
    
    # 1. Offline Check
    if not check_backend_health():
        st.markdown('''
        <div class="offline-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h2>Backend System Offline</h2>
            <p>The RAG FastAPI engine is unreachable on <code>http://localhost:8000</code>. Please start the backend service.</p>
        </div>
        ''', unsafe_allow_html=True)
        st.code("cd backend && python main.py", language="bash")
        st.stop()
        
    refresh_system_state()
    
    # 2. Header Brand Banner
    st.markdown('''
    <div class="app-brand-header">
        <div class="brand-title">Maester AI</div>
        <div class="brand-status"><span>●</span> System Active</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 3. Sidebar Navigation & Document Management
    with st.sidebar:
        st.markdown("### 📎 Attach & Index Document")
        
        # File Uploader
        sidebar_file = st.file_uploader(
            "Upload Document", 
            type=["pdf", "docx", "txt"], 
            label_visibility="collapsed",
            key="sidebar_uploader"
        )
        
        if sidebar_file:
            if st.button("📤 Process & Index Document", use_container_width=True):
                with st.spinner("Processing & Indexing..."):
                    res = upload_document(sidebar_file, chunking_strategy=strategy, alpha=alpha)
                if res.get("success"):
                    refresh_system_state(force=True)
                    st.toast(f"✅ Indexed {sidebar_file.name} ({res.get('chunks_created')} chunks created)!", icon="🎉")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📥 **System Event**: Successfully indexed document **`{sidebar_file.name}`** into the vector store. You can now ask questions about its contents!"
                    })
                    st.rerun()
                else:
                    st.error(f"❌ Ingestion Failed: {res.get('message')}")

        st.markdown("---")

        # Ingestion Strategy Accordion
        with st.expander("⚙️ Ingestion & Chunking Options", expanded=False):
            strategy = st.selectbox(
                "Chunking Strategy",
                options=["recursive", "semantic", "hierarchical"],
                format_func=lambda x: {
                    "recursive": "Recursive",
                    "semantic": "Semantic",
                    "hierarchical": "Hierarchical"
                }.get(x, x)
            )
            alpha = 1.0
            if strategy == "semantic":
                alpha = st.slider("Semantic Alpha Threshold", 0.1, 3.0, 1.0, 0.1)
        
        # Indexed Documents Drawer
        st.markdown("### 📁 Indexed Documents")
        indexed_files = st.session_state.indexed_files
        stats = st.session_state.stats
        
        if indexed_files:
            for f in indexed_files:
                ext_icon = "📄" if f.get("extension") == ".pdf" else ("📝" if f.get("extension") == ".docx" else "📜")
                size_kb = f"{f.get('size', 0) / 1024:.1f} KB"
                st.markdown(f'''
                <div class="file-item-card">
                    <div>
                        <div class="file-item-name">{ext_icon} {f.get("filename")}</div>
                        <div class="file-item-sub">{size_kb}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.caption("No documents attached yet.")
            
        if stats and stats.get("success"):
            st.caption(f"Total Chunks in DB: **{stats.get('total_documents', 0)}**")

        st.markdown("---")
        
        # Accordion: Pipeline Optimizations & Controls
        with st.expander("🤖 Pipeline & Agent Controls", expanded=True):
            use_agentic_rag = st.checkbox(
                "Enable Agentic RAG Mode",
                value=True,
                help="Enables intent routing, relevance grading, automatic query re-writing, and web fallback."
            )
            use_hyde = st.checkbox(
                "Enable HyDE Expansion",
                value=True,
                help="Generates a hypothetical document before retrieval to bridge vocabulary gaps."
            )
            use_litm_packing = st.checkbox(
                "Enable Lost-in-Middle Packing",
                value=True,
                help="Reorders retrieved context to place high relevance chunks at prompt boundaries."
            )

        # Accordion: Context Metadata Filters
        with st.expander("🔍 Context Metadata Filters", expanded=False):
            meta_vals = st.session_state.metadata_values
            filter_dict = {}
            
            if meta_vals and meta_vals.get("success") and (meta_vals.get("sources") or meta_vals.get("file_types")):
                sources = meta_vals.get("sources", [])
                file_types = meta_vals.get("file_types", [])
                
                enable_filt = st.checkbox("Apply Filters", value=False)
                if enable_filt:
                    sel_source = st.selectbox("Filter Source", ["All"] + sources)
                    sel_type = st.selectbox("Filter File Type", ["All"] + file_types)
                    
                    conds = []
                    if sel_source != "All": conds.append({"source": sel_source})
                    if sel_type != "All": conds.append({"file_type": sel_type})
                    
                    if len(conds) == 1: filter_dict = conds[0]
                    elif len(conds) > 1: filter_dict = {"$and": conds}
                    
                    st.info(f"Active Filter: `{filter_dict}`")
            else:
                st.caption("No indexed documents available for filtering.")

        st.markdown("---")
        
        # Quick actions
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💬 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear DB", type="secondary", use_container_width=True):
                with st.spinner("Clearing DB..."):
                    res = clear_database()
                    if res.get("success"):
                        st.session_state.messages = []
                        refresh_system_state(force=True)
                        st.toast("✅ Database cleared!", icon="🗑️")
                        st.rerun()

        st.markdown("<div style='font-size:0.75rem; color:#6b7280; text-align:center; margin-top:1rem;'>Maester AI Engine</div>", unsafe_allow_html=True)

    # 4. Main Conversational Workspace
    total_chunks = stats.get("total_documents", 0) if stats and stats.get("success") else 0
    
    # Empty State Hero Dashboard
    if total_chunks == 0 and not st.session_state.messages:
        st.markdown('''
        <div class="hero-container">
            <div class="hero-title">What would you like to discover today?</div>
            <div class="hero-subtitle">Attach research papers, policy documents, or technical specs in the sidebar to unlock grounded insights with cited sources.</div>
        </div>
        ''', unsafe_allow_html=True)

    else:
        # Display conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    render_citations(msg["sources"], expanded=False)
                    
        # Chat Input
        prompt_input = st.chat_input("Ask a question about your documents...")
        
        if prompt_input:
            # Display user question
            with st.chat_message("user"):
                st.markdown(prompt_input)
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            
            # Stream Assistant Response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_text = ""
                sources = []
                
                # Gemini/Claude Reasoning Status Block
                status_expander = None
                if use_agentic_rag:
                    status_expander = st.status("🧠 Agent Reasoning Process...", expanded=True)

                history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                
                # Call streaming backend
                stream = query_documents_stream(
                    question=prompt_input,
                    history=history_payload,
                    filters=filter_dict,
                    use_hyde=use_hyde,
                    use_litm_packing=use_litm_packing,
                    use_agentic_rag=use_agentic_rag
                )
                
                for event in stream:
                    etype = event.get("type")
                    
                    if etype == "thought" and status_expander is not None:
                        step = event.get("step", "Reasoning Step")
                        thought = event.get("thought", "")
                        st_icon = "🧠" if event.get("status") == "info" else ("✅" if event.get("status") == "success" else ("⚠️" if event.get("status") == "warning" else "❌"))
                        status_expander.write(f"{st_icon} **{step}**: {thought}")
                        
                    elif etype == "sources":
                        sources = event.get("sources", [])
                        
                    elif etype == "token":
                        if status_expander is not None:
                            status_expander.update(label="⚡ Reasoning Complete", state="complete", expanded=False)
                            status_expander = None
                        full_text += event.get("token", "")
                        response_placeholder.markdown(full_text + "▌")
                        
                    elif etype == "error":
                        if status_expander is not None:
                            status_expander.update(label="❌ Agent Error", state="error", expanded=True)
                        st.error(f"Error: {event.get('message', 'Unknown error')}")
                        break
                        
                response_placeholder.markdown(full_text)
                
                if sources:
                    render_citations(sources, expanded=True)
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "sources": sources
            })
            st.rerun()


if __name__ == "__main__":
    main()
