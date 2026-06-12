"""
Streamlit Frontend for RAG System

This creates an interactive web interface for:
1. Uploading documents
2. Asking questions
3. Viewing answers with sources
4. Managing the system

"""

import streamlit as st
import requests
from typing import Dict, List
import json
from pathlib import Path



# CONFIGURATION

# Backend API URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# HELPER FUNCTIONS

def check_backend_health() -> bool:
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_stats() -> Dict:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def upload_document(file) -> Dict:
    """Upload a document to the backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def query_documents(question: str, history: List[Dict] = None) -> Dict:
    """Ask a question"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "history": history or []
            }
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def clear_database() -> Dict:
    """Clear all documents"""
    try:
        response = requests.delete(f"{API_URL}/clear")
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.main {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* ===== HEADER ===== */
h1 {
    text-align: center;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
}

p {
    text-align: center;
    color: #94a3b8;
}

/* ===== CENTER CONTENT ===== */
.block-container {
    max-width: 800px;
    margin: auto;
    padding-top: 2rem;
}

/* ===== INPUT BOX (ChatGPT Style) ===== */
.stTextInput > div > div > input {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px;
    font-size: 15px;
}

.stTextInput > div > div > input:focus {
    border-color: #10a37f;
    box-shadow: 0 0 0 1px #10a37f;
}

/* ===== BUTTON ===== */
.stButton > button {
    background-color: #10a37f;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 500;
    border: none;
}

.stButton > button:hover {
    background-color: #0d8c6d;
}

/* ===== ANSWER BOX (Chat Bubble) ===== */
.chat-bubble {
    background-color: #1e293b;
    padding: 16px;
    border-radius: 12px;
    margin-top: 15px;
    line-height: 1.6;
    border: 1px solid #334155;
}

/* ===== SOURCES ===== */
.streamlit-expanderHeader {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px;
    border: 1px solid #334155;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1e293b;
}

/* ===== METRICS ===== */
[data-testid="stMetricValue"] {
    color: #f8fafc;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

/* ===== ALERTS ===== */
.stSuccess, .stInfo, .stWarning, .stError {
    background-color: #1e293b !important;
    border-radius: 10px;
    border: 1px solid #334155;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
}

.stTabs [aria-selected="true"] {
    color: #10a37f !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    border: 1px dashed #334155;
    border-radius: 10px;
}

/* ===== REMOVE EXTRA SPACING ===== */
.element-container {
    margin-bottom: 0.5rem !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem;">📚 RAG Document Q&A System</h1>
        <p style="font-size: 1.1rem; color: #64748b; margin: 0;">Upload documents and ask questions with cited, grounded answers</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Check backend health
    if not check_backend_health():
        st.error("⚠️ **Backend is not running!** Please start the backend server first.")
        st.code("cd backend && python main.py", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System stats
        stats = get_stats()
        if stats and stats.get("success"):
            st.subheader("📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Status", "✅")
            st.markdown("---")
        
        # Clear database button
        if st.button("🗑️ Clear Database", use_container_width=True):
            with st.spinner("Clearing..."):
                result = clear_database()
                if result.get("success"):
                    st.session_state.messages = []
                    st.success("✅ Database cleared!")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
        
        # Clear chat history button
        if st.button("💬 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat history cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # System info
        st.subheader("ℹ️ System Info")
        st.markdown("""
        **LLM:** Groq (Llama 3.3)  
        **Embeddings:** MiniLM-L6  
        **Vector DB:** ChromaDB  
        **Backend:** FastAPI  
        **Frontend:** Streamlit
        """)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📤 Upload Documents"])
    
    # ============================================
    # TAB 1: ASK QUESTIONS
    # ============================================
    with tab1:
        st.header("Ask Questions")
        
        # Check if documents are uploaded
        stats = get_stats()
        if not stats or stats.get("total_documents", 0) == 0:
            st.warning("📝 No documents uploaded yet! Please upload documents in the 'Upload Documents' tab.")
        else:
            st.success(f"✅ {stats.get('total_documents', 0)} document chunks available")
        
        # Initialize session state chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Check if there are saved sources for this message
                if "sources" in message and message["sources"]:
                    with st.expander("📑 Sources used for this answer"):
                        for i, source in enumerate(message["sources"], 1):
                            relevance = source.get('relevance_score', 0)
                            color = "#10b981" if relevance > 0.8 else "#f59e0b" if relevance > 0.6 else "#ef4444"
                            st.markdown(f"**Source {i}:** {source.get('source', 'Unknown')} (Relevance: {relevance:.1%})")
                            st.markdown(f"**Chunk:** {source.get('chunk_id', 'N/A')}")
                            
                            preview_html = f'''
                            <div style="
                                background-color: #0f172a;
                                padding: 1rem;
                                border-radius: 0.5rem;
                                border-left: 4px solid {color};
                                color: #cbd5e1;
                                font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
                                font-size: 0.875rem;
                                line-height: 1.6;
                                white-space: pre-wrap;
                                border: 1px solid #1e293b;
                            ">
{source.get('preview', '')}
                            </div>
                            '''
                            st.markdown(preview_html, unsafe_allow_html=True)
                            st.markdown("---")
        
        # React to user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Send query to backend
            with st.spinner("🤔 Thinking..."):
                # Pass previous messages (excluding the current prompt we just appended)
                history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                result = query_documents(prompt, history_payload)
                
            if result.get("success"):
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📑 Sources used for this answer", expanded=True):
                            for i, source in enumerate(sources, 1):
                                relevance = source.get('relevance_score', 0)
                                color = "#10b981" if relevance > 0.8 else "#f59e0b" if relevance > 0.6 else "#ef4444"
                                st.markdown(f"**Source {i}:** {source.get('source', 'Unknown')} (Relevance: {relevance:.1%})")
                                st.markdown(f"**Chunk:** {source.get('chunk_id', 'N/A')}")
                                
                                preview_html = f'''
                                <div style="
                                    background-color: #0f172a;
                                    padding: 1rem;
                                    border-radius: 0.5rem;
                                    border-left: 4px solid {color};
                                    color: #cbd5e1;
                                    font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
                                    font-size: 0.875rem;
                                    line-height: 1.6;
                                    white-space: pre-wrap;
                                    border: 1px solid #1e293b;
                                ">
{source.get('preview', '')}
                                </div>
                                '''
                                st.markdown(preview_html, unsafe_allow_html=True)
                                st.markdown("---")
                                
                # Add assistant response with sources to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            else:
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    # ============================================
    # TAB 2: UPLOAD DOCUMENTS
    # ============================================
    with tab2:
        st.header("Upload Documents")
        
        st.info("📄 **Supported formats:** PDF, Word (.docx), Text (.txt)")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "**Choose a file**",
            type=["pdf", "docx", "txt"],
            label_visibility="visible"
        )
        
        if uploaded_file:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Filename", uploaded_file.name)
            with col2:
                st.metric("📦 Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("📋 Type", uploaded_file.type.split('/')[-1].upper())
            
            # Upload button
            if st.button("📤 Upload & Process", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing document... This may take a minute..."):
                    result = upload_document(uploaded_file)
                
                if result.get("success"):
                    st.success("✅ **Document uploaded successfully!**")
                    
                    # Show processing details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chunks Created", result.get("chunks_created", 0))
                    with col2:
                        st.metric("Total in Database", result.get("total_documents_in_db", 0))
                    
                    st.balloons()
                else:
                    st.error(f"❌ **Error:** {result.get('message', 'Unknown error')}")
        
        st.markdown("---")
        st.info("💡 **Tip:** After uploading, go to the 'Ask Questions' tab to query your documents!")



if __name__ == "__main__":
    main()
