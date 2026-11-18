import os
import streamlit as st
from dotenv import load_dotenv
from vectorstore.chroma_client import get_chroma_vectorstore
from indexer.codebase_indexer import CodebaseIndexer
from rag.llm_backends import OllamaBackend, ClaudeBackend
from rag.retriever import retrieve_relevant_chunks
from rag.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context, extract_citations

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Codebase RAG Assistant",
    page_icon="AI",
    layout="wide"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Title */
.title-gradient {
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Sidebar cards */
.sidebar-section {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Premium Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.2) !important;
    width: 100% !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(168, 85, 247, 0.35) !important;
}

/* Warnings and callouts */
.warning-box {
    border-left: 4px solid #f59e0b;
    background-color: rgba(245, 158, 11, 0.08);
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}

.insufficient-context-banner {
    border-left: 4px solid #ef4444;
    background-color: rgba(239, 68, 68, 0.08);
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Helper to determine programming language for styling code blocks
def get_language_for_ext(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.md': 'markdown',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.sh': 'bash',
        '.go': 'go',
        '.java': 'java',
        '.rs': 'rust',
        '.sql': 'sql',
        '.xml': 'xml'
    }
    return mapping.get(ext, 'text')

# Helper to check if LLM response indicates insufficient context
def is_insufficient_context(response_text: str) -> bool:
    keywords = [
        "insufficient context", 
        "not enough context", 
        "insufficient information",
        "cannot answer",
        "do not have enough information",
        "not present in the provided context",
        "not present in the context",
        "no context provided"
    ]
    text_lower = response_text.lower()
    return any(kw in text_lower for kw in keywords)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_path" not in st.session_state:
    st.session_state.indexed_path = None

# Title
st.markdown("<h1 class='title-gradient'>Codebase RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask grounded, source-attributed questions about your codebase.</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.title("Configuration")

# 1. Codebase Directory Input
st.sidebar.subheader("Codebase Settings")
default_path = os.getcwd()
codebase_path = st.sidebar.text_input(
    "Codebase Root Path", 
    value=st.session_state.get("indexed_path", default_path),
    help="Absolute path to the codebase directory you want to search."
)

chroma_dir = os.path.join(default_path, "chroma_db")

# Index / Re-index button
if st.sidebar.button("Index / Re-index"):
    if not os.path.exists(codebase_path):
        st.sidebar.error(f"Directory does not exist: {codebase_path}")
    else:
        with st.spinner("Indexing codebase..."):
            # Setup log container
            log_container = st.sidebar.empty()
            progress_bar = st.sidebar.progress(0.0)
            
            def index_progress_callback(update: dict):
                status = update.get("status")
                msg = update.get("message", "")
                
                if status == "scanning":
                    log_container.info(msg)
                elif status == "deleting":
                    log_container.warning(msg)
                elif status == "indexing":
                    curr = update.get("current", 0)
                    total = update.get("total", 1)
                    file = update.get("file", "")
                    progress_bar.progress(curr / total)
                    log_container.text(f"[{curr}/{total}] Indexing {file}")
                elif status == "complete":
                    stats = update.get("stats", {})
                    progress_bar.progress(1.0)
                    log_container.success(
                        f"Indexing complete!\n\n"
                        f"- Scanned: {stats['scanned']} files\n"
                        f"- Indexed (new/modified): {stats['indexed']} files\n"
                        f"- Skipped: {stats['skipped']} files\n"
                        f"- Deleted: {stats['deleted']} files\n"
                        f"- Total Chunks: {stats['total_chunks']}"
                    )
            
            try:
                indexer = CodebaseIndexer(codebase_path, persist_directory=chroma_dir)
                indexer.index_codebase(progress_callback=index_progress_callback)
                st.session_state.vectorstore = indexer.vectorstore
                st.session_state.indexed_path = codebase_path
            except Exception as e:
                progress_bar.empty()
                log_container.empty()
                st.sidebar.error(f"Indexing failed: {e}")

# 2. LLM Backend Selector
st.sidebar.markdown("---")
st.sidebar.subheader("LLM Generation Settings")
backend_option = st.sidebar.selectbox(
    "LLM Backend", 
    ["Ollama (local - llama3.1)", "Claude API"],
    help="Select whether to run generation locally or use external Claude API."
)

backend = None
try:
    if backend_option == "Ollama (local - llama3.1)":
        backend = OllamaBackend()
    else:
        # This will raise an error if ANTHROPIC_API_KEY is not configured
        backend = ClaudeBackend()
except Exception as e:
    st.sidebar.error(f"LLM Config Error: {e}")

# 3. Retrieval Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval Hyperparameters")
k_val = st.sidebar.slider(
    "Top-k chunks to retrieve", 
    min_value=1, 
    max_value=20, 
    value=6,
    help="The number of relevant code chunks to supply as context."
)

threshold_val = st.sidebar.slider(
    "Relevance Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.40, 
    step=0.05,
    help="Minimum similarity score required to process the query. Below this, search fails immediately."
)

# ----------------- MAIN INTERFACE -----------------

# Lazy load vectorstore if already exists on disk but not in session state
if st.session_state.vectorstore is None and os.path.exists(chroma_dir):
    try:
        st.session_state.vectorstore = get_chroma_vectorstore(persist_directory=chroma_dir)
        st.session_state.indexed_path = codebase_path
    except Exception as e:
        # Fail silently here, user will get prompted when they index or search
        pass

# Display Status
if st.session_state.vectorstore is None:
    st.info("Welcome! Please enter a codebase path in the sidebar and click **Index / Re-index** to get started.")
else:
    st.success(f"Active Vector Database: `./chroma_db` (Grounding codebase: `{st.session_state.indexed_path}`)")

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_warning", False):
            st.markdown(f"<div class='warning-box'>{msg['content']}</div>", unsafe_allow_html=True)
        elif msg.get("is_insufficient", False):
            st.markdown("<div class='insufficient-context-banner'>[Insufficient Context Notice]</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])
            
        # Render citations
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"View Sources ({len(citations)})"):
                for cit in citations:
                    st.markdown(f"**`{cit['file_path']}` (Lines {cit['start_line']}-{cit['end_line']})**")
                    st.code(cit["doc_content"], language=get_language_for_ext(cit["file_path"]))

# User Query Input
if user_query := st.chat_input("Ask a question about the codebase..."):
    # Check if vectorstore is initialized
    if st.session_state.vectorstore is None:
        st.error("No active vector store found. Please index a codebase first.")
    elif backend is None:
        st.error("Selected LLM backend is not configured correctly. Check your settings/sidebar.")
    else:
        # 1. Add user message to UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Perform Retrieval & Generation
        with st.chat_message("assistant"):
            with st.spinner("Searching codebase..."):
                retrieved_results = retrieve_relevant_chunks(
                    st.session_state.vectorstore,
                    user_query,
                    k=k_val,
                    threshold=threshold_val
                )
                
            if not retrieved_results:
                # Similarity threshold check failed or no documents found
                warn_msg = "Warning: No relevant code found in the codebase matching your query (similarity score below threshold)."
                st.markdown(f"<div class='warning-box'>{warn_msg}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": warn_msg,
                    "is_warning": True,
                    "citations": []
                })
            else:
                docs = [doc for doc, score in retrieved_results]
                formatted_ctx = format_context(docs)
                user_prompt = USER_PROMPT_TEMPLATE.format(context=formatted_ctx, question=user_query)
                
                with st.spinner("Generating answer..."):
                    try:
                        response = backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
                        
                        # Extract citations
                        verified_citations = extract_citations(response, docs)
                        
                        # Check for insufficient context
                        insufficient = is_insufficient_context(response)
                        
                        if insufficient:
                            st.markdown("<div class='insufficient-context-banner'>[Insufficient Context Notice]</div>", unsafe_allow_html=True)
                        
                        st.markdown(response)
                        
                        # Render Sources
                        citations_to_save = []
                        if verified_citations:
                            with st.expander(f"View Sources ({len(verified_citations)})"):
                                for cit in verified_citations:
                                    st.markdown(f"**`{cit['file_path']}` (Lines {cit['start_line']}-{cit['end_line']})**")
                                    st.code(cit["doc"].page_content, language=get_language_for_ext(cit["file_path"]))
                                    
                                    # Prepare serializable list for session state
                                    citations_to_save.append({
                                        "file_path": cit["file_path"],
                                        "start_line": cit["start_line"],
                                        "end_line": cit["end_line"],
                                        "doc_content": cit["doc"].page_content
                                    })
                                    
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "is_insufficient": insufficient,
                            "citations": citations_to_save
                        })
                        
                    except Exception as ex:
                        st.error(f"Error during response generation: {ex}")
