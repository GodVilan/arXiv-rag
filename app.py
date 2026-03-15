"""
app.py – Streamlit UI for the RAG ML Research Paper Q&A system.

Run:
    streamlit run app.py
"""

import json
import time
from pathlib import Path

import streamlit as st

from rag import config
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.generation.generator import Generator
from rag.retrieval.bm25 import BM25Retriever

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · ML Paper Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1200px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: #c9c7d4 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { 
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b6880 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
}

/* Main title */
.rag-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0eeff;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0;
}
.rag-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #6b6880;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Chat messages */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}
.msg-user-bubble {
    background: #2d2b4e;
    border: 1px solid #3d3b5e;
    color: #e8e6f0;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 1rem 0;
    gap: 10px;
    align-items: flex-start;
}
.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-assistant-bubble {
    background: #13131f;
    border: 1px solid #1e1e2e;
    color: #c9c7d4;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.75;
}

/* Sources */
.sources-container {
    margin-top: 10px;
    border-top: 1px solid #1e1e2e;
    padding-top: 10px;
}
.sources-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a4860;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.source-chip {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 5px 10px;
    margin: 3px 3px 3px 0;
    font-size: 11px;
    color: #8b89a0;
    font-family: 'Inter', sans-serif;
}
.source-score {
    font-family: 'JetBrains Mono', monospace;
    color: #6c63ff;
    font-size: 10px;
    margin-left: 6px;
}

/* Stats bar */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #6b6880;
    margin-right: 8px;
}
.stat-value { color: #a09cc0; font-weight: 500; }

/* Welcome screen */
.welcome-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 1.5rem;
}
.welcome-card {
    background: #0d0d18;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.2s;
}
.welcome-card:hover { border-color: #3d3b5e; }
.welcome-card-icon { font-size: 18px; margin-bottom: 8px; }
.welcome-card-text {
    font-size: 13px;
    color: #8b89a0;
    line-height: 1.5;
    font-family: 'Inter', sans-serif;
}

/* Input area */
.stTextInput > div > div > input {
    background: #0d0d18 !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #4a4860 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #6c63ff !important; }

/* Metric cards in sidebar */
.metric-mini {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.metric-mini-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #4a4860;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-mini-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #a09cc0;
    line-height: 1.2;
}

/* Paper list */
.paper-item {
    background: #0d0d18;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #8b89a0;
}
.paper-item-title { color: #c9c7d4; font-weight: 500; font-size: 13px; }
.paper-item-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a4860;
    margin-top: 3px;
}

/* Dark page background */
.stApp { background: #080810; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_chunk" not in st.session_state:
    st.session_state.current_chunk = None


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_chunks(chunk_size: int):
    cache_path = config.DATA_DIR / f"chunks_{chunk_size}.json"
    if cache_path.exists():
        return load_chunks(cache_path)
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        papers = json.load(f)
    chunks = process_papers(papers, chunk_size=chunk_size)
    save_chunks(chunks, cache_path)
    return chunks


@st.cache_resource(show_spinner=False)
def get_retriever(model_key: str, chunk_size: int):
    chunks = get_chunks(chunk_size)
    if model_key == "BM25":
        return BM25Retriever(chunks)
    return Retriever.build(
        model_key=model_key,
        chunks=chunks,
        chunk_size=chunk_size,
        index_dir=config.RESULTS_DIR / "indices",
    )


@st.cache_resource(show_spinner=False)
def get_generator():
    return Generator()


@st.cache_data(show_spinner=False)
def get_metadata():
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        return json.load(f)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 700; color: #f0eeff; margin-bottom: 4px;">⚙ Configuration</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family: JetBrains Mono, monospace; font-size: 10px; color: #4a4860; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1.5rem;">RAG for Domain Specific Q&A</p>', unsafe_allow_html=True)

    model_key = st.selectbox(
        "Embedding Model",
        options=["BM25"] + list(config.EMBEDDING_MODELS.keys()),
        index=3,   # BGE default
        help="BM25 = sparse baseline  |  BGE = best dense model (MRR 0.97)"
    )

    chunk_size = st.selectbox(
        "Chunk Size",
        options=config.CHUNK_SIZES,
        index=1,  # 512 default
        help="512 tokens balances precision and recall"
    )

    top_k = st.slider("Top-K Sources", min_value=1, max_value=10, value=5)

    st.divider()

    # Model info
    model_names = {
        "MiniLM": "all-MiniLM-L6-v2",
        "MPNet":  "all-mpnet-base-v2",
        "BGE":    "bge-large-en",
    }
    mrr_scores = {"BM25": 0.634, "MiniLM": 0.960, "MPNet": 0.943, "BGE": 0.970}
    ar_scores  = {"BM25": 0.0,   "MiniLM": 0.586, "MPNet": 0.442, "BGE": 0.893}

    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-mini-label">MRR (chunk 1024)</div>
        <div class="metric-mini-value">{mrr_scores[model_key]:.3f}</div>
    </div>
    <div class="metric-mini">
        <div class="metric-mini-label">Answer Relevance</div>
        <div class="metric-mini-value">{ar_scores[model_key]:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Paper browser
    with st.expander("📚 Browse Papers", expanded=False):
        papers = get_metadata()
        if papers:
            search = st.text_input("Filter", placeholder="Search titles…", label_visibility="collapsed")
            filtered = [p for p in papers if search.lower() in p["title"].lower()] if search else papers[:30]
            for p in filtered[:25]:
                st.markdown(f"""
                <div class="paper-item">
                    <div class="paper-item-title">{p['title'][:70]}{'…' if len(p['title']) > 70 else ''}</div>
                    <div class="paper-item-meta">{p['paper_id']} · {p['published'][:10]}</div>
                </div>
                """, unsafe_allow_html=True)
            if len(filtered) > 25:
                st.caption(f"+{len(filtered)-25} more")
        else:
            st.caption("No papers loaded yet.")

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Load model (cached) ───────────────────────────────────────────────────────
chunks = get_chunks(chunk_size)
corpus_ready = len(chunks) > 0

if corpus_ready:
    with st.spinner(f"Loading {model_key} retriever…"):
        retriever = get_retriever(model_key, chunk_size)
    with st.spinner("Loading Gemini generator…"):
        generator = get_generator()
else:
    retriever = None
    generator = None


# ── Main area ─────────────────────────────────────────────────────────────────
col_title, col_stats = st.columns([3, 2])

with col_title:
    st.markdown('<h1 class="rag-title">ML Paper Q&A</h1>', unsafe_allow_html=True)
    st.markdown('<p class="rag-subtitle">RAG · arXiv cs.LG · 2026</p>', unsafe_allow_html=True)

with col_stats:
    if corpus_ready:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:flex-end; height:100%; padding-top:8px; flex-wrap:wrap; gap:6px">
            <span class="stat-pill">📄 <span class="stat-value">{len(get_metadata())}</span> papers</span>
            <span class="stat-pill">🧩 <span class="stat-value">{len(chunks):,}</span> chunks</span>
            <span class="stat-pill">🤖 <span class="stat-value">{model_key}</span></span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #1e1e2e;margin:12px 0 20px'>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1rem">
            <div style="font-size:2.5rem; margin-bottom:12px">🔬</div>
            <p style="font-family: Syne, sans-serif; font-size: 1.1rem; color: #8b89a0;">
                Ask anything about machine learning research
            </p>
            <p style="font-size: 12px; color: #4a4860; font-family: JetBrains Mono, monospace;">
                The system retrieves relevant paper segments and generates grounded answers
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Example questions as clickable buttons
        examples = [
            ("🤖", "What are recent approaches to continual learning in robotics?"),
            ("🧠", "How do transformer models handle long context windows?"),
            ("📊", "What evaluation methods are used for language model reliability?"),
            ("⚡", "What are the latest techniques for efficient inference?"),
        ]

        cols = st.columns(2)
        for i, (icon, example) in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"{icon} {example}", key=f"ex_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()

    else:
        # Render messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    chips = "".join([
                        f'<span class="source-chip">[{i+1}] {s["title"][:45]}{"…" if len(s["title"])>45 else ""}<span class="source-score">{s["score"]:.3f}</span></span>'
                        for i, s in enumerate(msg["sources"])
                    ])
                    sources_html = f'<div class="sources-container"><div class="sources-label">Retrieved sources</div>{chips}</div>'

                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-avatar">🔬</div>
                    <div class="msg-assistant-bubble">
                        {msg["content"]}
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

if not corpus_ready:
    st.error("⚠ No corpus found. Run `python data_collection.py` first.", icon="⚠️")
else:
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([8, 1])
        with col_input:
            user_input = st.text_input(
                "query",
                placeholder="Ask about ML research, methods, papers, findings…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("➤", use_container_width=True)

    # Form submission — just store the message, processing happens below
    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()

# ── Process any unanswered user message (form OR example buttons) ─────────────
if (
    corpus_ready
    and retriever is not None
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    query = st.session_state.messages[-1]["content"]

    with st.spinner("Retrieving relevant passages…"):
        results = retriever.retrieve(query, top_k=top_k)
        context = retriever.format_context(query, top_k=top_k)

    with st.spinner("Generating answer with Gemini…"):
        answer = generator.generate(query, context)

    sources = [
        {"title": c.title, "score": round(s, 3)}
        for c, s in results
    ]

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()