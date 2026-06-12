"""
rag_tab.py
──────────
Streamlit UI for the RAG (Retrieval-Augmented Generation) chatbot tab.

Drop this into app.py as a new tab:

    tab1, tab2 = st.tabs(["📊 Data Analysis", "🗂️ Chat with Documents"])
    with tab1:
        # ... existing app content ...
    with tab2:
        import rag_tab
        rag_tab.render(ollama_model)

No business logic lives here — all RAG logic is in rag.py.
"""

from __future__ import annotations

import streamlit as st

import rag as rg
import chat as ch  # Imported to fetch available chat models

# ──────────────────────────────────────────────────────────────────────────────
#  Session-state keys
# ──────────────────────────────────────────────────────────────────────────────

_KEY_COLLECTION  = "rag_collection"
_KEY_HISTORY     = "rag_history"
_KEY_FILE_NAME   = "rag_file_name"
_KEY_CHUNK_COUNT = "rag_chunk_count"
_KEY_SHOW_CTX    = "rag_show_context"


def _init_state() -> None:
    defaults = {
        _KEY_COLLECTION:  None,
        _KEY_HISTORY:     [],
        _KEY_FILE_NAME:   None,
        _KEY_CHUNK_COUNT: 0,
        _KEY_SHOW_CTX:    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
#  Quick questions for documents
# ──────────────────────────────────────────────────────────────────────────────

_QUICK_QUESTIONS = [
    "📋 Summarise the key points of this document",
    "🔍 What are the main topics covered?",
    "📈 Are there any trends or patterns mentioned?",
    "⚠️ What are the risks or issues highlighted?",
    "🔗 What conclusions or recommendations are made?",
    "🤔 What questions does this document leave unanswered?",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Core render function
# ──────────────────────────────────────────────────────────────────────────────

def render(chat_model: str = rg.DEFAULT_CHAT_MODEL) -> None:
    """Render the full RAG tab. Call from app.py inside a `with tab:` block."""
    _init_state()

    st.subheader("🗂️ Chat with Your Documents")
    st.caption(
        "Upload a document — PDF, CSV, DOCX, TXT, JSON, or HTML — "
        "and ask questions about it. Everything runs locally via Ollama."
    )

    # ── Sidebar: Settings ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🧩 RAG Settings")

    # Fetch available models for the chat dropdown
    available_chat_models = ch.ollama_available_models()
    if not available_chat_models:
        # Fallback if Ollama is unreachable
        available_chat_models = [chat_model, "llama3.2:latest", "llama3.2:1b"]
    
    # Try to set the default index to the one passed from the main app
    try:
        chat_index = available_chat_models.index(chat_model)
    except ValueError:
        chat_index = 0

    # Overwrite the variable with the user's actual selection from the sidebar
    chat_model = st.sidebar.selectbox(
        "Chat Model",
        options=available_chat_models,
        index=chat_index,
        help="The model used to generate answers from the document context.",
        key="rag_chat_model_select",
    )

    embed_options = rg.available_embed_models()
    embed_model = st.sidebar.selectbox(
        "Embedding Model",
        options=embed_options,
        index=0,
        help="Used to convert text chunks into vectors.",
        key="rag_embed_model",
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Parameters")
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (characters)", 
        min_value=100, max_value=2000, value=500, step=100,
        help="How large each block of text should be when indexing the document. Larger chunks give the model broader context, smaller chunks provide more specific retrieval."
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", 
        min_value=0, max_value=500, value=100, step=50,
        help="How many characters should overlap between chunks to prevent splitting ideas or sentences in half."
    )
    top_k = st.sidebar.slider(
        "Top-K Retrieved Chunks", 
        min_value=1, max_value=20, value=10,
        help="How many chunks of text to retrieve from the document and send to the AI for each question."
    )
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="0.0 makes the model strictly factual based on the text. Higher values make it more creative and descriptive."
    )

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload a Document")

    uploaded = st.file_uploader(
        "Supported formats: PDF, CSV, DOCX, TXT, JSON, HTML",
        type=[ext.lstrip(".") for ext in rg.SUPPORTED_EXTENSIONS],
        key="rag_uploader",
    )

    col_index, col_reset = st.columns([3, 1])

    with col_index:
        index_btn = st.button(
            "⚡ Index Document",
            key="rag_index",
            use_container_width=True,
            disabled=uploaded is None,
            type="primary",
        )

    with col_reset:
        reset_btn = st.button(
            "🗑️ Reset",
            key="rag_reset",
            use_container_width=True,
        )

    # ── Index action ──────────────────────────────────────────────────────────
    if index_btn and uploaded is not None:
        with st.spinner(f"📖 Parsing & indexing *{uploaded.name}* …"):
            try:
                collection, n_chunks = rg.build_index(
                    uploaded, 
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                st.session_state[_KEY_COLLECTION]  = collection
                st.session_state[_KEY_FILE_NAME]   = uploaded.name
                st.session_state[_KEY_CHUNK_COUNT] = n_chunks
                st.session_state[_KEY_HISTORY]     = []
                st.success(
                    f"✅ Indexed **{uploaded.name}** into **{n_chunks}** chunks. "
                    "Ask your first question below!"
                )
            except Exception as exc:
                st.error(f"❌ Indexing failed: {exc}")

    # ── Reset action ──────────────────────────────────────────────────────────
    if reset_btn:
        for k in (_KEY_COLLECTION, _KEY_FILE_NAME, _KEY_CHUNK_COUNT):
            st.session_state[k] = None if k != _KEY_CHUNK_COUNT else 0
        st.session_state[_KEY_HISTORY] = []
        st.rerun()

    # ── Status banner ─────────────────────────────────────────────────────────
    collection = st.session_state[_KEY_COLLECTION]

    if collection is not None:
        fname  = st.session_state[_KEY_FILE_NAME]
        nchunk = st.session_state[_KEY_CHUNK_COUNT]

        st.info(
            f"📄 **Active document:** {fname} &nbsp;|&nbsp; "
            f"**{nchunk}** chunks indexed &nbsp;|&nbsp; "
            f"Chat model: `{chat_model}` &nbsp;|&nbsp; "
            f"Embed model: `{embed_model}`"
        )

        st.markdown("---")

        # ── Quick questions ────────────────────────────────────────────────
        st.markdown("**✨ Quick Questions:**")
        q_cols = st.columns(2)
        for i, q in enumerate(_QUICK_QUESTIONS):
            if q_cols[i % 2].button(q, key=f"rag_qq_{i}", use_container_width=True):
                _send(q, collection, chat_model, embed_model, top_k, temperature)
                st.rerun()

        st.markdown("")

        # ── Show / hide retrieved context toggle ──────────────────────────
        st.session_state[_KEY_SHOW_CTX] = st.toggle(
            "🔍 Show retrieved context chunks",
            value=st.session_state[_KEY_SHOW_CTX],
            key="rag_ctx_toggle",
        )

        # ── Conversation history ───────────────────────────────────────────
        chat_box = st.container(height=450)
        with chat_box:
            for msg in st.session_state[_KEY_HISTORY]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    # Show retrieved chunks if toggle is on and stored
                    if (
                        msg["role"] == "assistant"
                        and st.session_state[_KEY_SHOW_CTX]
                        and msg.get("chunks")
                    ):
                        with st.expander("📎 Retrieved context chunks"):
                            for j, chunk in enumerate(msg["chunks"], 1):
                                st.markdown(
                                    f"**Chunk {j}:**\n\n"
                                    f"> {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
                                )

        # ── User input ────────────────────────────────────────────────────
        user_input = st.chat_input(
            f"Ask anything about {st.session_state[_KEY_FILE_NAME]} …"
        )
        if user_input:
            _send(user_input, collection, chat_model, embed_model, top_k, temperature)
            st.rerun()

        # ── Clear chat ────────────────────────────────────────────────────
        if st.session_state[_KEY_HISTORY]:
            if st.button("🗑️ Clear Chat History", key="rag_clear", use_container_width=True):
                st.session_state[_KEY_HISTORY] = []
                st.rerun()

    else:
        # No document indexed yet
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem; color: #888;">
                <div style="font-size: 3rem;">📂</div>
                <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                    Upload a document above and click <b>⚡ Index Document</b> to get started.
                </p>
                <p style="font-size: 0.9rem;">
                    Supported: PDF · CSV · DOCX · TXT · JSON · HTML
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Setup tips ─────────────────────────────────────────────────────
        with st.expander("🛠️ Setup Guide — first time?"):
            st.markdown(
                """
                **1. Make sure Ollama is running**
                ```bash
                ollama serve
                ```

                **2. Pull an embedding model** (recommended)
                ```bash
                ollama pull nomic-embed-text
                ```

                **3. Pull a chat model** (if you haven't already)
                ```bash
                ollama pull llama3.2
                ```

                **4. Install Python dependencies**
                ```bash
                pip install chromadb pymupdf python-docx beautifulsoup4 ollama
                ```

                **5. Upload your document and click ⚡ Index Document**
                """
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _send(
    question: str,
    collection,
    chat_model: str,
    embed_model: str,
    top_k: int,
    temperature: float,
) -> None:
    """Append user message, call RAG, append assistant reply."""
    # Build history without the chunk metadata (clean roles only)
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state[_KEY_HISTORY]
    ]

    st.session_state[_KEY_HISTORY].append(
        {"role": "user", "content": question}
    )

    try:
        reply, chunks = rg.rag_chat(
            collection,
            question,
            history=clean_history,
            chat_model=chat_model,
            embed_model=embed_model,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as exc:
        reply  = f"⚠️ Error: {exc}\n\nMake sure Ollama is running."
        chunks = []

    st.session_state[_KEY_HISTORY].append(
        {"role": "assistant", "content": reply, "chunks": chunks}
    )