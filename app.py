import streamlit as st
from src.ingestion import process_pdf_document, clear_vector_store
from src.rag import stream_financial_query
from src.helpers import reset_app, render_citations

st.set_page_config(page_title="Financial Report RAG MVP", page_icon="📈", layout="wide")

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Stored only in session memory.",
        placeholder="sk-...",
    )
    st.divider()

    st.subheader("📄 Document Ingestion")
    # Using dynamic key so the file uploader clears out visually on reset
    uploaded_file = st.file_uploader(
        "Upload Financial PDF (e.g. 10-K)",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=10,
        help="Upload a single PDF report.",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )
    st.divider()

    st.subheader("📊 Backend Status")
    if not openai_api_key:
        st.warning("⚠️ Enter your OpenAI API key to get started.")
    elif uploaded_file is None and st.session_state.vector_store is None:
        st.info("ℹ️ Upload a PDF report above.")
    elif st.session_state.vector_store is not None:
        st.success("🟢 Ready to query")
        st.caption(f"**Active Document:** {st.session_state.processed_file}")
        st.caption(f"**Indexed Chunks:** {st.session_state.total_chunks}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset All", type="primary", use_container_width=True):
                reset_app()
                st.rerun()


# ----------------- INGESTION TRIGGER -----------------
if uploaded_file and openai_api_key:
    if st.session_state.processed_file != uploaded_file.name:
        with st.sidebar:
            with st.status(
                "Ingesting and indexing document...", expanded=True
            ) as status:
                st.write("1. Chunking pages & generating embeddings...")

                # If an old collection was loaded, clear it first
                if st.session_state.vector_store is not None:
                    clear_vector_store(st.session_state.vector_store)

                vector_store, bm25_retriever, chunk_count = process_pdf_document(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    api_key=openai_api_key,
                )

                st.session_state.vector_store = vector_store
                st.session_state.bm25_retriever = bm25_retriever
                st.session_state.total_chunks = chunk_count
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []

                status.update(
                    label="Document indexed successfully!",
                    state="complete",
                    expanded=False,
                )
        st.rerun()


# ----------------- CHAT PRESENTATION -----------------
st.title("📈 Financial Report Analyst")
st.caption(
    "Ask questions about your uploaded financial document. Every response includes verifiable citations."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

if prompt := st.chat_input("Ask a question about this report..."):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    if not st.session_state.vector_store:
        st.error("Please upload a PDF document before asking questions.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        # We only want the spinner to show during retrieval, not during generation
        with st.spinner("Retrieving context..."):
            answer_stream, docs = stream_financial_query(
                query=prompt,
                vector_store=st.session_state.vector_store,
                bm25_retriever=st.session_state.bm25_retriever,
                api_key=openai_api_key,
            )

        # Stream the output directly to the UI
        # st.write_stream automatically consumes the generator and returns the final full string
        full_answer = st.write_stream(answer_stream)

        # Render citations below the streaming text
        render_citations(docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_stream, "citations": docs}
    )
