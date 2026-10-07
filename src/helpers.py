import streamlit as st
from src.ingestion import clear_vector_store


def reset_app():
    """Wipes ChromaDB and resets all session variables to fresh initial state."""
    clear_vector_store(st.session_state.vector_store)
    st.session_state.vector_store = None
    st.session_state.vector_store = None
    st.session_state.bm25_retriever = None
    st.session_state.processed_file = None
    st.session_state.total_chunks = 0
    st.session_state.messages = []
    # Incrementing the key forces Streamlit to re-mount a completely clean uploader widget
    st.session_state.uploader_key += 1


# ----------------- UI HELPERS -----------------
def render_citations(documents):
    """Renders a collapsible section detailing cited sources and pages."""
    with st.expander("🔍 View Retrieved Sources & Citations"):
        for i, doc in enumerate(documents, start=1):
            page_num = doc.metadata.get("page", 0) + 1
            st.markdown(f"**Source {i} — Page {page_num}:**")
            st.caption(doc.page_content)
            st.divider()
