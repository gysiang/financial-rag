from typing import List, Tuple, Iterator
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def retrieve_relevant_chunks(
    vector_store: Chroma, bm25_retriever: BM25Retriever, query: str, k: int = 4
) -> List[Document]:
    """Fetches the top-k most relevant document chunks based on semantic similarity."""
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever.k = k
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever.invoke(query)


def stream_grounded_answer(
    query: str,
    retrieved_docs: List[Document],
    api_key: str,
    model_name: str = "gpt-6-luna",
) -> Iterator[str]:
    """Yields the LLM response token-by-token."""
    context_text = "\n\n---\n\n".join(
        [
            f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    system_prompt = (
        "You are an expert financial analyst. Use the provided context extracted from a financial report "
        "to answer the question accurately and concisely.\n\n"
        "Rules:\n"
        "1. Only base your answers on the provided context.\n"
        "2. If the context does not contain the answer, explicitly state that you cannot find it in the report.\n"
        "3. When referencing financial numbers, cite the page number directly in the text (e.g., [Page 42]).\n\n"
        f"Context:\n{context_text}"
    )

    llm = ChatOpenAI(model=model_name, openai_api_key=api_key)

    # Use .stream() instead of .invoke()
    for chunk in llm.stream(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    ):
        # Yield only the string content so Streamlit can render it smoothly
        if chunk.content:
            yield chunk.content


def stream_financial_query(
    query: str,
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    api_key: str,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """Convenience pipeline function executing retrieval followed by generation."""
    docs = retrieve_relevant_chunks(vector_store, bm25_retriever, query, k=k)
    answer_stream = stream_grounded_answer(query, docs, api_key)
    return answer_stream, docs
