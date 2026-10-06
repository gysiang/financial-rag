import os
import tempfile
from typing import Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def process_pdf_document(
    file_bytes: bytes, filename: str, api_key: str
) -> Tuple[Chroma, int]:
    """Saves uploaded bytes to a temp file, chunks pages, and indexes them in Chroma."""
    suffix = os.path.splitext(filename)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=api_key
        )
        vector_store = Chroma.from_documents(splits, embeddings)
        return vector_store, len(splits)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def clear_vector_store(vector_store: Optional[Chroma]) -> None:
    """Explicitly deletes the collection from Chroma memory."""
    if vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception:
            pass
