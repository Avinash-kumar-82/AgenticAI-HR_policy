from functools import lru_cache
from typing import Sequence

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.core.config import get_settings


def _namespace() -> str | None:
    """Optional Pinecone namespace; must be the same for ingest and retrieval."""
    return getattr(get_settings(), "pinecone_namespace", None) or None


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and cache the Hugging Face embedding model."""
    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """Create and cache the Pinecone vector store."""
    settings = get_settings()

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    if not settings.pinecone_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is missing")

    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
        namespace=_namespace(),
    )


def get_retriever():
    """Return a retriever for similarity-based document search."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


def add_documents(chunks: Sequence[Document]) -> list[str]:
    """Embed and store document chunks in Pinecone."""
    if not chunks:
        return []

    return get_vectorstore().add_documents(documents=list(chunks))


def warmup() -> None:
    """Load the embedding model and Pinecone client at startup."""
    get_embeddings().embed_query("warmup")
    get_vectorstore()


def index_stats() -> dict:
    """Debug helper: show vector counts per namespace and index dimension."""
    from pinecone import Pinecone

    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    stats = index.describe_index_stats()
    return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)