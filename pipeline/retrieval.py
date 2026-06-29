from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from config import EMBED_MODEL, DB_PATH, SEARCH_K


def load_retrievers() -> tuple:
    """
    Initialise vector and BM25 retrievers from the Qdrant collection.

    Returns (vector_retriever, bm25_retriever, qdrant_client)
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(path=DB_PATH)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="local_docs",
        embedding=embeddings,
    )

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K},
    )

    print("📖 Building BM25 index from Qdrant...")
    all_points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name="local_docs",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    all_docs = [
        Document(
            page_content=p.payload.get("page_content", ""),
            metadata=p.payload.get("metadata", {}),
        )
        for p in all_points
    ]

    if not all_docs:
        raise RuntimeError(
            "No documents found in Qdrant. "
            "Run ingest.py first, or check DB_PATH in your .env"
        )

    print(f"   ✅ {len(all_docs)} chunks indexed for BM25")
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=SEARCH_K)

    return vector_retriever, bm25_retriever, client


def retrieve(question: str, vector_retriever, bm25_retriever) -> tuple[list[Document], list[Document]]:
    """Run both retrievers and return (vector_results, bm25_results)."""
    vec_results = vector_retriever.invoke(question)
    bm25_results = bm25_retriever.invoke(question)
    return vec_results, bm25_results
