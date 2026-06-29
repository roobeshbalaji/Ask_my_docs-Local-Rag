from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANK_TOP_N, RERANK_BATCH


def load_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker model."""
    print("🧠 Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    print("   ✅ Reranker ready")
    return reranker


def rerank(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_n: int = RERANK_TOP_N,
) -> tuple[list[Document], list[float]]:
    """
    Score (query, chunk) pairs with the cross-encoder and return the top N.

    Returns (top_docs, top_scores) so callers can inspect scores for CRAG.
    """
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH)

    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in reranked[:top_n]]
    top_scores = [float(score) for score, _ in reranked[:top_n]]

    return top_docs, top_scores
