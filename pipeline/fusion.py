from langchain_core.documents import Document

from config import RRF_K, SEARCH_K


def reciprocal_rank_fusion(
    results_a: list[Document],
    results_b: list[Document],
) -> list[Document]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF only cares about rank position, not raw scores, so it works across
    retrievers with incompatible score scales (cosine vs BM25).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in [results_a, results_b]:
        for rank, doc in enumerate(result_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RRF_K)
            doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:SEARCH_K]]
