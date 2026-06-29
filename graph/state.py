from typing import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str
    transformed_queries: list[str]
    vector_query: str
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    rerank_scores: list[float]
    relevance_verdict: str
    answer: str
    sources: list[str]
    retrieval_attempts: int
    route: str
