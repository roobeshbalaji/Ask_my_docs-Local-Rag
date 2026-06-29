"""
Query transformation techniques for improved retrieval.

- HyDE: generate a hypothetical answer, embed that for vector search
- Multi-Query: rephrase the question multiple ways, retrieve for each
- Adaptive: classify whether the query needs retrieval at all
"""
import re
from langchain_ollama import ChatOllama


def generate_hyde_document(question: str, llm: ChatOllama) -> str:
    """
    Generate a hypothetical answer paragraph to use as the vector search query.

    The hypothetical document shares vocabulary with real chunks in the corpus,
    which dramatically improves vector recall for conceptual questions where the
    user's phrasing doesn't match the chunk text.

    BM25 should still use the original question (it needs real keywords).
    """
    prompt = (
        "Write a short, factual paragraph (3-5 sentences) that directly answers "
        "the following question. Write as if you are quoting from an authoritative "
        "technical document. Do not hedge or say 'I don't know' — just write what "
        "a correct answer would look like.\n\n"
        f"Question: {question}"
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


def generate_multi_queries(question: str, llm: ChatOllama, n: int = 3) -> list[str]:
    """
    Generate n alternative phrasings of the question for broader retrieval.

    Each variant approaches the question from a different angle, increasing the
    chance that at least one phrasing matches the vocabulary in relevant chunks.
    The original question is always included in the output.
    """
    prompt = (
        f"Generate {n} different versions of the following question to help "
        "retrieve relevant documents from a vector database. Each version should "
        "approach the question from a different angle or use different terminology.\n\n"
        f"Original question: {question}\n\n"
        f"Output ONLY the {n} questions, one per line, numbered 1-{n}. "
        "Do not include any other text."
    )

    response = llm.invoke([{"role": "user", "content": prompt}])

    lines = response.content.strip().split("\n")
    queries = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned and len(cleaned) > 10:
            queries.append(cleaned)

    queries = queries[:n]

    if question not in queries:
        queries.insert(0, question)

    return queries


def classify_query_need(question: str, llm: ChatOllama) -> str:
    """
    Classify whether a query needs document retrieval or can be answered directly.

    Returns "retrieve" or "direct".

    For specialized corpora, most queries will route to "retrieve". This is most
    valuable when the system handles a mix of general and document-specific questions.
    """
    prompt = (
        "You are a query router for a document question-answering system. "
        "The document database contains financial reports (10-K filings) and "
        "technical research papers about RAG (Retrieval-Augmented Generation).\n\n"
        "Classify whether the following question requires searching the document "
        "database, or if it can be answered from general knowledge.\n\n"
        "Output ONLY one word: 'retrieve' or 'direct'.\n\n"
        f"Question: {question}"
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    answer = response.content.strip().lower()

    if "direct" in answer:
        return "direct"
    return "retrieve"
