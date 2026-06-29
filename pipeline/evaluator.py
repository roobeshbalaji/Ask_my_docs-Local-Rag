"""
Corrective RAG (CRAG) evaluator.

Two-tier chunk relevance evaluation:
1. Score-based fast check: if all reranker scores are below threshold, retrieval failed
2. LLM-based check: for ambiguous scores, ask the LLM to classify relevance

Returns a verdict that controls whether the pipeline should retry retrieval.
"""
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

SCORE_FAIL_THRESHOLD = 0.0
SCORE_AMBIGUOUS_UPPER = 2.0
MIN_POSITIVE_RATIO = 0.5


def evaluate_relevance_fast(top_scores: list[float]) -> str:
    """
    Quick heuristic using cross-encoder scores.

    Checks both the best score AND how many chunks are actually relevant.
    If only 1 out of 4 chunks has a positive score, the retrieval is weak
    even if that one chunk scored well.

    Returns:
        "correct"   — scores look good, proceed to generation
        "ambiguous" — scores are borderline, needs LLM verification
        "incorrect" — scores are all bad, retrieval likely failed
    """
    if not top_scores:
        return "incorrect"

    max_score = max(top_scores)
    positive_count = sum(1 for s in top_scores if s > SCORE_FAIL_THRESHOLD)
    positive_ratio = positive_count / len(top_scores)

    if max_score < SCORE_FAIL_THRESHOLD:
        return "incorrect"

    if positive_ratio < MIN_POSITIVE_RATIO:
        return "ambiguous"

    if max_score < SCORE_AMBIGUOUS_UPPER:
        return "ambiguous"

    return "correct"


def evaluate_relevance_llm(
    question: str,
    chunks: list[Document],
    llm: ChatOllama,
) -> str:
    """
    Ask the LLM to classify whether retrieved chunks are relevant to the question.

    Returns "correct", "incorrect", or "ambiguous".
    """
    chunk_texts = "\n\n---\n\n".join(
        f"Chunk {i+1}: {doc.page_content[:500]}"
        for i, doc in enumerate(chunks)
    )

    prompt = (
        "You are evaluating whether retrieved document chunks are relevant to a question.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{chunk_texts}\n\n"
        "Do these chunks contain information that can help answer the question?\n"
        "Output ONLY one word: 'correct' (chunks are relevant), "
        "'incorrect' (chunks are not relevant), or "
        "'ambiguous' (partially relevant or unclear)."
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    answer = response.content.strip().lower()

    if "incorrect" in answer:
        return "incorrect"
    elif "ambiguous" in answer:
        return "ambiguous"
    return "correct"


def evaluate_retrieval(
    question: str,
    top_docs: list[Document],
    top_scores: list[float],
    llm: ChatOllama,
) -> str:
    """
    Two-tier evaluation: fast score check, then LLM fallback if ambiguous.

    Returns "correct", "incorrect", or "ambiguous".
    """
    fast_verdict = evaluate_relevance_fast(top_scores)

    if fast_verdict == "correct":
        print("   ✅ CRAG: scores look good (fast check)")
        return "correct"

    if fast_verdict == "incorrect":
        print("   ❌ CRAG: all scores below threshold (fast check)")
        return "incorrect"

    # Ambiguous — ask the LLM
    print("   🤔 CRAG: ambiguous scores, running LLM evaluation...")
    llm_verdict = evaluate_relevance_llm(question, top_docs, llm)
    print(f"   CRAG LLM verdict: {llm_verdict}")
    return llm_verdict
