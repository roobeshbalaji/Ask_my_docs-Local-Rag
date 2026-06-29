"""
LangGraph agentic RAG pipeline.

Wraps all pipeline modules into a state machine with conditional edges:
- Adaptive routing (skip retrieval for simple questions)
- Query transformation (HyDE / multi-query)
- Hybrid retrieval + RRF + reranking
- CRAG evaluation with retry loop
- Answer generation
"""
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, START, END

from graph.state import RAGState
from pipeline.query_transform import (
    classify_query_need,
    generate_hyde_document,
    generate_multi_queries,
)
from pipeline.retrieval import retrieve
from pipeline.fusion import reciprocal_rank_fusion
from pipeline.rerank import rerank
from pipeline.evaluator import evaluate_retrieval
from pipeline.generation import generate_answer, generate_direct_answer

from config import (
    RERANK_TOP_N,
    ENABLE_HYDE,
    ENABLE_MULTI_QUERY,
    ENABLE_ADAPTIVE,
    ENABLE_CRAG,
)

MAX_RETRIEVAL_ATTEMPTS = 2

# These are set at build time via build_graph()
_vector_retriever = None
_bm25_retriever = None
_reranker = None
_llm = None


def classify_node(state: RAGState) -> dict:
    """Decide whether the query needs retrieval or can be answered directly."""
    question = state["question"]

    if ENABLE_ADAPTIVE:
        route = classify_query_need(question, _llm)
        print(f"   🔀 Route: {route}")
    else:
        route = "retrieve"

    return {"route": route}


def transform_node(state: RAGState) -> dict:
    """Apply query transformations: HyDE and/or multi-query."""
    question = state["question"]
    vector_query = question
    queries = [question]

    if ENABLE_HYDE:
        print("   🔮 Generating HyDE document...")
        vector_query = generate_hyde_document(question, _llm)
        print(f"   HyDE: {vector_query[:100]}...")

    if ENABLE_MULTI_QUERY:
        print("   🔄 Generating multi-query variants...")
        queries = generate_multi_queries(question, _llm)
        for i, q in enumerate(queries, 1):
            print(f"      Q{i}: {q}")

    return {
        "transformed_queries": queries,
        "vector_query": vector_query,
    }


def retrieve_node(state: RAGState) -> dict:
    """Run hybrid retrieval for all query variants."""
    queries = state.get("transformed_queries", [state["question"]])
    vector_query = state.get("vector_query", state["question"])
    attempt = state.get("retrieval_attempts", 0) + 1

    print(f"\n🔍 Running hybrid search (attempt {attempt}/{MAX_RETRIEVAL_ATTEMPTS})...")

    all_vec: list[Document] = []
    all_bm25: list[Document] = []

    for i, rq in enumerate(queries):
        vq = vector_query if i == 0 and ENABLE_HYDE else rq
        vec_r, bm25_r = retrieve(vq, _vector_retriever, _bm25_retriever)
        all_vec.extend(vec_r)
        all_bm25.extend(bm25_r)

    seen = set()
    deduped_vec = [d for d in all_vec if d.page_content not in seen and not seen.add(d.page_content)]

    seen2 = set()
    deduped_bm25 = [d for d in all_bm25 if d.page_content not in seen2 and not seen2.add(d.page_content)]

    print(f"   Vector hits: {len(deduped_vec)} | BM25 hits: {len(deduped_bm25)}")

    return {
        "retrieved_docs": deduped_vec + deduped_bm25,
        "retrieval_attempts": attempt,
    }


def fuse_rerank_node(state: RAGState) -> dict:
    """RRF fusion + cross-encoder reranking."""
    all_docs = state["retrieved_docs"]
    question = state["question"]

    mid = len(all_docs) // 2
    vec_docs = all_docs[:mid]
    bm25_docs = all_docs[mid:]

    fused = reciprocal_rank_fusion(vec_docs, bm25_docs)
    print(f"   After RRF fusion: {len(fused)} candidates")

    top_docs, top_scores = rerank(question, fused, _reranker, RERANK_TOP_N)

    print(f"   After reranking: kept top {len(top_docs)} chunks")
    for i, (score, doc) in enumerate(zip(top_scores, top_docs), 1):
        src = doc.metadata.get("source", "?")
        print(f"   [{i}] score={score:.3f}  source={src}")

    return {
        "reranked_docs": top_docs,
        "rerank_scores": top_scores,
    }


def evaluate_node(state: RAGState) -> dict:
    """CRAG: evaluate retrieval quality."""
    if not ENABLE_CRAG:
        return {"relevance_verdict": "correct"}

    verdict = evaluate_retrieval(
        state["question"],
        state["reranked_docs"],
        state["rerank_scores"],
        _llm,
    )
    return {"relevance_verdict": verdict}


def generate_node(state: RAGState) -> dict:
    """Generate the final answer from reranked chunks."""
    result = generate_answer(state["question"], state["reranked_docs"], _llm)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }


def direct_answer_node(state: RAGState) -> dict:
    """Answer without retrieval."""
    result = generate_direct_answer(state["question"], _llm)
    return {
        "answer": result["answer"],
        "sources": [],
    }


# --- Routing functions ---

def route_after_classify(state: RAGState) -> str:
    return "direct_answer" if state.get("route") == "direct" else "transform"


def route_after_evaluate(state: RAGState) -> str:
    verdict = state.get("relevance_verdict", "correct")
    attempts = state.get("retrieval_attempts", 0)

    if verdict == "correct" or attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate"
    else:
        print(f"   🔁 CRAG: retrieval {verdict}, reformulating for retry...")
        return "retry_transform"


def retry_transform_node(state: RAGState) -> dict:
    """Re-transform the query for a CRAG retry (uses multi-query only, no HyDE)."""
    question = state["question"]
    queries = generate_multi_queries(question, _llm)
    for i, q in enumerate(queries, 1):
        print(f"      Retry Q{i}: {q}")

    return {
        "transformed_queries": queries,
        "vector_query": question,
    }


# --- Graph construction ---

def build_graph(vector_retriever, bm25_retriever, reranker: CrossEncoder, llm: ChatOllama):
    """Build and compile the LangGraph state machine."""
    global _vector_retriever, _bm25_retriever, _reranker, _llm
    _vector_retriever = vector_retriever
    _bm25_retriever = bm25_retriever
    _reranker = reranker
    _llm = llm

    workflow = StateGraph(RAGState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("transform", transform_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("fuse_rerank", fuse_rerank_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retry_transform", retry_transform_node)

    workflow.add_edge(START, "classify")

    workflow.add_conditional_edges("classify", route_after_classify, {
        "transform": "transform",
        "direct_answer": "direct_answer",
    })

    workflow.add_edge("transform", "retrieve")
    workflow.add_edge("retrieve", "fuse_rerank")
    workflow.add_edge("fuse_rerank", "evaluate")

    workflow.add_conditional_edges("evaluate", route_after_evaluate, {
        "generate": "generate",
        "retry_transform": "retry_transform",
    })

    workflow.add_edge("retry_transform", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()
