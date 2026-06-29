from pipeline.retrieval import load_retrievers
from pipeline.rerank import load_reranker
from pipeline.generation import build_llm
from graph.agent import build_graph

from config import (
    ENABLE_HYDE,
    ENABLE_MULTI_QUERY,
    ENABLE_ADAPTIVE,
    ENABLE_CRAG,
)


if __name__ == "__main__":
    print("🚀 Booting retrieval components — this takes ~10-20s on first run...")
    vector_retriever, bm25_retriever, _ = load_retrievers()
    reranker = load_reranker()
    llm = build_llm()

    print("🔗 Building LangGraph agent pipeline...")
    graph = build_graph(vector_retriever, bm25_retriever, reranker, llm)

    flags = []
    if ENABLE_ADAPTIVE:
        flags.append("adaptive")
    if ENABLE_HYDE:
        flags.append("HyDE")
    if ENABLE_MULTI_QUERY:
        flags.append("multi-query")
    if ENABLE_CRAG:
        flags.append("CRAG")
    print(f"   Active features: {', '.join(flags) or 'none'}")
    print("\n✅ All components ready. Type your question or 'quit' to exit.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = graph.invoke({"question": question, "retrieval_attempts": 0})

        print(f"\n{'='*60}")
        print(f"📝 Answer:\n{result.get('answer', 'No answer generated')}")
        sources = result.get("sources", [])
        if sources:
            print(f"\n📂 Sources: {', '.join(sources)}")
        print(f"{'='*60}\n")
