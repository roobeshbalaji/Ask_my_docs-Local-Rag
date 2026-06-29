"""
Evaluate the RAG pipeline against the curated test set using RAGAS metrics.

Usage:
    python -m eval.run_eval              # run full eval
    python -m eval.run_eval --quick      # run first 5 questions only
"""
import sys
import json
import time
from pathlib import Path

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Add project root to path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OLLAMA_MODEL, EMBED_MODEL, RERANK_TOP_N
from pipeline.retrieval import load_retrievers, retrieve
from pipeline.fusion import reciprocal_rank_fusion
from pipeline.rerank import load_reranker, rerank
from pipeline.generation import build_llm, generate_answer
from eval.test_set import TEST_SET


def run_pipeline(question: str, vector_retriever, bm25_retriever, reranker, llm):
    """Run the full RAG pipeline and return (answer, contexts)."""
    vec_results, bm25_results = retrieve(question, vector_retriever, bm25_retriever)
    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    top_docs, top_scores = rerank(question, fused, reranker, RERANK_TOP_N)
    result = generate_answer(question, top_docs, llm)
    contexts = [doc.page_content for doc in top_docs]
    return result["answer"], contexts


def main():
    quick = "--quick" in sys.argv
    test_items = TEST_SET[:5] if quick else TEST_SET

    print("🚀 Loading retrieval components...")
    vector_retriever, bm25_retriever, _ = load_retrievers()
    reranker_model = load_reranker()
    llm = build_llm()

    print(f"\n📊 Running evaluation on {len(test_items)} questions...\n")

    samples = []
    results_log = []
    start_time = time.time()

    for i, item in enumerate(test_items, 1):
        q = item["question"]
        print(f"  [{i}/{len(test_items)}] {q}")

        answer, contexts = run_pipeline(
            q, vector_retriever, bm25_retriever, reranker_model, llm
        )

        print(f"           Answer: {answer[:100]}...")

        sample = SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
            reference=item["ground_truth"],
        )
        samples.append(sample)

        results_log.append({
            "question": q,
            "answer": answer,
            "ground_truth": item["ground_truth"],
            "category": item["category"],
            "source": item["source"],
            "num_contexts": len(contexts),
        })

    print(f"\n⏱️  Pipeline took {time.time() - start_time:.1f}s for {len(test_items)} questions")

    print("\n🔬 Computing RAGAS metrics (this may take a few minutes with local LLM)...")

    eval_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=8192))
    eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    dataset = EvaluationDataset(samples=samples)

    eval_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
        show_progress=True,
    )

    print("\n" + "=" * 60)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, score in eval_result.items():
        if isinstance(score, (int, float)):
            print(f"  {metric_name:.<40} {score:.4f}")

    print("=" * 60)

    output_path = Path(__file__).parent / "results.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions": len(test_items),
        "metrics": {k: v for k, v in eval_result.items() if isinstance(v, (int, float))},
        "per_question": results_log,
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\n💾 Full results saved to {output_path}")


if __name__ == "__main__":
    main()
