"""
Diagnostic: print the FULL text of the chunks being passed to the LLM.

If the revenue figure is visible in the output below, the problem is in
generation (prompt / model / num_ctx). If it is NOT visible, the problem is
retrieval/chunking and no LLM tuning will fix it.
"""
import sys
from pipeline.retrieval import load_retrievers, retrieve
from pipeline.fusion import reciprocal_rank_fusion
from pipeline.rerank import load_reranker, rerank

from config import RERANK_TOP_N

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "what was NVIDIA's total revenue in fiscal 2024?"

vector_retriever, bm25_retriever, _ = load_retrievers()
reranker_model = load_reranker()

vec_results, bm25_results = retrieve(QUESTION, vector_retriever, bm25_retriever)
fused = reciprocal_rank_fusion(vec_results, bm25_results)
top_docs, top_scores = rerank(QUESTION, fused, reranker_model, RERANK_TOP_N)

print("\n" + "=" * 80)
print("FULL TEXT OF TOP CHUNKS SENT TO THE LLM")
print("=" * 80)
for i, (doc, score) in enumerate(zip(top_docs, top_scores), 1):
    src = doc.metadata.get("source", "?")
    print(f"\n----- CHUNK [{i}]  source={src}  score={score:.3f}  len={len(doc.page_content)} chars -----")
    print(doc.page_content)
