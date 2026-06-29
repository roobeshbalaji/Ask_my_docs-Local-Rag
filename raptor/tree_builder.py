"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Builds a hierarchical summary tree from the base chunks in Qdrant:
1. Load all chunks with their embeddings
2. Cluster via KMeans
3. Summarize each cluster with the LLM
4. Embed summaries and store back in Qdrant with level metadata
5. Repeat for higher levels

Run after ingest.py:
    python -m raptor.tree_builder
"""
import sys
import time
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DB_PATH, EMBED_MODEL, EMBED_BATCH, NUM_CTX,
    RAPTOR_RAPTOR_CHUNKS_PER_CLUSTER, RAPTOR_RAPTOR_MIN_CLUSTERS, RAPTOR_MODEL,
)


def load_chunks_with_embeddings(client: QdrantClient, level: int = 0):
    """Load all chunks at a given level from Qdrant, with embeddings."""
    all_points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name="local_docs",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    filtered = [
        p for p in all_points
        if p.payload.get("metadata", {}).get("level", 0) == level
    ]

    docs = []
    embeddings = []
    for p in filtered:
        docs.append(Document(
            page_content=p.payload.get("page_content", ""),
            metadata=p.payload.get("metadata", {}),
        ))
        vec = p.vector
        if isinstance(vec, dict):
            vec = list(vec.values())[0]
        embeddings.append(vec)

    return docs, np.array(embeddings) if embeddings else np.array([])


def cluster_documents(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster embeddings with KMeans. Returns cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def summarize_cluster(docs: list[Document], llm: ChatOllama) -> str:
    """Generate a summary of a cluster of document chunks."""
    combined = "\n\n---\n\n".join(
        doc.page_content[:800] for doc in docs[:10]
    )

    prompt = (
        "You are summarizing a cluster of related document chunks. "
        "Write a single coherent paragraph (4-6 sentences) that captures "
        "the key information and themes across all the chunks below. "
        "Be specific — include names, numbers, and technical terms.\n\n"
        f"Chunks:\n{combined}"
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


def build_level(
    client: QdrantClient,
    vector_store: QdrantVectorStore,
    llm: ChatOllama,
    source_level: int,
):
    """Build one level of the RAPTOR tree from the previous level."""
    target_level = source_level + 1

    print(f"\n📊 Building level {target_level} summaries from level {source_level}...")
    docs, embeddings = load_chunks_with_embeddings(client, level=source_level)

    if len(docs) < RAPTOR_MIN_CLUSTERS * 2:
        print(f"   Only {len(docs)} docs at level {source_level} — too few to cluster further")
        return 0

    n_clusters = max(RAPTOR_MIN_CLUSTERS, len(docs) // RAPTOR_CHUNKS_PER_CLUSTER)
    print(f"   {len(docs)} documents → {n_clusters} clusters")

    labels = cluster_documents(embeddings, n_clusters)

    clusters: dict[int, list[Document]] = {}
    for doc, label in zip(docs, labels):
        clusters.setdefault(int(label), []).append(doc)

    summary_docs = []
    for cluster_id in sorted(clusters.keys()):
        cluster_docs = clusters[cluster_id]
        print(f"   Summarizing cluster {cluster_id + 1}/{n_clusters} ({len(cluster_docs)} chunks)...")

        summary_text = summarize_cluster(cluster_docs, llm)

        sources = list({
            d.metadata.get("source", "unknown")
            for d in cluster_docs
        })

        summary_doc = Document(
            page_content=summary_text,
            metadata={
                "source": ", ".join(sources),
                "level": target_level,
                "cluster_id": cluster_id,
                "type": "summary",
                "num_source_chunks": len(cluster_docs),
            },
        )
        summary_docs.append(summary_doc)

    print(f"   Storing {len(summary_docs)} level-{target_level} summaries...")
    for i in range(0, len(summary_docs), EMBED_BATCH):
        vector_store.add_documents(summary_docs[i : i + EMBED_BATCH])

    print(f"   ✅ Level {target_level} complete: {len(summary_docs)} summaries")
    return len(summary_docs)


def main():
    start = time.time()

    print("🌳 RAPTOR Tree Builder")
    print("=" * 50)

    client = QdrantClient(path=DB_PATH)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="local_docs",
        embedding=embeddings,
    )
    llm = ChatOllama(
        model=RAPTOR_MODEL,
        temperature=0,
        num_ctx=NUM_CTX,
        num_gpu=99,
    )

    # Ensure base chunks have level=0 in metadata
    base_docs, _ = load_chunks_with_embeddings(client, level=0)
    if not base_docs:
        print("⚠️  No base chunks found (level=0). Checking if chunks lack level metadata...")
        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name="local_docs", limit=500, offset=offset,
                with_payload=True, with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        no_level = [p for p in all_points if "level" not in p.payload.get("metadata", {})]
        if no_level:
            print(f"   Found {len(no_level)} chunks without level metadata — treating as level 0")
        else:
            print("❌ No chunks found in Qdrant. Run ingest.py first.")
            return

    total_summaries = 0
    max_levels = 3

    for level in range(max_levels):
        count = build_level(client, vector_store, llm, source_level=level)
        total_summaries += count
        if count == 0:
            break

    elapsed = time.time() - start
    print(f"\n🌳 RAPTOR tree complete!")
    print(f"   Total summaries created: {total_summaries}")
    print(f"   Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
