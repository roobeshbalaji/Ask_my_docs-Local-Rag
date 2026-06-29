"""
Search EVERY chunk in Qdrant for a literal substring, ignoring all the
retrieval/reranking machinery. This tells us whether the revenue figure even
exists in the database, and if so, what the chunk around it looks like.
"""
import sys
from qdrant_client import QdrantClient

from config import DB_PATH

# Try a few spellings of NVIDIA's FY2024 total revenue ($60,922 million).
# Pass your own terms on the command line to override:  python search_db.py 60,922 60922
TERMS = sys.argv[1:] or ["60,922", "60922", "60.9 billion", "Total revenue"]

client = QdrantClient(path=DB_PATH)

all_points = []
offset = None
while True:
    batch, next_offset = client.scroll(
        collection_name="local_docs",
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(batch)
    if next_offset is None:
        break
    offset = next_offset

print(f"Scanned {len(all_points)} chunks.\n")

for term in TERMS:
    hits = [
        p for p in all_points
        if term.lower() in p.payload.get("page_content", "").lower()
    ]
    print(f"=== '{term}': {len(hits)} chunk(s) ===")
    for p in hits[:3]:
        src = p.payload.get("metadata", {}).get("source", "?")
        content = p.payload.get("page_content", "")
        print(f"\n--- source={src} ---\n{content}\n")
    print()
