import os
import re
import gc
import time
from tqdm import tqdm

from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_BATCH,
    EMBED_MODEL, DB_PATH, DATA_PATH,
)


def convert_pdf(converter: MarkItDown, file_path: str) -> str:
    """Convert a PDF to markdown using MarkItDown."""
    result = converter.convert(file_path)
    return result.text_content


def init_qdrant(embed_dim: int) -> QdrantClient:
    client = QdrantClient(path=DB_PATH)

    if client.collection_exists("local_docs"):
        client.delete_collection("local_docs")

    client.create_collection(
        collection_name="local_docs",
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    return client


def extract_doc_title(markdown: str, filename: str) -> str:
    """Extract a document title from the first markdown heading, or fall back to filename."""
    for line in markdown.split("\n")[:20]:
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return os.path.splitext(filename)[0]


def extract_section_headers(markdown: str) -> list[tuple[int, str]]:
    """Extract (char_offset, header_text) pairs from markdown headings."""
    headers = []
    offset = 0
    for line in markdown.split("\n"):
        match = re.match(r"^(#{1,4})\s+(.+)", line)
        if match:
            headers.append((offset, match.group(2).strip()))
        offset += len(line) + 1
    return headers


def find_nearest_header(char_offset: int, headers: list[tuple[int, str]]) -> str:
    """Find the most recent section header before the given character offset."""
    nearest = ""
    for hdr_offset, hdr_text in headers:
        if hdr_offset <= char_offset:
            nearest = hdr_text
        else:
            break
    return nearest


def enrich_chunks(
    chunks: list[Document],
    markdown: str,
    filename: str,
) -> list[Document]:
    """
    Add contextual enrichment and richer metadata to each chunk.

    - Prepends "Document: X | Section: Y" to page_content
    - Adds chunk_index, section, doc_title, total_chunks to metadata
    """
    doc_title = extract_doc_title(markdown, filename)
    headers = extract_section_headers(markdown)

    enriched = []
    for idx, chunk in enumerate(chunks):
        section = find_nearest_header(
            markdown.find(chunk.page_content[:100]) if chunk.page_content[:100] in markdown else 0,
            headers,
        )

        prefix = f"Document: {filename}"
        if section:
            prefix += f" | Section: {section}"
        enriched_content = f"{prefix}\n\n{chunk.page_content}"

        enriched.append(Document(
            page_content=enriched_content,
            metadata={
                **chunk.metadata,
                "chunk_index": idx,
                "section": section,
                "doc_title": doc_title,
                "total_chunks": len(chunks),
            },
        ))

    return enriched


def run_ingestion():
    start = time.time()

    converter = MarkItDown()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    data_dir = DATA_PATH
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    print(f"🚀 Found {len(files)} PDF(s)")

    qdrant_client = None
    vector_store = None

    for file in tqdm(files, desc="Ingesting PDFs"):
        file_path = os.path.join(data_dir, file)
        tqdm.write(f"  📄 {file}")

        try:
            markdown = convert_pdf(converter, file_path)

            if not markdown or len(markdown.strip()) < 50:
                tqdm.write(f"  ⚠️  No content extracted from {file}, skipping")
                continue

            doc = Document(page_content=markdown, metadata={"source": file})
            chunks = splitter.split_documents([doc])

            if not chunks:
                tqdm.write(f"  ⚠️  No chunks from {file}, skipping")
                continue

            chunks = enrich_chunks(chunks, markdown, file)
            tqdm.write(f"    ✅ {len(chunks)} enriched chunks")

            if qdrant_client is None:
                sample_vec = embeddings.embed_query("probe")
                qdrant_client = init_qdrant(embed_dim=len(sample_vec))
                vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name="local_docs",
                    embedding=embeddings,
                )

            for i in range(0, len(chunks), EMBED_BATCH):
                vector_store.add_documents(chunks[i : i + EMBED_BATCH])

            del chunks, markdown, doc
            gc.collect()

        except Exception as e:
            tqdm.write(f"  ⚠️  Error on {file}: {e}")

    print(f"\n✅ Done! Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    run_ingestion()
