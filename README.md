# Ask My Docs (Local Agentic RAG)

A fully local, production-grade Retrieval-Augmented Generation (RAG) system that lets you query your own documents — financial reports, research papers, PDFs — using a local LLM. No API keys. No data leaving your machine.

Built with MarkItDown, LangGraph, Qdrant, BM25, a cross-encoder reranker, CRAG, HyDE, Multi-Query, RAPTOR, and Ollama.

---

## What it does

You point it at a folder of PDFs. It parses them, chunks them with contextual enrichment, embeds them into a local vector database, and gives you a CLI where you can ask natural language questions and get grounded answers with citations.

The retrieval is driven by a **LangGraph agentic state machine** — it classifies queries, transforms them (HyDE / Multi-Query), retrieves via hybrid search (vector + BM25), re-ranks, evaluates relevance (CRAG), retries if needed, and generates answers with citations.

---

## Tech stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| PDF parsing | MarkItDown (`markitdown[pdf]`) | Fast, clean PDF → text (no GPU, no whitespace bloat) |
| Vector DB | Qdrant (local) | Stores and searches embeddings |
| Embeddings | Ollama (`nomic-embed-text`) | 768-dim local embeddings |
| Keyword search | BM25Retriever (LangChain) | Exact-match / lexical retrieval |
| Fusion | Reciprocal Rank Fusion (RRF) | Merges vector + BM25 result lists |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder scoring of top candidates |
| LLM | Ollama (`mistral-nemo:12b`) | Local inference, citation-enforced answers |
| Orchestration | LangGraph | Agentic state machine for query routing + CRAG |
| Query transforms | HyDE, Multi-Query | Improve retrieval recall |
| CRAG | `pipeline/evaluator.py` | Corrective RAG — retries on low-relevance results |
| RAPTOR | `raptor/tree_builder.py` | Hierarchical cluster summaries for multi-hop queries |
| Evaluation | RAGAS + Ollama | Faithfulness, relevancy, context precision metrics |

Hybrid search (vector + BM25 → RRF → cross-encoder) makes this significantly more accurate than naive single-retriever RAG, especially for financial figures, specific dates, and proper nouns.

---

## Hardware requirements

MarkItDown replaced Docling as the PDF parser — GPU is no longer required for ingestion. Ollama still benefits from VRAM for LLM inference.

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 12 GB | 16 GB |
| VRAM | 0 GB (ingest) / 4 GB (LLM) | 8 GB NVIDIA (LLM) |
| Storage | 5 GB free | 10 GB (models + DB) |
| CPU | 4 cores | 8+ cores |

Developed on: Intel Core i7, 16 GB RAM, NVIDIA RTX 4060 Laptop (8 GB VRAM), Windows 11.

---

## Project structure

```
ask_my_docs/
├── data/                        # Drop your PDFs here
├── db/
│   └── qdrant_storage/          # Local Qdrant vector database (auto-created)
├── config.py                    # All constants, paths, and feature flags
├── ingest.py                    # PDF → enriched chunks → Qdrant
├── retriver.py                  # CLI entrypoint — invokes LangGraph pipeline
├── check_ingest.py              # Verify ingestion worked
├── debug_chunks.py              # Inspect chunks in the DB
├── search_db.py                 # Manual vector search utility
├── pipeline/
│   ├── retrieval.py             # Vector + BM25 retriever setup
│   ├── fusion.py                # Reciprocal Rank Fusion
│   ├── rerank.py                # Cross-encoder reranking
│   ├── generation.py            # Prompt + LLM answer generation
│   ├── query_transform.py       # HyDE, Multi-Query, Adaptive routing
│   └── evaluator.py             # CRAG chunk relevance evaluation
├── graph/
│   ├── state.py                 # RAGState TypedDict
│   └── agent.py                 # LangGraph state machine
├── raptor/
│   └── tree_builder.py          # RAPTOR hierarchical summaries
├── eval/
│   ├── test_set.py              # 15 curated Q&A pairs
│   └── run_eval.py              # RAGAS evaluation runner
├── .env                         # Your configuration (see setup below)
└── requirements.txt
```

---

## Setup

### 1. Create environment

```bash
conda create -n local_rag python=3.11
conda activate local_rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M
```

Keep Ollama running in a separate terminal:

```bash
ollama serve
```

### 4. Configure your `.env`

```env
OLLAMA_MODEL=mistral-nemo:12b-instruct-2407-q4_K_M
EMBED_MODEL=nomic-embed-text
DB_PATH=./db/qdrant_storage
DATA_PATH=./data
```

### 5. Add your PDFs

Drop any `.pdf` files into the `data/` folder. The system is tested with:
- NVIDIA 10-K annual reports (2023, 2024, 2025)
- Tesla 10-K annual reports (2022, 2023, 2024)
- Academic papers on RAG systems

---

## Usage

### Step 1 — Ingest your documents

```bash
python ingest.py
```

This will:
- Parse every PDF in `data/` using MarkItDown (fast, no GPU needed)
- Chunk each document into 1000-character overlapping segments
- Enrich each chunk with contextual metadata (document title + section prefix)
- Embed every chunk using `nomic-embed-text` via Ollama
- Save everything to a local Qdrant database at `DB_PATH`

Expect roughly 15–30 seconds per 100-page document (50–100x faster than the previous Docling-based parser).

### Step 2 — (Optional) Build RAPTOR summary tree

```bash
python -m raptor.tree_builder
```

Clusters chunks using KMeans, summarizes each cluster with the LLM, and stores multi-level summaries in Qdrant. Improves answers for broad multi-hop questions.

### Step 3 — Verify ingestion

```bash
python check_ingest.py
```

You should see `vectors_count > 0` and `size = 768` for `nomic-embed-text`.

### Step 4 — Ask questions

```bash
python retriver.py
```

Then type any question at the prompt:

```
What was NVIDIA's total revenue in fiscal 2024?
What retrieval method does RAG use for dense retrieval?
What was Tesla's gross profit margin in 2023?
```

Each answer includes citations like `[1]`, `[2]` that reference the source document and section the claim came from.

### Step 5 — (Optional) Run evaluation

```bash
python -m eval.run_eval          # Full 15-question RAGAS eval
python -m eval.run_eval --quick  # Quick 5-question smoke test
```

---

## How it works

### Ingestion pipeline (`ingest.py`)

```
PDF files → MarkItDown → clean text
         → RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
         → Contextual enrichment: "Document: X | Section: Y\n<chunk text>"
         → nomic-embed-text (768-dim vectors)
         → Qdrant local DB (with metadata: doc_title, section, chunk_index)
```

### Agentic retrieval pipeline (`graph/agent.py`)

```
User question
    │
    ▼
classify_node ──→ [direct] ──→ direct_answer_node (skip retrieval for simple Qs)
    │
    ▼ [retrieve]
transform_node  (HyDE: embed hypothetical answer / Multi-Query: 3 variants)
    │
    ▼
retrieve_node   (Vector search top-10 + BM25 top-10, per query variant)
    │
    ▼
fuse_rerank_node (RRF fusion → cross-encoder reranker → top-4 chunks)
    │
    ▼
evaluate_node   (CRAG: check reranker scores + positive-chunk ratio)
    │
    ├── [pass]  ──→ generate_node → Answer with [1][2] citations
    │
    └── [fail]  ──→ retry_transform_node → retrieve again (max 2 retries)
                        └── [fail after retries] → generate_node (best-effort)
```

---

## Configuration reference

All configuration lives in `config.py`. It reads `.env` for paths and model names.

### Feature flags

| Flag | Default | Description |
|---|---|---|
| `ENABLE_HYDE` | `True` | Generate hypothetical answer for query embedding |
| `ENABLE_MULTI_QUERY` | `True` | Generate 3 query variants and merge results |
| `ENABLE_CRAG` | `True` | Evaluate chunk relevance and retry on failure |
| `ENABLE_ADAPTIVE` | `True` | Classify query and skip retrieval for direct answers |

> **Note:** HyDE + Multi-Query stacked adds ~4–5 LLM calls per query (12–15s latency). Disable one or both in `config.py` if speed matters more than recall.

### Retrieval parameters

| Constant | Default | Effect |
|---|---|---|
| `SEARCH_K` | `10` | Candidates fetched per retriever before fusion |
| `RRF_K` | `60` | RRF smoothing constant |
| `RERANK_TOP_N` | `4` | Chunks passed to the LLM after reranking |

### Ingestion parameters

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `PAGE_BATCH` | `10` | PDF pages processed per MarkItDown pass |
| `EMBED_BATCH` | `32` | Chunks sent to Ollama per embedding call |

### RAPTOR parameters

| Constant | Default | Effect |
|---|---|---|
| `CHUNKS_PER_CLUSTER` | `100` | Chunks per KMeans cluster (higher = fewer LLM calls) |
| `RAPTOR_MODEL` | (set in config) | LLM used for cluster summarization (can differ from main LLM) |

**Critical:** `EMBED_MODEL` must match between ingest and retrieval. Changing it requires deleting and recreating the Qdrant collection.

---

## Troubleshooting

**Answers are wrong or hallucinated**
Check reranker scores printed during retrieval — if all scores are below 0, relevant chunks may not be in your corpus. Re-run `python ingest.py` and verify chunk previews with `python check_ingest.py`.

**CRAG keeps retrying**
Your question may be too broad or outside the corpus. Check `MIN_POSITIVE_RATIO` in `pipeline/evaluator.py` — lower it (e.g. 0.3) if too aggressive.

**`model requires more system memory than is available`**
Ollama cannot load the LLM. Switch to a smaller model like `mistral:7b-instruct-q4_K_M` in `.env`.

**`ollama._types.ResponseError` connection refused**
Ollama is not running. Open a separate terminal and run `ollama serve`.

**RAPTOR tree build is slow**
Increase `CHUNKS_PER_CLUSTER` in `config.py` to reduce the number of LLM summarization calls. Set `RAPTOR_MODEL` to a faster/smaller model.

---

## Roadmap

- [x] Phase 1 — PDF ingestion pipeline (MarkItDown, contextual enrichment)
- [x] Phase 2 — Hybrid vector + BM25 search with RRF fusion
- [x] Phase 3 — Cross-encoder reranking
- [x] Phase 4 — Local LLM generation with citation enforcement
- [x] Phase 5 — Agentic LangGraph pipeline (classify, transform, retrieve, evaluate, generate)
- [x] Phase 6 — HyDE + Multi-Query query transforms
- [x] Phase 7 — CRAG corrective retrieval with retry loop
- [x] Phase 8 — RAPTOR hierarchical summary tree
- [x] Phase 9 — RAGAS evaluation framework (faithfulness, relevancy, context precision)
- [ ] Phase 10 — Streamlit UI with drag-and-drop PDF upload and trace logging

---
