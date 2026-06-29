# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How Claude should use this file (READ FIRST — token policy)

This file is the project's **persistent memory across chat sessions**. The user
does NOT want to re-paste the whole previous conversation into a new chat — that
wastes tokens. Instead:

1. **At the start of every new session**, read this file (especially the
   **Session Log** below) to recover context on what's been done and what the
   current issue is. Do NOT ask the user to repeat history that's recorded here.
2. **Whenever you learn something new, make a code change, or change the current
   focus/issue**, append a dated entry to the **Session Log** — concise bullet
   points only. Record: what changed, why, the file(s) touched, and the current
   open question. This is what lets the next session pick up cold.
3. Keep entries **short and factual** (the goal is to save tokens, not write
   prose). Newest entry at the top. Prune entries that are fully superseded.
4. Don't duplicate what the code already says — record decisions, rationale, and
   state that aren't obvious from reading the source.

## Session Log

### 2026-06-21 — Switched PDF parser from Docling to MarkItDown

- **Problem:** Docling's markdown output had excessive whitespace padding in tables
  (e.g. `Net     income  attributable    to      common  stockholders`) wasting
  ~50-80% of chunk capacity. Also caused `std::bad_alloc` errors on complex pages,
  and required GPU + large PyTorch model downloads.
- **Fix:** Replaced Docling with Microsoft MarkItDown (`markitdown[pdf]` v0.1.6).
  MarkItDown produces clean text, no whitespace bloat, no GPU needed, 50-100x faster.
  Financial table data (revenue, net income) now appears cleanly in chunks.
- **Also fixed:** CRAG evaluator now checks positive-chunk ratio (not just max score).
  If <50% of chunks have positive reranker scores, flags as ambiguous → triggers
  LLM evaluation instead of passing through with bad chunks.
- **Files changed:** `ingest.py` (complete rewrite — removed Docling, torch, fitz
  imports; uses MarkItDown), `pipeline/evaluator.py` (added MIN_POSITIVE_RATIO),
  `config.py` (added RAPTOR settings, PAGE_BATCH lowered to 10)
- **Removed deps:** Docling, torch, fitz no longer needed by ingest.py (still used
  elsewhere). Added: `markitdown[pdf]`
- **Requires re-ingestion:** `python ingest.py`
- **RAPTOR:** increased CHUNKS_PER_CLUSTER from 40→100 to reduce LLM calls. Added
  RAPTOR_MODEL config for using a lighter model for summarization.

### 2026-06-21 — Full Agentic RAG Build-Out

**Major refactor: monolithic retriver.py → modular agentic pipeline.** All code
compiles and imports pass. User needs to re-ingest (enriched chunks) and test.

Changes made (all phases implemented in one session):

- **Phase 0 — Refactor:**
  - Created `config.py` — centralized constants + feature flags (`ENABLE_HYDE`,
    `ENABLE_MULTI_QUERY`, `ENABLE_CRAG`, `ENABLE_ADAPTIVE`)
  - Created `pipeline/` package: `retrieval.py`, `fusion.py`, `rerank.py`,
    `generation.py`, `query_transform.py`, `evaluator.py`
  - Created `eval/test_set.py` (15 curated Q&A pairs) + `eval/run_eval.py`
    (RAGAS metrics runner using Ollama locally)
  - Updated `debug_chunks.py`, `search_db.py`, `check_ingest.py` to use config.py

- **Phase 1 — Query-time enhancements (pipeline/query_transform.py):**
  - HyDE: generates hypothetical answer, embeds that for vector search
  - Multi-Query: generates 3 query variants, retrieves for each, merges
  - Adaptive: classifies query as "retrieve" or "direct" to skip retrieval

- **Phase 2 — Ingestion + CRAG:**
  - `ingest.py`: contextual enrichment — prepends "Document: X | Section: Y" to
    each chunk before embedding. Adds metadata: chunk_index, section, doc_title,
    total_chunks. **Requires re-ingestion.**
  - `pipeline/evaluator.py`: two-tier CRAG — fast score check + LLM fallback.
    Triggers multi-query reformulation on failure (max 2 attempts).

- **Phase 3 — LangGraph agentic pipeline (graph/):**
  - `graph/state.py`: RAGState TypedDict
  - `graph/agent.py`: full state machine with nodes (classify, transform,
    retrieve, fuse_rerank, evaluate, generate, direct_answer, retry_transform)
    and conditional edges for adaptive routing + CRAG retry loop
  - `retriver.py` now uses `graph.invoke()` instead of linear pipeline

- **Phase 4 — RAPTOR (raptor/tree_builder.py):**
  - KMeans clustering → LLM summarization → multi-level summary tree in Qdrant
  - Run after ingest: `python -m raptor.tree_builder`

**Next steps for user:**
1. Re-ingest: `python ingest.py` (will rebuild with enriched chunks)
2. Build RAPTOR tree: `python -m raptor.tree_builder`
3. Test: `python retriver.py` — should show feature flags, run graph pipeline
4. Eval baseline: `python -m eval.run_eval --quick`

**Open:** Features all default to enabled (`config.py`). User may want HyDE and
Multi-Query NOT stacked (adds 4-5 LLM calls = 12-15s latency per query). Toggle
in `config.py`.

### 2026-06-21 — Initial prompt fix (superseded by refactor above)
- Generation prompt rewritten to demand synthesis across chunks (now in
  `pipeline/generation.py`)

## Project Overview

Local RAG system: parses PDFs, indexes into Qdrant, answers questions via hybrid
search pipeline with Ollama LLMs. Fully offline — no API keys, no data leaves
the machine.

## Architecture

```
retriver.py (CLI)
  └── graph/agent.py (LangGraph state machine)
        ├── classify_node      → adaptive routing (retrieve vs direct)
        ├── transform_node     → HyDE / multi-query
        ├── retrieve_node      → hybrid vector + BM25
        ├── fuse_rerank_node   → RRF fusion + cross-encoder
        ├── evaluate_node      → CRAG relevance check
        ├── retry_transform    → reformulate on CRAG failure (max 2)
        ├── generate_node      → LLM answer with citations
        └── direct_answer_node → skip retrieval for simple Qs

ingest.py → Docling PDF → markdown → split → enrich (title+section prefix) → embed → Qdrant
raptor/tree_builder.py → cluster chunks → LLM summarize → embed → store as higher-level nodes
```

## Setup & Commands

```bash
conda activate local_rag
pip install -r requirements.txt

# Ollama (must be running: ollama serve)
ollama pull nomic-embed-text
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M

# Full pipeline
python ingest.py                  # Ingest PDFs (with contextual enrichment)
python -m raptor.tree_builder     # Build RAPTOR summary tree (optional)
python check_ingest.py            # Verify ingestion
python retriver.py                # Interactive CLI (agentic pipeline)
python -m eval.run_eval           # RAGAS evaluation (--quick for 5 Qs)
```

## Configuration

`config.py` is the single source of truth for all constants, paths, model names,
and feature flags. It loads `.env` for paths and model names.

Feature flags: `ENABLE_HYDE`, `ENABLE_MULTI_QUERY`, `ENABLE_CRAG`, `ENABLE_ADAPTIVE`
— all default `True`. Toggle in `config.py`.

**Critical:** `EMBED_MODEL` must match between ingest and retrieval. Changing it
requires deleting and recreating the Qdrant collection.

## Key Dependencies

PyTorch (CUDA cu126), Docling, LangChain, LangGraph, Qdrant, sentence-transformers,
RAGAS, scikit-learn, Ollama.

## File Map

| File | Role |
|---|---|
| `config.py` | All constants, paths, feature flags |
| `ingest.py` | PDF → enriched chunks → Qdrant |
| `retriver.py` | CLI entrypoint, invokes LangGraph |
| `pipeline/retrieval.py` | Vector + BM25 retriever setup |
| `pipeline/fusion.py` | Reciprocal Rank Fusion |
| `pipeline/rerank.py` | Cross-encoder reranking |
| `pipeline/generation.py` | Prompt + LLM answer generation |
| `pipeline/query_transform.py` | HyDE, Multi-Query, Adaptive routing |
| `pipeline/evaluator.py` | CRAG chunk relevance evaluation |
| `graph/state.py` | RAGState TypedDict |
| `graph/agent.py` | LangGraph state machine |
| `raptor/tree_builder.py` | RAPTOR hierarchical summaries |
| `eval/test_set.py` | Curated Q&A test pairs |
| `eval/run_eval.py` | RAGAS evaluation runner |
