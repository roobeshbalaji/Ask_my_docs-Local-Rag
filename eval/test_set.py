"""
Curated Q&A pairs for evaluating the RAG pipeline.

Each entry has:
  - question: the query to ask
  - ground_truth: a reference answer (used for context_recall and answer_correctness)
  - source: which PDF(s) should be cited
  - category: "factual", "conceptual", "multi-hop", or "out-of-corpus"
"""

TEST_SET = [
    # --- Factual questions (financial reports) ---
    {
        "question": "What was NVIDIA's total revenue in fiscal year 2025?",
        "ground_truth": "NVIDIA's total revenue in fiscal year 2025 was $130.5 billion.",
        "source": "nvidia10k2025.pdf",
        "category": "factual",
    },
    {
        "question": "What was Tesla's total revenue in fiscal year 2023?",
        "ground_truth": "Tesla's total revenue in fiscal year 2023 was $96.8 billion.",
        "source": "tesla10k2023.pdf",
        "category": "factual",
    },
    {
        "question": "How many employees did NVIDIA have at the end of fiscal year 2024?",
        "ground_truth": "NVIDIA had approximately 29,600 employees at the end of fiscal year 2024.",
        "source": "nvidia10k2024.pdf",
        "category": "factual",
    },
    {
        "question": "What was Tesla's automotive gross margin in 2022?",
        "ground_truth": "Tesla's automotive gross margin in 2022 was approximately 28.5%.",
        "source": "tesla10k2022.pdf",
        "category": "factual",
    },

    # --- Conceptual questions (RAG papers) ---
    {
        "question": "What retrieval method does RAG use for dense retrieval?",
        "ground_truth": "RAG uses Dense Passage Retrieval (DPR), which employs a bi-encoder architecture with BERT-based models to encode queries and passages into dense vectors, then uses Maximum Inner Product Search (MIPS) to find relevant passages.",
        "source": "rag2.pdf",
        "category": "conceptual",
    },
    {
        "question": "What is the difference between RAG-Sequence and RAG-Token models?",
        "ground_truth": "RAG-Sequence uses the same retrieved document to generate the entire output sequence, while RAG-Token can use different retrieved documents for each output token, marginalizing over documents at each generation step.",
        "source": "rag1.pdf",
        "category": "conceptual",
    },
    {
        "question": "How does the cross-encoder reranking approach differ from bi-encoder retrieval?",
        "ground_truth": "A bi-encoder encodes query and document separately into vectors and compares them, which is fast but less precise. A cross-encoder processes the query and document together as one input sequence, enabling cross-attention between them for more accurate relevance scoring, but at higher computational cost.",
        "source": "rag3.pdf",
        "category": "conceptual",
    },
    {
        "question": "What is Reciprocal Rank Fusion and why is it used?",
        "ground_truth": "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers by assigning scores based on rank position rather than raw scores. It avoids the problem of incompatible score scales between different retrieval methods.",
        "source": "rag4.pdf",
        "category": "conceptual",
    },
    {
        "question": "What role does BM25 play in hybrid retrieval systems?",
        "ground_truth": "BM25 is a keyword-based retrieval algorithm that scores documents by term frequency weighted by inverse document frequency. In hybrid systems, it complements dense vector retrieval by excelling at exact keyword matching for specific names, numbers, and terminology that embedding models may miss.",
        "source": "rag2.pdf",
        "category": "conceptual",
    },
    {
        "question": "How does chunking affect retrieval quality in RAG systems?",
        "ground_truth": "Chunking determines the granularity of retrieved information. Smaller chunks provide more precise retrieval but may lose context, while larger chunks preserve context but may include irrelevant information. Overlap between chunks helps maintain continuity across chunk boundaries.",
        "source": "rag5.pdf",
        "category": "conceptual",
    },

    # --- Multi-hop questions ---
    {
        "question": "Compare NVIDIA's revenue growth between fiscal years 2023 and 2024.",
        "ground_truth": "NVIDIA's revenue grew significantly from fiscal year 2023 to fiscal year 2024, driven primarily by strong demand in the Data Center segment for AI and accelerated computing products.",
        "source": "nvidia10k2023.pdf,nvidia10k2024.pdf",
        "category": "multi-hop",
    },
    {
        "question": "What are the main advantages and disadvantages of dense retrieval compared to sparse retrieval in RAG?",
        "ground_truth": "Dense retrieval captures semantic similarity and handles paraphrases well but may miss exact keyword matches. Sparse retrieval (like BM25) excels at exact matching and is efficient but fails to capture semantic relationships between terms.",
        "source": "rag2.pdf,rag3.pdf",
        "category": "multi-hop",
    },

    # --- Out-of-corpus questions (should return 'I don't have enough information') ---
    {
        "question": "What was Apple's revenue in Q4 2024?",
        "ground_truth": "I don't have enough information in the provided documents to answer that.",
        "source": "",
        "category": "out-of-corpus",
    },
    {
        "question": "What is the current stock price of Amazon?",
        "ground_truth": "I don't have enough information in the provided documents to answer that.",
        "source": "",
        "category": "out-of-corpus",
    },
    {
        "question": "How does GPT-4 handle context window limitations?",
        "ground_truth": "I don't have enough information in the provided documents to answer that.",
        "source": "",
        "category": "out-of-corpus",
    },
]
