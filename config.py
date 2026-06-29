import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH   = os.getenv("DB_PATH", "./db/qdrant_storage")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b-instruct-2407-q4_K_M")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH   = 32
PAGE_BATCH    = 10

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SEARCH_K      = 40
RRF_K         = 60
RERANK_TOP_N  = 4
RERANK_BATCH  = 32
NUM_CTX       = 8192

# ---------------------------------------------------------------------------
# RAPTOR
# ---------------------------------------------------------------------------
RAPTOR_CHUNKS_PER_CLUSTER = 100   # higher = fewer clusters = fewer LLM calls
RAPTOR_MIN_CLUSTERS = 3
RAPTOR_MODEL = os.getenv("RAPTOR_MODEL", OLLAMA_MODEL)  # override with a faster model if desired

# ---------------------------------------------------------------------------
# Feature flags — toggle query-time enhancements
# ---------------------------------------------------------------------------
ENABLE_HYDE        = True
ENABLE_MULTI_QUERY = True
ENABLE_CRAG        = True
ENABLE_ADAPTIVE    = True
