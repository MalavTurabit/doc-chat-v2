import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI — Embeddings ─────────────────────────────────────────────────
AZURE_OPENAI_EMB_KEY   = os.getenv("AZURE_OPENAI_EMB_KEY")
AZURE_EMB_ENDPOINT     = os.getenv("AZURE_EMB_ENDPOINT")
AZURE_EMB_API_VERSION  = os.getenv("AZURE_EMB_API_VERSION")
AZURE_EMB_DEPLOYMENT   = os.getenv("AZURE_EMB_DEPLOYMENT")

# ── Azure OpenAI — LLM ───────────────────────────────────────────────────────
AZURE_OPENAI_LLM_KEY   = os.getenv("AZURE_OPENAI_LLM_KEY")
AZURE_LLM_ENDPOINT     = os.getenv("AZURE_LLM_ENDPOINT")
AZURE_LLM_API_VERSION  = os.getenv("AZURE_LLM_API_VERSION")
AZURE_LLM_DEPLOYMENT   = os.getenv("AZURE_LLM_DEPLOYMENT")

# ── Extraction ───────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS   = {".pdf", ".docx", ".pptx", ".xlsx"}

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS      = 512
CHUNK_OVERLAP_TOKENS   = 50

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_DIM          = 3072   # text-embedding-3-large

# ── Milvus ───────────────────────────────────────────────────────────────────
MILVUS_DB_PATH         = "./doc_chat.db"
MILVUS_COLLECTION      = "doc_chunks"