# Doc Chat

An AI-powered document assistant that lets you upload multiple documents, chat with them, edit content using natural language, and get an audio explanation — all through a clean Streamlit UI backed by a FastAPI server.

---

## What it does

- **Multi-document chat** — upload multiple files and ask questions across all of them in a single unified chat. The bot tells you which document each answer came from.
- **Intent routing** — automatically detects whether you want to ask a question, get a summary, request a simple explanation, or make an edit.
- **Natural language editing** — tell the bot what to change ("change annual leave from 7 to 10 days") and it finds the right section, applies the edit, and re-indexes the chunk.
- **Conversation memory** — the bot remembers the last few turns so follow-up questions work naturally.
- **Audio explainer** — upload a document and get a spoken MP3 explanation in the style of a friend explaining it to you the night before an exam.
- **Download updated TXT** — after edits, download the updated document as plain text.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI — GPT-4.1-mini |
| Embeddings | Azure OpenAI — text-embedding-3-large (3072 dim) |
| RAG framework | LangGraph + LangChain |
| Vector store | Milvus-Lite (embedded, no server needed) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Text to speech | ElevenLabs |
| Chunking | Paragraph-aware greedy merger with tiktoken |
| Package manager | uv |
| Language | Python 3.12 |

---

## Supported file types

| Format | Parser |
|---|---|
| PDF | pymupdf (fitz) |
| DOCX | python-docx |
| PPTX | python-pptx |
| XLSX | openpyxl (row-based chunking for large files) |
| CSV | csv stdlib (row-based chunking for large files) |
| TXT | stdlib (paragraph-aware) |

---

## Project structure

```
doc-chat-v2/
├── api/
│   ├── main.py          # FastAPI app — all routes
│   ├── session.py       # In-memory session store + conversation memory
│   └── schemas.py       # Pydantic request/response models
├── ingestion/
│   └── parser.py        # Format router + per-format text extractors
├── vectorstore/
│   ├── chunker.py       # Paragraph-aware greedy chunker
│   ├── embedder.py      # Azure OpenAI embedding wrapper
│   └── milvus_client.py # Milvus-Lite collection + search + upsert
├── graph/
│   ├── state.py         # LangGraph DocState definition
│   ├── nodes.py         # Intent classifier, retriever, generate, edit nodes
│   └── graph.py         # Compiled LangGraph pipeline + run() entry point
├── export/
│   └── reconstructor.py # Apply edits to full_text, return updated TXT
├── pages/
│   └── audio_explainer.py  # Streamlit audio explainer page
├── app.py               # Streamlit main app
├── config.py            # Centralised config — loads from .env
├── requirements.txt
└── .env                 # API keys (not committed)
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/yourname/doc-chat-v2.git
cd doc-chat-v2
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
uv add pymupdf python-docx python-pptx openpyxl python-dotenv \
       pymilvus[milvus_lite] tiktoken openai \
       langgraph langchain-openai langchain-core \
       fastapi uvicorn python-multipart \
       streamlit elevenlabs
```

Or from requirements.txt:

```bash
uv pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
AZURE_OPENAI_EMB_KEY=your_embedding_key
AZURE_OPENAI_LLM_KEY=your_llm_key
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMB_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_API_VERSION=2025-01-01-preview
AZURE_EMB_API_VERSION=2024-12-01-preview
AZURE_EMB_DEPLOYMENT=text-embedding-3-large
AZURE_LLM_DEPLOYMENT=gpt-4.1-mini
ELEVENLABS_API_KEY=your_elevenlabs_key
```

---

## Running

Always start FastAPI first, then Streamlit — Milvus-Lite only allows one process to hold the database file at a time.

```bash
# Terminal 1 — FastAPI backend
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

FastAPI Swagger docs available at `http://localhost:8000/docs`.

---

## API routes

| Method | Route | Description |
|---|---|---|
| POST | `/session` | Create a new session, returns `session_id` |
| GET | `/session/{session_id}` | Get session info — docs, message count, edit count |
| POST | `/upload` | Upload and index a document into a session |
| POST | `/chat` | Send a query, get a response with sources and intent |
| GET | `/download/{session_id}` | Download updated TXT (all docs or single doc) |
| DELETE | `/session/{session_id}` | Clear session — removes docs from Milvus and memory |

---

## Chunking strategy

Documents are split using a **paragraph-aware greedy merger**:

- Headings are never chunked alone — they become `section_heading` metadata on the following chunk
- Tables always get their own chunk regardless of size
- For large CSV/XLSX files, rows are chunked in groups of ~300 tokens with the header row repeated on every chunk
- Chunk size limit: 480 tokens with 50 token overlap
- Token counting uses `tiktoken` with `cl100k_base` encoding (same as GPT-4)

---

## RAG pipeline

```
user query
    └── intent classifier (GPT-4.1-mini)
            ├── summarise → retrieve all chunks → generate
            ├── explain   → retrieve top-6 chunks → generate
            ├── qa        → retrieve top-6 chunks → generate
            └── edit      → retrieve top-3 → LLM picks best → apply edit
                                                             → re-embed
                                                             → upsert Milvus
```

All retrieval is scoped to the current `session_id` so queries search across all uploaded documents automatically. Each retrieved chunk carries its `filename` so the generator can cite the source in the answer.

---

## Audio explainer

Navigate to the **Audio Explainer** page from the sidebar. Upload any supported document and click **Generate audio explanation**. The pipeline:

1. Sends a summarise query through the RAG pipeline with a "friend explaining before an exam" prompt
2. Passes the generated script to ElevenLabs TTS
3. Returns an MP3 you can play in the browser or download

ElevenLabs free tier gives 10,000 characters/month. Scripts are kept under 900 words (~4,500 characters) to stay within limits.

---

## Git workflow

```bash
# after each module
git add .
git commit -m "feat: description"

# check history
git log --oneline

# undo uncommitted changes
git stash

# go back to last commit
git checkout .
```

---

## Known limitations

- Session memory is in-process only — restarting FastAPI clears all sessions and chat history
- Milvus-Lite allows only one process to open the `.db` file — always run FastAPI before Streamlit
- ElevenLabs free tier has a 10,000 character/month limit
- PDF heading detection uses font size heuristic (`>= 14pt`) — may need tuning per document
- Edit pipeline applies only the specific change requested — dependent calculations (e.g. totals in a spreadsheet) are not automatically recalculated

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Milvus-Lite](https://milvus.io/docs/milvus_lite.md)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [ElevenLabs](https://elevenlabs.io)
- [Streamlit](https://streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)