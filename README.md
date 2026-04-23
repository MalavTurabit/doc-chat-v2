# Doc Chat v2

An AI-powered document intelligence platform that lets you upload multiple documents, chat with them naturally, compare documents, perform cross-document analysis, edit content with natural language, view extracted images, and get audio explanations — all through a clean Streamlit UI backed by a FastAPI server.

---

## What it does

- **Multi-document chat** — upload multiple files and ask questions across all of them in a single unified chat. The bot tells you which document each answer came from.
- **Intent routing** — automatically detects whether you want to ask a question, summarise, explain, edit, compare, analyse, or view images.
- **Image understanding** — extracts and displays images from uploaded documents. Scanned/image-based PDFs are processed via RapidOCR. Embedded images in PDFs, DOCX, and PPTX can be described using GPT-4.1-mini vision (toggle in sidebar).
- **Image display in chat** — when an answer references a page or image, clickable image links appear directly in the chat.
- **Cross-document comparison** — upload two or more documents and ask the bot to compare them side by side.
- **Cross-document analysis** — find patterns, contradictions, insights, and gaps across all uploaded documents.
- **Natural language editing** — tell the bot what to change and it finds the right section, applies the edit, and re-indexes the chunk.
- **Smart retrieval** — rule-based query classifier routes queries to the right retrieval strategy: semantic search, keyword/identifier lookup, analytical sampling, or positional (page/section) filtering.
- **Follow-up query rewriting** — detects pronouns and references ("what about her salary?") and rewrites them into standalone queries using conversation memory.
- **Conversation memory** — the bot remembers the last 6 messages so follow-up questions work naturally.
- **Audio explainer** — upload a document and get a spoken MP3 explanation in the style of a friend explaining it to you the night before an exam.
- **Download updated TXT** — after edits, download the updated document as plain text.
- **Guardrails** — the bot stays strictly within the uploaded document context and will not answer unrelated questions.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI — GPT-4.1-mini |
| LLM Vision | Azure OpenAI — GPT-4.1-mini (image input) |
| Embeddings | Azure OpenAI — text-embedding-3-large (3072 dim) |
| OCR | RapidOCR (rapidocr-onnxruntime) |
| RAG framework | LangGraph + LangChain |
| Vector store | Milvus-Lite (embedded, no server needed) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Text to speech | ElevenLabs |
| Chunking | Paragraph-aware greedy merger with tiktoken |
| Image normalisation | Pillow (PIL) |
| Package manager | uv |
| Language | Python 3.12 |

---

## Supported file types

| Format | Parser | Image support |
|---|---|---|
| PDF (text layer) | pymupdf (fitz) | Embedded images via GPT-4.1-mini (toggle) |
| PDF (scanned/image-based) | RapidOCR fallback | Page images always saved for display |
| DOCX | python-docx | Embedded images via GPT-4.1-mini (toggle) |
| PPTX | python-pptx | Picture shapes via GPT-4.1-mini (toggle) |
| XLSX | openpyxl (row-based chunking) | — |
| CSV | csv stdlib (row-based chunking) | — |
| TXT | stdlib (paragraph-aware) | — |
| PNG / JPG / JPEG | RapidOCR + optional vision | Always saved for display |

---

## Project structure

```
doc-chat-v2/
├── api/
│   ├── main.py             # FastAPI app — all routes
│   ├── session.py          # In-memory session store + image path registry
│   └── schemas.py          # Pydantic request/response models
├── ingestion/
│   └── parser.py           # Format router, per-format extractors, OCR, vision
├── vectorstore/
│   ├── chunker.py          # Paragraph-aware greedy chunker
│   ├── embedder.py         # Azure OpenAI embedding wrapper with batching
│   └── milvus_client.py    # Milvus-Lite schema, search, upsert, keyword search
├── graph/
│   ├── state.py            # LangGraph DocState — includes image_refs field
│   ├── nodes.py            # All nodes: classifier, retriever, generate, edit,
│   │                       # compare, analyse, show_image, general
│   └── graph.py            # Compiled LangGraph pipeline + run() entry point
├── export/
│   └── reconstructor.py    # Apply edits to full_text, return updated TXT
├── pages/
│   └── audio_explainer.py  # Streamlit audio explainer page
├── extracted_images/       # Saved page/embedded images (gitignored)
├── app.py                  # Streamlit main app
├── config.py               # Centralised config — loads from .env
└── .env                    # API keys (not committed)
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/MalavTurabit/doc-chat-v2.git
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
       streamlit elevenlabs \
       rapidocr-onnxruntime Pillow
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

Always start FastAPI first — Milvus-Lite only allows one process to hold the database file at a time.

```bash
# Terminal 1 — FastAPI backend
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.
FastAPI Swagger docs: `http://localhost:8000/docs`

---

## API routes

| Method | Route | Description |
|---|---|---|
| POST | `/session` | Create a new session, returns `session_id` |
| GET | `/session/{session_id}` | Get session info |
| POST | `/upload` | Upload and index a document |
| POST | `/chat` | Send a query, get response + sources + image_refs |
| GET | `/image/{chunk_id}` | Serve extracted image as PNG |
| GET | `/image_b64/{chunk_id}` | Serve image as base64 JSON |
| GET | `/download/{session_id}` | Download updated TXT |
| DELETE | `/session/{session_id}` | Clear session from Milvus and memory |

---

## Intent routing

```
user query
    └── intent classifier (GPT-4.1-mini)
            ├── general    → friendly response about bot capabilities
            ├── summarise  → sample all chunks → generate summary
            ├── explain    → retrieve top-6 → explain simply
            ├── qa         → query classifier → smart retrieval → generate
            │                   ├── identifier  → keyword + semantic hybrid
            │                   ├── name        → keyword + semantic hybrid
            │                   ├── analytical  → all chunks sampled
            │                   ├── positional  → page/section metadata filter
            │                   └── semantic    → ANN vector search
            ├── edit       → retrieve top-3 → LLM picks chunk → apply edit
            │                              → re-embed → upsert Milvus
            ├── compare    → search per-doc → side-by-side LLM comparison
            ├── analyse    → search per-doc → cross-doc pattern analysis
            └── show_image → search for has_image chunks → return image links
```

---

## Image pipeline

```
Upload (any file type)
    ↓
parser extracts image bytes → image_map {block_index: png_bytes}
    ↓
api/main.py saves images to:
    ./extracted_images/{session_id}/{chunk_id}.png
    ↓
image_path stored on chunk → upserted to Milvus (has_image=True, image_path=...)
    ↓
Chat query
    ↓
retriever fetches chunks with has_image=True + image_path
    ↓
generate_node / show_image_node returns image_refs (list of chunk_ids)
    ↓
FastAPI GET /image/{chunk_id} → looks up image_path from Milvus → FileResponse
    ↓
Streamlit renders clickable links → user clicks → image opens in browser tab
```

**When images are extracted per file type:**

| File type | Toggle OFF | Toggle ON |
|---|---|---|
| PDF (scanned) | Page PNGs always saved | Same |
| PDF (text layer) | No images | Embedded charts/diagrams described + saved |
| DOCX | No images | Embedded images described + saved |
| PPTX | No images | Picture shapes described + saved |
| PNG / JPG | Always saved | Also described by GPT-4.1-mini |
| XLSX / CSV / TXT | Never | Never |

---

## Chunking strategy

Documents are split using a **paragraph-aware greedy merger**:

- Headings become `section_heading` metadata on the following chunk — never chunked alone
- Tables always get their own chunk regardless of size
- CSV/XLSX: rows chunked in groups of 20 rows with header repeated every chunk
- Chunk size: 480 tokens with 50 token overlap
- Token counting: `tiktoken` with `cl100k_base` encoding

---

## Session persistence

Session data is stored in two places:

| Data | Storage | Survives restart? |
|---|---|---|
| Chunk vectors + metadata | Milvus-Lite (`doc_chat.db`) | ✅ Yes |
| Image files | `./extracted_images/` on disk | ✅ Yes |
| Image paths | Milvus `image_path` field | ✅ Yes |
| Doc list, memory, edits | In-memory Python dicts | ❌ No |

After FastAPI restarts, `get_docs()` automatically rebuilds the doc list from Milvus so chat continues to work. Image paths are looked up directly from Milvus so images still display after restart.

---

## Audio explainer

Navigate to **Audio Explainer** from the sidebar. The pipeline:

1. Sends a summarise query with a "friend explaining before an exam" system prompt
2. Passes the script to ElevenLabs TTS (voice: George, model: eleven_multilingual_v2)
3. Returns an MP3 playable in the browser or downloadable

ElevenLabs free tier: 10,000 characters/month. Scripts are kept under 900 words.

---

## Known limitations

- Session memory (chat history, edit records) is lost on FastAPI restart — only vector data and images persist
- Milvus-Lite allows only one process to open the `.db` file — always start FastAPI before Streamlit
- ElevenLabs free tier has a 10,000 character/month limit
- Analytical queries (averages, counts) are approximate — LLM sees a sample of chunks, not all data
- Image-based PDFs with very low resolution or handwriting may produce poor OCR results
- Azure content safety filters may block responses for documents with certain content

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Milvus-Lite](https://milvus.io/docs/milvus_lite.md)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [ElevenLabs](https://elevenlabs.io)
- [Streamlit](https://streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)