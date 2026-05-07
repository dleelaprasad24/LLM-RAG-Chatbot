# LLM-Powered RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that lets
you chat with your own documents (PDF, TXT, MD, DOCX) and web pages. Built
with **LangChain**, **ChromaDB**, **HuggingFace sentence-transformers**, and
ships with **two interchangeable frontends** - a **Streamlit** UI for humans
and a **FastAPI** REST backend (with auto-generated Swagger docs) for
integrations. Pluggable LLM backends (Groq, OpenAI, or fully local Ollama).

> Designed as a portfolio / resume project: small, well-structured, easy to
> demo, and uses only free tools out-of-the-box.

---

## Features

- **Conversational, multi-turn RAG** with history-aware question rewriting.
- **Pluggable LLM provider** - switch between Groq (free & fast), OpenAI, or
  Ollama (fully local) via a single env var.
- **Local embeddings** with `sentence-transformers/all-MiniLM-L6-v2` - no
  embeddings API key required.
- **Persistent vector store** powered by ChromaDB.
- **Multiple document types**: PDF, TXT, Markdown, DOCX, and arbitrary URLs.
- **Strict, citation-aware system prompt** that refuses to answer when the
  context is insufficient (anti-hallucination guardrail).
- **Cited sources** displayed alongside every answer in the UI.
- **Three independent entry points** sharing the same core: a **Streamlit UI**,
  a **FastAPI REST backend** (with auto-generated Swagger docs at `/docs`),
  and a **CLI** (`python cli.py chat`).
- Fully type-hinted, modular code with **16 unit / integration tests** (`pytest`).

---

## Architecture

```
            +-----------------+         +-------------------+
 PDF / URL  |  Document       |         |  Recursive        |
 / TXT  ───▶|  Loaders        ├────────▶|  Text Splitter    |
            +-----------------+         +-------------------+
                                                 │
                                                 ▼
                                       +-------------------+
                                       |  Sentence-Transf. |
                                       |  Embeddings       |
                                       +---------┬---------+
                                                 ▼
                                       +-------------------+
                                       |  ChromaDB         |
                                       |  (persistent)     |
                                       +---------┬---------+
                                                 │ top-K retrieval
                                                 ▼
   User question ─▶ [history-aware rewrite] ─▶ Retriever ─▶ Prompt + Context
                                                                │
                                                                ▼
                                                  +--------------------------+
                                                  |  LLM (Groq/OpenAI/Ollama)|
                                                  +-------------┬------------+
                                                                ▼
                                                       Grounded answer + sources
```

---

## Tech stack

| Layer            | Choice                                            |
| ---------------- | ------------------------------------------------- |
| Orchestration    | LangChain (LCEL)                                  |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (local)  |
| Vector store     | ChromaDB (persistent on disk)                     |
| LLMs             | Groq · OpenAI · Ollama (configurable)             |
| Document loaders | pypdf · python-docx · BeautifulSoup4              |
| UI               | Streamlit                                         |
| REST backend     | FastAPI + Uvicorn (auto Swagger docs)             |
| Config           | Pydantic Settings + `.env`                        |
| Tests            | pytest + FastAPI TestClient                       |

---

## Quick start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd llm-rag-chatbot

python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set **one** LLM provider:

- **Groq** (recommended - free & fast). Get a key at <https://console.groq.com/keys>:

  ```env
  LLM_PROVIDER=groq
  GROQ_API_KEY=gsk_xxx...
  ```

- **OpenAI**:

  ```env
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-xxx...
  ```

- **Ollama** (fully local - run `ollama pull llama3.2` first):

  ```env
  LLM_PROVIDER=ollama
  OLLAMA_MODEL=llama3.2
  ```

### 3. Launch the UI

```bash
streamlit run app.py
```

Open <http://localhost:8501>, upload a PDF (or paste URLs) in the sidebar,
then ask questions in the chat.

### 4. Or launch the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
```

Then open:

- **Swagger UI:** <http://localhost:8000/docs>  (interactive try-it-out)
- **ReDoc:**     <http://localhost:8000/redoc>

#### Endpoints

| Method | Path              | Purpose                                                           |
| ------ | ----------------- | ----------------------------------------------------------------- |
| GET    | `/health`         | Liveness probe                                                    |
| GET    | `/status`         | Vector count, provider, embedding model, RAG hyperparameters      |
| POST   | `/ingest/files`   | Multipart upload of one or more PDF / TXT / MD / DOCX files       |
| POST   | `/ingest/urls`    | Body `{"urls": ["https://..."]}` - fetches and ingests each URL   |
| POST   | `/chat`           | Body `{"question": "...", "history": [{role, content}, ...]}`     |
| DELETE | `/collection`     | Wipes the vector store                                            |

#### Example `curl` calls

```bash
# Health
curl http://localhost:8000/health

# Upload a PDF
curl -X POST http://localhost:8000/ingest/files \
     -F "files=@./data/handbook.pdf"

# Ask a question
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is RAG?", "history": []}'
```

The Streamlit UI and the FastAPI backend are **independent** entry points
that both call the same `src/` core, so you can run either or both at the
same time on different ports (`8501` and `8000`).

### 5. Or use the CLI

```bash
# Ingest the bundled sample document
python cli.py ingest --path ./data/sample.md

# Ingest a folder of PDFs
python cli.py ingest --path ./data

# Ingest URLs
python cli.py ingest --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# Start an interactive chat
python cli.py chat

# Inspect or wipe the vector store
python cli.py status
python cli.py reset
```

---

## Project layout

```
llm-rag-chatbot/
├── app.py                     # Streamlit UI                (port 8501)
├── cli.py                     # Command-line interface
├── api/                       # FastAPI REST backend         (port 8000)
│   ├── __init__.py
│   ├── main.py                # App + routes
│   └── schemas.py             # Pydantic request/response models
├── src/                       # Shared core (used by all 3 entry points)
│   ├── config.py              # Pydantic settings
│   ├── document_loader.py     # PDF / TXT / DOCX / URL loaders
│   ├── text_splitter.py       # Recursive character splitter
│   ├── embeddings.py          # HuggingFace embeddings factory
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── llm.py                 # Multi-provider LLM factory
│   ├── prompts.py             # System + condense prompts
│   ├── rag_chain.py           # Conversational RAG pipeline
│   └── ingest.py              # File/URL → chunks → vector store
├── tests/
│   ├── test_config.py
│   ├── test_document_loader.py
│   ├── test_text_splitter.py
│   └── test_api.py            # FastAPI TestClient tests
├── requirements.txt
├── .env.example
├── README.md
├── data/                      # Drop your documents here
└── chroma_db/                 # Persistent vector store (auto-created)
```

---

## How it works

1. **Ingestion.** Files are parsed into LangChain `Document` objects, split
   into overlapping chunks (default 1000 chars / 200 overlap), embedded with
   a local sentence-transformer, and persisted in ChromaDB.
2. **History-aware retrieval.** When a user asks a follow-up like *"and the
   second one?"*, the chain first rewrites it into a standalone question
   using the chat history before retrieving.
3. **Top-K retrieval.** The standalone question is embedded and the K most
   similar chunks are pulled from Chroma.
4. **Grounded generation.** Retrieved chunks are injected into a strict
   system prompt that instructs the LLM to answer **only** from the context
   and cite the source filename or URL. If the context is insufficient, the
   model is told to say so explicitly.
5. **Sources surfaced.** The original retrieved chunks (with metadata such
   as page numbers) are returned alongside the answer and shown in the UI.

---

## Running the tests

```bash
pytest -q
```

The tests cover the splitter, document loaders, and configuration. (LLM and
vector-store integration tests are intentionally omitted to keep the suite
hermetic and fast.)

---

## Tuning

All hyperparameters live in `.env`:

| Variable        | Default | Purpose                                  |
| --------------- | ------- | ---------------------------------------- |
| `CHUNK_SIZE`    | 1000    | Characters per chunk                     |
| `CHUNK_OVERLAP` | 200     | Overlap between adjacent chunks          |
| `TOP_K`         | 4       | Chunks retrieved per query               |
| `TEMPERATURE`   | 0.2     | LLM sampling temperature                 |
| `MAX_TOKENS`    | 1024    | Maximum tokens in the generated answer   |

---

## Roadmap / ideas

- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Re-ranking with a cross-encoder
- [ ] Streaming responses in the Streamlit UI
- [ ] Per-user collections / multi-tenancy
- [ ] Dockerfile + `docker compose`
- [ ] Evaluation harness (Ragas)

---

## License

MIT - feel free to fork, extend, and use in your own portfolio.
