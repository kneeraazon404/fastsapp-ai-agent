# FastsApp AI Agent

A production-grade WhatsApp customer-service bot for **Adventist Medical Centre (Hong Kong)**.
Powered by **OpenAI**, **Cohere Rerank**, **ChromaDB**, **PostgreSQL**, and **Twilio WhatsApp**,
with five built-in agentic features.

---

## Architecture

```
Twilio WhatsApp
      │
      ▼ POST /message
┌─────────────────────────────────────────────┐
│  FastAPI webhook handler                    │
│                                             │
│  1. Intent + Sentiment Classification  ◄──── Feature 2
│  2. Load per-user Conversation History ◄──── Feature 1
│  3. Summarise long history (if needed) ◄──── Feature 3
│  4. Query ChromaDB → Cohere Rerank v4       │
│  5. Generate response via OpenAI            │
│     └─ Appointment path if intent=appt ◄──── Feature 5
│  6. Append escalation footer (if needed) ◄── Feature 4
│  7. Persist to PostgreSQL                   │
│  8. Send reply via Twilio                   │
└─────────────────────────────────────────────┘
      │
      ▼
  PostgreSQL (conversations table)
  ChromaDB   (persistent FAQ vectorstore)
```

### Module layout

```
fastsapp-ai-agent/
├── main.py                       # FastAPI app + lifespan startup
├── pyproject.toml                # Project metadata, pinned deps, pytest config
├── requirements.txt              # Flat install list
├── app/
│   ├── config.py                 # pydantic-settings centralised config
│   ├── database.py               # SQLAlchemy engine/session (psycopg3 dialect)
│   ├── models.py                 # Conversation ORM model
│   ├── routes/
│   │   ├── health.py             # GET /health
│   │   └── webhook.py            # POST /message
│   └── services/
│       ├── ai_service.py         # OpenAI chat completion
│       ├── vectorstore.py        # ChromaDB + Cohere rerank
│       ├── conversation_store.py # Feature 1 — per-user history
│       ├── intent_classifier.py  # Feature 2 — intent + sentiment
│       ├── summarizer.py         # Feature 3 — history summarisation
│       ├── sentiment_analyzer.py # Feature 4 — escalation
│       └── entity_extractor.py   # Feature 5 — appointment entity flow
├── content/
│   └── adventistFAQ.csv          # FAQ source data for RAG ingestion
├── tests/                        # Full mocked test suite (41 tests)
└── .env.example                  # Environment variable template
```

---

## Five Agentic Features

| # | Feature | Description |
|---|---|---|
| 1 | **Per-user Memory** | Every exchange stored in PostgreSQL; latest N turns injected into each prompt |
| 2 | **Intent Classification** | Single OpenAI call classifies intent (`faq`, `appointment`, `complaint`, `emergency`, `greeting`, `farewell`, `other`) + sentiment score |
| 3 | **Conversation Summarisation** | Older turns condensed into a compact context block when history exceeds threshold |
| 4 | **Sentiment Escalation** | Negative/urgent conversations flagged in DB, given a human-contact footer, and `WARNING`-logged |
| 5 | **Appointment Entity Flow** | Guided booking: confirms when all fields are present, asks for missing ones otherwise |

---

## Stack baseline

| Component | Recommended | Minimum | Notes |
|---|---|---|---|
| Python | **3.13.12** | 3.11 | 3.14.3 is latest stable but this repo suppresses a known Chroma deprecation on 3.14; 3.13.x is the safer production target |
| PostgreSQL | **17** | 14 | 18 is the newest major. PostgreSQL has no official "LTS" label — each major release gets 5 years of support |
| OpenAI model | `gpt-4o-mini` | — | Still documented with 128k context / 16,384 output tokens |
| Cohere rerank | `rerank-v4.0-fast` | — | v4.0 is the current family (32k context, 100+ languages); use `rerank-v4.0-pro` for higher-quality ranking |
| Embedding model | `text-embedding-3-small` | — | Current small embedding model |

---

## Prerequisites

- Python 3.13.12 (recommended) or 3.11+
- PostgreSQL 17 (recommended) or 14+
- Twilio account — use the [WhatsApp Sandbox](https://www.twilio.com/docs/whatsapp/sandbox) for dev, a WhatsApp-enabled number for production
- OpenAI API key (`gpt-4o-mini` access)
- Cohere API key

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/kneeraazon404/fastsapp-ai-agent.git
cd fastsapp-ai-agent
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Fill in your API keys and database credentials
```

**Required variables:**

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `COHERE_API_KEY` | Cohere API key |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_NUMBER` | WhatsApp-enabled Twilio number (e.g. `+14155238886`) |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | PostgreSQL host (default: `localhost`) |
| `DB_NAME` | Database name (default: `fastsapp_ai_agent`) |

### 4. Create the database

```bash
psql -U postgres -c "CREATE DATABASE fastsapp_ai_agent;"
```

The `conversations` table is created automatically on first startup.

### 5. Start the server

```bash
uvicorn main:app --reload --port 8000
```

The FAQ CSV is ingested into ChromaDB on first startup and skipped on subsequent restarts.

### 6. Expose for local Twilio testing

```bash
ngrok http 8000
```

Set `https://<ngrok-id>.ngrok.io/message` as the Twilio WhatsApp webhook URL.

---

## Configuration reference

All values have defaults. Override in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MAX_COMPLETION_TOKENS` | `512` | Max reply tokens |
| `TEMPERATURE` | `0.7` | Response creativity |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-fast` | Rerank model (`rerank-v4.0-pro` for higher quality) |
| `COHERE_RERANK_TOP_N` | `3` | Documents kept after rerank |
| `VECTORSTORE_QUERY_N` | `8` | Candidate documents from ChromaDB |
| `VECTORSTORE_PATH` | `./vectorstore` | ChromaDB persistence directory |
| `CONTENT_CSV_PATH` | `./content/adventistFAQ.csv` | FAQ source data |
| `MAX_HISTORY_MESSAGES` | `10` | Prior turns included in each prompt |
| `SUMMARIZE_THRESHOLD` | `8` | Message pairs before summarisation triggers |
| `SYSTEM_PROMPT` | (Chinese hospital persona) | Bot persona |
| `ESCALATION_SENTIMENT_THRESHOLD` | `-0.5` | Score below which escalation fires |
| `ESCALATION_CONTACT_PHONE` | `` | Phone to notify on escalation |

---

## Dependency snapshot

**Runtime**

| Package | Version |
|---|---|
| `fastapi` | 0.135.3 |
| `uvicorn[standard]` | 0.44.0 |
| `python-multipart` | 0.0.24 |
| `openai` | 2.30.0 |
| `cohere` | 5.21.1 |
| `chromadb` | 1.5.6 |
| `sqlalchemy` | 2.0.49 |
| `psycopg[binary]` | 3.3.3 |
| `pydantic` | 2.12.5 |
| `pydantic-settings` | 2.13.1 |
| `twilio` | 9.10.4 |

**Dev / test**

| Package | Version |
|---|---|
| `pytest` | 9.0.2 |
| `pytest-asyncio` | 1.3.0 |
| `httpx` | 0.28.1 |

All packages are pinned to their current latest versions as of April 2026.

---

## Testing

```bash
pytest tests/ -v
```

41 tests, no external services required — all API calls are mocked. Deprecation warnings in project code are treated as hard errors; a targeted suppression is in place for the known `asyncio.iscoroutinefunction` deprecation in chromadb's telemetry internals (third-party, not ours). This is the main reason **Python 3.13.x** is the recommended production target over 3.14.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/message` | Twilio webhook — `Body` + `From` form fields |

---

## Production deployment

1. Use managed PostgreSQL 17 (e.g. Neon, Supabase, RDS).
2. Mount persistent storage for `VECTORSTORE_PATH`.
3. Store all secrets in your platform's secret manager — never commit `.env`.
4. Run behind TLS (nginx, Caddy, or platform ingress).
5. Multi-worker: `uvicorn main:app --workers 2 --host 0.0.0.0 --port 8000`
6. Stay on Python **3.13.x** in production until you have fully validated 3.14 across the whole stack.

---

## Notable upgrade decisions

**psycopg2 → psycopg3** — actively maintained successor with native binary protocol and Python 3.12+ improvements. SQLAlchemy 2.x supports it via the `postgresql+psycopg` dialect.

**openai SDK 1.x → 2.x** — `max_tokens` replaced by `max_completion_tokens`. Chat Completions remains available; the OpenAI Responses API is the recommended path for new integrations.

**rerank-multilingual-v3.0 → rerank-v4.0-fast** — Cohere v4.0 is the current rerank family: unified multilingual model (no separate variant needed), 32k context, state-of-the-art on BEIR and domain-specific retrieval including hospitality/healthcare.

**PostgreSQL versioning** — PostgreSQL does not use an "LTS" label. Each major version receives 5 years of community support. PostgreSQL 17 is the recommended baseline; 18 is the newest major.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
