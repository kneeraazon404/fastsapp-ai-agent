# FastsApp AI Agent

A production-grade WhatsApp customer-service bot for **Adventist Medical Centre (Hong Kong)**.
Powered by OpenAI GPT-4o-mini, Retrieval-Augmented Generation (ChromaDB + Cohere rerank), and Twilio,
with five built-in agentic intelligence features.

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
│  4. Query ChromaDB → Cohere rerank (RAG)    │
│  5. Generate response (OpenAI gpt-4o-mini)  │
│     └─ Appointment path if intent=appt ◄──── Feature 5
│  6. Append escalation footer (if needed) ◄── Feature 4
│  7. Persist to PostgreSQL                   │
│  8. Send reply via Twilio                   │
└─────────────────────────────────────────────┘
      │
      ▼
  PostgreSQL (conversations table) — via psycopg3
  ChromaDB   (FAQ vectorstore, persistent on disk)
```

### Module layout

```
fastsapp-ai-agent/
├── main.py                       # FastAPI app + lifespan startup
├── pyproject.toml                # Project metadata, pinned deps, pytest config
├── requirements.txt              # Flat install list (mirrors pyproject.toml deps)
├── app/
│   ├── config.py                 # Centralised pydantic-settings config
│   ├── database.py               # SQLAlchemy engine/session (psycopg3 dialect)
│   ├── models.py                 # Conversation ORM model
│   ├── routes/
│   │   ├── health.py             # GET /health
│   │   └── webhook.py            # POST /message (Twilio webhook)
│   └── services/
│       ├── ai_service.py         # OpenAI v2 chat completion
│       ├── vectorstore.py        # ChromaDB ingestion + Cohere rerank
│       ├── conversation_store.py # Feature 1: per-user history
│       ├── intent_classifier.py  # Feature 2: intent + sentiment
│       ├── summarizer.py         # Feature 3: long-context summarisation
│       ├── sentiment_analyzer.py # Feature 4: escalation handling
│       └── entity_extractor.py   # Feature 5: appointment entity flow
├── content/
│   └── adventistFAQ.csv          # Source FAQ documents for RAG
├── tests/                        # Full test suite (mocked, no external APIs)
└── .env.example                  # Environment variable template
```

---

## Five Agentic Features

### Feature 1 — Per-user Conversation Memory
Every exchange is stored in PostgreSQL keyed by the user's WhatsApp number.
The most-recent N turns are retrieved and included in each OpenAI prompt,
enabling genuine multi-turn contextual conversations.

### Feature 2 — Intent Classification & Routing
Before generating a reply, a fast `gpt-4o-mini` call classifies each
message into one of: `faq`, `appointment`, `complaint`, `emergency`,
`greeting`, `farewell`, `other`. The intent steers the response path
(e.g. appointment → entity extractor) and is stored for analytics.

### Feature 3 — Conversation Summarisation
When a user's history exceeds `SUMMARIZE_THRESHOLD` message pairs, older
turns are automatically summarised into a compact paragraph and prepended
as context. This keeps the prompt within token limits while preserving
long-running conversation continuity.

### Feature 4 — Sentiment Analysis & Human Escalation
The intent classifier scores sentiment on a `-1.0` to `+1.0` scale.
When the score drops below `ESCALATION_SENTIMENT_THRESHOLD` (default `-0.5`)
or the intent is `emergency` / `urgent`, the conversation is:
- flagged `escalated = True` in the database
- given a human-contact footer appended to the bot's reply
- emitted as a `WARNING` log for operator monitoring

### Feature 5 — Structured Entity Extraction & Appointment Handling
For `appointment`-intent messages, extracted entities (name, date, time,
service type) drive a guided booking flow:
- **All fields present** → confirmation response + next-steps instructions
- **Fields missing** → focused follow-up question for only the missing data

---

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (16 or 17 LTS recommended)
- A [Twilio](https://twilio.com) account with a WhatsApp-enabled number
- An [OpenAI](https://platform.openai.com) API key (GPT-4o-mini access)
- A [Cohere](https://cohere.com) API key

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/kneeraazon404/fastsapp-ai-agent.git
cd fastsapp-ai-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual API keys and database credentials
```

Required variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `COHERE_API_KEY` | Cohere API key |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_NUMBER` | Your Twilio WhatsApp number (e.g. `+14155238886`) |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | PostgreSQL host (default: `localhost`) |
| `DB_NAME` | Database name (default: `fastsapp_ai_agent`) |

### 4. Create the database

```bash
psql -U postgres -c "CREATE DATABASE fastsapp_ai_agent;"
```

The application creates the `conversations` table automatically at startup.

### 5. Start the server

```bash
uvicorn main:app --reload --port 8000
```

On first startup the FAQ CSV is automatically ingested into the ChromaDB
vectorstore (stored in `./vectorstore/`). Subsequent restarts skip re-ingestion.

### 6. Expose via ngrok (development)

```bash
ngrok http 8000
```

Set the generated HTTPS URL + `/message` as your Twilio WhatsApp webhook:
`https://<ngrok-id>.ngrok.io/message`

---

## Configuration reference

All settings have sensible defaults. Override any of them in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MAX_COMPLETION_TOKENS` | `512` | Max reply tokens (openai v2 param) |
| `TEMPERATURE` | `0.7` | Response creativity |
| `COHERE_RERANK_MODEL` | `rerank-multilingual-v3.0` | Cohere rerank model |
| `COHERE_RERANK_TOP_N` | `3` | Documents kept after rerank |
| `VECTORSTORE_QUERY_N` | `8` | Candidate documents from ChromaDB |
| `VECTORSTORE_PATH` | `./vectorstore` | ChromaDB persistence directory |
| `CONTENT_CSV_PATH` | `./content/adventistFAQ.csv` | FAQ source data |
| `MAX_HISTORY_MESSAGES` | `10` | Prior turns in each prompt |
| `SUMMARIZE_THRESHOLD` | `8` | Pairs before summarisation triggers |
| `SYSTEM_PROMPT` | (Chinese hospital persona) | Bot persona |
| `ESCALATION_SENTIMENT_THRESHOLD` | `-0.5` | Score below which escalation fires |
| `ESCALATION_CONTACT_PHONE` | `` | Phone to notify on escalation |

---

## Testing

```bash
pytest tests/ -v
```

The test suite requires **no external services** — all API calls are mocked.
`pyproject.toml` configures pytest to treat any `DeprecationWarning` or
`PendingDeprecationWarning` in project code as a hard error, with a targeted
exemption for the known `asyncio.iscoroutinefunction` deprecation inside
chromadb's telemetry internals (a third-party bug, not ours).

```
tests/test_ai_service.py          - OpenAI v2 response generation
tests/test_conversation_store.py  - Per-user history (Feature 1)
tests/test_intent_classifier.py   - Intent + sentiment (Feature 2)
tests/test_summarizer.py          - History summarisation (Feature 3)
tests/test_sentiment_analyzer.py  - Escalation handling (Feature 4)
tests/test_entity_extractor.py    - Appointment entity flow (Feature 5)
tests/test_vectorstore.py         - ChromaDB + Cohere rerank
tests/test_webhook.py             - Full webhook pipeline
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` — for uptime monitoring |
| `POST` | `/message` | Twilio webhook: `Body` + `From` form fields |

---

## Notable upgrade decisions

**psycopg2 → psycopg3 (`psycopg[binary]`)**
psycopg2 is in maintenance-only mode. psycopg3 is the actively developed
successor with native binary protocol, Python 3.12+ improvements, and
async-ready design. SQLAlchemy 2.x supports it via the `postgresql+psycopg`
dialect string in `app/database.py`.

**openai SDK 1.x → 2.x**
The v2 client deprecates `max_tokens` in favour of `max_completion_tokens`.
All four service files that call the completions API were updated accordingly.
The chat completion interface (`client.chat.completions.create()`,
`response.choices[0].message.content`) is otherwise unchanged.

**`pyproject.toml` added**
Replaces ad-hoc pytest configuration. Contains pinned dependency versions,
`asyncio_mode = "strict"`, and the `filterwarnings` rules that enforce zero
deprecations in project code while tolerating the chromadb telemetry issue.

---

## Production deployment

1. Use a managed PostgreSQL 16/17 service (e.g. RDS, Supabase, Neon).
2. Mount a persistent volume for `VECTORSTORE_PATH` so embeddings survive restarts.
3. Set all secrets via your hosting platform's secret manager (never commit `.env`).
4. Run behind a reverse proxy (nginx, Caddy) with TLS.
5. Use `uvicorn main:app --workers 2 --host 0.0.0.0 --port 8000` for multi-worker mode.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
