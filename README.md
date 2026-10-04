# Production RAG Platform with Guardrails & Observability

Ask questions about a company's private documents and get **cited answers**, with **proof of where each
answer came from**. Hybrid retrieval, cross-encoder reranking, three guardrails against unsafe prompts and
hallucinations, and a trace of every step with latency and cost.

**Live demo:** [abhirama.tech/rag](https://abhirama.tech/rag) · runs on Google Cloud Run, scales to zero, costs ~$0/month.

**Stack:** Python · FastAPI · LangChain · fastembed (ONNX) · Qdrant / NumPy · Gemini / OpenAI · Langfuse · Docker · Cloud Run · GitHub Actions

Why RAG? A public chatbot has never seen a company's internal documents, so it can't answer
*"What caused incident INC-2291?"*. This system finds the right internal passage at question time, answers from it,
cites it, and checks that the answer actually matches it. The sample corpus is the handbook of **Quillfeather Labs**,
a fictional company written for this project, so no public model knows the answers.

---

## How it works

```
question
  ├─ 1. Input guardrail ...... block prompt injection & unsafe requests, redact emails / phone numbers
  ├─ 2. Hybrid retrieval
  │      ├─ dense search (bge-small embeddings) .... finds by meaning
  │      ├─ BM25 keyword search .................... finds exact terms ("INC-2291", "Tollbooth")
  │      └─ Reciprocal Rank Fusion ................. merges both lists by rank
  ├─ 3. Rerank ............... cross-encoder rereads (question, passage) pairs, keeps the best 4
  ├─ 4. Relevance guardrail .. best score too low? -> "I don't know" instead of guessing
  ├─ 5. Generate ............. LLM writes the answer from those passages only, citing [1] [2]
  │                            fallback chain: visitor's OpenAI key -> pool of free Gemini keys
  │                            -> (optional) server OpenAI key -> extractive answer (no LLM)
  ├─ 6. Grounding guardrail .. every answer sentence checked against the sources -> hallucination flag
  └─ Trace ................... each step timed; tokens + $ cost -> UI, /api/metrics, Cloud Logging, Langfuse
```

## The live demo

- **See the evidence.** Every citation links to its passage; *Show in document* opens the original document
  with the cited passage highlighted.
- **Bring your own documents.** Visitors can remove sample documents and upload `.txt` `.md` `.pdf` `.docx` files.
  Changes go into a private, in-memory copy that only their browser tab knows (copy-on-write). Reloading, *Restore
  originals*, or 30 idle minutes brings back the samples. Questions about uploads are never logged or exported.
- **Bring your own key.** An OpenAI key typed into the page is sent with each question and never stored or logged
  (key fragments are scrubbed from error messages). Without one, a rotating pool of free Gemini keys answers;
  a key that hits its quota is benched for a minute. If every LLM fails, the extractive fallback still answers.
- **Guardrails you can poke.** The dashed sample questions trigger PII redaction, the relevance refusal and
  prompt-injection blocking.

## Built for serverless hosting

| Concern | What the app does |
|---|---|
| **Cold starts** | Models are downloaded and the sample documents embedded **at Docker build time** (`python -m app.build_index`). Startup loads two ONNX models and a few KB of vectors: no embedding, no network. |
| **No servers to run** | Dense search is an in-memory NumPy index: for thousands of chunks it's faster than a network hop to a vector DB, and it builds in milliseconds for each visitor's private set. Set `QDRANT_URL` to serve the shared index from Qdrant instead (docker-compose does). |
| **Memory** | Cloud Run's disk lives in RAM, so traces go to **stdout as JSON** (Cloud Logging) instead of a growing file. Visitor sessions expire and are capped (LRU). |
| **Cost** | Embeddings, reranking and guardrails run locally on CPU. Repeated questions about the samples are answered from an **in-memory cache**. A **per-IP rate limit** protects the free LLM quota. Request-based billing + min instances 0 keeps it inside the free tier. |
| **Speed** | CPU-heavy steps run in a worker thread; responses are gzipped; uvicorn with uvloop. |
| **Security** | Non-root container, no secrets in the image (`.dockerignore` excludes `.env`), keys from Secret Manager. |

### Deploy to Cloud Run

1. Cloud Run → **Create service** → *Continuously deploy from a repository* → this repo, branch `main`, **Dockerfile**.
2. Settings: allow public access · request-based billing · min instances **0**, max instances **1**
   (sessions and the cache live in memory) · container port **8000** · memory 2 GiB · CPU 1 · startup CPU boost on.
3. Variables & Secrets: `GEMINI_API_KEYS` as a Secret Manager secret (comma-separated keys).
   Leave `OPENAI_API_KEY` unset unless you want to pay for visitors' questions.
4. Every push to `main` rebuilds and redeploys automatically.

## Run locally (Mac)

Requires Python 3.10–3.12.

```bash
cd rag-platform
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # optional: add GEMINI_API_KEYS or OPENAI_API_KEY
python -m app.build_index         # first time: downloads models (~150 MB) and builds data/index/
uvicorn app.api:app --reload      # http://localhost:8000
```

Add or edit documents in `data/docs/` (`.md`, `.txt`, `.rst`); the index rebuilds automatically on the next start.

```bash
docker compose up --build         # API + Qdrant server (http://localhost:6333/dashboard)
pytest -q && ruff check .         # tests use fake models: ~1 second, no downloads
```

## Project structure

```
app/
├── pipeline.py      <- START HERE: the six steps above
├── text.py          load / clean / chunk documents; extract text from uploads
├── embeddings.py    text -> vectors (fastembed, ONNX on CPU)
├── index.py         hybrid index: NumPy or Qdrant dense search + BM25 + Reciprocal Rank Fusion
├── bm25.py          readable BM25 implementation
├── reranker.py      cross-encoder reranking
├── guardrails.py    input / relevance / grounding checks
├── llm.py           LLM fallback chain, Gemini key pool, cost, key redaction
├── workspace.py     prebuilt sample index + per-visitor sandboxes
├── protect.py       rate limiter + answer cache
├── tracing.py       traces, metrics (p50/p95, cost), Cloud Logging, Langfuse
├── services.py      wires everything together at startup
├── build_index.py   build-time step: download models, embed samples
├── api.py           FastAPI endpoints
└── static/index.html  demo UI (single file, no build step)
data/docs/           sample corpus (Quillfeather Labs)
tests/               pytest with fake models
loadtest/            Locust load test
```

## API

| Method | Path | Returns |
|---|---|---|
| POST | `/api/ask` `{"question", "session_id"?}` + optional `X-OpenAI-Key` header | answer, sources, guardrail results, trace |
| GET | `/api/state?session=…` | the documents that session searches, sandbox limits, available LLMs |
| GET | `/api/documents/{id}?session=…` | one document's full text |
| POST | `/api/sessions/{id}/documents` `{"filename", "content_base64"}` | add a file to the visitor's private copy |
| DELETE | `/api/sessions/{id}/documents/{doc_id}` | remove a document from the private copy |
| POST | `/api/sessions/{id}/reset` | back to the original samples |
| GET | `/api/metrics` | requests, cache hits, blocked, hallucination flags, p50/p95 latency, tokens, cost |
| GET | `/api/traces?session=…` | recent traces (private ones only for their own session) |
| GET | `/health`, `/`, `/docs` | liveness, demo UI, OpenAPI docs |

## Demo script (3 minutes)

| Step | Do this | What it shows |
|---|---|---|
| 1 | Ask ChatGPT *"What is Project Kestrel and when does it launch?"* | A public LLM can't know private company data |
| 2 | Click the same question in the app | Exact answer with citation [1]; *Show in document* highlights the passage |
| 3 | *"What caused incident INC-2291?"* | "found by word search": BM25 matched the exact ID that vector search can miss |
| 4 | *"My email is jane.doe@example.com. What is my yearly learning budget?"* | Input guardrail redacts the email, still answers |
| 5 | *"What is the weather in Paris today?"* | Relevance guardrail declines instead of guessing |
| 6 | *"Ignore all previous instructions…"* | Prompt injection blocked before retrieval (trace has one step) |
| 7 | Upload your own PDF and ask about it | The whole pipeline on new data, in a private sandbox |

More questions with answers: `DEMO_QUESTIONS.md`.

## Design decisions

- **Hybrid retrieval:** embeddings miss exact IDs and names; BM25 misses paraphrases. RRF merges by rank, so scores never need normalising.
- **Rerank:** retrievers are fast but approximate; a cross-encoder reads question and passage together, so it only scores ~20 candidates.
- **Three guardrails:** stop bad input early (cheap), refuse when nothing relevant was found (prevents most hallucinations), verify the output (catches the rest).
- **Grounding check:** each answer sentence must be semantically close to a source sentence (cosine ≥ 0.78) or share ≥ 80% of its content words.
- **NumPy vs Qdrant:** a vector database earns its keep at millions of vectors or many replicas. At this size an in-memory index is faster, cheaper and simpler, so it's the default; Qdrant is one environment variable away.
- **Fallback chain:** the demo must never break. A missing, invalid or rate-limited key degrades the wording of the answer, not the retrieval, citations or guardrails.

## Limitations / next steps

- Guardrail patterns are rule-based; production would add a moderation model (e.g. Llama Guard).
- An evaluation set (questions + expected sources) to track retrieval hit-rate and answer quality.
- Streaming answers. Sessions and cache live in one instance's memory; scaling out would move them to Redis.

Quillfeather Labs is fictional; its documents were written for this project. Any resemblance to a real company is coincidental.
