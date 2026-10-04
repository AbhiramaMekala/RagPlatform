# Production image for Cloud Run (or any container host).
# Everything slow happens here, at build time, so a cold start only loads files:
#   1. install dependencies      2. download the two ONNX models      3. embed the sample documents
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/models
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# run as an unprivileged user (created before the big layers, so no costly chown afterwards)
RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models /app
USER app

# models layer: rebuilt only when requirements change
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY --chown=app app ./app
COPY --chown=app data/docs ./data/docs
RUN python -m app.build_index

# Cloud Run sets $PORT (8000 here matches the service's container port). One worker on purpose:
# visitor sessions and the answer cache live in this process's memory.
ENV PORT=8000
EXPOSE 8000
CMD exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75 --no-access-log
