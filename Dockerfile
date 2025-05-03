# ── common base image ─────────────────────────────────
FROM python:3.12-slim AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc build-essential libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

# ── crawler stage ─────────────────────
FROM base AS crawler

# Install Playwright (already satisfied) + all OS deps needed for headless Chromium
RUN pip install --no-cache-dir playwright \
 && playwright install-deps

# Download the Chromium browser bundle
RUN playwright install chromium

# ── init-db stage ─────────────────
FROM base AS init-db
# (no extra layers)

# ── app stage: only Streamlit + RAG code ───────────────────────────────────
FROM base AS app
EXPOSE 8501
CMD ["streamlit", "run", "context7_chat.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
