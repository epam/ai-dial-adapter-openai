FROM python:3.11-alpine AS builder
ARG TARGETARCH

RUN apk update && apk upgrade --no-cache libcrypto3 libssl3
# Installing Rust to build tiktoken from source on arm64
RUN apk add --no-cache alpine-sdk linux-headers \
    && if [ "$TARGETARCH" = "arm64" ] || [ "$(apk --print-arch)" = "aarch64" ]; then \
         apk add --no-cache rust cargo libffi-dev pkgconf openssl-dev; \
       fi
RUN pip install poetry==2.1.1

WORKDIR /app

# Install split into two steps (the dependencies and the sources)
# in order to leverage the Docker caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root \
  --no-directory --only main

# Download tiktoken model encodings
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
RUN .venv/bin/python -c "from tiktoken.model import (get_encoding as load, MODEL_TO_ENCODING as models); [(print(f'Loading tiktoken tokenizer {e}...'), load(e)) for e in set(models.values())]"

COPY aidial_adapter_openai aidial_adapter_openai
RUN poetry install --no-interaction --no-ansi --no-cache --only main

FROM python:3.11-alpine AS server
ARG TARGETARCH

RUN apk update && apk upgrade --no-cache libcrypto3 libssl3

# Runtime libs for arm64
RUN if [ "$TARGETARCH" = "arm64" ] || [ "$(apk --print-arch)" = "aarch64" ]; then \
      apk add --no-cache libffi libstdc++; \
    fi

# fix CVE-2023-52425
RUN apk upgrade --no-cache libexpat
# fix CVE-2025-47273
RUN pip install setuptools==78.1.1
# fix CVE-2025-6965
RUN apk upgrade --no-cache sqlite-libs

WORKDIR /app

# Copy the sources and virtual env. No poetry.
RUN adduser --uid 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder /app .

COPY ./scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

EXPOSE 5000

USER appuser
ENTRYPOINT ["/docker_entrypoint.sh"]

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache

CMD uvicorn aidial_adapter_openai.app:app --host 0.0.0.0 --port 5000 --timeout-keep-alive ${TIMEOUT_KEEP_ALIVE:-5}
