#!/bin/sh
set -e

if [ -n "${USE_SYSTEM_CA_CERTS:-}" ]; then
  TMPDIR="/tmp"
  SYSTEM_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

  if [ ! -w "$TMPDIR" ]; then
    echo "Using additional CA certificates requires write permissions to $TMPDIR." >&2
    exit 1
  fi

  if [ ! -r "$SYSTEM_BUNDLE" ]; then
    echo "System CA bundle not found at $SYSTEM_BUNDLE (did you install ca-certificates?)." >&2
    exit 1
  fi

  OUT_BUNDLE="$(mktemp "${TMPDIR}/ca-bundle.XXXXXX")" || {
    echo "Failed to create temp CA bundle in $TMPDIR" >&2
    exit 1
  }

  # Append the system trust store
  cat "$SYSTEM_BUNDLE" > "$OUT_BUNDLE"

  # Append mounted custom certs
  if [ -d /certificates ]; then
    for f in /certificates/*.crt; do
      [ -f "$f" ] || continue
      echo >> "$OUT_BUNDLE"
      cat "$f" >> "$OUT_BUNDLE"
    done
  fi

  # Tell Python HTTP client (httpx) to use the merged bundle
  export SSL_CERT_FILE="$OUT_BUNDLE"
  echo "Using CA bundle at $OUT_BUNDLE"
fi
