#!/bin/sh
set -e

. ./.venv/bin/activate

# If no args passed to `docker run`,
# then we assume the user is calling the adapter server
if [ $# -lt 1 ]; then
  exec uvicorn aidial_adapter_openai.app:app \
    --host 0.0.0.0 \
    --port 5000 \
    --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE:-5}"
fi

# Otherwise, we assume the user wants to run his own process,
# for example a `bash` shell to explore the container
exec "$@"