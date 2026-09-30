import threading
from typing import Any

import boto3

from aidial_adapter_openai.utils.concurrency import run_in_threadpool

# A single shared Session, so that every client reuses the botocore loader
# and its cache of parsed service models.
# Relevant discussion: https://github.com/boto/boto3/issues/1670
#
# NOTE: Session isn't thread-safe, but client is, and clients are created on
# worker threads (see `run_in_threadpool`), hence the lock.
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#caveats
_session = boto3.Session()
_session_lock = threading.Lock()


def create_client(service_name: str, **kwargs: Any) -> Any:
    with _session_lock:
        return _session.client(service_name, **kwargs)


async def close_client(client: Any) -> None:
    # Each client owns a connection pool of its own, which the shared Session
    # does nothing to pool together.
    await run_in_threadpool(client.close)
