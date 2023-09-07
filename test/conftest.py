from multiprocessing import Process

import pytest
import uvicorn

from app import app
from utils.server import ping_server, wait_for_server

DEFAULT_API_VERSION = "2023-03-15-preview"
HOST = "0.0.0.0"
PORT = 5001

BASE_URL = f"http://{HOST}:{PORT}"


def run_server():
    uvicorn.run(app, host=HOST, port=PORT)


@pytest.fixture(scope="module")
def server():
    already_exists = ping_server(BASE_URL)

    server_process: Process | None = None
    if not already_exists:
        server_process = Process(target=run_server)
        server_process.start()

    assert wait_for_server(BASE_URL), "Server didn't start in time!"

    yield

    if server_process is not None:
        server_process.terminate()
        server_process.join()
