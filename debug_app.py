import os

import uvicorn

from utils.args import get_host_port_args
from utils.init import init

if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))
    uvicorn.run("app:app", host=host, port=port, workers=workers)
