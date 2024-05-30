import logging
import os
import re
import sys
from logging import Filter, LogRecord

from aidial_sdk import logger as aidial_logger
from uvicorn.logging import DefaultFormatter

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DIAL_LOG_LEVEL = os.getenv("DIAL_LOG_LEVEL", "WARNING")
aidial_logger.setLevel(DIAL_LOG_LEVEL)


class HealthCheckFilter(Filter):
    def filter(self, record: LogRecord):
        return not re.search(r"(\s+)/health(\s+)", record.getMessage())


def configure_loggers():
    # Making the uvicorn logger delegate logging to the root logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True

    # Filter out health check requests from uvicorn logs
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # Setting up log levels
    for name in ["aidial_adapter_openai", "uvicorn"]:
        logging.getLogger(name).setLevel(LOG_LEVEL)

    # Configuring the root logger
    root = logging.getLogger()

    root_has_stderr_handler = any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream == sys.stderr
        for handler in root.handlers
    )

    # If stderr handler is already set, then no need to add another one
    if not root_has_stderr_handler:
        formatter = DefaultFormatter(
            fmt="%(levelprefix)s | %(asctime)s | %(name)s | %(process)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=True,
        )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)


logger = logging.getLogger("aidial_adapter_openai")
