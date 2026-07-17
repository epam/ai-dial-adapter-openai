import logging
import os
import re
from logging import Filter, LogRecord

from aidial_sdk import LogConfig, configure_root_logger


class HealthCheckFilter(Filter):
    def filter(self, record: LogRecord):
        return not re.search(r"(\s+)/health(\s+)", record.getMessage())


def configure_loggers():
    # By default (in prod) we don't want to print debug messages,
    # because they typically contain prompts.
    app_log_level = os.getenv("LOG_LEVEL", "INFO")

    configure_root_logger(
        LogConfig(
            text_format="%(levelprefix)s | %(asctime)s | %(name)s | %(process)d | %(message)s"
        )
    )

    # Filter out health check requests from uvicorn logs
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # Setting up log levels
    for name in [
        "aidial_adapter_openai",
        "aidial_adapter_anthropic",
        "uvicorn",
    ]:
        logging.getLogger(name).setLevel(app_log_level)


logger = logging.getLogger("aidial_adapter_openai")
