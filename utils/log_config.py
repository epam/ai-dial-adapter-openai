import logging
import os

from pydantic import BaseModel

# By default (in prod) we don't want to print debug messages,
# because they typically contain prompts.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s | %(asctime)s | %(name)s | %(process)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers = {
        "app": {"handlers": ["default"], "level": LOG_LEVEL},
        "vertex-ai": {"handlers": ["default"], "level": LOG_LEVEL},
        "uvicorn": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    }


# Loggers in order from high-level to low-level
# High-level logs from the adapter server
app_logger = logging.getLogger("app")

# LLM requests and responses
vertex_ai_logger = logging.getLogger("vertex-ai")
