import logging
import os

from pydantic import BaseModel

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class LogConfig(BaseModel):
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
        "aidial_adapter_openai": {"handlers": ["default"], "level": LOG_LEVEL},
        "uvicorn": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    }


logger = logging.getLogger("aidial_adapter_openai")
