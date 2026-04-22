import logging
import json
import sys
from datetime import datetime, timezone


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"adversescore.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, event: str, **kwargs) -> None:
    entry = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}
    logger.info(json.dumps(entry, default=str))
