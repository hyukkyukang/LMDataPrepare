import logging
import os
from typing import Optional


def log_if_rank_zero(logger: logging.Logger, message: str, level: str = "info") -> None:
    # Data-prep scripts are single-process by default; keep API-compatible helper.
    level_name: str = str(level).lower()
    level_map: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger.log(level_map.get(level_name, logging.INFO), message)


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    _ = file_path
    logger: logging.Logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
    root_logger: logging.Logger = logging.getLogger()
    if not root_logger.handlers:
        level: str = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    return logger


def setup_tqdm_friendly_logging() -> None:
    logging.getLogger("tqdm").setLevel(logging.WARNING)


def suppress_httpx_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)


def suppress_dataloader_workers_warning() -> None:
    logging.getLogger("torch.utils.data").setLevel(logging.WARNING)
