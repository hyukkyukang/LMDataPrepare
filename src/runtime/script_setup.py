import logging
import os
import warnings

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.runtime.logging import (
    log_if_rank_zero,
    setup_tqdm_friendly_logging,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
)
from src.runtime.seed import set_seed


def normalize_tag(tag: object | None) -> str | None:
    if tag is None:
        return None
    tag_value: str = str(tag).strip()
    if not tag_value:
        return None
    return tag_value


def _resolve_tagged_log_dir(log_dir_base: str, tag: str | None) -> str:
    tag_value: str | None = normalize_tag(tag)
    if tag_value is None:
        return log_dir_base
    return os.path.join(log_dir_base, tag_value)


def _register_tagged_log_dir_resolver() -> None:
    resolver_name: str = "tagged_log_dir"
    has_resolver: bool = False
    if hasattr(OmegaConf, "has_resolver"):
        has_resolver = OmegaConf.has_resolver(resolver_name)
    if has_resolver:
        return
    try:
        OmegaConf.register_new_resolver(resolver_name, _resolve_tagged_log_dir)
    except ValueError:
        return


def configure_script_environment(
    *,
    load_env: bool,
    set_tokenizers_parallelism: bool,
    set_matmul_precision: bool,
    suppress_lightning_tips: bool,
    suppress_httpx: bool,
    suppress_dataloader_workers: bool,
) -> None:
    _ = set_matmul_precision
    _ = suppress_lightning_tips
    _register_tagged_log_dir_resolver()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    logging.captureWarnings(True)

    if load_env:
        load_dotenv()
    if set_tokenizers_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if suppress_httpx:
        suppress_httpx_logging()
    if suppress_dataloader_workers:
        suppress_dataloader_workers_warning()


def initialize_run(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
    suppress_lightning_tips: bool = True,
) -> None:
    _ = suppress_lightning_tips
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(int(cfg.seed))
    log_if_rank_zero(logger, f"Random seed set to: {int(cfg.seed)}")
