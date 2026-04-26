"""Central configuration for the platewatcher (LLM license plate extractor)."""

import os
from dataclasses import dataclass
from pathlib import Path

from platewatcher.constants import APP_NAME, DEFAULT_LLM_MODEL, DEFAULT_LLM_URL

PLATEWATCHER_APP_DIR = os.environ.get("PLATEWATCHER_APP_DIR", None)
PLATEWATCHER_EVENTS_DIR = os.environ.get("PLATEWATCHER_EVENTS_DIR", None)
PLATEWATCHER_DB_URL = os.environ.get("PLATEWATCHER_DB_URL", None)
PLATEWATCHER_LOG_FILE = os.environ.get("PLATEWATCHER_LOG_FILE", None)
PLATEWATCHER_LLM_TIMEOUT_SECONDS = os.environ.get(
    "PLATEWATCHER_LLM_TIMEOUT_SECONDS", None
)
PLATEWATCHER_POLL_INTERVAL_SECONDS = os.environ.get(
    "PLATEWATCHER_POLL_INTERVAL_SECONDS", None
)
PLATEWATCHER_MAX_IMAGE_WIDTH = os.environ.get(
    "PLATEWATCHER_MAX_IMAGE_WIDTH", None
)
PLATEWATCHER_MAX_IMAGE_HEIGHT = os.environ.get(
    "PLATEWATCHER_MAX_IMAGE_HEIGHT", None
)
PLATEWATCHER_MAX_IMAGES_PER_EVENT = os.environ.get(
    "PLATEWATCHER_MAX_IMAGES_PER_EVENT", None
)

DEFAULT_APP_DIR = Path(
    PLATEWATCHER_APP_DIR or f"~/.local/state/{APP_NAME}"
).expanduser()
DEFAULT_EVENTS_DIR = Path(
    PLATEWATCHER_EVENTS_DIR or str(DEFAULT_APP_DIR / "events")
)
DEFAULT_DB_URL = (
    PLATEWATCHER_DB_URL or f"sqlite:///{DEFAULT_APP_DIR / 'license_plates.db'}"
)
DEFAULT_LOG_FILE = Path(
    PLATEWATCHER_LOG_FILE or str(DEFAULT_APP_DIR / "platewatcher.log")
)


@dataclass
class Config:
    events_dir: Path = DEFAULT_EVENTS_DIR
    db_url: str = DEFAULT_DB_URL
    log_file: Path = DEFAULT_LOG_FILE
    log_to_file: bool = False

    llm_base_url: str = os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_URL
    vision_model: str = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL
    llm_timeout_seconds: int = int(PLATEWATCHER_LLM_TIMEOUT_SECONDS or "90")

    poll_interval_seconds: int = int(PLATEWATCHER_POLL_INTERVAL_SECONDS or "10")

    # Resize images to this before sending to the LLM — cuts RAM usage on Pi
    max_image_width: int = int(PLATEWATCHER_MAX_IMAGE_WIDTH or "1280")
    max_image_height: int = int(PLATEWATCHER_MAX_IMAGE_HEIGHT or "720")
    max_images_per_event: int = int(PLATEWATCHER_MAX_IMAGES_PER_EVENT or "10")
    # safety cap; more = slower
    supported_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Minimum and maximum cleaned alphanumeric length to accept as a real plate
    plate_min_length: int = 2
    plate_max_length: int = 12
