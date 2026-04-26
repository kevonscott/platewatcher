"""Entry point for the LLM platewatcher - license plate extractor.

The process runs indefinitely, watching for new event directories and
processing them via the vision workflow.
"""

import argparse
import logging
import sys

from platewatcher.config import Config
from platewatcher.constants import DEFAULT_LLM_MODEL
from platewatcher.database import Database
from platewatcher.event_processor import EventProcessor
from platewatcher.image_processor import ImageProcessor
from platewatcher.ollama_client import (
    LLMEndpointUnavailableError,
    LLMModelUnavailableError,
    OllamaClient,
)
from platewatcher.watcher import DirectoryWatcher


def setup_logging(config: Config, level: int | str) -> None:
    """Configure root logger handlers and formatting.

    Parameters
    ----------
    config
        Application configuration.
    level
        Logging level to configure.

    Returns
    -------
        Configures the global logging system.
    """
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if config.log_to_file:
        handlers.append(logging.FileHandler(config.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=fmt, datefmt=datefmt, handlers=handlers
    )


def get_watcher(config: Config, llm_client: OllamaClient) -> DirectoryWatcher:
    """Wire dependencies and build the directory watcher.

    Parameters
    ----------
    config
        Application configuration.

    Returns
    -------
        Ready-to-run watcher instance.
    """
    db = Database(config)
    image_processor = ImageProcessor(config)
    event_processor = EventProcessor(config, db, llm_client, image_processor)
    return DirectoryWatcher(config, db, event_processor)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
        Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        prog="platewatcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        help="Logging level",
        choices=logging._nameToLevel.keys(),
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_LLM_MODEL,
        help="Vision model to use for prediction",
    )
    return parser.parse_args()


def _ensure_working_directories(config: Config) -> None:
    """Create required runtime directories before startup.

    Parameters
    ----------
    config
        Application configuration.

    Returns
    -------
        Ensures event and log directories exist.
    """
    config.events_dir.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the platewatcher application entrypoint.

    Returns
    -------
        Initializes services and starts the watcher loop.
    """
    args = _parse_args()
    config = Config(vision_model=args.model)
    _ensure_working_directories(config)
    setup_logging(config, level=args.log_level)

    logger = logging.getLogger(__name__)

    logger.info("PlateWatcher starting.")
    logger.info("  Events dir  : %s", config.events_dir.resolve())
    logger.info("  Database    : %s", config.db_url)
    logger.info("  LLM URL  : %s", config.llm_base_url)
    logger.info("  Model       : %s", config.vision_model)
    logger.info("  Poll every  : %ds", config.poll_interval_seconds)

    llm_client = OllamaClient(config)
    try:
        llm_client.ensure_endpoint_reachable()
    except LLMEndpointUnavailableError as exc:
        logger.error("Startup aborted: %s", exc)
        sys.exit(1)

    watcher = get_watcher(config, llm_client)
    try:
        watcher.run()
    except (LLMEndpointUnavailableError, LLMModelUnavailableError) as exc:
        logger.error("Shutdown due to LLM unavailability: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C). Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
