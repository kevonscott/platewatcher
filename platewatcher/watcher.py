"""Polls the events directory for unprocessed event subdirectories.

Simple polling (time.sleep) over OS inotify/watchdog to avoid the need for
extra dependencies and is immune to missed events on high-frequency writes.

Event directories are sorted by name (timestamps) so the oldest events
are always processed first.

A single iteration processes ALL unprocessed events before sleeping again so
that a backlog after a restart is burned down quickly.
"""

import logging
import time
from pathlib import Path

from platewatcher.config import Config
from platewatcher.database import Database
from platewatcher.event_processor import EventProcessor
from platewatcher.ollama_client import (
    LLMEndpointUnavailableError,
    LLMModelUnavailableError,
)

logger = logging.getLogger(__name__)


class DirectoryWatcher:
    """Poll events directory and dispatch pending events for processing."""

    def __init__(
        self,
        config: Config,
        db: Database,
        processor: EventProcessor,
    ) -> None:
        self._events_dir = config.events_dir
        self._interval = config.poll_interval_seconds
        self._db = db
        self._processor = processor

    def run(self) -> None:
        """Run the watcher loop indefinitely.

        Returns
        -------
            Blocks and repeatedly scans for pending events.
        """
        logger.info(
            "Watcher started — polling '%s' every %ds",
            self._events_dir,
            self._interval,
        )
        while True:
            try:
                self._scan_once()
            except (LLMEndpointUnavailableError, LLMModelUnavailableError):
                logger.critical(
                    "LLM is unavailable; stopping watcher so the app can exit."
                )
                raise
            except Exception:
                logger.exception("Unexpected error during scan; will retry.")
            time.sleep(self._interval)

    def _scan_once(self) -> None:
        """Scan and process all currently pending event directories.

        Returns
        -------
            Performs one polling cycle.
        """
        if not self._events_dir.exists():
            logger.warning("Events directory '%s' missing.", self._events_dir)
            return

        pending, total_dirs, processed_dirs = self._find_pending()
        if not pending:
            if total_dirs == 0:
                logger.info("No event directories found.")
            else:
                logger.info(
                    "No pending events (%d already processed).",
                    processed_dirs,
                )
            return

        logger.info(
            "Found %d pending event(s) (%d already processed).",
            len(pending),
            processed_dirs,
        )
        for event_dir in pending:
            self._process_one(event_dir)

    def _find_pending(self) -> tuple[list[Path], int, int]:
        """Find event directories that still need processing.

        Returns
        -------
            Pending directories, total directories, and processed directory count.
        """
        event_dirs = sorted(
            [event for event in self._events_dir.iterdir() if event.is_dir()]
        )
        pending = [
            event
            for event in event_dirs
            if not self._db.is_event_processed(event.name)
        ]
        total_dirs = len(event_dirs)
        processed_dirs = total_dirs - len(pending)
        return pending, total_dirs, processed_dirs

    def _process_one(self, event_dir: Path) -> None:
        """Process one event directory.

        Parameters
        ----------
        event_dir
            Event directory to process.

        Returns
        -------
            Logs failures and allows the main loop to continue.
        """
        try:
            self._processor.process(event_dir)
        except (LLMEndpointUnavailableError, LLMModelUnavailableError):
            logger.critical(
                "LLM is unavailable while processing event '%s'.",
                event_dir.name,
            )
            raise
        except Exception:
            logger.exception(
                "Failed to process event '%s'; it will be retried on next poll.",
                event_dir.name,
            )
