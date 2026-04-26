"""Image loading, validation, and pre-processing.

The Pi has limited RAM.  Every image is downscaled to fit within the
configured maximum resolution before it is base64-encoded and sent to the
LLM, which keeps memory pressure low and speeds up model inference.

Resized copies are written to the same directory with a leading "_resized_"
so the watcher and cleanup logic can distinguish them from originals.
"""

import logging
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from platewatcher.config import Config

logger = logging.getLogger(__name__)

# Prefix used for temporary resized copies so they are easy to find/delete.
_RESIZE_PREFIX = "_resized_"


class ImageProcessor:
    """Load, validate, resize, and clean up event images."""

    def __init__(self, config: Config) -> None:
        self._max_size = (config.max_image_width, config.max_image_height)
        self._max_images = config.max_images_per_event
        self._image_extensions = set(config.supported_image_extensions)

    def get_event_images(self, event_dir: Path) -> List[Path]:
        """Collect image files from an event directory.

        Parameters
        ----------
        event_dir
            Event directory to scan.

        Returns
        -------
            Sorted valid image paths, capped by configured max images.
        """
        candidates: list[Path] = sorted(
            [
                item
                for item in event_dir.iterdir()
                if item.is_file()
                and item.suffix.lower() in self._image_extensions
                and not item.name.startswith(_RESIZE_PREFIX)
            ]
        )

        if len(candidates) > self._max_images:
            logger.warning(
                "Event %s has %d images; capping at %d",
                event_dir.name,
                len(candidates),
                self._max_images,
            )
            candidates = candidates[: self._max_images]

        valid_images = []
        for path in candidates:
            if self._is_valid_image(path):
                valid_images.append(path)
            else:
                logger.warning("Skipping unreadable image: %s", path.name)

        logger.debug(
            "Event %s: %d valid image(s)", event_dir.name, len(valid_images)
        )
        return valid_images

    def prepare_for_llm(self, image_path: Path) -> Path:
        """Return an image path suitable for model input.

        Parameters
        ----------
        image_path
            Source image path.

        Returns
        -------
            Original path when already within limits, otherwise resized copy.
        """
        try:
            with Image.open(image_path) as img:
                if (
                    img.width <= self._max_size[0]
                    and img.height <= self._max_size[1]
                ):
                    return image_path

                # Copy before the context manager closes because resizing
                # edits the image in place.
                img_copy = img.copy()

            # Resize the image in place.
            # LANCZOS provides best quality but worst performance, if this
            # becomes a bottleneck, we can switch to something more performant
            # like NEAREST.
            img_copy.thumbnail(self._max_size, Image.Resampling.LANCZOS)

            resized_path = (
                image_path.parent / f"{_RESIZE_PREFIX}{image_path.name}"
            )
            img_copy.save(resized_path)
            logger.debug(
                "Resized %s: %dx%d → saved as %s",
                image_path.name,
                img_copy.width,
                img_copy.height,
                resized_path.name,
            )
            return resized_path

        except Exception as exc:
            logger.warning(
                "Could not resize %s (%s); sending original.",
                image_path.name,
                exc,
            )
            return image_path

    def cleanup_temp_files(self, event_dir: Path) -> None:
        """Delete resized temporary files created for model input.

        Parameters
        ----------
        event_dir
            Event directory containing resized temporary files.

        Returns
        -------
            Removes matching resized files in-place.
        """
        for tmp_file in event_dir.glob(f"{_RESIZE_PREFIX}*"):
            tmp_file.unlink(missing_ok=True)
            logger.debug("Removed temp file: %s", tmp_file.name)

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        """Validate that a file is a readable image.

        Parameters
        ----------
        path
            File path to validate.

        Returns
        -------
            True when the file can be opened and verified as an image.
        """
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, Exception):
            return False
