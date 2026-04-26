"""Core functionality for a single captured event.

1. Load images from the event directory.
2. Stage 1 — Vehicle detection:
    Ask LLM whether each frame contains a vehicle.
    In the same call-group also ask for vehicle color + type.
    Stop at the first "yes". If None → log as unresolved, delete image, done.
3. Stage 2 — License Plate extraction:
    For each frame, ask LLM for the plate text.
    Stop early when a HIGH-confidence plate is found.
    Otherwise keep the best result seen so far.
4. Write a DB record (status = 'found' or 'unresolved').
5. Delete the event directory to free storage.

LLM works best with short, direct questions and a single task per
prompt. We therefore use three separate prompts rather than one compound
prompt, even though that costs two extra round trips per frame.
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from platewatcher.config import Config
from platewatcher.database import Database, DetectionRecord
from platewatcher.image_processor import ImageProcessor
from platewatcher.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


_PROMPT_IS_VEHICLE = "Is there a vehicle in this image"

_PROMPT_PLATE = (
    "Extract the license plate text if available and rate your confidence.\n"
    #     "RULES TO FOLLOW:\n"
    #     "- confidence: high if plate is clear and legible, medium if partially unclear, low if heavily obscured"
    #     "- license_plate_number: alphanumeric (A-Z, 0-9) and hyphens only, 'None' if unreadable\n"
)


class LLMImageVehicleDescription(BaseModel):
    """Result from a 'is vehicle' query."""

    description: str
    is_vehicle: bool
    vehicle_color: str | None = None
    type_of_vehicle: (
        Literal["car", "truck", "van", "motorcycle", "bus"] | None
    ) = None


class LLMImagePlateDetails(BaseModel):
    """Result from a license plate query."""

    is_vehicle: bool
    is_license_plate_readable: bool
    confidence: Literal["high", "medium", "low"]
    license_plate_number: str | None


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


@dataclass
class _PlateResult:
    """Intermediate plate extraction result.

    Parameters
    ----------
    plate
        Cleaned alphanumeric plate text.
    color
        Vehicle color when detected.
    vehicle_type
        Vehicle type when detected.
    confidence
        Confidence label for the plate reading.
    """

    plate: str  # cleaned alphanumeric plate text
    color: str | None
    vehicle_type: str | None
    confidence: Confidence


def _rank(confidence: Confidence) -> int:
    """Map confidence label to a sortable rank.

    Parameters
    ----------
    confidence
        Confidence label to rank.

    Returns
    -------
        Integer rank where higher is better.
    """
    return _CONFIDENCE_RANK.get(confidence, 0)


def _clean_plate(raw: str) -> str | None:
    """Normalize raw plate text to an uppercase compact token.

    Parameters
    ----------
    raw
        Raw model output to clean.

    Returns
    -------
        Cleaned plate string, or None when empty after normalization.
    """
    cleaned = re.sub(r"[^A-Z0-9\-]", "", raw.upper())
    # Remove isolated hyphens or leading/trailing hyphens
    cleaned = cleaned.strip("-")
    return cleaned or None


class EventProcessor:
    """Run the full detection, extraction, persistence, and cleanup process."""

    def __init__(
        self,
        config: Config,
        db: Database,
        llm_client: OllamaClient,
        image_proc: ImageProcessor,
    ) -> None:
        self._cfg = config
        self._db = db
        self._llm_client = llm_client
        self._img = image_proc
        self._plate_min_length = config.plate_min_length
        self._plate_max_length = config.plate_max_length

    def process(self, event_dir: Path) -> None:
        """Process one event directory from start to finish.

        Parameters
        ----------
        event_dir
            Event directory containing captured frames.

        Returns
        -------
            Persists a result and cleans up event files.
        """
        event_id = event_dir.name
        logger.info("Processing event: %s ", event_id)

        images = self._img.get_event_images(event_dir)

        if not images:
            logger.warning("No valid images in %s; skipping.", event_id)
            self._record_and_cleanup(
                event_dir, status="unresolved", notes="No valid images found"
            )
            return

        # Vehicle detection
        is_vehicle, vehicle_description = self._any_vehicle(images)
        if not is_vehicle:
            logger.info("No vehicle detected in event %s.", event_id)
            self._record_and_cleanup(
                event_dir,
                status="unresolved",
                image_count=len(images),
                notes="No vehicle detected in any frame",
            )
            return

        # plate extraction
        result = self._extract_best_plate(
            images=images, vehicle_description=vehicle_description
        )

        if result is not None:
            logger.info(
                "Plate found in event %s: %s (confidence=%s)",
                event_id,
                result.plate,
                result.confidence,
            )
            self._record_and_cleanup(
                event_dir,
                status="found",
                plate_text=result.plate,
                vehicle_color=result.color,
                vehicle_type=result.vehicle_type,
                confidence=result.confidence,
                image_count=len(images),
            )
        else:
            logger.info(
                "Vehicle found but plate unreadable in event %s.", event_id
            )
            self._record_and_cleanup(
                event_dir,
                status="unresolved",
                image_count=len(images),
                notes="Vehicle detected; plate not readable from any frame",
            )

    def _any_vehicle(
        self, images: list
    ) -> tuple[bool, LLMImageVehicleDescription]:
        """Check whether any frame contains a vehicle.

        Parameters
        ----------
        images
            Ordered list of image paths to inspect.

        Returns
        -------
            Tuple of True when a vehicle is detected in at least one frame.
            And json LLM response in form of LLMImageVehicleDescription
        """
        # Default to None found
        image_vehicle_description = LLMImageVehicleDescription(
            description="",
            is_vehicle=False,
            vehicle_color=None,
            type_of_vehicle=None,
        )
        for path in images:
            llm_input: Path = self._img.prepare_for_llm(path)
            response: str | None = self._llm_client.query(
                prompt=_PROMPT_IS_VEHICLE,
                image_path=llm_input,
                # Force LLM to return a structured JSON format.
                format=LLMImageVehicleDescription.model_json_schema(),
            )
            if response:
                json_response = json.loads(response)
                is_vehicle = json_response.get("is_vehicle", False)
                if is_vehicle:
                    image_vehicle_description = LLMImageVehicleDescription(
                        **json_response
                    )
                    logger.debug("Vehicle confirmed in %s", path.name)
                    break

        return image_vehicle_description.is_vehicle, image_vehicle_description

    def _extract_best_plate(
        self, images: list, vehicle_description: LLMImageVehicleDescription
    ) -> _PlateResult | None:
        """Extract the highest-confidence plate result from image frames.

        Parameters
        ----------
        images
            Ordered list of image paths to inspect.

        vehicle_description
            Description of the vehicle returned by LLM

        Returns
        -------
            Best candidate result, or None when no readable plate is found.
        """
        best: _PlateResult | None = None

        color = vehicle_description.vehicle_color
        v_type = vehicle_description.type_of_vehicle

        for path in images:
            llm_input = self._img.prepare_for_llm(path)
            plate: LLMImagePlateDetails = self._query_plate(llm_input)
            confidence = Confidence[plate.confidence.upper()]

            if plate.license_plate_number is None:
                if best is None:
                    # Store as a placeholder so we have color/type info.
                    # plate=None signals "vehicle seen, plate unreadable".
                    best = _PlateResult(
                        plate="",
                        color=color,
                        vehicle_type=v_type,
                        confidence=confidence,
                    )
                continue

            candidate = _PlateResult(
                plate=plate.license_plate_number,
                color=color,
                vehicle_type=v_type,
                confidence=confidence,
            )

            if confidence == Confidence.HIGH:
                return candidate  # good enough result, exit early

            if best is None or _rank(confidence) > _rank(best.confidence):
                best = candidate

        if best is not None and best.plate:
            return best

        # Return None if all we stored was the placeholder (no readable plate)
        return None

    def _query_plate(self, image_path: Path) -> LLMImagePlateDetails:
        """Query the model for license plate text and confidence from one image.

        Parameters
        ----------
        image_path
            Image path to query.

        Returns
        -------
            Plate text and LLM-assessed confidence level.
        """
        # Start out by assuming no match then only update plate_query if
        # we find a match
        plate_query = LLMImagePlateDetails(
            is_vehicle=False,
            is_license_plate_readable=False,
            license_plate_number=None,
            confidence=Confidence.LOW.value,
        )
        data: str | None = self._llm_client.query(
            prompt=_PROMPT_PLATE,
            image_path=image_path,
            format=LLMImagePlateDetails.model_json_schema(),
        )
        if not data:
            return plate_query
        data_json = json.loads(data)
        cleaned_plate = _clean_plate(data_json.get("license_plate_number", ""))
        is_vehicle = data_json.get("is_vehicle", False)
        is_license_plate_readable = data_json.get(
            "is_license_plate_readable", False
        )
        confidence = data_json.get("confidence", Confidence.LOW)

        if not is_vehicle:
            logger.debug(
                "Plate query results rejected: Image does not contain any vehicle",
            )
            return plate_query

        if cleaned_plate is not None:
            if (
                self._plate_min_length
                <= len(cleaned_plate)
                <= self._plate_max_length
            ):
                # Replace plate_query with matched query
                plate_query = LLMImagePlateDetails(
                    is_vehicle=is_vehicle,
                    is_license_plate_readable=is_license_plate_readable,
                    license_plate_number=cleaned_plate,
                    confidence=confidence,
                )
            else:
                logger.debug(
                    "Plate '%s' rejected: length %d outside [%d, %d]",
                    cleaned_plate,
                    len(cleaned_plate),
                    self._plate_min_length,
                    self._plate_max_length,
                )
        return plate_query

    def _record_and_cleanup(
        self,
        event_dir: Path,
        *,
        status: str,
        plate_text: str | None = None,
        vehicle_color: str | None = None,
        vehicle_type: str | None = None,
        confidence: str | None = None,
        image_count: int = 0,
        notes: str | None = None,
    ) -> None:
        """Persist detection result and clean up processed files.

        Parameters
        ----------
        event_dir
            Processed event directory.
        status
            Final detection status.
        plate_text
            Plate text to persist.
        vehicle_color
            Vehicle color to persist.
        vehicle_type
            Vehicle type to persist.
        confidence
            Confidence label to persist.
        image_count
            Number of frames processed.
        notes
            Optional note to persist for unresolved events.
        """
        event_id = event_dir.name
        detection_record = DetectionRecord(
            event_id=event_id,
            event_timestamp=event_id,  # folder name IS the timestamp string
            status=status,
            plate_text=plate_text,
            vehicle_color=vehicle_color,
            vehicle_type=vehicle_type,
            confidence=confidence,
            image_count=image_count,
            notes=notes,
        )
        self._db.record_detection(detection_record)
        self._img.cleanup_temp_files(event_dir)
        shutil.rmtree(event_dir, ignore_errors=True)
        logger.info("Event %s cleaned up.", event_id)
