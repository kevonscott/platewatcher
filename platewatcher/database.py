"""DB setup and all persistence operations."""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import CheckConstraint, Index, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from platewatcher.config import Config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""


class Detection(Base):
    """ORM model for persisted detection records."""

    __tablename__ = "detections"
    __table_args__ = (
        CheckConstraint(
            "status IN ('found', 'unresolved')", name="ck_detections_status"
        ),
        CheckConstraint(
            "confidence IN ('high', 'medium', 'low') OR confidence IS NULL",
            name="ck_detections_confidence",
        ),
        # Index some columns to speed up lookup
        Index("idx_plate", "plate_text"),
        Index("idx_event_ts", "event_timestamp"),
        Index("idx_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    event_timestamp: Mapped[str] = mapped_column(String, nullable=False)
    processed_at: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    plate_text: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicle_color: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicle_type: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String, nullable=True)
    image_count: Mapped[int] = mapped_column(nullable=False, default=0)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


@dataclass(slots=True)
class DetectionRecord:
    """Payload container for detection writes.

    Parameters
    ----------
    event_id
        Stable event identifier.
    event_timestamp
        Event timestamp string associated with the event identifier.
    status
        Detection status value.
    plate_text
        Extracted plate text when available.
    vehicle_color
        Extracted vehicle color when available.
    vehicle_type
        Extracted vehicle type when available.
    confidence
        Confidence label for the extracted result.
    image_count
        Number of frames processed for the event.
    notes
        Optional notes for unresolved or partial detections.
    """

    event_id: str
    event_timestamp: str
    status: str
    plate_text: str | None = None
    vehicle_color: str | None = None
    vehicle_type: str | None = None
    confidence: str | None = None
    image_count: int = 0
    notes: str | None = None


@dataclass(slots=True, frozen=True)
class DetectionResult:
    """Container for read/query results from detections.

    Parameters
    ----------
    id
        Database primary key.
    event_id
        Stable event identifier.
    event_timestamp
        Event timestamp string associated with the event identifier.
    processed_at
        Processing completion timestamp.
    status
        Detection status value.
    plate_text
        Extracted plate text when available.
    vehicle_color
        Extracted vehicle color when available.
    vehicle_type
        Extracted vehicle type when available.
    confidence
        Confidence label for the extracted result.
    image_count
        Number of frames processed for the event.
    notes
        Optional notes for unresolved or partial detections.
    """

    id: int
    event_id: str
    event_timestamp: str
    processed_at: str
    status: str
    plate_text: str | None
    vehicle_color: str | None
    vehicle_type: str | None
    confidence: str | None
    image_count: int
    notes: str | None


class Database:
    def __init__(self, config: Config) -> None:
        self._url: str = config.db_url
        self._engine: Engine = create_engine(self._url, future=True)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        self._init()

    def _init(self) -> None:
        """Initialize database schema.

        Returns
        -------
            Creates missing tables and logs initialization.
        """
        Base.metadata.create_all(self._engine)
        logger.info("Database ready at %s", self._url)

    @staticmethod
    def _to_detection_result(detection: Detection) -> DetectionResult:
        """Convert an ORM row to a typed query result.

        Parameters
        ----------
        detection
            ORM detection row to convert.

        Returns
        -------
            Immutable read-model projection for callers.
        """
        return DetectionResult(
            id=detection.id,
            event_id=detection.event_id,
            event_timestamp=detection.event_timestamp,
            processed_at=detection.processed_at,
            status=detection.status,
            plate_text=detection.plate_text,
            vehicle_color=detection.vehicle_color,
            vehicle_type=detection.vehicle_type,
            confidence=detection.confidence,
            image_count=detection.image_count,
            notes=detection.notes,
        )

    def is_event_processed(self, event_id: str) -> bool:
        """Check whether an event has already been persisted.

        Parameters
        ----------
        event_id
            Event identifier to look up.

        Returns
        -------
            True when a row exists for the event identifier, else False.
        """
        stmt = (
            select(Detection.id).where(Detection.event_id == event_id).limit(1)
        )
        with self._session_factory() as session:
            return session.execute(stmt).scalar_one_or_none() is not None

    def get_detection_by_event_id(
        self, event_id: str
    ) -> DetectionResult | None:
        """Fetch one detection by event identifier.

        Parameters
        ----------
        event_id
            Event identifier to fetch.

        Returns
        -------
            Detection details when found, otherwise None.
        """
        stmt = select(Detection).where(Detection.event_id == event_id).limit(1)
        with self._session_factory() as session:
            detection = session.execute(stmt).scalar_one_or_none()
        if detection is None:
            return None
        return self._to_detection_result(detection)

    def list_recent_detections(self, limit: int = 100) -> list[DetectionResult]:
        """Return recent detections in descending processed order.

        Parameters
        ----------
        limit
            Maximum number of rows to return.

        Returns
        -------
            Recent detections ordered newest first.
        """
        stmt = (
            select(Detection)
            .order_by(Detection.processed_at.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            detections = session.execute(stmt).scalars().all()
        return [self._to_detection_result(item) for item in detections]

    def record_detection(self, detection_record: DetectionRecord) -> None:
        """Insert or update a detection record for an event.

        Parameters
        ----------
        detection_record
            Detection payload to persist.

        Returns
        -------
            Writes to the database and emits a log record.
        """
        processed_at = datetime.now().isoformat(timespec="seconds")
        with self._session_factory.begin() as session:
            stmt = select(Detection).where(
                Detection.event_id == detection_record.event_id
            )
            detection = session.execute(stmt).scalar_one_or_none()

            if detection is None:
                detection = Detection(
                    event_id=detection_record.event_id,
                    event_timestamp=detection_record.event_timestamp,
                )
                session.add(detection)

            detection.event_timestamp = detection_record.event_timestamp
            detection.processed_at = processed_at
            detection.status = detection_record.status
            detection.plate_text = detection_record.plate_text
            detection.vehicle_color = detection_record.vehicle_color
            detection.vehicle_type = detection_record.vehicle_type
            detection.confidence = detection_record.confidence
            detection.image_count = detection_record.image_count
            detection.notes = detection_record.notes

        logger.info(
            "DB record — event=%s  status=%s  plate=%s  confidence=%s",
            detection_record.event_id,
            detection_record.status,
            detection_record.plate_text,
            detection_record.confidence,
        )
