"""Thin wrapper around the Ollama /api/generate endpoint for vision prompts."""

import base64
import logging
from pathlib import Path

import httpx

from platewatcher.config import Config

logger = logging.getLogger(__name__)


class LLMEndpointUnavailableError(RuntimeError):
    """Raised when the LLM endpoint cannot be reached."""


class LLMModelUnavailableError(RuntimeError):
    """Raised when the configured model is not available on the endpoint."""


class OllamaClient:
    """Client for sending vision prompts to an Ollama endpoint."""

    def __init__(self, config: Config) -> None:
        self._base_url = config.llm_base_url.rstrip("/")
        self._url = f"{self._base_url}/api/generate"
        self._model = config.vision_model
        self._timeout = config.llm_timeout_seconds

    def _raise_endpoint_unavailable(self, message: str) -> None:
        """Log and raise a consistent endpoint-unavailable error."""
        logger.warning(message)
        raise LLMEndpointUnavailableError(message) from None

    def _raise_model_unavailable(self, message: str) -> None:
        """Log and raise a consistent model-unavailable error."""
        logger.warning(message)
        raise LLMModelUnavailableError(message) from None

    def ensure_endpoint_reachable(self) -> None:
        """Validate that the LLM endpoint is reachable.

        Returns
        -------
            Confirms endpoint connectivity.

        Raises
        ------
        LLMEndpointUnavailableError
            Raised when endpoint connectivity validation fails.
        """
        try:
            with httpx.Client(timeout=min(self._timeout, 10)) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
        except httpx.TimeoutException:
            self._raise_endpoint_unavailable(
                "LLM endpoint check timed out after "
                f"{min(self._timeout, 10)}s at {self._base_url}"
            )
        except httpx.ConnectError:
            self._raise_endpoint_unavailable(
                f"Cannot connect to LLM endpoint at {self._base_url} "
                "— is it running?"
            )
        except httpx.HTTPStatusError as exc:
            self._raise_endpoint_unavailable(
                "LLM endpoint health check failed with HTTP "
                f"{exc.response.status_code} at {self._base_url}"
            )

    def query(
        self,
        prompt: str,
        image_path: Path | None = None,
        format: dict | None = None,
    ) -> str | None:
        """Send a prompt request to the configured vision model.

        Parameters
        ----------
        prompt
            Prompt text for the model.
        image_path
            Optional image path to attach to the request.

        Returns
        -------
            Stripped model response text, or None on failure.

        Raises
        ------
        LLMEndpointUnavailableError
            Raised when the LLM endpoint cannot be reached.
        LLMModelUnavailableError
            Raised when the configured model is unavailable.
        """
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if image_path is not None:
            encoded = self._encode_image(image_path)
            if encoded is None:
                return None
            payload["images"] = [encoded]

        return self._post(payload)

    @staticmethod
    def _encode_image(path: Path) -> str | None:
        """Encode one image file as base64.

        Parameters
        ----------
        path
            Image file path to encode.

        Returns
        -------
            Base64-encoded image string, or None on I/O failure.
        """
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except OSError as exc:
            logger.error("Cannot read image %s: %s", path, exc)
            return None

    def _post(self, payload: dict) -> str | None:
        """Send a POST request to the model endpoint.

        Parameters
        ----------
        payload
            Request body to send to the model endpoint.

        Returns
        -------
            Response text when successful, otherwise None.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                text: str = resp.json().get("response", "").strip()
                logger.debug(
                    "LLM response (%d chars): %s", len(text), text[:120]
                )
                return text or None
        except httpx.TimeoutException:
            self._raise_endpoint_unavailable(
                f"LLM request timed out after {self._timeout}s (model={self._model})"
            )
        except httpx.ConnectError:
            self._raise_endpoint_unavailable(
                f"Cannot connect to LLM endpoint at {self._base_url} "
                "— is it running?"
            )
        except httpx.HTTPStatusError as exc:
            details = exc.response.text[:200]
            if exc.response.status_code == 404 and "model" in details.lower():
                self._raise_model_unavailable(
                    f"Configured model '{self._model}' unavailable: {details}"
                )
            self._raise_endpoint_unavailable(
                f"LLM HTTP {exc.response.status_code}: {details}"
            )
        except Exception as exc:  # pragma: no cover
            self._raise_endpoint_unavailable(
                f"Unexpected LLM error contacting {self._base_url}: {exc}"
            )

        return None
