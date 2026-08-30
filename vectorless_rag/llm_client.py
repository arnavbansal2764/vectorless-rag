"""
llm_client.py
-------------
Thin wrapper around the `ollama` Python library for structured JSON generation.

Uses the locally running Ollama daemon — no HTTP configuration needed.
Retry logic handles cold starts, GPU memory pressure, and transient errors.

Strategy: ollama.chat() + format=<JSON schema> (grammar-based structured output).
This forces the model to output valid JSON matching the schema even for
thinking/reasoning models that otherwise emit preamble text before JSON.
"""

from __future__ import annotations

import json
import logging

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from vectorless_rag.config import LLMConfig
from vectorless_rag.tree_model import LLMTreeResponse

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `ollama` package is required. Install it with: pip install ollama"
    ) from exc


class LLMClient:
    """
    Stateless client for generating structured JSON from a local Ollama model.

    Uses ollama.chat() with a Pydantic schema as the `format` argument so the
    model's output is grammar-constrained — reliable even for thinking models.

    Example
    -------
    >>> client = LLMClient(LLMConfig(model="gpt-oss:20b"))
    >>> response = client.generate(system="...", prompt="...")
    >>> print(response.operations)
    """

    # Lazily computed once per class — the JSON schema for LLMTreeResponse
    _RESPONSE_SCHEMA: dict = LLMTreeResponse.model_json_schema()

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(self, system: str, prompt: str) -> LLMTreeResponse:
        """
        Send *system* + *prompt* to the local Ollama model and return a
        parsed :class:`LLMTreeResponse`.

        Retries up to ``config.max_retries`` times with exponential backoff.
        On repeated failure, returns an empty response with a logged warning.
        """
        try:
            return self._generate_with_retry(system=system, prompt=prompt)
        except Exception:
            logger.exception(
                "LLM generation failed after %d retries — returning empty response.",
                self.config.max_retries,
            )
            return LLMTreeResponse(reasoning="[LLM call failed]", operations=[])

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _generate_with_retry(self, system: str, prompt: str) -> LLMTreeResponse:
        """Decorated version with retry; separated so tenacity can wrap it."""

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> LLMTreeResponse:
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format=self._RESPONSE_SCHEMA,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.num_ctx,
                },
            )

            raw: str = response["message"]["content"]
            logger.debug(
                "LLM tokens — prompt: %s | completion: %s",
                response.get("prompt_eval_count", "?"),
                response.get("eval_count", "?"),
            )
            logger.debug("Raw LLM response: %s", raw[:1000])

            return self._parse_response(raw)

        return _call()

    def _parse_response(self, raw: str) -> LLMTreeResponse:
        """Parse raw JSON string into LLMTreeResponse.

        Attempts direct parse first, then falls back to regex extraction for
        any model that still emits surrounding text despite format constraints.
        """
        raw = raw.strip()

        # 1. Direct parse (expected path with structured output)
        try:
            data = json.loads(raw)
            return LLMTreeResponse.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # 2. Regex fallback: find the outermost JSON object
        import re
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if match:
            blob = match.group(1)
            try:
                data = json.loads(blob)
                return LLMTreeResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        logger.warning("JSON parsing failed. Raw response snippet:\n%s", raw[:800])
        raise ValueError("Could not extract valid JSON from LLM response.")
