"""OpenAI-compatible chat model adapter.

Works with any endpoint exposing the OpenAI chat-completions API (OpenAI,
Azure, vLLM, DeepSeek, ...). Endpoint, model and API key come from
configuration and environment variables — nothing is hardcoded, and the API
key is never logged or persisted.
"""

from __future__ import annotations

import json
import logging
import os
import re

from web_harness.core.errors import ModelApiError, ModelOutputParseError
from web_harness.core.models import (
    ActionDecision,
    ModelOutput,
    Observation,
    PromptBundle,
    StepRecord,
    TaskSpec,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# exception names that a retry may plausibly fix; everything else
# (authentication, permission, bad request, invalid model/config) is terminal
_TRANSIENT_API_EXCEPTIONS = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "APIStatusError",  # generic 5xx-ish status wrapper
}


def _is_transient_api_error(exc: Exception) -> bool:
    """Classify an OpenAI SDK exception as transient (retryable) or terminal.

    Matching is by exception class name so the adapter does not need to import
    provider SDK types directly (keeps provider coupling minimal).
    """
    return type(exc).__name__ in _TRANSIENT_API_EXCEPTIONS


def parse_structured_action(raw_text: str) -> ActionDecision:
    """Extract {"action": ..., "short_reason": ...} from model output.

    Accepts a bare JSON object or one embedded in surrounding text. Raises
    ModelOutputParseError (carrying the raw text for tracing) when no valid
    action can be recovered.
    """
    candidates = [raw_text.strip()]
    candidates.extend(m.group(0) for m in _JSON_BLOCK.finditer(raw_text))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("action"), str):
            action = data["action"].strip()
            if action:
                reason = data.get("short_reason")
                return ActionDecision(
                    action=action,
                    short_reason=reason if isinstance(reason, str) else None,
                )
    raise ModelOutputParseError(
        f"could not parse an action JSON object from model output "
        f"({len(raw_text)} chars)",
        raw_text=raw_text,
    )


class OpenAICompatibleModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        base_url_env: str = "MODEL_BASE_URL",
        api_key_env: str = "MODEL_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        resolved_base_url = base_url or os.environ.get(base_url_env, "")
        resolved_api_key = os.environ.get(api_key_env, "")
        if not resolved_base_url:
            raise ModelApiError(
                f"base URL is empty: set config model.base_url or env {base_url_env}"
            )
        if not resolved_api_key:
            raise ModelApiError(
                f"API key is empty: set env {api_key_env} (never hardcode it)"
            )
        # key stays inside the client; it is never stored on self
        from openai import OpenAI

        self._client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout_s,
        )

    def _complete(self, prompt: PromptBundle) -> tuple[str, int | None, int | None]:
        """One chat completion: returns (raw_text, input_tokens, output_tokens).

        Shared by generate_action and generate_structured (Phase 1D)."""
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout_s,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelApiError(
                f"model API call failed: {type(exc).__name__}",
                transient=_is_transient_api_error(exc),
            ) from exc

        try:
            raw_text = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise ModelApiError("model API returned an unexpected response shape") from exc

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return raw_text, (
            int(input_tokens) if input_tokens is not None else None
        ), (int(output_tokens) if output_tokens is not None else None)

    def generate_action(
        self,
        *,
        task: TaskSpec,
        observation: Observation,
        history: list[StepRecord],
        prompt: PromptBundle,
    ) -> ModelOutput:
        raw_text, input_tokens, output_tokens = self._complete(prompt)

        try:
            decision = parse_structured_action(raw_text)
        except ModelOutputParseError as exc:
            # the failed call's tokens are already spent: attach usage so the
            # harness never records a parse failure as a zero-cost call
            raise ModelOutputParseError(
                exc.message,
                raw_text=exc.raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name=self.model,
            ) from exc

        return ModelOutput(
            decision=decision,
            model_name=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_text=raw_text,
        )

    def generate_structured(
        self,
        *,
        prompt: PromptBundle,
    ) -> ModelOutput:
        """Provider-neutral structured generation (Phase 1D): the raw model
        text is returned unparsed in `decision.action`; JSON validation and
        schema interpretation belong to the caller (Replanner)."""
        raw_text, input_tokens, output_tokens = self._complete(prompt)
        return ModelOutput(
            decision=ActionDecision(action=raw_text),
            model_name=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_text=raw_text,
        )
