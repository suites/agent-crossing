import json
import os
import ssl
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from typing import Callable, TypeAlias, cast
from urllib import error
from urllib.parse import urlencode, urlsplit

import certifi
from settings import EMBEDDING_DIMENSION

from .ollama_client import OllamaGenerateOptions

JsonObject: TypeAlias = dict[str, object]

RequestFn = Callable[[str, JsonObject, float], JsonObject]


class GoogleAiStudioClientError(RuntimeError):
    pass


def _build_ssl_context() -> ssl.SSLContext:
    ssl_cert_file = os.getenv("SSL_CERT_FILE", "").strip()
    if ssl_cert_file:
        return ssl.create_default_context(cafile=ssl_cert_file)

    return ssl.create_default_context(cafile=certifi.where())


def _coerce_float_vector(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None

    raw_values = cast(list[object], value)
    vector: list[float] = []
    for item in raw_values:
        if not isinstance(item, (int, float)):
            return None
        vector.append(float(item))

    return vector


@dataclass(frozen=True)
class GoogleAiStudioClient:
    api_key: str
    timeout_seconds: float = 10.0
    base_url: str = "https://generativelanguage.googleapis.com"
    default_generate_model: str = "gemini-1.5-flash"
    default_embedding_model: str = "text-embedding-004"
    request_fn: RequestFn | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.api_key.strip():
            raise GoogleAiStudioClientError("GOOGLE_AI_STUDIO_API_KEY is required")
        object.__setattr__(
            self,
            "request_fn",
            self.request_fn or self._default_request,
        )

    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        options: OllamaGenerateOptions | None = None,
        format_json: bool = False,
        model: str | None = None,
    ) -> str:
        final_options = options or OllamaGenerateOptions()
        generation_config: JsonObject = {
            "temperature": final_options.temperature,
            "topP": final_options.top_p,
            "maxOutputTokens": final_options.num_predict,
        }
        if format_json:
            generation_config["responseMimeType"] = "application/json"

        payload: JsonObject = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        if system is not None:
            payload["systemInstruction"] = {
                "parts": [{"text": system}],
            }

        selected_model = model or self.default_generate_model
        url = self._build_url(
            path=f"/v1beta/models/{selected_model}:generateContent",
        )
        data = cast(RequestFn, self.request_fn)(url, payload, self.timeout_seconds)

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing 'candidates'"
            )

        first_candidate = cast(object, candidates[0])
        if not isinstance(first_candidate, dict):
            raise GoogleAiStudioClientError(
                "Google AI Studio response has invalid candidate shape"
            )

        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing 'content'"
            )

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing 'parts'"
            )

        first_part = cast(object, parts[0])
        if not isinstance(first_part, dict):
            raise GoogleAiStudioClientError(
                "Google AI Studio response has invalid part shape"
            )

        text = first_part.get("text")
        if not isinstance(text, str):
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing 'text'"
            )

        return text

    def embed(
        self,
        *,
        input: str,
        model: str | None = None,
        truncate: bool = True,
        keep_alive: str = "30m",
        expected_dimension: int | None = None,
    ) -> list[float]:
        _ = truncate
        _ = keep_alive
        expected_dim = (
            expected_dimension
            if expected_dimension is not None
            else EMBEDDING_DIMENSION
        )
        selected_model = model or self.default_embedding_model
        payload: JsonObject = {
            "content": {
                "parts": [{"text": input}],
            },
            "outputDimensionality": expected_dim,
        }
        url = self._build_url(
            path=f"/v1beta/models/{selected_model}:embedContent",
        )
        data = cast(RequestFn, self.request_fn)(url, payload, self.timeout_seconds)

        embedding = data.get("embedding")
        if not isinstance(embedding, dict):
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing 'embedding'"
            )

        values = embedding.get("values")
        embedding_vector = _coerce_float_vector(values)
        if embedding_vector is None:
            raise GoogleAiStudioClientError(
                "Google AI Studio response is missing embedding values"
            )
        if len(embedding_vector) != expected_dim:
            raise GoogleAiStudioClientError(
                "Embedding dimension mismatch: "
                + f"expected {expected_dim}, got {len(embedding_vector)}"
            )
        return embedding_vector

    def _build_url(self, *, path: str) -> str:
        query = urlencode({"key": self.api_key})
        return f"{self.base_url}{path}?{query}"

    @staticmethod
    def _default_request(
        url: str, payload: JsonObject, timeout_seconds: float
    ) -> JsonObject:
        body = json.dumps(payload).encode("utf-8")
        parsed_url = urlsplit(url)
        path = parsed_url.path or "/"
        if parsed_url.query:
            path = f"{path}?{parsed_url.query}"

        if parsed_url.scheme == "https":
            connection = HTTPSConnection(
                parsed_url.netloc,
                timeout=timeout_seconds,
                context=_build_ssl_context(),
            )
        else:
            connection = HTTPConnection(parsed_url.netloc, timeout=timeout_seconds)
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response: HTTPResponse = connection.getresponse()
            raw_bytes = response.read()
        except (OSError, error.URLError) as exc:
            raise GoogleAiStudioClientError(
                f"Failed to connect to Google AI Studio: {exc}"
            ) from exc
        finally:
            connection.close()

        raw = raw_bytes.decode("utf-8")
        try:
            parsed = cast(object, json.loads(raw))
        except json.JSONDecodeError as exc:
            raise GoogleAiStudioClientError(
                "Invalid JSON from Google AI Studio"
            ) from exc

        if not isinstance(parsed, dict):
            raise GoogleAiStudioClientError(
                "Unexpected Google AI Studio response shape"
            )

        error_payload = parsed.get("error")
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
            if isinstance(message, str) and message.strip():
                raise GoogleAiStudioClientError(message)
            raise GoogleAiStudioClientError("Google AI Studio returned an error")

        return cast(JsonObject, parsed)
