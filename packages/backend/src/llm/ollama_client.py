import json
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from typing import Callable, TypeAlias, cast

from settings import EMBEDDING_DIMENSION
from urllib import error
from urllib.parse import urlsplit

JsonObject: TypeAlias = dict[str, object]

RequestFn = Callable[[str, JsonObject, float], JsonObject]


class OllamaClientError(RuntimeError):
    pass


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
class OllamaGenerateOptions:
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 80
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 10.0,
        request_fn: RequestFn | None = None,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.timeout_seconds: float = timeout_seconds
        self._request_fn: RequestFn = request_fn or self._default_request

    def generate(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        prompt: str,
        options: OllamaGenerateOptions | None = None,
        system: str | None = None,
        format_json: bool = False,
    ) -> str:
        final_options = options or OllamaGenerateOptions()
        option_payload: JsonObject = {
            "temperature": final_options.temperature,
            "top_p": final_options.top_p,
            "num_predict": final_options.num_predict,
        }
        if final_options.repeat_penalty is not None:
            option_payload["repeat_penalty"] = final_options.repeat_penalty
        if final_options.presence_penalty is not None:
            option_payload["presence_penalty"] = final_options.presence_penalty
        if final_options.frequency_penalty is not None:
            option_payload["frequency_penalty"] = final_options.frequency_penalty

        payload: JsonObject = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": option_payload,
        }

        if system is not None:
            payload["system"] = system

        if format_json:
            payload["format"] = "json"

        url = f"{self.base_url}/api/generate"
        data = self._request_fn(url, payload, self.timeout_seconds)

        response = data.get("response")
        if not isinstance(response, str):
            raise OllamaClientError("Ollama response is missing 'response' text")

        return response

    def embed(
        self,
        *,
        model: str = "bge-m3",
        input: str,
        truncate: bool = True,
        keep_alive: str = "30m",
        expected_dimension: int | None = None,
    ) -> list[float]:
        expected_dim = (
            expected_dimension
            if expected_dimension is not None
            else EMBEDDING_DIMENSION
        )

        payload: JsonObject = {
            "model": model,
            "input": input,
            "truncate": truncate,
            "keep_alive": keep_alive,
            "dimensions": expected_dim,
        }

        url = f"{self.base_url}/api/embed"
        data = self._request_fn(url, payload, self.timeout_seconds)

        embeddings = data.get("embeddings")
        embedding_vector: list[float] | None = None

        if (
            isinstance(embeddings, list)
            and embeddings
            and isinstance(embeddings[0], list)
        ):
            first_embedding = cast(object, embeddings[0])
            embedding_vector = _coerce_float_vector(first_embedding)

        if embedding_vector is None:
            single_embedding = data.get("embedding")
            embedding_vector = _coerce_float_vector(single_embedding)

        if embedding_vector is None:
            raise OllamaClientError(
                "Ollama response is missing 'embeddings' list for /api/embed"
            )

        if len(embedding_vector) != expected_dim:
            raise OllamaClientError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding_vector)}"
            )

        return embedding_vector

    @staticmethod
    def _default_request(
        url: str, payload: JsonObject, timeout_seconds: float
    ) -> JsonObject:
        body = json.dumps(payload).encode("utf-8")
        parsed_url = urlsplit(url)
        path = parsed_url.path or "/"
        if parsed_url.query:
            path = f"{path}?{parsed_url.query}"

        connection_cls = (
            HTTPSConnection if parsed_url.scheme == "https" else HTTPConnection
        )
        connection = connection_cls(parsed_url.netloc, timeout=timeout_seconds)

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
            raise OllamaClientError(f"Failed to connect to Ollama: {exc}") from exc
        finally:
            connection.close()

        raw = raw_bytes.decode("utf-8")

        try:
            parsed = cast(object, json.loads(raw))
        except json.JSONDecodeError as exc:
            raise OllamaClientError("Invalid JSON from Ollama") from exc

        if not isinstance(parsed, dict):
            raise OllamaClientError("Unexpected Ollama response shape")

        return cast(JsonObject, parsed)
