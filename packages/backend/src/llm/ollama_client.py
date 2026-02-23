import json
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from dataclasses import dataclass
from typing import Callable, TypeAlias, cast
from urllib import error
from urllib.parse import urlsplit


JsonObject: TypeAlias = dict[str, object]

RequestFn = Callable[[str, JsonObject, float], JsonObject]


class OllamaClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaGenerateOptions:
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 80


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
        model: str,
        prompt: str,
        options: OllamaGenerateOptions,
        system: str | None = None,
        format_json: bool = False,
    ) -> str:
        payload: JsonObject = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "num_predict": options.num_predict,
            },
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
