"""OpenAI兼容接口；仅Step 3的pilot/full模式会实例化。"""

from __future__ import annotations

import json
import os
import time
import http.client
import urllib.error
import urllib.request
from typing import Any

from steps.step0.source import PipelineError


class OpenAICompatibleClient:
    def __init__(self, model: str, max_tokens: int | None = None) -> None:
        api_key = os.environ.get("STEP3_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            raise PipelineError("STEP3_API_KEY is required for Step 3 pilot/full")
        base_url = os.environ.get(
            "STEP3_BASE_URL",
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ).rstrip("/")
        self.url = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens or int(os.environ.get("STEP3_MAX_TOKENS", "65536"))
        self.timeout = int(os.environ.get("STEP3_TIMEOUT", "900"))
        self.retries = int(os.environ.get("STEP3_RETRIES", "3"))
        self.stream = os.environ.get("STEP3_STREAM", "false").strip().lower() == "true"
        self.max_tokens_field = os.environ.get("STEP3_MAX_TOKENS_FIELD", "max_tokens")
        if self.max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise PipelineError("STEP3_MAX_TOKENS_FIELD must be max_tokens or max_completion_tokens")

    @staticmethod
    def _read_stream(response: Any) -> dict[str, Any]:
        """读取OpenAI兼容SSE，在本地合并为非流式响应结构。"""
        content_parts: list[str] = []
        reasoning_returned = False
        response_id = None
        usage: dict[str, Any] = {}
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            response_id = chunk.get("id") or response_id
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or choice.get("message") or {}
                if delta.get("content") is not None:
                    content_parts.append(str(delta["content"]))
                reasoning_returned = reasoning_returned or bool(delta.get("reasoning_content"))
        return {
            "id": response_id,
            "usage": usage,
            "choices": [
                {
                    "message": {
                        "content": "".join(content_parts),
                        "reasoning_content": "streamed" if reasoning_returned else None,
                    }
                }
            ],
        }

    def complete_json(self, prompt: str) -> dict[str, Any]:
        """请求合法JSON；空content、限流和临时错误都会有限重试。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": os.environ.get("STEP3_THINKING", "enabled")},
            "reasoning_effort": os.environ.get("STEP3_REASONING_EFFORT", "high"),
            "stream": self.stream,
        }
        payload[self.max_tokens_field] = self.max_tokens
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # 部分企业网关会拦截Python urllib的默认签名。
                "User-Agent": os.environ.get(
                    "STEP3_USER_AGENT",
                    "Mozilla/5.0 AgentDoG-DataPipeline/1.1",
                ),
            },
            method="POST",
        )

        last_error = "unknown error"
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = (
                        self._read_stream(response)
                        if self.stream
                        else json.loads(response.read().decode("utf-8"))
                    )
                message = result["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                if not content:
                    raise ValueError("API returned empty content")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("API JSON output must be an object")
                return {
                    "json": parsed,
                    "content": content,
                    "response_id": result.get("id"),
                    "usage": result.get("usage", {}),
                    # reasoning_content不进入数据文件，避免存储无用的内部推理文本。
                    "reasoning_returned": bool(message.get("reasoning_content")),
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
                last_error = str(exc)
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 30))
        raise PipelineError(f"API request failed after {self.retries} attempts: {last_error}")
