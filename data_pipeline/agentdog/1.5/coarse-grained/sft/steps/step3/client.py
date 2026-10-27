"""DeepSeek OpenAI兼容接口；仅Step 3的pilot/full模式会实例化。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from steps.step0.source import PipelineError


class DeepSeekClient:
    def __init__(self, model: str, max_tokens: int | None = None) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise PipelineError("DEEPSEEK_API_KEY is required for Step 3 pilot/full")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.url = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens or int(os.environ.get("STEP3_MAX_TOKENS", "65536"))
        self.timeout = int(os.environ.get("STEP3_TIMEOUT", "900"))
        self.retries = int(os.environ.get("STEP3_RETRIES", "3"))

    def complete_json(self, prompt: str) -> dict[str, Any]:
        """请求合法JSON；空content、限流和临时错误都会有限重试。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error = "unknown error"
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                message = result["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                if not content:
                    raise ValueError("DeepSeek returned empty content")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("DeepSeek JSON output must be an object")
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = str(exc)
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 30))
        raise PipelineError(f"DeepSeek request failed after {self.retries} attempts: {last_error}")
