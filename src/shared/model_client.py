"""Small synchronous client for role models exposing the `/correct` contract."""
from __future__ import annotations

import ipaddress
import json
import re
import urllib.request
from typing import Any, Optional, Protocol
from urllib.parse import urlsplit


class RoleModelError(RuntimeError):
    pass


class RoleModelClient(Protocol):
    def complete(self, source: str, instruction: str) -> str: ...


class CorrectEndpointClient:
    def __init__(
        self,
        url: str,
        *,
        timeout: float = 120.0,
        max_tokens: int = 512,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.calls = 0
        self._opener = None
        if _is_loopback_url(self.url):
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(self, source: str, instruction: str) -> str:
        self.calls += 1
        request = urllib.request.Request(
            self.url,
            data=json.dumps(
                {
                    "benign": source,
                    "question": instruction,
                    "max_tokens": self.max_tokens,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            open_request = (
                self._opener.open
                if self._opener is not None
                else urllib.request.urlopen
            )
            with open_request(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except Exception as exc:
            raise RoleModelError(f"role model request failed: {exc}") from exc
        answer = (payload.get("answer") or "").strip()
        if not answer:
            raise RoleModelError("role model returned an empty answer")
        return answer


class OpenAICompatibleRoleModelClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        model: str = "local-calibration",
        api_key: str = "tmi-local",
        max_tokens: int = 512,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.max_tokens = max_tokens
        self.calls = 0
        client_kwargs = {
            "api_key": api_key or "tmi-local",
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        if _is_loopback_url(self.base_url):
            client_kwargs["trust_env"] = False
        self.client = _build_openai_client(
            **client_kwargs,
        )

    def complete(self, source: str, instruction: str) -> str:
        self.calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": source},
            ],
            stream=False,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.9,
        )
        answer = (
            (response.choices[0].message.content if response.choices else "") or ""
        ).strip()
        if not answer:
            raise RoleModelError("role model returned an empty answer")
        return answer


def _build_openai_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float,
    trust_env: Optional[bool] = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RoleModelError(
            "openai package is required for OpenAI-compatible role model client"
        ) from exc
    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
    }
    if trust_env is False:
        try:
            import httpx
        except ImportError as exc:
            raise RoleModelError(
                "httpx package is required to disable proxies for a loopback role model"
            ) from exc
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
    return OpenAI(**kwargs)


def _is_loopback_url(url: str) -> bool:
    """Detect loopback literals/localhost without DNS or environment proxy I/O."""

    try:
        host = (urlsplit(str(url)).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_json_object(text: str) -> dict:
    """Extract one JSON object from plain text or a fenced model response."""

    stripped = extract_task_output_text(text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise RoleModelError("role model did not return a JSON object")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RoleModelError(f"invalid role-model JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RoleModelError("role model JSON must be an object")
    return value


def split_hidden_thinking(text: str) -> tuple[str, str]:
    """Split visible answer from complete or orphaned ``</think>`` output."""

    stripped = str(text or "").strip()
    closing = stripped.rfind("</think>")
    if closing >= 0:
        thinking = stripped[:closing].strip()
        if thinking.startswith("<think>"):
            thinking = thinking[len("<think>") :].strip()
        answer = stripped[closing + len("</think>") :].strip()
        return thinking, answer
    if stripped.startswith("<think>"):
        return stripped[len("<think>") :].strip(), ""
    return "", stripped


def extract_task_output_text(text: str) -> str:
    _, stripped = split_hidden_thinking(text)
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    xml_match = re.search(r"<task_output>\s*([\s\S]*?)\s*</task_output>", stripped)
    if xml_match:
        stripped = xml_match.group(1).strip()
    return stripped
