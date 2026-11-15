"""Thin OpenAI-compatible chat client with a disk cache, retries, and token/cost accounting.

The cache makes runs resumable and reproducible: re-running the benchmark replays
identical calls from disk (including their recorded token counts and latency), so the
numbers in the README can be regenerated for free.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# USD per million tokens (input, output). Override with LLM_PRICE_IN / LLM_PRICE_OUT.
PRICES = {"gpt-5.6-luna": (0.20, 1.20)}


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_seconds: float = 0.0
    cached_calls: int = 0
    cost_usd: float = 0.0  # from the recorded token counts and the configured prices
    log: list[dict] = field(default_factory=list)


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "gpt-5.6-luna")
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.is_openai = "api.openai.com" in base_url
        key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        # OpenAI's flex tier is about half price for slower responses; dropped automatically if unsupported.
        self.service_tier = os.getenv("LLM_SERVICE_TIER", "flex" if self.is_openai else "")
        timeout = 300 if self.service_tier == "flex" else 60
        self.client = OpenAI(api_key=key or "missing", base_url=base_url, max_retries=0, timeout=timeout)
        self.has_key = bool(key)
        price = PRICES.get(self.model, (0.0, 0.0))
        self.price_in = float(os.getenv("LLM_PRICE_IN", price[0]))
        self.price_out = float(os.getenv("LLM_PRICE_OUT", price[1]))

        cache_dir = ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(cache_dir / "llm.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.usage = Usage()

    def reset_usage(self) -> Usage:
        u, self.usage = self.usage, Usage()
        return u

    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 256, tag: str = "") -> str:
        if self.model.startswith("gpt-5"):
            # gpt-5.x: fixed temperature; reasoning tokens count against max_completion_tokens
            payload = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens + 2000}
        else:
            payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # the cache key ignores the service tier: same model, same answer distribution
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        row = self.db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            rec = json.loads(row[0])
            self.usage.cached_calls += 1
        else:
            if not self.has_key:
                raise RuntimeError("No API key: put OPENAI_API_KEY in .env (see .env.example).")
            rec = self._call(payload)
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, json.dumps(rec)))
            self.db.commit()

        u = self.usage
        u.calls += 1
        u.prompt_tokens += rec["prompt_tokens"]
        u.completion_tokens += rec["completion_tokens"]
        u.api_seconds += rec["seconds"]
        discount = 0.5 if rec.get("service_tier") == "flex" else 1.0
        u.cost_usd += discount * (rec["prompt_tokens"] * self.price_in + rec["completion_tokens"] * self.price_out) / 1e6
        u.log.append({"tag": tag, "out": rec["text"]})
        return rec["text"]

    def chat_json(self, messages: list[dict], **kw) -> dict:
        text = self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            try:
                return json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                return {}

    def _call(self, payload: dict) -> dict:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        delay = 2.0
        for _ in range(12):
            try:
                t = time.perf_counter()
                r = self.client.chat.completions.create(**payload)
                return {
                    "text": (r.choices[0].message.content or "").strip(),
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                    "completion_tokens": r.usage.completion_tokens if r.usage else 0,
                    "seconds": time.perf_counter() - t,
                    "service_tier": getattr(r, "service_tier", None),
                }
            except RateLimitError as e:
                if "insufficient_quota" in str(e):
                    raise RuntimeError("The API account has no remaining credit (insufficient_quota).") from e
                time.sleep(min((_retry_after(e) or delay) + 0.5, 180))
                delay = min(delay * 2, 60)
            except APIStatusError as e:
                if e.status_code == 400 and payload.get("service_tier") and "service_tier" in str(e).lower():
                    print(f"  service tier {payload['service_tier']!r} not supported; using the default tier", flush=True)
                    self.service_tier = ""
                    payload = {k: v for k, v in payload.items() if k != "service_tier"}
                    continue
                if e.status_code < 500:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except APIConnectionError:
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("LLM call failed after retries")


def _retry_after(e) -> float | None:
    try:
        if e.response.headers.get("retry-after"):
            return float(e.response.headers["retry-after"])
    except Exception:
        pass
    return None
