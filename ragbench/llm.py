"""Thin OpenAI-compatible chat client with a disk cache, retries and usage accounting.

The cache makes runs resumable and reproducible: re-running the benchmark
replays identical calls from disk (including their recorded token counts and
latency), so the numbers in the README can be regenerated for free.
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


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_seconds: float = 0.0
    cached_calls: int = 0
    failed_calls: int = 0  # model kept emitting tool calls; answer treated as empty
    cost_usd: float = 0.0  # from the recorded token counts and the configured prices
    log: list[dict] = field(default_factory=list)


# USD per million tokens (input, output). Override with LLM_PRICE_IN / LLM_PRICE_OUT.
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.reasoning_model = "gpt-oss" in self.model or self.model.startswith(("gpt-5", "o3", "o4"))
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low" if self.reasoning_model else "")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.is_openai = "api.openai.com" in base_url
        if self.is_openai:
            key = os.getenv("OPENAI_API_KEY")
        else:
            key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        # Flex processing is ~half price for slower responses; dropped automatically if unsupported.
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
        self.deadline = _next_clock_time(os.getenv("RAGBENCH_STOP_AT", ""))  # e.g. "06:55" for nightly runs

    def reset_usage(self) -> Usage:
        u, self.usage = self.usage, Usage()
        return u

    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 256, tag: str = "") -> str:
        if self.is_openai and self.model.startswith("gpt-5"):
            # gpt-5.x: fixed temperature, reasoning tokens count against max_completion_tokens
            payload = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens + 2000}
        else:
            payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
            if self.reasoning_effort:
                payload["max_tokens"] = max_tokens + 400  # reasoning tokens count against max_tokens
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
                raise RuntimeError("No API key: put GROQ_API_KEY in .env (see .env.example).")
            if self.deadline and time.time() >= self.deadline:
                raise RuntimeError("Reached RAGBENCH_STOP_AT; stopping (progress is cached).")
            rec = self._call(payload)
            if rec.get("failed"):
                self.usage.failed_calls += 1  # not cached, so a later re-run retries it
            else:
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
        tool_failures = 0
        errors = 0  # rate-limit waits don't count; other transient errors do
        while errors < 12:
            try:
                t = time.perf_counter()
                r = self.client.chat.completions.create(**payload)
                secs = time.perf_counter() - t
                return {
                    "text": (r.choices[0].message.content or "").strip(),
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                    "completion_tokens": r.usage.completion_tokens if r.usage else 0,
                    "seconds": secs,
                    "service_tier": getattr(r, "service_tier", None),
                }
            except RateLimitError as e:
                if "insufficient_quota" in str(e):
                    raise RuntimeError("OpenAI account has no remaining credit/quota (insufficient_quota).") from e
                wait = _retry_after(e) or delay
                if "per day" in str(e).lower():
                    # Groq's daily cap is a rolling window: tokens free up continuously, so by
                    # default wait exactly as long as asked and carry on (set the env var to 0 in CI).
                    max_wait = float(os.getenv("RAGBENCH_MAX_DAILY_WAIT_MIN", "60")) * 60
                    past_deadline = self.deadline and time.time() + wait >= self.deadline
                    if os.getenv("RAGBENCH_WAIT_ON_DAILY_LIMIT", "1") != "1" or wait > max_wait or past_deadline:
                        raise RuntimeError(f"Daily rate limit hit; re-run later to resume from cache. {e}") from e
                    print(f"  daily token cap reached; waiting {wait / 60:.1f} min ({time.strftime('%H:%M')})", flush=True)
                    time.sleep(min(wait + 5, 3600))
                    continue
                time.sleep(min(wait + 0.5, 180))
                delay = min(delay * 2, 60)
            except APIStatusError as e:
                if e.status_code == 400 and ("called a tool" in str(e) or "tool_use_failed" in str(e)):
                    # gpt-oss occasionally emits a tool call even though no tools exist.
                    # Retry: first as-is, then with a little randomness and an explicit note.
                    tool_failures += 1
                    if tool_failures >= 4:
                        return {"text": "", "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "failed": True}
                    if tool_failures >= 2:
                        msgs = [dict(m) for m in payload["messages"]]
                        msgs[0]["content"] += " No tools are available: reply directly in text."
                        payload = {**payload, "messages": msgs, "temperature": 0.3}
                    continue
                if e.status_code == 400 and payload.get("service_tier") and "service_tier" in str(e).lower():
                    print(f"  service tier {payload['service_tier']!r} not supported; using the default tier", flush=True)
                    self.service_tier = ""
                    payload = {k: v for k, v in payload.items() if k != "service_tier"}
                    continue
                if e.status_code == 400 and payload.get("response_format"):
                    # some models reject JSON mode on odd outputs; retry once without it
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    continue
                if e.status_code < 500:
                    raise
                errors += 1
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except APIConnectionError:
                errors += 1
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("LLM call failed after retries")


def _next_clock_time(hhmm: str) -> float | None:
    """Epoch seconds of the next occurrence of local time HH:MM, or None if unset."""
    if not hhmm:
        return None
    import datetime as dt

    h, m = (int(x) for x in hhmm.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def _retry_after(e) -> float | None:
    try:
        h = e.response.headers
        if h.get("retry-after"):
            return float(h["retry-after"])
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(e))
    if m and any(m.groups()):
        h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mins * 60 + secs
    return None
