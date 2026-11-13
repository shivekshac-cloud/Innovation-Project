"""Minimal client for any OpenAI-compatible chat endpoint: LM Studio, Ollama, OpenAI, Groq, vLLM...

Configure with environment variables (defaults point at LM Studio's local server):
  CLEANAGENT_LLM_BASE_URL  e.g. http://localhost:1234/v1 (LM Studio) or http://localhost:11434/v1 (Ollama)
  CLEANAGENT_LLM_MODEL     the model name the server expects
  CLEANAGENT_LLM_API_KEY   only for hosted APIs
"""
from __future__ import annotations
import json
import os
import re
import urllib.request


class OpenAICompatLLM:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=120):
        self.base_url = (base_url or os.getenv("CLEANAGENT_LLM_BASE_URL", "http://localhost:1234/v1")).rstrip("/")
        self.model = model or os.getenv("CLEANAGENT_LLM_MODEL", "local-model")
        self.api_key = api_key if api_key is not None else os.getenv("CLEANAGENT_LLM_API_KEY", "")
        self.timeout = timeout
        self.calls = 0

    def complete(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)
        self.calls += 1
        return data["choices"][0]["message"]["content"]


def extract_json(text):
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


class CachedLLM:
    """Wraps an LLM and stores every reply in a JSON file, keyed by the exact prompt.

    mode="auto":   use the cache when possible, otherwise ask the model and save the answer.
    mode="replay": never contact a model (offline demo); a prompt not in the cache raises an error,
                   and the agent then falls back to rules.
    Run once at home with the model, commit llm_cache.json, and the presentation works with no model running.
    """

    def __init__(self, llm=None, path="llm_cache.json", mode="auto"):
        import hashlib
        from pathlib import Path
        self.llm, self.path, self.mode, self._h = llm, Path(path), mode, hashlib.sha256
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.hits = self.misses = 0
        self.model = getattr(llm, "model", "cache")

    def complete(self, messages):
        key = self._h(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]["reply"]
        if self.mode == "replay" or self.llm is None:
            self.misses += 1
            raise RuntimeError("prompt not in llm_cache.json (replay mode)")
        self.misses += 1
        reply = self.llm.complete(messages)
        self.cache[key] = {"model": self.model, "prompt": messages[-1]["content"][:300], "reply": reply}
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return reply
