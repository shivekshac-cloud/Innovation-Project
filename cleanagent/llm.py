"""Minimal client for any OpenAI-compatible chat endpoint: LM Studio, Ollama, OpenAI, Groq, vLLM...

Configure with environment variables (defaults point at LM Studio's local server):
  CLEANAGENT_LLM_BASE_URL  e.g. http://localhost:1234/v1 (LM Studio) or http://localhost:11434/v1 (Ollama)
  CLEANAGENT_LLM_MODEL     the model name the server expects
  CLEANAGENT_LLM_API_KEY   only for hosted APIs
"""
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
