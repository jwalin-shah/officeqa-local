"""Shared LLM client + model selection for all pipeline modules.

Provider priority:
  1. NVIDIA   — set NVIDIA_API_KEY (uses NVIDIA_API_BASE or default NVIDIA endpoint)
  2. Dedalus  — set DEDALUS_API_KEY + DEDALUS_API_BASE (default)

Override model via OFFICEQA_MODEL env var regardless of provider.

Usage:
    from llm_client import client, MODEL, strip_thinking, rate_limiter

    # In decompose.py or extract.py before making an LLM call:
    rate_limiter.acquire()  # block until token available
    response = client.chat.completions.create(...)
"""

import os
import re
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_nvidia_key = os.getenv("NVIDIA_API_KEY")

if _nvidia_key:
    _api_key = _nvidia_key
    _base_url = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    _default_model = "minimaxai/minimax-m2.7"
else:
    _api_key = os.getenv("DEDALUS_API_KEY")
    _base_url = os.getenv("DEDALUS_API_BASE")
    _default_model = "deepseek/deepseek-chat"

MODEL: str = os.getenv("OFFICEQA_MODEL", _default_model)

# Per-phase model overrides — set in .env to use a different model for
# extraction without changing the decompose/verify model.
# E.g. OFFICEQA_EXTRACT_MODEL=deepseek/deepseek-chat with Dedalus client.
EXTRACT_MODEL: str = os.getenv("OFFICEQA_EXTRACT_MODEL", MODEL)

# If a separate extraction client is needed (different provider/key for extract),
# configure via OFFICEQA_EXTRACT_API_KEY + OFFICEQA_EXTRACT_API_BASE.
_extract_key = os.getenv("OFFICEQA_EXTRACT_API_KEY")
_extract_base = os.getenv("OFFICEQA_EXTRACT_API_BASE")
if _extract_key and _extract_base:
    extract_client: OpenAI = OpenAI(api_key=_extract_key, base_url=_extract_base)
else:
    extract_client = None  # type: ignore[assignment]  # falls back to main client

client: OpenAI = OpenAI(api_key=_api_key, base_url=_base_url)


class RateLimiter:
    """Token bucket rate limiter for LLM API calls.

    Enforces a ceiling on requests per minute via a background daemon thread
    that releases tokens at a fixed rate. Callers block on acquire() until
    a token is available.
    """

    def __init__(self, rpm: int = 40):
        """Initialize rate limiter at rpm tokens per minute."""
        self.rpm = rpm
        self.tokens = rpm  # Start with a full bucket
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = True

        # Background thread refills tokens every 60/rpm seconds
        self.refill_thread = threading.Thread(
            target=self._refill_loop, daemon=True, name="RateLimiterRefill"
        )
        self.refill_thread.start()

    def _refill_loop(self):
        """Background thread: refill tokens at fixed rate."""
        interval = 60.0 / self.rpm  # seconds per token
        while self.running:
            time.sleep(interval)
            with self.cv:
                if self.tokens < self.rpm:
                    self.tokens += 1
                    self.cv.notify_all()

    def acquire(self):
        """Block until a token is available, then consume it."""
        with self.cv:
            while self.tokens <= 0:
                self.cv.wait()
            self.tokens -= 1

    def shutdown(self):
        """Stop the refill thread (for cleanup on exit)."""
        self.running = False
        self.refill_thread.join(timeout=1)


# Module-level rate limiter, configurable via OFFICEQA_RPM env var
_RPM = int(os.getenv("OFFICEQA_RPM", "40"))
rate_limiter: RateLimiter = RateLimiter(rpm=_RPM)

# Disable chain-of-thought thinking tokens by default — they add 5-10x latency
# with no benefit for structured extraction tasks. Set OFFICEQA_ENABLE_THINKING=1
# to re-enable for hard reasoning questions.
_ENABLE_THINKING = os.getenv("OFFICEQA_ENABLE_THINKING", "").lower() in ("1", "true", "yes")
# MiniMax M2.7 on NVIDIA ignores "enable_thinking": False — try multiple
# known disable-thinking param names so at least one sticks.
THINKING_EXTRA_BODY: dict = (
    {}
    if _ENABLE_THINKING
    else {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }
)

_SHOW_THINKING = os.getenv("OFFICEQA_SHOW_THINKING", "").lower() in ("1", "true", "yes")


def strip_thinking(text: str, print_thinking: bool = False) -> str:
    """Strip <think>...</think> reasoning blocks, returning only the answer.

    MiniMax-M2.7 (and DeepSeek-R1-style models) emit chain-of-thought inside
    <think>...</think> before the actual answer. Downstream parsers need clean
    text/JSON. The thinking is NOT discarded — set OFFICEQA_SHOW_THINKING=1 or
    pass print_thinking=True to log it, or call extract_thinking() separately.

    Handles unclosed <think> blocks (model truncated mid-reasoning): if no
    </think> is present, strips everything from <think> to the first JSON-like
    content ('{' or '['), so a partial answer can still be salvaged.
    """
    if print_thinking or _SHOW_THINKING:
        thinking = extract_thinking(text)
        if thinking:
            print(f"[think] {thinking[:600]}{'...' if len(thinking) > 600 else ''}", flush=True)
    # Normal case: well-formed <think>...</think> block
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    # Unclosed <think> — model ran out of tokens before </think>.
    # Try to find any JSON-like content after the opening tag.
    if "<think>" in text and "</think>" not in text:
        after = text[text.index("<think>") + len("<think>") :]
        # Look for start of JSON object or array after the thinking
        for marker in ("{", "["):
            idx = after.rfind(marker)  # last occurrence more likely to be final JSON
            if idx != -1:
                candidate = after[idx:]
                if candidate.strip():
                    return candidate.strip()
        return ""  # nothing salvageable
    return stripped


def extract_thinking(text: str) -> str:
    """Extract the <think>...</think> block from a reasoning model response."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""
