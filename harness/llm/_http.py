"""Shared HTTP retry logic for the stdlib LLM clients."""
from __future__ import annotations
import time, urllib.error, urllib.request
from .base import LLMError

RETRY_CODES = {408, 409, 429, 500, 502, 503, 504}


def open_with_retry(req: urllib.request.Request, timeout: float, max_retries: int):
    """Return an open response. Retries 429 and 5xx with exponential backoff and honors Retry-After.
    Other HTTP errors raise LLMError at once. The caller must close the response."""
    last = ""
    for attempt in range(max_retries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in RETRY_CODES:
                raise LLMError(last)
            try:
                delay = float(e.headers.get("Retry-After") or 2 ** attempt)
            except ValueError:
                delay = 2 ** attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last, delay = f"network error: {e}", 2 ** attempt
        if attempt < max_retries:
            time.sleep(min(delay, 30))
    raise LLMError(f"gave up after {max_retries + 1} attempts: {last}")
