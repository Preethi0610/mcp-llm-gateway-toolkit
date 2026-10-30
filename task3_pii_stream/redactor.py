"""Incremental PII redaction over a text stream with a *bounded* holdback buffer.

Problem: a secret can be split across deltas ("john@exa" | "mple.com"). Naively redacting each
chunk misses it; buffering the whole response kills TTFT and memory.

Approach: keep a small `pending` tail. For each delta:
  1. scan pending+delta with one combined regex,
  2. choose a split point: everything before it can no longer change, everything after might still
     become (part of) a match once more text arrives. Candidates for "still open":
       - the trailing whitespace-free token (could be an email still being typed), cap 320 chars
       - the trailing run of digits/spaces/dashes (could be an SSN / card), cap 24 chars
     If a completed match straddles the split, the split moves to the match start.
  3. emit redacted text before the split, keep the rest as pending.
So memory is O(cap) regardless of response size, and ordinary prose is held back by at most the
current word. `flush()` at end-of-stream redacts whatever remains.

Correctness property (tested exhaustively): for ANY chunking of a text, the concatenated output
equals redact(text) computed in one shot.
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"
MAX_TOKEN = 320   # RFC 5321 caps an address at 254; slack for odd forms
MAX_NUMERIC_RUN = 24  # 19 digits + up to 4 separators, +1

_EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
_SSN = r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])"
_CARD = r"(?<![0-9])(?:[0-9][ \-]?){12,18}[0-9](?![0-9])"
PATTERN = re.compile(f"(?P<email>{_EMAIL})|(?P<ssn>{_SSN})|(?P<card>{_CARD})")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _is_sensitive(m: re.Match[str]) -> bool:
    if m.lastgroup == "card":
        digits = re.sub(r"[ \-]", "", m.group())
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)  # Luhn cuts false positives (order numbers etc.)
    return True


def redact(text: str) -> str:
    """One-shot reference implementation (also used by the property tests)."""
    return PATTERN.sub(lambda m: REDACTED if _is_sensitive(m) else m.group(), text)


class StreamRedactor:
    def __init__(self) -> None:
        self._pending = ""
        self._ctx = ""  # last char already emitted: lookbehind context for (?<![0-9])
        self.max_pending = 0  # observability: high-water mark, asserted in tests

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        buf = self._ctx + self._pending + delta
        off = len(self._ctx)
        matches = [m for m in PATTERN.finditer(buf) if m.start() >= off and _is_sensitive(m)]
        split = self._safe_split(buf, off)
        for m in matches:
            if m.start() < split < m.end():
                split = m.start()
        out = self._render(buf, off, split, matches)
        self._pending = buf[split:]
        if split > off:
            self._ctx = buf[split - 1]
        self.max_pending = max(self.max_pending, len(self._pending))
        return out

    def flush(self) -> str:
        buf = self._ctx + self._pending
        off = len(self._ctx)
        matches = [m for m in PATTERN.finditer(buf) if m.start() >= off and _is_sensitive(m)]
        out = self._render(buf, off, len(buf), matches)
        self._pending = ""
        return out

    @staticmethod
    def _render(buf: str, off: int, split: int, matches: list[re.Match[str]]) -> str:
        parts, pos = [], off
        for m in matches:
            if m.end() > split:
                break
            parts.append(buf[pos:m.start()])
            parts.append(REDACTED)
            pos = m.end()
        parts.append(buf[pos:split])
        return "".join(parts)

    @staticmethod
    def _safe_split(buf: str, floor: int) -> int:
        n = len(buf)
        # trailing whitespace-free token
        i = n
        while i > floor and n - i < MAX_TOKEN and not buf[i - 1].isspace():
            i -= 1
        token_start = i
        # trailing numeric run, trimmed to its first digit
        j = n
        while j > floor and n - j < MAX_NUMERIC_RUN and buf[j - 1] in "0123456789 -":
            j -= 1
        while j < n and not buf[j].isdigit():
            j += 1
        return min(token_start, j)
