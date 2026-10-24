"""
Day 39 (Week 6): prompt-injection and exfiltration defences, in one place.

Threat model in one paragraph: anything a tool returns that originated in
a file (chunk text, filenames) is attacker-controllable -- whoever can
upload a document can put instructions in it. The model cannot reliably
tell instructions from data, so we assume a poisoned document CAN steer
the model, and we bound the damage structurally:

  1. Tools are authorized in code (scope.py, tools.py), not in the prompt.
  2. Untrusted text is delimited and labelled (`wrap_untrusted`) -- this
     raises the bar for the model, it does not guarantee anything.
  3. The output channel is sanitized (`sanitize_output`, `StreamSanitizer`)
     so the classic exfiltration route -- the model emits a markdown
     image/link whose URL carries stolen data and the client's renderer
     fetches it -- is closed even if 1 and 2 fail.
  4. A heuristic detector (`flag_injection`) FLAGS suspicious text for
     logs/trace. It is telemetry, not a defence: regexes are trivially
     bypassed (translation, encoding, paraphrase) and nothing here relies
     on them.
"""
import re

# ---- detection (telemetry only) -------------------------------------------

_INJECTION_PATTERNS = {
    "ignore_previous": r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)*(previous|prior|above|earlier|preceding)",
    "reveal_prompt": r"(reveal|print|show|repeat|output|leak)\s+(your\s+|the\s+)*(system\s+)?(prompt|instructions)",
    "role_override": r"\byou\s+are\s+now\b|\bnew\s+instructions?\s*:|\bact\s+as\b",
    "tool_steering": r"\b(call|invoke|run|use)\s+(the\s+)?\w*\s*(tool|function)\b|\bdocument_id\s*=",
    "fake_role_tags": r"<\s*/?\s*(system|assistant|tool|user)\s*>",
    "image_beacon": r"!\[[^\]]*\]\(\s*https?://",
}
_INJECTION_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _INJECTION_PATTERNS.items()}


def flag_injection(text: str) -> list[str]:
    """Names of the heuristics that matched. Empty list = nothing noticed
    (NOT 'safe')."""
    return [name for name, rx in _INJECTION_RES.items() if rx.search(text)]


# ---- delimiting untrusted text --------------------------------------------

_DELIM_TAG_RE = re.compile(r"<\s*/?\s*(retrieved_document|tool_data)[^>]*>?", re.IGNORECASE)


def _clean_label(label: str) -> str:
    return re.sub(r"[<>\"\n\r]", "", label)[:200]


def wrap_untrusted(label: str, text: str, tag: str = "retrieved_document") -> str:
    """
    Frame tool output so the model can see where untrusted text starts and
    ends. Any delimiter-looking tag INSIDE the text is removed first, so a
    poisoned document can't close the block early and write "outside" it.
    """
    safe = _DELIM_TAG_RE.sub("[tag removed]", text)
    return f'<{tag} source="{_clean_label(label)}" trust="untrusted">\n{safe}\n</{tag}>'


# ---- output sanitization ---------------------------------------------------

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_RE = re.compile(
    r"<\s*(?:img|script|iframe|a|link|style|svg|object|embed)\b[^>]*>", re.IGNORECASE
)
_URL_RE = re.compile(r"https?://\S*", re.IGNORECASE)


def sanitize_output(text: str) -> str:
    """
    Remove the constructs a markdown/HTML renderer would turn into an
    outbound request or a clickable link: images, links (keep the visible
    text), risky HTML tags, bare URLs. Plain `[1]` citations are untouched.

    Deliberately blunt: this agent has no legitimate reason to emit a
    URL, so the safest policy is to emit none. Not a general HTML
    sanitizer -- if the client renders raw HTML, escape it there too.
    """
    text = _IMG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_RE.sub("", text)
    return _URL_RE.sub("[link removed]", text)


# Suffix patterns: "the buffer ENDS inside a construct that is not finished
# yet, so we can't know what sanitize_output will do with it".
_HOLD_RES = [
    re.compile(r"!$"),  # could become "!["
    re.compile(r"!?\[[^\]]*$"),  # "[" or "![" still open
    re.compile(r"!?\[[^\]]*\]$"),  # "[..]" could still become "[..](url)"
    re.compile(r"!?\[[^\]]*\]\([^)]*$"),  # "[..](" url still open
    re.compile(r"h(?:t(?:t(?:p(?:s?(?::/?)?)?)?)?)?$", re.IGNORECASE),  # "http://" prefix
    re.compile(r"https?://\S*$", re.IGNORECASE),  # URL in progress
    re.compile(r"<\s*[a-z]*$", re.IGNORECASE),  # "<" / "<im" could become a tag
    re.compile(
        r"<\s*(?:img|script|iframe|a|link|style|svg|object|embed)\b[^>]*$", re.IGNORECASE
    ),
]
_MAX_HOLD_CHARS = 1000  # never buffer unboundedly; see StreamSanitizer


def _hold_index(buf: str) -> int:
    cut = len(buf)
    for rx in _HOLD_RES:
        m = rx.search(buf)
        if m:
            cut = min(cut, m.start())
    # A cut must never land INSIDE a construct that is already complete
    # (e.g. the "URL in progress" rule fires on the URL inside a finished
    # "[text](https://...)", which would release "[text](" and sanitize the
    # URL separately -- found by the chunking-equivalence test). Pull the
    # cut back to the start of any complete construct it would split.
    moved = True
    while moved:
        moved = False
        for rx in (_IMG_RE, _LINK_RE, _HTML_RE):
            for m in rx.finditer(buf):
                if m.start() < cut < m.end():
                    cut = m.start()
                    moved = True
    return cut


class StreamSanitizer:
    """
    Sanitizes a token stream without ever emitting a half-finished
    dangerous construct. `feed()` returns only the text that is already
    settled; anything that might still become `![..](http://..)` etc. is
    held back until it resolves. Property: for ANY chunking of a string,
    ''.join(feed outputs) + flush() == sanitize_output(whole string).
    That property is tested, because a streaming sanitizer that differs
    from the batch one is exactly where bypasses hide.

    Cost: a token like "[" or "http" is delayed until the next token
    arrives -- a few tokens of latency, only around these constructs.
    The hold is capped (_MAX_HOLD_CHARS); past the cap the held text is
    released, which is safe here because bare URLs are always rewritten,
    so a released fragment can no longer form a working beacon.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        cut = _hold_index(self._buf)
        if len(self._buf) - cut > _MAX_HOLD_CHARS:
            cut = len(self._buf)
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return sanitize_output(ready)

    def flush(self) -> str:
        ready, self._buf = self._buf, ""
        return sanitize_output(ready)
