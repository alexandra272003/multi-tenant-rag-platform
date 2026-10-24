"""
Chunking: cut cleaned text into pieces small enough to embed and retrieve.

Two ideas drive the design:

1. Chunk on natural boundaries (sentences / lines) where possible, so a chunk
   is a coherent thought rather than text sliced mid-sentence.
2. Overlap: each chunk repeats the tail of the previous one, so an answer that
   straddles a boundary still appears whole in at least one chunk.

Every chunk records char_start/char_end into the cleaned text, so a citation
can later point back to exactly where the passage came from.
"""
import re
from dataclasses import dataclass
from typing import Callable

# Approximate token counter: words and punctuation marks each count as one.
# Real LLM tokenizers (BPE) split differently -- roughly 1.3 tokens per
# English word -- so treat sizes here as "about N tokens". The counter is a
# parameter so a real tokenizer can be swapped in without touching the logic.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


# A "unit" is the smallest piece chunks are built from: a sentence, or a line
# (headings / list items). Split after . ! ? followed by whitespace, or at
# newlines.
_UNIT_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int
    token_count: int


def _split_units(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _UNIT_BREAK.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _split_oversize(text, start, end, chunk_size, count) -> list[tuple[int, int]]:
    """A single sentence bigger than chunk_size: fall back to packing words."""
    pieces: list[tuple[int, int]] = []
    piece_start = piece_end = None
    tokens = 0
    for m in re.finditer(r"\S+", text[start:end]):
        w_start, w_end = start + m.start(), start + m.end()
        w_tokens = count(m.group())
        if piece_start is not None and tokens + w_tokens > chunk_size:
            pieces.append((piece_start, piece_end))
            piece_start, tokens = None, 0
        if piece_start is None:
            piece_start = w_start
        piece_end = w_end
        tokens += w_tokens
    if piece_start is not None:
        pieces.append((piece_start, piece_end))
    return pieces


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    count: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    # 1. Break the text into units, each measured in tokens.
    units: list[tuple[int, int, int]] = []  # (start, end, tokens)
    for s, e in _split_units(text):
        t = count(text[s:e])
        if t <= chunk_size:
            units.append((s, e, t))
        else:
            for ps, pe in _split_oversize(text, s, e, chunk_size, count):
                units.append((ps, pe, count(text[ps:pe])))

    # 2. Greedily pack units into chunks, stepping back for overlap.
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (j == i or total + units[j][2] <= chunk_size):
            total += units[j][2]
            j += 1

        start, end = units[i][0], units[j - 1][1]
        body = text[start:end]
        chunks.append(Chunk(body, start, end, count(body)))

        if j >= len(units):
            break

        # Choose where the next chunk starts: walk back from j while the
        # carried-over units fit the overlap budget AND still leave room for
        # the next new unit (otherwise we'd emit a chunk that is only a
        # repeat of the previous one).
        nxt = units[j][2]
        k, acc = j, 0
        while (
            k - 1 > i
            and acc + units[k - 1][2] <= overlap
            and acc + units[k - 1][2] + nxt <= chunk_size
        ):
            acc += units[k - 1][2]
            k -= 1
        i = k  # k >= i + 1 always, so every iteration makes progress

    return chunks
