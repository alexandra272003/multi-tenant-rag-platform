"""
Text cleaning. Raw text from PDFs is messy: hard-wrapped lines, words split
across lines with hyphens, ligature glyphs, stray control characters. If that
noise goes into chunks, it goes into embeddings (Day 29) and into the LLM's
prompt (Day 31). Cleaning happens BEFORE chunking so chunk boundaries and
character offsets refer to the text we actually store.
"""
import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# "informa-\ntion" -> "information". Restricted to lowercase-on-both-sides so
# "well-\nknown" style compounds are the main false positive; that's an
# accepted trade-off (see README).
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")

_BLANK_LINES = re.compile(r"\n\s*\n+")

# Lines that carry structure (markdown headings, list items). Blocks
# containing these keep their line breaks instead of being re-flowed.
_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s)")


def clean_text(raw: str) -> str:
    # NFKC folds ligatures ("ﬁ" -> "fi") and non-breaking spaces into plain
    # characters, so the same word always has the same spelling.
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    paragraphs: list[str] = []
    for block in _BLANK_LINES.split(text):
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in block.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        if any(_STRUCTURAL.match(ln) for ln in lines):
            paragraphs.append("\n".join(lines))  # keep list/heading structure
        else:
            paragraphs.append(" ".join(lines))  # undo hard line-wrapping
    return "\n\n".join(paragraphs)
