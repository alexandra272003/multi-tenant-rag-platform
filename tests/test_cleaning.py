from app.ingestion.cleaning import clean_text


def test_joins_hard_wrapped_lines_into_one_paragraph():
    assert clean_text("The quick brown\nfox jumps over\nthe lazy dog.") == (
        "The quick brown fox jumps over the lazy dog."
    )


def test_keeps_paragraph_breaks():
    assert clean_text("First paragraph.\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."


def test_repairs_hyphenated_line_breaks():
    assert clean_text("The informa-\ntion was useful.") == "The information was useful."


def test_collapses_runs_of_whitespace():
    assert clean_text("too    many \t  spaces") == "too many spaces"


def test_folds_ligatures_and_nbsp():
    assert clean_text("the \ufb01nal\u00a0answer") == "the final answer"


def test_removes_control_characters():
    assert clean_text("bad\x00 byte\x07s") == "bad bytes"


def test_preserves_list_and_heading_structure():
    raw = "# Title\n\n- one\n- two\n- three"
    assert clean_text(raw) == "# Title\n\n- one\n- two\n- three"


def test_windows_line_endings_are_normalised():
    assert clean_text("a\r\nb\r\n\r\nc") == "a b\n\nc"


def test_empty_and_whitespace_only_become_empty():
    assert clean_text("") == ""
    assert clean_text("  \n\n \t ") == ""
