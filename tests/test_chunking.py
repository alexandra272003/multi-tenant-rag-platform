import pytest

from app.ingestion.chunking import chunk_text, count_tokens
from tests.conftest import LONG_TEXT


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("Just one short sentence.", chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."


def test_empty_text_gives_no_chunks():
    assert chunk_text("", 50, 5) == []


def test_offsets_slice_back_to_the_exact_chunk_text():
    for c in chunk_text(LONG_TEXT, chunk_size=40, overlap=10):
        assert LONG_TEXT[c.char_start : c.char_end] == c.text


def test_chunks_respect_the_size_limit():
    for c in chunk_text(LONG_TEXT, chunk_size=40, overlap=10):
        assert c.token_count <= 40


def test_overlap_repeats_text_from_the_previous_chunk():
    chunks = chunk_text(LONG_TEXT, chunk_size=40, overlap=12)
    assert len(chunks) > 2
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.char_start < prev.char_end  # they share characters


def test_zero_overlap_means_chunks_do_not_share_text():
    chunks = chunk_text(LONG_TEXT, chunk_size=40, overlap=0)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.char_start >= prev.char_end


def test_larger_overlap_produces_more_chunks():
    small = chunk_text(LONG_TEXT, chunk_size=60, overlap=0)
    large = chunk_text(LONG_TEXT, chunk_size=60, overlap=30)
    assert len(large) > len(small)


def test_every_sentence_appears_in_at_least_one_chunk():
    chunks = chunk_text(LONG_TEXT, chunk_size=40, overlap=10)
    for i in range(40):
        assert any(f"Sentence number {i} " in c.text for c in chunks)


def test_chunk_starts_strictly_increase_so_the_loop_always_progresses():
    starts = [c.char_start for c in chunk_text(LONG_TEXT, chunk_size=25, overlap=20)]
    assert starts == sorted(set(starts))


def test_a_sentence_longer_than_chunk_size_is_split_by_words():
    long_sentence = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = chunk_text(long_sentence, chunk_size=30, overlap=0)
    assert len(chunks) >= 4
    assert all(c.token_count <= 30 for c in chunks)


def test_no_chunk_is_merely_a_repeat_of_the_previous_one():
    # Two tiny sentences, then a 29-token sentence, with chunk_size=30. If the
    # tiny sentence were carried over as overlap, the big one could no longer
    # fit beside it, and the "next" chunk would be just a copy of text the
    # previous chunk already contained. The overlap must be dropped instead.
    big = " ".join(f"w{i}" for i in range(28)) + "."
    text = f"Tiny one. Tiny two. {big} Tail sentence."
    chunks = chunk_text(text, chunk_size=30, overlap=8)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.char_end > prev.char_end, "chunk adds nothing beyond the previous one"
    assert any("Tail sentence." in c.text for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_count_tokens_counts_words_and_punctuation():
    assert count_tokens("Hello, world!") == 4
