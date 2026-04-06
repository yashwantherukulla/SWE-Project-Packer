from src.eval.verify import compute_ngram_overlap, verify_exact_match


def test_verify_exact_match_strips_whitespace():
    assert verify_exact_match("hello\n", "hello")
    assert not verify_exact_match("hello world", "hello")


def test_ngram_overlap_detects_shared_phrases():
    target = "one two three four five six seven"
    generated = "zero one two three four five six nine"
    overlap = compute_ngram_overlap(generated, target, n=3)
    assert 0.0 < overlap <= 1.0


def test_ngram_overlap_returns_zero_for_short_text():
    assert compute_ngram_overlap("tiny", "also tiny", n=5) == 0.0
