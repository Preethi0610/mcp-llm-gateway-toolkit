import itertools, random, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from redactor import StreamRedactor, redact, MAX_TOKEN, REDACTED

TEXTS = [
    "Mail john.doe@example.com or ssn 123-45-6789, card 4111 1111 1111 1111. Order 1234 5678 9012 3456 ok.",
    "a@b.co,x_y+z@sub.domain.org;5500-0000-0000-0004 and 078-05-1120!",
    "digits 12345 and 123-45-678 and 1234-56-7890 are not PII; 4111111111111112 fails Luhn.",
    "",
    "no pii here at all, just a plain sentence.",
]


def stream(text, cuts):
    chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
    r = StreamRedactor()
    return "".join(r.feed(c) for c in chunks) + r.flush()


@pytest.mark.parametrize("text", TEXTS)
def test_every_single_and_double_split_equals_oneshot(text):
    want = redact(text)
    n = len(text)
    for c in range(1, n):
        assert stream(text, [c]) == want, f"single cut {c}"
    for a, b in itertools.combinations(range(1, n), 2):
        assert stream(text, [a, b]) == want, f"cuts {a},{b}"


def test_char_by_char_and_random_chunking():
    t = TEXTS[0]
    assert stream(t, list(range(1, len(t)))) == redact(t)
    random.seed(7)
    for _ in range(2000):
        cuts = sorted(random.sample(range(1, len(t)), random.randint(1, 15)))
        assert stream(t, cuts) == redact(t)


def test_expected_redactions():
    out = redact(TEXTS[0])
    assert out == f"Mail {REDACTED} or ssn {REDACTED}, card {REDACTED}. Order 1234 5678 9012 3456 ok."


def test_memory_stays_bounded_on_large_streams():
    r = StreamRedactor()
    prose = ("lorem ipsum dolor sit amet " * 40000)  # ~1 MB
    out = "".join(r.feed(prose[i:i + 7]) for i in range(0, len(prose), 7)) + r.flush()
    assert out == prose and r.max_pending <= MAX_TOKEN
    r2 = StreamRedactor()  # worst case: no whitespace at all
    blob = "x" * 1_000_000
    out2 = "".join(r2.feed(blob[i:i + 50]) for i in range(0, len(blob), 50)) + r2.flush()
    assert out2 == blob and r2.max_pending <= MAX_TOKEN + 50


def test_prose_is_not_delayed_beyond_current_word():
    r = StreamRedactor()
    assert r.feed("Hello there, ") == "Hello there, "  # nothing held after whitespace
    assert r.feed("wor") == ""                           # current word held (could be email local-part)
    assert r.feed("ld ") == "world "
