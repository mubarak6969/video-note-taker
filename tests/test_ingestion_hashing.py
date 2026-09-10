from ingestion.hashing import hash_bytes


def test_same_content_produces_same_hash_regardless_of_filename():
    # This is the property duplicate-detection relies on: identical bytes
    # must hash identically even if re-uploaded under a different name.
    data = b"the quick brown fox"
    assert hash_bytes(data) == hash_bytes(data)


def test_different_content_produces_different_hash():
    assert hash_bytes(b"content A") != hash_bytes(b"content B")


def test_hash_is_short_and_stable_length():
    h = hash_bytes(b"anything", length=16)
    assert len(h) == 16
    assert h == hash_bytes(b"anything", length=16)
