import hashlib


def hash_bytes(data: bytes, length: int = 16) -> str:
    """A short, stable content hash used as the source_id for uploaded
    files, so the same file - even re-uploaded under a different name -
    is recognized as the same source instead of being re-ingested."""
    return hashlib.sha256(data).hexdigest()[:length]
