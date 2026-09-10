class IngestionError(Exception):
    """A user-facing ingestion failure: unsupported file type, oversized
    file, unreadable/empty content, and similar - the message is safe to
    show directly in the UI."""
