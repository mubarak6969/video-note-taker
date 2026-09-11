from ui.formatting import format_processed_date, safe_download_filename


def test_format_processed_date_formats_iso_timestamp():
    assert format_processed_date("2026-09-05T12:30:00+00:00") == "Sep 05, 2026"


def test_format_processed_date_empty_for_missing_or_bad_input():
    assert format_processed_date("") == ""
    assert format_processed_date(None) == ""
    assert format_processed_date("not-a-date") == ""


def test_safe_download_filename_normal_title():
    assert safe_download_filename("My Video Title", "notes", ".md") == "My Video Title.md"


def test_safe_download_filename_strips_path_separators():
    # No path separators must survive, regardless of how many ".." segments
    # were in the title - with no "/" or "\" left, stray ".." characters
    # are just inert text, not a traversal path.
    name = safe_download_filename("evil/../../etc/passwd", "notes", ".md")
    assert "/" not in name
    assert "\\" not in name


def test_safe_download_filename_strips_control_and_header_injection_chars():
    name = safe_download_filename('bad"name\r\nContent-Type: evil', "notes", ".md")
    assert "\r" not in name
    assert "\n" not in name
    assert '"' not in name


def test_safe_download_filename_falls_back_to_default_when_title_empty():
    assert safe_download_filename("", "notes", ".md") == "notes.md"
    assert safe_download_filename(None, "notes", ".md") == "notes.md"


def test_safe_download_filename_never_contains_a_path_separator_even_when_title_is_only_separators():
    # Every unsafe char is replaced 1:1, so a title of only separators
    # becomes a string of underscores rather than falling back to the
    # default - either way, the safety property (no "/" or "\") holds.
    name = safe_download_filename("///\\\\", "notes", ".md")
    assert "/" not in name
    assert "\\" not in name
    assert name.endswith(".md")


def test_safe_download_filename_truncates_very_long_titles():
    name = safe_download_filename("x" * 500, "notes", ".md")
    assert len(name) <= 154  # 150 chars + ".md"
