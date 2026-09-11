from url_validation import is_youtube_url


def test_accepts_standard_watch_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_accepts_short_youtu_be_url():
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")


def test_accepts_bare_youtube_com_host():
    assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")


def test_accepts_mobile_host():
    assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")


def test_rejects_empty_or_none():
    assert not is_youtube_url("")
    assert not is_youtube_url(None)


def test_rejects_unrelated_url():
    assert not is_youtube_url("https://example.com/watch?v=dQw4w9WgXcQ")


def test_rejects_host_spoofed_via_query_string():
    # A naive substring check ("youtube.com" appears in the string) would
    # wrongly accept this - the host is actually evil.example.
    assert not is_youtube_url("https://evil.example/?next=youtube.com/watch")


def test_rejects_host_spoofed_via_path():
    assert not is_youtube_url("https://evil.example/youtube.com/watch")


def test_rejects_lookalike_subdomain():
    # "youtube.com.evil.example" is a subdomain of evil.example, not of
    # youtube.com - urlparse's hostname must not be fooled by this.
    assert not is_youtube_url("https://youtube.com.evil.example/watch")


def test_rejects_non_http_scheme():
    assert not is_youtube_url("file:///etc/passwd")
    assert not is_youtube_url("javascript:alert(1)//youtube.com")


def test_rejects_malformed_url_without_raising():
    assert not is_youtube_url("::not a url::")
