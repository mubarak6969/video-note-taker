import time

import rate_limiter


def test_is_allowed_true_when_under_limit():
    timestamps = [time.time(), time.time()]
    assert rate_limiter.is_allowed(timestamps, limit=3)


def test_is_allowed_false_when_at_limit():
    timestamps = [time.time(), time.time(), time.time()]
    assert not rate_limiter.is_allowed(timestamps, limit=3)


def test_is_allowed_true_when_empty():
    assert rate_limiter.is_allowed([], limit=1)


def test_limit_zero_or_negative_means_unlimited():
    many = [time.time()] * 1000
    assert rate_limiter.is_allowed(many, limit=0)
    assert rate_limiter.is_allowed(many, limit=-1)


def test_prune_old_drops_entries_outside_the_window():
    now = time.time()
    timestamps = [now - 7200, now - 3700, now - 10]  # 2h ago, ~1h ago, 10s ago
    pruned = rate_limiter.prune_old(timestamps, window_seconds=3600)
    assert pruned == [now - 10]


def test_is_allowed_ignores_entries_outside_the_window():
    now = time.time()
    old_timestamps = [now - 7200] * 10  # all 2 hours old, outside a 1h window
    assert rate_limiter.is_allowed(old_timestamps, limit=1, window_seconds=3600)


def test_record_appends_and_prunes():
    now = time.time()
    timestamps = [now - 7200]  # stale, should be dropped
    updated = rate_limiter.record(timestamps, window_seconds=3600)
    assert len(updated) == 1
    assert updated[0] > now - 1  # the freshly recorded entry


def test_record_then_is_allowed_reaches_the_limit():
    timestamps = []
    for _ in range(3):
        assert rate_limiter.is_allowed(timestamps, limit=3)
        timestamps = rate_limiter.record(timestamps)
    assert not rate_limiter.is_allowed(timestamps, limit=3)
