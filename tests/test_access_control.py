from ui.access_control import check_password


def test_correct_password_matches():
    assert check_password("hunter2", "hunter2")


def test_incorrect_password_rejected():
    assert not check_password("wrong", "hunter2")


def test_empty_entered_password_rejected():
    assert not check_password("", "hunter2")


def test_none_entered_password_does_not_raise():
    assert not check_password(None, "hunter2")


def test_no_configured_password_always_rejects():
    # If nothing is configured, require_password() never calls this (it's
    # a no-op) - but the function itself should still fail closed rather
    # than treating an empty configured value as "anything matches".
    assert not check_password("hunter2", "")
    assert not check_password("", "")
