"""Public-deployment access control: an optional shared-password gate and
a per-session rate limit on costly actions (ingestion, questions). Both
are no-ops unless explicitly configured, so local development stays
frictionless - see config.APP_PASSWORD / RATE_LIMIT_* for how to turn
them on before a public deployment.
"""
import hmac
import logging

import streamlit as st

import config
import rate_limiter

logger = logging.getLogger(__name__)


def _has_secret(name: str) -> bool:
    try:
        return name in st.secrets
    except Exception:
        return False


def _configured_password() -> str:
    if config.APP_PASSWORD:
        return config.APP_PASSWORD
    if _has_secret("APP_PASSWORD"):
        return st.secrets["APP_PASSWORD"]
    return ""


def check_password(entered: str, configured: str) -> bool:
    """Constant-time comparison so response timing can't be used to guess
    the password character by character. Pulled out as a plain function
    so it's unit-testable without a Streamlit session."""
    if not configured:
        return False
    return hmac.compare_digest(entered or "", configured)


def require_password():
    """Gates the rest of the app behind a shared password, if one is
    configured (APP_PASSWORD env var or Streamlit secret). A no-op when
    it isn't set. Call this before rendering anything else."""
    configured = _configured_password()
    if not configured:
        return
    if st.session_state.get("authenticated"):
        return

    st.title("🎯 Deep-Dive Knowledge Assistant")
    st.write("This deployment is password-protected.")
    entered = st.text_input("Password", type="password", key="access_password_input")
    if st.button("Unlock", type="primary", key="access_password_submit"):
        if check_password(entered, configured):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def check_ingestion_allowed() -> bool:
    """Call before starting an ingestion; shows the limit message and
    returns False if this session should be blocked."""
    timestamps = st.session_state.setdefault("ingestion_timestamps", [])
    if config.RATE_LIMIT_INGESTIONS_PER_HOUR > 0 and not rate_limiter.is_allowed(
        timestamps, config.RATE_LIMIT_INGESTIONS_PER_HOUR
    ):
        st.error(
            f"⏳ You've reached the limit of {config.RATE_LIMIT_INGESTIONS_PER_HOUR} "
            "sources processed per hour on this session. Please try again later."
        )
        return False
    return True


def record_ingestion():
    st.session_state.ingestion_timestamps = rate_limiter.record(
        st.session_state.get("ingestion_timestamps", [])
    )


def check_question_allowed() -> bool:
    """Call before answering a question; shows the limit message and
    returns False if this session should be blocked."""
    timestamps = st.session_state.setdefault("question_timestamps", [])
    if config.RATE_LIMIT_QUESTIONS_PER_HOUR > 0 and not rate_limiter.is_allowed(
        timestamps, config.RATE_LIMIT_QUESTIONS_PER_HOUR
    ):
        st.error(
            f"⏳ You've reached the limit of {config.RATE_LIMIT_QUESTIONS_PER_HOUR} "
            "questions per hour on this session. Please try again later."
        )
        return False
    return True


def record_question():
    st.session_state.question_timestamps = rate_limiter.record(
        st.session_state.get("question_timestamps", [])
    )
