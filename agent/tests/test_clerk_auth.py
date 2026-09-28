"""
Tests for Clerk-backed authentication in api.clerk_auth and the /me endpoint.

No real Clerk instance is contacted: verify_token_async is monkeypatched, so
these exercise the decision logic (precedence, fall-through, enforcement)
deterministically and offline.
"""

import pytest
from fastapi.testclient import TestClient

from api.config import CONFIG
from api.clerk_auth import resolve_auth_state
from clerk_backend_api.security import (
    TokenVerificationError,
    TokenVerificationErrorReason,
)


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


VALID_CLERK_PAYLOAD = {
    "sub": "user_2abc",
    "sid": "sess_2xyz",
    "azp": "https://app.example.com",
    "org_id": "org_2org",
    "org_role": "org:admin",
    "org_slug": "acme",
}


def _patch_verify(monkeypatch, payload_or_error):
    """Patch verify_token_async in the auth module's namespace."""
    from api import clerk_auth

    if isinstance(payload_or_error, Exception):

        async def _raise(token, options):
            raise payload_or_error

        monkeypatch.setattr(clerk_auth, "verify_token_async", _raise)
    else:

        async def _ok(token, options):
            return payload_or_error

        monkeypatch.setattr(clerk_auth, "verify_token_async", _ok)


@pytest.fixture
def clerk_enabled(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_dummy")


@pytest.fixture
def clerk_disabled(monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)


def _bearer(token: str):
    from fastapi.security import HTTPAuthorizationCredentials

    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# --- CONFIG properties -----------------------------------------------------


def test_config_defaults(clerk_disabled, monkeypatch):
    monkeypatch.delenv("BUILDER_TOKEN", raising=False)
    assert CONFIG.clerk_secret_key is None
    assert CONFIG.clerk_authorized_parties == []
    assert CONFIG.clerk_enforce is False


def test_config_authorized_parties_parsing(monkeypatch):
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", "https://a.com, https://b.com")
    assert CONFIG.clerk_authorized_parties == ["https://a.com", "https://b.com"]
    monkeypatch.setenv("CLERK_AUTHORIZED_PARTIES", "")
    assert CONFIG.clerk_authorized_parties == []


def test_config_enforce_parsing(monkeypatch):
    for raw, expected in [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("", False),
        ("0", False),
    ]:
        monkeypatch.setenv("CLERK_ENFORCE", raw)
        assert CONFIG.clerk_enforce is expected


# --- authenticate_with_clerk -------------------------------------------------


async def test_clerk_disabled_returns_none(clerk_disabled):
    from api.clerk_auth import authenticate_with_clerk

    assert await authenticate_with_clerk(_bearer("anything")) is None


async def test_clerk_no_credentials_returns_none(clerk_enabled):
    from api.clerk_auth import authenticate_with_clerk

    assert await authenticate_with_clerk(None) is None


async def test_clerk_valid_token_maps_payload(clerk_enabled, monkeypatch):
    from api.clerk_auth import authenticate_with_clerk

    _patch_verify(monkeypatch, VALID_CLERK_PAYLOAD)
    identity = await authenticate_with_clerk(_bearer("sess-token"))
    assert identity == {
        "provider": "clerk",
        "user_id": "user_2abc",
        "session_id": "sess_2xyz",
        "azp": "https://app.example.com",
        "org_id": "org_2org",
        "org_role": "org:admin",
        "org_slug": "acme",
    }


async def test_clerk_invalid_token_returns_none(clerk_enabled, monkeypatch):
    from api.clerk_auth import authenticate_with_clerk

    error = TokenVerificationError(TokenVerificationErrorReason.TOKEN_INVALID)
    _patch_verify(monkeypatch, error)
    assert await authenticate_with_clerk(_bearer("bad-token")) is None


# --- resolve_auth_state precedence ---------------------------------------------


async def test_precedence_clerk_over_builder_token(clerk_enabled, monkeypatch):
    _patch_verify(monkeypatch, VALID_CLERK_PAYLOAD)
    monkeypatch.setenv("BUILDER_TOKEN", "builder-secret")
    state = await resolve_auth_state(_bearer("sess-token"))
    assert state.authenticated is True
    assert state.mode == "clerk"


async def test_builder_token_still_works_with_clerk_enabled(clerk_enabled, monkeypatch):
    error = TokenVerificationError(TokenVerificationErrorReason.TOKEN_INVALID)
    _patch_verify(monkeypatch, error)
    monkeypatch.setenv("BUILDER_TOKEN", "builder-secret")
    state = await resolve_auth_state(_bearer("builder-secret"))
    assert state.authenticated is True
    assert state.mode == "builder_token"


async def test_open_access_preserved_without_any_keys(clerk_disabled, monkeypatch):
    monkeypatch.delenv("BUILDER_TOKEN", raising=False)
    state = await resolve_auth_state(None)
    assert state.mode == "open"
    assert state.authenticated is False


async def test_open_access_blocked_by_enforce(clerk_enabled, monkeypatch):
    monkeypatch.setenv("CLERK_ENFORCE", "true")
    monkeypatch.delenv("BUILDER_TOKEN", raising=False)
    state = await resolve_auth_state(None)
    assert state.authenticated is False
    assert state.failure_reason == "clerk_required"


async def test_builder_token_missing_fails_when_configured(clerk_disabled, monkeypatch):
    monkeypatch.setenv("BUILDER_TOKEN", "builder-secret")
    state = await resolve_auth_state(None)
    assert state.authenticated is False
    assert state.failure_reason == "missing"


async def test_builder_token_invalid_fails(clerk_disabled, monkeypatch):
    monkeypatch.setenv("BUILDER_TOKEN", "builder-secret")
    state = await resolve_auth_state(_bearer("wrong"))
    assert state.authenticated is False
    assert state.failure_reason == "invalid"


# --- /me endpoint --------------------------------------------------------------


@pytest.fixture
def client():
    from api.agent_server.async_server import app

    with TestClient(app) as c:
        yield c


def test_me_open_mode(client, clerk_disabled, monkeypatch):
    monkeypatch.delenv("BUILDER_TOKEN", raising=False)
    r = client.get("/me")
    assert r.status_code == 200
    assert r.json() == {"authenticated": False, "mode": "open", "identity": None}


def test_me_clerk_identity(client, clerk_enabled, monkeypatch):
    _patch_verify(monkeypatch, VALID_CLERK_PAYLOAD)
    monkeypatch.setenv("BUILDER_TOKEN", "builder-secret")
    r = client.get("/me", headers={"Authorization": "Bearer sess-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["mode"] == "clerk"
    assert body["identity"]["user_id"] == "user_2abc"


def test_me_rejects_enforced_missing_token(client, clerk_enabled, monkeypatch):
    monkeypatch.setenv("CLERK_ENFORCE", "true")
    monkeypatch.delenv("BUILDER_TOKEN", raising=False)
    r = client.get("/me")
    assert r.status_code == 401


def test_root_still_serves(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
