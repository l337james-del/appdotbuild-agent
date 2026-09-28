"""
Clerk session-token authentication for the agent server.

Verifies Clerk-issued session JWTs against the instance's JWKS via the
official `clerk-backend-api` SDK, and runs alongside the existing static
BUILDER_TOKEN check with additive semantics:

- No CLERK_SECRET_KEY configured -> Clerk auth is disabled and behavior is
  exactly what it was before (builder token if set, otherwise open access).
- A request bearing a valid Clerk session token -> authenticated as that
  Clerk user (takes precedence over the builder-token path).
- Otherwise -> fall through to the existing builder-token behavior.
- CLERK_ENFORCE=true (opt-in) -> when Clerk is enabled, requests that carry
  neither a valid Clerk token nor a valid builder token are rejected with
  401 instead of being allowed by open access.

The module never raises on Clerk failure; callers decide how to respond via
AuthState. No outbound call is made unless a Bearer token is presented and
CLERK_SECRET_KEY is set.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.security import HTTPAuthorizationCredentials

from clerk_backend_api.security import (
    TokenVerificationError,
    VerifyTokenOptions,
    verify_token_async,
)

from api.config import CONFIG
from log import get_logger

logger = get_logger(__name__)


@dataclass
class AuthState:
    """Result of resolving a request's credentials against all auth modes."""

    authenticated: bool
    # Which mechanism applies to this request: "clerk", "builder_token", or "open".
    mode: str
    identity: dict | None = None
    # Why authentication failed: "missing" | "invalid" | "clerk_required" | None.
    failure_reason: str | None = None


def _identity_from_payload(payload: dict) -> dict:
    """Map a verified Clerk session JWT payload to our identity shape."""
    identity: dict = {
        "provider": "clerk",
        "user_id": payload.get("sub"),
        "session_id": payload.get("sid"),
    }
    for claim in ("org_id", "org_role", "org_slug"):
        if payload.get(claim):
            identity[claim] = payload[claim]
    if payload.get("azp"):
        identity["azp"] = payload["azp"]
    return identity


async def authenticate_with_clerk(
    credentials: HTTPAuthorizationCredentials | None,
) -> dict | None:
    """Return a Clerk identity dict when a valid Clerk session token is
    presented, or None when Clerk is disabled / no token / verification
    fails (so callers can fall through to the next auth mechanism)."""
    secret_key = CONFIG.clerk_secret_key
    if not secret_key:
        return None
    if (
        not credentials
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
    ):
        return None

    options = VerifyTokenOptions(
        secret_key=secret_key,
        authorized_parties=CONFIG.clerk_authorized_parties or None,
    )
    try:
        payload = await verify_token_async(credentials.credentials, options)
    except TokenVerificationError as e:
        logger.info(f"Clerk token verification failed: {e}")
        return None
    except Exception as e:
        # The SDK can surface wrapped JWT/network errors; treat any of them
        # as "not a Clerk identity" so the builder-token path still applies.
        logger.warning(f"Clerk token verification error (falling through): {e}")
        return None

    identity = _identity_from_payload(payload)
    logger.info(f"Authenticated via Clerk: user={identity.get('user_id')}")
    return identity


async def resolve_auth_state(
    credentials: HTTPAuthorizationCredentials | None,
) -> AuthState:
    """Resolve credentials across Clerk and builder-token modes.

    Order of precedence:
      1. Valid Clerk session token  -> authenticated, mode "clerk".
      2. Valid builder token        -> authenticated, mode "builder_token".
      3. Clerk enforcement enabled  -> not authenticated, mode "clerk".
      4. No builder token configured-> open access, mode "open".
      5. Otherwise                  -> builder-token failure (missing/invalid).
    """
    clerk_identity = await authenticate_with_clerk(credentials)
    if clerk_identity is not None:
        return AuthState(authenticated=True, mode="clerk", identity=clerk_identity)

    valid_token = CONFIG.builder_token
    if not valid_token:
        if CONFIG.clerk_enforce:
            return AuthState(
                authenticated=False, mode="clerk", failure_reason="clerk_required"
            )
        return AuthState(authenticated=False, mode="open")

    if not credentials or credentials.scheme.lower() != "bearer":
        return AuthState(
            authenticated=False, mode="builder_token", failure_reason="missing"
        )
    if credentials.credentials != valid_token:
        return AuthState(
            authenticated=False, mode="builder_token", failure_reason="invalid"
        )
    return AuthState(authenticated=True, mode="builder_token")


__all__ = ["AuthState", "authenticate_with_clerk", "resolve_auth_state"]
