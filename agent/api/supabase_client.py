"""
Supabase integration for the agent server.

Provides an agent-state persistence backend backed by a Supabase Postgres
table (via the official `supabase-py` SDK, postgrest transport). This gives
multi-instance deployments a durable store for `request.agent_state`, so a
message can be resumed on any replica.

Additive design, mirroring the Clerk integration:

- No SUPABASE_URL configured -> the integration is disabled; every function
  returns a neutral result and the server behaves exactly as before.
- SUPABASE_URL + SUPABASE_ANON_KEY only -> client is created for future
  anon (RLS-scoped) reads; persistence calls are no-ops unless the
  service role key is also present.
- SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY -> `save_agent_state` /
  `load_agent_state` actively persist and restore state rows in the
  `agent_states` table.

Persistence failures are logged and returned as neutral results; they never
break an in-flight agent run. Client construction itself issues no network
call, so the module imports and builds a client safely in any environment.
"""

from __future__ import annotations

from typing import Any

import anyio
from supabase import Client, create_client

from api.config import CONFIG
from log import get_logger

logger = get_logger(__name__)

AGENT_STATES_TABLE = "agent_states"

_client: Client | None = None
_client_config: tuple[str, str] | None = None


def _client_key() -> str:
    """anon key, service key, or empty when Supabase is not configured."""
    if not CONFIG.supabase_url:
        return ""
    return CONFIG.supabase_service_role_key or CONFIG.supabase_anon_key or ""


def get_supabase_client() -> Client | None:
    """Lazily build and cache the singleton Supabase client.

    Returns None when SUPABASE_URL is not set. The key must also be
    present; a URL without a key is treated as not configured (the SDK
    itself rejects an empty key).
    """
    global _client, _client_config
    if not CONFIG.supabase_url:
        return None
    key = _client_key()
    if not key:
        return None
    config = (CONFIG.supabase_url, key)
    if _client is None or _client_config != config:
        _client = create_client(CONFIG.supabase_url, key)
        _client_config = config
        logger.info(
            f"Supabase client created for {CONFIG.supabase_url} "
            f"({'service_role' if CONFIG.supabase_service_role_key else 'anon'} key)"
        )
    return _client


def reset_client_cache() -> None:
    """Drop the cached client (used by tests to re-read config)."""
    global _client, _client_config
    _client = None
    _client_config = None


def is_configured() -> bool:
    """True when persistence can actually run (URL + service role key)."""
    return bool(CONFIG.supabase_url and CONFIG.supabase_service_role_key)


def _upsert_row(row: dict[str, Any]) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        (
            client.table(AGENT_STATES_TABLE)
            .upsert(row, on_conflict="application_id")
            .execute()
        )
        return True
    except Exception as e:
        logger.warning(f"Supabase agent-state upsert failed (ignored): {e}")
        return False


def _load_row(application_id: str) -> dict[str, Any] | None:
    client = get_supabase_client()
    if client is None:
        return None
    try:
        response = (
            client.table(AGENT_STATES_TABLE)
            .select("state")
            .eq("application_id", application_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return rows[0]["state"] if rows and rows[0].get("state") is not None else None
    except Exception as e:
        logger.warning(f"Supabase agent-state load failed (ignored): {e}")
        return None


async def save_agent_state(
    application_id: str, trace_id: str, agent_state: dict[str, Any]
) -> bool:
    """Persist the current agent state. Fire-and-forget safe: returns False
    on any failure instead of raising, so the agent run is never broken."""
    if not is_configured():
        return False
    row = {
        "application_id": application_id,
        "last_trace_id": trace_id,
        "state": agent_state,
    }
    return await anyio.to_thread.run_sync(lambda: _upsert_row(row))


async def load_agent_state(application_id: str) -> dict[str, Any] | None:
    """Load the most recent agent state for an application, if any."""
    if not is_configured():
        return None
    return await anyio.to_thread.run_sync(lambda: _load_row(application_id))


__all__ = [
    "AGENT_STATES_TABLE",
    "get_supabase_client",
    "is_configured",
    "load_agent_state",
    "reset_client_cache",
    "save_agent_state",
]
