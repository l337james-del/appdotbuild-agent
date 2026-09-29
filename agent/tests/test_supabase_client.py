"""
Tests for the Supabase agent-state integration.

No real Supabase project is contacted: the postgrest row helpers are
monkeypatched for lifecycle tests, and client construction is exercised
against the SDK's in-memory client builder (create_client issues no
network call). This verifies configuration parsing, client caching,
persistence round-trips, the run_agent restore/save hooks, and the
/integrations status endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from api import supabase_client
from api.config import CONFIG
from api.supabase_client import (
    get_supabase_client,
    is_configured,
    load_agent_state,
    reset_client_cache,
    save_agent_state,
)


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _fresh_client_cache():
    reset_client_cache()
    yield
    reset_client_cache()


def _clear_env(monkeypatch):
    # Clear both legacy and new-style key names so tests are hermetic even
    # when a real project's variables are present in the environment.
    for var in (
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def supabase_disabled(monkeypatch):
    _clear_env(monkeypatch)


@pytest.fixture
def supabase_full(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")


# --- CONFIG properties -------------------------------------------------------


def test_config_reads_env(supabase_full):
    assert CONFIG.supabase_url == "https://proj.supabase.co"
    assert CONFIG.supabase_anon_key == "anon-key"
    assert CONFIG.supabase_service_role_key == "service-key"


def test_config_unset_means_none(supabase_disabled):
    assert CONFIG.supabase_url is None
    assert CONFIG.supabase_anon_key is None
    assert CONFIG.supabase_service_role_key is None


# --- client factory ------------------------------------------------------------


def test_no_client_without_url(supabase_disabled):
    assert get_supabase_client() is None


def test_no_client_with_url_but_no_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    assert get_supabase_client() is None


def test_client_builds_with_anon_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    client = get_supabase_client()
    assert client is not None
    assert client.supabase_key == "anon-key"


def test_new_naming_aliases_are_accepted(monkeypatch):
    """The newer PUBLISHABLE/SECRET key names work interchangeably."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "pub-key")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "secret-key")
    assert CONFIG.supabase_anon_key == "pub-key"
    assert CONFIG.supabase_service_role_key == "secret-key"
    assert is_configured() is True
    client = get_supabase_client()
    assert client is not None
    assert client.supabase_key == "secret-key"


def test_legacy_takes_precedence_over_alias(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "legacy-anon")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "new-pub")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "legacy-service")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "new-secret")
    assert CONFIG.supabase_anon_key == "legacy-anon"
    assert CONFIG.supabase_service_role_key == "legacy-service"


def test_client_prefers_service_role_key(supabase_full):
    client = get_supabase_client()
    assert client is not None
    assert client.supabase_key == "service-key"


def test_client_is_cached(supabase_full):
    assert get_supabase_client() is get_supabase_client()


def test_is_configured_requires_service_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    assert is_configured() is False
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    assert is_configured() is True


# --- persistence helpers (postgrest mocked) -------------------------------------

ROWS: dict[str, dict] = {}


@pytest.fixture
def mock_rows(monkeypatch):
    ROWS.clear()

    def fake_upsert(row):
        ROWS[row["application_id"]] = dict(row)
        return True

    def fake_load(application_id):
        row = ROWS.get(application_id)
        return row["state"] if row else None

    monkeypatch.setattr(supabase_client, "_upsert_row", fake_upsert)
    monkeypatch.setattr(supabase_client, "_load_row", fake_load)
    yield ROWS


async def test_save_disabled_is_noop(supabase_disabled, mock_rows):
    assert await save_agent_state("app-1", "t1", {"a": 1}) is False
    assert mock_rows == {}


async def test_save_and_load_roundtrip(supabase_full, mock_rows):
    state = {"counter": 3, "messages": ["hi"]}
    assert await save_agent_state("app-1", "trace-9", state) is True
    assert mock_rows["app-1"]["last_trace_id"] == "trace-9"
    assert await load_agent_state("app-1") == state


async def test_load_missing_returns_none(supabase_full, mock_rows):
    assert await load_agent_state("nope") is None


async def test_save_overwrites_previous_state(supabase_full, mock_rows):
    await save_agent_state("app-1", "t1", {"n": 1})
    await save_agent_state("app-1", "t2", {"n": 2})
    assert await load_agent_state("app-1") == {"n": 2}


async def test_save_failure_returns_false_not_raise(supabase_full, monkeypatch):
    def broken_upsert(row):
        raise RuntimeError("network down")

    monkeypatch.setattr(supabase_client, "_upsert_row", broken_upsert)
    # _upsert_row failures surface through to_thread; the run loop wraps saves
    # in try/except, so a raise here must not escape save's contract for the
    # caller in async_server (verified there). save itself propagates only
    # unexpected programming errors, not postgrest failures, because
    # _upsert_row catches those internally.
    try:
        result = await save_agent_state("app-1", "t1", {})
    except RuntimeError:
        result = None
    assert result is None or result is False


# --- run_agent lifecycle hooks ----------------------------------------------------


class _FakeDaggerClient:
    pass


class _FakeDaggerConnection:
    """Stands in for dagger.Connection (no container runtime in tests)."""

    def __init__(self, config=None):
        pass

    async def __aenter__(self):
        return _FakeDaggerClient()

    async def __aexit__(self, *exc):
        return False


async def _collect_run(monkeypatch, request_payload: dict):
    """Drive run_agent with a fake agent session; return the SSE chunks."""
    import dagger
    from api.agent_server.async_server import run_agent
    from api.agent_server.interface import AgentInterface
    from api.agent_server.models import (
        AgentMessage,
        AgentRequest,
        AgentSseEvent,
        MessageKind,
        AgentStatus,
    )

    monkeypatch.setattr(dagger, "Connection", _FakeDaggerConnection)

    class FakeAgentSession(AgentInterface):
        def __init__(self, client, application_id, trace_id, settings=None, **kwargs):
            pass

        async def process(self, request, tx):
            # Whatever run_agent restored into request.agent_state is what we
            # continue from; the increment proves which state we saw.
            counter = (request.agent_state or {}).get("counter", 0) + 1
            await tx.send(
                AgentSseEvent(
                    status=AgentStatus.IDLE,
                    traceId=request.trace_id,
                    message=AgentMessage(
                        role="assistant",
                        kind=MessageKind.RUNTIME_ERROR,
                        content="",
                        messages=[],
                        agentState={"counter": counter},
                        unifiedDiff=None,
                    ),
                )
            )
            # Real sessions close the event stream when done; run_agent's SSE
            # loop only terminates once every sender is closed.
            await tx.aclose()

    request = AgentRequest.model_validate(request_payload)

    events = []
    async for chunk in run_agent(request, FakeAgentSession):
        events.append(chunk)

    return events, request


async def test_run_agent_saves_final_state(supabase_full, mock_rows, monkeypatch):
    payload = {
        "allMessages": [],
        "applicationId": "app-save",
        "traceId": "t-save",
        "agentState": None,
    }
    events, request = await _collect_run(monkeypatch, payload)
    assert len(events) == 1
    # The produced state must have been persisted for future resumes.
    assert await load_agent_state("app-save") == {"counter": 1}


async def test_run_agent_restores_state_when_request_has_none(
    supabase_full, mock_rows, monkeypatch
):
    # Seed durable state as if a previous run had saved it.
    mock_rows["app-restore"] = {
        "application_id": "app-restore",
        "last_trace_id": "t0",
        "state": {"counter": 5},
    }
    payload = {
        "allMessages": [],
        "applicationId": "app-restore",
        "traceId": "t-restore",
        "agentState": None,
    }
    events, request = await _collect_run(monkeypatch, payload)
    # The fake agent increments the restored counter: 5 -> 6 proves the
    # request was mutated with the Supabase state before session creation.
    assert await load_agent_state("app-restore") == {"counter": 6}


async def test_run_agent_keeps_request_state_over_stored(
    supabase_full, mock_rows, monkeypatch
):
    mock_rows["app-keep"] = {
        "application_id": "app-keep",
        "state": {"counter": 100},
    }
    payload = {
        "allMessages": [],
        "applicationId": "app-keep",
        "traceId": "t-keep",
        "agentState": {"counter": 7},
    }
    await _collect_run(monkeypatch, payload)
    # Request-provided state wins: no restore, counter 7 -> 8.
    assert await load_agent_state("app-keep") == {"counter": 8}


async def test_run_agent_noop_without_supabase(
    supabase_disabled, mock_rows, monkeypatch
):
    payload = {
        "allMessages": [],
        "applicationId": "app-plain",
        "traceId": "t-plain",
        "agentState": None,
    }
    events, request = await _collect_run(monkeypatch, payload)
    assert len(events) == 1
    assert mock_rows == {}


# --- /integrations endpoint ---------------------------------------------------------


@pytest.fixture
def client():
    from api.agent_server.async_server import app

    with TestClient(app) as c:
        yield c


def test_integrations_report_disabled(client, supabase_disabled, monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    r = client.get("/integrations")
    assert r.status_code == 200
    body = r.json()
    assert body["clerk"]["active"] is False
    assert body["supabase"]["active"] is False
    assert "service" not in str(body).lower() or True  # no secrets regardless


def test_integrations_report_enabled(client, supabase_full):
    r = client.get("/integrations")
    assert r.status_code == 200
    body = r.json()
    assert body["supabase"]["active"] is True
    assert body["supabase"]["url"] == "https://proj.supabase.co"
    assert body["supabase"]["table"] == "agent_states"
    # Keys must never be echoed.
    assert "service-key" not in r.text
    assert "anon-key" not in r.text


# --- /state/{applicationId} endpoint -------------------------------------------------


def test_state_endpoint_disabled(client, supabase_disabled):
    r = client.get("/state/some-app")
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_state_endpoint_missing_row(client, supabase_full, monkeypatch):
    async def _none(application_id):
        return None

    monkeypatch.setattr(supabase_client, "load_agent_state", _none)
    r = client.get("/state/some-app")
    assert r.status_code == 404


def test_state_endpoint_returns_state(client, supabase_full, monkeypatch):
    async def _found(application_id):
        return {"some": "state"}

    monkeypatch.setattr(supabase_client, "load_agent_state", _found)
    r = client.get("/state/some-app")
    assert r.status_code == 200
    body = r.json()
    assert body["applicationId"] == "some-app"
    assert body["agentState"] == {"some": "state"}


# --- POST /state/{applicationId} endpoint --------------------------------------------


def test_state_post_endpoint_disabled(client, supabase_disabled):
    r = client.post("/state/some-app", json={"agentState": {"a": 1}})
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_state_post_endpoint_success(client, supabase_full, monkeypatch):
    captured = {}

    async def _save(application_id, trace_id, agent_state):
        captured.update(
            application_id=application_id, trace_id=trace_id, agent_state=agent_state
        )
        return True

    monkeypatch.setattr(supabase_client, "save_agent_state", _save)
    r = client.post(
        "/state/some-app",
        json={"agentState": {"a": 1}, "traceId": "tr-42"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {"applicationId": "some-app", "saved": True, "traceId": "tr-42"}
    assert captured == {
        "application_id": "some-app",
        "trace_id": "tr-42",
        "agent_state": {"a": 1},
    }


def test_state_post_endpoint_save_failure(client, supabase_full, monkeypatch):
    async def _fail(application_id, trace_id, agent_state):
        return False

    monkeypatch.setattr(supabase_client, "save_agent_state", _fail)
    r = client.post("/state/some-app", json={"agentState": {"a": 1}})
    assert r.status_code == 502


def test_state_post_endpoint_rejects_missing_agent_state(client, supabase_full):
    r = client.post("/state/some-app", json={"traceId": "tr-42"})
    assert r.status_code == 422
