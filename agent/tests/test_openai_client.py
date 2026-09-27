"""
Unit tests for the OpenAILLM construction path.

These lock in the `openai` 2.x (2.54) upgrade without touching the network:
constructing an `AsyncOpenAI` only builds an in-memory HTTP client, so a bare
`OpenAILLM(...)` instantiation is enough to catch a future openai bump that
renames/removes the `AsyncOpenAI` kwargs we rely on, changes required
arguments, or drops the client attributes we read.

Note: openai is imported and the client is constructed directly, so no
provider API key and no container runtime are needed. The tests intentionally
never issue a request.
"""

import re

import openai
import pytest
from openai import AsyncOpenAI

from llm.openai_client import OpenAILLM

pytestmark = pytest.mark.anyio

# Minimum openai version this suite was written and verified against.
MIN_OPENAI_VERSION = (2, 54)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _version_tuple(raw: str) -> tuple[int, ...]:
    """Parse a dotted version string, tolerating pre-release/build suffixes."""
    parts: list[int] = []
    for piece in raw.split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def test_openai_version_is_locked_in():
    """The installed openai must be at least the version we validated."""
    installed = _version_tuple(openai.__version__)
    assert installed >= MIN_OPENAI_VERSION, (
        f"openai {openai.__version__} is older than the validated "
        f"{'.'.join(str(p) for p in MIN_OPENAI_VERSION)}"
    )


async def test_client_constructs_with_explicit_api_key():
    llm = OpenAILLM(api_key="test-key", model_name="gpt-4o-mini")
    try:
        assert isinstance(llm.client, AsyncOpenAI)
        assert llm.client.api_key == "test-key"
        # Falls back to the SDK default endpoint when no base_url is given.
        assert str(llm.client.base_url).rstrip("/") == "https://api.openai.com/v1"
        assert llm.model_name == "gpt-4o-mini"
        assert llm.default_model == "gpt-4o-mini"
        # Attribute chain exercised by completion().
        assert callable(llm.client.chat.completions.create)
    finally:
        await llm.client.close()


async def test_client_constructs_with_custom_base_url_and_account():
    llm = OpenAILLM(
        api_key="test-key",
        base_url="https://proxy.example.invalid/v1",
        organization="org-123",
        project="proj-456",
        provider_name="CustomProvider",
    )
    try:
        assert isinstance(llm.client, AsyncOpenAI)
        assert str(llm.client.base_url).rstrip("/") == (
            "https://proxy.example.invalid/v1"
        )
        assert llm.client.organization == "org-123"
        assert llm.client.project == "proj-456"
        assert llm.provider_name == "CustomProvider"
    finally:
        await llm.client.close()


async def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm = OpenAILLM()
    try:
        assert llm.client.api_key == "env-key"
    finally:
        await llm.client.close()


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLM()
