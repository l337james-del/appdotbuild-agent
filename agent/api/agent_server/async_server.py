"""
FastAPI implementation for the agent server.

This server handles API requests initiated by clients (e.g., test clients),
coordinates agent logic using components from `core` and specific agent
implementations like `trpc_agent`. It utilizes `models.py` for
request/response validation and interacts with LLMs via the `llm` wrappers
(indirectly through agents).

Refer to `architecture.puml` for a visual overview.
"""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from laravel_agent.agent_session import LaravelAgentSession
import uvicorn
from fire import Fire
import os

# Disable dagger telemetry
os.environ["DO_NOT_TRACK"] = "1"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

import dagger
import json
from brotli_asgi import BrotliMiddleware
from dotenv import load_dotenv

from api.agent_server.models import (
    AgentRequest,
    AgentSseEvent,
    AgentMessage,
    AgentStatus,
    MessageKind,
    ErrorResponse,
    ExternalContentBlock,
    StateUpdateRequest,
)
from api.agent_server.interface import AgentInterface
from trpc_agent.agent_session import TrpcAgentSession
from nicegui_agent.agent_session import NiceguiAgentSession
from api.agent_server.template_diff_impl import TemplateDiffAgentImplementation
from api.config import CONFIG

from log import get_logger, configure_uvicorn_logging, set_trace_id, clear_trace_id
from api.clerk_auth import resolve_auth_state
from api import supabase_client
from llm.telemetry import save_cumulative_stats

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Initializing Async Agent Server API")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"AWS_SECRET_ACCESS_KEY: {'SET' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'NOT_SET'}"
    )
    logger.info(
        f"ANTHROPIC_API_KEY: {'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT_SET'}"
    )
    logger.info(
        f"GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT_SET'}"
    )

    yield
    logger.info("Shutting down Async Agent Server API")

    # save cumulative telemetry stats on shutdown
    save_cumulative_stats()


app = FastAPI(
    title="Async Agent Server API",
    description="Async API for communication between the Platform (Backend) and the Agent Server",
    version="1.0.0",
    lifespan=lifespan,
)

# add Brotli compression middleware with optimized settings for SSE
app.add_middleware(
    BrotliMiddleware,
    quality=4,  # balanced compression/speed for streaming
    minimum_size=500,  # compress responses >= 500 bytes
    gzip_fallback=True,  # fallback to gzip for older clients
)

bearer_scheme = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Resolve request credentials (Clerk session token or builder token).

    Behavior is unchanged from the original builder-token-only flow when no
    CLERK_SECRET_KEY is configured; see api.clerk_auth for the additive
    Clerk semantics.
    """
    state = await resolve_auth_state(credentials)

    if state.authenticated:
        return True

    if state.failure_reason == "clerk_required":
        logger.info("Clerk enforcement enabled and no valid credentials provided")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized - a valid Clerk session token is required",
        )
    if state.failure_reason == "missing":
        logger.info("Missing authentication token")
        raise HTTPException(
            status_code=401, detail="Unauthorized - missing authentication token"
        )
    if state.failure_reason == "invalid":
        logger.info("Invalid authentication token")
        raise HTTPException(
            status_code=403, detail="Unauthorized - invalid authentication token"
        )

    # No builder token configured and Clerk disabled/enforcement off: open access.
    return True


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_or_create_session[T: AgentInterface](
        self,
        client: dagger.Client,
        request: AgentRequest,
        agent_class: type[T],
        *args,
        **kwargs,
    ) -> T:
        session_id = f"{request.application_id}:{request.trace_id}"

        logger.info(f"Creating new agent session for {session_id}")
        agent = agent_class(
            client=client,
            application_id=request.application_id,
            trace_id=request.trace_id,
            settings=request.settings,
            *args,
            **kwargs,
        )
        return agent

    def cleanup_session(self, application_id: str, trace_id: str) -> None:
        session_id = f"{application_id}:{trace_id}"
        if session_id in self.sessions:
            logger.info(f"Removing session for {session_id}")
            del self.sessions[session_id]


session_manager = SessionManager()


async def run_agent[T: AgentInterface](
    request: AgentRequest,
    agent_class: type[T],
    *args,
    **kwargs,
) -> AsyncGenerator[str, None]:
    logger.info(
        f"Running agent for session {request.application_id}:{request.trace_id}"
    )

    async with dagger.Connection(
        dagger.Config(log_output=open(os.devnull, "w"))
    ) as client:
        # Establish Dagger connection for the agent's execution context
        # Restore durable state from Supabase when the request carries none,
        # so a conversation can be resumed on any replica of a multi-instance
        # deployment. No-op when Supabase is not configured.
        if request.agent_state is None and supabase_client.is_configured():
            restored = await supabase_client.load_agent_state(request.application_id)
            if restored is not None:
                logger.info(
                    f"Restored agent state from Supabase for {request.application_id}"
                )
                request.agent_state = restored

        agent = session_manager.get_or_create_session(
            client, request, agent_class, *args, **kwargs
        )

        event_tx, event_rx = anyio.create_memory_object_stream[AgentSseEvent](
            max_buffer_size=0
        )
        keep_alive_tx = (
            event_tx.clone()
        )  # Clone the sender for use in the keep-alive task
        final_state = None

        # Use this flag to control the keep-alive task
        keep_alive_running = True

        async def send_keep_alive():
            try:
                # Use shorter sleep intervals to be more responsive
                keep_alive_interval = 30
                sleep_interval = 0.5  # Check every 500ms
                elapsed = 0.0

                while keep_alive_running:
                    await anyio.sleep(sleep_interval)
                    elapsed += sleep_interval

                    if keep_alive_running and elapsed >= keep_alive_interval:
                        keep_alive_event = AgentSseEvent(
                            status=AgentStatus.RUNNING,
                            traceId=request.trace_id,
                            message=AgentMessage(
                                role="assistant",
                                kind=MessageKind.KEEP_ALIVE,
                                content="",
                                messages=[],
                                agentState=None,
                                unifiedDiff=None,
                            ),
                        )
                        await keep_alive_tx.send(keep_alive_event)
                        elapsed = 0.0  # Reset elapsed time

            except Exception:
                pass
            finally:
                await keep_alive_tx.aclose()

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(agent.process, request, event_tx)

                # Start the keep-alive task
                tg.start_soon(send_keep_alive)

                async with event_rx:
                    async for event in event_rx:
                        # Keep track of the last state in events with non-null state
                        if event.message and event.message.agent_state:
                            final_state = event.message.agent_state
                            # Persist the newest state so a future request without
                            # agentState can resume from Supabase. Fire-and-forget:
                            # persistence problems never break the SSE stream.
                            if supabase_client.is_configured():
                                try:
                                    await supabase_client.save_agent_state(
                                        request.application_id,
                                        request.trace_id,
                                        final_state,
                                    )
                                except Exception as save_error:
                                    logger.warning(
                                        f"Supabase agent-state save error (ignored): {save_error}"
                                    )

                        # Format SSE event properly with data: prefix and double newline at the end
                        # This ensures compatibility with SSE standard
                        yield f"data: {event.to_json()}\n\n"

                        if event.status == AgentStatus.IDLE:
                            keep_alive_running = False

                            # Only log that we'll clean up later - don't do the actual cleanup here
                            # The actual cleanup happens in the finally block
                            if request.agent_state is None:
                                logger.info(
                                    f"Agent idle, will clean up session for {request.application_id}:{request.trace_id} when all events are processed"
                                )

        except* Exception as excgroup:
            for e in excgroup.exceptions:
                # Log the specific exception from the group with traceback
                logger.exception(
                    f"Error in SSE generator TaskGroup for trace {request.trace_id}:",
                    exc_info=e,
                )
                error_event = AgentSseEvent(
                    status=AgentStatus.IDLE,
                    traceId=request.trace_id,
                    message=AgentMessage(
                        role="assistant",
                        kind=MessageKind.RUNTIME_ERROR,
                        content=json.dumps(
                            [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Error processing request: {str(e)}",
                                        }
                                    ],
                                }
                            ]
                        ),
                        messages=[
                            ExternalContentBlock(
                                content=f"Error processing request: {str(e)}",
                                # timestamp=datetime.datetime.now(datetime.UTC)
                            )
                        ],
                        agentState=None,
                        unifiedDiff="",
                    ),
                )
                # Format error SSE event properly
                yield f"data: {error_event.to_json()}\n\n"

                # On error, remove the session entirely
                session_manager.cleanup_session(
                    request.application_id, request.trace_id
                )
        finally:
            # For requests without agent state or where the session completed, clean up
            # Ensure cleanup happens outside the dagger connection if needed, though session removal should be fine
            if request.agent_state is None and (
                final_state is None or final_state == {}
            ):
                logger.info(
                    f"Cleaning up completed agent session for {request.application_id}:{request.trace_id}"
                )
                session_manager.cleanup_session(
                    request.application_id, request.trace_id
                )
                clear_trace_id()


@app.post("/message", response_model=None)
async def message(
    request: AgentRequest, token: str = Depends(verify_token)
) -> StreamingResponse:
    """
    Send a message to the agent and stream responses via SSE.

    Platform (Backend) -> Agent Server API Spec:
    POST Request:
    - allMessages: [str] - history of all user messages
    - applicationId: str - required for Agent Server for tracing
    - traceId: str - required - a string used in SSE events
    - agentState: {..} or null - the full state of the Agent to restore from
    - settings: {...} - json with settings with number of iterations etc

    SSE Response:
    - status: "running" | "idle" - defines if the Agent stopped or continues running
    - traceId: corresponding traceId of the input
    - message: {kind, content, agentState, unifiedDiff} - response from the Agent Server

    Args:
        request: The agent request containing all necessary fields
        token: Authentication token (automatically verified by verify_token dependency)

    Returns:
        Streaming response with SSE events according to the API spec
    """
    try:
        logger.info(
            f"Received message request for application {request.application_id}, trace {request.trace_id}"
        )
        set_trace_id(request.trace_id)
        logger.info("Starting SSE stream for application")
        template_id = request.template_id or CONFIG.agent_type
        logger.info(f"Using template: {template_id}")

        agent_types = {
            "template_diff": TemplateDiffAgentImplementation,
            "trpc_agent": TrpcAgentSession,
            "nicegui_agent": NiceguiAgentSession,
            "laravel_agent": LaravelAgentSession,
        }

        if template_id not in agent_types:
            logger.warning(
                f"Unknown template {template_id}, available types: {agent_types}, falling back to default"
            )
            template_id = CONFIG.agent_type

        return StreamingResponse(
            run_agent(request, agent_types[template_id]), media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error processing message request: {str(e)}")
        # Return an HTTP error response for non-SSE errors
        error_response = ErrorResponse(error="Internal Server Error", details=str(e))
        raise HTTPException(status_code=500, detail=error_response.to_json())


@app.get("/")
async def root():
    """Service info and lightweight readiness probe target.

    Unlike /health, this does not open a Dagger connection, so it works
    in environments without a container runtime.
    """
    return {
        "service": "fullstack-agent",
        "status": "running",
        "default_template": CONFIG.default_template_id,
        "templates": CONFIG.available_templates,
        "endpoints": {
            "message": "/message",
            "templates": "/templates",
            "health": "/health",
            "state": "/state/{applicationId}",
        },
    }


@app.get("/me")
async def me(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Return the authenticated identity for the current request.

    Resolution order: Clerk session token, then builder token, then open
    access. Mode "open" reports authenticated=False; when CLERK_ENFORCE is
    enabled, unauthenticated requests are rejected to mirror the guarded
    endpoints' policy.
    """
    state = await resolve_auth_state(credentials)
    if state.failure_reason == "clerk_required":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized - a valid Clerk session token is required",
        )
    if state.mode == "clerk" and state.identity is not None:
        return {"authenticated": True, "mode": "clerk", "identity": state.identity}
    if state.mode == "builder_token" and state.authenticated:
        return {
            "authenticated": True,
            "mode": "builder_token",
            "identity": {"provider": "builder", "user_id": "builder"},
        }
    return {"authenticated": False, "mode": state.mode, "identity": None}


@app.get("/integrations")
async def integrations_status():
    """Report which optional integrations are active (never returns secrets)."""
    return {
        "clerk": {"active": CONFIG.clerk_secret_key is not None},
        "supabase": {
            "active": supabase_client.is_configured(),
            "url": CONFIG.supabase_url,
            "table": supabase_client.AGENT_STATES_TABLE,
            "persistence_ready": supabase_client.is_configured(),
        },
    }


@app.post("/state/{application_id}")
async def save_stored_state(
    application_id: str,
    payload: StateUpdateRequest,
    token: str = Depends(verify_token),
) -> dict:
    """Persist agent state for an application to Supabase.

    Complements GET /state/{application_id} so that durable persistence
    can be exercised over HTTP without executing an agent run (which
    requires a container runtime).
    """
    if not supabase_client.is_configured():
        raise HTTPException(
            status_code=503, detail="Supabase persistence is not configured"
        )
    saved = await supabase_client.save_agent_state(
        application_id, payload.trace_id or "", payload.agent_state
    )
    if not saved:
        raise HTTPException(
            status_code=502, detail="Failed to persist agent state"
        )
    return {
        "applicationId": application_id,
        "saved": True,
        "traceId": payload.trace_id,
    }


@app.get("/state/{application_id}")
async def get_stored_state(
    application_id: str, token: str = Depends(verify_token)
) -> dict:
    """Inspect the persisted Supabase agent state for an application.

    Complements the save/load hooks in run_agent so that durable
    persistence can be verified over HTTP without executing an agent
    run (which requires a container runtime).
    """
    if not supabase_client.is_configured():
        raise HTTPException(
            status_code=503, detail="Supabase persistence is not configured"
        )
    state = await supabase_client.load_agent_state(application_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No persisted state for application {application_id}",
        )
    return {"applicationId": application_id, "agentState": state}


@app.get("/templates")
async def list_templates():
    """List available templates"""
    return {
        "templates": CONFIG.available_templates,
        "default": CONFIG.default_template_id,
    }


@app.get("/health")
async def dagger_healthcheck():
    """Dagger connection health check endpoint"""
    async with dagger.Connection(
        dagger.Config(log_output=open(os.devnull, "w"))
    ) as client:
        # Try a simple Dagger operation to verify connectivity
        container = client.container().from_("alpine:latest")
        version = await container.with_exec(["cat", "/etc/alpine-release"]).stdout()
        return {
            "status": "healthy",
            "dagger_connection": "successful",
            "alpine_version": version.strip(),
        }


def main(
    host: str = "0.0.0.0",
    port: int = 8001,
    reload: bool = False,
    log_level: str = "info",
):
    uvicorn.run(
        "api.agent_server.async_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        log_config=configure_uvicorn_logging(),
    )


if __name__ == "__main__":
    Fire(main)
