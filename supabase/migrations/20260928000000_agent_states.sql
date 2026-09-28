-- Durable agent-state storage for the agent server.
-- Lets a conversation be resumed on any replica of a multi-instance
-- deployment via api.supabase_client.save_agent_state / load_agent_state.

create table if not exists public.agent_states (
    application_id text primary key,
    last_trace_id text,
    state jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

-- Server-side access only: the service/secret key bypasses RLS, while
-- anon/authenticated roles get no access until policies are added.
alter table public.agent_states enable row level security;
