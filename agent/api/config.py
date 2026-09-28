import os


class Config:
    _instance = None

    # Available agent template IDs, matching the agent_types registry in api.agent_server.async_server
    AVAILABLE_TEMPLATES = (
        "template_diff",
        "trpc_agent",
        "nicegui_agent",
        "laravel_agent",
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    @property
    def agent_type(self):
        return os.getenv("CODEGEN_AGENT", "trpc_agent")

    @property
    def available_templates(self):
        return list(self.AVAILABLE_TEMPLATES)

    @property
    def default_template_id(self):
        return self.agent_type

    @property
    def builder_token(self):
        return os.getenv("BUILDER_TOKEN")

    @property
    def snapshot_bucket(self):
        return os.getenv("SNAPSHOT_BUCKET", None)

    @property
    def clerk_secret_key(self):
        """Clerk secret key (sk_...); when unset, Clerk auth is disabled."""
        return os.getenv("CLERK_SECRET_KEY")

    @property
    def clerk_authorized_parties(self):
        """Comma-separated list of allowed `azp` values for Clerk tokens."""
        raw = os.getenv("CLERK_AUTHORIZED_PARTIES", "")
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def clerk_enforce(self):
        """When True, requests without a valid Clerk or builder token are rejected."""
        return os.getenv("CLERK_ENFORCE", "").lower() in ("1", "true", "yes")

    @property
    def supabase_url(self):
        """Supabase project URL; when unset, Supabase persistence is disabled."""
        return os.getenv("SUPABASE_URL")

    @property
    def supabase_anon_key(self):
        """Anon/publishable key for RLS-scoped client access.

        Accepts both the legacy SUPABASE_ANON_KEY and the newer
        SUPABASE_PUBLISHABLE_KEY naming.
        """
        return os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")

    @property
    def supabase_service_role_key(self):
        """Service/secret key enabling server-side agent-state persistence.

        Accepts both the legacy SUPABASE_SERVICE_ROLE_KEY and the newer
        SUPABASE_SECRET_KEY naming.
        """
        return os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
            "SUPABASE_SECRET_KEY"
        )


CONFIG = Config()
