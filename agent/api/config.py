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


CONFIG = Config()
