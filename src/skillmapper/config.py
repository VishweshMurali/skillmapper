import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    agent_confidence_threshold: float = 0.65
    agent_max_retries: int = 2

    # Optional: LangSmith tracing of the LangGraph agent workflow (retrieve
    # -> compare -> confidence_check -> retry/escalate). Off unless both
    # langsmith_tracing and an API key are set -- no-op otherwise.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "skillmapper"
    langsmith_endpoint: str | None = None

    data_dir: Path = REPO_ROOT / "data"

    @property
    def synthetic_resumes_dir(self) -> Path:
        return self.data_dir / "synthetic" / "resumes"

    @property
    def synthetic_jobs_dir(self) -> Path:
        return self.data_dir / "synthetic" / "job_descriptions"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "store" / "skillmapper.db"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


settings = Settings()

# LangChain/LangSmith read tracing config from real process env vars, not
# from this Settings object -- bridge the two here, once, on import.
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
