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


settings = Settings()
