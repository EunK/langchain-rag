import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    supabase_url: str
    supabase_service_key: str
    storage_bucket: str = "manual-pages"

    # Retrieval
    top_k: int = 10

    # UI slider default = 0.00
    similarity_threshold: float = 0.00

    # Related pages
    max_related_pages: int = 6

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Models
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536


def _ensure_trailing_slash(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    return url if url.endswith("/") else (url + "/")


load_dotenv()


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        supabase_url=_ensure_trailing_slash(os.getenv("SUPABASE_URL", "")),
        supabase_service_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
