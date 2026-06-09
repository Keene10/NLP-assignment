import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_path(relative_path: str) -> str:
    return os.path.join(ROOT_DIR, relative_path)


def _default_reranker_model() -> str:
    configured = os.getenv("PAGE_RERANKER_MODEL", "").strip()
    if configured:
        return configured
    for relative_path in (
        "models/bge-reranker-v2-m3",
        "bge-reranker-v2-m3",
    ):
        if os.path.isdir(_project_path(relative_path)):
            return f"./{relative_path}"
    return "BAAI/bge-reranker-v2-m3"

# OpenAI configuration. The project can run retrieval without this key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

# Model configuration. m3e-small is included in this workspace and works well
# for Chinese financial-report retrieval.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "./m3e-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Vector database configuration.
# auto: use Chroma when installed, otherwise use the local simple backend.
VECTOR_DB_BACKEND = os.getenv("VECTOR_DB_BACKEND", "auto").lower()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./outputs/vector_db_multimodal")
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_CHUNKS_PATH = os.getenv(
    "RETRIEVAL_CHUNKS_PATH",
    "./outputs/extracted_text_ocr_multimodal/chunks.jsonl",
)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
RETRIEVAL_VECTOR_WEIGHT = float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.5"))
RETRIEVAL_KEYWORD_WEIGHT = float(os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.3"))
RETRIEVAL_BONUS_WEIGHT = float(os.getenv("RETRIEVAL_BONUS_WEIGHT", "1.5"))
RETRIEVAL_EXACT_WEIGHT = float(os.getenv("RETRIEVAL_EXACT_WEIGHT", "0.4"))
RETRIEVAL_EARLY_PAGE_PENALTY = float(os.getenv("RETRIEVAL_EARLY_PAGE_PENALTY", "1.6"))
FINAL_SOURCE_OVERRIDE_MARGIN = float(os.getenv("FINAL_SOURCE_OVERRIDE_MARGIN", "0.12"))
FINAL_SOURCE_OVERRIDE_RATIO = float(os.getenv("FINAL_SOURCE_OVERRIDE_RATIO", "1.10"))
PAGE_RERANKER_MODEL = _default_reranker_model()
PAGE_RERANKER_CANDIDATES = int(os.getenv("PAGE_RERANKER_CANDIDATES", "10"))
PAGE_RERANKER_BATCH_SIZE = int(os.getenv("PAGE_RERANKER_BATCH_SIZE", "8"))
PAGE_RERANKER_MAX_CHARS = int(os.getenv("PAGE_RERANKER_MAX_CHARS", "1200"))
PAGE_RERANKER_WEIGHT = float(os.getenv("PAGE_RERANKER_WEIGHT", "0.85"))
PAGE_RERANKER_NEIGHBOR_PAGES = int(os.getenv("PAGE_RERANKER_NEIGHBOR_PAGES", "1"))
STRICT_LOCAL_RERANKER = os.getenv("STRICT_LOCAL_RERANKER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
TARGETED_RETRIEVAL_ENABLED = os.getenv("TARGETED_RETRIEVAL_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
CHART_PAGE_BOOST = float(os.getenv("CHART_PAGE_BOOST", "4.0"))
CHART_DIRECTORY_PENALTY = float(os.getenv("CHART_DIRECTORY_PENALTY", "3.0"))
SECTION_ROUTE_BONUS = float(os.getenv("SECTION_ROUTE_BONUS", "3.0"))
SEMANTIC_SECTION_ROUTING_ENABLED = os.getenv(
    "SEMANTIC_SECTION_ROUTING_ENABLED", "1"
).strip().lower() not in {"0", "false", "no"}
SEMANTIC_SECTION_TOP_K = int(os.getenv("SEMANTIC_SECTION_TOP_K", "2"))
SEMANTIC_SECTION_BONUS = float(os.getenv("SEMANTIC_SECTION_BONUS", "0.8"))
SEMANTIC_SECTION_MIN_SCORE = float(os.getenv("SEMANTIC_SECTION_MIN_SCORE", "0.35"))
MANUAL_SECTION_RULES_ENABLED = os.getenv("MANUAL_SECTION_RULES_ENABLED", "0").strip().lower() not in {
    "0",
    "false",
    "no",
}
CONTENT_ANCHOR_BOOSTS_ENABLED = os.getenv("CONTENT_ANCHOR_BOOSTS_ENABLED", "0").strip().lower() not in {
    "0",
    "false",
    "no",
}

# 0 disables filtering. For Chroma this is treated as a similarity target;
# for the simple backend it is cosine similarity.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0"))

# App configuration.
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
