import os
from dotenv import load_dotenv

load_dotenv()

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
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./outputs/vector_db")
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_CHUNKS_PATH = os.getenv(
    "RETRIEVAL_CHUNKS_PATH",
    "./outputs/extracted_text_ocr/chunks.jsonl",
)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
RETRIEVAL_VECTOR_WEIGHT = float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.5"))
RETRIEVAL_KEYWORD_WEIGHT = float(os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.3"))
RETRIEVAL_BONUS_WEIGHT = float(os.getenv("RETRIEVAL_BONUS_WEIGHT", "1.5"))
RETRIEVAL_EXACT_WEIGHT = float(os.getenv("RETRIEVAL_EXACT_WEIGHT", "0.4"))
RETRIEVAL_EARLY_PAGE_PENALTY = float(os.getenv("RETRIEVAL_EARLY_PAGE_PENALTY", "1.6"))
FINAL_SOURCE_OVERRIDE_MARGIN = float(os.getenv("FINAL_SOURCE_OVERRIDE_MARGIN", "0.12"))
FINAL_SOURCE_OVERRIDE_RATIO = float(os.getenv("FINAL_SOURCE_OVERRIDE_RATIO", "1.10"))

# 0 disables filtering. For Chroma this is treated as a similarity target;
# for the simple backend it is cosine similarity.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0"))

# App configuration.
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
