# config/settings.py

import os
import json
import logging
import boto3
from botocore.exceptions import ClientError
from pathlib import Path # 프로젝트 루트 경로를 얻기 위해 추가

logger = logging.getLogger(__name__)

# --- 프로젝트 루트 경로 ---
# 이 settings.py 파일의 위치를 기준으로 프로젝트 루트를 추정합니다.
# settings.py가 config 폴더 안에 있으므로, Path(__file__).resolve().parent.parent는 프로젝트 루트가 됩니다.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# --- 환경 변수 및 기본 설정 ---
SQS_QUEUE_URL = os.environ.get("AI_DOC_PROCESSING_SQS_URL")
if not SQS_QUEUE_URL:
    logger.warning("필수 환경 변수 누락: AI_DOC_PROCESSING_SQS_URL. SQS 관련 기능이 작동하지 않을 수 있습니다.")

SPRING_BOOT_API_ENDPOINT = os.environ.get("SPRING_BOOT_API_ENDPOINT")
if not SPRING_BOOT_API_ENDPOINT:
    logger.warning("필수 환경 변수 누락: SPRING_BOOT_API_ENDPOINT. 결과 저장이 실패할 수 있습니다.")

INGESTION_S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
if not INGESTION_S3_BUCKET_NAME:
    logger.warning("환경 변수 S3_BUCKET_NAME (for ingestion text storage)이 설정되지 않았습니다.")

FAISS_INDEX_S3_BUCKET = os.environ.get('FAISS_INDEX_S3_BUCKET')
FAISS_INDEX_S3_PREFIX = os.environ.get('FAISS_INDEX_S3_PREFIX', 'vector_indexes/')
if not FAISS_INDEX_S3_BUCKET:
    logger.warning("필수 환경 변수 누락: FAISS_INDEX_S3_BUCKET. 벡터 저장소 작업이 실패할 수 있습니다.")

FAISS_INDEX_FILE_NAME = "main_faiss_index.idx"
FAISS_METADATA_FILE_NAME = "main_faiss_metadata.json"

# --- NLP Model & Processing Configurations ---
EMBEDDING_MODEL_NAME_OR_PATH = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH", "jhgan/ko-sroberta-multitask")
EMBEDDING_CHUNK_SIZE_CHARS = int(os.environ.get("EMBEDDING_CHUNK_SIZE_CHARS", 1000))
EMBEDDING_CHUNK_OVERLAP_CHARS = int(os.environ.get("EMBEDDING_CHUNK_OVERLAP_CHARS", 100))
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 768))

SUMMARIZER_MODEL_NAME = os.environ.get("SUMMARIZER_MODEL_NAME", "llama3-8b-8192")
SUMMARIZER_MAX_CHARS_SINGLE_API_CALL = int(os.environ.get("SUMMARIZER_MAX_CHARS_SINGLE_API_CALL", 15000))
SUMMARIZER_CHUNK_TARGET_SIZE_CHARS = int(os.environ.get("SUMMARIZER_CHUNK_TARGET_SIZE_CHARS", 3500))
SUMMARIZER_CHUNK_OVERLAP_CHARS = int(os.environ.get("SUMMARIZER_CHUNK_OVERLAP_CHARS", 300))
SUMMARIZER_MAX_TOKENS_CHUNK_SUMMARY = int(os.environ.get("SUMMARIZER_MAX_TOKENS_CHUNK_SUMMARY", 250))
SUMMARIZER_MIN_TEXT_LENGTH_LONG_SUMMARY_CHARS = int(os.environ.get("SUMMARIZER_MIN_TEXT_LENGTH_LONG_SUMMARY_CHARS", 400))
SUMMARIZER_LONG_SUMMARY_TARGET_CHARS = int(os.environ.get("SUMMARIZER_LONG_SUMMARY_TARGET_CHARS", 800))
SUMMARIZER_LONG_SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARIZER_LONG_SUMMARY_MAX_TOKENS", 450))
SUMMARIZER_SHORT_SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARIZER_SHORT_SUMMARY_MAX_TOKENS", 150))

TAG_EXTRACTOR_NUM_TAGS = int(os.environ.get("TAG_EXTRACTOR_NUM_TAGS", 3))
CATEGORIZER_SIMILARITY_THRESHOLD = float(os.environ.get("CATEGORIZER_SIMILARITY_THRESHOLD", 0.05))

# 카테고리 정의 파일 경로 설정
# 환경 변수 CATEGORIES_FILE_PATH가 있으면 그 값을 사용하고, 없으면 프로젝트 루트의 categories.json을 기본값으로 사용
DEFAULT_CATEGORIES_FILE_PATH = PROJECT_ROOT_DIR / "categories.json"
CATEGORIES_FILE_PATH = os.environ.get("CATEGORIES_FILE_PATH", str(DEFAULT_CATEGORIES_FILE_PATH))
if not Path(CATEGORIES_FILE_PATH).is_file():
    logger.warning(f"Categories definition file not found at: {CATEGORIES_FILE_PATH}. Categorizer may not work correctly.")


# --- Worker Settings ---
FAISS_SAVE_INTERVAL_MESSAGES = int(os.environ.get("FAISS_SAVE_INTERVAL_MESSAGES", 10))

# --- 민감 정보 로드 (AWS Secrets Manager 사용) ---
# (기존 Secrets Manager 로직 유지)
GROQ_API_KEY_SECRET_ARN = os.environ.get('GROQ_API_KEY_SECRET_ARN')
GROQ_API_KEY_FALLBACK = os.environ.get('GROQ_API_KEY_FALLBACK')
_groq_api_key_cache: str | None = None
_secrets_manager_client = None

def _get_secrets_manager_client():
    global _secrets_manager_client
    if _secrets_manager_client is None:
        logger.debug("Initializing Secrets Manager client.")
        try:
            _secrets_manager_client = boto3.client('secretsmanager')
        except Exception as e:
            logger.error(f"Failed to create boto3 secretsmanager client: {e}", exc_info=True)
    return _secrets_manager_client

def get_groq_api_key_from_secrets_manager() -> str | None:
    if not GROQ_API_KEY_SECRET_ARN:
        logger.info("GROQ_API_KEY_SECRET_ARN is not set. Skipping Secrets Manager.")
        return None
    client = _get_secrets_manager_client()
    if not client:
        logger.error("Secrets Manager client is not available. Cannot fetch Groq API key.")
        return None
    logger.info(f"Attempting to fetch Groq API key from Secrets Manager: {GROQ_API_KEY_SECRET_ARN}")
    try:
        secret_value_response = client.get_secret_value(SecretId=GROQ_API_KEY_SECRET_ARN)
        if 'SecretString' in secret_value_response:
            secret_data_str = secret_value_response['SecretString']
            try:
                secret_json = json.loads(secret_data_str)
                api_key = secret_json.get('GROQ_API_KEY')
                if api_key:
                    logger.info("Successfully parsed Groq API key from JSON secret in Secrets Manager.")
                    return api_key
                else:
                    logger.warning(f"Specified key ('GROQ_API_KEY') not found in the JSON secret. Trying to use the whole SecretString.")
                    return secret_data_str.strip() if secret_data_str.strip() else None
            except json.JSONDecodeError:
                logger.info("Successfully fetched Groq API key as plaintext from Secrets Manager.")
                return secret_data_str.strip() if secret_data_str.strip() else None
        else:
            logger.error("Groq API key in Secrets Manager is binary, not string. Cannot use.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        logger.error(f"Secrets Manager ClientError fetching Groq API key (ARN: {GROQ_API_KEY_SECRET_ARN}): {error_code} - {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error fetching Groq API key from Secrets Manager: {e}", exc_info=True)
    return None

def load_groq_api_key() -> str | None:
    global _groq_api_key_cache
    if _groq_api_key_cache:
        logger.debug("Returning cached Groq API key.")
        return _groq_api_key_cache
    logger.debug("Groq API key not in cache. Attempting to load.")
    api_key_from_sm = get_groq_api_key_from_secrets_manager()
    if api_key_from_sm:
        _groq_api_key_cache = api_key_from_sm
        return _groq_api_key_cache
    if GROQ_API_KEY_FALLBACK:
        logger.warning(f"Using Groq API key from fallback environment variable GROQ_API_KEY_FALLBACK. This is NOT recommended for production.")
        _groq_api_key_cache = GROQ_API_KEY_FALLBACK
        return _groq_api_key_cache
    logger.error("Groq API key could not be retrieved from Secrets Manager AND no fallback environment variable (GROQ_API_KEY_FALLBACK) is set.")
    return None

GROQ_API_KEY: str | None = load_groq_api_key()

# --- Testing Configurations ---
MOCK_S3 = None
TMP_DIR = os.environ.get("TMP_DIR", "/tmp")


logger.info("Configuration module (config.settings) initialized with NLP, Testing, and Categories File Path parameters.")
logger.info(f"Categories Definition File Path: {CATEGORIES_FILE_PATH}")
# (기존 로깅 메시지 유지)
logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME_OR_PATH}, Dimension: {EMBEDDING_DIMENSION}")
logger.info(f"Summarizer Model: {SUMMARIZER_MODEL_NAME}")
logger.info(f"Temporary Directory (TMP_DIR): {TMP_DIR}")
if GROQ_API_KEY:
    logger.info("Groq API Key has been loaded.")
else:
    logger.warning("Groq API Key is NOT available after initialization.")

