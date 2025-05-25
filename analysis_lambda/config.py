# analysis_worker_app/config.py

import os
import json
import logging
import boto3
from botocore.exceptions import ClientError

# 로거 설정 (환경 변수를 통해 로그 레벨 제어 가능)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# boto3와 같은 다른 라이브러리의 로그 레벨도 조정하려면 여기서 추가 설정 가능
# logging.getLogger("boto3").setLevel(logging.WARNING)
# logging.getLogger("botocore").setLevel(logging.WARNING)

# 애플리케이션 로거 (이름을 __name__ 대신 고정 문자열로 하거나, 최상위 패키지 이름 사용도 가능)
logger = logging.getLogger("AnalysisWorkerConfig")
logger.setLevel(LOG_LEVEL)
# 핸들러 설정 (Lambda 환경에서는 기본 핸들러가 이미 있을 수 있으나, 서버 환경에서는 명시적 설정 권장)
# if not logger.handlers: # 핸들러 중복 추가 방지
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
# 위 로깅 설정은 worker.py의 logging.basicConfig와 중복될 수 있으니,
# worker.py에서 logging.basicConfig를 한 번만 호출하고, 각 모듈에서는 getLogger(__name__)만 사용하도록 통일하는 것이 좋습니다.
# 여기서는 getLogger만 사용하고, 기본 설정은 worker.py에 있다고 가정하겠습니다. (이전 worker.py 예시처럼)
logger = logging.getLogger(__name__) # worker.py의 기본 설정을 따름


# --- 일반 설정 (환경 변수에서 직접 로드) ---
# Spring Boot 백엔드 API 엔드포인트
SPRING_BOOT_API_ENDPOINT = os.environ.get("SPRING_BOOT_API_ENDPOINT")
if not SPRING_BOOT_API_ENDPOINT:
    logger.warning("필수 환경 변수 누락: SPRING_BOOT_API_ENDPOINT. 결과 저장이 실패할 수 있습니다.")
    # 운영 환경에서는 애플리케이션 시작 시 에러를 발생시켜 빠르게 인지하도록 할 수 있습니다.
    # raise EnvironmentError("SPRING_BOOT_API_ENDPOINT is required for the application to run.")

# SQS 큐 URL (worker.py에서 직접 os.environ.get으로 사용하므로 여기서는 중복 정의하지 않거나,
# 여기서 정의하고 worker.py에서 config.SQS_QUEUE_URL 형태로 사용)
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
if not SQS_QUEUE_URL:
    logger.warning("필수 환경 변수 누락: SQS_QUEUE_URL. 워커가 SQS 메시지를 가져올 수 없습니다.")


# --- Ingestion Service Configuration (수정/추가) ---
# 수집된 텍스트가 저장될 S3 버킷 이름
INGESTION_S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME') # 기존 환경 변수명 유지 또는 변경 가능
if not INGESTION_S3_BUCKET_NAME:
    logger.warning("환경 변수 S3_BUCKET_NAME (for ingestion text storage)이 설정되지 않았습니다.")
    # 필요시 raise EnvironmentError(...)

# --- FAISS Index S3 Storage Configuration ---
FAISS_INDEX_S3_BUCKET = os.environ.get('FAISS_INDEX_S3_BUCKET')
FAISS_INDEX_S3_PREFIX = os.environ.get('FAISS_INDEX_S3_PREFIX', 'vector_indexes/') # 기본값: 'vector_indexes/'

if not FAISS_INDEX_S3_BUCKET:
    logger.warning("필수 환경 변수 누락: FAISS_INDEX_S3_BUCKET. 벡터 저장소 작업이 실패할 수 있습니다.")

# FAISS 파일 이름 (애플리케이션 내에서 일관되게 사용될 상수)
FAISS_INDEX_FILE_NAME = "main_faiss_index.idx"
FAISS_METADATA_FILE_NAME = "main_faiss_metadata.json"


# --- 민감 정보 로드 (AWS Secrets Manager 사용) ---
# GROQ_API_KEY_SECRET_ARN 환경 변수에는 Secrets Manager에 저장된 Groq API 키 Secret의 전체 ARN을 지정합니다.
GROQ_API_KEY_SECRET_ARN = os.environ.get('GROQ_API_KEY_SECRET_ARN')
# GROQ_API_KEY_FALLBACK 환경 변수는 로컬 개발/테스트 시 또는 Secrets Manager 접근 실패 시 사용할 수 있는 비상용 API 키입니다.
GROQ_API_KEY_FALLBACK = os.environ.get('GROQ_API_KEY_FALLBACK')

_groq_api_key_cache: str | None = None # API 키 캐시 (웜 스타트 시 재사용)
_secrets_manager_client = None # Secrets Manager 클라이언트 캐시

def _get_secrets_manager_client():
    """Secrets Manager 클라이언트를 반환합니다 (필요시 초기화)."""
    global _secrets_manager_client
    if _secrets_manager_client is None:
        logger.debug("Initializing Secrets Manager client.")
        try:
            # 실행 환경(로컬, ECS Fargate)에 따라 리전 명시 필요 여부 확인
            # Fargate에서는 태스크 IAM 역할의 리전을 따름
            _secrets_manager_client = boto3.client('secretsmanager') # region_name=os.environ.get("AWS_REGION")
        except Exception as e:
            logger.error(f"Failed to create boto3 secretsmanager client: {e}", exc_info=True)
    return _secrets_manager_client

def get_groq_api_key_from_secrets_manager() -> str | None:
    """AWS Secrets Manager에서 Groq API 키를 가져옵니다."""
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
            # Secret 값이 JSON 형태 {"GROQ_API_KEY": "your_actual_key"} 로 저장된 경우:
            try:
                secret_json = json.loads(secret_data_str)
                # 실제 Secret에 저장된 JSON 키 이름으로 변경해야 합니다.
                # 예를 들어, Secret 값이 {"myGroqApiKey": "sk-..."} 라면 secret_json.get('myGroqApiKey')
                api_key = secret_json.get('GROQ_API_KEY') # 또는 'api_key', 'groq_api_key' 등 실제 저장된 키
                if api_key:
                    logger.info("Successfully parsed Groq API key from JSON secret in Secrets Manager.")
                    return api_key
                else:
                    logger.warning(f"Specified key ('GROQ_API_KEY') not found in the JSON secret from Secrets Manager: {secret_data_str}. Trying to use the whole SecretString as key.")
                    # 키를 못 찾았지만 SecretString 자체가 키일 수 있으므로 반환 시도
                    return secret_data_str.strip() if secret_data_str.strip() else None
            except json.JSONDecodeError:
                # Secret 값이 JSON 형태가 아니라 Plaintext로 저장된 경우:
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
    """
    Groq API 키를 로드합니다. 먼저 캐시를 확인하고, 없으면 Secrets Manager,
    그래도 없으면 Fallback 환경 변수를 사용합니다.
    """
    global _groq_api_key_cache
    if _groq_api_key_cache:
        logger.debug("Returning cached Groq API key.")
        return _groq_api_key_cache

    logger.debug("Groq API key not in cache. Attempting to load.")
    api_key_from_sm = get_groq_api_key_from_secrets_manager()

    if api_key_from_sm:
        _groq_api_key_cache = api_key_from_sm
        return _groq_api_key_cache

    # Secrets Manager에서 가져오기 실패했거나 ARN이 없는 경우, Fallback 환경 변수 시도
    if GROQ_API_KEY_FALLBACK:
        logger.warning(f"Using Groq API key from fallback environment variable GROQ_API_KEY_FALLBACK. This is NOT recommended for production.")
        _groq_api_key_cache = GROQ_API_KEY_FALLBACK
        return _groq_api_key_cache

    logger.error("Groq API key could not be retrieved from Secrets Manager AND no fallback environment variable (GROQ_API_KEY_FALLBACK) is set.")
    return None

# --- 모듈 로드 시 (애플리케이션 시작 시) 주요 설정값 초기화 ---
# GROQ_API_KEY는 이제 다른 모듈에서 config.GROQ_API_KEY로 바로 사용 가능
GROQ_API_KEY: str | None = load_groq_api_key()

logger.info("Configuration module initialized for Analysis Worker.")
# 초기화 시 주요 설정값 상태 로깅
if GROQ_API_KEY:
    logger.info("Groq API Key has been loaded and is available.")
else:
    logger.warning("Groq API Key is NOT available after initialization. Summarization feature will likely fail or be skipped.")

if SPRING_BOOT_API_ENDPOINT:
    logger.info(f"Spring Boot API Endpoint: {SPRING_BOOT_API_ENDPOINT}")
else:
    logger.warning("Spring Boot API Endpoint is NOT configured.")

if FAISS_INDEX_S3_BUCKET:
    logger.info(f"FAISS S3 Bucket: {FAISS_INDEX_S3_BUCKET}, Prefix: {FAISS_INDEX_S3_PREFIX.rstrip('/')}")
else:
    logger.warning("FAISS S3 Bucket is NOT configured. Vector store operations will fail.")
