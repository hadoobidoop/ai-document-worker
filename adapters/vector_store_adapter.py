# analysis_lambda/adapters/vector_store_adapter.py
import logging
import numpy as np
import boto3
from botocore.exceptions import ClientError
import json
import os # For /tmp/ path

from langchain.vectorstores import faiss

from analysis_lambda import config

logger = logging.getLogger(__name__)
s3_client = boto3.client('s3')

faiss_index = None # 실제로는 faiss.Index 객체
index_to_metadata_map = []
_faiss_index_loaded_from_s3 = False
EMBEDDING_DIMENSION = 768 # 실제 사용하는 모델의 차원 수로 변경해야 함!

def load_or_initialize_faiss_index():
    global faiss_index, index_to_metadata_map, _faiss_index_loaded_from_s3
    if _faiss_index_loaded_from_s3: return
    logger.info("FAISS Index 로드/초기화 시도 (현재는 Placeholder)...")
    # Priority 2에서 S3로부터 로드 또는 신규 생성 로직 구현
    # 예: faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    _faiss_index_loaded_from_s3 = True
    logger.info("FAISS Index 로드/초기화 Placeholder 완료.")

def save_faiss_index_and_metadata_to_s3() -> bool:
    logger.info("FAISS Index S3 저장 시도 (현재는 Placeholder)...")
    if faiss_index is None:
        logger.warning("FAISS 인덱스가 없어 저장할 수 없습니다 (Placeholder 동작).")
        return False
    # Priority 2에서 실제 S3 저장 로직 구현
    logger.info("FAISS Index S3 저장 Placeholder 완료.")
    return True

def add_embeddings_to_faiss(embeddings_data: list[dict]) -> bool:
    logger.info(f"{len(embeddings_data)}개의 임베딩 FAISS 추가 시도 (현재는 Placeholder)...")
    if faiss_index is None:
        logger.error("FAISS 인덱스가 초기화되지 않아 임베딩을 추가할 수 없습니다 (Placeholder 동작).")
        return False
    # Priority 2에서 실제 FAISS 인덱스에 벡터 추가 및 메타데이터 업데이트 로직 구현
    logger.info("임베딩 FAISS 추가 Placeholder 완료. (S3 저장은 save_faiss_index_and_metadata_to_s3가 담당)")
    # 중요: 이 함수는 메모리 내 인덱스만 업데이트하고, S3 저장은 별도 호출(예: worker.py의 종료 시점 또는 주기적 백업)로 가정
    return True # 여기서는 S3 저장 성공 여부가 아닌, 메모리 내 추가 성공 여부 (또는 시도 여부)

# --- FAISS Index 전역 변수 (Lambda 호출 간 상태 유지) ---
faiss_index: faiss.Index | None = None
# index_to_metadata_map는 FAISS 인덱스의 내부 ID (0부터 시작하는 정수)를
# 실제 문서/청크 정보 (doc_id, chunk_id, chunk_text_preview)에 매핑합니다.
index_to_metadata_map: list[dict] = [] # list of {'doc_id': str, 'chunk_id': int, 'text': str}
# 이 리스트의 인덱스가 FAISS 인덱스 ID와 일치

# FAISS 인덱스 차원 (사용하는 임베딩 모델에 따라 다름)
# 예: jhgan/ko-sroberta-multitask는 768차원
EMBEDDING_DIMENSION = 768 # 사용하는 임베딩 모델의 차원 수로 반드시 변경!

_faiss_index_loaded_from_s3 = False # S3에서 로드되었는지 여부 플래그

def _get_s3_index_path():
    return os.path.join(config.FAISS_INDEX_S3_PREFIX, config.FAISS_INDEX_FILE_NAME).replace("\\", "/")

def _get_s3_metadata_path():
    return os.path.join(config.FAISS_INDEX_S3_PREFIX, config.FAISS_METADATA_FILE_NAME).replace("\\", "/")

def _get_local_tmp_index_path():
    return f"/tmp/{config.FAISS_INDEX_FILE_NAME}" # Lambda 임시 스토리지

def _get_local_tmp_metadata_path():
    return f"/tmp/{config.FAISS_METADATA_FILE_NAME}"

