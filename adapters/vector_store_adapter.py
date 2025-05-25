# analysis_worker_app/adapters/vector_store_adapter.py
import logging
import faiss # faiss-cpu
import numpy as np
import boto3
from botocore.exceptions import ClientError
import json
import os # Lambda/Fargate의 /tmp/ 경로 사용 위함

from analysis_lambda import config

from config import settings

# config 모듈에서 S3 버킷, 경로 등 설정 가져오기

logger = logging.getLogger(__name__)

# --- FAISS Index 및 S3 클라이언트 (모듈 로드 시 초기화) ---
s3_client = boto3.client('s3')

# FAISS 인덱스 객체 및 메타데이터 맵 (메모리에 유지)
# faiss_index: 실제 FAISS 인덱스 객체
# index_to_metadata_map: FAISS 인덱스의 내부 ID (0부터 시작하는 정수)를
#                        실제 문서/청크 정보 (doc_id, chunk_id, chunk_text_preview)에 매핑하는 리스트.
#                        이 리스트의 인덱스가 FAISS 인덱스 ID와 일치해야 합니다.
faiss_index: faiss.Index | None = None
index_to_metadata_map: list[dict] = []

# 사용하는 임베딩 모델의 차원 수 (매우 중요! 실제 모델과 일치시켜야 함)
# 예: "jhgan/ko-sroberta-multitask" 모델은 768차원 벡터를 생성합니다.
EMBEDDING_DIMENSION = 768

# S3에서 성공적으로 로드되었거나, 새로 초기화되었는지 나타내는 플래그
_faiss_store_initialized_successfully = False

def _get_s3_index_key() -> str:
    """S3에 저장될 FAISS 인덱스 파일의 전체 키(경로)를 반환합니다."""
    # config.FAISS_INDEX_S3_PREFIX가 "/"로 끝나지 않을 경우를 대비하여 os.path.join 사용 회피
    prefix = config.FAISS_INDEX_S3_PREFIX.rstrip('/')
    return f"{prefix}/{config.FAISS_INDEX_FILE_NAME}"

def _get_s3_metadata_key() -> str:
    """S3에 저장될 메타데이터 파일의 전체 키(경로)를 반환합니다."""
    prefix = config.FAISS_INDEX_S3_PREFIX.rstrip('/')
    return f"{prefix}/{config.FAISS_METADATA_FILE_NAME}"

def _get_local_tmp_index_path() -> str:
    """Lambda/Fargate의 임시 저장소(/tmp) 내 FAISS 인덱스 파일 경로를 반환합니다."""
    return f"/tmp/{config.FAISS_INDEX_FILE_NAME}"

def _get_local_tmp_metadata_path() -> str:
    """Lambda/Fargate의 임시 저장소(/tmp) 내 메타데이터 파일 경로를 반환합니다."""
    return f"/tmp/{config.FAISS_METADATA_FILE_NAME}"


def load_or_initialize_faiss_index():
    """
    S3에서 FAISS 인덱스와 메타데이터를 로드합니다.
    S3에 해당 파일이 없으면, 새로운 빈 FAISS 인덱스와 메타데이터를 초기화합니다.
    이 함수는 워커 애플리케이션 시작 시 (콜드 스타트 시) 한 번만 호출되어야 합니다.
    """
    global faiss_index, index_to_metadata_map, _faiss_store_initialized_successfully
    if _faiss_store_initialized_successfully: # 이미 성공적으로 로드/초기화 되었다면 중복 실행 방지
        logger.info("FAISS index and metadata already loaded/initialized.")
        return

    if not settings.FAISS_INDEX_S3_BUCKET:
        logger.error("FAISS_INDEX_S3_BUCKET is not configured in settings.py. Cannot load or initialize FAISS index.")
        # _faiss_store_initialized_successfully는 False로 유지하여 이후 작업 실패 유도
        return

    s3_index_key = _get_s3_index_key()
    s3_metadata_key = _get_s3_metadata_key()
    local_tmp_index_path = _get_local_tmp_index_path()
    local_tmp_metadata_path = _get_local_tmp_metadata_path()

    index_loaded = False
    metadata_loaded = False

    try:
        # 1. S3에서 인덱스 파일 다운로드 시도
        logger.info(f"Attempting to download FAISS index from s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_index_key} to {local_tmp_index_path}")
        s3_client.download_file(config.FAISS_INDEX_S3_BUCKET, s3_index_key, local_tmp_index_path)
        faiss_index = faiss.read_index(local_tmp_index_path)
        logger.info(f"FAISS index loaded from S3. Index contains {faiss_index.ntotal} vectors.")
        index_loaded = True

        # 2. S3에서 메타데이터 파일 다운로드 시도
        logger.info(f"Attempting to download FAISS metadata from s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_metadata_key} to {local_tmp_metadata_path}")
        s3_client.download_file(config.FAISS_INDEX_S3_BUCKET, s3_metadata_key, local_tmp_metadata_path)
        with open(local_tmp_metadata_path, 'r', encoding='utf-8') as f:
            index_to_metadata_map = json.load(f)
        logger.info(f"FAISS metadata loaded from S3. Contains {len(index_to_metadata_map)} entries.")
        metadata_loaded = True

        if faiss_index.ntotal != len(index_to_metadata_map):
            logger.critical(
                f"CRITICAL: FAISS index vector count ({faiss_index.ntotal}) and "
                f"metadata count ({len(index_to_metadata_map)}) mismatch! "
                f"This indicates a corrupted or inconsistent state. Manual intervention may be required."
            )
            # 이 경우, 애플리케이션을 중단하거나, 빈 인덱스로 강제 초기화하는 등의 정책 필요
            # 여기서는 일단 로깅만 하고 진행 (매우 위험한 상태)
            # raise Exception("FAISS index and metadata count mismatch, potential data corruption.")
            _faiss_store_initialized_successfully = False # 오류 상태로 설정
            return

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ['404', 'NoSuchKey', '403']: # 403은 파일은 있는데 권한 없는 경우도 포함될 수 있음
            logger.warning(f"FAISS index or metadata file not found on S3 (or access denied). Error code: {error_code}. Path: s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_index_key} or {s3_metadata_key}. Initializing a new empty index.")
            # 파일이 없을 경우 새 인덱스 생성
            if not index_loaded: faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            if not metadata_loaded: index_to_metadata_map = []
        else:
            logger.error(f"S3 ClientError during FAISS index/metadata download: {e}", exc_info=False)
            # 여기서 예외 발생 시 faiss_index가 None으로 유지될 수 있음.
            _faiss_store_initialized_successfully = False # 오류 상태로 설정
            return
    except FileNotFoundError: # 로컬 임시 파일 관련 (download_file 실패 후 read_index 시도 등)
        logger.warning(f"Local FAISS index/metadata file not found at /tmp/ after S3 download attempt. This might indicate issues with /tmp/ or S3 download. Initializing new index.")
        if not index_loaded: faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        if not metadata_loaded: index_to_metadata_map = []
    except Exception as e: # faiss.read_index() 또는 json.load() 실패 등
        logger.error(f"Unexpected error loading or initializing FAISS index: {e}", exc_info=True)
        _faiss_store_initialized_successfully = False # 오류 상태로 설정
        return
    finally:
        # 사용한 임시 파일 정리 (선택 사항, /tmp/는 실행 환경 종료 시 자동 정리됨)
        if os.path.exists(local_tmp_index_path): os.remove(local_tmp_index_path)
        if os.path.exists(local_tmp_metadata_path): os.remove(local_tmp_metadata_path)

    if faiss_index is not None:
        _faiss_store_initialized_successfully = True
        logger.info("FAISS index and metadata are ready.")
    else: # 모든 시도 후에도 faiss_index가 None이면 초기화 실패
        logger.critical("Failed to load or initialize FAISS index after all attempts.")
        _faiss_store_initialized_successfully = False


def save_faiss_index_and_metadata_to_s3() -> bool:
    """현재 메모리 상의 FAISS 인덱스와 메타데이터를 S3에 저장(덮어쓰기)합니다."""
    global faiss_index, index_to_metadata_map
    if not _faiss_store_initialized_successfully or faiss_index is None: # 초기화가 성공적으로 안됐거나 인덱스가 없으면 저장 불가
        logger.error("FAISS index is not properly initialized. Cannot save to S3.")
        return False
    if not config.FAISS_INDEX_S3_BUCKET:
        logger.error("FAISS_INDEX_S3_BUCKET is not configured. Cannot save to S3.")
        return False

    s3_index_key = _get_s3_index_key()
    s3_metadata_key = _get_s3_metadata_key()
    local_tmp_index_path = _get_local_tmp_index_path()
    local_tmp_metadata_path = _get_local_tmp_metadata_path()

    # 동시성 문제 방지를 위한 매우 간단한 버전 관리 또는 락킹 메커니즘 고려 가능 (MVP 이후)
    # 예: S3 객체 ETag를 확인하여 다른 프로세스가 중간에 수정했는지 감지 등

    logger.info(f"Attempting to save FAISS index and metadata to S3.")
    try:
        # 1. 로컬 임시 파일에 인덱스 저장
        logger.debug(f"Saving FAISS index ({faiss_index.ntotal} vectors) to local path: {local_tmp_index_path}")
        faiss.write_index(faiss_index, local_tmp_index_path)

        # 2. 로컬 임시 파일에 메타데이터 저장
        logger.debug(f"Saving FAISS metadata ({len(index_to_metadata_map)} entries) to local path: {local_tmp_metadata_path}")
        with open(local_tmp_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(index_to_metadata_map, f, ensure_ascii=False) # indent=2 등으로 가독성 높일 수 있음

        # 3. S3에 업로드
        logger.info(f"Uploading FAISS index to s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_index_key}")
        s3_client.upload_file(local_tmp_index_path, config.FAISS_INDEX_S3_BUCKET, s3_index_key)

        logger.info(f"Uploading FAISS metadata to s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_metadata_key}")
        s3_client.upload_file(local_tmp_metadata_path, config.FAISS_INDEX_S3_BUCKET, s3_metadata_key)

        logger.info("FAISS index and metadata successfully saved to S3.")
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS index or metadata to S3: {e}", exc_info=True)
        return False
    finally:
        if os.path.exists(local_tmp_index_path): os.remove(local_tmp_index_path)
        if os.path.exists(local_tmp_metadata_path): os.remove(local_tmp_metadata_path)


def add_embeddings_to_faiss(embeddings_data: list[dict]) -> bool:
    """
    주어진 임베딩 데이터 리스트를 메모리 내 FAISS 인덱스에 추가하고, 메타데이터를 업데이트합니다.
    실제 S3 저장은 `save_faiss_index_and_metadata_to_s3` 함수를 통해 별도로 호출되어야 합니다.
    (주의: 이 함수 자체는 S3에 즉시 저장하지 않음. 동시성 문제를 고려한 설계)

    Args:
        embeddings_data: 각 요소가 {'document_id': str, 'chunk_id': int,
                                  'chunk_text': str, 'embedding_vector': list[float]}
                         형태의 딕셔너리인 리스트.
    Returns:
        메모리 내 인덱스에 성공적으로 추가되었으면 True, 아니면 False.
    """
    global faiss_index, index_to_metadata_map
    if not _faiss_store_initialized_successfully or faiss_index is None:
        logger.error("FAISS index not properly initialized. Cannot add embeddings to in-memory index.")
        return False

    if not embeddings_data:
        logger.info("No new embeddings data to add to FAISS.")
        return True # 추가할 내용 없으므로 성공으로 간주

    logger.info(f"Adding {len(embeddings_data)} new embedding(s) to in-memory FAISS index.")

    vectors_to_add = []
    new_metadata_entries = []

    for item in embeddings_data:
        vector = item.get('embedding_vector')
        # EMBEDDING_DIMENSION은 사용하는 모델에 따라 정확히 설정되어야 함
        if vector and len(vector) == EMBEDDING_DIMENSION:
            vectors_to_add.append(vector)
            new_metadata_entries.append({
                "doc_id": str(item.get("document_id")), # document_id를 문자열로 통일
                "chunk_id": int(item.get("chunk_id")), # chunk_id를 정수로 통일
                "text_preview": item.get("chunk_text", "")[:250] + "..." # 검색 결과에 보여줄 텍스트 미리보기 (길이 조절)
            })
        else:
            logger.warning(
                f"Invalid or missing embedding vector for doc_id {item.get('document_id')}, "
                f"chunk_id {item.get('chunk_id')}. Expected dimension {EMBEDDING_DIMENSION}, got {len(vector) if vector else 'None'}. Skipping."
            )

    if not vectors_to_add:
        logger.info("No valid vectors to add to FAISS after filtering.")
        return True # 추가할 유효한 벡터가 없으므로 성공으로 간주

    try:
        # FAISS는 float32 타입의 numpy array를 기대합니다.
        np_vectors_to_add = np.array(vectors_to_add).astype(np.float32)

        # FAISS 인덱스에 벡터 추가 (메모리 내에서만 발생)
        faiss_index.add(np_vectors_to_add)

        # 메타데이터 리스트 확장 (메모리 내에서만 발생)
        # 이 순서는 FAISS 내부 ID와 정확히 일치해야 합니다.
        index_to_metadata_map.extend(new_metadata_entries)

        logger.info(f"Successfully added {len(vectors_to_add)} vectors to in-memory FAISS index. Total vectors in memory: {faiss_index.ntotal}")
        # 현재 index_to_metadata_map의 길이와 faiss_index.ntotal이 일치하는지 확인하는 로직 추가 가능
        if faiss_index.ntotal != len(index_to_metadata_map):
            logger.critical(f"CRITICAL INCONSISTENCY after adding to FAISS: index count {faiss_index.ntotal}, metadata count {len(index_to_metadata_map)}")
            # 이 경우 심각한 문제이므로, 롤백 또는 에러 처리가 필요할 수 있음
            return False
        return True

    except Exception as e:
        logger.error(f"Error adding embeddings to in-memory FAISS index: {e}", exc_info=True)
        return False

# --- (검색 함수 예시 - 실제 채팅 API에서 사용, MVP 이후 구현) ---
def search_in_faiss(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    메모리 내 FAISS 인덱스에서 주어진 쿼리 임베딩과 유사한 상위 K개의 결과를 검색합니다.
    """
    global faiss_index, index_to_metadata_map
    if not _faiss_store_initialized_successfully or faiss_index is None or faiss_index.ntotal == 0:
        logger.warning("FAISS index not initialized or empty. Cannot perform search.")
        return []

    if not query_embedding or len(query_embedding) != EMBEDDING_DIMENSION:
        logger.error(f"Invalid query embedding for FAISS search. Expected dimension {EMBEDDING_DIMENSION}, got {len(query_embedding) if query_embedding else 'None'}.")
        return []

    logger.info(f"Searching for top {top_k} similar items in FAISS index (total items: {faiss_index.ntotal}).")
    try:
        # 쿼리 임베딩도 FAISS가 기대하는 형태로 변환 (1, EMBEDDING_DIMENSION) 크기의 float32 numpy array
        query_vector_np = np.array([query_embedding]).astype(np.float32)

        # search 메소드는 (distances, indices) 튜플을 반환
        # distances: 각 결과까지의 L2 거리 (작을수록 유사)
        # indices: 매칭된 벡터들의 FAISS 내부 ID (0부터 시작)
        distances, indices = faiss_index.search(query_vector_np, top_k)

        results = []
        if indices.size > 0: # 검색 결과가 있을 경우
            for i, faiss_id in enumerate(indices[0]): # indices[0]이 실제 인덱스 ID들의 배열
                if 0 <= faiss_id < len(index_to_metadata_map): # 유효한 FAISS ID인지 확인
                    metadata = index_to_metadata_map[faiss_id]
                    results.append({
                        "document_id": metadata.get("doc_id"),
                        "chunk_id": metadata.get("chunk_id"),
                        "text_preview": metadata.get("text_preview"),
                        # L2 거리는 제곱값이므로, 실제 거리로 사용하려면 제곱근을 취하거나,
                        # 또는 코사인 유사도로 변환하는 것이 더 직관적일 수 있음 (별도 정규화 및 내적 필요)
                        # 여기서는 FAISS가 반환한 raw distance (L2 제곱 거리)를 사용
                        "distance_score": float(distances[0][i])
                    })
                else:
                    logger.warning(f"Search returned an invalid FAISS internal ID: {faiss_id}, which is out of bounds for metadata map (size: {len(index_to_metadata_map)}).")

        logger.info(f"FAISS search completed. Found {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return []

# `worker.py`의 `initialize_app`에서 `vector_store_adapter.load_or_initialize_faiss_index()`를 호출합니다.
# `worker.py`의 Graceful Shutdown 로직에서 `vector_store_adapter.save_faiss_index_and_metadata_to_s3()`를 호출합니다.
# 주기적인 백업 로직도 `worker.py`의 메인 루프나 별도 스레드에서 구현할 수 있습니다.