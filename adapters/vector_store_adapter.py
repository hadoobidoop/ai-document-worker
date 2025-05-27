# analysis_worker_app/adapters/vector_store_adapter.py
import logging
import faiss # faiss-cpu
import numpy as np
import boto3
from botocore.exceptions import ClientError
import json
import os # Lambda/Fargate의 /tmp/ 경로 사용 위함

# analysis_lambda는 현재 프로젝트 구조에 없으므로, config를 직접 임포트합니다.
# from analysis_lambda import config # 이 줄은 제거하거나 주석 처리
import config # 최상위 config 패키지 사용 가정, 또는 from config import settings

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
EMBEDDING_DIMENSION = 768 # config.py 또는 settings.py에서 가져오는 것을 고려할 수 있습니다.

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

    if not config.FAISS_INDEX_S3_BUCKET: # settings.py 대신 config 모듈 직접 사용으로 변경 (config/__init__.py에 정의됨)
        logger.error("FAISS_INDEX_S3_BUCKET is not configured. Cannot load or initialize FAISS index.")
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

        # --- MMAP 적용 시작 ---
        try:
            logger.info(f"Attempting to read FAISS index from {local_tmp_index_path} using mmap.")
            faiss_index = faiss.read_index(local_tmp_index_path, faiss.IO_FLAG_MMAP) # IO_FLAG_MMAP 추가
            logger.info(f"FAISS index (mmap) loaded from S3. Index contains {faiss_index.ntotal} vectors.")
            index_loaded = True
        except RuntimeError as re_mmap: # faiss mmap 관련 특정 오류 처리
            logger.error(f"FAISS RuntimeError while reading index with mmap: {re_mmap}. Falling back to default read.", exc_info=True)
            # mmap 실패 시 일반 로드로 재시도
            try:
                faiss_index = faiss.read_index(local_tmp_index_path)
                logger.info(f"FAISS index (fallback regular load) loaded. Index contains {faiss_index.ntotal} vectors.")
                index_loaded = True
            except Exception as e_fallback: # 일반 로드도 실패하는 경우
                logger.error(f"FAISS fallback regular read also failed: {e_fallback}", exc_info=True)
                # index_loaded는 False로 유지, 이후 로직에서 처리
        except Exception as e_read: # mmap 외 다른 read_index 오류
            logger.error(f"Failed to read FAISS index (path: {local_tmp_index_path}): {e_read}", exc_info=True)
            # index_loaded는 False로 유지
        # --- MMAP 적용 끝 ---

        # 인덱스 로드 성공 여부 재확인 (mmap 또는 fallback 로드 결과)
        if not index_loaded:
            # 로깅은 위에서 이미 했으므로, 여기서는 빈 인덱스로 초기화하는 로직으로 넘어갈 수 있도록 조건 설정
            logger.warning("FAISS index could not be loaded. Will attempt to initialize a new one if metadata also fails or is missing.")


        # 2. S3에서 메타데이터 파일 다운로드 시도
        logger.info(f"Attempting to download FAISS metadata from s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_metadata_key} to {local_tmp_metadata_path}")
        s3_client.download_file(config.FAISS_INDEX_S3_BUCKET, s3_metadata_key, local_tmp_metadata_path)
        with open(local_tmp_metadata_path, 'r', encoding='utf-8') as f:
            index_to_metadata_map = json.load(f)
        logger.info(f"FAISS metadata loaded from S3. Contains {len(index_to_metadata_map)} entries.")
        metadata_loaded = True

        if index_loaded and faiss_index is not None and faiss_index.ntotal != len(index_to_metadata_map): # index_loaded 조건 추가
            logger.critical(
                f"CRITICAL: FAISS index vector count ({faiss_index.ntotal}) and "
                f"metadata count ({len(index_to_metadata_map)}) mismatch! "
                f"This indicates a corrupted or inconsistent state. Manual intervention may be required."
            )
            _faiss_store_initialized_successfully = False
            return

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        # 파일이 없는 경우는 정상적인 초기화 흐름으로 간주
        if error_code in ['404', 'NoSuchKey']: # 403(Access Denied)은 다른 문제로 간주
            logger.warning(f"FAISS index or metadata file not found on S3 (Error code: {error_code}). Path: s3://{config.FAISS_INDEX_S3_BUCKET}/{s3_index_key} or {s3_metadata_key}. Initializing a new empty index.")
            if not index_loaded: # 인덱스 다운로드/로드 실패 시 (NoSuchKey 포함) 새 인덱스 생성
                logger.info("Initializing new empty FAISS index (L2).")
                faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            if not metadata_loaded: # 메타데이터 다운로드 실패 시 (NoSuchKey 포함) 빈 리스트로 초기화
                logger.info("Initializing new empty FAISS metadata map.")
                index_to_metadata_map = []
        else: # '404', 'NoSuchKey' 외의 ClientError (예: AccessDenied)
            logger.error(f"S3 ClientError during FAISS index/metadata download: {e}", exc_info=False)
            _faiss_store_initialized_successfully = False
            return
    except FileNotFoundError: # 로컬 임시 파일 관련 (download_file 실패 후 read_index 시도 등)
        logger.warning(f"Local FAISS index/metadata file not found at /tmp/ after S3 download attempt. This might indicate issues with /tmp/ or S3 download. Initializing new index if not already done.")
        if not index_loaded:
            logger.info("Initializing new empty FAISS index (L2) due to FileNotFoundError.")
            faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        if not metadata_loaded:
            logger.info("Initializing new empty FAISS metadata map due to FileNotFoundError.")
            index_to_metadata_map = []
    except Exception as e: # faiss.read_index() 또는 json.load() 실패 등
        logger.error(f"Unexpected error loading or initializing FAISS index: {e}", exc_info=True)
        _faiss_store_initialized_successfully = False
        return
    finally:
        # 사용한 임시 파일 정리
        if os.path.exists(local_tmp_index_path):
            try:
                os.remove(local_tmp_index_path)
            except OSError as e_remove:
                logger.warning(f"Could not remove temporary index file {local_tmp_index_path}: {e_remove}")
        if os.path.exists(local_tmp_metadata_path):
            try:
                os.remove(local_tmp_metadata_path)
            except OSError as e_remove:
                logger.warning(f"Could not remove temporary metadata file {local_tmp_metadata_path}: {e_remove}")


    if faiss_index is not None: # 최종적으로 faiss_index 객체가 유효한지 확인
        _faiss_store_initialized_successfully = True
        logger.info("FAISS index and metadata are ready.")
        if index_to_metadata_map is None: # 만약 위 로직에서 metadata_loaded가 false였고 초기화도 안됐다면
            logger.warning("FAISS metadata map is None after initialization attempt. Initializing to empty list.")
            index_to_metadata_map = []
    else:
        logger.critical("Failed to load or initialize FAISS index after all attempts. FAISS index is None.")
        _faiss_store_initialized_successfully = False


def save_faiss_index_and_metadata_to_s3() -> bool:
    """현재 메모리 상의 FAISS 인덱스와 메타데이터를 S3에 저장(덮어쓰기)합니다."""
    global faiss_index, index_to_metadata_map # index_to_metadata_map은 이미 global이지만 명시
    if not _faiss_store_initialized_successfully or faiss_index is None:
        logger.error("FAISS index is not properly initialized. Cannot save to S3.")
        return False
    if not config.FAISS_INDEX_S3_BUCKET:
        logger.error("FAISS_INDEX_S3_BUCKET is not configured. Cannot save to S3.")
        return False
    # 메타데이터가 None인 경우 (이론적으로 발생하면 안되지만 방어 코드)
    if index_to_metadata_map is None:
        logger.error("FAISS metadata (index_to_metadata_map) is None. Cannot save to S3.")
        return False


    s3_index_key = _get_s3_index_key()
    s3_metadata_key = _get_s3_metadata_key()
    local_tmp_index_path = _get_local_tmp_index_path()
    local_tmp_metadata_path = _get_local_tmp_metadata_path()

    logger.info(f"Attempting to save FAISS index and metadata to S3.")
    try:
        # 1. 로컬 임시 파일에 인덱스 저장
        logger.debug(f"Saving FAISS index ({faiss_index.ntotal} vectors) to local path: {local_tmp_index_path}")
        faiss.write_index(faiss_index, local_tmp_index_path)

        # 2. 로컬 임시 파일에 메타데이터 저장
        logger.debug(f"Saving FAISS metadata ({len(index_to_metadata_map)} entries) to local path: {local_tmp_metadata_path}")
        with open(local_tmp_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(index_to_metadata_map, f, ensure_ascii=False, indent=2) # indent 추가로 가독성 향상

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
        if os.path.exists(local_tmp_index_path):
            try:
                os.remove(local_tmp_index_path)
            except OSError as e_remove:
                logger.warning(f"Could not remove temporary index file {local_tmp_index_path}: {e_remove}")
        if os.path.exists(local_tmp_metadata_path):
            try:
                os.remove(local_tmp_metadata_path)
            except OSError as e_remove:
                logger.warning(f"Could not remove temporary metadata file {local_tmp_metadata_path}: {e_remove}")


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

    # 메타데이터가 None인 경우 (이론적으로 발생하면 안되지만 방어 코드)
    if index_to_metadata_map is None:
        logger.error("FAISS metadata (index_to_metadata_map) is None. Cannot add embeddings.")
        return False


    if not embeddings_data:
        logger.info("No new embeddings data to add to FAISS.")
        return True

    logger.info(f"Adding {len(embeddings_data)} new embedding(s) to in-memory FAISS index.")

    vectors_to_add = []
    new_metadata_entries = []

    for item in embeddings_data:
        vector = item.get('embedding_vector')
        if vector and len(vector) == EMBEDDING_DIMENSION:
            vectors_to_add.append(vector)
            new_metadata_entries.append({
                "doc_id": str(item.get("document_id")),
                "chunk_id": int(item.get("chunk_id")),
                "text_preview": item.get("chunk_text", "")[:250] + "..."
            })
        else:
            logger.warning(
                f"Invalid or missing embedding vector for doc_id {item.get('document_id')}, "
                f"chunk_id {item.get('chunk_id')}. Expected dimension {EMBEDDING_DIMENSION}, got {len(vector) if vector else 'None'}. Skipping."
            )

    if not vectors_to_add:
        logger.info("No valid vectors to add to FAISS after filtering.")
        return True

    try:
        np_vectors_to_add = np.array(vectors_to_add).astype(np.float32)
        faiss_index.add(np_vectors_to_add)
        index_to_metadata_map.extend(new_metadata_entries)

        logger.info(f"Successfully added {len(vectors_to_add)} vectors to in-memory FAISS index. Total vectors in memory: {faiss_index.ntotal}")
        if faiss_index.ntotal != len(index_to_metadata_map):
            logger.critical(f"CRITICAL INCONSISTENCY after adding to FAISS: index count {faiss_index.ntotal}, metadata count {len(index_to_metadata_map)}")
            return False
        return True

    except Exception as e:
        logger.error(f"Error adding embeddings to in-memory FAISS index: {e}", exc_info=True)
        return False

def search_in_faiss(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    메모리 내 FAISS 인덱스에서 주어진 쿼리 임베딩과 유사한 상위 K개의 결과를 검색합니다.
    """
    global faiss_index, index_to_metadata_map
    if not _faiss_store_initialized_successfully or faiss_index is None or faiss_index.ntotal == 0:
        logger.warning("FAISS index not initialized or empty. Cannot perform search.")
        return []

    # 메타데이터가 None인 경우 (이론적으로 발생하면 안되지만 방어 코드)
    if index_to_metadata_map is None:
        logger.error("FAISS metadata (index_to_metadata_map) is None. Cannot perform search.")
        return []


    if not query_embedding or len(query_embedding) != EMBEDDING_DIMENSION:
        logger.error(f"Invalid query embedding for FAISS search. Expected dimension {EMBEDDING_DIMENSION}, got {len(query_embedding) if query_embedding else 'None'}.")
        return []

    logger.info(f"Searching for top {top_k} similar items in FAISS index (total items: {faiss_index.ntotal}).")
    try:
        query_vector_np = np.array([query_embedding]).astype(np.float32)
        distances, indices = faiss_index.search(query_vector_np, top_k)

        results = []
        if indices.size > 0:
            for i, faiss_id in enumerate(indices[0]):
                if 0 <= faiss_id < len(index_to_metadata_map):
                    metadata = index_to_metadata_map[faiss_id]
                    results.append({
                        "document_id": metadata.get("doc_id"),
                        "chunk_id": metadata.get("chunk_id"),
                        "text_preview": metadata.get("text_preview"),
                        "distance_score": float(distances[0][i])
                    })
                else:
                    logger.warning(f"Search returned an invalid FAISS internal ID: {faiss_id}, which is out of bounds for metadata map (size: {len(index_to_metadata_map)}).")

        logger.info(f"FAISS search completed. Found {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return []