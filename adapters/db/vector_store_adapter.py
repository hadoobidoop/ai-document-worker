# analysis_worker_app/adapters/vector_store_adapter.py
import logging
from pathlib import Path # pathlib.Path 사용
from typing import Any, Dict, List, Optional, Sequence

import boto3 # boto3는 get_s3_client 내에서 호출되므로 여기서는 직접 사용하지 않아도 될 수 있음
from botocore.exceptions import ClientError
import faiss
import numpy as np
import json

import config # config 모듈 임포트

logger = logging.getLogger(__name__)

# --- Dependency injection for easier testing ---
def get_s3_client():
    """
    Returns an S3 client. In tests, config.MOCK_S3 can be set to a fake client.
    config.MOCK_S3가 정의되어 있지 않다면 실제 boto3.client를 반환합니다.
    """
    # config 모듈에 MOCK_S3 속성이 있고, 그 값이 None이 아닌 경우 해당 값을 사용
    if hasattr(config, "MOCK_S3") and config.MOCK_S3 is not None:
        logger.info("Using MOCK_S3 client from config.")
        return config.MOCK_S3
    logger.info("Initializing new boto3 S3 client.")
    return boto3.client("s3")

# 모듈 레벨 전역 변수
s3_client = get_s3_client() # S3 클라이언트 인스턴스
index_to_metadata_map: List[Dict[str, Any]] = [] # 메타데이터 맵
faiss_index: Optional[faiss.Index] = None # FAISS 인덱스 객체

# config.TMP_DIR이 settings.py에 정의되어 있어야 합니다. 예: TMP_DIR = "/tmp"
# 또는 os.environ.get("TMP_DIR", "/tmp")


def is_initialized() -> bool:
    """
    Returns True if the FAISS index and metadata are loaded and consistent.
    FAISS 인덱스가 존재하고, 메타데이터 맵의 길이와 인덱스의 총 벡터 수가 일치하는지 확인합니다.
    """
    return (
            isinstance(faiss_index, faiss.Index)
            and len(index_to_metadata_map) == faiss_index.ntotal
    )


def _get_s3_key(filename: str) -> str:
    """S3 객체 키를 생성합니다."""
    prefix = config.FAISS_INDEX_S3_PREFIX.rstrip('/')
    return f"{prefix}/{filename}"


def _get_local_path(filename: str) -> Path:
    """로컬 임시 파일 경로를 생성합니다. config.TMP_DIR이 정의되어 있어야 합니다."""
    # TMP_DIR이 config에 정의되어 있는지 확인하거나 기본값을 사용합니다.
    tmp_dir_path = getattr(config, "TMP_DIR", "/tmp") # config에 TMP_DIR이 없으면 /tmp 사용
    return Path(tmp_dir_path) / filename


def _download_from_s3(bucket: str, key: str, local_path: Path) -> bool:
    """
    S3에서 파일을 로컬 경로로 다운로드합니다. 성공 시 True를 반환합니다.
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True) # 필요시 부모 디렉토리 생성
        with local_path.open('wb') as f:
            s3_client.download_fileobj(bucket, key, f)
        logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
        return True
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code in ('404', 'NoSuchKey'): # 파일이 없는 경우 경고 로깅
            logger.warning(f"S3 key not found: s3://{bucket}/{key} (code {code})")
        else: # 기타 S3 오류
            logger.error(f"Error downloading s3://{bucket}/{key}: {e}")
        return False
    except Exception as e: # 예상치 못한 오류
        logger.error(f"Unexpected error downloading s3://{bucket}/{key}: {e}")
        return False


def _load_index(local_path: Path) -> Optional[faiss.Index]:
    """
    디스크에서 FAISS 인덱스를 로드합니다. mmap 시도 후 실패 시 일반 로드로 대체합니다.
    """
    try:
        idx = faiss.read_index(str(local_path), faiss.IO_FLAG_MMAP) # mmap으로 로드 시도
        logger.info(f"Loaded FAISS index via mmap from {local_path} ({idx.ntotal} vectors)")
        return idx
    except RuntimeError as e_mmap: # mmap 실패 시
        logger.warning(f"Mmap load from {local_path} failed ({e_mmap}), retrying regular read.")
    except Exception as e: # 기타 mmap 로드 오류
        logger.error(f"Failed mmap load from {local_path}: {e}")

    try: # 일반 로드 시도
        idx = faiss.read_index(str(local_path))
        logger.info(f"Loaded FAISS index via regular read from {local_path} ({idx.ntotal} vectors)")
        return idx
    except Exception as e: # 일반 로드도 실패 시
        logger.error(f"Regular load from {local_path} failed: {e}")
        return None


def _load_metadata(local_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    디스크에서 메타데이터 JSON 파일을 로드합니다.
    """
    try:
        with local_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded metadata from {local_path} ({len(data)} entries)")
        return data
    except Exception as e:
        logger.error(f"Failed to load metadata from {local_path}: {e}")
        return None


def _validate_counts(index: faiss.Index, metadata: List[Dict[str, Any]]) -> bool:
    """
    인덱스와 메타데이터의 길이가 일치하는지 확인합니다.
    """
    if index.ntotal != len(metadata):
        logger.critical(
            f"CRITICAL MISMATCH: Index count {index.ntotal} != metadata count {len(metadata)}"
        )
        return False
    return True


def load_or_initialize_faiss_index() -> None:
    """
    S3에서 FAISS 인덱스와 메타데이터를 로드하거나, 없다면 새로 초기화합니다.
    이 함수는 애플리케이션 시작 시 호출되어야 합니다.
    """
    global faiss_index, index_to_metadata_map

    if is_initialized():
        logger.info("FAISS index and metadata are already initialized and consistent.")
        return

    bucket = config.FAISS_INDEX_S3_BUCKET
    if not bucket:
        logger.error("FAISS_INDEX_S3_BUCKET is not configured. Cannot proceed.")
        # 이전 버전에서는 여기서 return 했지만, 중요한 설정이므로 에러 발생시키는 것이 더 적절할 수 있음
        raise RuntimeError("FAISS_INDEX_S3_BUCKET is not configured.")

    index_key = _get_s3_key(config.FAISS_INDEX_FILE_NAME)
    meta_key = _get_s3_key(config.FAISS_METADATA_FILE_NAME)
    local_index_path = _get_local_path(config.FAISS_INDEX_FILE_NAME)
    local_meta_path = _get_local_path(config.FAISS_METADATA_FILE_NAME)

    idx_loaded_successfully = False
    meta_loaded_successfully = False

    if _download_from_s3(bucket, index_key, local_index_path):
        loaded_idx_obj = _load_index(local_index_path)
        if loaded_idx_obj:
            faiss_index = loaded_idx_obj
            idx_loaded_successfully = True

    if _download_from_s3(bucket, meta_key, local_meta_path):
        loaded_meta_obj = _load_metadata(local_meta_path)
        if loaded_meta_obj is not None:
            index_to_metadata_map = loaded_meta_obj
            meta_loaded_successfully = True

    for path in (local_index_path, local_meta_path):
        try:
            path.unlink(missing_ok=True)
        except Exception as e_unlink:
            logger.warning(f"Could not remove temporary file {path}: {e_unlink}")

    if not idx_loaded_successfully:
        logger.info(f"Initializing new empty FAISS IndexFlatL2 (dimension={config.EMBEDDING_DIMENSION}) as index was not loaded.")
        faiss_index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
        if meta_loaded_successfully and len(index_to_metadata_map) > 0: # 새 인덱스 생성 시 메타데이터도 비워야 함
            logger.warning("New FAISS index created, but old metadata was loaded. Clearing metadata for consistency.")
            index_to_metadata_map = []
            meta_loaded_successfully = False # 메타데이터도 새로 시작하는 것으로 간주 (아래 if not meta_loaded_successfully에서 처리됨)


    if not meta_loaded_successfully: # 인덱스는 로드/생성되었지만 메타데이터가 로드 안된 경우
        logger.info("Initializing new empty metadata map as it was not loaded or cleared.")
        index_to_metadata_map = []

    if not is_initialized(): # 최종 유효성 검사
        # 이 지점에 도달했다면, faiss_index가 None이거나 (매우 예외적), 개수가 맞지 않는 경우
        current_idx_total = faiss_index.ntotal if faiss_index else "None"
        logger.error(
            f"FAISS initialization failed or inconsistent. Index vectors: {current_idx_total}, Metadata entries: {len(index_to_metadata_map)}. Vector store may not function correctly."
        )
        # 애플리케이션의 안정성을 위해 중요한 초기화 실패 시 에러 발생
        raise RuntimeError(f"Failed to initialize FAISS index and metadata consistently. Index: {current_idx_total}, Meta: {len(index_to_metadata_map)}")
    else:
        logger.info(
            f"FAISS ready. Index vectors: {faiss_index.ntotal}, Metadata entries: {len(index_to_metadata_map)}"
        )


def save_faiss_index_and_metadata_to_s3() -> bool | None:
    """
    FAISS 인덱스와 메타데이터를 S3에 저장합니다.
    """
    if not is_initialized():
        logger.error("FAISS index is not properly initialized or inconsistent; cannot save.")
        return False

    bucket = config.FAISS_INDEX_S3_BUCKET
    if not bucket:
        logger.error("FAISS_INDEX_S3_BUCKET S3 bucket not configured; cannot save.")
        return False

    index_key = _get_s3_key(config.FAISS_INDEX_FILE_NAME)
    meta_key = _get_s3_key(config.FAISS_METADATA_FILE_NAME)
    local_index_path = _get_local_path(config.FAISS_INDEX_FILE_NAME)
    local_meta_path = _get_local_path(config.FAISS_METADATA_FILE_NAME)

    try:
        local_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(local_index_path))
        logger.debug(f"FAISS index saved locally to {local_index_path}")

        with local_meta_path.open('w', encoding='utf-8') as f:
            json.dump(index_to_metadata_map, f, ensure_ascii=False, indent=2)
        logger.debug(f"FAISS metadata saved locally to {local_meta_path}")

        s3_client.upload_file(str(local_index_path), bucket, index_key)
        logger.info(f"FAISS index uploaded to s3://{bucket}/{index_key}")
        s3_client.upload_file(str(local_meta_path), bucket, meta_key)
        logger.info(f"FAISS metadata uploaded to s3://{bucket}/{meta_key}")

        logger.info("FAISS index and metadata successfully saved to S3.")
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS data to S3: {e}", exc_info=True)
        return False
    finally:
        for path in (local_index_path, local_meta_path):
            try:
                path.unlink(missing_ok=True)
            except Exception as e_unlink:
                logger.warning(f"Could not remove temporary file {path} during save: {e_unlink}")


def add_embeddings_to_faiss(embeddings_data: Sequence[Dict[str, Any]]) -> bool:
    """
    새로운 임베딩 (메타데이터 포함)을 인메모리 FAISS 인덱스에 추가합니다.
    """
    global faiss_index, index_to_metadata_map

    if not faiss_index: # FAISS 인덱스 객체 존재 여부 우선 확인
        logger.error("FAISS index object does not exist; cannot add embeddings. Initialize first.")
        return False
    # is_initialized()를 사용하여 이전 상태가 일관되었는지 확인하는 것도 좋음
    # if not is_initialized():
    #     logger.warning("FAISS store was not in a consistent state before adding. Proceeding with caution.")


    if not embeddings_data:
        logger.info("No embeddings data provided to add.")
        return True

    vectors_to_add = []
    metadata_to_add: List[Dict[str, Any]] = []

    for item in embeddings_data:
        vector = item.get('embedding_vector')
        if vector and len(vector) == config.EMBEDDING_DIMENSION:
            vectors_to_add.append(vector)
            metadata_to_add.append({
                'doc_id': str(item.get('document_id', 'N/A')),
                'chunk_id': int(item.get('chunk_id', -1)),
                'text_preview': str(item.get('chunk_text', ''))[:250] + "..."
            })
        else:
            logger.warning(
                f"Skipping invalid or missing embedding vector for doc_id {item.get('document_id')}. "
                f"Expected dimension {config.EMBEDDING_DIMENSION}, got {len(vector) if vector else 'None'}."
            )

    if not vectors_to_add:
        logger.info("No valid vectors to add after filtering.")
        return True

    try:
        np_vectors = np.array(vectors_to_add, dtype=np.float32)
        faiss_index.add(np_vectors)
        index_to_metadata_map.extend(metadata_to_add)

        if not _validate_counts(faiss_index, index_to_metadata_map): # 추가 후 일관성 검증
            logger.error("CRITICAL: Inconsistency after adding embeddings. Manual check needed.")
            # 여기서 롤백은 복잡하므로, 일단 실패로 처리하고 상태를 남김
            return False

        logger.info(
            f"Successfully added {len(vectors_to_add)} vectors. "
            f"Total vectors in FAISS index: {faiss_index.ntotal}."
        )
        return True
    except Exception as e:
        logger.error(f"Error adding embeddings to FAISS index: {e}", exc_info=True)
        return False


def search_in_faiss(
        query_embedding: Sequence[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    주어진 쿼리 임베딩에 대해 상위 K개의 가장 유사한 이웃을 검색합니다.
    """
    if not is_initialized() or (faiss_index and faiss_index.ntotal == 0): # is_initialized() 및 인덱스 비어있는지 확인
        logger.warning("FAISS index not initialized, empty, or inconsistent. Cannot perform search.")
        return []

    if not query_embedding or len(query_embedding) != config.EMBEDDING_DIMENSION:
        logger.error(
            f"Invalid query embedding for FAISS search. Expected dimension {config.EMBEDDING_DIMENSION}, "
            f"got {len(query_embedding) if query_embedding else 'None'}."
        )
        return []

    logger.info(f"Searching for top {top_k} similar items in FAISS index (total items: {faiss_index.ntotal if faiss_index else 0}).")
    try:
        query_vector_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_index.search(query_vector_np, top_k)

        results: List[Dict[str, Any]] = []
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

