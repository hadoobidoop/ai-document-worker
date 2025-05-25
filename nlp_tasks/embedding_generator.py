# analysis_worker_app/nlp_tasks/embedding_generator.py
import logging
import numpy as np # sentence_transformers.encode()가 numpy array를 반환하므로 필요
from sentence_transformers import SentenceTransformer # Lambda Layer에 패키징 필요
from langchain.text_splitter import RecursiveCharacterTextSplitter # Lambda Layer에 패키징 필요

logger = logging.getLogger(__name__)

# --- 임베딩 모델 및 청킹 설정 ---
# 사용할 한국어 Sentence Transformer 모델 이름 또는 Layer 내 경로
# 예: "jhgan/ko-sroberta-multitask", "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
# config.py에서 이 값을 가져오도록 수정할 수도 있습니다.
EMBEDDING_MODEL_NAME_OR_PATH = "jhgan/ko-sroberta-multitask" # 예시 모델

# SentenceTransformer 모델 인스턴스 (워커 시작 시 초기화)
embedding_model_instance: SentenceTransformer | None = None

# Langchain Text Splitter 인스턴스 (임베딩용 청킹)
# 임베딩 모델의 입력 길이 제한(보통 512 토큰)과 검색 효율성을 고려하여 설정합니다.
# 1토큰 ~ 2.5자 가정 시, 512 토큰은 약 1280자.
# 여기서는 청크 크기를 1000자, 겹침을 100자로 설정합니다.
EMBEDDING_CHUNK_SIZE_CHARS = 1000
EMBEDDING_CHUNK_OVERLAP_CHARS = 100

# text_splitter는 모듈 로드 시 한 번만 초기화
embedding_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=EMBEDDING_CHUNK_SIZE_CHARS,
    chunk_overlap=EMBEDDING_CHUNK_OVERLAP_CHARS,
    length_function=len,
    # add_start_index=True, # 필요시 청크의 원본 내 시작 위치 정보 추가
)

def initialize_embedding_model():
    """
    Sentence Transformer 임베딩 모델을 초기화합니다.
    워커 애플리케이션 시작 시 (콜드 스타트 시) 한 번만 호출되어야 합니다.
    """
    global embedding_model_instance
    if embedding_model_instance is not None: # 이미 초기화 되었다면 중복 실행 방지
        logger.info("Embedding model is already initialized.")
        return

    logger.info(f"Initializing sentence embedding model: {EMBEDDING_MODEL_NAME_OR_PATH}...")
    try:
        # Lambda 환경은 보통 CPU를 사용하므로 device='cpu'를 명시해줄 수 있습니다.
        # 모델 파일이 Lambda Layer에 포함되어 있다면, 해당 경로를 EMBEDDING_MODEL_NAME_OR_PATH에 지정할 수 있습니다.
        # 예: EMBEDDING_MODEL_NAME_OR_PATH = "/opt/python/embedding_models/my_sbert_model"
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME_OR_PATH, device='cpu')
        logger.info("Sentence embedding model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize sentence embedding model ({EMBEDDING_MODEL_NAME_OR_PATH}): {e}", exc_info=True)
        embedding_model_instance = None # 실패 시 None으로 설정


def generate_and_chunk_embeddings(text_content: str, document_id: str) -> list[dict] | None:
    """
    주어진 텍스트를 청크로 나누고, 각 청크에 대한 임베딩을 생성합니다.

    Args:
        text_content: 임베딩을 생성할 전체 텍스트.
        document_id: 해당 텍스트의 문서 ID.

    Returns:
        각 청크의 정보(document_id, chunk_id, chunk_text, embedding_vector)를 담은
        딕셔너리 리스트. 오류 발생 또는 임베딩 모델 미초기화 시 None 반환.
        처리할 내용이 없는 경우 빈 리스트 반환.
    """
    if embedding_model_instance is None:
        logger.error("Embedding model is not initialized. Cannot generate embeddings. Make sure initialize_embedding_model() was called at startup.")
        return None # 모델 미초기화 시 처리 불가

    if not text_content or not text_content.strip():
        logger.info(f"Document ID {document_id}: Text content is empty. No embeddings to generate.")
        return [] # 빈 내용이면 빈 리스트 반환

    try:
        logger.info(f"Splitting text for document_id: {document_id} into chunks for embedding. Target chunk size: {EMBEDDING_CHUNK_SIZE_CHARS} chars.")
        # Langchain의 RecursiveCharacterTextSplitter 사용
        chunks = embedding_text_splitter.split_text(text_content)

        if not chunks:
            logger.info(f"Document ID {document_id}: No chunks generated from text. Length: {len(text_content)} chars.")
            return []

        logger.info(f"Document ID {document_id}: Split text into {len(chunks)} chunks. Now generating embeddings for them...")

        # SentenceTransformer.encode()는 문자열 리스트를 받아 각 문자열에 대한 임베딩 리스트 (numpy array)를 반환합니다.
        # show_progress_bar=False: 서버 환경에서는 프로그레스 바 불필요
        chunk_embeddings_np_array = embedding_model_instance.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

        if not isinstance(chunk_embeddings_np_array, np.ndarray) or chunk_embeddings_np_array.ndim != 2:
            logger.error(f"Document ID {document_id}: Expected a 2D numpy array of embeddings, but got {type(chunk_embeddings_np_array)} with ndim {chunk_embeddings_np_array.ndim if isinstance(chunk_embeddings_np_array, np.ndarray) else 'N/A'}")
            return None

        embeddings_data = []
        for i, chunk_text in enumerate(chunks):
            if i < chunk_embeddings_np_array.shape[0]: # Defensive check
                embedding_vector = chunk_embeddings_np_array[i].tolist() # numpy array to Python list for JSON serialization
                embeddings_data.append({
                    "document_id": str(document_id), # Ensure document_id is string
                    "chunk_id": i, # 0-based index for chunk within this document
                    "chunk_text": chunk_text,
                    "embedding_vector": embedding_vector
                })
            else:
                # 이 경우는 encode()가 입력 청크 수보다 적은 임베딩을 반환한 경우 (매우 드묾)
                logger.warning(f"Document ID {document_id}: Mismatch between number of chunks ({len(chunks)}) and generated embeddings ({chunk_embeddings_np_array.shape[0]}). Skipping chunk {i}.")

        logger.info(f"Successfully generated {len(embeddings_data)} embeddings for document_id: {document_id}.")
        return embeddings_data

    except Exception as e:
        logger.error(f"Error during embedding generation or chunking for document_id {document_id}: {e}", exc_info=True)
        return None

# `worker.py`의 `initialize_app` 함수 내에서 `embedding_generator.initialize_embedding_model()`를 호출하여
# 워커 애플리케이션 시작 시 임베딩 모델을 미리 로드해야 합니다.