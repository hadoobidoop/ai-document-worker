# analysis_worker_app/worker.py

import logging
import signal
import time
import json
import os
import boto3
from botocore.exceptions import ClientError

# 애플리케이션 모듈 임포트
from adapters import s3_reader, backend_api_adapter, vector_store_adapter
from analysis_lambda import config
from nlp_tasks import summarizer, categorizer, tag_extractor, embedding_generator

# --- 로깅 설정 ---
# LOG_LEVEL은 config.py에서도 설정할 수 있지만, 메인 애플리케이션에서 명시적으로 설정하는 것이 좋습니다.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# boto3와 같은 외부 라이브러리의 로그 레벨을 조정하여 너무 많은 로그 방지
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING) # sentence-transformers 로그 레벨 조정
logging.getLogger("PIL.Image").setLevel(logging.WARNING) # PIL 이미지 로드 경고 방지

logger = logging.getLogger(__name__) # 현재 모듈의 로거

# --- Boto3 SQS 클라이언트 및 전역 변수 ---
sqs_client = None
# SQS_QUEUE_URL은 config.py에서 로드합니다. (config.SQS_QUEUE_URL 사용)

# Graceful shutdown을 위한 플래그
shutdown_flag = False
# 주기적인 FAISS 인덱스 저장을 위한 카운터 및 임계값 (예시)
FAISS_SAVE_INTERVAL_MESSAGES = int(os.environ.get("FAISS_SAVE_INTERVAL_MESSAGES", "10")) # 10개 메시지 처리마다 저장
messages_processed_since_last_faiss_save = 0

# --- 애플리케이션 초기화 ---
def initialize_app():
    """애플리케이션 시작 시 필요한 모든 컴포넌트를 초기화합니다."""
    global sqs_client
    logger.info("=====================================================================")
    logger.info("               AI Analysis Worker Application Starting               ")
    logger.info("=====================================================================")

    # config 모듈은 임포트 시점에 이미 로그를 남기므로 여기서는 상태만 확인
    logger.info(f"SQS Queue URL from config: {config.SQS_QUEUE_URL}")
    if not config.SQS_QUEUE_URL:
        logger.critical("필수 환경 변수 누락: SQS_QUEUE_URL. 워커를 시작할 수 없습니다.")
        raise EnvironmentError("SQS_QUEUE_URL is required for the worker to run.")

    sqs_client = boto3.client('sqs') # 리전은 실행 환경(예: ECS Fargate)에서 자동 설정
    logger.info("SQS client initialized.")

    logger.info("Initializing AI modules and FAISS vector store...")
    categorizer.initialize_categorizer()
    embedding_generator.initialize_embedding_model() # 이 함수가 embedding_generator.embedding_model_instance를 설정
    tag_extractor.initialize_tag_extractor_components(
        use_konlpy_okt=True # True로 설정하여 Okt 사용 시도 (konlpy 설치 필요)
    )
    # summarizer는 별도 초기화 함수가 현재 없음 (필요시 추가)
    vector_store_adapter.load_or_initialize_faiss_index()

    logger.info("---------------------------------------------------------------------")
    logger.info("               Application Initialization Complete                   ")
    logger.info("---------------------------------------------------------------------")

# --- SQS 메시지 처리 ---
def process_message(message: dict) -> bool:
    """단일 SQS 메시지를 처리하고, 모든 AI 분석 파이프라인을 실행합니다."""
    global shutdown_flag, messages_processed_since_last_faiss_save
    if shutdown_flag:
        logger.info("Shutdown signal received. Halting new message processing.")
        return False

    receipt_handle = message.get('ReceiptHandle')
    message_id = message.get('MessageId', 'N/A')
    logger.info(f"Received message (ID: {message_id}). Starting processing pipeline.")

    document_id_for_log = "N/A"
    processing_status_for_backend = "PROCESSING_STARTED"
    error_messages_for_backend = [] # 여러 단계에서 오류 발생 시 누적

    try:
        body_str = message.get('Body', '{}')
        message_body = json.loads(body_str)

        s3_path = message_body.get('s3_path')
        ingestion_details = message_body.get('ingestion_details', {})
        document_id = ingestion_details.get('document_id')
        user_id = ingestion_details.get('user_id')
        original_url = ingestion_details.get('source_identifier') if ingestion_details.get('source_type') == 'url' else None
        document_id_for_log = document_id

        if not s3_path or not document_id:
            logger.error(f"Message (ID: {message_id}) is missing s3_path or document_id. Body: {body_str}")
            if receipt_handle and sqs_client:
                sqs_client.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            return True # Invalid message, but loop continues

        logger.info(f"Processing document_id: {document_id}, S3 Path: {s3_path}, User ID: {user_id}")

        # 1. S3 텍스트 로드
        text_content = s3_reader.get_text_from_s3(s3_path)
        if text_content is None:
            logger.error(f"Document ID {document_id}: Failed to load text from S3. Aborting processing for this message.")
            processing_status_for_backend = "TEXT_LOAD_FAILURE"
            error_messages_for_backend.append("Failed to load text from S3.")
            # 이 경우에도 백엔드에 상태 업데이트 시도
            backend_api_adapter.save_analysis_to_backend(
                document_id, user_id, s3_path, original_url,
                processing_status_for_backend, "; ".join(error_messages_for_backend)
            )
            if receipt_handle and sqs_client: # 처리 불가능한 오류로 간주하고 메시지 삭제
                sqs_client.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            return True

        logger.info(f"Document ID {document_id}: Text loaded from S3. Length: {len(text_content)} chars.")
        processing_status_for_backend = "TEXT_LOADED_SUCCESSFULLY" # 각 단계 성공 시 상태 업데이트

        # --- AI 분석 파이프라인 ---
        short_summary, long_markdown_summary, assigned_category, extracted_hashtags = None, None, None, None
        num_embedded_chunks = 0

        # 2. 요약 수행
        try:
            if config.GROQ_API_KEY: # API 키가 있을 때만 시도
                short_summary = summarizer.summarize_with_groq(text_content, summary_type="short")
                long_markdown_summary = summarizer.summarize_with_groq(text_content, summary_type="long_markdown")
                logger.info(f"DocID {document_id}: Summaries generated. Short: {len(short_summary or '')} chars, Long: {len(long_markdown_summary or '')} chars.")
            else:
                logger.warning(f"DocID {document_id}: Groq API key not configured. Skipping summarization.")
                error_messages_for_backend.append("Summarization skipped: API key missing.")
        except Exception as e_sum:
            logger.error(f"DocID {document_id}: Summarization step failed: {e_sum}", exc_info=True)
            error_messages_for_backend.append(f"Summarization failed: {str(e_sum)[:100]}")


        # 3. 카테고리 분류 수행
        try:
            assigned_category = categorizer.classify_text_tfidf(text_content)
            logger.info(f"DocID {document_id}: Classified as '{assigned_category}'.")
        except Exception as e_cat:
            logger.error(f"DocID {document_id}: Categorization step failed: {e_cat}", exc_info=True)
            error_messages_for_backend.append(f"Categorization failed: {str(e_cat)[:100]}")


        # 4. 해시태그 추출
        try:
            if embedding_generator.embedding_model_instance: # 임베딩 모델이 로드되어야 KeyBERT 사용 가능
                extracted_hashtags = tag_extractor.extract_hashtags_with_keybert(
                    text_content=text_content,
                    embedding_model=embedding_generator.embedding_model_instance,
                    num_tags=3,
                    use_korean_noun_extraction_if_available=True # konlpy 사용 여부
                    # language_hint=detected_language # 언어 감지 기능 추가 시 전달
                )
                logger.info(f"DocID {document_id}: Extracted hashtags: {extracted_hashtags}")
            else:
                logger.warning(f"DocID {document_id}: Embedding model for KeyBERT not available. Skipping hashtag extraction.")
                error_messages_for_backend.append("Hashtag extraction skipped: Embedding model not ready.")
        except Exception as e_tag:
            logger.error(f"DocID {document_id}: Hashtag extraction step failed: {e_tag}", exc_info=True)
            error_messages_for_backend.append(f"Hashtag extraction failed: {str(e_tag)[:100]}")


        # 5. 임베딩 생성 및 FAISS에 추가
        try:
            if vector_store_adapter.faiss_index is not None and embedding_generator.embedding_model_instance:
                embeddings_data = embedding_generator.generate_and_chunk_embeddings(text_content, document_id)
                if embeddings_data:
                    if vector_store_adapter.add_embeddings_to_faiss(embeddings_data): # 메모리 내 인덱스에만 추가
                        num_embedded_chunks = len(embeddings_data)
                        logger.info(f"DocID {document_id}: Embeddings ({num_embedded_chunks} chunks) added to in-memory FAISS index.")
                        messages_processed_since_last_faiss_save += 1 # FAISS 저장 카운터 증가
                    else: # add_embeddings_to_faiss가 False 반환 (메모리 내 추가 실패)
                        logger.error(f"DocID {document_id}: Failed to add embeddings to in-memory FAISS index.")
                        error_messages_for_backend.append("Failed to add embeddings to in-memory FAISS.")
                elif embeddings_data == []:
                    logger.info(f"DocID {document_id}: No embeddings generated as content was empty or too short.")
                else: # None 반환 (생성 오류)
                    logger.warning(f"DocID {document_id}: Embedding generation returned None.")
                    error_messages_for_backend.append("Embedding generation failed.")
            else:
                logger.warning(f"DocID {document_id}: FAISS index or embedding model not available. Skipping embedding.")
                error_messages_for_backend.append("Embedding skipped: FAISS index or model not ready.")
        except Exception as e_emb:
            logger.error(f"DocID {document_id}: Embedding/FAISS step failed: {e_emb}", exc_info=True)
            error_messages_for_backend.append(f"Embedding/FAISS step failed: {str(e_emb)[:100]}")

        # --- AI 분석 파이프라인 종료 ---

        # 최종 처리 상태 결정
        if error_messages_for_backend:
            processing_status_for_backend = "PARTIAL_FAILURE" if processing_status_for_backend != "TEXT_LOAD_FAILURE" else processing_status_for_backend
        else:
            processing_status_for_backend = "COMPLETED"

        final_error_message = "; ".join(error_messages_for_backend) if error_messages_for_backend else None

        # 6. 모든 분석 결과 DB에 저장
        save_success = backend_api_adapter.save_analysis_to_backend(
            document_id=document_id,
            user_id=user_id,
            s3_path=s3_path,
            original_url=original_url,
            short_summary=short_summary,
            long_markdown_summary=long_markdown_summary,
            category=assigned_category,
            hashtags=extracted_hashtags,
            num_embedded_chunks=num_embedded_chunks,
            analysis_status=processing_status_for_backend,
            failure_reason=final_error_message
        )

        if not save_success:
            logger.error(f"CRITICAL: Failed to save final analysis results to backend for document_id: {document_id}. This data might be lost if not retried.")
            # 이 경우, SQS 메시지를 삭제하지 않고 재처리되도록 하거나 DLQ로 보내는 것이 중요.
            # 현재는 로깅만 하고 아래에서 메시지 삭제 (정책 결정 필요)
            return False # 메시지 삭제 방지를 위해 False 반환하여 재처리 유도

        logger.info(f"Message (ID: {message_id}, DocID: {document_id}) processed with status: {processing_status_for_backend}.")

        # 성공적으로 모든 단계가 완료되었거나, 재처리 불가능한 오류로 판단되면 메시지 삭제
        if receipt_handle and sqs_client:
            sqs_client.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            logger.info(f"Message (ID: {message_id}, DocID: {document_id}) deleted from SQS.")

        return True # 성공

    except json.JSONDecodeError as e:
        logger.error(f"Message (ID: {message_id}) body JSON parsing failed: {e}. Body: {message.get('Body', '')}", exc_info=True)
        if receipt_handle and sqs_client: # 잘못된 형식은 재처리해도 소용 없으므로 삭제
            sqs_client.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
        return True
    except Exception as e: # process_message 함수의 최상위 예외 처리
        logger.error(f"CRITICAL unhandled error in process_message for DocID: {document_id_for_log}, Message ID: {message_id}. Error: {e}", exc_info=True)
        # 이 메시지는 재처리될 가능성이 높음 (삭제 안 했으므로)
        return False # 처리 실패


# --- Graceful Shutdown 처리 ---
def signal_handler(signum, frame):
    global shutdown_flag
    if not shutdown_flag: # 중복 호출 방지
        logger.info(f"Shutdown signal ({signal.Signals(signum).name}) received. Initiating graceful shutdown...")
        shutdown_flag = True
    else:
        logger.info("Shutdown already in progress.")

# --- 메인 SQS 폴링 루프 ---
def main_loop():
    global shutdown_flag, messages_processed_since_last_faiss_save
    if not sqs_client: # SQS 클라이언트 초기화 실패 시 실행 불가
        logger.critical("SQS client not initialized. Worker cannot start. Check SQS_QUEUE_URL.")
        return

    logger.info(f"Starting SQS polling loop for queue: {config.SQS_QUEUE_URL}")
    while not shutdown_flag:
        try:
            response = sqs_client.receive_message(
                QueueUrl=config.SQS_QUEUE_URL,
                MaxNumberOfMessages=1, # 한 번에 하나의 메시지만 처리하여 개별 오류 영향 최소화 (조정 가능)
                WaitTimeSeconds=10,    # Long Polling
                VisibilityTimeout=300  # 처리 시간에 맞춰 가시성 제한 시간 설정 (예: 5분)
                # Lambda 최대 시간 15분, 서버는 이보다 길게 설정 가능
            )

            messages = response.get('Messages', [])
            if not messages:
                # logger.debug("No messages received from SQS. Continuing to poll.")
                # time.sleep(1) # Long Polling 사용 시 메시지 없을 때 CPU 점유 방지 (WaitTimeSeconds가 이 역할)
                continue

            # 현재는 한 번에 하나의 메시지만 가져오도록 설정 (MaxNumberOfMessages=1)
            if process_message(messages[0]): # True 반환 시 성공 또는 복구 불가능 오류로 메시지 삭제됨
                if messages_processed_since_last_faiss_save >= FAISS_SAVE_INTERVAL_MESSAGES:
                    logger.info(f"Processed {messages_processed_since_last_faiss_save} messages. Saving FAISS index to S3...")
                    if vector_store_adapter.save_faiss_index_and_metadata_to_s3():
                        messages_processed_since_last_faiss_save = 0 # 카운터 초기화
                    else:
                        logger.error("Failed to save FAISS index to S3 during periodic save.")
                        # 이 경우, 다음 저장 주기 또는 종료 시 다시 시도. 심각한 문제 시 알림 필요.
            else: # process_message가 False 반환 시 (재처리 가능한 오류)
                logger.warning("Message processing failed, message will likely be re-processed by SQS after visibility timeout.")
                time.sleep(5) # 짧은 대기 후 다음 폴링 시도

        except ClientError as e:
            logger.error(f"SQS receive_message ClientError: {e}", exc_info=True)
            time.sleep(10)
        except Exception as e:
            logger.error(f"Unexpected error in SQS polling loop: {e}", exc_info=True)
            time.sleep(10)

    logger.info("SQS polling loop has finished due to shutdown signal.")

# --- 애플리케이션 실행 ---
if __name__ == "__main__":
    # 종료 신호(SIGINT: Ctrl+C, SIGTERM: ECS 태스크 중지 등) 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    final_exit_code = 0
    try:
        initialize_app() # 애플리케이션 및 AI 모듈 초기화
        if config.SQS_QUEUE_URL and sqs_client : # SQS 설정이 올바를 때만 메인 루프 실행
            main_loop()      # SQS 메시지 처리 루프 시작
        else:
            logger.critical("SQS_QUEUE_URL not configured or SQS client failed to initialize. Worker cannot poll SQS.")
            final_exit_code = 1 # 오류 상태로 종료

    except Exception as e: # 최상위 예외 처리 (예: initialize_app 실패)
        logger.critical(f"Unhandled exception at worker's top level: {e}", exc_info=True)
        final_exit_code = 1
    finally:
        # 애플리케이션 종료 전 반드시 수행해야 할 정리 작업
        logger.info("Worker application is shutting down. Performing final cleanup...")
        if vector_store_adapter.faiss_index is not None:
            logger.info("Attempting to save FAISS index and metadata to S3 before final exit...")
            if vector_store_adapter.save_faiss_index_and_metadata_to_s3():
                logger.info("FAISS index and metadata successfully saved to S3 on shutdown.")
            else:
                logger.error("CRITICAL: Failed to save FAISS index and metadata to S3 during shutdown.")
                final_exit_code = 1 # 저장 실패 시 오류 코드 반환
        else:
            logger.info("No FAISS index loaded or initialized, skipping save on shutdown.")

        logger.info(f"Worker application shutdown complete. Exiting with code {final_exit_code}.")
        # sys.exit(final_exit_code) # 컨테이너 환경에서는 명시적 exit 필요 없을 수 있음