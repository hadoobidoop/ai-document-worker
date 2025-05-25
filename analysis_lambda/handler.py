# analysis_lambda/handler.py (최종 통합)

import json
import logging

from ingestion_lambda.adapters import backend_api_adapter
from . import config
from . import summarizer
# 모듈 임포트
from .adapters import s3_reader, vector_store_adapter  # vector_store_adapter 추가
from .nlp_tasks import categorizer, embedding_generator, tag_extractor

# from langchain.text_splitter import RecursiveCharacterTextSplitter # 핸들러에서는 직접 사용 안 함 (embedding_generator 내부 사용)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Lambda 콜드 스타트 시 모듈 초기화 ---
logger.info("Executing initializations for analysis_lambda (cold start)...")
if config.GROQ_API_KEY:
    logger.info("Groq API Key IS configured.")
else:
    logger.warning("Groq API Key is NOT configured. Summarization will be skipped or fail.")
logger.info(f"Spring Boot API Endpoint configured: {config.SPRING_BOOT_API_ENDPOINT}")

categorizer.initialize_categorizer()
embedding_generator.initialize_embedding_model()
vector_store_adapter.load_or_initialize_faiss_index()  # FAISS 인덱스 로드/초기화
logger.info("All initializations complete.")


# --- 초기화 끝 ---


# save_analysis_results 함수는 DB에 요약, 카테고리 등 메타데이터 저장 역할
# (이전 응답의 save_analysis_results 함수 예시 사용 또는 수정)
def save_analysis_results(document_id: str, user_id: str | None,
                          short_summary: str | None, long_summary: str | None,
                          category: str | None, num_embedded_chunks: int):
    if not config.SPRING_BOOT_API_ENDPOINT:
        logger.error("Spring Boot API endpoint not configured. Cannot save analysis metadata.")
        return

    payload = {
        "documentId": document_id,
        "userId": user_id,
        "shortSummary": short_summary,
        "longSummary": long_summary,
        "category": category,
        "analysisStatus": "COMPLETED",  # 분석 완료 상태
        "numEmbeddedChunks": num_embedded_chunks,  # 임베딩된 청크 수
        # 필요한 다른 메타데이터 추가
    }
    logger.info(
        f"Attempting to save analysis results to Spring Boot API for document_id: {document_id}. Payload: {json.dumps(payload)}")
    try:
        # import requests # Lambda Layer에 포함 필요
        # response = requests.post(
        #     f"{config.SPRING_BOOT_API_ENDPOINT}/api/ai/analysis-results", # API 엔드포인트 예시
        #     json=payload,
        #     headers={'Content-Type': 'application/json'}
        # )
        # response.raise_for_status()
        # logger.info(f"Successfully saved analysis results via API for document_id: {document_id}")
        logger.info(
            f"Placeholder: Successfully 'sent' analysis results for document_id: {document_id} to Spring Boot API.")
    except Exception as e:  # requests.exceptions.RequestException
        logger.error(f"Failed to save analysis results via API for document_id {document_id}: {e}", exc_info=True)

def lambda_handler(event, context):
    logger.info(f"Received SQS event (showing first 1000 chars): {json.dumps(event)[:1000]}")

    for record in event.get('Records', []):
        document_id_for_log = "N/A"
        processing_status_for_backend = "COMPLETED" # 기본 상태
        error_message_for_backend = None

        try:
            message_body = json.loads(record.get('body', '{}'))
            s3_path = message_body.get('s3_path')
            ingestion_details = message_body.get('ingestion_details', {})

            document_id = ingestion_details.get('document_id')
            user_id = ingestion_details.get('user_id')
            original_url = ingestion_details.get('source_identifier') if ingestion_details.get('source_type') == 'url' else None
            document_id_for_log = document_id

            if not s3_path or not document_id:
                logger.error(f"Critical: Missing s3_path or document_id in SQS message: {message_body}")
                # 이 메시지는 처리 불가, DLQ로 보내거나 그냥 넘어갈 수 있음 (현재는 continue)
                continue

            logger.info(f"Processing document_id: {document_id} from S3: {s3_path} for user_id: {user_id}")

            # 1. S3 텍스트 로드
            text_content = s3_reader.get_text_from_s3(s3_path)
            if text_content is None:
                logger.error(f"Failed to load text for document_id: {document_id}. Marking as failure.")
                processing_status_for_backend = "TEXT_LOAD_FAILURE"
                error_message_for_backend = "Failed to load text from S3."
                # 실패 시에도 백엔드에 상태 업데이트
                backend_api_adapter.save_analysis_to_backend(
                    document_id, user_id, s3_path, original_url, None, None, None, None, 0,
                    processing_status_for_backend, error_message_for_backend
                )
                continue # 다음 레코드로

            logger.info(f"Successfully loaded text for document_id: {document_id}. Length: {len(text_content)} chars.")

            # 2. 요약 수행 (오류 발생 가능성 고려)
            short_summary, long_markdown_summary = None, None
            try:
                short_summary = summarizer.summarize_with_groq(text_content, summary_type="short")
                long_markdown_summary = summarizer.summarize_with_groq(text_content, summary_type="long_markdown")
                logger.info(f"DocID {document_id}: Summaries generated.")
            except Exception as e_sum: # 개별 단계 실패 시 로깅하고 진행 (부분 성공 처리)
                logger.error(f"Summarization step failed for document_id {document_id}: {e_sum}", exc_info=True)
                processing_status_for_backend = "PARTIAL_FAILURE"
                error_message_for_backend = (error_message_for_backend + "; " if error_message_for_backend else "") + f"Summarization failed: {str(e_sum)[:100]}"


            # 3. 카테고리 분류 수행
            assigned_category = None
            try:
                assigned_category = categorizer.classify_text_tfidf(text_content)
                logger.info(f"DocID {document_id}: Classified as '{assigned_category}'.")
            except Exception as e_cat:
                logger.error(f"Categorization step failed for document_id {document_id}: {e_cat}", exc_info=True)
                processing_status_for_backend = "PARTIAL_FAILURE"
                error_message_for_backend = (error_message_for_backend + "; " if error_message_for_backend else "") + f"Categorization failed: {str(e_cat)[:100]}"


            # 4. 해시태그 추출
            extracted_hashtags = None
            try:
                if embedding_generator.embedding_model_instance: # 임베딩 모델이 로드되었는지 확인
                    extracted_hashtags = tag_extractor.extract_hashtags_with_keybert(
                        text_content, embedding_generator.embedding_model_instance, num_tags=3
                    )
                    logger.info(f"DocID {document_id}: Extracted hashtags: {extracted_hashtags}")
                else:
                    logger.warning(f"DocID {document_id}: Embedding model for KeyBERT not available. Skipping hashtag extraction.")
            except Exception as e_tag:
                logger.error(f"Hashtag extraction step failed for document_id {document_id}: {e_tag}", exc_info=True)
                processing_status_for_backend = "PARTIAL_FAILURE"
                error_message_for_backend = (error_message_for_backend + "; " if error_message_for_backend else "") + f"Hashtag extraction failed: {str(e_tag)[:100]}"


            # 5. 임베딩 생성 및 FAISS에 추가
            num_embedded_chunks = 0
            try:
                if vector_store_adapter.faiss_index is not None and embedding_generator.embedding_model_instance:
                    embeddings_data = embedding_generator.generate_and_chunk_embeddings(text_content, document_id)
                    if embeddings_data:
                        if vector_store_adapter.add_embeddings_to_faiss(embeddings_data):
                            num_embedded_chunks = len(embeddings_data)
                            logger.info(f"DocID {document_id}: Embeddings added to FAISS ({num_embedded_chunks} chunks).")
                        else: # add_embeddings_to_faiss가 False 반환 (저장 실패)
                            processing_status_for_backend = "PARTIAL_FAILURE"
                            error_message_for_backend = (error_message_for_backend + "; " if error_message_for_backend else "") + "Failed to save embeddings to FAISS/S3."
                            logger.error(f"DocID {document_id}: Failed to add embeddings to FAISS or save index to S3.")
                    # embeddings_data가 비어있거나 None인 경우는 이미 embedding_generator에서 로깅하므로 여기선 추가 처리 불필요
                else:
                    logger.warning(f"DocID {document_id}: FAISS index or embedding model not available. Skipping embedding.")
            except Exception as e_emb:
                logger.error(f"Embedding/FAISS step failed for document_id {document_id}: {e_emb}", exc_info=True)
                processing_status_for_backend = "PARTIAL_FAILURE"
                error_message_for_backend = (error_message_for_backend + "; " if error_message_for_backend else "") + f"Embedding/FAISS failed: {str(e_emb)[:100]}"


            # 6. 모든 분석 결과 DB에 저장 (어댑터 사용)
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
                processing_status=processing_status_for_backend,
                failure_reason=error_message_for_backend
            )

            if not save_success:
                logger.error(f"CRITICAL: Failed to save final analysis results to backend for document_id: {document_id}. This data might be lost if not retried.")
                # 이 경우, SQS 메시지가 DLQ로 가거나 재처리되도록 조치하는 것이 매우 중요합니다.
                # 예를 들어 여기서 예외를 발생시키면 Lambda 설정에 따라 SQS가 재시도할 수 있습니다.
                # raise Exception(f"Failed to save results to backend for {document_id}")

            logger.info(f"Successfully processed and sent analysis results for document_id: {document_id} with status: {processing_status_for_backend}")

        except Exception as e: # 핸들러의 최상위 예외 처리
            logger.error(f"CRITICAL unhandled error in handler for document_id: {document_id_for_log}. Error: {e}", exc_info=True)
            # 이 경우도 DLQ 처리 및 알림이 중요합니다.
            # 필요하다면 여기서도 backend_api_adapter를 호출하여 실패 상태를 기록할 수 있습니다.
            # backend_api_adapter.save_analysis_to_backend(
            #     document_id_for_log, "N/A", "N/A", None, None, None, None, None, 0,
            #     "TOTAL_FAILURE", f"Unhandled handler error: {str(e)[:200]}"
            # )
            # SQS 재시도를 유도하려면 여기서 예외를 다시 발생시킵니다.
            # raise e

    return {
        'statusCode': 200, # SQS 트리거 Lambda는 보통 성공 시 200 OK (또는 오류 발생 시 예외)
        'body': json.dumps('AI analysis batch processing attempt completed.')
    }