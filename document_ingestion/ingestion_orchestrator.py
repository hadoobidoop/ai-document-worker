# ingestion_lambda/ingestion_orchestrator.py

import logging
import uuid

# 같은 ingestion_lambda 디렉토리 내 모듈들을 임포트
from . import text_cleaner
from .content_fetch_coordinator import coordinate_fetch_from_url # 새로 추가

# 어댑터 모듈 임포트
from .adapters import s3_storage
from .adapters import sqs_publisher

logger = logging.getLogger(__name__) # __name__ 사용 권장

# is_sufficient_text 함수는 content_analyzer.py로 이동했으므로 여기서 제거합니다.

def process_input(text: str | None = None, url: str | None = None, metadata: dict | None = None): # metadata 추가
    """
    입력 텍스트 또는 URL을 받아 클리닝하고 S3 저장 후 SQS 메시지를 발행합니다.
    Phase 1의 전체 워크플로우를 오케스트레이션합니다.
    """
    # metadata가 None일 경우 빈 딕셔너리로 초기화하여 안전하게 .get() 사용
    if metadata is None:
        metadata = {}

    final_cleaned_text: str | None = None
    # processing_details는 전반적인 처리 과정을 담고,
    # fetch_specific_details는 content_fetch_coordinator 또는 raw text 처리에서 오는 세부 정보
    processing_details = {}
    fetch_specific_details = {}

    source_identifier = url if url else (metadata.get('source_identifier_from_text') or 'raw_text_input')
    source_type = 'raw_text' if text is not None else ('url' if url is not None else 'unknown')

    processing_details['source_identifier'] = source_identifier
    processing_details['source_type'] = source_type
    # metadata에서 사용자 ID 등 추가 정보 가져오기
    processing_details['user_id'] = metadata.get('user_id')
    processing_details['document_id'] = metadata.get('document_id') # 원본 문서 ID (예: DB의 PK)


    logger.info(f"Starting ingestion process for: {source_identifier}, User: {processing_details.get('user_id')}")

    # --- 1. 자료 가져오기 및 클리닝 ---
    try:
        if text is not None:
            logger.info("Input is raw text. Starting cleaning.")
            final_cleaned_text = text_cleaner.clean_raw_text(text)
            fetch_specific_details = {
                'method': 'raw_text_input',
                'original_length': len(text) # 원본 raw text 길이
            }

        elif url is not None:
            logger.info(f"Input is URL: {url}. Delegating to Content Fetch Coordinator.")
            # content_fetch_coordinator가 (cleaned_text, fetch_details)를 반환
            final_cleaned_text, fetch_specific_details = coordinate_fetch_from_url(url)
            # fetch_specific_details에는 'original_url', 'method', 'static_fetch_length' 등이 포함됨

        else:
            logger.error("Orchestrator received no valid input (text or url).")
            processing_details['reason'] = 'No valid input provided to orchestrator'
            # 모든 세부사항을 병합하여 반환
            return {'status': 'failed', 'details': {**processing_details, **fetch_specific_details}}

        # fetch_specific_details를 processing_details에 병합
        processing_details.update(fetch_specific_details)

        # --- 최종 클린 텍스트 확인 ---
        if not final_cleaned_text or not final_cleaned_text.strip():
            logger.warning(f"No usable cleaned text produced for input: {source_identifier}. Final length: {len(final_cleaned_text) if final_cleaned_text else 0}")
            processing_details['reason'] = 'No usable text extracted or cleaned after all fetch/clean attempts'
            # 실패 이유에 대한 더 상세한 정보는 fetch_specific_details에 이미 'method' 등으로 기록되어 있음
            return {'status': 'failed', 'details': processing_details}

        # 클린 텍스트 길이 제한 (예시)
        max_cleaned_text_length = 10 * 1024 * 1024 # 10MB
        if len(final_cleaned_text) > max_cleaned_text_length:
            logger.warning(f"Cleaned text length ({len(final_cleaned_text)}) exceeds recommended max length ({max_cleaned_text_length}).")
            processing_details['warning'] = f"Cleaned text exceeds max recommended length ({max_cleaned_text_length})"
            # 여기서 텍스트를 자르거나, 그대로 진행할 수 있습니다. 현재는 경고만 로깅.

        processing_details['cleaned_length'] = len(final_cleaned_text)
        logger.info(f"Successfully produced cleaned text for {source_identifier}. Length: {processing_details['cleaned_length']}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during text ingestion/cleaning for {source_identifier}: {e}", exc_info=True)
        processing_details['reason'] = 'Unexpected error during ingestion/cleaning orchestration'
        processing_details['error_details'] = str(e)
        return {'status': 'failed', 'details': processing_details}

    # --- 2. S3에 cleaned_text 저장 ---
    s3_object_key = None
    s3_full_path = None # SQS 메시지에 넣기 위해 미리 선언
    try:
        object_uuid = uuid.uuid4().hex
        # 사용자별/문서 ID별 경로 구성 (metadata 활용)
        user_id_path = processing_details.get('user_id', 'unknown_user')
        doc_id_path = processing_details.get('document_id', object_uuid) # document_id가 없으면 uuid 사용

        s3_object_key = f"cleaned-text/{user_id_path}/{doc_id_path}.txt"

        logger.info(f"Initiating S3 upload for object key: {s3_object_key}")
        s3_full_path = s3_storage.upload_text(s3_object_key, final_cleaned_text)

        processing_details['s3_key'] = s3_object_key
        processing_details['s3_full_path'] = s3_full_path

    except Exception as e:
        logger.error(f"Error uploading cleaned text to S3 for {source_identifier}: {e}", exc_info=True)
        processing_details['reason'] = 'S3 upload failed'
        processing_details['s3_upload_error'] = str(e)
        return {'status': 'failed', 'details': processing_details}

    # --- 3. AI 분석 SQS 큐에 메시지 발행 ---
    sqs_message_id = None
    try:
        sqs_message_body = {
            's3_path': s3_full_path, # S3 전체 경로
            's3_key': s3_object_key,
            # 'original_source_type': source_type, # processing_details에 이미 포함
            # 'source_identifier': source_identifier, # processing_details에 이미 포함
            # user_id, document_id 등 필요한 정보는 processing_details에 이미 포함되어 있음
            'ingestion_details': processing_details, # 전체 처리 과정을 담은 ingestion_details 전달
        }
        # metadata의 다른 정보들도 필요시 sqs_message_body에 직접 추가 가능
        # 예: sqs_message_body['custom_tag'] = metadata.get('custom_tag')

        logger.info(f"Initiating SQS message publishing for S3 path: {s3_full_path}")
        sqs_message_id = sqs_publisher.publish_analysis_request(sqs_message_body)

        processing_details['sqs_message_id'] = sqs_message_id

    except Exception as e:
        logger.error(f"Error publishing SQS message for S3 path {s3_full_path}: {e}", exc_info=True)
        processing_details['reason'] = 'SQS publish failed'
        processing_details['sqs_publish_error'] = str(e)
        # SQS 발행 실패 시 고급 오류 처리 (예: S3 파일 롤백 또는 별도 알림) 고려 가능
        return {'status': 'failed', 'details': processing_details}

    logger.info(f"Ingestion process completed and SQS initiation successful for: {source_identifier}. SQS Message ID: {sqs_message_id}")
    return {
        'status': 'initiated',
        'details': processing_details # 최종적으로 모든 정보가 담긴 details
    }