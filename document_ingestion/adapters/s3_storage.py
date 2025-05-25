# ingestion_lambda/adapters/s3_storage.py

import boto3
import logging

from analysis_lambda.config import INGESTION_S3_BUCKET_NAME

logger = logging.getLogger()
# 로거 레벨은 ingestion_lambda/handler.py 등에서 설정되었거나 Lambda 환경 설정 따름

# S3 클라이언트 초기화 (Lambda 실행 환경 재사용을 위해 전역으로 선언)
# AWS SDK는 Lambda 환경에서 AWS 자격 증명 및 리전을 자동으로 찾습니다.
# 명시적으로 region_name을 지정할 수도 있습니다.
s3_client = boto3.client('s3')

def upload_text(object_key: str, text_content: str) -> str:
    """
    주어진 텍스트 내용을 S3 버킷의 지정된 객체 키로 업로드합니다.

    Args:
        object_key: S3 버킷 내 파일 경로 (예: 'cleaned-text/my-document.txt'). 앞에 슬래시(/)는 붙이지 않습니다.
        text_content: S3에 저장할 텍스트 내용. None인 경우 빈 파일로 저장합니다.

    Returns:
        업로드된 S3 객체의 전체 경로 (예: 's3://your-bucket-name/cleaned-text/my-document.txt').
        업로드 실패 시 예외 발생.
    """
    # 환경 변수 미설정 시 오류 체크
    if not INGESTION_S3_BUCKET_NAME:
        logger.error(f"S3 버킷 이름이 구성되지 않아 {object_key} 업로드 실패.")
        raise EnvironmentError("S3_BUCKET_NAME environment variable not set.")

    if not object_key or object_key.startswith('/'):
        logger.error(f"유효하지 않은 S3 객체 키 형식: {object_key}. 키는 비어있거나 '/'로 시작할 수 없습니다.")
        raise ValueError("S3 object key must be a non-empty string and cannot start with '/'.")

    # text_content가 None이면 빈 문자열로 처리
    content_to_upload = text_content if text_content is not None else ""

    try:
        logger.info(f"S3 업로드 시도: 버킷={INGESTION_S3_BUCKET_NAME}, 키={object_key}")
        s3_client.put_object(
            Bucket=INGESTION_S3_BUCKET_NAME, # 수정
            Key=object_key,
            Body=content_to_upload.encode('utf-8'),
            ContentType='text/plain; charset=utf-8' # 필요시 Content-Type 지정
        )

        s3_path = f"s3://{INGESTION_S3_BUCKET_NAME}/{object_key}"
        logger.info(f"S3 업로드 성공: {s3_path}")

        return s3_path

    except Exception as e:
        # Boto3 S3 관련 예외 발생 시
        logger.error(f"S3 업로드 실패: 버킷={S3_BUCKET_NAME}, 키={object_key}, 오류={e}", exc_info=True)
        # 업로드 실패 시 예외를 다시 발생시켜 호출자(orchestrator)에게 알림
        raise # 예외 다시 발생