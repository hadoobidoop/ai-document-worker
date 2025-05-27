# analysis_worker_app/adapters/s3_client.py

import logging
import boto3
from botocore.exceptions import ClientError

# 로거 설정 (worker.py에서 logging.basicConfig로 전역 설정된 것을 따름)
logger = logging.getLogger(__name__)

# S3 클라이언트는 모듈 로드 시 (워커 애플리케이션 시작 시) 한 번만 초기화되어
# 이후의 Lambda 호출(이 경우 워커의 SQS 메시지 처리) 간에 재사용됩니다.
s3_client = boto3.client('s3')

def get_text_from_s3(s3_path: str) -> str | None:
    """
    주어진 S3 경로에서 텍스트 파일을 읽어 그 내용을 반환합니다.

    Args:
        s3_path: S3 객체의 전체 경로 (예: "s3://your-bucket-name/path/to/your-file.txt")

    Returns:
        파일의 텍스트 내용 (str) 또는 오류 발생 시 None.
    """
    if not s3_path or not s3_path.startswith("s3://"):
        logger.error(f"잘못된 S3 경로 형식입니다: {s3_path}")
        return None

    try:
        # "s3://" 접두사 제거 후 버킷 이름과 객체 키 분리
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        if len(path_parts) < 2: # 객체 키가 없는 경우 (예: "s3://bucket-name/")
            logger.error(f"잘못된 S3 경로입니다 - 객체 키가 없습니다: {s3_path}")
            return None

        bucket_name = path_parts[0]
        object_key = path_parts[1]

        logger.info(f"S3에서 파일 읽기 시도: Bucket='{bucket_name}', Key='{object_key}'")

        # S3에서 객체 가져오기
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)

        # 객체의 본문(Body) 내용을 UTF-8로 디코딩하여 텍스트로 변환
        text_content = response['Body'].read().decode('utf-8')

        logger.info(f"S3에서 성공적으로 텍스트를 읽었습니다. 내용 길이: {len(text_content)} 글자.")
        return text_content

    except ClientError as e:
        # Boto3 ClientError 처리 (예: NoSuchKey, AccessDenied 등)
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'NoSuchKey':
            logger.error(f"S3 객체를 찾을 수 없습니다 (NoSuchKey): s3://{bucket_name}/{object_key}")
        elif error_code == 'AccessDenied':
            logger.error(f"S3 객체 접근 권한이 없습니다 (AccessDenied): s3://{bucket_name}/{object_key}")
        else:
            logger.error(
                f"S3 ClientError 발생 (경로: {s3_path}): {error_code} - {e}",
                exc_info=False # ClientError는 이미 상세 정보를 담고 있으므로 스택 트레이스는 간결하게
            )
        return None
    except Exception as e:
        # 기타 예외 처리
        logger.error(f"S3 경로 {s3_path} 에서 파일 읽기 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return None