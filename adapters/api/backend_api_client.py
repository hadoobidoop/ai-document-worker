# analysis_worker_app/adapters/backend_api_client.py
# 이제 이 파일이 표준 백엔드 API 어댑터입니다.

import logging
import requests # HTTP API 호출을 위해 requests 라이브러리 사용
from typing import List, Optional, Dict, Any # 타입 힌팅용

from config import settings

# config 모듈 임포트 경로 수정
# worker.py가 프로젝트 루트에 있고, config.py가 analysis_lambda 폴더 하위에 있으므로
# analysis_lambda를 패키지로 인식할 수 있도록 PYTHONPATH 설정이 되어있거나,
# worker.py와 동일한 레벨로 config 관련 부분이 이동되어야 할 수 있습니다.
# 여기서는 analysis_lambda가 PYTHONPATH에 있다고 가정합니다.

# 로거 설정 (worker.py에서 logging.basicConfig로 전역 설정된 것을 따름)
logger = logging.getLogger(__name__)

# Spring Boot API 내 결과 저장을 위한 구체적인 경로 (실제 백엔드 API 설계에 따라 변경)
ANALYSIS_RESULTS_API_PATH = "/api/ai/processed-documents" # API 경로 일관성 유지

def save_analysis_results_to_backend(
        document_id: str,
        user_id: Optional[str],
        s3_path: str, # 원본 텍스트가 저장된 S3 경로
        original_url: Optional[str], # 문서의 원본 URL (수집 시 URL이었다면)
        # --- AI 분석 결과 필드들 ---
        short_summary: Optional[str],
        long_markdown_summary: Optional[str],
        category: Optional[str],
        hashtags: Optional[List[str]],
        num_embedded_chunks: int,
        # --- 처리 상태 관련 필드 ---
        analysis_status: str, # 현재 처리 상태 (예: "COMPLETED", "PARTIAL_FAILURE", "TEXT_LOAD_FAILURE")
        failure_reason: Optional[str] = None # 기존 error_message 대신 failure_reason 사용 (worker.py와 일치)
) -> bool:
    """
    분석된 결과 및 처리 상태를 Spring Boot 백엔드 API를 통해 저장합니다.

    Args:
        document_id: 원본 문서의 고유 ID.
        user_id: 사용자 ID.
        s3_path: S3에 저장된 원본 텍스트 파일의 경로.
        original_url: 문서의 원본 URL (있는 경우).
        short_summary: 짧은 요약.
        long_markdown_summary: 긴 마크다운 형식 요약.
        category: 분류된 카테고리.
        hashtags: 추출된 해시태그 리스트.
        num_embedded_chunks: 임베딩된 청크의 수.
        analysis_status: 분석 처리 상태.
        failure_reason: 처리 실패 시 원인.

    Returns:
        API 호출 성공 시 True, 실패 시 False.
    """
    if not settings.SPRING_BOOT_API_ENDPOINT:
        logger.error("SPRING_BOOT_API_ENDPOINT가 config.py에 설정되지 않았습니다. 분석 결과를 백엔드로 전송할 수 없습니다.")
        return False

    # 전체 API URL 구성
    api_url = f"{settings.SPRING_BOOT_API_ENDPOINT.rstrip('/')}{ANALYSIS_RESULTS_API_PATH}"

    # 백엔드 API로 전송할 데이터 페이로드 구성
    payload: Dict[str, Any] = {
        "documentId": document_id,
        "userId": user_id,
        "s3Path": s3_path,
        "originalUrl": original_url,
        "summaryShort": short_summary if short_summary is not None else "",
        "summaryLongMarkdown": long_markdown_summary if long_markdown_summary is not None else "",
        "category": category if category is not None else "미분류",
        "hashtags": hashtags if hashtags is not None else [],
        "embeddedChunkCount": num_embedded_chunks,
        "status": analysis_status, # worker.py에서 사용하는 필드명과 일치
        "errorMessage": failure_reason # worker.py에서 사용하는 필드명과 일치
    }

    logger.info(f"백엔드 API로 분석 결과(상태: {analysis_status}) 전송 시도. 문서 ID: {document_id}.")
    # logger.debug(f"API URL: {api_url}")
    # logger.debug(f"백엔드 API 페이로드: {json.dumps(payload, ensure_ascii=False)}") # 필요시 로깅

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=20  # 백엔드 API 호출 타임아웃 (초 단위, 적절히 조절)
        )
        response.raise_for_status()

        logger.info(f"백엔드 API로 분석 결과 성공적으로 전송 완료 (문서 ID: {document_id}). 응답 코드: {response.status_code}")
        return True

    except requests.exceptions.Timeout:
        logger.error(f"백엔드 API 호출 시간 초과 (문서 ID: {document_id}, URL: {api_url}).", exc_info=False)
    except requests.exceptions.HTTPError as e:
        error_content = e.response.text if e.response else "응답 내용 없음"
        logger.error(
            f"백엔드 API 호출 중 HTTP 오류 발생 (문서 ID: {document_id}): {e.response.status_code if e.response else 'N/A'} - 오류 내용: {error_content[:200]}...",
            exc_info=False
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"백엔드 API 호출 중 RequestException 발생 (문서 ID: {document_id}): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"백엔드 API 호출 중 예상치 못한 오류 발생 (문서 ID: {document_id}): {e}", exc_info=True)

    return False