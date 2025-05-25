# analysis_lambda/adapters/backend_api_adapter.py

import logging
import requests # HTTP API 호출용
from typing import List, Optional, Dict, Any # 타입 힌팅용

from analysis_lambda import config

# config 모듈에서 Spring Boot API 엔드포인트 가져오기

logger = logging.getLogger(__name__)

# Spring Boot API 내 결과 저장을 위한 구체적인 경로 (예시)
# 이 경로는 실제 백엔드 API 설계에 따라 결정됩니다.
ANALYSIS_RESULTS_API_PATH = "/api/v1/ai/processed-documents" # 예시 경로

def save_analysis_to_backend(
        document_id: str,
        user_id: Optional[str],
        s3_path: str,
        original_url: Optional[str],
        short_summary: Optional[str],
        long_markdown_summary: Optional[str],
        category: Optional[str],
        hashtags: Optional[List[str]],
        num_embedded_chunks: int,
        processing_status: str = "COMPLETED", # "COMPLETED", "PARTIAL_FAILURE", "TOTAL_FAILURE" 등
        failure_reason: Optional[str] = None
) -> bool:
    """
    분석된 결과를 Spring Boot 백엔드 API를 통해 저장합니다.

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
        processing_status: 분석 처리 상태.
        failure_reason: 처리 실패 시 원인.

    Returns:
        API 호출 성공 시 True, 실패 시 False.
    """
    if not config.SPRING_BOOT_API_ENDPOINT:
        logger.error("SPRING_BOOT_API_ENDPOINT is not configured in config.py. Cannot save analysis results.")
        return False

    # 최종 API URL 구성
    # config.SPRING_BOOT_API_ENDPOINT가 "http://localhost:8080" 와 같은 형태라고 가정
    api_url = f"{config.SPRING_BOOT_API_ENDPOINT.rstrip('/')}{ANALYSIS_RESULTS_API_PATH}"

    # 백엔드 API로 전송할 데이터 페이로드 구성
    # 이 페이로드의 필드명은 Spring Boot API가 기대하는 DTO의 필드명과 일치해야 합니다.
    payload: Dict[str, Any] = {
        "documentId": document_id,
        "userId": user_id,
        "s3Path": s3_path,
        "originalUrl": original_url,
        "summaryShort": short_summary if short_summary is not None else "",
        "summaryLongMarkdown": long_markdown_summary if long_markdown_summary is not None else "",
        "category": category if category is not None else "미분류", # 기본값 설정
        "hashtags": hashtags if hashtags is not None else [], # 빈 리스트로 기본값
        "embeddedChunkCount": num_embedded_chunks,
        "status": processing_status,
        "errorMessage": failure_reason
    }

    logger.info(f"Attempting to save analysis results to backend API for document_id: {document_id}.")
    logger.debug(f"API URL: {api_url}")
    # 페이로드 전체를 로깅하는 것은 민감한 내용(요약 등)이 포함될 수 있으므로 주의.
    # 필요시 주요 ID나 상태만 로깅: logger.debug(f"Payload keys: {list(payload.keys())}")
    # logger.debug(f"Payload for Spring Boot API: {json.dumps(payload, ensure_ascii=False)}")


    try:
        response = requests.post(
            api_url,
            json=payload, # dict를 자동으로 JSON 문자열로 변환하여 전송
            headers={'Content-Type': 'application/json'},
            timeout=20  # 백엔드 API 호출 타임아웃 (초 단위, 적절히 조절)
        )
        # HTTP 오류 코드 (4xx, 5xx) 발생 시 예외 발생
        response.raise_for_status()

        logger.info(f"Successfully saved analysis results via API for document_id: {document_id}. Response status: {response.status_code}")
        # API 응답 본문이 있다면 여기서 확인 가능: response.json()
        return True

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while calling backend API for document_id {document_id} at {api_url}.", exc_info=False) # 스택 트레이스 없이
    except requests.exceptions.HTTPError as e:
        # HTTP 오류 발생 시 응답 내용 로깅
        error_content = e.response.text if e.response else "No response content"
        logger.error(
            f"HTTPError while calling backend API for document_id {document_id}: {e.response.status_code if e.response else 'N/A'} - {error_content}",
            exc_info=False # 스택 트레이스 없이
        )
    except requests.exceptions.RequestException as e:
        # 기타 requests 관련 예외 (예: ConnectionError)
        logger.error(f"RequestException (e.g., connection error) while calling backend API for document_id {document_id}: {e}", exc_info=True)
    except Exception as e:
        # 예상치 못한 기타 예외
        logger.error(f"An unexpected error occurred while trying to save analysis results via API for document_id {document_id}: {e}", exc_info=True)

    return False