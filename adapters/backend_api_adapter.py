# analysis_worker_app/adapters/backend_api_adapter.py

import logging
import json
import requests # HTTP API 호출을 위해 requests 라이브러리 사용
from typing import List, Optional, Dict, Any # 타입 힌팅용

# config 모듈에서 Spring Boot API 엔드포인트 등의 설정을 가져옵니다.
# 이 파일은 adapters 폴더 안에 있으므로, 부모 폴더의 config.py를 참조하기 위해 ".." 사용
from .. import config

# 로거 설정 (worker.py에서 logging.basicConfig로 전역 설정된 것을 따름)
logger = logging.getLogger(__name__)

# Spring Boot API 내 결과 저장을 위한 구체적인 경로 (실제 백엔드 API 설계에 따라 변경)
# 예시: /api/v1/analysis/update-status 또는 /api/v1/documents/{documentId}/analysis
# 여기서는 문서를 생성하거나 업데이트하는 단일 엔드포인트를 가정합니다.
ANALYSIS_RESULTS_API_PATH = "/api/ai/processed-documents" # 이전 제안과 동일하게 유지

def save_analysis_results_to_backend(
        document_id: str,
        user_id: Optional[str],
        s3_path: str, # 원본 텍스트가 저장된 S3 경로
        original_url: Optional[str], # 문서의 원본 URL (수집 시 URL이었다면)
        analysis_status: str, # 현재 처리 상태 (예: "TEXT_LOADED", "PROCESSING_FAILED", "COMPLETED_PLACEHOLDER")
        error_message: Optional[str] = None,
        # --- 아래는 Priority 2에서 채워질 AI 분석 결과 필드들 (현재는 None 또는 기본값 전달) ---
        short_summary: Optional[str] = None,
        long_markdown_summary: Optional[str] = None,
        category: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
        num_embedded_chunks: int = 0 # 초기값 0
) -> bool:
    """
    분석된 결과 (또는 처리 상태)를 Spring Boot 백엔드 API를 통해 저장합니다.
    Priority 1에서는 기본적인 상태 업데이트 위주로 사용됩니다.

    Args:
        document_id: 원본 문서의 고유 ID.
        user_id: 사용자 ID.
        s3_path: S3에 저장된 원본 텍스트 파일의 경로.
        original_url: 문서의 원본 URL (있는 경우).
        analysis_status: 분석 처리 상태.
        error_message: 처리 실패 시 원인.
        short_summary, long_markdown_summary, category, hashtags, num_embedded_chunks:
            AI 분석 결과 (Priority 1에서는 대부분 None 또는 기본값).

    Returns:
        API 호출 성공 시 True, 실패 시 False.
    """
    if not config.SPRING_BOOT_API_ENDPOINT:
        logger.error("SPRING_BOOT_API_ENDPOINT가 config.py에 설정되지 않았습니다. 분석 결과를 백엔드로 전송할 수 없습니다.")
        return False

    # 전체 API URL 구성
    api_url = f"{config.SPRING_BOOT_API_ENDPOINT.rstrip('/')}{ANALYSIS_RESULTS_API_PATH}"

    # 백엔드 API로 전송할 데이터 페이로드 구성
    # 이 페이로드의 필드명은 Spring Boot API가 기대하는 DTO(Data Transfer Object)의 필드명과 정확히 일치해야 합니다.
    payload: Dict[str, Any] = {
        "documentId": document_id,
        "userId": user_id,
        "s3Path": s3_path,
        "originalUrl": original_url,
        "status": analysis_status, # 필수: 현재 처리 상태
        "errorMessage": error_message,
        # Priority 1에서는 아래 값들이 대부분 기본값이거나 None일 수 있습니다.
        "summaryShort": short_summary if short_summary is not None else "",
        "summaryLongMarkdown": long_markdown_summary if long_markdown_summary is not None else "",
        "category": category if category is not None else "미분류", # 기본값
        "hashtags": hashtags if hashtags is not None else [],       # 기본값
        "embeddedChunkCount": num_embedded_chunks
    }

    logger.info(f"백엔드 API로 분석 결과(상태: {analysis_status}) 전송 시도. 문서 ID: {document_id}.")
    # 디버그 시에만 페이로드 전체 로깅 (민감 정보 포함 가능성 주의)
    # logger.debug(f"API URL: {api_url}")
    # logger.debug(f"백엔드 API 페이로드: {json.dumps(payload, ensure_ascii=False)}")

    try:
        response = requests.post(
            api_url,
            json=payload, # dict가 자동으로 JSON 문자열로 변환되어 전송됨
            headers={'Content-Type': 'application/json'},
            timeout=15  # 백엔드 API 호출 타임아웃 (초 단위, 네트워크 상황에 따라 조절)
        )
        # HTTP 오류 코드 (4xx 클라이언트 오류, 5xx 서버 오류) 발생 시 예외 발생
        response.raise_for_status()

        logger.info(f"백엔드 API로 분석 결과 성공적으로 전송 완료 (문서 ID: {document_id}). 응답 코드: {response.status_code}")
        # 필요시 API 응답 본문 확인: response_data = response.json()
        return True

    except requests.exceptions.Timeout:
        logger.error(f"백엔드 API 호출 시간 초과 (문서 ID: {document_id}, URL: {api_url}).", exc_info=False)
    except requests.exceptions.HTTPError as e:
        # HTTP 오류 발생 시 응답 내용 로깅 (민감 정보가 없을 경우)
        error_content = e.response.text if e.response else "응답 내용 없음"
        logger.error(
            f"백엔드 API 호출 중 HTTP 오류 발생 (문서 ID: {document_id}): {e.response.status_code if e.response else 'N/A'} - 오류 내용: {error_content[:200]}...", # 오류 내용은 너무 길 수 있으므로 일부만 로깅
            exc_info=False
        )
    except requests.exceptions.RequestException as e:
        # 기타 requests 라이브러리 관련 예외 (예: ConnectionError)
        logger.error(f"백엔드 API 호출 중 RequestException 발생 (문서 ID: {document_id}): {e}", exc_info=True)
    except Exception as e:
        # 예상치 못한 기타 예외
        logger.error(f"백엔드 API 호출 중 예상치 못한 오류 발생 (문서 ID: {document_id}): {e}", exc_info=True)

    return False