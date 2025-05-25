# ingestion_lambda/static_html_fetcher.py

import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlsplit

# 같은 ingestion_lambda 디렉토리 내 다른 모듈 임포트
from . import text_cleaner
from .html_constants import UNWANTED_HTML_TAGS

logger = logging.getLogger()
# 로거 레벨은 handler.py에서 설정되었거나 Lambda 환경 설정 따름

def is_valid_url(url):
    """간단한 URL 유효성 검사"""
    try:
        result = urlsplit(url)
        # scheme(http/https), netloc(도메인)이 있어야 유효한 URL로 간주
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

# 반환 타입 힌트 변경: str -> tuple[str, BeautifulSoup | None]
def fetch_and_clean_static_html(url: str) -> tuple[str, BeautifulSoup | None]:
    """
    Static HTML 페이지를 가져와서 불필요한 내용을 제거하고 클린 텍스트를 반환하며,
    파싱된 BeautifulSoup 객체도 함께 반환합니다.
    네트워크 오류, 파싱 오류 등을 처리합니다.
    """
    if not is_valid_url(url):
        logger.warning(f"Invalid URL provided: {url}")
        return "", None # 유효하지 않은 URL은 빈 문자열과 None 반환

    logger.info(f"Attempting to fetch static HTML from: {url}")
    cleaned_text = ""
    soup = None # BeautifulSoup 객체를 저장할 변수

    try:
        # 웹페이지 HTML 내용 가져오기
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)

        response.raise_for_status() # 200번대 코드가 아니면 예외 발생

        logger.info(f"Successfully fetched HTML from {url}. Status Code: {response.status_code}")

        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')

        # -- 텍스트 추출 및 클리닝 (기존 로직 유지) --
        # 스크립트, 스타일 등 불필요한 태그 제거
        for tag_name in UNWANTED_HTML_TAGS: # 수정: 공통 상수 사용
            for tag_to_remove in soup(tag_name): # soup(tag_name)으로 직접 사용 가능
              tag_to_remove.extract()

        # 텍스트 내용 추출
        page_text = soup.get_text(separator='\n', strip=True)

        # text_cleaner 모듈을 사용하여 최종 클리닝 적용
        cleaned_text = text_cleaner.clean_raw_text(page_text)
        # -- 텍스트 추출 및 클리닝 끝 --


        logger.debug(f"Cleaned text after static fetch (Length: {len(cleaned_text)}) and returning soup object.")

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching URL: {url}")
        cleaned_text = ""
        soup = None # 오류 발생 시 soup도 None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        cleaned_text = ""
        soup = None # 오류 발생 시 soup도 None

    except Exception as e:
        logger.error(f"An unexpected error occurred during static HTML processing for {url}: {e}", exc_info=True)
        cleaned_text = ""
        soup = None # 오류 발생 시 soup도 None

    # 클리닝된 텍스트와 BeautifulSoup 객체를 함께 반환
    return cleaned_text, soup