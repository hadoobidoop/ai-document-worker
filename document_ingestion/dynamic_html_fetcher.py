# ingestion_lambda/dynamic_html_fetcher.py

import logging
import os
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, Playwright, TimeoutError as PlaywrightTimeoutError # TimeoutError 임포트

from bs4 import BeautifulSoup

from . import text_cleaner
from .html_constants import UNWANTED_HTML_TAGS
from .static_html_fetcher import is_valid_url

logger = logging.getLogger(__name__) # __name__ 사용 권장

playwright_instance: Playwright | None = None
browser = None
browser_context = None

def initialize_playwright():
    """Playwright 브라우저 및 컨텍스트를 초기화합니다."""
    global playwright_instance, browser, browser_context

    # browser.is_connected() 체크 추가하여 브라우저 연결 상태 확인
    if browser_context is None or browser is None or not browser.is_connected():
        logger.info("Initializing Playwright browser and context...")
        try:
            if playwright_instance is None or not playwright_instance._was_started: # _was_started는 내부 API일 수 있으므로 주의, 혹은 단순히 playwright_instance is None으로 체크
                playwright_instance = sync_playwright().start()

            browser = playwright_instance.chromium.launch(
                headless=True,
                args=[
                    "--single-process",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process", # 메모리 절약 시도
                    "--disable-extensions", # 확장 기능 비활성화
                    "--disable-component-extensions-with-background-pages",
                    "--disable-default-apps",
                    "--disable-sync",
                    # "--blink-settings=imagesEnabled=false" # 이미지 로딩 비활성화 (page.route로 제어하는 것이 더 유연)
                ],
                # executable_path="/opt/chromium/chromium-linux/chrome" # Layer 사용 시 필요하면 경로 지정
            )
            browser_context = browser.new_context(
                ignore_https_errors=True,
                # 사용자 에이전트는 필요시 설정 (현재는 기본값 사용)
                # viewport={"width": 1280, "height": 800} # 필요시 뷰포트 설정
            )
            logger.info("Playwright browser and context initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}", exc_info=True)
            # 전역 변수들을 None으로 다시 설정하여 다음 호출 시 재시도 유도
            playwright_instance = None
            browser = None
            browser_context = None
            # 여기서 예외를 다시 발생시켜 호출자에게 알릴 수도 있습니다.
            # raise

def fetch_and_clean_dynamic_html(url: str, unwanted_tags_for_dynamic=None) -> str:
    """
    Dynamic HTML 페이지를 Headless Browser로 가져와서 클린 텍스트를 반환합니다.
    리소스 차단 기능이 추가되었습니다.
    """
    if not is_valid_url(url):
        logger.warning(f"Invalid URL provided for dynamic fetch: {url}")
        return ""

    initialize_playwright()

    if browser_context is None: # initialize_playwright 실패 시 browser_context가 None일 수 있음
        logger.error("Playwright browser context is not initialized. Cannot fetch dynamic page.")
        return "" # 초기화 실패 시 빈 문자열 반환

    logger.info(f"Attempting to fetch dynamic HTML from: {url}")
    cleaned_text = ""
    page = None

    try:
        page = browser_context.new_page()

        # --- 리소스 차단 로직 시작 ---
        def handle_route(route):
            resource_type = route.request.resource_type
            # 차단할 리소스 유형 목록 (서비스에 맞게 조정)
            # "stylesheet"을 차단하면 일부 사이트의 레이아웃이 깨져 내용 추출에 영향 줄 수 있으므로 주의
            # 테스트 후 문제가 없다면 포함, 문제 있다면 제외하거나 특정 CSS만 허용/차단
            blocked_resource_types = [
                "image", "media", "font", "manifest",
                # "stylesheet", # 스타일시트 차단은 신중하게 결정
                # "script", # 스크립트 차단은 SPA의 경우 핵심 콘텐츠 로드를 막을 수 있으므로 매우 주의
            ]
            # 특정 도메인 차단 목록 (예: 분석, 광고)
            blocked_domains = [
                "google-analytics.com", "googletagmanager.com", "doubleclick.net",
                "facebook.net", "twitter.com", # 소셜 위젯 등
                # 필요에 따라 더 많은 광고/추적 도메인 추가
            ]

            if resource_type in blocked_resource_types:
                # logger.debug(f"Blocking resource: {route.request.url} (type: {resource_type})")
                route.abort()
                return # 명시적 return으로 아래 로직 스킵

            for domain in blocked_domains:
                if domain in route.request.url:
                    # logger.debug(f"Blocking resource: {route.request.url} (domain: {domain})")
                    route.abort()
                    return # 명시적 return

            route.continue_()

        page.route("**/*", handle_route) # 모든 요청(* 모든 경로, * 모든 호스트)에 대해 handle_route 함수 적용
        # --- 리소스 차단 로직 끝 ---

        logger.info(f"Navigating to {url} and waiting for networkidle...")
        page.goto(url, wait_until="networkidle", timeout=30000) # 30초 타임아웃

        logger.info(f"Page loaded dynamically: {url}. Extracting content.")
        rendered_html = page.content()

        soup = BeautifulSoup(rendered_html, 'html.parser')

        # text_cleaner.py에서 사용할 unwanted_tags와 동일하게 유지하거나,
        # dynamic_html_fetcher 전용으로 다르게 가져갈 수도 있습니다.
        # 현재는 static_html_fetcher.py와 동일한 태그 목록을 사용한다고 가정합니다.
        # (이 부분은 static_html_fetcher.py의 unwanted_tags 목록과 동기화 필요)
        for tag_name in UNWANTED_HTML_TAGS: # 수정: 공통 상수 사용
            for tag in soup.find_all(tag_name):
                tag.extract()

        page_text = soup.get_text(separator='\n', strip=True)
        cleaned_text = text_cleaner.clean_raw_text(page_text)

        logger.debug(f"Cleaned text after dynamic fetch (Length: {len(cleaned_text)})")

    except PlaywrightTimeoutError: # playwright.sync_api.TimeoutError
        logger.error(f"TimeoutError during dynamic HTML fetch for {url}", exc_info=False) # 스택 트레이스는 간결하게
        cleaned_text = ""
    except Exception as e:
        logger.error(f"Error during dynamic HTML fetch for {url}: {e}", exc_info=True)
        cleaned_text = ""
    finally:
        if page:
            try:
                page.close()
                logger.info(f"Closed page for {url}.")
            except Exception as close_e:
                logger.error(f"Error closing page for {url}: {close_e}")

    return cleaned_text

# Lambda 환경 종료 시 브라우저를 정리하는 함수 (선택 사항, 고급)
# def cleanup_playwright():
#     global playwright_instance, browser, browser_context
#     if browser:
#         try:
#             browser.close()
#             logger.info("Playwright browser closed.")
#         except Exception as e:
#             logger.error(f"Error closing browser: {e}")
#     if playwright_instance and playwright_instance._was_started:
#         try:
#             playwright_instance.stop()
#             logger.info("Playwright instance stopped.")
#         except Exception as e:
#             logger.error(f"Error stopping playwright_instance: {e}")
#     playwright_instance = None
#     browser = None
#     browser_context = None

# Lambda 핸들러의 최상위 레벨이나, Lambda Extension 등을 사용하여
# Lambda 실행 컨테이너가 종료되기 직전에 cleanup_playwright()를 호출하도록 구성할 수 있습니다.
# (일반적으로는 Lambda가 알아서 정리하지만, 명시적 정리가 필요하다고 판단될 경우)