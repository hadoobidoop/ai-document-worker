# ingestion_lambda/content_fetch_coordinator.py

import logging

from core.fechers import static_html_fetcher, dynamic_html_fetcher
from core.processing import content_analyzer

logger = logging.getLogger(__name__)

def coordinate_fetch_from_url(url: str) -> tuple[str | None, dict]:
    """
    주어진 URL로부터 웹 콘텐츠를 가져오는 과정을 조정합니다.
    정적 분석을 우선 시도하고, content_analyzer의 판단에 따라 동적 분석을 수행합니다.

    Args:
        url: 콘텐츠를 가져올 URL

    Returns:
        tuple: (추출 및 정리된 텍스트 | None, 처리 상세 정보 딕셔너리)
               추출된 텍스트가 없는 경우 None 대신 빈 문자열("")을 반환할 수도 있습니다.
    """
    fetch_processing_details = {
        'original_url': url,
        'method': 'initial', # 초기 상태
        'static_fetch_length': 0,
        'static_soup_available': False,
        'dynamic_fetch_attempted': False,
        'dynamic_fetch_length': 0,
    }
    final_cleaned_text: str | None = None # 최종적으로 반환될 텍스트

    # 1. 정적 HTML 가져오기 및 기본 정리
    logger.info(f"Attempting Static HTML fetch for: {url}")
    try:
        # static_html_fetcher는 (cleaned_text, soup)를 반환합니다.
        static_text, static_soup = static_html_fetcher.fetch_and_clean_static_html(url)

        # static_text는 이미 static_html_fetcher 내부에서 text_cleaner.clean_raw_text를 거친 결과입니다.
        fetch_processing_details['static_fetch_length'] = len(static_text) if static_text else 0
        fetch_processing_details['static_soup_available'] = static_soup is not None

    except Exception as e:
        logger.error(f"Static HTML fetch failed catastrophically for {url}: {e}", exc_info=True)
        # 정적 분석 자체가 완전 실패하면, 동적 분석을 시도하거나 여기서 바로 실패 처리 가능
        # 여기서는 일단 빈 값으로 두고, 동적 분석 시도 여지는 남겨둘 수 있으나,
        # 보통은 정적 분석 실패 시 바로 종료하는 경우가 많습니다.
        # 현재 로직에서는 아래 is_sufficient_text에서 static_text가 None이나 빈 값일 경우 처리됩니다.
        static_text = "" # 또는 None, 오류 발생 시 빈 문자열로 초기화
        static_soup = None


    # 2. 정적 분석 결과의 충분성 판단
    # content_analyzer.is_sufficient_text는 cleaned_text와 soup을 받습니다.
    # static_text는 static_html_fetcher에서 이미 내부적으로 text_cleaner를 통과했습니다.
    if content_analyzer.is_sufficient_text(static_text, static_soup):
        logger.info(f"Static fetch deemed sufficient for {url}.")
        final_cleaned_text = static_text
        fetch_processing_details['method'] = 'static_html_sufficient'
    else:
        logger.warning(f"Static fetch deemed insufficient for {url}. Attempting Dynamic HTML fetch.")
        fetch_processing_details['dynamic_fetch_attempted'] = True
        fetch_processing_details['method'] = 'dynamic_html_attempted' # 동적 분석 시도 기록

        # 3. 동적 HTML 가져오기 (정적 분석 결과가 불충분할 경우)
        try:
            # dynamic_html_fetcher는 최종 정리된 텍스트를 반환합니다.
            dynamic_text = dynamic_html_fetcher.fetch_and_clean_dynamic_html(url)
            fetch_processing_details['dynamic_fetch_length'] = len(dynamic_text) if dynamic_text else 0

            if dynamic_text and dynamic_text.strip(): # 동적 분석으로 유의미한 텍스트를 얻은 경우
                final_cleaned_text = dynamic_text.strip() # 이미 dynamic_html_fetcher에서 strip() 등을 했을 수 있지만, 최종 확인
                fetch_processing_details['method'] = 'dynamic_html_success'
                logger.info(f"Dynamic fetch successful for {url}.")
            else: # 동적 분석을 시도했으나 빈 텍스트/결과가 없는 경우
                # 이 경우, 불충분했던 static_text를 사용할지, 아니면 아예 빈 값으로 처리할지 결정.
                # 원본 ingestion_orchestrator.py 에서는 이런 경우 cleaned_text = "" 로 처리했습니다.
                # 즉, 동적 분석을 시도했다면 그 결과(성공 또는 빈 값)를 따릅니다.
                final_cleaned_text = ""
                fetch_processing_details['method'] = 'dynamic_html_empty_result'
                logger.warning(f"Dynamic fetch for {url} returned empty or whitespace text.")

        except Exception as e:
            logger.error(f"Error during Dynamic HTML fetch for {url}: {e}", exc_info=True)
            # 동적 분석 중 예외 발생 시에도 빈 값으로 처리 (원본 로직 따름)
            final_cleaned_text = ""
            fetch_processing_details['method'] = 'dynamic_html_exception'
            fetch_processing_details['dynamic_fetch_error'] = str(e)

    # 최종 반환값은 (정리된 텍스트, 처리 상세 정보)
    # final_cleaned_text가 None일 수 있으므로, 호출부에서 빈 문자열로 처리하거나 여기서 처리
    return final_cleaned_text if final_cleaned_text is not None else "", fetch_processing_details