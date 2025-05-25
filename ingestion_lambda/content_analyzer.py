# ingestion_lambda/content_analyzer.py

import logging
import re
from bs4 import BeautifulSoup

# text_cleaner 모듈 임포트 (실제 프로젝트 구조에 맞게)
from . import text_cleaner
# 만약 text_cleaner가 최상위 폴더의 shared 등에 있다면 from shared import text_cleaner 와 같이 변경

logger = logging.getLogger(__name__) # __name__을 사용하여 로거 이름 지정 권장

# --- 설정값 (테스트를 통해 서비스에 맞게 조정 필요) ---
MIN_OVERALL_TEXT_LENGTH = 150
MIN_PRIMARY_CONTENT_LENGTH = 100
MIN_P_TAG_AGGREGATE_LENGTH = 250
VERY_SHORT_TEXT_THRESHOLD = 50

CONTENT_SELECTORS = [
    'article', 'main', '.post-content', '.entry-content', '.td-post-content',
    'div.story', 'div.content', 'div#main-content', 'div.main-content',
    'div.article-body', 'div[itemprop="articleBody"]', 'div.blog-post-content',
    'section.entry-content', 'section.article-content', 'div.text-content'
]

def _get_non_whitespace_length(text: str) -> int:
    """
    주어진 텍스트에서 모든 공백 문자를 제거한 후의 길이를 반환합니다.
    text_cleaner.clean_raw_text를 거친 텍스트에 사용하는 것이 좋습니다.
    """
    if not text:
        return 0
    # text_cleaner.clean_raw_text가 이미 연속 공백을 단일 공백으로 변경하고 strip() 했으므로,
    # 여기서는 모든 공백을 제거하여 순수 문자열 길이만 계산합니다.
    return len(re.sub(r'\s+', '', text))

def is_sufficient_text(
        cleaned_statically_fetched_text: str,
        statically_fetched_soup: BeautifulSoup | None
) -> bool:
    """
    정적 HTML 분석 결과(텍스트 및 파싱된 soup 객체)를 바탕으로
    콘텐츠가 충분한지, 아니면 동적 분석이 필요한지를 판단합니다.

    Args:
        cleaned_statically_fetched_text: 정적 분석기가 추출하고 text_cleaner로 정리한 텍스트.
        statically_fetched_soup: 정적 분석기가 파싱하고 주요 불필요 태그를 제거한 BeautifulSoup 객체.
                                 (주의: 이 soup에서 script 태그 개수 확인은 static_html_fetcher가
                                  script를 제거하기 전에 수행해야 효과적입니다.)

    Returns:
        True이면 정적 콘텐츠가 충분하다고 판단, False이면 부족하여 동적 분석을 고려.
    """
    overall_text_length = _get_non_whitespace_length(cleaned_statically_fetched_text)
    logger.debug(f"Sufficiency Check: Initial overall non-whitespace text length: {overall_text_length}")

    if overall_text_length < VERY_SHORT_TEXT_THRESHOLD:
        logger.info(
            f"Sufficiency Check: Text is very short ({overall_text_length} < {VERY_SHORT_TEXT_THRESHOLD}). "
            f"Declaring insufficient."
        )
        return False

    if overall_text_length < MIN_OVERALL_TEXT_LENGTH:
        logger.info(
            f"Sufficiency Check: Overall text length ({overall_text_length}) is less than "
            f"MIN_OVERALL_TEXT_LENGTH ({MIN_OVERALL_TEXT_LENGTH}). Declaring insufficient."
        )
        return False

    if not statically_fetched_soup:
        logger.info(
            "Sufficiency Check: No BeautifulSoup object for structural analysis. "
            "Overall text length was sufficient. Declaring sufficient."
        )
        return True

    for selector in CONTENT_SELECTORS:
        content_element = statically_fetched_soup.select_one(selector)
        if content_element:
            element_text = text_cleaner.clean_raw_text(
                content_element.get_text(separator=' ', strip=True)
            )
            element_text_length = _get_non_whitespace_length(element_text)

            if element_text_length >= MIN_PRIMARY_CONTENT_LENGTH:
                logger.info(
                    f"Sufficiency Check: Found sufficient content in primary selector '{selector}' "
                    f"(length: {element_text_length} >= {MIN_PRIMARY_CONTENT_LENGTH}). Declaring sufficient."
                )
                return True

    logger.debug(
        "Sufficiency Check: No single primary selector met criteria. "
        "Checking aggregate <p> tag content."
    )
    total_p_text_length = 0
    all_p_tags = statically_fetched_soup.find_all('p')

    if not all_p_tags:
        logger.info("Sufficiency Check: No <p> tags found for aggregate check.")
    else:
        for p_tag in all_p_tags:
            p_text = text_cleaner.clean_raw_text(
                p_tag.get_text(separator=' ', strip=True)
            )
            total_p_text_length += _get_non_whitespace_length(p_text)

        logger.debug(f"Sufficiency Check: Aggregate <p> tag non-whitespace text length: {total_p_text_length}")
        if total_p_text_length >= MIN_P_TAG_AGGREGATE_LENGTH:
            logger.info(
                f"Sufficiency Check: Aggregate <p> tag content is sufficient "
                f"({total_p_text_length} >= {MIN_P_TAG_AGGREGATE_LENGTH}). Declaring sufficient."
            )
            return True

    logger.warning(
        f"Sufficiency Check: Overall text length ({overall_text_length}) was initially adequate, "
        "but structural checks (primary selectors, p-tags) failed to find sufficient concentrated content. "
        "Declaring insufficient to be cautious."
    )
    return False