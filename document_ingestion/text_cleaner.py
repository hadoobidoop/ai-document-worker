# ingestion_lambda/text_cleaner.py

import re
import logging

logger = logging.getLogger()
# 로거 레벨은 handler.py에서 설정되었거나 Lambda 환경 설정 따름


def clean_raw_text(text: str) -> str:
    """
    Raw 텍스트 입력 또는 추출된 텍스트에 대한 기본적인 클리닝을 수행합니다.
    - 앞뒤 공백 제거
    - 여러 개의 연속된 공백, 탭, 줄바꿈 등을 하나로 줄임
    - 기타 기본적인 문자 정제 (필요시 추가)
    """
    if not isinstance(text, str):
        logger.warning(f"Input to clean_raw_text is not a string: {type(text)}. Returning empty string.")
        return ""

    # 1. 앞뒤 공백 제거
    cleaned_text = text.strip()

    # 2. 여러 개의 연속된 공백 (스페이스, 탭, 줄바꿈 등 \s에 포함되는 모든 문자)을 하나로 줄임
    # 이 과정에서 기존 줄바꿈이 모두 공백으로 바뀝니다. 필요에 따라 \s를 [ \t]+ (공백과 탭만) 등으로 변경할 수 있습니다.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 3. 특정 제어 문자나 불필요한 유니코드 문자 제거 (선택 사항, 필요시 추가)
    # 예: ASCII가 아닌 제어 문자 제거
    # cleaned_text = ''.join(c for c in cleaned_text if c.isprintable() or c in ('\n', '\r', '\t'))

    # 4. HTML 엔티티 디코딩 (BeautifulSoup 파싱 후 대부분 처리되지만, Raw text나 다른 소스 시 필요할 수 있음)
    # 예: &amp; -> &, &lt; -> < 변환 등
    # from html import unescape
    # cleaned_text = unescape(cleaned_text)


    logger.debug(f"Cleaned raw text (Original length: {len(text)}, Cleaned length: {len(cleaned_text)})") # 디버그 레벨 로깅

    return cleaned_text.strip() # 최종적으로 다시 앞뒤 공백 제거 (혹시 모를 경우 대비)