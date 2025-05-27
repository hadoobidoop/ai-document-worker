# analysis_worker_app/nlp_tasks/tag_extractor.py
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

try:
    from konlpy.tag import Okt
    konlpy_available = True
except ImportError:
    konlpy_available = False
    Okt = None

logger = logging.getLogger(__name__)

okt_tokenizer_instance: Okt | None = None

def initialize_tag_extractor_components(use_konlpy_okt: bool = True):
    global okt_tokenizer_instance
    if use_konlpy_okt and konlpy_available and okt_tokenizer_instance is None:
        logger.info("Initializing Okt tokenizer for KeyBERT candidate extraction...")
        try:
            okt_tokenizer_instance = Okt()
            logger.info("Okt tokenizer initialized successfully for tag extraction.")
        except Exception as e:
            logger.error(f"Failed to initialize Okt tokenizer: {e}", exc_info=True)
            okt_tokenizer_instance = None
    elif not konlpy_available and use_konlpy_okt:
        logger.warning("konlpy is not available or Okt could not be imported. Korean noun extraction for KeyBERT will be skipped.")
    else:
        logger.info("Okt tokenizer initialization skipped (not requested or already done).")

def _format_as_hashtags(keywords_with_scores: list[tuple[str, float]], num_tags: int) -> list[str]:
    # (이전 답변과 동일한 _format_as_hashtags 함수 내용)
    hashtags = []
    if not keywords_with_scores:
        return []

    for keyword_phrase, score in keywords_with_scores:
        tag_candidate = "".join(keyword_phrase.split())
        if not tag_candidate:
            continue
        formatted_tag = f"#{tag_candidate}"
        if formatted_tag not in hashtags:
            hashtags.append(formatted_tag)
        if len(hashtags) >= num_tags:
            break
    return hashtags

def extract_hashtags_with_keybert(
        text_content: str,
        embedding_model: SentenceTransformer,
        num_tags: int = 3,
        use_korean_noun_extraction_if_available: bool = True,
        language_hint: str | None = None # 예: 'ko', 'en'
) -> list[str]:
    if not text_content or not text_content.strip():
        logger.info("Text content is empty. No hashtags to extract.")
        return []
    if embedding_model is None:
        logger.error("Embedding model not provided to KeyBERT. Cannot extract hashtags.")
        return []

    try:
        kw_model = KeyBERT(model=embedding_model)
        candidates = None

        # --- 한국어 명사 추출 로직 시작 ---
        is_korean_text_for_noun_extraction = False
        if use_korean_noun_extraction_if_available and okt_tokenizer_instance:
            if language_hint == 'ko':
                is_korean_text_for_noun_extraction = True
                logger.debug("Language hint is 'ko'. Proceeding with Korean noun extraction.")
            elif language_hint is None: # 언어 힌트가 없을 경우, 내용 기반으로 추정
                # 간단한 한글 포함 여부로 한국어 추정 (텍스트의 처음 200자 검사)
                # 좀 더 정확하려면 외부에서 langdetect 등을 사용한 결과를 language_hint로 전달하는 것이 좋음
                if any('\uAC00' <= char <= '\uD7A3' for char in text_content[:200]):
                    is_korean_text_for_noun_extraction = True
                    logger.debug("No language hint, but Hangul detected. Proceeding with Korean noun extraction.")
                else:
                    logger.debug("No language hint and no Hangul detected in initial part. Skipping Korean noun extraction.")
            # language_hint가 'en' 등으로 명시적으로 주어지면 is_korean_text_for_noun_extraction은 False 유지
            else:
                logger.debug(f"Language hint is '{language_hint}'. Skipping Korean noun extraction.")


            if is_korean_text_for_noun_extraction:
                logger.info("Attempting Korean noun extraction for KeyBERT candidates.")
                try:
                    nouns = okt_tokenizer_instance.nouns(text_content)
                    if nouns:
                        # 명사 중에서도 너무 짧거나 일반적인 단어 제외 로직 추가 가능
                        # 예: KOREAN_COMMON_NOUN_STOPWORDS = {"것", "수", "때", "나", "너", "우리"}
                        candidates = [n for n in list(set(nouns)) if len(n) > 1] # 1글자 명사 제외 및 중복 제거
                        if candidates:
                            logger.info(f"Extracted {len(candidates)} unique noun candidates (longer than 1 char) for KeyBERT.")
                            logger.debug(f"Noun candidates preview (first 10): {candidates[:10]}")
                        else:
                            logger.info("No suitable noun candidates (longer than 1 char) found after Okt extraction.")
                    else:
                        logger.info("Okt noun extraction returned no nouns.")
                except Exception as e_okt:
                    logger.warning(f"Okt noun extraction failed: {e_okt}. Proceeding without candidates.", exc_info=True)
                    candidates = None # Okt 실패 시 후보군 없이 진행 (KeyBERT가 전체 텍스트 사용)
        elif use_korean_noun_extraction_if_available and okt_tokenizer_instance is None:
            logger.warning("Korean noun extraction requested, but Okt tokenizer is not initialized. Skipping.")
        # --- 한국어 명사 추출 로직 끝 ---

        # KeyBERT로 키워드 추출
        # 영어의 경우 'english' 불용어 사용, 그 외 (한국어 포함)는 None 또는 사용자 정의 불용어 사용
        # language_hint가 'ko'이고 candidates가 있다면, stop_words는 크게 중요하지 않을 수 있음 (명사만 사용하므로)
        current_stop_words = 'english' if language_hint == 'en' else None

        keywords_with_scores = kw_model.extract_keywords(
            docs=text_content,
            keyphrase_ngram_range=(1, 2),
            stop_words=current_stop_words,
            use_mmr=True, diversity=0.7,
            top_n=num_tags * 3, # 요청 개수보다 넉넉하게 뽑아서 후처리에서 선택
            candidates=candidates # 명사 추출 사용 시 여기에 후보군 전달, None이면 KeyBERT가 알아서 처리
        )

        if not keywords_with_scores:
            logger.info("KeyBERT did not extract any keywords.")
            return []

        logger.info(f"KeyBERT extracted raw keywords (with scores): {keywords_with_scores}")

        hashtags = _format_as_hashtags(keywords_with_scores, num_tags)
        logger.info(f"Formatted hashtags for document: {hashtags}")
        return hashtags

    except Exception as e:
        logger.error(f"Error during KeyBERT hashtag extraction: {e}", exc_info=True)
        return []