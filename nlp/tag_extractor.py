# analysis_worker_app/nlp_tasks/tag_extractor.py
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

import config
from nlp import nlp_context # nlp_context 임포트

logger = logging.getLogger(__name__)


def initialize_tag_extractor_components(use_konlpy_okt: bool = True):
    """
    태그 추출기 컴포넌트 초기화 (주로 로깅 및 상태 확인용).
    Okt 인스턴스 자체는 worker.py에서 nlp_context를 통해 공유됩니다.
    """
    if use_konlpy_okt:
        if nlp_context.konlpy_available_for_nlp and nlp_context.shared_okt_instance:
            logger.info("Shared Okt tokenizer is available and will be used by KeyBERT if applicable for Korean text.")
        elif nlp_context.konlpy_available_for_nlp and not nlp_context.shared_okt_instance:
            logger.warning("Konlpy is available, but shared Okt instance is not initialized. Korean noun extraction for KeyBERT might be skipped or fail.")
        elif not nlp_context.konlpy_available_for_nlp:
            logger.warning("Konlpy is not available. Korean noun extraction for KeyBERT will be skipped.")
    else:
        logger.info("Okt tokenizer usage not requested for tag extraction.")


def _format_as_hashtags(keywords_with_scores: list[tuple[str, float]], num_tags: int) -> list[str]:
    # (기존과 동일)
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
        num_tags: int | None = None,
        use_korean_noun_extraction_if_available: bool = True, # 이 파라미터는 worker에서 nlp_context.konlpy_available_for_nlp 값으로 전달받음
        language_hint: str | None = None
) -> list[str]:
    if not text_content or not text_content.strip():
        logger.info("Text content is empty. No hashtags to extract.")
        return []
    if embedding_model is None:
        logger.error("Embedding model not provided to KeyBERT. Cannot extract hashtags.")
        return []

    final_num_tags = num_tags if num_tags is not None else config.TAG_EXTRACTOR_NUM_TAGS
    current_okt_instance = nlp_context.shared_okt_instance # 공유 인스턴스 사용

    try:
        kw_model = KeyBERT(model=embedding_model)
        candidates = None
        is_korean_text_for_noun_extraction = False

        # use_korean_noun_extraction_if_available 플래그와 공유 Okt 인스턴스 상태 모두 확인
        if use_korean_noun_extraction_if_available and nlp_context.konlpy_available_for_nlp and current_okt_instance:
            if language_hint == 'ko':
                is_korean_text_for_noun_extraction = True
            elif language_hint is None:
                if any('\uAC00' <= char <= '\uD7A3' for char in text_content[:200]):
                    is_korean_text_for_noun_extraction = True
            # (기존 언어 감지 로직 유지)

            if is_korean_text_for_noun_extraction:
                logger.info("Attempting Korean noun extraction for KeyBERT candidates using shared Okt instance.")
                try:
                    nouns = current_okt_instance.nouns(text_content) # 공유 인스_instance 사용
                    if nouns:
                        candidates = [n for n in list(set(nouns)) if len(n) > 1]
                        if candidates:
                            logger.info(f"Extracted {len(candidates)} unique noun candidates for KeyBERT.")
                        else:
                            logger.info("No suitable noun candidates found after Okt extraction.")
                    else:
                        logger.info("Okt noun extraction returned no nouns.")
                except Exception as e_okt:
                    logger.warning(f"Okt noun extraction failed: {e_okt}. Proceeding without candidates.", exc_info=True)
                    candidates = None
        elif use_korean_noun_extraction_if_available and (not nlp_context.konlpy_available_for_nlp or not current_okt_instance):
            logger.warning("Korean noun extraction requested, but Konlpy/Okt is not available or not initialized. Skipping.")

        current_stop_words = 'english' if language_hint == 'en' else None
        keywords_with_scores = kw_model.extract_keywords(
            docs=text_content,
            keyphrase_ngram_range=(1, 2),
            stop_words=current_stop_words,
            use_mmr=True, diversity=0.7,
            top_n=final_num_tags * 3,
            candidates=candidates
        )

        if not keywords_with_scores:
            logger.info("KeyBERT did not extract any keywords.")
            return []

        logger.info(f"KeyBERT extracted raw keywords (with scores): {keywords_with_scores}")
        hashtags = _format_as_hashtags(keywords_with_scores, final_num_tags)
        logger.info(f"Formatted hashtags for document: {hashtags}")
        return hashtags

    except Exception as e:
        logger.error(f"Error during KeyBERT hashtag extraction: {e}", exc_info=True)
        return []
