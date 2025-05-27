# analysis_worker_app/nlp_tasks/categorizer.py
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json # JSON 파일을 읽기 위해 추가
from pathlib import Path # 파일 경로 처리를 위해 추가

import config
from nlp import nlp_context

logger = logging.getLogger(__name__)

vectorizer_instance: TfidfVectorizer | None = None
category_names_list: list[str] = []
category_vectors_matrix: np.ndarray | None = None

def _load_category_definitions_from_file(file_path_str: str) -> dict:
    """
    지정된 경로의 JSON 파일에서 카테고리 정의를 로드합니다.
    파일이 없거나 JSON 파싱 오류 발생 시 빈 딕셔너리를 반환하고 오류를 로깅합니다.
    """
    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.error(f"Category definition file not found at: {file_path_str}. Returning empty definitions.")
        return {}
    try:
        with file_path.open('r', encoding='utf-8') as f:
            definitions = json.load(f)
        logger.info(f"Successfully loaded category definitions from {file_path_str}")
        return definitions
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from category file {file_path_str}: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading category file {file_path_str}: {e}", exc_info=True)
        return {}

def _korean_tokenizer_okt(text: str) -> list[str]:
    current_okt_instance = nlp_context.shared_okt_instance
    if not nlp_context.konlpy_available_for_nlp or current_okt_instance is None:
        logger.warning("Shared Okt tokenizer is not available for categorizer. Falling back to space tokenizer.")
        return text.split()
    try:
        nouns = current_okt_instance.nouns(text)
        meaningful_nouns = [noun for noun in nouns if len(noun) > 1]
        if not meaningful_nouns and nouns:
            meaningful_nouns = nouns
        return meaningful_nouns
    except Exception as e:
        logger.error(f"Error during Okt tokenization for text '{text[:50]}...': {e}", exc_info=True)
        return text.split()


def initialize_categorizer():
    global vectorizer_instance, category_names_list, category_vectors_matrix

    if vectorizer_instance is not None:
        logger.info("TF-IDF categorizer is already initialized.")
        return

    logger.info("Initializing TF-IDF categorizer...")

    # config.CATEGORIES_FILE_PATH에서 카테고리 정의 로드
    category_definitions = _load_category_definitions_from_file(config.CATEGORIES_FILE_PATH)

    if not category_definitions:
        logger.error("No category definitions loaded. Categorizer initialization cannot proceed.")
        return # 카테고리 정의가 없으면 초기화 중단

    try:
        vectorizer_instance = TfidfVectorizer(
            tokenizer=_korean_tokenizer_okt,
            min_df=2,
            ngram_range=(1,2),
            stop_words=None
        )

        temp_category_names = []
        corpus_for_categories = []

        for cat_name, representative_texts in category_definitions.items():
            if isinstance(representative_texts, list) and all(isinstance(text, str) for text in representative_texts):
                combined_text = " ".join(representative_texts)
                corpus_for_categories.append(combined_text)
                temp_category_names.append(cat_name)
            else:
                logger.warning(f"Invalid format for category '{cat_name}' in definitions. Skipping this category. Expected a list of strings.")


        if not corpus_for_categories:
            logger.warning("No valid category data to process after parsing definitions. Categorizer might not work.")
            return

        category_vectors_matrix = vectorizer_instance.fit_transform(corpus_for_categories)
        category_names_list = temp_category_names
        logger.info(f"TF-IDF categorizer initialized with {len(category_names_list)} categories. Vocabulary size: {len(vectorizer_instance.vocabulary_)}")

    except Exception as e:
        logger.error(f"Failed to initialize TF-IDF categorizer: {e}", exc_info=True)
        vectorizer_instance = None


def classify_text_tfidf(text_to_classify: str, similarity_threshold: float | None = None) -> str:
    if vectorizer_instance is None or category_vectors_matrix is None or not category_names_list:
        logger.error("Categorizer is not properly initialized. Returning '미분류'.")
        return "미분류"

    if not text_to_classify or not text_to_classify.strip():
        logger.info("Text to classify is empty. Returning '미분류'.")
        return "미분류"

    final_similarity_threshold = similarity_threshold if similarity_threshold is not None else config.CATEGORIZER_SIMILARITY_THRESHOLD

    try:
        text_vector = vectorizer_instance.transform([text_to_classify])
        similarities = cosine_similarity(text_vector, category_vectors_matrix)

        if similarities.size == 0:
            logger.warning("Could not compute similarities for the given text.")
            return "미분류"

        best_match_index = np.argmax(similarities[0])
        best_similarity_score = similarities[0][best_match_index]

        logger.info(f"Text classification: Best match '{category_names_list[best_match_index]}' with score {best_similarity_score:.4f}")

        if best_similarity_score >= final_similarity_threshold:
            return category_names_list[best_match_index]
        else:
            logger.info(f"Best similarity score {best_similarity_score:.4f} is below threshold {final_similarity_threshold}. Assigning to '기타'.")
            return "기타" # '기타'는 JSON 파일에 정의되어 있지 않아도 여기서 반환 가능
    except Exception as e:
        logger.error(f"Error during TF-IDF classification for text '{text_to_classify[:50]}...': {e}", exc_info=True)
        return "미분류"
