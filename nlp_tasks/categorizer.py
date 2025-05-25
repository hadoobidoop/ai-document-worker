# analysis_worker_app/nlp_tasks/categorizer.py
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# konlpy Okt 임포트
from konlpy.tag import Okt # Okt가 시스템에 설치되어 있어야 함 (Java 필요)

logger = logging.getLogger(__name__)

# --- 카테고리 정의 (이전과 동일, 15개 카테고리) ---
CATEGORY_DEFINITIONS = {
    "뉴스/시사": ["속보 주요뉴스 헤드라인 브리핑 오늘의소식 사건사고 논평 칼럼 인터뷰", ...],
    "IT/기술": ["AI 인공지능 머신러닝 딥러닝 빅데이터 클라우드 SaaS PaaS IaaS API", ...],
    "경제/금융": ["주식 코스피 코스닥 나스닥 증권 시황 분석 투자전략 재테크 부동산 시장 전망", ...],
    "경영/비즈니스": ["기업전략 마케팅광고 브랜딩 시장조사 소비자분석 고객경험 CX UX UI", ...],
    "학문/교육": ["논문 학술지 연구자료 연구보고서 학회 컨퍼런스 세미나 강연 발표자료", ...],
    "건강/의학": ["질병예방 건강검진 치료법 의학정보 의료뉴스 병원 의사 약사 간호사", ...],
    "여행/레저": ["국내여행 해외여행 추천코스 관광지 명소 맛집 탐방 숙소 호텔 리조트 항공권", ...],
    "문화/예술": ["영화리뷰 영화추천 독립영화 영화제 OTT서비스 드라마 예능 방송프로그램", ...],
    "패션/뷰티": ["패션트렌드 스타일링 코디법 의류 신발 가방 액세서리 쇼핑정보 브랜드", ...],
    "음식/요리": ["레시피 요리법 집밥 밑반찬 간편요리 베이킹 디저트 음료 커피 와인", ...],
    "생활/리빙": ["인테리어 가구 홈데코 홈스타일링 DIY가구 집꾸미기 미니멀라이프", ...],
    "자기계발": ["생산성향상 시간관리 목표설정 습관만들기 GTD 업무효율 커뮤니케이션 스킬", ...],
    "자동차/교통": ["자동차리뷰 신차 중고차 전기차 자율주행 자동차정비 튜닝 세차 차량관리", ...],
    "환경/기후": ["기후변화 지구온난화 탄소중립 ESG 환경오염 미세먼지 대기 수질 토양", ...],
    "기타/커뮤니티": ["유머 웃긴글 짤방 밈 인터넷커뮤니티 게시판 Q&A 질문답변", ...]
} # 각 카테고리별 키워드는 이전 답변의 상세 목록을 사용해주세요.

# --- TF-IDF Vectorizer, 카테고리 벡터, Okt 객체 (워커 시작 시 초기화) ---
vectorizer_instance: TfidfVectorizer | None = None
category_names_list: list[str] = []
category_vectors_matrix: np.ndarray | None = None # scipy.sparse.csr_matrix일 수도 있음
okt_tokenizer: Okt | None = None # Okt 객체

def _korean_tokenizer_okt(text: str) -> list[str]:
    """
    konlpy의 Okt 형태소 분석기를 사용하여 명사를 추출하는 토크나이저.
    """
    global okt_tokenizer
    if okt_tokenizer is None:
        # 이 경우는 initialize_categorizer가 먼저 호출되지 않았다는 의미이므로,
        # 사실상 오류 상황이거나, 매우 드문 경우에 대한 방어 코드.
        logger.warning("Okt tokenizer not initialized. Attempting to initialize now (this should ideally not happen here).")
        try:
            okt_tokenizer = Okt()
        except Exception as e:
            logger.error(f"Failed to initialize Okt tokenizer in _korean_tokenizer_okt: {e}")
            # Okt 초기화 실패 시 기본 공백 토크나이저로 fallback (품질 저하)
            return text.split()

    try:
        # 명사만 추출하거나, 필요시 다른 품사(형용사 등)도 포함 가능
        # 예: nouns = okt_tokenizer.nouns(text)
        # 예: morphs = okt_tokenizer.morphs(text, stem=True) # 어간 추출
        # 여기서는 간단히 명사만 추출하고, 길이가 2 이상인 단어만 사용
        nouns = okt_tokenizer.nouns(text)
        meaningful_nouns = [noun for noun in nouns if len(noun) > 1]
        if not meaningful_nouns and nouns : # 2글자 이상 명사가 없으면 1글자 명사라도 사용
            meaningful_nouns = nouns
        # logger.debug(f"Tokenized with Okt (nouns > 1 char): {meaningful_nouns}")
        return meaningful_nouns
    except Exception as e:
        logger.error(f"Error during Okt tokenization for text '{text[:50]}...': {e}")
        return text.split() # 오류 발생 시 기본 공백 토크나이저로 fallback


def initialize_categorizer():
    global vectorizer_instance, category_names_list, category_vectors_matrix, okt_tokenizer
    if vectorizer_instance is not None:
        return

    logger.info("Initializing TF-IDF categorizer with Okt for Korean tokenization...")
    try:
        # Okt 초기화 (Java 경로 등 환경 설정이 필요할 수 있음)
        if okt_tokenizer is None: # Okt 객체가 아직 없다면 생성
            logger.info("Initializing Okt tokenizer instance...")
            okt_tokenizer = Okt()
            logger.info("Okt tokenizer instance created successfully.")

        vectorizer_instance = TfidfVectorizer(
            tokenizer=_korean_tokenizer_okt, # Okt 토크나이저 사용
            min_df=2,              # 최소 2개 카테고리 대표 텍스트에서 등장하는 단어만 고려 (조정 가능)
            ngram_range=(1,2),     # 유니그램 및 바이그램(연속된 두 단어) 고려
            stop_words=None        # 필요시 한국어 불용어 사전 전달 ['은', '는', '이', '가', ...]
        )

        temp_category_names = []
        corpus_for_categories = []

        for cat_name, representative_texts in CATEGORY_DEFINITIONS.items():
            combined_text = " ".join(representative_texts)
            corpus_for_categories.append(combined_text)
            temp_category_names.append(cat_name)

        if not corpus_for_categories:
            logger.warning("No category definitions found. Categorizer might not work.")
            return

        category_vectors_matrix = vectorizer_instance.fit_transform(corpus_for_categories)
        category_names_list = temp_category_names
        logger.info(f"TF-IDF categorizer initialized with {len(category_names_list)} categories. Vocabulary size: {len(vectorizer_instance.vocabulary_)}")

    except Exception as e:
        logger.error(f"Failed to initialize TF-IDF categorizer: {e}", exc_info=True)
        vectorizer_instance = None
        okt_tokenizer = None


def classify_text_tfidf(text_to_classify: str, similarity_threshold: float = 0.05) -> str:
    if vectorizer_instance is None or category_vectors_matrix is None or not category_names_list:
        logger.error("Categorizer is not properly initialized. Returning '미분류'.")
        # initialize_categorizer() # 워커 시작 시 초기화되므로, 여기서 재호출은 보통 불필요
        return "미분류"

    if not text_to_classify or not text_to_classify.strip():
        logger.info("Text to classify is empty. Returning '미분류'.")
        return "미분류"

    try:
        text_vector = vectorizer_instance.transform([text_to_classify])
        similarities = cosine_similarity(text_vector, category_vectors_matrix)

        if similarities.size == 0:
            logger.warning("Could not compute similarities for the given text.")
            return "미분류"

        best_match_index = np.argmax(similarities[0])
        best_similarity_score = similarities[0][best_match_index]

        logger.info(f"Text classification: Best match '{category_names_list[best_match_index]}' with score {best_similarity_score:.4f}")

        if best_similarity_score >= similarity_threshold:
            return category_names_list[best_match_index]
        else:
            logger.info(f"Best similarity score {best_similarity_score:.4f} is below threshold {similarity_threshold}. Assigning to '기타'.")
            return "기타"
    except Exception as e:
        logger.error(f"Error during TF-IDF classification for text '{text_to_classify[:50]}...': {e}", exc_info=True)
        return "미분류"

