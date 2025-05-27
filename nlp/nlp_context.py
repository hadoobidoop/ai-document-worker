# nlp/nlp_context.py
from typing import Optional
# konlpy.tag에서 Okt를 임포트할 수 있지만, 실제 인스턴스는 worker에서 주입받으므로 타입 힌팅용으로만 사용
from konlpy.tag import Okt as OktType # 타입 힌팅을 위해 OktType으로 별칭 사용

# 공유될 Okt 토크나이저 인스턴스
# 이 변수는 worker.py에서 초기화 후 할당됩니다.
shared_okt_instance: Optional[OktType] = None

# konlpy 사용 가능 여부 플래그 (worker.py에서 설정 가능)
konlpy_available_for_nlp: bool = False
