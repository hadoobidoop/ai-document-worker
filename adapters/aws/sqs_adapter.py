# adapters/aws/sqs_adapter.py

import boto3
import json
import logging
# import os # os 모듈 직접 사용 불필요

# config.settings에서 표준화된 SQS_QUEUE_URL 임포트
from config.settings import SQS_QUEUE_URL

logger = logging.getLogger(__name__)

# SQS 클라이언트 초기화
sqs_client = boto3.client('sqs')

# SQS_QUEUE_URL이 config.settings를 통해 로드되었는지 확인
if not SQS_QUEUE_URL:
    logger.error("SQS_QUEUE_URL이 config.settings를 통해 설정되지 않았습니다. SQS 메시지 발행이 불가능합니다.")
    # 실제 운영 환경에서는 애플리케이션 시작 시점에서 오류를 발생시켜 배포 실패로 이어지도록 하는 것이 좋습니다.
    # raise EnvironmentError("SQS_QUEUE_URL not configured via config.settings (expected from AI_DOC_PROCESSING_SQS_URL env var).")


def publish_analysis_request(message_body: dict) -> str:
    """
    AI 분석 요청 메시지를 SQS 큐에 발행합니다.

    Args:
        message_body: SQS 메시지에 담을 딕셔너리 형태의 본문.
                      이 딕셔너리는 JSON 문자열로 변환되어 SQS에 전송됩니다.

    Returns:
        성공 시 SQS 메시지 ID (str). 발행 실패 시 예외 발생.
    """
    if not SQS_QUEUE_URL: # 함수 호출 시점에서도 SQS_QUEUE_URL 유효성 재확인
        logger.error("SQS_QUEUE_URL이 설정되지 않아 메시지를 발행할 수 없습니다.")
        raise EnvironmentError("SQS_QUEUE_URL is not configured. Cannot publish message.")

    if not isinstance(message_body, dict):
        logger.error(f"메시지 본문이 딕셔너리가 아닙니다: {type(message_body)}. 메시지 발행 실패.")
        raise ValueError("SQS message body must be a dictionary.")

    try:
        message_body_json = json.dumps(message_body)

        if len(message_body_json.encode('utf-8')) > 256 * 1024:
            logger.error(f"SQS 메시지 크기 초과: {len(message_body_json.encode('utf-8'))} bytes.")
            raise ValueError("SQS message body exceeds the 256KB limit.")

        logger.info(f"SQS 메시지 발행 시도: 큐={SQS_QUEUE_URL}")
        # logger.debug(f"SQS 메시지 본문: {message_body_json}") # 필요시 로깅 (민감 정보 주의)

        response = sqs_client.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=message_body_json
        )

        message_id = response.get('MessageId')
        if message_id:
            logger.info(f"SQS 메시지 발행 성공. 메시지 ID: {message_id}")
            return message_id
        else:
            logger.error("SQS send_message 호출 성공, but MessageId가 반환되지 않음.")
            raise RuntimeError("SQS send_message failed to return MessageId.")

    except Exception as e:
        logger.error(f"SQS 메시지 발행 실패: 큐={SQS_QUEUE_URL}, 오류={e}", exc_info=True)
        raise
