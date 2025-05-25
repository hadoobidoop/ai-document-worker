# ingestion_lambda/adapters/sqs_publisher.py

import boto3
import json
import logging
import os

logger = logging.getLogger()
# 로거 레벨은 ingestion_lambda/handler.py 등에서 설정되었거나 Lambda 환경 설정 따름

# SQS 클라이언트 초기화 (Lambda 실행 환경 재사용을 위해 전역으로 선언)
# AWS SDK는 Lambda 환경에서 AWS 자격 증명 및 리전을 자동으로 찾습니다.
sqs_client = boto3.client('sqs')

# AI 분석 SQS 큐 URL은 환경 변수에서 가져오는 것을 권장
# 이 Lambda 함수의 환경 변수에 AI_ANALYSIS_SQS_QUEUE_URL을 설정해야 합니다.
AI_ANALYSIS_SQS_QUEUE_URL = os.environ.get('AI_ANALYSIS_SQS_QUEUE_URL')
if not AI_ANALYSIS_SQS_QUEUE_URL:
    logger.error("환경 변수 AI_ANALYSIS_SQS_QUEUE_URL이 설정되지 않았습니다. SQS 메시지 발행을 수행할 수 없습니다.")
    # 실제 운영 환경에서는 이 경우 심각한 오류로 처리해야 합니다. 배포 전 확인 필수.
    # raise EnvironmentError("AI_ANALYSIS_SQS_QUEUE_URL environment variable not set.") # 배포 파이프라인에서 체크하도록 할 수도 있습니다.


def publish_analysis_request(message_body: dict) -> str:
    """
    AI 분석 요청 메시지를 SQS 큐에 발행합니다.

    Args:
        message_body: SQS 메시지에 담을 딕셔너리 형태의 본문.
                      이 딕셔너리는 JSON 문자열로 변환되어 SQS에 전송됩니다.
                      (예: {'s3_path': '...', 'original_source_type': '...', ...})

    Returns:
        성공 시 SQS 메시지 ID (str). 발행 실패 시 예외 발생.
    """
    # 환경 변수 미설정 시 오류 체크
    if not AI_ANALYSIS_SQS_QUEUE_URL:
        logger.error("AI 분석 SQS 큐 URL이 구성되지 않아 메시지 발행 실패.")
        raise EnvironmentError("AI_ANALYSIS_SQS_QUEUE_URL environment variable not set.")

    if not isinstance(message_body, dict):
        logger.error(f"메시지 본문이 딕셔너리가 아닙니다: {type(message_body)}. 메시지 발행 실패.")
        raise ValueError("SQS message body must be a dictionary.")

    try:
        # 딕셔너리를 JSON 문자열로 변환하여 SQS 메시지 본문으로 사용
        message_body_json = json.dumps(message_body)

        # SQS 메시지 크기 제한 (256KB) 고려 필요.
        # 우리 설계에서는 클린 텍스트 자체는 S3에 저장하고 SQS에는 S3 경로와 메타데이터만 담으므로
        # 메시지 크기 제한에 걸릴 가능성은 매우 낮습니다.
        if len(message_body_json.encode('utf-8')) > 256 * 1024:
            logger.error(f"SQS 메시지 크기 초과: {len(message_body_json.encode('utf-8'))} bytes.")
            raise ValueError("SQS message body exceeds the 256KB limit.")


        logger.info(f"SQS 메시지 발행 시도: 큐={AI_ANALYSIS_SQS_QUEUE_URL}")
        # logger.debug(f"SQS 메시지 본문: {message_body_json}") # 상세 본문 로깅 (민감 정보 주의)

        response = sqs_client.send_message(
            QueueUrl=AI_ANALYSIS_SQS_QUEUE_URL,
            MessageBody=message_body_json
        )

        message_id = response.get('MessageId')
        if message_id:
            logger.info(f"SQS 메시지 발행 성공. 메시지 ID: {message_id}")
            return message_id
        else:
            # send_message 호출 자체는 성공했지만 MessageId가 반환되지 않은 경우
            logger.error("SQS send_message 호출 성공, but MessageId가 반환되지 않음.")
            raise RuntimeError("SQS send_message failed to return MessageId.")


    except Exception as e:
        # Boto3 SQS 관련 예외 발생 시
        logger.error(f"SQS 메시지 발행 실패: 큐={AI_ANALYSIS_SQS_QUEUE_URL}, 오류={e}", exc_info=True)
        # 발행 실패 시 예외를 다시 발생시켜 호출자(orchestrator)에게 알림
        raise # 예외 다시 발생