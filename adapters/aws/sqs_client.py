import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from typing import List, Dict, Optional

# 로깅 설정 (worker.py와 동일한 포맷 사용 또는 별도 설정)
logger = logging.getLogger(__name__)

class SQSAdapter:
    def __init__(self, queue_url: str, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, region_name: Optional[str] = None):
        self.queue_url = queue_url
        try:
            self.sqs = boto3.client(
                'sqs',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            logger.info(f"SQSAdapter initialized for queue: {queue_url} in region: {region_name if region_name else 'default'}")
        except Exception as e:
            logger.error(f"Failed to initialize SQS client: {e}", exc_info=True)
            raise

    def send_message(self, message_body: str, message_attributes: Optional[Dict] = None) -> Dict:
        try:
            response = self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message_body,
                MessageAttributes=message_attributes or {}
            )
            logger.info(f"Message sent to SQS queue {self.queue_url}. Message ID: {response.get('MessageId')}")
            return response
        except NoCredentialsError:
            logger.error("AWS credentials not found.")
            raise
        except PartialCredentialsError:
            logger.error("Incomplete AWS credentials.")
            raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            error_message = e.response.get("Error", {}).get("Message")
            logger.error(f"Failed to send message to SQS. Error Code: {error_code}. Message: {error_message}. Full error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending message to SQS: {e}", exc_info=True)
            raise

    def receive_messages(self, max_number_of_messages: int = 1, wait_time_seconds: int = 0, visibility_timeout: int = 30) -> List[Dict]:
        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                AttributeNames=['All'], # 모든 속성 가져오기
                MessageAttributeNames=['All'], # 모든 메시지 속성 가져오기
                MaxNumberOfMessages=max_number_of_messages,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=visibility_timeout # 메시지 처리 예상 시간으로 설정
            )
            messages = response.get('Messages', [])
            if messages:
                logger.info(f"Received {len(messages)} message(s) from SQS queue {self.queue_url}")
            return messages
        except NoCredentialsError:
            logger.error("AWS credentials not found.")
            raise
        except PartialCredentialsError:
            logger.error("Incomplete AWS credentials.")
            raise
        except ClientError as e: # receive_message 호출 시 발생할 수 있는 ClientError 처리
            error_code = e.response.get("Error", {}).get("Code")
            error_message = e.response.get("Error", {}).get("Message")
            logger.error(f"Failed to receive messages from SQS. Error Code: {error_code}. Message: {error_message}. Full error: {e}", exc_info=True)
            raise # main loop에서 이 예외를 처리하도록 다시 발생시킴
        except Exception as e:
            logger.error(f"An unexpected error occurred while receiving messages from SQS: {e}", exc_info=True)
            raise

    def delete_message(self, receipt_handle: str):
        if not receipt_handle:
            logger.error("Receipt handle is required to delete a message but was not provided.")
            # 이 경우 예외를 발생시켜 호출자가 문제를 인지하도록 할 수 있습니다.
            # raise ValueError("Receipt handle cannot be None or empty.")
            return # 또는 단순히 반환하여 아무 작업도 하지 않음 (호출자 로직에 따라 결정)

        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info(f"Message with receipt handle {receipt_handle} deleted successfully from SQS queue {self.queue_url}.")
        except NoCredentialsError:
            logger.error("AWS credentials not found.")
            raise
        except PartialCredentialsError:
            logger.error("Incomplete AWS credentials.")
            raise
        except ClientError as e:
            # ClientError 발생 시 더 자세한 정보 로깅
            error_code = e.response.get("Error", {}).get("Code")
            error_message = e.response.get("Error", {}).get("Message")
            logger.error(f"Failed to delete message with receipt_handle {receipt_handle}. Error Code: {error_code}. Message: {error_message}. Full error: {e}", exc_info=True)
            # 특정 오류 코드에 따라 다른 동작을 수행할 수 있습니다.
            # 예: if error_code == 'ReceiptHandleIsInvalid': logger.warning("Attempted to delete with an invalid receipt handle.")
            raise # 예외를 다시 발생시켜 호출한 쪽에서 처리할 수 있도록 함
        except Exception as e:
            logger.error(f"An unexpected error occurred while deleting message {receipt_handle}: {e}", exc_info=True)
            raise
