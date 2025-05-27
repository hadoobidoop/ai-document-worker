# analysis_worker_app/worker.py

import logging
import signal
import time
import json
import os
import boto3
from botocore.exceptions import ClientError
from botocore.client import BaseClient # SQS 클라이언트 타입 힌팅을 위해 추가
from typing import Optional # Optional 타입 힌팅을 위해 추가

import config

from adapters.api import backend_api_client
from adapters.aws.s3_client import get_text_from_s3
from adapters.db import vector_store_adapter
from nlp import categorizer, embedding_generator, tag_extractor, summarizer
from nlp import nlp_context

try:
    from konlpy.tag import Okt
    konlpy_available = True
except ImportError:
    Okt = None
    konlpy_available = False
    logging.warning("Konlpy or Okt not found. Korean NLP features depending on Okt will be limited.")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# SQS 클라이언트, 전역 변수로 선언하고 타입 힌트 추가
sqs_client: Optional[BaseClient] = None
shutdown_flag = False
messages_processed_since_last_faiss_save = 0

def initialize_app():
    global sqs_client
    logger.info("=====================================================================")
    logger.info("               AI Analysis Worker Application Starting               ")
    logger.info("=====================================================================")

    logger.info(f"Attempting to initialize SQS client. Current SQS_QUEUE_URL from config: {config.SQS_QUEUE_URL}")
    if not config.SQS_QUEUE_URL:
        logger.critical("SQS_QUEUE_URL is not configured in config.py. Worker cannot start.")
        raise EnvironmentError("SQS_QUEUE_URL is required for the worker to run.")

    try:
        # boto3.client('sqs')는 BaseClient를 상속하는 SQS 클라이언트 객체를 반환합니다.
        sqs_client = boto3.client('sqs')
        if sqs_client:
            logger.info(f"SQS client initialized successfully. Type: {type(sqs_client)}. Object ID: {id(sqs_client)}")
        else:
            logger.error("boto3.client('sqs') returned None without raising an exception. This is unexpected.")
            sqs_client = None
            raise RuntimeError("SQS client could not be initialized (boto3.client returned None).")
    except Exception as e:
        logger.error(f"Failed to initialize SQS client with boto3: {e}", exc_info=True)
        sqs_client = None
        raise

    if sqs_client is None:
        logger.critical("SQS client is None after initialization attempt. Halting further initialization.")
        raise RuntimeError("SQS client initialization failed, client is None.")

    logger.info("Initializing AI modules and FAISS vector store...")
    nlp_context.konlpy_available_for_nlp = konlpy_available
    if konlpy_available and Okt is not None:
        logger.info("Attempting to initialize shared Okt tokenizer instance...")
        try:
            nlp_context.shared_okt_instance = Okt()
            logger.info("Shared Okt tokenizer instance initialized successfully.")
        except Exception as e_okt:
            logger.error(f"Failed to initialize shared Okt tokenizer: {e_okt}", exc_info=True)
            nlp_context.shared_okt_instance = None
            nlp_context.konlpy_available_for_nlp = False
    else:
        logger.warning("Konlpy (Okt) is not available or Okt class is None. Shared Okt instance will not be initialized.")
        nlp_context.shared_okt_instance = None

    categorizer.initialize_categorizer()
    embedding_generator.initialize_embedding_model()
    tag_extractor.initialize_tag_extractor_components(use_konlpy_okt=konlpy_available)
    vector_store_adapter.load_or_initialize_faiss_index()

    logger.info("---------------------------------------------------------------------")
    logger.info(f"FAISS save interval: {config.FAISS_SAVE_INTERVAL_MESSAGES} messages.")
    logger.info("               Application Initialization Complete                   ")
    logger.info("---------------------------------------------------------------------")

def _delete_sqs_message(receipt_handle: str, context_msg_id: str = "N/A"):
    """Helper function to delete SQS message with robust checks."""
    if not receipt_handle:
        logger.warning(f"Message (ID: {context_msg_id}): Receipt handle is missing, cannot delete message.")
        return

    # 이 시점에서 sqs_client는 BaseClient 타입이거나 None일 수 있습니다.
    # 아래 if 문에서 None이 아님을 확인하면, 그 이후에는 BaseClient 타입으로 간주할 수 있습니다.
    if sqs_client is None:
        logger.error(f"Message (ID: {context_msg_id}): SQS client is None. Cannot delete message. This indicates a critical issue with SQS client lifecycle.")
        return
    if not config.SQS_QUEUE_URL:
        logger.error(f"Message (ID: {context_msg_id}): SQS_QUEUE_URL is not configured. Cannot delete message.")
        return

    try:
        logger.debug(f"Message (ID: {context_msg_id}): Attempting to delete message from SQS. SQS Client Object ID: {id(sqs_client)}")
        # 이 시점에서 sqs_client는 None이 아님이 보장됩니다.
        sqs_client.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
        logger.info(f"Message (ID: {context_msg_id}): Successfully deleted from SQS.")
    except Exception as e_del:
        logger.error(f"Message (ID: {context_msg_id}): Failed to delete message from SQS: {e_del}", exc_info=True)


def process_message(message: dict) -> bool:
    global shutdown_flag, messages_processed_since_last_faiss_save
    if shutdown_flag:
        logger.info("Shutdown signal received. Halting new message processing.")
        return False

    receipt_handle = message.get('ReceiptHandle')
    message_id = message.get('MessageId', 'N/A')
    logger.info(f"Received message (ID: {message_id}). Starting processing pipeline.")
    logger.debug(f"Current SQS Client in process_message: Type={type(sqs_client)}, ID={id(sqs_client)}")


    document_id_for_log = "N/A"
    processing_status_for_backend = "PROCESSING_STARTED"
    error_messages_for_backend = []

    try:
        body_str = message.get('Body', '{}')
        message_body = json.loads(body_str)

        s3_path = message_body.get('s3_path')
        ingestion_details = message_body.get('ingestion_details', {})
        document_id = ingestion_details.get('document_id')
        user_id = ingestion_details.get('user_id')
        original_url = ingestion_details.get('source_identifier') if ingestion_details.get('source_type') == 'url' else None
        document_id_for_log = document_id

        if not s3_path or not document_id:
            logger.error(f"Message (ID: {message_id}) is missing s3_path or document_id. Body: {body_str}")
            _delete_sqs_message(receipt_handle, message_id)
            return True

        logger.info(f"Processing document_id: {document_id}, S3 Path: {s3_path}, User ID: {user_id}")

        text_content = get_text_from_s3(s3_path)
        if text_content is None:
            logger.error(f"Document ID {document_id}: Failed to load text from S3. Aborting processing for this message.")
            processing_status_for_backend = "TEXT_LOAD_FAILURE"
            error_messages_for_backend.append("Failed to load text from S3.")
            backend_api_client.save_analysis_results_to_backend(
                document_id, user_id, s3_path, original_url, None, None, None, None, 0,
                processing_status_for_backend, "; ".join(error_messages_for_backend)
            )
            _delete_sqs_message(receipt_handle, message_id)
            return True

        logger.info(f"Document ID {document_id}: Text loaded from S3. Length: {len(text_content)} chars.")
        processing_status_for_backend = "TEXT_LOADED_SUCCESSFULLY"

        short_summary, long_markdown_summary, assigned_category, extracted_hashtags = None, None, None, None
        num_embedded_chunks = 0

        # (NLP 처리 로직은 이전과 동일하게 유지)
        try:
            if config.GROQ_API_KEY:
                short_summary = summarizer.summarize_with_groq(text_content, summary_type="short", short_summary_max_tokens=config.SUMMARIZER_SHORT_SUMMARY_MAX_TOKENS)
                long_markdown_summary = summarizer.summarize_with_groq(text_content, summary_type="long_markdown", long_summary_target_chars=config.SUMMARIZER_LONG_SUMMARY_TARGET_CHARS, long_summary_max_tokens=config.SUMMARIZER_LONG_SUMMARY_MAX_TOKENS)
                logger.info(f"DocID {document_id}: Summaries generated.")
            else:
                logger.warning(f"DocID {document_id}: Groq API key not configured. Skipping summarization.")
                error_messages_for_backend.append("Summarization skipped: API key missing.")
        except Exception as e_sum:
            logger.error(f"DocID {document_id}: Summarization step failed: {e_sum}", exc_info=True)
            error_messages_for_backend.append(f"Summarization failed: {str(e_sum)[:100]}")

        try:
            assigned_category = categorizer.classify_text_tfidf(text_content, similarity_threshold=config.CATEGORIZER_SIMILARITY_THRESHOLD)
            logger.info(f"DocID {document_id}: Classified as '{assigned_category}'.")
        except Exception as e_cat:
            logger.error(f"DocID {document_id}: Categorization step failed: {e_cat}", exc_info=True)
            error_messages_for_backend.append(f"Categorization failed: {str(e_cat)[:100]}")

        try:
            if embedding_generator.embedding_model_instance:
                extracted_hashtags = tag_extractor.extract_hashtags_with_keybert(
                    text_content=text_content, embedding_model=embedding_generator.embedding_model_instance,
                    num_tags=config.TAG_EXTRACTOR_NUM_TAGS, use_korean_noun_extraction_if_available=nlp_context.konlpy_available_for_nlp)
                logger.info(f"DocID {document_id}: Extracted hashtags: {extracted_hashtags}")
            else:
                logger.warning(f"DocID {document_id}: Embedding model for KeyBERT not available. Skipping hashtag extraction.")
                error_messages_for_backend.append("Hashtag extraction skipped: Embedding model not ready.")
        except Exception as e_tag:
            logger.error(f"DocID {document_id}: Hashtag extraction step failed: {e_tag}", exc_info=True)
            error_messages_for_backend.append(f"Hashtag extraction failed: {str(e_tag)[:100]}")

        try:
            if vector_store_adapter.is_initialized() and embedding_generator.embedding_model_instance:
                embeddings_data = embedding_generator.generate_and_chunk_embeddings(text_content, document_id)
                if embeddings_data:
                    if vector_store_adapter.add_embeddings_to_faiss(embeddings_data):
                        num_embedded_chunks = len(embeddings_data)
                        logger.info(f"DocID {document_id}: Embeddings ({num_embedded_chunks} chunks) added to in-memory FAISS index.")
                        messages_processed_since_last_faiss_save += 1
                    else:
                        logger.error(f"DocID {document_id}: Failed to add embeddings to in-memory FAISS index.")
                        error_messages_for_backend.append("Failed to add embeddings to in-memory FAISS.")
                elif not embeddings_data:
                    logger.info(f"DocID {document_id}: No embeddings generated.")
                else:
                    logger.warning(f"DocID {document_id}: Embedding generation returned None.")
                    error_messages_for_backend.append("Embedding generation failed.")
            else:
                logger.warning(f"DocID {document_id}: FAISS index or embedding model not available. Skipping embedding.")
                error_messages_for_backend.append("Embedding skipped: FAISS index or model not ready.")
        except Exception as e_emb:
            logger.error(f"DocID {document_id}: Embedding/FAISS step failed: {e_emb}", exc_info=True)
            error_messages_for_backend.append(f"Embedding/FAISS step failed: {str(e_emb)[:100]}")


        if error_messages_for_backend:
            processing_status_for_backend = "PARTIAL_FAILURE" if processing_status_for_backend != "TEXT_LOAD_FAILURE" else processing_status_for_backend
        else:
            processing_status_for_backend = "COMPLETED"
        final_error_message = "; ".join(error_messages_for_backend) if error_messages_for_backend else None

        save_success = backend_api_client.save_analysis_results_to_backend(
            document_id=document_id, user_id=user_id, s3_path=s3_path, original_url=original_url,
            short_summary=short_summary, long_markdown_summary=long_markdown_summary, category=assigned_category,
            hashtags=extracted_hashtags, num_embedded_chunks=num_embedded_chunks,
            analysis_status=processing_status_for_backend, failure_reason=final_error_message
        )

        if not save_success:
            logger.error(f"CRITICAL: Failed to save final analysis results to backend for document_id: {document_id}.")
            return False

        logger.info(f"Message (ID: {message_id}, DocID: {document_id}) processed with status: {processing_status_for_backend}.")
        _delete_sqs_message(receipt_handle, message_id)
        return True

    except json.JSONDecodeError as e:
        logger.error(f"Message (ID: {message_id}) body JSON parsing failed: {e}. Body: {message.get('Body', '')}", exc_info=True)
        _delete_sqs_message(receipt_handle, message_id)
        return True
    except Exception as e:
        logger.error(f"CRITICAL unhandled error in process_message for DocID: {document_id_for_log}, Message ID: {message_id}. Error: {e}", exc_info=True)
        return False


def signal_handler(signum, frame):
    global shutdown_flag
    if not shutdown_flag:
        logger.info(f"Shutdown signal ({signal.Signals(signum).name}) received. Initiating graceful shutdown...")
        shutdown_flag = True
    else:
        logger.info("Shutdown already in progress.")

def main_loop():
    global shutdown_flag, messages_processed_since_last_faiss_save
    if sqs_client is None:
        logger.critical("SQS client is None at the start of main_loop. Worker cannot poll SQS. This indicates a failure in initialize_app.")
        return
    if not config.SQS_QUEUE_URL:
        logger.critical("SQS_QUEUE_URL not configured in config.py. Worker cannot poll SQS.")
        return

    logger.info(f"Starting SQS polling loop for queue: {config.SQS_QUEUE_URL}. SQS Client Object ID: {id(sqs_client)}")
    while not shutdown_flag:
        try:
            logger.debug(f"Polling SQS. SQS Client in loop: Type={type(sqs_client)}, ID={id(sqs_client)}")
            if sqs_client is None:
                logger.error("SQS client became None during main_loop. Stopping poll.")
                break

            response = sqs_client.receive_message(
                QueueUrl=config.SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
                VisibilityTimeout=300
            )
            messages = response.get('Messages', [])
            if not messages:
                continue

            if process_message(messages[0]):
                if messages_processed_since_last_faiss_save >= config.FAISS_SAVE_INTERVAL_MESSAGES:
                    logger.info(f"Processed {messages_processed_since_last_faiss_save} messages. Saving FAISS index to S3...")
                    if vector_store_adapter.save_faiss_index_and_metadata_to_s3():
                        messages_processed_since_last_faiss_save = 0
                    else:
                        logger.error("Failed to save FAISS index to S3 during periodic save.")
            else:
                logger.warning("Message processing failed, message will likely be re-processed by SQS after visibility timeout.")
                time.sleep(5)

        except ClientError as e:
            logger.error(f"SQS receive_message ClientError: {e}", exc_info=True)
            time.sleep(10)
        except Exception as e:
            logger.error(f"Unexpected error in SQS polling loop: {e}", exc_info=True)
            time.sleep(10)

    logger.info("SQS polling loop has finished due to shutdown signal.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    final_exit_code = 0
    try:
        logger.info("Worker process started. Calling initialize_app().")
        initialize_app()
        logger.info(f"initialize_app() completed. SQS Client: Type={type(sqs_client)}, ID={id(sqs_client)}. SQS_QUEUE_URL: {config.SQS_QUEUE_URL}")

        if config.SQS_QUEUE_URL and sqs_client is not None:
            logger.info("SQS_QUEUE_URL is configured and SQS client is initialized. Starting main_loop.")
            main_loop()
        else:
            logger.critical("Post-initialization check failed: SQS_QUEUE_URL not configured OR SQS client is None. Worker cannot poll SQS.")
            if not config.SQS_QUEUE_URL:
                logger.error("Reason: SQS_QUEUE_URL is not configured.")
            if sqs_client is None:
                logger.error("Reason: SQS client is None.")
            final_exit_code = 1
    except Exception as e:
        logger.critical(f"Unhandled exception at worker's top level (likely from initialize_app or main_loop): {e}", exc_info=True)
        final_exit_code = 1
    finally:
        logger.info("Worker application is shutting down. Performing final cleanup...")
        if vector_store_adapter.is_initialized():
            logger.info("Attempting to save FAISS index and metadata to S3 before final exit...")
            if vector_store_adapter.save_faiss_index_and_metadata_to_s3():
                logger.info("FAISS index and metadata successfully saved to S3 on shutdown.")
            else:
                logger.error("CRITICAL: Failed to save FAISS index and metadata to S3 during shutdown.")
                final_exit_code = 1
        else:
            logger.info("No FAISS index loaded or initialized, or it's in an inconsistent state. Skipping save on shutdown.")

        logger.info(f"Worker application shutdown complete. Exiting with code {final_exit_code}.")
