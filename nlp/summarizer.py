# analysis_worker_app/nlp_tasks/summarizer.py
import logging
import requests
# import re # Langchain text_splitter 사용으로 re는 직접 필요 없을 수 있음

# config 모듈에서 API 키 및 설정을 가져옵니다.

# Langchain Text Splitter 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import GROQ_API_KEY

logger = logging.getLogger(__name__)

# --- 청킹 관련 상수 ---
# 단일 API 호출로 처리할 최대 원본 텍스트 글자 수 (Llama3 8B 8k 토큰 고려)
MAX_CHARS_FOR_SINGLE_API_CALL = 15000
# 각 청크의 목표 글자 수
CHUNK_TARGET_SIZE_CHARS = 3500
# 청크 간 겹치는 글자 수
CHUNK_OVERLAP_CHARS = 300
# 각 청크(중간) 요약 시 생성될 최대 토큰 수
MAX_TOKENS_FOR_CHUNK_SUMMARY = 250
MIN_TEXT_LENGTH_FOR_LONG_SUMMARY_CHARS = 400 # 긴 요약 요청 시 원본 길이 판단용

# --- 모델 및 기타 상수 ---
MODEL_TO_USE = "llama3-8b-8192" # 사용자 지정 모델 (8k 컨텍스트)

# Langchain Text Splitter 초기화 (모듈 로드 시 한 번만)
# chunk_size는 글자 수 기준, length_function=len이 기본값
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TARGET_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    length_function=len,
)

def _call_groq_api(prompt_content: str, max_tokens: int, system_prompt_content: str) -> str | None:
    """Groq API를 호출하여 응답을 반환하는 내부 헬퍼 함수입니다."""
    if not GROQ_API_KEY:
        logger.error("Groq API key is not configured in settings.py.")
        return None

    request_payload = {
        "model": MODEL_TO_USE,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6, # 요약의 일관성을 위해 약간 낮게 설정
        "top_p": 0.9,
    }

    logger.debug(
        f"Sending request to Groq API. Model: {request_payload['model']}, "
        f"Max Tokens: {max_tokens}, "
        f"System Prompt: '{system_prompt_content[:70].replace('\n', ' ')}...', "
        f"User Prompt Preview (first 100 chars): '{prompt_content[:100].replace('\n', ' ')}...'"
    )

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        # Groq API 엔드포인트 확인 필요
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=request_payload,
            timeout=90 # API 호출 타임아웃 (초 단위, 필요시 조절)
        )
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생

        response_data = response.json()
        summary = response_data["choices"][0]["message"]["content"].strip()

        usage_info = response_data.get("usage", {})
        total_tokens = usage_info.get("total_tokens", "N/A")
        logger.info(f"Groq API call successful. Total tokens used: {total_tokens}. Output length: {len(summary)} chars.")
        return summary

    except requests.exceptions.Timeout:
        logger.error(f"Groq API request timed out.", exc_info=False)
    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API request failed: {e}", exc_info=True)
    except KeyError as e:
        response_text = response_data if 'response_data' in locals() else 'N/A (response data not captured)'
        logger.error(f"Failed to parse Groq API response: {e}. Response: {response_text}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during Groq API call: {e}", exc_info=True)

    return None


def summarize_with_groq(
        text_to_summarize: str,
        summary_type: str = "short",
        long_summary_target_chars: int = 800,
        long_summary_max_tokens: int = 450, # 최종 800자 요약 생성 시
        short_summary_max_tokens: int = 150  # 최종 1-3줄 요약 생성 시
) -> str | None:
    if not text_to_summarize or not text_to_summarize.strip():
        logger.info("Text to summarize is empty.")
        return ""

    original_text_char_length = len(text_to_summarize)
    system_prompt_for_final_summary = "You are an expert summarization assistant. Follow the user's instructions carefully for summary length, format, and handling of short or pre-summarized inputs."

    perform_chunking = original_text_char_length > MAX_CHARS_FOR_SINGLE_API_CALL

    text_for_final_summary_pass: str

    if perform_chunking:
        logger.info(f"Text length ({original_text_char_length} chars) exceeds threshold ({MAX_CHARS_FOR_SINGLE_API_CALL} chars). Applying chunking strategy.")
        # Langchain의 RecursiveCharacterTextSplitter 사용
        chunks = text_splitter.split_text(text_to_summarize)
        logger.info(f"Split text into {len(chunks)} chunks using Langchain splitter. Target chunk size: {CHUNK_TARGET_SIZE_CHARS} chars, Overlap: {CHUNK_OVERLAP_CHARS} chars.")

        intermediate_summaries = []
        chunk_system_prompt = "You are an assistant that summarizes segments of a larger document. Be concise, capture key information accurately, and maintain context if possible."
        for i, chunk_content in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)} (length: {len(chunk_content)} chars).")
            # 각 청크에 대한 프롬프트
            chunk_prompt = f"This is segment {i+1} of {len(chunks)}. Summarize this text segment concisely, capturing its main points and key details. This summary will be used to create a final, more comprehensive summary of the entire document:\n\n{chunk_content}"
            intermediate_summary = _call_groq_api(chunk_prompt, MAX_TOKENS_FOR_CHUNK_SUMMARY, chunk_system_prompt)
            if intermediate_summary:
                intermediate_summaries.append(intermediate_summary)
            else:
                logger.warning(f"Failed to summarize chunk {i+1}. It will be omitted from the final summary.")

        if not intermediate_summaries:
            logger.error("No intermediate summaries could be generated from chunks. Cannot proceed with summarization.")
            return None

        intermediate_summaries_concatenated = "\n\n---\n\n".join(intermediate_summaries) # 각 중간 요약 사이에 구분자 추가
        text_for_final_summary_pass = intermediate_summaries_concatenated
        logger.info(f"Generated {len(intermediate_summaries)} intermediate summaries. Total intermediate length: {len(text_for_final_summary_pass)} chars.")

        if len(text_for_final_summary_pass) > MAX_CHARS_FOR_SINGLE_API_CALL :
            logger.warning(
                f"Concatenated intermediate summaries ({len(text_for_final_summary_pass)} chars) are still too long for a single final pass "
                f"(limit: {MAX_CHARS_FOR_SINGLE_API_CALL} chars). Truncating intermediate summaries to fit."
            )
            # 이 경우, text_for_final_summary_pass를 다시 청킹하거나, 앞부분만 사용하거나,
            # 또는 단순히 잘라낼 수 있습니다. 여기서는 간단히 앞부분만 사용합니다.
            text_for_final_summary_pass = text_for_final_summary_pass[:MAX_CHARS_FOR_SINGLE_API_CALL]

    else: # 청킹 불필요
        logger.info(f"Text length ({original_text_char_length} chars) is within single API call limit ({MAX_CHARS_FOR_SINGLE_API_CALL} chars). No chunking needed.")
        text_for_final_summary_pass = text_to_summarize

    # 2. 최종 요약 생성 요청
    final_prompt_content = ""
    max_tokens_for_final_request = 0

    if summary_type == "short":
        user_prompt_instruction = "Based on the following text (which might be a collection of summaries from a longer document, or the original short document itself), provide a very concise overall summary in one to three sentences:"
        final_prompt_content = f"{user_prompt_instruction}\n\n{text_for_final_summary_pass}"
        max_tokens_for_final_request = short_summary_max_tokens
        logger.info(f"Requesting final short summary (max_tokens: {max_tokens_for_final_request}).")

    elif summary_type == "long_markdown":
        # 원본 텍스트가 짧아서 청킹을 안했고, MIN_TEXT_LENGTH_FOR_LONG_SUMMARY_CHARS 보다도 짧은 경우에 대한 프롬프트 조정
        if not perform_chunking and original_text_char_length < MIN_TEXT_LENGTH_FOR_LONG_SUMMARY_CHARS:
            user_prompt_instruction = (
                f"The following text is quite short (around {original_text_char_length} characters). "
                f"Provide a concise but well-structured summary using Markdown for formatting if appropriate. "
                f"If the text is already a good summary, you can return it with minimal changes or apply appropriate Markdown formatting. "
                f"The summary should not exceed {long_summary_target_chars} characters and should be suitable for the original text's length."
            )
        else: # 청킹을 했거나, 청킹 안 했지만 MIN_TEXT_LENGTH_FOR_LONG_SUMMARY_CHARS 이상인 텍스트인 경우
            user_prompt_instruction = (
                f"Synthesize the provided text (which might be a collection of summaries from a longer document, or the original document itself) "
                f"into a coherent, detailed, and well-structured summary, up to a maximum of {long_summary_target_chars} characters. "
                f"Use Markdown for formatting (e.g., headings, bullet points, bold text) to enhance readability. "
                f"The summary should cover key aspects, arguments, and conclusions from the original content."
            )
        final_prompt_content = f"{user_prompt_instruction}\n\n{text_for_final_summary_pass}"
        max_tokens_for_final_request = long_summary_max_tokens
        logger.info(f"Requesting final long markdown summary (target_chars: {long_summary_target_chars}, max_tokens: {max_tokens_for_final_request}).")
    else:
        logger.error(f"Invalid summary_type: {summary_type}.")
        return None

    final_summary = _call_groq_api(final_prompt_content, max_tokens_for_final_request, system_prompt_for_final_summary)

    # 최종 요약 길이 후처리 (long_markdown 경우에만)
    if final_summary and summary_type == "long_markdown":
        if len(final_summary) > long_summary_target_chars * 1.15: # 목표 글자수보다 15% 이상 길 경우 조정
            logger.warning(
                f"Generated long_markdown summary significantly exceeded target chars ({len(final_summary)} > {long_summary_target_chars}). "
                f"Attempting to truncate to {long_summary_target_chars} characters."
            )
            # 더 정교한 자르기(예: 문장 경계)도 가능하지만, 우선 간단히 자름. Markdown 구조가 깨질 수 있음에 유의.
            final_summary = final_summary[:long_summary_target_chars]
            # 잘린 후 마지막 문장이 불완전할 수 있으므로, 마지막 문장 종결부호까지 찾아서 자르는 것이 더 좋음
            # 예: last_period = final_summary.rfind('.'); if last_period > 0: final_summary = final_summary[:last_period+1]

    if final_summary:
        logger.info(f"Successfully generated final {summary_type} summary. Final length: {len(final_summary)} chars.")
    else:
        logger.error(f"Failed to generate final {summary_type} summary for text (length {original_text_char_length} chars).")

    return final_summary