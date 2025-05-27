# analysis_worker_app/nlp_tasks/summarizer.py
import logging
import requests

# config 모듈 임포트
import config

# Langchain Text Splitter 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Langchain Text Splitter 초기화 (config 값 사용)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.SUMMARIZER_CHUNK_TARGET_SIZE_CHARS,
    chunk_overlap=config.SUMMARIZER_CHUNK_OVERLAP_CHARS,
    length_function=len,
)

def _call_groq_api(prompt_content: str, max_tokens: int, system_prompt_content: str) -> str | None:
    """Groq API를 호출하여 응답을 반환하는 내부 헬퍼 함수입니다."""
    if not config.GROQ_API_KEY:
        logger.error("Groq API key is not configured in settings.py.")
        return None

    request_payload = {
        "model": config.SUMMARIZER_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6,
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
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=request_payload,
            timeout=90
        )
        response.raise_for_status()

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
        # 파라미터 기본값을 None으로 설정하고, None일 경우 config 값 사용
        long_summary_target_chars: int | None = None,
        long_summary_max_tokens: int | None = None,
        short_summary_max_tokens: int | None = None
) -> str | None:
    if not text_to_summarize or not text_to_summarize.strip():
        logger.info("Text to summarize is empty.")
        return ""

    # 파라미터가 None이면 config에서 기본값 로드
    final_long_summary_target_chars = long_summary_target_chars if long_summary_target_chars is not None else config.SUMMARIZER_LONG_SUMMARY_TARGET_CHARS
    final_long_summary_max_tokens = long_summary_max_tokens if long_summary_max_tokens is not None else config.SUMMARIZER_LONG_SUMMARY_MAX_TOKENS
    final_short_summary_max_tokens = short_summary_max_tokens if short_summary_max_tokens is not None else config.SUMMARIZER_SHORT_SUMMARY_MAX_TOKENS

    original_text_char_length = len(text_to_summarize)
    system_prompt_for_final_summary = "You are an expert summarization assistant. Follow the user's instructions carefully for summary length, format, and handling of short or pre-summarized inputs."

    perform_chunking = original_text_char_length > config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL
    text_for_final_summary_pass: str

    if perform_chunking:
        logger.info(f"Text length ({original_text_char_length} chars) exceeds threshold ({config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL} chars). Applying chunking strategy.")
        chunks = text_splitter.split_text(text_to_summarize)
        logger.info(f"Split text into {len(chunks)} chunks using Langchain splitter. Target chunk size: {config.SUMMARIZER_CHUNK_TARGET_SIZE_CHARS} chars, Overlap: {config.SUMMARIZER_CHUNK_OVERLAP_CHARS} chars.")

        intermediate_summaries = []
        chunk_system_prompt = "You are an assistant that summarizes segments of a larger document. Be concise, capture key information accurately, and maintain context if possible."
        for i, chunk_content in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)} (length: {len(chunk_content)} chars).")
            chunk_prompt = f"This is segment {i+1} of {len(chunks)}. Summarize this text segment concisely, capturing its main points and key details. This summary will be used to create a final, more comprehensive summary of the entire document:\n\n{chunk_content}"
            intermediate_summary = _call_groq_api(chunk_prompt, config.SUMMARIZER_MAX_TOKENS_CHUNK_SUMMARY, chunk_system_prompt)
            if intermediate_summary:
                intermediate_summaries.append(intermediate_summary)
            else:
                logger.warning(f"Failed to summarize chunk {i+1}. It will be omitted from the final summary.")

        if not intermediate_summaries:
            logger.error("No intermediate summaries could be generated from chunks. Cannot proceed with summarization.")
            return None

        intermediate_summaries_concatenated = "\n\n---\n\n".join(intermediate_summaries)
        text_for_final_summary_pass = intermediate_summaries_concatenated
        logger.info(f"Generated {len(intermediate_summaries)} intermediate summaries. Total intermediate length: {len(text_for_final_summary_pass)} chars.")

        if len(text_for_final_summary_pass) > config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL :
            logger.warning(
                f"Concatenated intermediate summaries ({len(text_for_final_summary_pass)} chars) are still too long for a single final pass "
                f"(limit: {config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL} chars). Truncating intermediate summaries to fit."
            )
            text_for_final_summary_pass = text_for_final_summary_pass[:config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL]
    else:
        logger.info(f"Text length ({original_text_char_length} chars) is within single API call limit ({config.SUMMARIZER_MAX_CHARS_SINGLE_API_CALL} chars). No chunking needed.")
        text_for_final_summary_pass = text_to_summarize

    final_prompt_content = ""
    max_tokens_for_final_request = 0

    if summary_type == "short":
        user_prompt_instruction = "Based on the following text (which might be a collection of summaries from a longer document, or the original short document itself), provide a very concise overall summary in one to three sentences:"
        final_prompt_content = f"{user_prompt_instruction}\n\n{text_for_final_summary_pass}"
        max_tokens_for_final_request = final_short_summary_max_tokens # 수정된 변수 사용
        logger.info(f"Requesting final short summary (max_tokens: {max_tokens_for_final_request}).")

    elif summary_type == "long_markdown":
        if not perform_chunking and original_text_char_length < config.SUMMARIZER_MIN_TEXT_LENGTH_LONG_SUMMARY_CHARS:
            user_prompt_instruction = (
                f"The following text is quite short (around {original_text_char_length} characters). "
                f"Provide a concise but well-structured summary using Markdown for formatting if appropriate. "
                f"If the text is already a good summary, you can return it with minimal changes or apply appropriate Markdown formatting. "
                f"The summary should not exceed {final_long_summary_target_chars} characters and should be suitable for the original text's length." # 수정된 변수 사용
            )
        else:
            user_prompt_instruction = (
                f"Synthesize the provided text (which might be a collection of summaries from a longer document, or the original document itself) "
                f"into a coherent, detailed, and well-structured summary, up to a maximum of {final_long_summary_target_chars} characters. " # 수정된 변수 사용
                f"Use Markdown for formatting (e.g., headings, bullet points, bold text) to enhance readability. "
                f"The summary should cover key aspects, arguments, and conclusions from the original content."
            )
        final_prompt_content = f"{user_prompt_instruction}\n\n{text_for_final_summary_pass}"
        max_tokens_for_final_request = final_long_summary_max_tokens # 수정된 변수 사용
        logger.info(f"Requesting final long markdown summary (target_chars: {final_long_summary_target_chars}, max_tokens: {max_tokens_for_final_request}).") # 수정된 변수 사용
    else:
        logger.error(f"Invalid summary_type: {summary_type}.")
        return None

    final_summary = _call_groq_api(final_prompt_content, max_tokens_for_final_request, system_prompt_for_final_summary)

    if final_summary and summary_type == "long_markdown":
        if len(final_summary) > final_long_summary_target_chars * 1.15: # 수정된 변수 사용
            logger.warning(
                f"Generated long_markdown summary significantly exceeded target chars ({len(final_summary)} > {final_long_summary_target_chars}). " # 수정된 변수 사용
                f"Attempting to truncate to {final_long_summary_target_chars} characters." # 수정된 변수 사용
            )
            final_summary = final_summary[:final_long_summary_target_chars] # 수정된 변수 사용

    if final_summary:
        logger.info(f"Successfully generated final {summary_type} summary. Final length: {len(final_summary)} chars.")
    else:
        logger.error(f"Failed to generate final {summary_type} summary for text (length {original_text_char_length} chars).")

    return final_summary
