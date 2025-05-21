import json
import time
import math
import asyncio
import base64 # Added for tokenizer logic
from typing import List, Dict, Any, Callable, Union, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from google.genai.types import HttpOptions # Added for tokenizer logic
from google import genai
from openai import AsyncOpenAI

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    deobfuscate_text,
    convert_to_openai_format,
    convert_chunk_to_openai,
    create_final_chunk,
    split_text_by_completion_tokens # Added
)
import config as app_config

def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
            "param": None,
        }
    }

def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config = {}
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.presence_penalty is not None: config["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None: config["frequency_penalty"] = request.frequency_penalty
    if request.n is not None: config["candidate_count"] = request.n
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF")
    ]
    return config

def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'text') and isinstance(candidate.text, str) and candidate.text.strip(): return True
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and isinstance(part.text, str) and part.text.strip(): return True
    return False

async def _base_fake_stream_engine(
    api_call_task_creator: Callable[[], asyncio.Task],
    extract_text_from_response_func: Callable[[Any], str], # To get the *full* text before splitting
    response_id: str,
    sse_model_name: str,
    is_auto_attempt: bool,
    is_valid_response_func: Callable[[Any], bool],
    process_text_func: Optional[Callable[[str, str], str]] = None,
    check_block_reason_func: Optional[Callable[[Any], None]] = None,
    # New parameters for pre-split content
    reasoning_text_to_yield: Optional[str] = None,
    actual_content_text_to_yield: Optional[str] = None
):
    api_call_task = api_call_task_creator()

    while not api_call_task.done():
        keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"delta": {"reasoning_content": ""}, "index": 0, "finish_reason": None}]}
        yield f"data: {json.dumps(keep_alive_data)}\n\n"
        await asyncio.sleep(app_config.FAKE_STREAMING_INTERVAL_SECONDS)
    
    try:
        full_api_response = await api_call_task

        if check_block_reason_func:
            check_block_reason_func(full_api_response)

        if not is_valid_response_func(full_api_response):
             raise ValueError(f"Invalid/empty response in fake stream for model {sse_model_name} (validation failed): {str(full_api_response)[:200]}")

        # Determine content to chunk
        content_to_chunk = ""
        if actual_content_text_to_yield is not None:
            content_to_chunk = actual_content_text_to_yield
            if process_text_func: # Process only the actual content part if pre-split
                 content_to_chunk = process_text_func(content_to_chunk, sse_model_name)
        else: # Fallback to old method if no pre-split content provided
            content_to_chunk = extract_text_from_response_func(full_api_response)
            if process_text_func:
                content_to_chunk = process_text_func(content_to_chunk, sse_model_name)
        
        # Yield reasoning chunk first if available
        if reasoning_text_to_yield:
            reasoning_delta_data = {
                "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": sse_model_name, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning_text_to_yield}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(reasoning_delta_data)}\n\n"
            await asyncio.sleep(0.05) # Small delay after reasoning

        # Chunk and yield the main content
        chunk_size = max(20, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 0
        
        if not content_to_chunk and content_to_chunk != "":
            empty_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(empty_delta_data)}\n\n"
        else:
            for i in range(0, len(content_to_chunk), chunk_size):
                chunk_text = content_to_chunk[i:i+chunk_size]
                content_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]}
                yield f"data: {json.dumps(content_delta_data)}\n\n"
                if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)

        yield create_final_chunk(sse_model_name, response_id)
        yield "data: [DONE]\n\n"

    except Exception as e:
        err_msg_detail = f"Error in _base_fake_stream_engine (model: '{sse_model_name}'): {type(e).__name__} - {str(e)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e) 
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_for_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_for_fake_stream_error = json.dumps(err_resp_for_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_for_fake_stream_error}\n\n"
            yield "data: [DONE]\n\n"
        raise

def gemini_fake_stream_generator(
    gemini_model_instance: genai.GenerativeModel, 
    prompt_for_api_call: Union[types.Content, List[types.Content]],
    gen_config_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
):
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model: '{gemini_model_instance.model_name}')")
    def _create_gemini_api_task() -> asyncio.Task:
        return asyncio.create_task(gemini_model_instance.generate_content_async(contents=prompt_for_api_call, generation_config=gen_config_for_api_call))
    def _extract_gemini_text(response: Any) -> str:
        # ... (extraction logic as before) ...
        full_text = ""
        if hasattr(response, 'text') and response.text is not None: full_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'text') and candidate.text is not None: full_text = candidate.text
            elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                texts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text is not None]
                full_text = "".join(texts)
        return full_text
    def _process_gemini_text(text: str, sse_model_name: str) -> str:
        if sse_model_name.endswith("-encrypt-full"): return deobfuscate_text(text)
        return text
    def _check_gemini_block(response: Any):
        if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {response.prompt_feedback.block_reason}"
            if hasattr(response.prompt_feedback, 'block_reason_message') and response.prompt_feedback.block_reason_message: block_message += f" (Message: {response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)
    response_id = f"chatcmpl-{int(time.time())}"
    return _base_fake_stream_engine(
        api_call_task_creator=_create_gemini_api_task,
        extract_text_from_response_func=_extract_gemini_text,
        process_text_func=_process_gemini_text,
        check_block_reason_func=_check_gemini_block,
        is_valid_response_func=is_gemini_response_valid, 
        response_id=response_id, sse_model_name=request_obj.model,
        keep_alive_interval_seconds=app_config.FAKE_STREAMING_INTERVAL_SECONDS,
        is_auto_attempt=is_auto_attempt
        # reasoning_text_to_yield and actual_content_text_to_yield are not used for Gemini
    )

async def openai_fake_stream_generator( # Changed to async to await the tokenizer
    openai_client: AsyncOpenAI,
    openai_params: Dict[str, Any], 
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    # New params for tokenizer
    gcp_credentials: Any, 
    gcp_project_id: str, 
    gcp_location: str,
    base_model_id_for_tokenizer: str 
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI): Prep for '{request_obj.model}' (API model: '{api_model_name}') with reasoning split.")

    response_id = f"chatcmpl-{int(time.time())}"
    
    # This task creator now involves the full API call and subsequent token splitting.
    # The _base_fake_stream_engine will then use the pre-split text.
    async def _openai_api_call_and_split_task_creator_wrapper():
        # This inner async function will be what the asyncio.Task runs.
        # It first makes the API call, then does the sync tokenization in a thread.
        
        # 1. Make the non-streaming API call
        _api_call_task = asyncio.create_task(
            openai_client.chat.completions.create(
                **openai_params, extra_body=openai_extra_body, stream=False
            )
        )
        raw_response = await _api_call_task # This is the openai.types.chat.ChatCompletion object

        # 2. Extract full content and usage for splitting
        full_content_from_api = ""
        if raw_response.choices and raw_response.choices[0].message and raw_response.choices[0].message.content is not None:
            full_content_from_api = raw_response.choices[0].message.content
        
        vertex_completion_tokens = 0
        if raw_response.usage and raw_response.usage.completion_tokens is not None:
            vertex_completion_tokens = raw_response.usage.completion_tokens

        reasoning_text = ""
        actual_content_text = full_content_from_api # Default if split fails or not applicable

        if full_content_from_api and vertex_completion_tokens > 0:
            # 3. Perform synchronous tokenization and splitting in a separate thread
            reasoning_text, actual_content_text, _ = await asyncio.to_thread(
                split_text_by_completion_tokens, # Use imported function
                gcp_credentials, gcp_project_id, gcp_location,
                base_model_id_for_tokenizer, # The base model for the tokenizer
                full_content_from_api,
                vertex_completion_tokens
            )
            if reasoning_text:
                 print(f"DEBUG_FAKE_REASONING_SPLIT: Success. Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content_text)}")

        # We pass the raw_response and the split text to the base engine.
        # The base engine still needs the raw_response for initial validation,
        # but will use the pre-split text for yielding chunks.
        return raw_response, reasoning_text, actual_content_text

    # The main generator logic starts here:
    # Initial keep-alive loop
    temp_task_for_keepalive_check = asyncio.create_task(_openai_api_call_and_split_task_creator_wrapper())
    while not temp_task_for_keepalive_check.done():
        keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
        yield f"data: {json.dumps(keep_alive_data)}\n\n"
        await asyncio.sleep(app_config.FAKE_STREAMING_INTERVAL_SECONDS)

    try:
        # Get the results from our wrapper task
        full_api_response, separated_reasoning_text, separated_actual_content_text = await temp_task_for_keepalive_check

        # Define OpenAI specific helpers for _base_fake_stream_engine
        def _extract_openai_full_text(response: Any) -> str: # Still needed for initial validation if used
            if response.choices and response.choices[0].message and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""
        def _is_openai_response_valid(response: Any) -> bool:
            return bool(response.choices and response.choices[0].message is not None)

        # Now, iterate through the base engine using the results
        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(asyncio.sleep(0, result=full_api_response)), # Dummy task, result already known
            extract_text_from_response_func=_extract_openai_full_text, # For potential use by is_valid_response_func
            is_valid_response_func=_is_openai_response_valid,
            response_id=response_id,
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0, # Keep-alive handled above for the combined op
            is_auto_attempt=is_auto_attempt,
            reasoning_text_to_yield=separated_reasoning_text,
            actual_content_text_to_yield=separated_actual_content_text
        ):
            yield chunk
            
    except Exception as e_outer: # Catch errors from the _openai_api_call_and_split_task_creator_wrapper or subsequent base engine
        err_msg_detail = f"Error in openai_fake_stream_generator outer (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        # No re-raise here as we've handled sending the error via SSE.
        # If auto-mode needs to retry, the exception from the inner task would have been raised before this point.


async def execute_gemini_call(
    current_client: Any, model_to_call: str, 
    prompt_func: Callable[[List[OpenAIMessage]], Union[types.Content, List[types.Content]]], 
    gen_config_for_call: Dict[str, Any], request_obj: OpenAIRequest,
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    gemini_model_instance: Optional[genai.GenerativeModel] = None
    if hasattr(current_client, 'get_model') and callable(getattr(current_client, 'get_model')):
        try: gemini_model_instance = current_client.get_model(model_name=model_to_call)
        except Exception as e: raise ValueError(f"Could not get Gemini model '{model_to_call}' Express: {e}") from e
    elif isinstance(current_client, genai.GenerativeModel):
        if model_to_call not in current_client.model_name: print(f"WARNING: Mismatch! model_to_call='{model_to_call}', client.model_name='{current_client.model_name}'")
        gemini_model_instance = current_client
    else: raise ValueError(f"Unsupported current_client for Gemini: {type(current_client)}")
    if not gemini_model_instance: raise ValueError(f"Failed to get GeminiModel for '{model_to_call}'.")

    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(gemini_fake_stream_generator(gemini_model_instance, actual_prompt_for_call, gen_config_for_call, request_obj, is_auto_attempt), media_type="text/event-stream")
        response_id_for_stream, cand_count_stream = f"chatcmpl-{int(time.time())}", request_obj.n or 1
        async def _gemini_real_stream_generator_inner():
            try:
                async for chunk_item_call in gemini_model_instance.generate_content_async(contents=actual_prompt_for_call, generation_config=gen_config_for_call, stream=True):
                    yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                yield create_final_chunk(request_obj.model, response_id_for_stream, cand_count_stream)
                yield "data: [DONE]\n\n"
            except Exception as e:
                # ... (error handling as before) ...
                err_msg_detail_stream = f"Streaming Error (Gemini model: '{gemini_model_instance.model_name}'): {type(e).__name__} - {str(e)}"
                print(f"ERROR: {err_msg_detail_stream}")
                s_err = str(e); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                err_resp = create_openai_error_response(500,s_err,"server_error")
                j_err = json.dumps(err_resp)
                if not is_auto_attempt: yield f"data: {j_err}\n\n"; yield "data: [DONE]\n\n"
                raise e
        return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else:
        response_obj_call = await gemini_model_instance.generate_content_async(contents=actual_prompt_for_call, generation_config=gen_config_for_call)
        if hasattr(response_obj_call, 'prompt_feedback') and hasattr(response_obj_call.prompt_feedback, 'block_reason') and response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and response_obj_call.prompt_feedback.block_reason_message: block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        if not is_gemini_response_valid(response_obj_call): raise ValueError(f"Invalid non-streaming Gemini response for '{gemini_model_instance.model_name}'. Resp: {str(response_obj_call)[:200]}")
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))