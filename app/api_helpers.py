import json
import time
import math
import asyncio
import base64 
from typing import List, Dict, Any, Callable, Union, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from google.genai.types import HttpOptions 
from google import genai # Original import
from openai import AsyncOpenAI

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    deobfuscate_text, 
    convert_to_openai_format, 
    convert_chunk_to_openai, 
    create_final_chunk,
    split_text_by_completion_tokens,
    parse_gemini_response_for_reasoning_and_content # Added import
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
                for part_item in candidate.content.parts: 
                    if hasattr(part_item, 'text') and isinstance(part_item.text, str) and part_item.text.strip(): return True
    return False

async def _base_fake_stream_engine(
    api_call_task_creator: Callable[[], asyncio.Task],
    extract_text_from_response_func: Callable[[Any], str],  # May not be used if content is pre-processed
    response_id: str,
    sse_model_name: str,
    is_auto_attempt: bool,
    is_valid_response_func: Callable[[Any], bool],
    keep_alive_interval_seconds: float,
    actual_content_text_to_yield: str, # Expects already combined and tagged content
    process_text_func: Optional[Callable[[str, str], str]] = None, # For potential post-processing like deobfuscation
    check_block_reason_func: Optional[Callable[[Any], None]] = None
):
    api_call_task = api_call_task_creator() # This task should ideally just return the already fetched full_api_response

    if keep_alive_interval_seconds > 0:
        # This keep-alive runs while the initial full API call (created by the caller) is pending.
        # The api_call_task here is usually a dummy one just holding the response.
        # The actual waiting with keep-alive should be in the caller of _base_fake_stream_engine.
        # For simplicity, we'll assume keep_alive_interval_seconds passed here is for the chunking phase, if any.
        # A better design would separate the API call wait (with its keep-alive) from chunking.
        # However, to minimize structural changes to this function now:
        pass # Keep-alive during chunking is less common. Primary keep-alive is while waiting for full_api_response.

    try:
        full_api_response = await api_call_task # If task creator is just `asyncio.sleep(0, result=response)`, this is instant.

        if check_block_reason_func:
            check_block_reason_func(full_api_response) # check_block_reason_func operates on the raw API response object.

        # is_valid_response_func also operates on the raw API response object.
        if not is_valid_response_func(full_api_response):
             raise ValueError(f"Invalid/empty API response in fake stream for model {sse_model_name}: {str(full_api_response)[:200]}")

        # Text processing (e.g., deobfuscation) applied to the fully formed text
        content_to_chunk = actual_content_text_to_yield
        if process_text_func:
            content_to_chunk = process_text_func(content_to_chunk, sse_model_name)

        content_to_chunk = content_to_chunk or "" # Ensure it's a string
        chunk_size = max(20, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 0

        if not content_to_chunk and content_to_chunk != "": # Handles None or purely empty string
            empty_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(empty_delta_data)}\n\n"
        else:
            for i in range(0, len(content_to_chunk), chunk_size):
                chunk_text = content_to_chunk[i:i+chunk_size]
                content_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]}
                yield f"data: {json.dumps(content_delta_data)}\n\n"
                if len(content_to_chunk) > chunk_size and i + chunk_size < len(content_to_chunk): # Sleep if not the last chunk and more chunks exist
                    await asyncio.sleep(0.05) # Simulate processing delay

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

async def gemini_fake_stream_generator( # Changed to async
    gemini_client_instance: Any, 
    model_for_api_call: str, 
    prompt_for_api_call: Union[types.Content, List[types.Content]],
    gen_config_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    thought_tag_marker: str
):
    model_name_for_log = getattr(gemini_client_instance, 'model_name', 'unknown_gemini_model_object')
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', client obj: '{model_name_for_log}') with thought_tag_marker: '{thought_tag_marker}'.")
    response_id = f"chatcmpl-{int(time.time())}"

    # 1. Create and await the API call task
    api_call_task = asyncio.create_task(
        gemini_client_instance.aio.models.generate_content(
            model=model_for_api_call, 
            contents=prompt_for_api_call, 
            config=gen_config_for_api_call
        )
    )

    # Keep-alive loop while the main API call is in progress
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]} # Changed reasoning_content to content
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_response = await api_call_task

        separated_reasoning_text = ""
        separated_actual_content_text = ""
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            separated_reasoning_text, separated_actual_content_text = parse_gemini_response_for_reasoning_and_content(raw_response.candidates[0])
        elif hasattr(raw_response, 'text') and raw_response.text is not None:
             separated_actual_content_text = raw_response.text
        
        # Combine reasoning and actual content with the tag
        final_text_to_stream = separated_actual_content_text
        if separated_reasoning_text and thought_tag_marker:
            final_text_to_stream = f"<{thought_tag_marker}>{separated_reasoning_text}</{thought_tag_marker}>{separated_actual_content_text}"
            print(f"DEBUG_GEMINI_FAKE_STREAM_TAGGED: Reasoning tagged and prepended. Tag: {thought_tag_marker}")

        def _process_gemini_text_if_needed(text: str, model_name: str) -> str:
            if model_name.endswith("-encrypt-full"): # This processing (deobfuscate) should apply to the combined text
                return deobfuscate_text(text)
            return text

        # process_text_func (deobfuscation) will be applied by _base_fake_stream_engine
        # to the final_text_to_stream.

        def _check_gemini_block_wrapper(response_to_check: Any):
            if hasattr(response_to_check, 'prompt_feedback') and hasattr(response_to_check.prompt_feedback, 'block_reason') and response_to_check.prompt_feedback.block_reason:
                block_message = f"Response blocked by Gemini safety filter: {response_to_check.prompt_feedback.block_reason}"
                if hasattr(response_to_check.prompt_feedback, 'block_reason_message') and response_to_check.prompt_feedback.block_reason_message:
                    block_message += f" (Message: {response_to_check.prompt_feedback.block_reason_message})"
                raise ValueError(block_message)

        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(asyncio.sleep(0, result=raw_response)),
            extract_text_from_response_func=lambda r: "", # Not used as content is pre-formed
            is_valid_response_func=is_gemini_response_valid,
            check_block_reason_func=_check_gemini_block_wrapper,
            process_text_func=_process_gemini_text_if_needed, # Pass deobfuscation for _base_fake_stream_engine to apply
            response_id=response_id,
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0,
            is_auto_attempt=is_auto_attempt,
            actual_content_text_to_yield=final_text_to_stream # Pass the combined and tagged text
        ):
            yield chunk

    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        # Consider re-raising if auto-mode needs to catch this: raise e_outer_gemini


async def openai_fake_stream_generator(
    openai_client: AsyncOpenAI,
    openai_params: Dict[str, Any], 
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    gcp_credentials: Any, 
    gcp_project_id: str, 
    gcp_location: str,
    base_model_id_for_tokenizer: str,
    thought_tag_marker: str
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI): Prep for '{request_obj.model}' (API model: '{api_model_name}') with thought_tag_marker: '{thought_tag_marker}'.")
    response_id = f"chatcmpl-{int(time.time())}"
    
    async def _openai_api_call_and_split_task_creator_wrapper():
        params_for_non_stream_call = openai_params.copy()
        params_for_non_stream_call['stream'] = False
        
        _api_call_task = asyncio.create_task(
            openai_client.chat.completions.create(**params_for_non_stream_call, extra_body=openai_extra_body)
        )
        raw_response = await _api_call_task
        full_content_from_api = ""
        if raw_response.choices and raw_response.choices[0].message and raw_response.choices[0].message.content is not None:
            full_content_from_api = raw_response.choices[0].message.content
        vertex_completion_tokens = 0
        if raw_response.usage and raw_response.usage.completion_tokens is not None:
            vertex_completion_tokens = raw_response.usage.completion_tokens
        reasoning_text = ""
        actual_content_text = full_content_from_api
        if full_content_from_api and vertex_completion_tokens > 0:
            reasoning_text, actual_content_text, _ = await asyncio.to_thread(
                split_text_by_completion_tokens, 
                gcp_credentials, gcp_project_id, gcp_location,
                base_model_id_for_tokenizer, 
                full_content_from_api,
                vertex_completion_tokens
            )
            if reasoning_text:
                 print(f"DEBUG_FAKE_REASONING_SPLIT: Success. Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content_text)}")
        return raw_response, reasoning_text, actual_content_text

    temp_task_for_keepalive_check = asyncio.create_task(_openai_api_call_and_split_task_creator_wrapper())
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not temp_task_for_keepalive_check.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]} # Keep-alive has empty content
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        full_api_response, separated_reasoning_text, separated_actual_content_text = await temp_task_for_keepalive_check
        
        # Combine reasoning and actual content with the tag
        final_text_to_stream = separated_actual_content_text
        if separated_reasoning_text and thought_tag_marker:
            final_text_to_stream = f"<{thought_tag_marker}>{separated_reasoning_text}</{thought_tag_marker}>{separated_actual_content_text}"
            print(f"DEBUG_OPENAI_FAKE_STREAM_TAGGED: Reasoning tagged and prepended. Tag: {thought_tag_marker}")

        def _is_openai_response_valid(response: Any) -> bool: # full_api_response is the raw one
            return bool(response.choices and response.choices[0].message is not None)

        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(asyncio.sleep(0, result=full_api_response)),
            extract_text_from_response_func=lambda r: "", # Not used, content is pre-formed
            is_valid_response_func=_is_openai_response_valid,
            # No specific block checking for OpenAI direct path currently, relies on API errors.
            # No specific process_text_func like deobfuscation defined for OpenAI direct path.
            response_id=response_id,
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0,
            is_auto_attempt=is_auto_attempt,
            actual_content_text_to_yield=final_text_to_stream # Pass the combined and tagged text
        ):
            yield chunk
            
    except Exception as e_outer:
        err_msg_detail = f"Error in openai_fake_stream_generator outer (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"

async def execute_gemini_call(
    current_client: Any, 
    model_to_call: str,  
    prompt_func: Callable[[List[OpenAIMessage]], Union[types.Content, List[types.Content]]], 
    gen_config_for_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    thought_tag_marker: str, # Added
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    client_model_name_for_log = getattr(current_client, 'model_name', 'unknown_direct_client_object')
    print(f"INFO: execute_gemini_call for requested API model '{model_to_call}', client obj '{client_model_name_for_log}'. Original request model: '{request_obj.model}', thought_tag_marker: '{thought_tag_marker}'")

    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                gemini_fake_stream_generator( 
                    current_client, 
                    model_to_call,
                    actual_prompt_for_call,
                    gen_config_for_call,
                    request_obj,
                    is_auto_attempt,
                    thought_tag_marker # Pass to gemini_fake_stream_generator
                ),
                media_type="text/event-stream"
            )
        
        response_id_for_stream = f"chatcmpl-{int(time.time())}"
        cand_count_stream = request_obj.n or 1
        
        async def _gemini_real_stream_generator_inner():
            try:
                async for chunk_item_call in await current_client.aio.models.generate_content_stream(
                    model=model_to_call, 
                    contents=actual_prompt_for_call, 
                    config=gen_config_for_call
                ):
                    yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                yield create_final_chunk(request_obj.model, response_id_for_stream, cand_count_stream)
                yield "data: [DONE]\n\n"
            except Exception as e_stream_call:
                err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                print(f"ERROR: {err_msg_detail_stream}")
                s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                err_resp = create_openai_error_response(500,s_err,"server_error")
                j_err = json.dumps(err_resp)
                if not is_auto_attempt: 
                    yield f"data: {j_err}\n\n"
                    yield "data: [DONE]\n\n"
                raise e_stream_call
        return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: 
        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call, 
            contents=actual_prompt_for_call, 
            config=gen_config_for_call
        )
        if hasattr(response_obj_call, 'prompt_feedback') and hasattr(response_obj_call.prompt_feedback, 'block_reason') and response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and response_obj_call.prompt_feedback.block_reason_message: 
                block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        
        if not is_gemini_response_valid(response_obj_call):
            raise ValueError(f"Invalid non-streaming Gemini response for model string '{model_to_call}'. Resp: {str(response_obj_call)[:200]}")
        # For non-streaming Gemini, convert_to_openai_format needs to handle the tagging
        # This will be addressed by modifying convert_to_openai_format in message_processing.py
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model, thought_tag_marker))