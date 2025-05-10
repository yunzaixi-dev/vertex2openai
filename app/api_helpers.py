import json
import time
import math
import asyncio
from typing import List, Dict, Any, Callable, Union
from fastapi.responses import JSONResponse, StreamingResponse

from google.auth.transport.requests import Request as AuthRequest
from google.genai import types 
from google import genai # Needed if _execute_gemini_call uses genai.Client directly

# Local module imports
from models import OpenAIRequest, OpenAIMessage # Changed from relative
from message_processing import deobfuscate_text, convert_to_openai_format, convert_chunk_to_openai, create_final_chunk # Changed from relative
import config as app_config # Changed from relative

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

def is_response_valid(response):
    if response is None: return False
    if hasattr(response, 'text') and response.text: return True
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'text') and candidate.text: return True
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text: return True
    if hasattr(response, 'candidates') and response.candidates: return True # For fake streaming
    for attr in dir(response):
        if attr.startswith('_'): continue
        try:
            if isinstance(getattr(response, attr), str) and getattr(response, attr): return True
        except: pass
    print("DEBUG: Response is invalid, no usable content found")
    return False

async def fake_stream_generator(client_instance, model_name: str, prompt: Union[types.Content, List[types.Content]], current_gen_config: Dict[str, Any], request_obj: OpenAIRequest):
    response_id = f"chatcmpl-{int(time.time())}"
    async def fake_stream_inner():
        print(f"FAKE STREAMING: Making non-streaming request to Gemini API (Model: {model_name})")
        api_call_task = asyncio.create_task(
            client_instance.aio.models.generate_content(
                model=model_name, contents=prompt, config=current_gen_config
            )
        )
        while not api_call_task.done():
            keep_alive_data = {
                "id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]
            }
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(app_config.FAKE_STREAMING_INTERVAL_SECONDS)
        try:
            response = api_call_task.result()
            if not is_response_valid(response): 
                raise ValueError(f"Invalid/empty response in fake stream: {str(response)[:200]}")
            full_text = ""
            if hasattr(response, 'text'): full_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'text'): full_text = candidate.text
                elif hasattr(candidate.content, 'parts'):
                    full_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            if request_obj.model.endswith("-encrypt-full"):
                full_text = deobfuscate_text(full_text)
            
            chunk_size = max(20, math.ceil(len(full_text) / 10))
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i+chunk_size]
                delta_data = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": request_obj.model, "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(delta_data)}\n\n"
                await asyncio.sleep(0.05)
            yield create_final_chunk(request_obj.model, response_id)
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_msg = f"Error in fake_stream_generator: {str(e)}"
            print(err_msg)
            err_resp = create_openai_error_response(500, err_msg, "server_error")
            yield f"data: {json.dumps(err_resp)}\n\n"
            yield "data: [DONE]\n\n"
    return fake_stream_inner()

async def execute_gemini_call(
    current_client: Any, # Should be genai.Client or similar AsyncClient
    model_to_call: str, 
    prompt_func: Callable[[List[OpenAIMessage]], Union[types.Content, List[types.Content]]], 
    gen_config_for_call: Dict[str, Any],
    request_obj: OpenAIRequest # Pass the whole request object
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    
    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                await fake_stream_generator(current_client, model_to_call, actual_prompt_for_call, gen_config_for_call, request_obj), 
                media_type="text/event-stream"
            )

        response_id_for_stream = f"chatcmpl-{int(time.time())}"
        cand_count_stream = request_obj.n or 1
        
        async def _stream_generator_inner_for_execute(): # Renamed to avoid potential clashes
            try:
                for c_idx_call in range(cand_count_stream):
                    async for chunk_item_call in await current_client.aio.models.generate_content_stream(
                        model=model_to_call, contents=actual_prompt_for_call, config=gen_config_for_call
                    ):
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, c_idx_call)
                yield create_final_chunk(request_obj.model, response_id_for_stream, cand_count_stream)
                yield "data: [DONE]\n\n"
            except Exception as e_stream_call:
                print(f"Streaming Error in _execute_gemini_call: {e_stream_call}")
                err_resp_content_call = create_openai_error_response(500, str(e_stream_call), "server_error")
                yield f"data: {json.dumps(err_resp_content_call)}\n\n"
                yield "data: [DONE]\n\n"
                raise # Re-raise to be caught by retry logic if any
        return StreamingResponse(_stream_generator_inner_for_execute(), media_type="text/event-stream")
    else: 
        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call, contents=actual_prompt_for_call, config=gen_config_for_call
        )
        if not is_response_valid(response_obj_call):
            raise ValueError("Invalid/empty response from non-streaming Gemini call in _execute_gemini_call.")
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))