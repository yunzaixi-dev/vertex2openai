import asyncio
import json # Needed for error streaming
from fastapi import APIRouter, Depends, Request # Added Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any

# Google and OpenAI specific imports
from google.genai import types
from google import genai

# Local module imports (now absolute from app/ perspective)
from models import OpenAIRequest, OpenAIMessage
from auth import get_api_key
# from main import credential_manager # Removed, will use request.app.state
import config as app_config
from vertex_ai_init import VERTEX_EXPRESS_MODELS
from message_processing import (
    create_gemini_prompt,
    create_encrypted_gemini_prompt,
    create_encrypted_full_gemini_prompt
)
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call
)

router = APIRouter()

async def _temp_list_models_for_validation():
    return {"data": [{"id": model_name} for model_name in VERTEX_EXPRESS_MODELS]}


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        # Access credential_manager from app state
        credential_manager_instance = fastapi_request.app.state.credential_manager
        models_response = await _temp_list_models_for_validation()
        available_models_ids = [model["id"] for model in models_response.get("data", [])]
        # This list should be kept in sync with the models actually supported by the adapter's logic.
        extended_available_models = set(available_models_ids + [
            "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro-exp-03-25-search", "gemini-2.5-pro-exp-03-25-encrypt", "gemini-2.5-pro-exp-03-25-encrypt-full", "gemini-2.5-pro-exp-03-25-auto",
            "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-preview-03-25-search", "gemini-2.5-pro-preview-03-25-encrypt", "gemini-2.5-pro-preview-03-25-encrypt-full", "gemini-2.5-pro-preview-03-25-auto",
            "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-05-06-search", "gemini-2.5-pro-preview-05-06-encrypt", "gemini-2.5-pro-preview-05-06-encrypt-full", "gemini-2.5-pro-preview-05-06-auto",
            "gemini-2.0-flash", "gemini-2.0-flash-search", "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-search",
            "gemini-2.0-pro-exp-02-05", "gemini-1.5-flash",
            "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-preview-04-17-encrypt", "gemini-2.5-flash-preview-04-17-nothinking", "gemini-2.5-flash-preview-04-17-max",
            "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-1.0-pro-002", "gemini-1.0-pro-vision-001", "gemini-embedding-exp"
        ])

        if not request.model or request.model not in extended_available_models:
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' not found or not supported by this adapter.", "invalid_request_error"))

        is_auto_model = request.model.endswith("-auto")
        is_grounded_search = request.model.endswith("-search")
        is_encrypted_model = request.model.endswith("-encrypt")
        is_encrypted_full_model = request.model.endswith("-encrypt-full")
        is_nothinking_model = request.model.endswith("-nothinking")
        is_max_thinking_model = request.model.endswith("-max")
        base_model_name = request.model

        if is_auto_model: base_model_name = request.model.replace("-auto", "")
        elif is_grounded_search: base_model_name = request.model.replace("-search", "")
        elif is_encrypted_model: base_model_name = request.model.replace("-encrypt", "")
        elif is_encrypted_full_model: base_model_name = request.model.replace("-encrypt-full", "")
        elif is_nothinking_model:
            base_model_name = request.model.replace("-nothinking","")
            if base_model_name != "gemini-2.5-flash-preview-04-17":
                return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' does not support -nothinking variant", "invalid_request_error"))
        elif is_max_thinking_model:
            base_model_name = request.model.replace("-max","")
            if base_model_name != "gemini-2.5-flash-preview-04-17":
                return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' does not support -max variant", "invalid_request_error"))

        generation_config = create_generation_config(request)

        client_to_use = None
        express_api_key_val = app_config.VERTEX_EXPRESS_API_KEY_VAL

        if express_api_key_val and base_model_name in VERTEX_EXPRESS_MODELS:
            try:
                client_to_use = genai.Client(vertexai=True, api_key=express_api_key_val)
                print(f"INFO: Using Vertex Express Mode for model {base_model_name}.")
            except Exception as e:
                print(f"ERROR: Vertex Express Mode client init failed: {e}. Falling back.")
                client_to_use = None

        if client_to_use is None:
            rotated_credentials, rotated_project_id = credential_manager_instance.get_random_credentials()
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="us-central1")
                    print(f"INFO: Using rotated credential for project: {rotated_project_id}")
                except Exception as e:
                    print(f"ERROR: Rotated credential client init failed: {e}. Falling back.")
                    client_to_use = None
        
        if client_to_use is None:
            print("ERROR: No Vertex AI client could be initialized via Express Mode or Rotated Credentials.")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Vertex AI client not available. Ensure credentials are set up correctly (env var or files).", "server_error"))

        encryption_instructions = ["// Protocol Instructions Placeholder //"]

        if is_auto_model:
            print(f"Processing auto model: {request.model}")
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": encryption_instructions}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}                  
            ]
            last_err = None
            for attempt in attempts:
                print(f"Auto-mode attempting: '{attempt['name']}'")
                current_gen_config = attempt["config_modifier"](generation_config.copy())
                try:
                    return await execute_gemini_call(client_to_use, attempt["model"], attempt["prompt_func"], current_gen_config, request)
                except Exception as e_auto:
                    last_err = e_auto
                    print(f"Auto-attempt '{attempt['name']}' failed: {e_auto}")
                    await asyncio.sleep(1)
            
            print(f"All auto attempts failed. Last error: {last_err}")
            err_msg = f"All auto-mode attempts failed for {request.model}. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream: 
                async def final_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    yield f"data: {json.dumps(err_content)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed without specific error.", "server_error"))

        else: 
            current_prompt_func = create_gemini_prompt
            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                generation_config["tools"] = [search_tool]
            elif is_encrypted_model:
                generation_config["system_instruction"] = encryption_instructions
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                generation_config["system_instruction"] = encryption_instructions
                current_prompt_func = create_encrypted_full_gemini_prompt
            elif is_nothinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 0}
            elif is_max_thinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 24576}
            
            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, generation_config, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))