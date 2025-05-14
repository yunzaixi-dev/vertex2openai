import asyncio
import json # Needed for error streaming
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any

# Google and OpenAI specific imports
from google.genai import types
from google import genai

# Local module imports
from models import OpenAIRequest, OpenAIMessage
from auth import get_api_key
# from main import credential_manager # Removed to prevent circular import; accessed via request.app.state
import config as app_config
from model_loader import get_vertex_models, get_vertex_express_models # Import from model_loader
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

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        
        # Dynamically fetch allowed models for validation
        vertex_model_ids = await get_vertex_models()
        # Suffixes that can be appended to base models.
        # The remote model config should ideally be the source of truth for all valid permutations.
        standard_suffixes = ["-search", "-encrypt", "-encrypt-full", "-auto"]
        # No longer using special_suffix_map, will use prefix check instead

        all_allowed_model_ids = set(vertex_model_ids) # Start with base models from config
        for base_id in vertex_model_ids: # Iterate over base models to add suffixed versions
            # Apply standard suffixes only if not gemini-2.0
            if not base_id.startswith("gemini-2.0"):
                for suffix in standard_suffixes:
                    all_allowed_model_ids.add(f"{base_id}{suffix}")
            
            # Apply special suffixes for models starting with "gemini-2.5-flash"
            if base_id.startswith("gemini-2.5-flash"):
                special_flash_suffixes = ["-nothinking", "-max"]
                for special_suffix in special_flash_suffixes:
                    all_allowed_model_ids.add(f"{base_id}{special_suffix}")
        
        # Add express models to the allowed list as well.
        # These should be full names from the remote config.
        vertex_express_model_ids = await get_vertex_express_models()
        all_allowed_model_ids.update(vertex_express_model_ids)


        if not request.model or request.model not in all_allowed_model_ids:
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' not found or not supported by this adapter. Valid models are: {sorted(list(all_allowed_model_ids))}", "invalid_request_error"))

        is_auto_model = request.model.endswith("-auto")
        is_grounded_search = request.model.endswith("-search")
        is_encrypted_model = request.model.endswith("-encrypt")
        is_encrypted_full_model = request.model.endswith("-encrypt-full")
        is_nothinking_model = request.model.endswith("-nothinking")
        is_max_thinking_model = request.model.endswith("-max")
        base_model_name = request.model

        # Determine base_model_name by stripping known suffixes
        # This order matters if a model could have multiple (e.g. -encrypt-auto, though not currently a pattern)
        if is_auto_model: base_model_name = request.model[:-len("-auto")]
        elif is_grounded_search: base_model_name = request.model[:-len("-search")]
        elif is_encrypted_full_model: base_model_name = request.model[:-len("-encrypt-full")] # Must be before -encrypt
        elif is_encrypted_model: base_model_name = request.model[:-len("-encrypt")]
        elif is_nothinking_model: base_model_name = request.model[:-len("-nothinking")]
        elif is_max_thinking_model: base_model_name = request.model[:-len("-max")]
        
        # Specific model variant checks (if any remain exclusive and not covered dynamically)
        if is_nothinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-nothinking) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))
        if is_max_thinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-max) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))

        generation_config = create_generation_config(request)

        client_to_use = None
        express_api_key_val = app_config.VERTEX_EXPRESS_API_KEY_VAL
        
        # Use dynamically fetched express models list for this check
        if express_api_key_val and base_model_name in vertex_express_model_ids: # Check against base_model_name
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

        encryption_instructions_placeholder = ["// Protocol Instructions Placeholder //"] # Actual instructions are in message_processing

        if is_auto_model:
            print(f"Processing auto model: {request.model}")
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": encryption_instructions_placeholder}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}                  
            ]
            last_err = None
            for attempt in attempts:
                print(f"Auto-mode attempting: '{attempt['name']}' for model {attempt['model']}")
                current_gen_config = attempt["config_modifier"](generation_config.copy())
                try:
                    return await execute_gemini_call(client_to_use, attempt["model"], attempt["prompt_func"], current_gen_config, request)
                except Exception as e_auto:
                    last_err = e_auto
                    print(f"Auto-attempt '{attempt['name']}' for model {attempt['model']} failed: {e_auto}")
                    await asyncio.sleep(1)
            
            print(f"All auto attempts failed. Last error: {last_err}")
            err_msg = f"All auto-mode attempts failed for model {request.model}. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream: 
                async def final_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    yield f"data: {json.dumps(err_content)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed without specific error.", "server_error"))

        else: # Not an auto model
            current_prompt_func = create_gemini_prompt
            # Determine the actual model string to call the API with (e.g., "gemini-1.5-pro-search")
            api_model_string = request.model 

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                generation_config["tools"] = [search_tool]
            elif is_encrypted_model:
                generation_config["system_instruction"] = encryption_instructions_placeholder
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                generation_config["system_instruction"] = encryption_instructions_placeholder
                current_prompt_func = create_encrypted_full_gemini_prompt
            elif is_nothinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 0}
            elif is_max_thinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 24576}
            
            # For non-auto models, the 'base_model_name' might have suffix stripped.
            # We should use the original 'request.model' for API call if it's a suffixed one,
            # or 'base_model_name' if it's truly a base model without suffixes.
            # The current logic uses 'base_model_name' for the API call in the 'else' block.
            # This means if `request.model` was "gemini-1.5-pro-search", `base_model_name` becomes "gemini-1.5-pro"
            # but the API call might need the full "gemini-1.5-pro-search".
            # Let's use `request.model` for the API call here, and `base_model_name` for checks like Express eligibility.
            return await execute_gemini_call(client_to_use, api_model_string, current_prompt_func, generation_config, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))