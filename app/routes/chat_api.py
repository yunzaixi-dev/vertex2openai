import asyncio
import json # Needed for error streaming
import random
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any

# Google and OpenAI specific imports
from google.genai import types
from google import genai
import openai
from credentials_manager import _refresh_auth

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
        OPENAI_DIRECT_SUFFIX = "-openai"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        
        # Model validation based on a predefined list has been removed as per user request.
        # The application will now attempt to use any provided model string.
        # We still need to fetch vertex_express_model_ids for the Express Mode logic.
        vertex_express_model_ids = await get_vertex_express_models()

        # Updated logic for is_openai_direct_model
        is_openai_direct_model = False
        if request.model.endswith(OPENAI_DIRECT_SUFFIX):
            temp_name_for_marker_check = request.model[:-len(OPENAI_DIRECT_SUFFIX)]
            if temp_name_for_marker_check.startswith(PAY_PREFIX):
                is_openai_direct_model = True
            elif EXPERIMENTAL_MARKER in temp_name_for_marker_check:
                is_openai_direct_model = True
        is_auto_model = request.model.endswith("-auto")
        is_grounded_search = request.model.endswith("-search")
        is_encrypted_model = request.model.endswith("-encrypt")
        is_encrypted_full_model = request.model.endswith("-encrypt-full")
        is_nothinking_model = request.model.endswith("-nothinking")
        is_max_thinking_model = request.model.endswith("-max")
        base_model_name = request.model

        # Determine base_model_name by stripping known suffixes
        # This order matters if a model could have multiple (e.g. -encrypt-auto, though not currently a pattern)
        if is_openai_direct_model:
            # The general PAY_PREFIX stripper later will handle if this result starts with [PAY]
            base_model_name = request.model[:-len(OPENAI_DIRECT_SUFFIX)]
        elif is_auto_model: base_model_name = request.model[:-len("-auto")]
        elif is_grounded_search: base_model_name = request.model[:-len("-search")]
        elif is_encrypted_full_model: base_model_name = request.model[:-len("-encrypt-full")] # Must be before -encrypt
        elif is_encrypted_model: base_model_name = request.model[:-len("-encrypt")]
        elif is_nothinking_model: base_model_name = request.model[:-len("-nothinking")]
        elif is_max_thinking_model: base_model_name = request.model[:-len("-max")]
        
        # After all suffix stripping, if PAY_PREFIX is still at the start of base_model_name, remove it.
        # This handles cases like "[PAY]model-id-search" correctly.
        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]
            
        # Specific model variant checks (if any remain exclusive and not covered dynamically)
        if is_nothinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-nothinking) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))
        if is_max_thinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-max) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))

        generation_config = create_generation_config(request)

        client_to_use = None
        express_api_keys_list = app_config.VERTEX_EXPRESS_API_KEY_VAL
        
        # Use dynamically fetched express models list for this check
        if express_api_keys_list and base_model_name in vertex_express_model_ids: # Check against base_model_name
            indexed_keys = list(enumerate(express_api_keys_list))
            random.shuffle(indexed_keys)
            
            for original_idx, key_val in indexed_keys:
                try:
                    client_to_use = genai.Client(vertexai=True, api_key=key_val)
                    print(f"INFO: Using Vertex Express Mode for model {base_model_name} with API key (original index: {original_idx}).")
                    break # Successfully initialized client
                except Exception as e:
                    print(f"WARNING: Vertex Express Mode client init failed for API key (original index: {original_idx}): {e}. Trying next key if available.")
                    client_to_use = None # Ensure client_to_use is None if this attempt fails
            
            if client_to_use is None:
                print(f"WARNING: All {len(express_api_keys_list)} Vertex Express API key(s) failed to initialize for model {base_model_name}. Falling back.")

        if client_to_use is None:
            rotated_credentials, rotated_project_id = credential_manager_instance.get_random_credentials()
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                    print(f"INFO: Using rotated credential for project: {rotated_project_id}")
                except Exception as e:
                    print(f"ERROR: Rotated credential client init failed: {e}. Falling back.")
                    client_to_use = None
        
        if client_to_use is None:
            print("ERROR: No Vertex AI client could be initialized via Express Mode or Rotated Credentials.")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Vertex AI client not available. Ensure credentials are set up correctly (env var or files).", "server_error"))

        encryption_instructions_placeholder = ["// Protocol Instructions Placeholder //"] # Actual instructions are in message_processing
        if is_openai_direct_model:
            print(f"INFO: Using OpenAI Direct Path for model: {request.model}")
            # This mode exclusively uses rotated credentials, not express keys.
            rotated_credentials, rotated_project_id = credential_manager_instance.get_random_credentials()

            if not rotated_credentials or not rotated_project_id:
                error_msg = "OpenAI Direct Mode requires GCP credentials, but none were available or loaded successfully."
                print(f"ERROR: {error_msg}")
                return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))

            print(f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
            gcp_token = _refresh_auth(rotated_credentials)

            if not gcp_token:
                error_msg = f"Failed to obtain valid GCP token for OpenAI client (Source: Credential Manager, Project: {rotated_project_id})."
                print(f"ERROR: {error_msg}")
                return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))

            PROJECT_ID = rotated_project_id
            LOCATION = "global" # Fixed as per user confirmation
            VERTEX_AI_OPENAI_ENDPOINT_URL = (
                f"https://aiplatform.googleapis.com/v1beta1/"
                f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"
            )
            # base_model_name is already extracted (e.g., "gemini-1.5-pro-exp-v1")
            UNDERLYING_MODEL_ID = f"google/{base_model_name}"

            openai_client = openai.AsyncOpenAI(
                base_url=VERTEX_AI_OPENAI_ENDPOINT_URL,
                api_key=gcp_token, # OAuth token
            )

            openai_safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": 'HARM_CATEGORY_CIVIC_INTEGRITY', "threshold": 'OFF'}
            ]

            openai_params = {
                "model": UNDERLYING_MODEL_ID,
                "messages": [msg.model_dump(exclude_unset=True) for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stream": request.stream,
                "stop": request.stop,
                "seed": request.seed,
                "n": request.n,
            }
            openai_params = {k: v for k, v in openai_params.items() if v is not None}

            openai_extra_body = {
                'google': {
                    'safety_settings': openai_safety_settings
                }
            }

            if request.stream:
                async def openai_stream_generator():
                    try:
                        stream_response = await openai_client.chat.completions.create(
                            **openai_params,
                            extra_body=openai_extra_body
                        )
                        async for chunk in stream_response:
                            try:
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            except Exception as chunk_serialization_error:
                                error_msg_chunk = f"Error serializing OpenAI chunk for {request.model}: {str(chunk_serialization_error)}. Chunk: {str(chunk)[:200]}"
                                print(f"ERROR: {error_msg_chunk}")
                                # Truncate
                                if len(error_msg_chunk) > 1024:
                                    error_msg_chunk = error_msg_chunk[:1024] + "..."
                                error_response_chunk = create_openai_error_response(500, error_msg_chunk, "server_error")
                                json_payload_for_chunk_error = json.dumps(error_response_chunk)
                                print(f"DEBUG: Yielding chunk serialization error JSON payload (OpenAI path): {json_payload_for_chunk_error}")
                                yield f"data: {json_payload_for_chunk_error}\n\n"
                                yield "data: [DONE]\n\n"
                                return # Stop further processing for this request
                        yield "data: [DONE]\n\n"
                    except Exception as stream_error:
                        original_error_message = str(stream_error)
                        # Truncate very long error messages
                        if len(original_error_message) > 1024:
                            original_error_message = original_error_message[:1024] + "..."
                        
                        error_msg_stream = f"Error during OpenAI client streaming for {request.model}: {original_error_message}"
                        print(f"ERROR: {error_msg_stream}")
                        
                        error_response_content = create_openai_error_response(500, error_msg_stream, "server_error")
                        json_payload_for_stream_error = json.dumps(error_response_content)
                        print(f"DEBUG: Yielding stream error JSON payload (OpenAI path): {json_payload_for_stream_error}")
                        yield f"data: {json_payload_for_stream_error}\n\n"
                        yield "data: [DONE]\n\n"
                return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
            else: # Not streaming
                try:
                    response = await openai_client.chat.completions.create(
                        **openai_params,
                        extra_body=openai_extra_body
                    )
                    return JSONResponse(content=response.model_dump(exclude_unset=True))
                except Exception as generate_error:
                    error_msg_generate = f"Error calling OpenAI client for {request.model}: {str(generate_error)}"
                    print(f"ERROR: {error_msg_generate}")
                    error_response = create_openai_error_response(500, error_msg_generate, "server_error")
                    return JSONResponse(status_code=500, content=error_response)
        elif is_auto_model:
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
            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, generation_config, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))