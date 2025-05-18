import time
from fastapi import APIRouter, Depends, Request # Added Request
from typing import List, Dict, Any
from auth import get_api_key
from model_loader import get_vertex_models, get_vertex_express_models, refresh_models_config_cache
import config as app_config # Import config
from credentials_manager import CredentialManager # To check its type

router = APIRouter()

@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    await refresh_models_config_cache()
    
    OPENAI_DIRECT_SUFFIX = "-openai"
    EXPERIMENTAL_MARKER = "-exp-"
    PAY_PREFIX = "[PAY]"
    # Access credential_manager from app state
    credential_manager_instance: CredentialManager = fastapi_request.app.state.credential_manager

    has_sa_creds = credential_manager_instance.get_total_credentials() > 0
    has_express_key = bool(app_config.VERTEX_EXPRESS_API_KEY_VAL)

    raw_vertex_models = await get_vertex_models()
    raw_express_models = await get_vertex_express_models()
    
    candidate_model_ids = set()

    if has_express_key:
        candidate_model_ids.update(raw_express_models)
        # If *only* express key is available, only express models (and their variants) should be listed.
        # The current `vertex_model_ids` from remote config might contain non-express models.
        # The `get_vertex_express_models()` should be the source of truth for express-eligible base models.
        if not has_sa_creds:
            # Only list models that are explicitly in the express list.
            # Suffix generation will apply only to these if they are not gemini-2.0
            all_model_ids = set(raw_express_models)
        else:
            # Both SA and Express are available, combine all known models
            all_model_ids = set(raw_vertex_models + raw_express_models)
    elif has_sa_creds:
        # Only SA creds available, use all vertex_models (which might include express-eligible ones)
        all_model_ids = set(raw_vertex_models)
    else:
        # No credentials available
        all_model_ids = set()
    
    # Create extended model list with variations (search, encrypt, auto etc.)
    # This logic might need to be more sophisticated based on actual supported features per base model.
    # For now, let's assume for each base model, we might have these variations.
    # A better approach would be if the remote config specified these variations.
    
    dynamic_models_data: List[Dict[str, Any]] = []
    current_time = int(time.time())

    # Add base models and their variations
    for original_model_id in sorted(list(all_model_ids)):
        current_display_prefix = ""
        # Only add PAY_PREFIX if the model is not already an EXPRESS model (which has its own prefix)
        if not original_model_id.startswith("[EXPRESS]") and \
           has_sa_creds and not has_express_key and EXPERIMENTAL_MARKER not in original_model_id:
            current_display_prefix = PAY_PREFIX
        
        base_display_id = f"{current_display_prefix}{original_model_id}"
        
        dynamic_models_data.append({
            "id": base_display_id, "object": "model", "created": current_time, "owned_by": "google",
            "permission": [], "root": original_model_id, "parent": None
        })
        
        # Conditionally add common variations (standard suffixes)
        if not original_model_id.startswith("gemini-2.0"): # Suffix rules based on original_model_id
            standard_suffixes = ["-search", "-encrypt", "-encrypt-full", "-auto"]
            for suffix in standard_suffixes:
                # Suffix is applied to the original model ID part
                suffixed_model_part = f"{original_model_id}{suffix}"
                # Then the whole thing is prefixed
                final_suffixed_display_id = f"{current_display_prefix}{suffixed_model_part}"
                
                # Check if this suffixed ID is already in all_model_ids (unlikely with prefix) or already added
                if final_suffixed_display_id not in all_model_ids and not any(m['id'] == final_suffixed_display_id for m in dynamic_models_data):
                    dynamic_models_data.append({
                        "id": final_suffixed_display_id, "object": "model", "created": current_time, "owned_by": "google",
                        "permission": [], "root": original_model_id, "parent": None
                    })
        
        # Apply special suffixes for models starting with "gemini-2.5-flash"
        if original_model_id.startswith("gemini-2.5-flash"): # Suffix rules based on original_model_id
            special_flash_suffixes = ["-nothinking", "-max"]
            for special_suffix in special_flash_suffixes:
                suffixed_model_part = f"{original_model_id}{special_suffix}"
                final_special_suffixed_display_id = f"{current_display_prefix}{suffixed_model_part}"

                if final_special_suffixed_display_id not in all_model_ids and not any(m['id'] == final_special_suffixed_display_id for m in dynamic_models_data):
                    dynamic_models_data.append({
                        "id": final_special_suffixed_display_id, "object": "model", "created": current_time, "owned_by": "google",
                        "permission": [], "root": original_model_id, "parent": None
                    })

        # Ensure uniqueness again after adding suffixes
        # Add OpenAI direct variations if SA creds are available
        if has_sa_creds: # OpenAI direct mode only works with SA credentials
            # `all_model_ids` contains the comprehensive list of base models that are eligible based on current credentials
            # We iterate through this to determine which ones get an -openai variation.
            # `raw_vertex_models` is used here to ensure we only add -openai suffix to models that are
            # fundamentally Vertex models, not just any model that might appear in `all_model_ids` (e.g. from Express list exclusively)
            # if express only key is provided.
            # We iterate through the base models from the main Vertex list.
            for base_model_id_for_openai in raw_vertex_models: # Iterate through original list of GAIA/Vertex base models
                display_model_id = ""
                if EXPERIMENTAL_MARKER in base_model_id_for_openai:
                    display_model_id = f"{base_model_id_for_openai}{OPENAI_DIRECT_SUFFIX}"
                else:
                    display_model_id = f"{PAY_PREFIX}{base_model_id_for_openai}{OPENAI_DIRECT_SUFFIX}"
                
                # Check if already added (e.g. if remote config somehow already listed it or added as a base model)
                if display_model_id and not any(m['id'] == display_model_id for m in dynamic_models_data):
                    dynamic_models_data.append({
                        "id": display_model_id, "object": "model", "created": current_time, "owned_by": "google",
                        "permission": [], "root": base_model_id_for_openai, "parent": None
                    })
    # final_models_data_map = {m["id"]: m for m in dynamic_models_data}
    # model_list = list(final_models_data_map.values())
    # model_list.sort()
    
    return {"object": "list", "data": sorted(dynamic_models_data, key=lambda x: x['id'])}