import time
from fastapi import APIRouter, Depends
from typing import List, Dict, Any # Will be needed for constructing model dicts
from auth import get_api_key
from model_loader import get_vertex_models, get_vertex_express_models, refresh_models_config_cache # Changed from relative

router = APIRouter()

@router.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    # Attempt to refresh the cache. If it fails, getters will use the old cache.
    await refresh_models_config_cache()
    
    vertex_model_ids = await get_vertex_models()
    vertex_express_model_ids = await get_vertex_express_models()
    
    # Combine and unique model IDs.
    # We should also consider creating the OpenAI model suffixes (-search, -encrypt, -auto)
    # based on the base models available, similar to how chat_api.py currently does.
    # For simplicity here, we'll list all unique base models from the config
    # and then also list the specific variations.
    
    all_model_ids = set(vertex_model_ids + vertex_express_model_ids)
    
    # Create extended model list with variations (search, encrypt, auto etc.)
    # This logic might need to be more sophisticated based on actual supported features per base model.
    # For now, let's assume for each base model, we might have these variations.
    # A better approach would be if the remote config specified these variations.
    
    dynamic_models_data: List[Dict[str, Any]] = []
    current_time = int(time.time())

    # Add base models and their variations
    for model_id in sorted(list(all_model_ids)):
        dynamic_models_data.append({
            "id": model_id, "object": "model", "created": current_time, "owned_by": "google",
            "permission": [], "root": model_id, "parent": None
        })
        
        # Conditionally add common variations (standard suffixes)
        if not model_id.startswith("gemini-2.0"):
            standard_suffixes = ["-search", "-encrypt", "-encrypt-full", "-auto"]
            for suffix in standard_suffixes:
                suffixed_id = f"{model_id}{suffix}"
                # Check if this suffixed ID is already in all_model_ids (fetched from remote) or already added to dynamic_models_data
                if suffixed_id not in all_model_ids and not any(m['id'] == suffixed_id for m in dynamic_models_data):
                    dynamic_models_data.append({
                        "id": suffixed_id, "object": "model", "created": current_time, "owned_by": "google",
                        "permission": [], "root": model_id, "parent": None
                    })
        
        # Apply special suffixes for models starting with "gemini-2.5-flash"
        if model_id.startswith("gemini-2.5-flash"):
            special_flash_suffixes = ["-nothinking", "-max"]
            for special_suffix in special_flash_suffixes:
                suffixed_id = f"{model_id}{special_suffix}"
                if suffixed_id not in all_model_ids and not any(m['id'] == suffixed_id for m in dynamic_models_data):
                    dynamic_models_data.append({
                        "id": suffixed_id, "object": "model", "created": current_time, "owned_by": "google",
                        "permission": [], "root": model_id, "parent": None
                    })

    # Ensure uniqueness again after adding suffixes
    final_models_data_map = {m["id"]: m for m in dynamic_models_data}
    
    return {"object": "list", "data": list(final_models_data_map.values())}