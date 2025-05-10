import time
from fastapi import APIRouter, Depends
# from typing import List, Dict, Any # Removed as unused

from auth import get_api_key # Changed from relative

router = APIRouter()

@router.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    # This model list should ideally be dynamic or configurable
    models_data = [
        {"id": "gemini-2.5-pro-exp-03-25", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-exp-03-25-search", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-exp-03-25-encrypt", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-exp-03-25-encrypt-full", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-exp-03-25-auto", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-03-25", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-03-25-search", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-03-25-encrypt", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-03-25-encrypt-full", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-03-25-auto", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-05-06", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-05-06-search", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-05-06-encrypt", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-05-06-encrypt-full", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-pro-preview-05-06-auto", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.0-flash", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.0-flash-search", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.0-flash-lite", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.0-flash-lite-search", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.0-pro-exp-02-05", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-1.5-flash", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-flash-preview-04-17", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-flash-preview-04-17-encrypt", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-flash-preview-04-17-nothinking", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-2.5-flash-preview-04-17-max", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-1.5-flash-8b", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-1.5-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-1.0-pro-002", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-1.0-pro-vision-001", "object": "model", "created": int(time.time()), "owned_by": "google"},
        {"id": "gemini-embedding-exp", "object": "model", "created": int(time.time()), "owned_by": "google"}
    ]
    # Add root and parent for consistency with OpenAI-like response
    for model_info in models_data:
        model_info.setdefault("permission", [])
        model_info.setdefault("root", model_info["id"]) # Typically the model ID itself
        model_info.setdefault("parent", None) # Typically None for base models
    return {"object": "list", "data": models_data}