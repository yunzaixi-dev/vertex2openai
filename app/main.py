from fastapi import FastAPI, Depends # Depends might be used by root endpoint
# from fastapi.responses import JSONResponse # Not used
from fastapi.middleware.cors import CORSMiddleware
# import asyncio # Not used
# import os # Not used


# Local module imports
from auth import get_api_key # Potentially for root endpoint
from credentials_manager import CredentialManager
from vertex_ai_init import init_vertex_ai

# Routers
from routes import models_api
from routes import chat_api

# import config as app_config # Not directly used in main.py

app = FastAPI(title="OpenAI to Gemini Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

credential_manager = CredentialManager()
app.state.credential_manager = credential_manager # Store manager on app state

# Include API routers
app.include_router(models_api.router) 
app.include_router(chat_api.router)

@app.on_event("startup")
async def startup_event():
    if init_vertex_ai(credential_manager):
        print("INFO: Fallback Vertex AI client initialization check completed successfully.")
    else:
        print("ERROR: Failed to initialize a fallback Vertex AI client. API will likely fail.")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "OpenAI to Gemini Adapter is running."
    }
