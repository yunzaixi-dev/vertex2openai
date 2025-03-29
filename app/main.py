from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import os
import google.auth
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, SafetySetting
import vertexai

app = FastAPI(title="OpenAI to Gemini Adapter")

# Define data models
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

# Configure authentication
def init_vertex_ai():
    try:
        # Check for file path credentials
        file_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if file_path and os.path.exists(file_path):
            credentials = service_account.Credentials.from_service_account_file(file_path)
            project_id = credentials.project_id
            vertexai.init(credentials=credentials, project=project_id, location="us-central1")
            print(f"Initialized Vertex AI with project: {project_id}")
            return True
        else:
            print(f"Error: GOOGLE_APPLICATION_CREDENTIALS file not found at {file_path}")
            return False
    except Exception as e:
        print(f"Error initializing authentication: {e}")
        return False

# Initialize Vertex AI at startup
@app.on_event("startup")
async def startup_event():
    if not init_vertex_ai():
        print("WARNING: Failed to initialize Vertex AI authentication")

# Conversion functions
def create_gemini_prompt(messages: List[OpenAIMessage]) -> str:
    prompt = ""
    
    # Extract system message if present
    system_message = None
    for message in messages:
        if message.role == "system":
            system_message = message.content
            break
    
    # If system message exists, prepend it
    if system_message:
        prompt += f"System: {system_message}\n\n"
    
    # Add other messages
    for message in messages:
        if message.role == "system":
            continue  # Already handled
        
        if message.role == "user":
            prompt += f"Human: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"AI: {message.content}\n"
    
    # Add final AI prompt if last message was from user
    if messages[-1].role == "user":
        prompt += "AI: "
    
    return prompt

def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config = {}
    
    if request.temperature is not None:
        config["temperature"] = request.temperature
    
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    
    if request.top_p is not None:
        config["top_p"] = request.top_p
    
    if request.top_k is not None:
        config["top_k"] = request.top_k
    
    if request.stop is not None:
        config["stop_sequences"] = request.stop
    
    return config

# Response format conversion
def convert_to_openai_format(gemini_response, model: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": gemini_response.text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Would need token counting logic
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

def convert_chunk_to_openai(chunk, model: str, response_id: str) -> str:
    chunk_content = chunk.text if hasattr(chunk, 'text') else ""
    
    chunk_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": chunk_content
                },
                "finish_reason": None
            }
        ]
    }
    
    return f"data: {json.dumps(chunk_data)}\n\n"

def create_final_chunk(model: str, response_id: str) -> str:
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    
    return f"data: {json.dumps(final_chunk)}\n\n"

# /v1/models endpoint
@app.get("/v1/models")
async def list_models():
    # Based on current information for Vertex AI models
    models = [
        {
            "id": "gemini-2.5-pro-exp-03-25",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-2.5-pro-exp-03-25",
            "parent": None,
        },
        {
            "id": "gemini-2.0-flash",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-2.0-flash",
            "parent": None,
        },
        {
            "id": "gemini-2.0-flash-lite",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-2.0-flash-lite",
            "parent": None,
        },
        {
            "id": "gemini-2.0-pro-exp-02-05",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-2.0-pro-exp-02-05",
            "parent": None,
        },
        {
            "id": "gemini-1.5-flash",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-1.5-flash",
            "parent": None,
        },
        {
            "id": "gemini-1.5-flash-8b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-1.5-flash-8b",
            "parent": None,
        },
        {
            "id": "gemini-1.5-pro",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-1.5-pro",
            "parent": None,
        },
        {
            "id": "gemini-1.0-pro-002",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-1.0-pro-002",
            "parent": None,
        },
        {
            "id": "gemini-1.0-pro-vision-001",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-1.0-pro-vision-001",
            "parent": None,
        },
        {
            "id": "gemini-embedding-exp",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": "gemini-embedding-exp",
            "parent": None,
        }
    ]
    
    return {"object": "list", "data": models}

# Main chat completion endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest):
    try:
        # Use the model name directly as provided
        gemini_model = request.model
        
        # Create generation config
        generation_config = create_generation_config(request)
        
        # Initialize Gemini model
        model = GenerativeModel(gemini_model)

        # safety_settings = [
        #     {
        #         "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #         "threshold": HarmBlockThreshold.BLOCK_NONE
        #     },
        #     {
        #         "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        #         "threshold": HarmBlockThreshold.BLOCK_NONE
        #     },
        #     {
        #         "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        #         "threshold": HarmBlockThreshold.BLOCK_NONE
        #     },
        #     {
        #         "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        #         "threshold": HarmBlockThreshold.BLOCK_NONE
        #     }
        # ]
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,threshold=HarmBlockThreshold.BLOCK_NONE)
        ]
                
        # Create prompt from messages
        prompt = create_gemini_prompt(request.messages)
        
        if request.stream:
            # Handle streaming response
            def stream_generator():
                response_id = f"chatcmpl-{int(time.time())}"

                print("aaaaa")
                
                # Generate content with streaming
                responses = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True
                )

                print("bbbbb")
                
                # Convert and yield each chunk
                for response in responses:
                    yield convert_chunk_to_openai(response, request.model, response_id)
                
                # Send final chunk
                yield create_final_chunk(request.model, response_id)
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Handle non-streaming response
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            openai_response = convert_to_openai_format(response, request.model)
            return JSONResponse(content=openai_response)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
