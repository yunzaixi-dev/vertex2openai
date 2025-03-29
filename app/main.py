from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import os
import glob
import random
import google.auth
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, SafetySetting
import vertexai

app = FastAPI(title="OpenAI to Gemini Adapter")

# Credential Manager for handling multiple service accounts
class CredentialManager:
    def __init__(self, default_credentials_dir="/app/credentials"):
        # Use environment variable if set, otherwise use default
        self.credentials_dir = os.environ.get("CREDENTIALS_DIR", default_credentials_dir)
        self.credentials_files = []
        self.current_index = 0
        self.credentials = None
        self.project_id = None
        self.load_credentials_list()
    
    def load_credentials_list(self):
        """Load the list of available credential files"""
        # Look for all .json files in the credentials directory
        pattern = os.path.join(self.credentials_dir, "*.json")
        self.credentials_files = glob.glob(pattern)
        
        if not self.credentials_files:
            print(f"No credential files found in {self.credentials_dir}")
            return False
        
        print(f"Found {len(self.credentials_files)} credential files: {[os.path.basename(f) for f in self.credentials_files]}")
        return True
    
    def refresh_credentials_list(self):
        """Refresh the list of credential files (useful if files are added/removed)"""
        old_count = len(self.credentials_files)
        self.load_credentials_list()
        new_count = len(self.credentials_files)
        
        if old_count != new_count:
            print(f"Credential files updated: {old_count} -> {new_count}")
        
        return len(self.credentials_files) > 0
    
    def get_next_credentials(self):
        """Rotate to the next credential file and load it"""
        if not self.credentials_files:
            return None, None
        
        # Get the next credential file in rotation
        file_path = self.credentials_files[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.credentials_files)
        
        try:
            credentials = service_account.Credentials.from_service_account_file(file_path)
            project_id = credentials.project_id
            print(f"Loaded credentials from {file_path} for project: {project_id}")
            self.credentials = credentials
            self.project_id = project_id
            return credentials, project_id
        except Exception as e:
            print(f"Error loading credentials from {file_path}: {e}")
            # Try the next file if this one fails
            if len(self.credentials_files) > 1:
                print("Trying next credential file...")
                return self.get_next_credentials()
            return None, None
    
    def get_random_credentials(self):
        """Get a random credential file and load it"""
        if not self.credentials_files:
            return None, None
        
        # Choose a random credential file
        file_path = random.choice(self.credentials_files)
        
        try:
            credentials = service_account.Credentials.from_service_account_file(file_path)
            project_id = credentials.project_id
            print(f"Loaded credentials from {file_path} for project: {project_id}")
            self.credentials = credentials
            self.project_id = project_id
            return credentials, project_id
        except Exception as e:
            print(f"Error loading credentials from {file_path}: {e}")
            # Try another random file if this one fails
            if len(self.credentials_files) > 1:
                print("Trying another credential file...")
                return self.get_random_credentials()
            return None, None

# Initialize the credential manager
credential_manager = CredentialManager()

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
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    logprobs: Optional[int] = None
    response_logprobs: Optional[bool] = None
    n: Optional[int] = None  # Maps to candidate_count in Vertex AI

# Configure authentication
def init_vertex_ai():
    try:
        # First try to use the credential manager to get credentials
        credentials, project_id = credential_manager.get_next_credentials()
        
        if credentials and project_id:
            vertexai.init(credentials=credentials, project=project_id, location="us-central1")
            print(f"Initialized Vertex AI with project: {project_id}")
            return True
        
        # Fall back to environment variable if credential manager fails
        file_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if file_path and os.path.exists(file_path):
            credentials = service_account.Credentials.from_service_account_file(file_path)
            project_id = credentials.project_id
            vertexai.init(credentials=credentials, project=project_id, location="us-central1")
            print(f"Initialized Vertex AI with project: {project_id} (using GOOGLE_APPLICATION_CREDENTIALS)")
            return True
        else:
            print(f"Error: No valid credentials found. GOOGLE_APPLICATION_CREDENTIALS file not found at {file_path}")
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
    
    # Basic parameters that were already supported
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
    
    # Additional parameters with direct mappings
    if request.presence_penalty is not None:
        config["presence_penalty"] = request.presence_penalty
    
    if request.frequency_penalty is not None:
        config["frequency_penalty"] = request.frequency_penalty
    
    if request.seed is not None:
        config["seed"] = request.seed
    
    if request.logprobs is not None:
        config["logprobs"] = request.logprobs
    
    if request.response_logprobs is not None:
        config["response_logprobs"] = request.response_logprobs
    
    # Map OpenAI's 'n' parameter to Vertex AI's 'candidate_count'
    if request.n is not None:
        config["candidate_count"] = request.n
    
    return config

# Response format conversion
def convert_to_openai_format(gemini_response, model: str) -> Dict[str, Any]:
    # Handle multiple candidates if present
    if hasattr(gemini_response, 'candidates') and len(gemini_response.candidates) > 1:
        choices = []
        for i, candidate in enumerate(gemini_response.candidates):
            choices.append({
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": candidate.text
                },
                "finish_reason": "stop"
            })
    else:
        # Handle single response (backward compatibility)
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": gemini_response.text
                },
                "finish_reason": "stop"
            }
        ]
    
    # Include logprobs if available
    for i, choice in enumerate(choices):
        if hasattr(gemini_response, 'candidates') and i < len(gemini_response.candidates):
            candidate = gemini_response.candidates[i]
            if hasattr(candidate, 'logprobs'):
                choice["logprobs"] = candidate.logprobs
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,  # Would need token counting logic
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

def convert_chunk_to_openai(chunk, model: str, response_id: str, candidate_index: int = 0) -> str:
    chunk_content = chunk.text if hasattr(chunk, 'text') else ""
    
    chunk_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": candidate_index,
                "delta": {
                    "content": chunk_content
                },
                "finish_reason": None
            }
        ]
    }
    
    # Add logprobs if available
    if hasattr(chunk, 'logprobs'):
        chunk_data["choices"][0]["logprobs"] = chunk.logprobs
    
    return f"data: {json.dumps(chunk_data)}\n\n"

def create_final_chunk(model: str, response_id: str, candidate_count: int = 1) -> str:
    choices = []
    for i in range(candidate_count):
        choices.append({
            "index": i,
            "delta": {},
            "finish_reason": "stop"
        })
    
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices
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
# OpenAI-compatible error response
def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
            "param": None,
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest):
    try:
        # Validate model availability
        if not request.model or not any(model["id"] == request.model for model in list_models().get("data", [])):
            error_response = create_openai_error_response(
                400, f"Model '{request.model}' not found", "invalid_request_error"
            )
            return JSONResponse(status_code=400, content=error_response)
        
        # Use the model name directly as provided
        gemini_model = request.model
        
        # Create generation config
        generation_config = create_generation_config(request)
        
        # Get fresh credentials for this request
        credentials, project_id = credential_manager.get_next_credentials()
        
        if not credentials or not project_id:
            error_response = create_openai_error_response(
                500, "Failed to obtain valid credentials", "server_error"
            )
            return JSONResponse(status_code=500, content=error_response)
        
        # Initialize Vertex AI with the rotated credentials
        try:
            vertexai.init(credentials=credentials, project=project_id, location="us-central1")
            print(f"Using credentials for project: {project_id}")
        except Exception as auth_error:
            error_response = create_openai_error_response(
                500, f"Failed to initialize authentication: {str(auth_error)}", "server_error"
            )
            return JSONResponse(status_code=500, content=error_response)
        
        # Initialize Gemini model
        try:
            model = GenerativeModel(gemini_model)
        except Exception as model_error:
            error_response = create_openai_error_response(
                400, f"Failed to initialize model '{gemini_model}': {str(model_error)}", "invalid_request_error"
            )
            return JSONResponse(status_code=400, content=error_response)

        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,threshold=HarmBlockThreshold.OFF),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,threshold=HarmBlockThreshold.OFF),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,threshold=HarmBlockThreshold.OFF),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,threshold=HarmBlockThreshold.OFF)
        ]
                
        # Create prompt from messages
        prompt = create_gemini_prompt(request.messages)
        
        if request.stream:
            # Handle streaming response
            async def stream_generator():
                response_id = f"chatcmpl-{int(time.time())}"
                candidate_count = request.n or 1
                
                try:
                    # For streaming, we can only handle one candidate at a time
                    # If multiple candidates are requested, we'll generate them sequentially
                    for candidate_index in range(candidate_count):
                        # Generate content with streaming
                        responses = model.generate_content(
                            prompt,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                            stream=True
                        )
                        
                        # Convert and yield each chunk
                        for response in responses:
                            yield convert_chunk_to_openai(response, request.model, response_id, candidate_index)
                    
                    # Send final chunk with all candidates
                    yield create_final_chunk(request.model, response_id, candidate_count)
                    yield "data: [DONE]\n\n"
                
                except Exception as stream_error:
                    # Format streaming errors in SSE format
                    error_msg = f"Error during streaming: {str(stream_error)}"
                    print(error_msg)
                    error_response = create_openai_error_response(500, error_msg, "server_error")
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Handle non-streaming response
            try:
                # If multiple candidates are requested, set candidate_count
                if request.n and request.n > 1:
                    # Make sure generation_config has candidate_count set
                    if "candidate_count" not in generation_config:
                        generation_config["candidate_count"] = request.n
                
                response = model.generate_content(
                    prompt,
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                
                openai_response = convert_to_openai_format(response, request.model)
                return JSONResponse(content=openai_response)
            except Exception as generate_error:
                error_msg = f"Error generating content: {str(generate_error)}"
                print(error_msg)
                error_response = create_openai_error_response(500, error_msg, "server_error")
                return JSONResponse(status_code=500, content=error_response)
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(error_msg)
        error_response = create_openai_error_response(500, error_msg, "server_error")
        return JSONResponse(status_code=500, content=error_response)

# Health check endpoint
@app.get("/health")
def health_check():
    # Refresh the credentials list to get the latest status
    credential_manager.refresh_credentials_list()
    
    return {
        "status": "ok",
        "credentials": {
            "available": len(credential_manager.credentials_files),
            "files": [os.path.basename(f) for f in credential_manager.credentials_files],
            "current_index": credential_manager.current_index
        }
    }
