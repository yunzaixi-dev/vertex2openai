import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, List, Union, Optional, AsyncGenerator
import time

# Global cache for project IDs: {api_key: project_id}
PROJECT_ID_CACHE: Dict[str, str] = {}


class DirectVertexClient:
    """
    A client that connects to Vertex AI using direct URLs instead of the SDK.
    Mimics the interface of genai.Client for seamless integration.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.project_id: Optional[str] = None
        self.base_url = "https://aiplatform.googleapis.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        # Mimic the model_name attribute that might be accessed
        self.model_name = "direct_vertex_client"
        
        # Create nested structure to mimic genai.Client interface
        self.aio = self._AioNamespace(self)
    
    class _AioNamespace:
        def __init__(self, parent):
            self.parent = parent
            self.models = self._ModelsNamespace(parent)
        
        class _ModelsNamespace:
            def __init__(self, parent):
                self.parent = parent
            
            async def generate_content(self, model: str, contents: Any, config: Dict[str, Any]) -> Any:
                """Non-streaming content generation"""
                return await self.parent._generate_content(model, contents, config, stream=False)
            
            async def generate_content_stream(self, model: str, contents: Any, config: Dict[str, Any]):
                """Streaming content generation - returns an async generator"""
                # This needs to be an async method that returns the generator
                # to match the SDK's interface where you await the method call
                return self.parent._generate_content_stream(model, contents, config)
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def discover_project_id(self) -> None:
        """Discover project ID by triggering an intentional error"""
        # Check cache first
        if self.api_key in PROJECT_ID_CACHE:
            self.project_id = PROJECT_ID_CACHE[self.api_key]
            print(f"INFO: Using cached project ID: {self.project_id}")
            return
        
        await self._ensure_session()
        
        # Use a non-existent model to trigger error
        error_url = f"{self.base_url}/publishers/google/models/gemini-2.7-pro-preview-05-06:streamGenerateContent?key={self.api_key}"
        
        try:
            # Send minimal request to trigger error
            payload = {
                "contents": [{"role": "user", "parts": [{"text": "test"}]}]
            }
            
            async with self.session.post(error_url, json=payload) as response:
                response_text = await response.text()
                
                try:
                    # Try to parse as JSON first
                    error_data = json.loads(response_text)
                    
                    # Handle array response format
                    if isinstance(error_data, list) and len(error_data) > 0:
                        error_data = error_data[0]
                    
                    if "error" in error_data:
                        error_message = error_data["error"].get("message", "")
                        # Extract project ID from error message
                        # Pattern: "projects/39982734461/locations/..."
                        match = re.search(r'projects/(\d+)/locations/', error_message)
                        if match:
                            self.project_id = match.group(1)
                            PROJECT_ID_CACHE[self.api_key] = self.project_id
                            print(f"INFO: Discovered project ID: {self.project_id}")
                            return
                except json.JSONDecodeError:
                    # If not JSON, try to find project ID in raw text
                    match = re.search(r'projects/(\d+)/locations/', response_text)
                    if match:
                        self.project_id = match.group(1)
                        PROJECT_ID_CACHE[self.api_key] = self.project_id
                        print(f"INFO: Discovered project ID from raw response: {self.project_id}")
                        return
                
                raise Exception(f"Failed to discover project ID. Status: {response.status}, Response: {response_text[:500]}")
                
        except Exception as e:
            print(f"ERROR: Failed to discover project ID: {e}")
            raise
    
    
    async def _generate_content(self, model: str, contents: Any, config: Dict[str, Any], stream: bool = False) -> Any:
        """Internal method for content generation"""
        if not self.project_id:
            raise ValueError("Project ID not discovered. Call discover_project_id() first.")
        
        await self._ensure_session()
        
        # Build URL
        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/projects/{self.project_id}/locations/global/publishers/google/models/{model}:{endpoint}?key={self.api_key}"
        
        # The contents and config are already in the correct format
        # But config parameters need to be nested under generationConfig for the REST API
        payload = {
            "contents": contents
        }
        
        # Extract specific config sections
        if "system_instruction" in config:
            payload["systemInstruction"] = config["system_instruction"]
        
        if "safety_settings" in config:
            payload["safetySettings"] = config["safety_settings"]
        
        if "tools" in config:
            payload["tools"] = config["tools"]
        
        # All other config goes under generationConfig
        generation_config = {}
        for key, value in config.items():
            if key not in ["system_instruction", "safety_settings", "tools"]:
                generation_config[key] = value
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                    raise Exception(f"Vertex AI API error: {error_msg}")
                
                # Get the JSON response
                response_data = await response.json()
                
                # Convert dict to object with attributes for compatibility
                return self._dict_to_obj(response_data)
                
        except Exception as e:
            print(f"ERROR: Direct Vertex API call failed: {e}")
            raise
    
    def _dict_to_obj(self, data):
        """Convert a dict to an object with attributes"""
        if isinstance(data, dict):
            # Create a simple object that allows attribute access
            class AttrDict:
                def __init__(self, d):
                    for key, value in d.items():
                        setattr(self, key, self._convert_value(value))
                
                def _convert_value(self, value):
                    if isinstance(value, dict):
                        return AttrDict(value)
                    elif isinstance(value, list):
                        return [self._convert_value(item) for item in value]
                    else:
                        return value
            
            return AttrDict(data)
        elif isinstance(data, list):
            return [self._dict_to_obj(item) for item in data]
        else:
            return data
    
    async def _generate_content_stream(self, model: str, contents: Any, config: Dict[str, Any]) -> AsyncGenerator:
        """Internal method for streaming content generation"""
        if not self.project_id:
            raise ValueError("Project ID not discovered. Call discover_project_id() first.")
        
        await self._ensure_session()
        
        # Build URL for streaming
        url = f"{self.base_url}/projects/{self.project_id}/locations/global/publishers/google/models/{model}:streamGenerateContent?key={self.api_key}"
        
        # The contents and config are already in the correct format
        # But config parameters need to be nested under generationConfig for the REST API
        payload = {
            "contents": contents
        }
        
        # Extract specific config sections
        if "system_instruction" in config:
            payload["systemInstruction"] = config["system_instruction"]
        
        if "safety_settings" in config:
            payload["safetySettings"] = config["safety_settings"]
        
        if "tools" in config:
            payload["tools"] = config["tools"]
        
        # All other config goes under generationConfig
        generation_config = {}
        for key, value in config.items():
            if key not in ["system_instruction", "safety_settings", "tools"]:
                generation_config[key] = value
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_data = await response.json()
                    # Handle array response format
                    if isinstance(error_data, list) and len(error_data) > 0:
                        error_data = error_data[0]
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}") if isinstance(error_data, dict) else str(error_data)
                    raise Exception(f"Vertex AI API error: {error_msg}")
                
                # The Vertex AI streaming endpoint returns Server-Sent Events
                # We need to parse these and yield them as objects
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode('utf-8')
                    
                    # Process complete SSE messages
                    while '\n\n' in buffer:
                        message, buffer = buffer.split('\n\n', 1)
                        
                        if not message.strip():
                            continue
                        
                        # Parse SSE format
                        if message.startswith('data: '):
                            data_str = message[6:]
                            
                            if data_str.strip() == '[DONE]':
                                return
                            
                            try:
                                # Parse JSON and convert to object
                                chunk_data = json.loads(data_str)
                                yield self._dict_to_obj(chunk_data)
                            except json.JSONDecodeError:
                                continue
                            
        except Exception as e:
            print(f"ERROR: Direct Vertex streaming API call failed: {e}")
            raise