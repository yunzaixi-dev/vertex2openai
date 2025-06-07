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
    
    def _convert_contents(self, contents: Any) -> List[Dict[str, Any]]:
        """Convert SDK Content objects to REST API format"""
        if isinstance(contents, list):
            return [self._convert_content_item(item) for item in contents]
        else:
            return [self._convert_content_item(contents)]
    
    def _convert_content_item(self, content: Any) -> Dict[str, Any]:
        """Convert a single content item to REST API format"""
        if isinstance(content, dict):
            return content
        
        # Handle SDK Content objects
        result = {}
        if hasattr(content, 'role'):
            result['role'] = content.role
        if hasattr(content, 'parts'):
            result['parts'] = []
            for part in content.parts:
                if isinstance(part, dict):
                    result['parts'].append(part)
                elif hasattr(part, 'text'):
                    result['parts'].append({'text': part.text})
                elif hasattr(part, 'inline_data'):
                    result['parts'].append({
                        'inline_data': {
                            'mime_type': part.inline_data.mime_type,
                            'data': part.inline_data.data
                        }
                    })
        return result
    
    def _convert_safety_settings(self, safety_settings: Any) -> List[Dict[str, str]]:
        """Convert SDK SafetySetting objects to REST API format"""
        if not safety_settings:
            return []
        
        result = []
        for setting in safety_settings:
            if isinstance(setting, dict):
                result.append(setting)
            elif hasattr(setting, 'category') and hasattr(setting, 'threshold'):
                # Convert SDK SafetySetting to dict
                result.append({
                    'category': setting.category,
                    'threshold': setting.threshold
                })
        return result
    
    def _convert_tools(self, tools: Any) -> List[Dict[str, Any]]:
        """Convert SDK Tool objects to REST API format"""
        if not tools:
            return []
        
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            else:
                # Convert SDK Tool object to dict
                result.append(self._convert_tool_item(tool))
        return result
    
    def _convert_tool_item(self, tool: Any) -> Dict[str, Any]:
        """Convert a single tool item to REST API format"""
        if isinstance(tool, dict):
            return tool
        
        tool_dict = {}
        
        # Convert all non-private attributes
        if hasattr(tool, '__dict__'):
            for attr_name, attr_value in tool.__dict__.items():
                if not attr_name.startswith('_'):
                    # Convert attribute names from snake_case to camelCase for REST API
                    rest_api_name = self._to_camel_case(attr_name)
                    
                    # Special handling for known types
                    if attr_name == 'google_search' and attr_value is not None:
                        tool_dict[rest_api_name] = {}  # GoogleSearch is empty object in REST
                    elif attr_name == 'function_declarations' and attr_value is not None:
                        tool_dict[rest_api_name] = attr_value
                    elif attr_value is not None:
                        # Recursively convert any other SDK objects
                        tool_dict[rest_api_name] = self._convert_sdk_object(attr_value)
        
        return tool_dict
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def _convert_sdk_object(self, obj: Any) -> Any:
        """Generic SDK object converter"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._convert_sdk_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_sdk_object(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert SDK object to dict
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    result[self._to_camel_case(key)] = self._convert_sdk_object(value)
            return result
        else:
            return obj
    
    async def _generate_content(self, model: str, contents: Any, config: Dict[str, Any], stream: bool = False) -> Any:
        """Internal method for content generation"""
        if not self.project_id:
            raise ValueError("Project ID not discovered. Call discover_project_id() first.")
        
        await self._ensure_session()
        
        # Build URL
        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/projects/{self.project_id}/locations/global/publishers/google/models/{model}:{endpoint}?key={self.api_key}"
        
        # Convert contents to REST API format
        payload = {
            "contents": self._convert_contents(contents)
        }
        
        # Extract specific config sections
        if "system_instruction" in config:
            # System instruction should be a content object
            if isinstance(config["system_instruction"], dict):
                payload["systemInstruction"] = config["system_instruction"]
            else:
                payload["systemInstruction"] = self._convert_content_item(config["system_instruction"])
        
        if "safety_settings" in config:
            payload["safetySettings"] = self._convert_safety_settings(config["safety_settings"])
        
        if "tools" in config:
            payload["tools"] = self._convert_tools(config["tools"])
        
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
        
        # Convert contents to REST API format
        payload = {
            "contents": self._convert_contents(contents)
        }
        
        # Extract specific config sections
        if "system_instruction" in config:
            # System instruction should be a content object
            if isinstance(config["system_instruction"], dict):
                payload["systemInstruction"] = config["system_instruction"]
            else:
                payload["systemInstruction"] = self._convert_content_item(config["system_instruction"])
        
        if "safety_settings" in config:
            payload["safetySettings"] = self._convert_safety_settings(config["safety_settings"])
        
        if "tools" in config:
            payload["tools"] = self._convert_tools(config["tools"])
        
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
                
                # The Vertex AI streaming endpoint returns JSON array elements
                # We need to parse these as they arrive
                buffer = ""
                
                async for chunk in response.content.iter_any():
                    decoded_chunk = chunk.decode('utf-8')
                    buffer += decoded_chunk
                    
                    # Try to extract complete JSON objects from the buffer
                    while True:
                        # Skip whitespace and array brackets
                        buffer = buffer.lstrip()
                        if buffer.startswith('['):
                            buffer = buffer[1:].lstrip()
                            continue
                        if buffer.startswith(']'):
                            # End of array
                            return
                        
                        # Skip comma and whitespace between objects
                        if buffer.startswith(','):
                            buffer = buffer[1:].lstrip()
                            continue
                        
                        # Look for a complete JSON object
                        if buffer.startswith('{'):
                            # Find the matching closing brace
                            brace_count = 0
                            in_string = False
                            escape_next = False
                            
                            for i, char in enumerate(buffer):
                                if escape_next:
                                    escape_next = False
                                    continue
                                
                                if char == '\\' and in_string:
                                    escape_next = True
                                    continue
                                
                                if char == '"' and not in_string:
                                    in_string = True
                                elif char == '"' and in_string:
                                    in_string = False
                                elif char == '{' and not in_string:
                                    brace_count += 1
                                elif char == '}' and not in_string:
                                    brace_count -= 1
                                    
                                    if brace_count == 0:
                                        # Found complete object
                                        obj_str = buffer[:i+1]
                                        buffer = buffer[i+1:]
                                        
                                        try:
                                            chunk_data = json.loads(obj_str)
                                            converted_obj = self._dict_to_obj(chunk_data)
                                            yield converted_obj
                                        except json.JSONDecodeError as e:
                                            print(f"ERROR: DirectVertexClient - Failed to parse JSON: {e}")
                                        
                                        break
                            else:
                                # No complete object found, need more data
                                break
                        else:
                            # No more objects to process in current buffer
                            break
                            
        except Exception as e:
            print(f"ERROR: Direct Vertex streaming API call failed: {e}")
            raise