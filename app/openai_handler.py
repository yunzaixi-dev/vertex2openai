"""
OpenAI handler module for creating clients and processing OpenAI Direct mode responses.
This module encapsulates all OpenAI-specific logic that was previously in chat_api.py.
"""
import json
import time
import asyncio
from typing import Dict, Any, AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse
import openai
from google.auth.transport.requests import Request as AuthRequest

from models import OpenAIRequest
from config import VERTEX_REASONING_TAG
import config as app_config
from api_helpers import (
    create_openai_error_response,
    openai_fake_stream_generator,
    StreamingReasoningProcessor
)
from message_processing import extract_reasoning_by_tags
from credentials_manager import _refresh_auth


class OpenAIDirectHandler:
    """Handles OpenAI Direct mode operations including client creation and response processing."""
    
    def __init__(self, credential_manager):
        self.credential_manager = credential_manager
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": 'HARM_CATEGORY_CIVIC_INTEGRITY', "threshold": 'OFF'}
        ]
    
    def create_openai_client(self, project_id: str, gcp_token: str, location: str = "global") -> openai.AsyncOpenAI:
        """Create an OpenAI client configured for Vertex AI endpoint."""
        endpoint_url = (
            f"https://aiplatform.googleapis.com/v1beta1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )
        
        return openai.AsyncOpenAI(
            base_url=endpoint_url,
            api_key=gcp_token,  # OAuth token
        )
    
    def prepare_openai_params(self, request: OpenAIRequest, model_id: str) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        params = {
            "model": model_id,
            "messages": [msg.model_dump(exclude_unset=True) for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": request.stream,
            "stop": request.stop,
            "seed": request.seed,
            "n": request.n,
        }
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    
    def prepare_extra_body(self) -> Dict[str, Any]:
        """Prepare extra body parameters for OpenAI API call."""
        return {
            "extra_body": {
                'google': {
                    'safety_settings': self.safety_settings,
                    'thought_tag_marker': VERTEX_REASONING_TAG
                }
            }
        }
    
    async def handle_streaming_response(
        self, 
        openai_client: openai.AsyncOpenAI,
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> StreamingResponse:
        """Handle streaming responses for OpenAI Direct mode."""
        if app_config.FAKE_STREAMING_ENABLED:
            print(f"INFO: OpenAI Fake Streaming (SSE Simulation) ENABLED for model '{request.model}'.")
            return StreamingResponse(
                openai_fake_stream_generator(
                    openai_client=openai_client,
                    openai_params=openai_params,
                    openai_extra_body=openai_extra_body,
                    request_obj=request,
                    is_auto_attempt=False
                ),
                media_type="text/event-stream"
            )
        else:
            print(f"INFO: OpenAI True Streaming ENABLED for model '{request.model}'.")
            return StreamingResponse(
                self._true_stream_generator(openai_client, openai_params, openai_extra_body, request),
                media_type="text/event-stream"
            )
    
    async def _true_stream_generator(
        self,
        openai_client: openai.AsyncOpenAI,
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> AsyncGenerator[str, None]:
        """Generate true streaming response."""
        try:
            # Ensure stream=True is explicitly passed for real streaming
            openai_params_for_stream = {**openai_params, "stream": True}
            stream_response = await openai_client.chat.completions.create(
                **openai_params_for_stream,
                extra_body=openai_extra_body
            )
            
            # Create processor for tag-based extraction across chunks
            reasoning_processor = StreamingReasoningProcessor(VERTEX_REASONING_TAG)
            chunk_count = 0
            has_sent_content = False
            
            async for chunk in stream_response:
                chunk_count += 1
                try:
                    chunk_as_dict = chunk.model_dump(exclude_unset=True, exclude_none=True)
                    
                    choices = chunk_as_dict.get('choices')
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        delta = choices[0].get('delta')
                        if delta and isinstance(delta, dict):
                            # Always remove extra_content if present
                            if 'extra_content' in delta:
                                del delta['extra_content']
                            
                            content = delta.get('content', '')
                            if content:
                                # print(f"DEBUG: Chunk {chunk_count} - Raw content: '{content}'")
                                # Use the processor to extract reasoning
                                processed_content, current_reasoning = reasoning_processor.process_chunk(content)
                                
                                # Debug logging for processing results
                                # if processed_content or current_reasoning:
                                #     print(f"DEBUG: Chunk {chunk_count} - Processed content: '{processed_content}', Reasoning: '{current_reasoning[:50]}...' if len(current_reasoning) > 50 else '{current_reasoning}'")
                                
                                # Send chunks for both reasoning and content as they arrive
                                chunks_to_send = []
                                
                                # If we have reasoning content, send it
                                if current_reasoning:
                                    reasoning_chunk = chunk_as_dict.copy()
                                    reasoning_chunk['choices'][0]['delta'] = {'reasoning_content': current_reasoning}
                                    chunks_to_send.append(reasoning_chunk)
                                
                                # If we have regular content, send it
                                if processed_content:
                                    content_chunk = chunk_as_dict.copy()
                                    content_chunk['choices'][0]['delta'] = {'content': processed_content}
                                    chunks_to_send.append(content_chunk)
                                    has_sent_content = True
                                
                                # Send all chunks
                                for chunk_to_send in chunks_to_send:
                                    yield f"data: {json.dumps(chunk_to_send)}\n\n"
                            else:
                                # Still yield the chunk even if no content (could have other delta fields)
                                yield f"data: {json.dumps(chunk_as_dict)}\n\n"
                    else:
                        # Yield chunks without choices too (they might contain metadata)
                        yield f"data: {json.dumps(chunk_as_dict)}\n\n"

                except Exception as chunk_error:
                    error_msg = f"Error processing OpenAI chunk for {request.model}: {str(chunk_error)}"
                    print(f"ERROR: {error_msg}")
                    if len(error_msg) > 1024:
                        error_msg = error_msg[:1024] + "..."
                    error_response = create_openai_error_response(500, error_msg, "server_error")
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            
            # Debug logging for buffer state and chunk count
            # print(f"DEBUG: Stream ended after {chunk_count} chunks. Buffer state - tag_buffer: '{reasoning_processor.tag_buffer}', "
            #       f"inside_tag: {reasoning_processor.inside_tag}, "
            #       f"reasoning_buffer: '{reasoning_processor.reasoning_buffer[:50]}...' if reasoning_processor.reasoning_buffer else ''")
            
            # Flush any remaining buffered content
            remaining_content, remaining_reasoning = reasoning_processor.flush_remaining()
            
            # Send any remaining reasoning first
            if remaining_reasoning:
                # print(f"DEBUG: Flushing remaining reasoning: '{remaining_reasoning[:50]}...' if len(remaining_reasoning) > 50 else '{remaining_reasoning}'")
                reasoning_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": remaining_reasoning}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(reasoning_chunk)}\n\n"
            
            # Send any remaining content
            if remaining_content:
                # print(f"DEBUG: Flushing remaining content: '{remaining_content}'")
                final_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": remaining_content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                has_sent_content = True
            
            # Always send a finish reason chunk
            finish_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as stream_error:
            error_msg = str(stream_error)
            if len(error_msg) > 1024:
                error_msg = error_msg[:1024] + "..."
            error_msg_full = f"Error during OpenAI streaming for {request.model}: {error_msg}"
            print(f"ERROR: {error_msg_full}")
            error_response = create_openai_error_response(500, error_msg_full, "server_error")
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def handle_non_streaming_response(
        self,
        openai_client: openai.AsyncOpenAI,
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> JSONResponse:
        """Handle non-streaming responses for OpenAI Direct mode."""
        try:
            # Ensure stream=False is explicitly passed
            openai_params_non_stream = {**openai_params, "stream": False}
            response = await openai_client.chat.completions.create(
                **openai_params_non_stream,
                extra_body=openai_extra_body
            )
            response_dict = response.model_dump(exclude_unset=True, exclude_none=True)
            
            try:
                choices = response_dict.get('choices')
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message_dict = choices[0].get('message')
                    if message_dict and isinstance(message_dict, dict):
                        # Always remove extra_content from the message if it exists
                        if 'extra_content' in message_dict:
                            del message_dict['extra_content']
                        
                        # Extract reasoning from content
                        full_content = message_dict.get('content')
                        actual_content = full_content if isinstance(full_content, str) else ""
                        
                        if actual_content:
                            print(f"INFO: OpenAI Direct Non-Streaming - Applying tag extraction with fixed marker: '{VERTEX_REASONING_TAG}'")
                            reasoning_text, actual_content = extract_reasoning_by_tags(actual_content, VERTEX_REASONING_TAG)
                            message_dict['content'] = actual_content
                            if reasoning_text:
                                message_dict['reasoning_content'] = reasoning_text
                                # print(f"DEBUG: Tag extraction success. Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content)}")
                            # else:
                            #     print(f"DEBUG: No content found within fixed tag '{VERTEX_REASONING_TAG}'.")
                        else:
                            print(f"WARNING: OpenAI Direct Non-Streaming - No initial content found in message.")
                            message_dict['content'] = ""
                            
            except Exception as e_reasoning:
                print(f"WARNING: Error during non-streaming reasoning processing for model {request.model}: {e_reasoning}")
            
            return JSONResponse(content=response_dict)
            
        except Exception as e:
            error_msg = f"Error calling OpenAI client for {request.model}: {str(e)}"
            print(f"ERROR: {error_msg}")
            return JSONResponse(
                status_code=500, 
                content=create_openai_error_response(500, error_msg, "server_error")
            )
    
    async def process_request(self, request: OpenAIRequest, base_model_name: str):
        """Main entry point for processing OpenAI Direct mode requests."""
        print(f"INFO: Using OpenAI Direct Path for model: {request.model}")
        
        # Get credentials
        rotated_credentials, rotated_project_id = self.credential_manager.get_credentials()
        
        if not rotated_credentials or not rotated_project_id:
            error_msg = "OpenAI Direct Mode requires GCP credentials, but none were available or loaded successfully."
            print(f"ERROR: {error_msg}")
            return JSONResponse(
                status_code=500, 
                content=create_openai_error_response(500, error_msg, "server_error")
            )
        
        print(f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
        gcp_token = _refresh_auth(rotated_credentials)
        
        if not gcp_token:
            error_msg = f"Failed to obtain valid GCP token for OpenAI client (Project: {rotated_project_id})."
            print(f"ERROR: {error_msg}")
            return JSONResponse(
                status_code=500, 
                content=create_openai_error_response(500, error_msg, "server_error")
            )
        
        # Create client and prepare parameters
        openai_client = self.create_openai_client(rotated_project_id, gcp_token)
        model_id = f"google/{base_model_name}"
        openai_params = self.prepare_openai_params(request, model_id)
        openai_extra_body = self.prepare_extra_body()
        
        # Handle streaming vs non-streaming
        if request.stream:
            return await self.handle_streaming_response(
                openai_client, openai_params, openai_extra_body, request
            )
        else:
            return await self.handle_non_streaming_response(
                openai_client, openai_params, openai_extra_body, request
            )