"""
Tests for the unified OpenAI-compatible `/chat` endpoint.

This module tests the new unified endpoint that handles:
- Text-only chat
- Multimodal (image) chat
- File uploads (PDF, DOCX, TXT)
- Streaming responses
- Error handling
"""

import json
import base64
import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# =============================================================================
# Basic Text Chat Tests
# =============================================================================

class TestBasicTextChat:
    """Tests for basic text chat functionality."""
    
    def test_chat_simple_text(self, client: TestClient, valid_chat_request: dict):
        """Test basic message with simple text content."""
        response = client.post("/chat", json=valid_chat_request)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "conversation_id" in data
        assert "user_id" in data
        content = data["choices"][0]["message"]["content"]
        assert content == "Hello! I'm doing well, thank you for asking. How can I help you today?"
    
    def test_chat_with_system_message(self, client: TestClient):
        """Test chat with system message included."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to reverse a string."}
            ],
            "stream": False
        }

        response = client.post("/chat", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["user_id"] is not None
    
    def test_chat_conversation_history(self, client: TestClient, sample_messages_with_history: list[dict]):
        """Test chat with multiple messages (conversation history)."""
        request = {
            "model": "gpt-4o",
            "messages": sample_messages_with_history,
            "stream": False,
            "conversation_id": "conv_existing123456789012"
        }

        response = client.post("/chat", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "conversation_id" in data
        # Should maintain the provided conversation ID
        assert data["conversation_id"] == "conv_existing123456789012"
    
    def test_chat_with_parameters(self, client: TestClient, sample_messages: list[dict]):
        """Test chat with various parameters (temperature, max_tokens, etc.)."""
        request = {
            "model": "gpt-4o",
            "messages": sample_messages,
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "frequency_penalty": -0.5,
            "user_id": "custom_user_456",
            "conversation_id": None
        }

        response = client.post("/chat", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["user_id"] == "custom_user_456"
    
    def test_chat_auto_generates_user_id(self, client: TestClient, sample_messages: list[dict]):
        """Test that user_id is auto-generated when not provided."""
        request = {
            "model": "gpt-4o",
            "messages": sample_messages,
            "stream": False
            # No user_id provided
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert data["user_id"].startswith("user_")
    
    def test_chat_creates_conversation_when_none_provided(self, client: TestClient, sample_messages: list[dict]):
        """Test that new conversation is created when conversation_id is not provided."""
        request = {
            "model": "gpt-4o",
            "messages": sample_messages,
            "stream": False,
            "conversation_id": None
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert data["conversation_id"] is not None


# =============================================================================
# Multimodal Tests (Images)
# =============================================================================

class TestMultimodalChat:
    """Tests for multimodal chat with images."""
    
    def test_chat_with_image_base64(self, client: TestClient, sample_image_data_url: str):
        """Test chat with base64-encoded image."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": sample_image_data_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_chat_with_image_url(self, client: TestClient):
        """Test chat with image as HTTP URL."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "auto"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_chat_with_multiple_images(self, client: TestClient, sample_image_data_url: str):
        """Test chat with multiple images in one message."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two images:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": sample_image_data_url, "detail": "low"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": sample_image_data_url, "detail": "low"}
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_chat_with_text_and_image(self, client: TestClient, sample_image_data_url: str):
        """Test mixed text and image content."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First, tell me about Python."},
                        {"type": "text", "text": "Then describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": sample_image_data_url}
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_chat_image_with_detail_levels(self, client: TestClient, sample_image_data_url: str):
        """Test image with different detail levels (low, high, auto)."""
        for detail in ["low", "high", "auto"]:
            request = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": sample_image_data_url,
                                    "detail": detail
                                }
                            }
                        ]
                    }
                ],
                "stream": False
            }
            
            response = client.post("/chat", json=request)
            assert response.status_code == 200, f"Failed with detail={detail}"


# =============================================================================
# File Upload Tests (Documents)
# =============================================================================

class TestFileUploadChat:
    """Tests for file upload functionality."""
    
    def test_chat_with_pdf_upload(self, client_with_s3: TestClient, mock_s3: MagicMock, sample_pdf_content: bytes):
        """Test PDF file upload via multipart form."""
        messages = json.dumps([{"role": "user", "content": "Summarize this document"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "stream": "false"
            },
            files={"files": ("document.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        mock_s3.upload_bytes.assert_called_once()
    
    def test_chat_with_docx_upload(self, client_with_s3: TestClient, mock_s3: MagicMock, sample_docx_content: bytes):
        """Test DOCX file upload via multipart form."""
        messages = json.dumps([{"role": "user", "content": "Extract key points from this document"}])
        
        # Mock docx extraction
        with patch('docx.Document') as mock_docx:
            mock_para = MagicMock()
            mock_para.text = "This is extracted text from the DOCX file."
            mock_docx.return_value.paragraphs = [mock_para]
            
            response = client_with_s3.post(
                "/chat",
                data={
                    "messages": messages,
                    "model": "gpt-4o",
                    "stream": "false"
                },
                files={"files": ("document.docx", io.BytesIO(sample_docx_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_chat_with_txt_upload(self, client_with_s3: TestClient, mock_s3: MagicMock, sample_txt_content: bytes):
        """Test TXT file upload via multipart form."""
        messages = json.dumps([{"role": "user", "content": "What does this file say?"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "stream": "false"
            },
            files={"files": ("document.txt", io.BytesIO(sample_txt_content), "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        mock_s3.upload_bytes.assert_called_once()
    
    def test_chat_with_multiple_files(self, client_with_s3: TestClient, sample_pdf_content: bytes, sample_txt_content: bytes):
        """Test uploading multiple files at once."""
        messages = json.dumps([{"role": "user", "content": "Compare these documents"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "stream": "false"
            },
            files=[
                ("files", ("doc1.pdf", io.BytesIO(sample_pdf_content), "application/pdf")),
                ("files", ("doc2.txt", io.BytesIO(sample_txt_content), "text/plain"))
            ]
        )
        
        assert response.status_code == 200


# =============================================================================
# Streaming Tests
# =============================================================================

class TestStreamingChat:
    """Tests for streaming response functionality."""
    
    def test_chat_streaming_text(self, client: TestClient, mock_openai_stream_chunks: list[MagicMock]):
        """Test basic streaming response."""
        import server
        
        # Mock the streaming response
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_openai_stream_chunks))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": True
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        
        # Parse SSE chunks
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content
    
    def test_chat_streaming_with_image(self, client: TestClient, mock_openai_stream_chunks: list[MagicMock], sample_image_data_url: str):
        """Test streaming with image content."""
        import server
        
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_openai_stream_chunks))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": sample_image_data_url}
                        }
                    ]
                }
            ],
            "stream": True
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
    
    def test_chat_streaming_format(self, client: TestClient, mock_openai_stream_chunks: list[MagicMock], parse_sse_chunks):
        """Test SSE format verification."""
        import server
        
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_openai_stream_chunks))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        response = client.post("/chat", json=request)
        chunks = parse_sse_chunks(response.text)
        
        # Verify OpenAI-compatible format
        for chunk in chunks:
            assert "id" in chunk
            assert "object" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "created" in chunk
            assert "model" in chunk
            assert "choices" in chunk
            
            for choice in chunk["choices"]:
                assert "index" in choice
                assert "delta" in choice
                assert "finish_reason" in choice


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and validation."""
    
    def test_chat_invalid_model(self, client: TestClient):
        """Test request with unsupported model - model is passed to OpenAI directly."""
        request = {
            "model": "unsupported-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }

        response = client.post("/chat", json=request)

        # Model is passed through to OpenAI - server doesn't validate model names
        assert response.status_code in [200, 400, 404, 422]
    
    def test_chat_invalid_messages_format(self, client: TestClient):
        """Test request with bad message format."""
        request = {
            "model": "gpt-4o",
            "messages": "this should be an array, not a string",
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 422
    
    def test_chat_missing_messages(self, client: TestClient):
        """Test request without messages field."""
        request = {
            "model": "gpt-4o",
            "stream": False
            # No messages field
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 422
    
    def test_chat_empty_messages(self, client: TestClient):
        """Test request with empty messages array."""
        request = {
            "model": "gpt-4o",
            "messages": [],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 422
    
    def test_chat_invalid_role(self, client: TestClient):
        """Test message with invalid role."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "invalid_role", "content": "Hello"}
            ],
            "stream": False
        }
        
        response = client.post("/chat", json=request)
        
        assert response.status_code == 422
    
    def test_chat_invalid_image_url(self, client: TestClient):
        """Test with malformed image URL - URL is passed to OpenAI directly."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "not-a-valid-url-or-data-uri",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }

        response = client.post("/chat", json=request)

        # URL is passed through to OpenAI - server doesn't validate image URLs
        assert response.status_code in [200, 400, 422]
    
    def test_chat_rate_limit(self, client: TestClient, sample_messages: list[dict]):
        """Test rate limiting response."""
        import server
        from exceptions import RateLimitExceededError
        
        # Mock rate limiter to always fail
        mock_limiter = MagicMock()
        mock_limiter.acquire.side_effect = RateLimitExceededError(
            message="Rate limit exceeded",
            requests_per_minute=10,
            retry_after=60
        )
        
        with patch.object(server, 'rate_limiters', {"user_123": mock_limiter}):
            request = {
                "model": "gpt-4o",
                "messages": sample_messages,
                "stream": False,
                "user_id": "user_123"
            }
            
            response = client.post("/chat", json=request)
            
            assert response.status_code == 429
            assert "Retry-After" in response.headers or "retry_after" in response.json().get("error", "")
    
    def test_chat_file_too_large(self, client_with_s3: TestClient):
        """Test file upload exceeding size limit."""
        # Create a "file" larger than the limit (e.g., 50MB)
        large_content = b"x" * (50 * 1024 * 1024)
        
        messages = json.dumps([{"role": "user", "content": "Process this"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o"
            },
            files={"files": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
        )
        
        assert response.status_code == 413


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    """Tests for response format and structure."""
    
    def test_chat_response_format_openai_compatible(self, client: TestClient, valid_chat_request: dict):
        """Verify response follows OpenAI API format."""
        response = client.post("/chat", json=valid_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check OpenAI-compatible fields
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert "model" in data
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) > 0
        
        # Check choice structure
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        # Check message structure
        message = choice["message"]
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message
    
    def test_chat_response_includes_usage(self, client: TestClient, valid_chat_request: dict):
        """Verify response includes usage statistics."""
        response = client.post("/chat", json=valid_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
    
    def test_chat_response_extensions(self, client: TestClient, valid_chat_request: dict):
        """Verify TextLLM-specific extensions in response."""
        response = client.post("/chat", json=valid_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # TextLLM extensions
        assert "conversation_id" in data
        assert "user_id" in data
        assert "cost" in data
        assert isinstance(data["cost"], float)


