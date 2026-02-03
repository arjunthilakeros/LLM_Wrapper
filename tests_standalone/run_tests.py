#!/usr/bin/env python3
"""
Standalone Test Runner for TextLLM Unified /chat Endpoint

This script runs comprehensive tests for the OpenAI-compatible API.
No external dependencies except: pytest, fastapi, httpx

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py -v                 # Verbose output
    python run_tests.py -k test_chat       # Run specific tests
    python run_tests.py --quick            # Quick smoke tests only
"""

import sys
import os
import json
import base64
import io
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Generator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required packages
try:
    import pytest
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pytest fastapi httpx")
    HAS_DEPS = False
    sys.exit(1)


# =============================================================================
# Mock Data Fixtures
# =============================================================================

class MockData:
    """Mock data for testing."""
    
    @staticmethod
    def sample_image_base64() -> str:
        """Return a small valid base64 image."""
        # 1x1 pixel red PNG
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    @staticmethod
    def valid_chat_request() -> dict:
        """Return a valid chat request."""
        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False,
            "temperature": 1.0
        }
    
    @staticmethod
    def multimodal_request() -> dict:
        """Return a multimodal request with image."""
        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{MockData.sample_image_base64()}"}}
                ]}
            ]
        }
    
    @staticmethod
    def openai_response() -> dict:
        """Return a mock OpenAI API response."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        }


# =============================================================================
# Test Suite
# =============================================================================

class TestChatEndpoint:
    """Tests for the unified /chat endpoint."""
    
    def test_basic_text_chat(self, client, mock_openai):
        """Test basic text chat request."""
        response = client.post("/chat", json=MockData.valid_chat_request())
        
        assert response.status_code == 200
        data = response.json()
        
        # Check OpenAI-compatible format
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        
        print("✓ Basic text chat works")
    
    def test_chat_with_system_message(self, client, mock_openai):
        """Test chat with system message."""
        request = {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a Python function to reverse a string."}
            ]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
        
        print("✓ System message works")
    
    def test_chat_conversation_history(self, client, mock_openai):
        """Test chat with conversation history."""
        request = {
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What's my name?"}
            ]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        
        print("✓ Conversation history works")
    
    def test_chat_with_parameters(self, client, mock_openai):
        """Test chat with various parameters."""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        
        print("✓ Parameters work")
    
    def test_chat_auto_generates_user_id(self, client, mock_openai):
        """Test that user_id is auto-generated if not provided."""
        request = {
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert data["user_id"].startswith("user_")
        
        print("✓ Auto user_id generation works")
    
    def test_chat_creates_conversation(self, client, mock_openai):
        """Test that conversation is created when not provided."""
        request = {
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert data["conversation_id"].startswith("conv_")
        
        print("✓ Auto conversation creation works")


class TestMultimodalChat:
    """Tests for multimodal (image) support."""
    
    def test_chat_with_image_base64(self, client, mock_openai):
        """Test chat with base64-encoded image."""
        request = MockData.multimodal_request()
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        
        print("✓ Base64 image works")
    
    def test_chat_with_image_url(self, client, mock_openai):
        """Test chat with image URL."""
        request = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            }]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        
        print("✓ Image URL works")
    
    def test_chat_with_multiple_images(self, client, mock_openai):
        """Test chat with multiple images."""
        img = MockData.sample_image_base64()
        request = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                ]
            }]
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        
        print("✓ Multiple images work")


class TestStreamingChat:
    """Tests for streaming (SSE) support."""
    
    def test_chat_streaming_text(self, client, mock_openai_streaming):
        """Test streaming text response."""
        request = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        }
        
        response = client.post("/chat", json=request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Read SSE events
        content = response.content.decode()
        assert "data:" in content
        assert "[DONE]" in content
        
        print("✓ Streaming works")
    
    def test_chat_streaming_format(self, client, mock_openai_streaming):
        """Test streaming format is OpenAI-compatible."""
        request = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        }
        
        response = client.post("/chat", json=request)
        content = response.content.decode()
        
        # Check for OpenAI-compatible SSE format
        lines = [line for line in content.split("\n") if line.startswith("data:")]
        assert len(lines) > 0
        
        # Parse first data line
        first_data = lines[0].replace("data: ", "")
        chunk = json.loads(first_data)
        
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert "delta" in chunk["choices"][0]
        
        print("✓ Streaming format is OpenAI-compatible")


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_chat_missing_messages(self, client):
        """Test error when messages are missing."""
        response = client.post("/chat", json={"model": "gpt-4o"})
        assert response.status_code == 422
        
        print("✓ Missing messages error works")
    
    def test_chat_empty_messages(self, client):
        """Test error when messages array is empty."""
        response = client.post("/chat", json={"messages": []})
        assert response.status_code == 422
        
        print("✓ Empty messages error works")
    
    def test_chat_invalid_role(self, client):
        """Test error when role is invalid."""
        response = client.post("/chat", json={
            "messages": [{"role": "invalid_role", "content": "Hello!"}]
        })
        # Should either error or normalize the role
        assert response.status_code in [200, 422]
        
        print("✓ Invalid role handling works")
    
    def test_chat_invalid_model(self, client):
        """Test error when model is not supported."""
        response = client.post("/chat", json={
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello!"}]
        })
        # Should either error or fallback to default
        assert response.status_code in [200, 400, 422]
        
        print("✓ Invalid model handling works")


class TestResponseFormat:
    """Tests for response format compliance."""
    
    def test_chat_response_is_openai_compatible(self, client, mock_openai):
        """Test that response matches OpenAI format exactly."""
        response = client.post("/chat", json=MockData.valid_chat_request())
        data = response.json()
        
        # Required fields per OpenAI spec
        required_fields = ["id", "object", "created", "model", "choices"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check object type
        assert data["object"] == "chat.completion"
        
        # Check choices structure
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]
        assert "finish_reason" in choice
        
        # Check usage if present
        if "usage" in data:
            assert "prompt_tokens" in data["usage"]
            assert "completion_tokens" in data["usage"]
            assert "total_tokens" in data["usage"]
        
        print("✓ Response is OpenAI-compatible")
    
    def test_chat_response_has_extensions(self, client, mock_openai):
        """Test that TextLLM extensions are present."""
        response = client.post("/chat", json=MockData.valid_chat_request())
        data = response.json()
        
        # TextLLM-specific fields
        assert "conversation_id" in data
        assert "user_id" in data
        assert "cost" in data
        
        print("✓ TextLLM extensions present")


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy endpoints."""
    
    def test_legacy_chat_endpoint_exists(self, client):
        """Test that legacy /chat/legacy endpoint exists."""
        response = client.post("/chat/legacy", json={"message": "Hello!"})
        # Should work even if deprecated
        assert response.status_code in [200, 307, 308]
        
        print("✓ Legacy endpoint exists")
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        
        print("✓ Health endpoint works")
    
    def test_docs_endpoint(self, client):
        """Test Swagger docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        print("✓ Docs endpoint works")


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_openai():
    """Mock OpenAI API client."""
    with patch("server.openai_client") as mock:
        mock.chat.completions.create.return_value = MagicMock(
            id="chatcmpl-test123",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model="gpt-4o",
            choices=[MagicMock(
                index=0,
                message=MagicMock(role="assistant", content="Hello! How can I help?"),
                finish_reason="stop"
            )],
            usage=MagicMock(
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20
            )
        )
        yield mock


@pytest.fixture
def mock_openai_streaming():
    """Mock OpenAI API client for streaming."""
    def generate_stream():
        chunks = [
            MagicMock(
                id="chatcmpl-test123",
                object="chat.completion.chunk",
                created=int(datetime.now().timestamp()),
                model="gpt-4o",
                choices=[MagicMock(index=0, delta=MagicMock(role="assistant"), finish_reason=None)]
            ),
            MagicMock(
                id="chatcmpl-test123",
                object="chat.completion.chunk",
                created=int(datetime.now().timestamp()),
                model="gpt-4o",
                choices=[MagicMock(index=0, delta=MagicMock(content="Hello"), finish_reason=None)]
            ),
            MagicMock(
                id="chatcmpl-test123",
                object="chat.completion.chunk",
                created=int(datetime.now().timestamp()),
                model="gpt-4o",
                choices=[MagicMock(index=0, delta=MagicMock(content="!"), finish_reason="stop")]
            )
        ]
        for chunk in chunks:
            yield chunk
    
    with patch("server.openai_client") as mock:
        mock.chat.completions.create.return_value = generate_stream()
        yield mock


@pytest.fixture
def mock_db():
    """Mock database."""
    with patch("server.db") as mock:
        mock.create_conversation.return_value = {"id": "conv_test123"}
        mock.update_conversation_usage.return_value = None
        mock.get_conversation.return_value = {"message_count": 1}
        yield mock


@pytest.fixture
def mock_storage():
    """Mock S3 storage."""
    with patch("server.storage") as mock:
        mock.upload_bytes.return_value = {
            "key": "test/key",
            "url": "https://s3.example.com/test"
        }
        yield mock


@pytest.fixture
def client(mock_openai, mock_db, mock_storage):
    """Create test client."""
    # Import here to use mocked dependencies
    from server import app
    return TestClient(app)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_quick_tests():
    """Run quick smoke tests without pytest."""
    print("\n" + "="*60)
    print("TextLLM Unified /chat Endpoint - Quick Smoke Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Import server module", lambda: __import__("server")),
        ("Import validators", lambda: __import__("validators")),
        ("Import exceptions", lambda: __import__("exceptions")),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TextLLM Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-k", "--keyword", help="Run tests matching keyword")
    parser.add_argument("--quick", action="store_true", help="Quick smoke tests only")
    parser.add_argument("--no-mock", action="store_true", help="Run without mocks (requires real services)")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    # Run pytest
    pytest_args = [__file__, "-v" if args.verbose else "-q"]
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
