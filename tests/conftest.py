"""
Pytest configuration and fixtures for LLM_Wrapper tests.

This module provides fixtures for mocking external dependencies
and setting up the test environment.
"""

import json
import base64
import io
import os
import sys
from datetime import datetime
from typing import Generator, Any
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure LLM_Wrapper is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample valid messages array."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_messages_with_history() -> list[dict]:
    """Sample messages with conversation history."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What can I do with it?"}
    ]


@pytest.fixture
def sample_multimodal_messages() -> list[dict]:
    """Sample multimodal messages with text and image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD...",
                        "detail": "high"
                    }
                }
            ]
        }
    ]


@pytest.fixture
def sample_image_base64() -> str:
    """Sample base64-encoded JPEG image (1x1 pixel)."""
    # 1x1 red pixel JPEG
    image_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xB5, 0xFB, 0xB7, 0xFF,
        0xD9
    ])
    return base64.b64encode(image_bytes).decode('utf-8')


@pytest.fixture
def sample_image_data_url(sample_image_base64: str) -> str:
    """Sample image as data URL."""
    return f"data:image/jpeg;base64,{sample_image_base64}"


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Sample minimal PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n196\n%%EOF"


@pytest.fixture
def sample_docx_content() -> bytes:
    """Sample minimal DOCX content (actually a ZIP with XML)."""
    # Return minimal bytes that represent a docx structure
    # In tests, we'll mock the actual extraction
    return b"PK\x03\x04" + b"\x00" * 100  # ZIP header


@pytest.fixture
def sample_txt_content() -> bytes:
    """Sample text file content."""
    return b"This is a test document.\nIt has multiple lines.\nHello World!"


@pytest.fixture
def valid_chat_request(sample_messages: list[dict]) -> dict:
    """Sample valid chat request."""
    return {
        "model": "gpt-4o",
        "messages": sample_messages,
        "stream": False,
        "temperature": 1.0,
        "max_tokens": None,
        "user_id": "user_123",
        "conversation_id": None
    }


@pytest.fixture
def valid_multimodal_request(sample_multimodal_messages: list[dict]) -> dict:
    """Sample valid multimodal chat request."""
    return {
        "model": "gpt-4o",
        "messages": sample_multimodal_messages,
        "stream": False,
        "temperature": 0.7
    }


@pytest.fixture
def valid_streaming_request(sample_messages: list[dict]) -> dict:
    """Sample valid streaming chat request."""
    return {
        "model": "gpt-4o",
        "messages": sample_messages,
        "stream": True,
        "temperature": 1.0
    }


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Create a mock OpenAI Chat Completions API response."""
    mock = MagicMock()
    mock.id = "chatcmpl-8x7a06d52a79c6b0797d7163798c3b5a"
    mock.object = "chat.completion"
    mock.created = int(datetime.now().timestamp())
    mock.model = "gpt-4o"

    # Chat Completions format: choices[0].message.content
    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = "Hello! I'm doing well, thank you for asking. How can I help you today?"

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock.choices = [mock_choice]

    # Usage stats - Chat Completions format
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 25
    mock_usage.completion_tokens = 15
    mock_usage.total_tokens = 40
    mock.usage = mock_usage

    return mock


@pytest.fixture
def mock_openai_stream_chunks() -> list[MagicMock]:
    """Create mock OpenAI Chat Completions streaming response chunks."""
    chunks = []
    words = ["Hello", "!", " How", " can", " I", " help"]

    for word in words:
        chunk = MagicMock()
        mock_delta = MagicMock()
        mock_delta.content = word
        mock_delta.role = None
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        chunk.choices = [mock_choice]
        chunk.usage = None
        chunks.append(chunk)

    # Final chunk with finish_reason and usage
    final = MagicMock()
    mock_delta = MagicMock()
    mock_delta.content = None
    mock_delta.role = None
    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"
    final.choices = [mock_choice]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 6
    mock_usage.total_tokens = 16
    final.usage = mock_usage
    chunks.append(final)

    return chunks


@pytest.fixture
def mock_openai_client(mock_openai_response: MagicMock) -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()

    # Mock chat.completions.create (Chat Completions API)
    mock_client.chat.completions.create.return_value = mock_openai_response

    return mock_client


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock database."""
    mock = MagicMock()
    
    # Mock conversation operations
    mock.create_conversation.return_value = {
        "id": "conv_test1234567890123456789",
        "user_id": "user_123",
        "created_at": datetime.now().isoformat(),
        "message_count": 0
    }
    
    mock.get_conversation.return_value = {
        "id": "conv_test1234567890123456789",
        "user_id": "user_123",
        "message_count": 5,
        "total_tokens_input": 100,
        "total_tokens_output": 50,
        "total_cost": 0.001
    }
    
    mock.update_conversation_usage.return_value = None
    
    # Mock session operations
    mock.get_session.return_value = None  # No existing session
    mock.create_session.return_value = {
        "id": "session_user_123_20260101",
        "user_id": "user_123",
        "created_at": datetime.now().isoformat()
    }
    mock.update_session_stats.return_value = None
    
    # Mock user stats
    mock.get_user_stats.return_value = {
        "total_conversations": 10,
        "total_messages": 50,
        "total_tokens_input": 1000,
        "total_tokens_output": 500,
        "total_cost": 0.01
    }
    
    return mock


@pytest.fixture
def mock_s3() -> MagicMock:
    """Create a mock S3 storage."""
    mock = MagicMock()
    
    mock.upload_bytes.return_value = {
        "success": True,
        "key": "LLM/chat_images/user_123/20260101_120000_test.jpg",
        "url": "https://s3.example.com/LLM/chat_images/user_123/20260101_120000_test.jpg",
        "size": 1024,
        "hash": "abc123"
    }
    
    mock.upload_file.return_value = {
        "success": True,
        "key": "LLM/chat_files/user_123/20260101_120000_doc.pdf",
        "url": "https://s3.example.com/LLM/chat_files/user_123/20260101_120000_doc.pdf",
        "size": 2048,
        "hash": "def456"
    }
    
    mock.get_url.return_value = "https://s3.example.com/presigned-url"
    
    return mock


@pytest.fixture
def mock_rate_limiter() -> MagicMock:
    """Create a mock rate limiter."""
    mock = MagicMock()
    mock.acquire.return_value = True
    mock.can_proceed.return_value = True
    mock.get_remaining.return_value = 10
    mock.is_limited = False
    return mock


# =============================================================================
# FastAPI App Fixtures
# =============================================================================

@pytest.fixture
def app(mock_openai_client: MagicMock, mock_db: MagicMock, mock_s3: MagicMock) -> Generator[Any, None, None]:
    """
    Create a FastAPI app with mocked dependencies.
    
    This fixture patches the global dependencies before importing the app.
    """
    # Set environment variables for testing
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    os.environ["DATABASE_ENABLED"] = "false"
    os.environ["S3_ENABLED"] = "false"
    
    # Import after setting env vars
    import server
    
    # Patch global instances
    with patch.object(server, 'openai_client', mock_openai_client), \
         patch.object(server, 'db', None), \
         patch.object(server, 'storage', None):
        yield server.app


@pytest.fixture
def client(app) -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def client_with_db(app, mock_db: MagicMock) -> TestClient:
    """Create a FastAPI test client with mocked database."""
    import server
    with patch.object(server, 'db', mock_db):
        yield TestClient(app)


@pytest.fixture
def client_with_s3(app, mock_s3: MagicMock) -> TestClient:
    """Create a FastAPI test client with mocked S3 storage."""
    import server
    with patch.object(server, 'storage', mock_s3):
        yield TestClient(app)


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def parse_sse_chunks():
    """Utility function to parse SSE chunks from streaming response."""
    def _parse(response_text: str) -> list[dict]:
        chunks = []
        for line in response_text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data != '[DONE]':
                    try:
                        chunks.append(json.loads(data))
                    except json.JSONDecodeError:
                        pass
        return chunks
    return _parse


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
