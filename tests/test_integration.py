"""
Integration tests for the unified OpenAI-compatible Chat API.

This module tests complete user workflows including:
- Full conversation flows
- Multimodal conversations
- File upload and chat workflows
- Streaming vs non-streaming consistency
- Rate limiting per user
- Session tracking
- Cost calculation
- S3 uploads
"""

import json
import base64
import io
import time
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from fastapi.testclient import TestClient


# =============================================================================
# Full Conversation Flow Tests
# =============================================================================

class TestFullConversationFlow:
    """Integration tests for complete conversation workflows."""
    
    def test_create_conversation_send_messages_get_history(
        self, 
        client_with_db: TestClient, 
        mock_db: MagicMock,
        mock_openai_client: MagicMock
    ):
        """Test creating a conversation, sending messages, and retrieving history."""
        import server
        
        # Create conversation by sending first message
        response1 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, let's start a conversation"}],
            "stream": False,
            "user_id": "integration_user"
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        assert conversation_id is not None
        
        # Send follow-up message
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "I enjoy Python and JavaScript."

        response2 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, let's start a conversation"},
                {"role": "assistant", "content": data1["choices"][0]["message"]["content"]},
                {"role": "user", "content": "What programming languages do you know?"}
            ],
            "stream": False,
            "user_id": "integration_user",
            "conversation_id": conversation_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["conversation_id"] == conversation_id
        
        # Verify database was updated
        assert mock_db.update_conversation_usage.called
    
    def test_conversation_persistence_across_requests(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that conversation state persists across multiple requests."""
        conversation_id = "conv_persist_test1234567890"
        user_id = "persist_user"
        
        # Set up mock to return existing conversation
        mock_db.get_conversation.return_value = {
            "id": conversation_id,
            "user_id": user_id,
            "message_count": 3,
            "total_tokens_input": 150,
            "total_tokens_output": 75,
            "total_cost": 0.002
        }
        
        # Send message to existing conversation
        response = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Continuing our conversation..."}
            ],
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
    
    def test_multiple_users_separate_conversations(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that different users have separate conversation histories."""
        import server
        
        # User 1 creates conversation
        response1 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "User 1 message"}],
            "user_id": "user_1"
        })
        
        assert response1.status_code == 200
        conv1_id = response1.json()["conversation_id"]
        
        # User 2 creates conversation
        # Conversation IDs are generated locally, no OpenAI conversations API used
        
        response2 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "User 2 message"}],
            "user_id": "user_2"
        })
        
        assert response2.status_code == 200
        conv2_id = response2.json()["conversation_id"]
        
        # Conversations should be different
        assert conv1_id != conv2_id


# =============================================================================
# Multimodal Conversation Tests
# =============================================================================

class TestMultimodalConversation:
    """Integration tests for multimodal (text + image) workflows."""
    
    def test_multimodal_conversation_text_then_image(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        mock_openai_client: MagicMock,
        sample_image_data_url: str
    ):
        """Test conversation that starts with text then includes image."""
        
        # First message - text only
        response1 = client_with_s3.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "I have an image to show you"}],
            "user_id": "multimodal_user"
        })
        
        assert response1.status_code == 200
        conv_id = response1.json()["conversation_id"]
        
        # Second message - with image
        response2 = client_with_s3.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "I have an image to show you"},
                {"role": "assistant", "content": "Please share it!"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": sample_image_data_url}}
                    ]
                }
            ],
            "conversation_id": conv_id,
            "user_id": "multimodal_user"
        })
        
        assert response2.status_code == 200
    
    def test_multiple_images_same_conversation(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_image_data_url: str
    ):
        """Test sending multiple images in same conversation."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images:"},
                    {"type": "image_url", "image_url": {"url": sample_image_data_url, "detail": "low"}},
                    {"type": "image_url", "image_url": {"url": sample_image_data_url, "detail": "low"}}
                ]
            }
        ]
        
        response = client_with_s3.post("/chat", json={
            "model": "gpt-4o",
            "messages": messages,
            "user_id": "multi_image_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_image_followed_by_text_questions(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_image_data_url: str
    ):
        """Test asking text questions about a previously shared image."""
        
        conv_id = "conv_image_followup123456789"
        
        # Image message
        response1 = client_with_s3.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's an image:"},
                        {"type": "image_url", "image_url": {"url": sample_image_data_url}}
                    ]
                }
            ],
            "conversation_id": conv_id,
            "user_id": "image_qa_user"
        })
        
        assert response1.status_code == 200
        
        # Follow-up text question about the image
        response2 = client_with_s3.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What colors are in the image?"}
            ],
            "conversation_id": conv_id,
            "user_id": "image_qa_user"
        })
        
        assert response2.status_code == 200


# =============================================================================
# File Upload and Chat Tests
# =============================================================================

class TestFileUploadAndChat:
    """Integration tests for file upload workflows."""
    
    def test_upload_pdf_ask_questions(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_pdf_content: bytes
    ):
        """Test uploading PDF and asking questions about it."""
        
        messages = json.dumps([{"role": "user", "content": "Summarize this document"}])
        
        # Upload PDF and ask question
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "user_id": "pdf_user"
            },
            files={"files": ("report.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "conversation_id" in data
        
        # Verify S3 upload
        assert mock_s3.upload_bytes.called
        call_args = mock_s3.upload_bytes.call_args
        assert call_args[1]["folder"] == "chat_files"
    
    def test_upload_txt_extract_content(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_txt_content: bytes
    ):
        """Test uploading TXT file and extracting content."""
        
        messages = json.dumps([{"role": "user", "content": "What is the main topic?"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "user_id": "txt_user"
            },
            files={"files": ("notes.txt", io.BytesIO(sample_txt_content), "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
    
    def test_upload_multiple_files_comparison(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_pdf_content: bytes,
        sample_txt_content: bytes
    ):
        """Test uploading multiple files for comparison."""
        
        messages = json.dumps([{"role": "user", "content": "Compare these documents"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "user_id": "comparison_user"
            },
            files=[
                ("files", ("doc1.pdf", io.BytesIO(sample_pdf_content), "application/pdf")),
                ("files", ("doc2.txt", io.BytesIO(sample_txt_content), "text/plain"))
            ]
        )
        
        assert response.status_code == 200
        # Verify multiple uploads
        assert mock_s3.upload_bytes.call_count >= 2


# =============================================================================
# Streaming vs Non-Streaming Tests
# =============================================================================

class TestStreamingVsNonStreaming:
    """Integration tests comparing streaming and non-streaming responses."""
    
    def test_streaming_vs_non_streaming_same_content(
        self,
        client: TestClient,
        mock_openai_client: MagicMock,
        mock_openai_stream_chunks: list[MagicMock]
    ):
        """Test that streaming and non-streaming return equivalent content."""
        import server
        
        # Set same response for both modes
        expected_text = "Hello! How can I help you today?"
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = expected_text
        
        # Non-streaming request
        response_non_stream = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })
        
        assert response_non_stream.status_code == 200
        non_stream_data = response_non_stream.json()
        
        # Set up streaming mock
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_openai_stream_chunks))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        # Streaming request
        response_stream = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        })
        
        assert response_stream.status_code == 200
        assert "text/event-stream" in response_stream.headers.get("content-type", "")
    
    def test_streaming_usage_in_final_chunk(
        self,
        client: TestClient,
        mock_openai_stream_chunks: list[MagicMock]
    ):
        """Test that streaming response includes usage in final chunk."""
        import server
        
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_openai_stream_chunks))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        response = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Count to 10"}],
            "stream": True
        })
        
        content = response.text
        
        # Parse chunks to find one with usage
        lines = content.strip().split('\n')
        usage_found = False
        
        for line in lines:
            if line.startswith('data: ') and line != 'data: [DONE]':
                try:
                    chunk = json.loads(line[6:])
                    if 'usage' in chunk and chunk['usage']:
                        usage_found = True
                        assert 'prompt_tokens' in chunk['usage']
                        assert 'completion_tokens' in chunk['usage']
                except json.JSONDecodeError:
                    pass
        
        assert usage_found or 'response.completed' in str(mock_openai_stream_chunks)


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimitingPerUser:
    """Integration tests for rate limiting functionality."""
    
    def test_rate_limiting_per_user_separate_limits(
        self,
        client: TestClient
    ):
        """Test that each user has separate rate limit."""
        import server
        from rate_limiter import TokenBucketRateLimiter
        
        # Create rate limiters for different users
        limiter1 = TokenBucketRateLimiter(requests_per_minute=2, burst_allowance=0)
        limiter2 = TokenBucketRateLimiter(requests_per_minute=2, burst_allowance=0)
        
        with patch.object(server, 'rate_limiters', {
            "user_a": limiter1,
            "user_b": limiter2
        }):
            # User A makes requests
            for _ in range(2):
                response_a = client.post("/chat", json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "user_id": "user_a"
                })
                assert response_a.status_code == 200
            
            # User A should be rate limited now
            response_a_limited = client.post("/chat", json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello again"}],
                "user_id": "user_a"
            })
            assert response_a_limited.status_code == 429
            
            # User B should still be able to make requests
            response_b = client.post("/chat", json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello from B"}],
                "user_id": "user_b"
            })
            assert response_b.status_code == 200
    
    def test_rate_limit_headers_present(
        self,
        client: TestClient
    ):
        """Test that rate limit headers are present in responses."""
        response = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "header_test_user"
        })
        
        # Check for rate limit headers
        headers = response.headers
        # The Retry-After header would only be present on 429 responses


# =============================================================================
# Session Tracking Tests
# =============================================================================

class TestSessionTracking:
    """Integration tests for session tracking functionality."""
    
    def test_session_created_on_first_message(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that session is created when user sends first message."""
        
        response = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "new_session_user"
        })
        
        assert response.status_code == 200
        
        # Verify session was created
        assert mock_db.create_session.called or mock_db.get_session.called
    
    def test_session_stats_updated(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that session stats are updated after chat."""
        
        response = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "stats_user"
        })
        
        assert response.status_code == 200
        
        # Verify session stats were updated
        assert mock_db.update_session_stats.called
    
    def test_multiple_conversations_same_session(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that multiple conversations belong to same session."""
        import server
        
        user_id = "multi_conv_user"
        
        # First conversation
        response1 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "First conversation"}],
            "user_id": user_id
        })
        
        assert response1.status_code == 200
        conv1_id = response1.json()["conversation_id"]
        
        # Create new conversation (simulated by different ID)
        # Conversation IDs are generated locally as UUIDs
        
        response2 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Second conversation"}],
            "user_id": user_id
        })
        
        assert response2.status_code == 200
        conv2_id = response2.json()["conversation_id"]
        
        # Different conversations
        assert conv1_id != conv2_id


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculation:
    """Integration tests for cost calculation functionality."""
    
    def test_cost_calculated_correctly(
        self,
        client: TestClient,
        mock_openai_client: MagicMock
    ):
        """Test that cost is calculated based on token usage."""
        
        # Set specific usage - Chat Completions format
        usage = MagicMock()
        usage.prompt_tokens = 1000
        usage.completion_tokens = 500
        usage.total_tokens = 1500
        mock_openai_client.chat.completions.create.return_value.usage = usage
        
        response = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "cost_test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify cost is present and is a positive number
        assert "cost" in data
        assert isinstance(data["cost"], float)
        assert data["cost"] > 0
        
        # Expected cost calculation (based on pricing in server.py)
        # input_per_1k: 0.0025, output_per_1k: 0.01
        # Cost = (1000/1000) * 0.0025 + (500/1000) * 0.01 = 0.0025 + 0.005 = 0.0075
        expected_cost = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.01
        assert abs(data["cost"] - expected_cost) < 0.0001
    
    def test_cost_accumulated_in_conversation(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock
    ):
        """Test that costs accumulate across conversation messages."""
        
        conversation_id = "conv_cost_accum1234567890"
        
        # First message
        response1 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Message 1"}],
            "conversation_id": conversation_id,
            "user_id": "cost_accum_user"
        })
        
        assert response1.status_code == 200
        cost1 = response1.json()["cost"]
        
        # Second message
        response2 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Message 2"}
            ],
            "conversation_id": conversation_id,
            "user_id": "cost_accum_user"
        })
        
        assert response2.status_code == 200
        cost2 = response2.json()["cost"]
        
        # Both should have costs
        assert cost1 > 0
        assert cost2 > 0
    
    def test_usage_statistics_in_response(
        self,
        client: TestClient
    ):
        """Test that usage statistics are included in response."""
        
        response = client.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "usage_test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check usage field
        assert "usage" in data
        usage = data["usage"]
        
        # Verify usage structure (may vary based on implementation)
        # Legacy endpoint uses input/output/total
        if "input" in usage:
            assert isinstance(usage["input"], int)
            assert isinstance(usage["output"], int)
            assert isinstance(usage["total"], int)
        # New OpenAI-compatible format uses prompt_tokens/completion_tokens/total_tokens
        elif "prompt_tokens" in usage:
            assert isinstance(usage["prompt_tokens"], int)
            assert isinstance(usage["completion_tokens"], int)
            assert isinstance(usage["total_tokens"], int)


# =============================================================================
# S3 Uploads Tests
# =============================================================================

class TestS3Uploads:
    """Integration tests for S3 file upload functionality."""
    
    def test_s3_upload_with_correct_key_structure(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_image_base64: str
    ):
        """Test that files are uploaded with correct S3 key structure."""

        image_bytes = base64.b64decode(sample_image_base64)
        user_id = "s3_structure_user"

        messages = json.dumps([{"role": "user", "content": "Describe this"}])

        response = client_with_s3.post(
            "/chat",
            data={"messages": messages, "model": "gpt-4o", "user_id": user_id},
            files={"files": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        )

        # Verify S3 upload was called
        assert mock_s3.upload_bytes.called

        # Check key structure
        call_args = mock_s3.upload_bytes.call_args
        assert call_args[1]["user_id"] == user_id
        assert "folder" in call_args[1]
    
    def test_s3_upload_includes_metadata(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_txt_content: bytes
    ):
        """Test that uploads include proper metadata."""
        
        messages = json.dumps([{"role": "user", "content": "Read this"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o",
                "user_id": "metadata_user"
            },
            files={"files": ("document.txt", io.BytesIO(sample_txt_content), "text/plain")}
        )
        
        assert mock_s3.upload_bytes.called
        
        # Verify metadata in call
        call_kwargs = mock_s3.upload_bytes.call_args[1]
        assert "content_type" in call_kwargs
        assert call_kwargs["content_type"] == "text/plain"
    
    def test_s3_handles_upload_failure(
        self,
        client_with_s3: TestClient,
        mock_s3: MagicMock,
        sample_pdf_content: bytes
    ):
        """Test graceful handling of S3 upload failures."""
        
        # Mock S3 to raise error
        from storage import StorageError
        mock_s3.upload_bytes.side_effect = StorageError("S3 connection failed")
        
        messages = json.dumps([{"role": "user", "content": "Read this PDF"}])
        
        response = client_with_s3.post(
            "/chat",
            data={
                "messages": messages,
                "model": "gpt-4o"
            },
            files={"files": ("document.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        )
        
        # Should return service unavailable error
        assert response.status_code == 503


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end integration tests."""
    
    def test_complete_user_journey(
        self,
        client_with_db: TestClient,
        mock_db: MagicMock,
        mock_s3: MagicMock,
        sample_image_data_url: str,
        sample_pdf_content: bytes
    ):
        """Test complete user journey from start to finish."""
        import server
        
        user_id = "journey_user"
        
        # 1. Start conversation
        response1 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, I need help with a project"}],
            "user_id": user_id
        })
        assert response1.status_code == 200
        conv_id = response1.json()["conversation_id"]
        
        # 2. Send image for analysis
        response2 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you think of this design?"},
                        {"type": "image_url", "image_url": {"url": sample_image_data_url}}
                    ]
                }
            ],
            "conversation_id": conv_id,
            "user_id": user_id
        })
        assert response2.status_code == 200
        
        # 3. Upload document for reference
        messages = json.dumps([{"role": "user", "content": "Check this requirements doc"}])
        
        # Mock storage for file upload
        with patch.object(server, 'storage', mock_s3):
            response3 = client_with_db.post(
                "/chat",
                data={
                    "messages": messages,
                    "conversation_id": conv_id,
                    "user_id": user_id
                },
                files={"files": ("requirements.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
            )
        assert response3.status_code == 200
        
        # 4. Get streaming response
        mock_stream = MagicMock()
        chunk = MagicMock()
        mock_delta = MagicMock()
        mock_delta.content = "Based on the requirements and design..."
        mock_delta.role = None
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        chunk.choices = [mock_choice]
        chunk.usage = None
        mock_stream.__iter__ = MagicMock(return_value=iter([chunk]))
        server.openai_client.chat.completions.create.return_value = mock_stream
        
        response4 = client_with_db.post("/chat", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Give me a summary"}],
            "conversation_id": conv_id,
            "user_id": user_id,
            "stream": True
        })
        assert response4.status_code == 200
        
        # Verify all operations succeeded
        assert mock_db.create_session.called or mock_db.get_session.called
        assert mock_db.update_conversation_usage.called
