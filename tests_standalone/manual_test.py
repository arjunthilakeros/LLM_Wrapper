#!/usr/bin/env python3
"""
Manual Test Script for TextLLM Unified /chat Endpoint

This script provides a simple way to test the API without pytest.
It uses requests to hit the running server.

Usage:
    # Start server first
    python server.py
    
    # Then run tests
    python tests_standalone/manual_test.py
    python tests_standalone/manual_test.py --stream
    python tests_standalone/manual_test.py --image
"""

import sys
import os
import json
import base64
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


BASE_URL = "http://localhost:8000"


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")


def print_header(title):
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.YELLOW}{title}{Colors.END}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}\n")


def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def test_health():
    """Test health endpoint."""
    print_info("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Server is healthy (status: {data.get('status')})")
        return True
    else:
        print_error(f"Health check failed: {response.status_code}")
        return False


def test_chat_simple():
    """Test simple text chat."""
    print_info("Testing simple text chat...")
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from TextLLM!' and nothing else."}
        ],
        "temperature": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code != 200:
        print_error(f"Request failed: {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    
    # Check OpenAI-compatible format
    checks = [
        ("id" in data, "Has 'id' field"),
        (data.get("object") == "chat.completion", "Has correct 'object' type"),
        ("created" in data, "Has 'created' timestamp"),
        ("model" in data, "Has 'model' field"),
        ("choices" in data and len(data["choices"]) > 0, "Has 'choices' array"),
        ("message" in data["choices"][0], "Has 'message' in choice"),
        ("content" in data["choices"][0]["message"], "Has 'content' in message"),
        ("usage" in data, "Has 'usage' field"),
        ("conversation_id" in data, "Has TextLLM 'conversation_id' extension"),
        ("cost" in data, "Has TextLLM 'cost' extension"),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check:
            print_success(desc)
        else:
            print_error(desc)
            all_passed = False
    
    if all_passed:
        content = data["choices"][0]["message"]["content"]
        print_info(f"Response content: {content[:100]}...")
        print_info(f"Tokens used: {data['usage']}")
        print_info(f"Cost: ${data.get('cost', 'N/A')}")
    
    return all_passed


def test_chat_streaming():
    """Test streaming chat."""
    print_info("Testing streaming chat...")
    
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
        "temperature": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
    
    if response.status_code != 200:
        print_error(f"Request failed: {response.status_code}")
        return False
    
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        print_error(f"Wrong content-type: {content_type}")
        return False
    
    print_success("Content-Type is text/event-stream")
    
    # Read SSE events
    events = []
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data = line_str[6:]  # Remove "data: " prefix
                events.append(data)
    
    if len(events) == 0:
        print_error("No SSE events received")
        return False
    
    print_success(f"Received {len(events)} SSE events")
    
    # Check for [DONE] marker
    if "[DONE]" in events:
        print_success("Found [DONE] marker")
    else:
        print_error("Missing [DONE] marker")
    
    # Check OpenAI-compatible chunk format
    try:
        # Parse first non-done event
        for event in events:
            if event != "[DONE]":
                chunk = json.loads(event)
                assert chunk.get("object") == "chat.completion.chunk"
                assert "choices" in chunk
                assert "delta" in chunk["choices"][0]
                print_success("Chunk format is OpenAI-compatible")
                break
    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        print_error(f"Invalid chunk format: {e}")
        return False
    
    return True


def test_chat_with_image():
    """Test chat with image (base64)."""
    print_info("Testing multimodal chat with image...")
    
    # Create a small 1x1 red PNG
    img_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    payload = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this square? Answer with just the color name."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }],
        "temperature": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code != 200:
        print_error(f"Request failed: {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    
    print_success("Multimodal request succeeded")
    print_info(f"Response: {content[:100]}...")
    
    return True


def test_conversation_context():
    """Test conversation context preservation."""
    print_info("Testing conversation context...")
    
    # First message
    payload1 = {
        "messages": [{"role": "user", "content": "My name is Alice. Remember it."}],
        "temperature": 0.0
    }
    
    response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
    if response1.status_code != 200:
        print_error("First message failed")
        return False
    
    data1 = response1.json()
    conversation_id = data1.get("conversation_id")
    
    if not conversation_id:
        print_error("No conversation_id returned")
        return False
    
    print_success(f"Created conversation: {conversation_id}")
    
    # Second message with context
    payload2 = {
        "conversation_id": conversation_id,
        "messages": [
            {"role": "user", "content": "My name is Alice. Remember it."},
            {"role": "assistant", "content": "Hello Alice! I'll remember your name."},
            {"role": "user", "content": "What's my name? Answer with just the name."}
        ],
        "temperature": 0.0
    }
    
    response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
    if response2.status_code != 200:
        print_error("Second message failed")
        return False
    
    data2 = response2.json()
    content = data2["choices"][0]["message"]["content"]
    
    if "Alice" in content:
        print_success("Conversation context preserved!")
    else:
        print_error(f"Context not preserved. Response: {content}")
    
    return True


def test_error_handling():
    """Test error handling."""
    print_info("Testing error handling...")
    
    tests = [
        ("Missing messages", {"model": "gpt-4o"}, 422),
        ("Empty messages", {"messages": []}, 422),
        ("Invalid model", {"model": "invalid-model", "messages": [{"role": "user", "content": "Hi"}]}, [200, 400, 422, 502]),
    ]
    
    all_passed = True
    for name, payload, expected in tests:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if isinstance(expected, list):
            passed = response.status_code in expected
        else:
            passed = response.status_code == expected
        
        if passed:
            print_success(f"{name}: got {response.status_code}")
        else:
            print_error(f"{name}: expected {expected}, got {response.status_code}")
            all_passed = False
    
    return all_passed


def test_docs():
    """Test Swagger docs endpoint."""
    print_info("Testing /docs endpoint...")
    
    response = requests.get(f"{BASE_URL}/docs")
    
    if response.status_code == 200:
        print_success("Swagger docs available")
        return True
    else:
        print_error(f"Docs not available: {response.status_code}")
        return False


def main():
    global BASE_URL
    
    parser = argparse.ArgumentParser(description="Test TextLLM Unified /chat Endpoint")
    parser.add_argument("--stream", action="store_true", help="Test streaming only")
    parser.add_argument("--image", action="store_true", help="Test image only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--url", default=BASE_URL, help="Base URL (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    BASE_URL = args.url
    
    print_header("TextLLM Unified /chat Endpoint - Manual Tests")
    
    # Check server
    if not check_server():
        print_error(f"Server not running at {BASE_URL}")
        print_info("Start server with: python server.py")
        return 1
    
    print_success(f"Server is running at {BASE_URL}")
    
    # Run tests
    results = []
    
    if args.stream:
        results.append(("Streaming", test_chat_streaming()))
    elif args.image:
        results.append(("Image", test_chat_with_image()))
    elif args.all:
        results.append(("Health", test_health()))
        results.append(("Docs", test_docs()))
        results.append(("Simple Chat", test_chat_simple()))
        results.append(("Streaming", test_chat_streaming()))
        results.append(("Image", test_chat_with_image()))
        results.append(("Conversation Context", test_conversation_context()))
        results.append(("Error Handling", test_error_handling()))
    else:
        # Default: run basic tests
        results.append(("Health", test_health()))
        results.append(("Docs", test_docs()))
        results.append(("Simple Chat", test_chat_simple()))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
