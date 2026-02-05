# LLM Chat API

REST API for conversational AI using OpenAI's Responses API with PostgreSQL metadata storage and S3 integration.

**Base URL:** `http://localhost:8000`
**Docs:** `http://localhost:8000/docs`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your credentials

# Run
python server.py
```

## API Endpoints

### Chat

#### POST /chat
Send a message and get a response. Supports text, images, and streaming.

**Request:**
```json
{
  "input": "Hello, how are you?",
  "user_id": "user123",
  "conversation_id": "conv_xxx",
  "model": "gpt-4o",
  "instructions": "You are a helpful assistant",
  "stream": false,
  "temperature": 1.0,
  "max_tokens": 1000,
  "top_p": 1.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| input | string/array | Yes | User message (text or multimodal) |
| user_id | string | No | User identifier for rate limiting |
| conversation_id | string | No | Existing conversation ID (auto-creates if omitted) |
| model | string | No | Model to use (default: gpt-4o) |
| instructions | string | No | System instructions for the assistant |
| stream | boolean | No | Enable streaming (default: false) |
| temperature | float | No | Sampling temperature 0-2 (default: 1.0) |
| max_tokens | int | No | Max output tokens |
| top_p | float | No | Nucleus sampling 0-1 (default: 1.0) |

**Response:**
```json
{
  "id": "resp_xxx",
  "object": "response",
  "created_at": 1234567890,
  "model": "gpt-4o",
  "output_text": "Hello! I'm doing well, thank you for asking.",
  "conversation_id": "conv_xxx",
  "title": "Friendly Greeting",
  "message_count": 1,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 15,
    "total_tokens": 25
  },
  "cost": 0.0002
}
```

#### Chat with Image (Multimodal)

```json
{
  "input": [
    {
      "type": "text",
      "text": "What is in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/image.jpg"
      }
    }
  ],
  "model": "gpt-4o",
  "user_id": "user123"
}
```

Supports:
- **HTTPS URLs:** Direct image links
- **S3 URLs:** Pre-signed S3 URLs
- **Base64:** `data:image/png;base64,iVBORw0KGgo...`

#### Chat with System Instructions

```json
{
  "input": "Tell me a joke",
  "instructions": "You are a pirate. Always respond like a pirate with arrr and matey.",
  "user_id": "user123"
}
```

#### Streaming Response

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "Write a story", "stream": true}'
```

Returns Server-Sent Events (SSE):
```
data: {"delta": {"type": "text", "text": "Once"}}
data: {"delta": {"type": "text", "text": " upon"}}
data: {"delta": {"type": "text", "text": " a time"}}
data: {"done": true, "conversation_id": "conv_xxx"}
data: [DONE]
```

---

### Conversations

#### POST /conversations
Create a new conversation.

```json
{
  "user_id": "user123",
  "title": "My Chat",
  "metadata": {
    "category": "support"
  }
}
```

#### GET /conversations?user_id=xxx
List conversations for a user.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| user_id | string | Required | User ID |
| limit | int | 50 | Max results (1-100) |
| offset | int | 0 | Pagination offset |

#### GET /conversations/{id}
Get a single conversation.

#### PATCH /conversations/{id}
Update conversation title or metadata.

```json
{
  "title": "New Title",
  "metadata": {"updated": true}
}
```

#### DELETE /conversations/{id}?soft=true
Delete a conversation.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| soft | boolean | true | Soft delete (keeps data, marks as deleted) |

#### GET /conversations/{id}/messages
Get messages from OpenAI's Conversations API.

```json
[
  {"id": "msg_xxx", "role": "user", "content": "What is Python?"},
  {"id": "msg_yyy", "role": "assistant", "content": "Python is..."}
]
```

---

### Health & Info

| Endpoint | Description |
|----------|-------------|
| GET /health | Health check with system status |
| GET /config | Current configuration (non-sensitive) |
| GET / | API information |

---

## Features

### Auto Conversation Creation
When you send a chat without a `conversation_id`, one is automatically created in OpenAI and tracked locally.

### Auto Title Generation
First message triggers AI-generated title using gpt-4o-mini.

### Conversation Memory
OpenAI stores full conversation history. Use the same `conversation_id` to continue conversations - the AI remembers everything.

### Conversation Summary
Long conversations are automatically summarized to reduce token usage.

### Message Storage
Messages are stored in OpenAI's Conversations API, not locally. Local database stores only metadata:
- Title and summary
- Message count
- Token usage and cost
- Timestamps

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-xxx

# Database (Required for persistence)
DATABASE_URL=postgresql://user:pass@host:5432/db
DATABASE_ENABLED=true

# S3 Storage (Optional - for file uploads)
S3_ENABLED=true
S3_ENDPOINT=https://s3.example.com
S3_ACCESS_KEY=xxx
S3_SECRET_KEY=xxx
S3_BUCKET=bucket-name
S3_URL_EXPIRATION=86400

# Server
ENVIRONMENT=development
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=10
```

### config.yaml

```yaml
model: gpt-4o
rate_limit_per_minute: 10
max_input_length: 10000
max_file_size_mb: 20

context_management:
  mode: full              # "full" or "summary_window"
  window_size: 5          # Message pairs to keep (summary_window mode)
  summarize_after_messages: 10
  summary_update_interval: 5

pricing:
  input_per_1k: 0.0025
  output_per_1k: 0.01
```

---

## Database Schema

Schema: `textllm`

```sql
CREATE TABLE textllm.conversations (
    id VARCHAR(255) PRIMARY KEY,        -- OpenAI conversation ID (conv_xxx)
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    summary TEXT,
    message_count INTEGER DEFAULT 0,
    summary_message_count INTEGER DEFAULT 0,
    summary_updated_at TIMESTAMP,
    total_tokens_input INTEGER DEFAULT 0,
    total_tokens_output INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 6) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

---

## Examples

### Basic Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Python?", "user_id": "user1"}'
```

### Continue Conversation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Tell me more about lists",
    "user_id": "user1",
    "conversation_id": "conv_xxx"
  }'
```

### Chat with Image
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ],
    "user_id": "user1"
  }'
```

### Chat with Custom Instructions
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Explain quantum computing",
    "instructions": "Explain like I am 5 years old",
    "user_id": "user1"
  }'
```

### Streaming Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "Write a poem", "stream": true, "user_id": "user1"}'
```

### List User Conversations
```bash
curl "http://localhost:8000/conversations?user_id=user1&limit=20"
```

### Get Conversation Messages
```bash
curl "http://localhost:8000/conversations/conv_xxx/messages?limit=50&order=asc"
```

### Update Conversation Title
```bash
curl -X PATCH http://localhost:8000/conversations/conv_xxx \
  -H "Content-Type: application/json" \
  -d '{"title": "Python Tutorial"}'
```

### Delete Conversation
```bash
# Soft delete (recoverable)
curl -X DELETE "http://localhost:8000/conversations/conv_xxx?soft=true"

# Hard delete (permanent)
curl -X DELETE "http://localhost:8000/conversations/conv_xxx?soft=false"
```

---

## Error Handling

All errors return:
```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "error_code"
  }
}
```

| Status | Description |
|--------|-------------|
| 400 | Bad request |
| 404 | Not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 502 | OpenAI API error |
| 503 | Service unavailable |
