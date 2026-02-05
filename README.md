# LLM Chat API

REST API for conversational AI using OpenAI's Responses API with PostgreSQL metadata storage.

**Base URL:** `http://localhost:8000`
**Docs:** `http://localhost:8000/docs`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
export OPENAI_API_KEY=sk-your-key
export DATABASE_URL=postgresql://user:pass@host:5432/db
export DATABASE_ENABLED=true

# Run
python server.py
```

## API Endpoints

### Chat

#### POST /chat
Send a message and get a response.

**Request:**
```json
{
  "input": "Hello, how are you?",
  "user_id": "user123",
  "conversation_id": "conv_xxx",
  "model": "gpt-4o",
  "stream": false,
  "temperature": 1.0,
  "max_tokens": 1000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| input | string | Yes | User message |
| user_id | string | No | User identifier for rate limiting |
| conversation_id | string | No | Existing conversation ID (auto-creates if omitted) |
| model | string | No | Model to use (default: gpt-4o) |
| stream | boolean | No | Enable streaming (default: false) |
| temperature | float | No | Sampling temperature 0-2 (default: 1.0) |
| max_tokens | int | No | Max output tokens |

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

**Streaming:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "stream": true}'
```
Returns Server-Sent Events (SSE).

---

### Conversations

#### POST /conversations
Create a new conversation.

**Request:**
```json
{
  "user_id": "user123",
  "title": "My Chat"
}
```

**Response:**
```json
{
  "id": "conv_xxx",
  "user_id": "user123",
  "created_at": "2025-01-01T00:00:00",
  "message_count": 0,
  "title": "My Chat",
  "summary": null,
  "total_tokens": 0,
  "total_cost": 0
}
```

#### GET /conversations?user_id=xxx
List conversations for a user.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| user_id | string | Required | User ID |
| limit | int | 50 | Max results (1-100) |
| offset | int | 0 | Pagination offset |

#### GET /conversations/{id}
Get a single conversation.

**Response:**
```json
{
  "id": "conv_xxx",
  "user_id": "user123",
  "created_at": "2025-01-01T00:00:00",
  "message_count": 15,
  "title": "Python Learning",
  "summary": "User is learning Python basics including variables, data types, and functions.",
  "total_tokens": 5000,
  "total_cost": 0.015
}
```

#### PATCH /conversations/{id}
Update conversation title or metadata.

**Request:**
```json
{
  "title": "New Title"
}
```

#### DELETE /conversations/{id}
Delete a conversation.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| soft | boolean | true | Soft delete (keeps data) |

#### GET /conversations/{id}/messages
Get messages from OpenAI's Conversations API.

**Response:**
```json
[
  {
    "id": "msg_xxx",
    "role": "user",
    "content": "What is Python?",
    "created_at": "2025-01-01T00:00:00"
  },
  {
    "id": "msg_yyy",
    "role": "assistant",
    "content": "Python is a programming language...",
    "created_at": "2025-01-01T00:00:01"
  }
]
```

---

### Health & Info

#### GET /health
Health check with system status.

#### GET /config
Current configuration (non-sensitive).

#### GET /
API information.

---

## Features

### Auto Title Generation
When you send the first message to a conversation without a title, an AI-generated title is created automatically using gpt-4o-mini.

### Conversation Summary
Long conversations are automatically summarized to reduce token usage. The summary is stored and updated every 5 new messages after the threshold is reached.

### Message Storage
Messages are stored in OpenAI's Conversations API, not locally. The local database stores only metadata:
- Title and summary
- Message count
- Token usage and cost
- Timestamps

---

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-xxx          # Required
DATABASE_URL=postgresql://...   # Required for persistence
DATABASE_ENABLED=true           # Enable database
S3_ENABLED=true                 # Enable S3 storage
S3_ENDPOINT=https://...
S3_ACCESS_KEY=xxx
S3_SECRET_KEY=xxx
S3_BUCKET=bucket-name
```

### config.yaml
```yaml
model: gpt-4o
rate_limit_per_minute: 10
max_input_length: 10000

context_management:
  mode: summary_window    # "full" or "summary_window"
  window_size: 5          # Message pairs to keep
  summarize_after_messages: 10
  summary_update_interval: 5

pricing:
  input_per_1k: 0.0025
  output_per_1k: 0.01
```

---

## Database Schema

```sql
CREATE TABLE conversations (
    id VARCHAR(255) PRIMARY KEY,
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

### List User Conversations
```bash
curl "http://localhost:8000/conversations?user_id=user1"
```

### Get Conversation with Summary
```bash
curl http://localhost:8000/conversations/conv_xxx
```

### Update Title
```bash
curl -X PATCH http://localhost:8000/conversations/conv_xxx \
  -H "Content-Type: application/json" \
  -d '{"title": "Python Tutorial"}'
```
