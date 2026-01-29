# TextLLM - Production Architecture Documentation

> **Version:** 1.0.0  
> **Last Updated:** January 2026  
> **Status:** Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Request Pipeline](#request-pipeline)
4. [API Reference](#api-reference)
5. [Data Models](#data-models)
6. [Configuration](#configuration)
7. [Deployment](#deployment)
8. [File Reference](#file-reference)

---

## System Overview

TextLLM is a production-ready conversational AI platform built on OpenAI's Conversations API. It provides:

- **Stateful Conversations** - Messages persist across sessions via OpenAI's conversation threading
- **Multimodal Support** - Text, images (GPT-4 Vision), and document processing (PDF/DOCX/TXT)
- **Multiple Clients** - REST API server + Terminal CLI client
- **Enterprise Storage** - PostgreSQL for metadata, S3 for file uploads
- **Production Features** - Rate limiting, health checks, cost tracking, session management

---

## Architecture Diagram

```mermaid
graph TD
    subgraph Clients
        A[Terminal Chatbot<br/>terminal_chatbot.py]
        B[REST API Clients<br/>Web/Mobile/Postman]
    end

    subgraph "FastAPI Server (Port 8000)"
        C[server.py]
        
        subgraph Middleware
            M1[Rate Limiter]
            M2[CORS]
            M3[Error Handler]
        end
        
        subgraph Endpoints
            E1[/chat]
            E2[/chat/stream]
            E3[/chat/image]
            E4[/chat/file]
            E5[/conversations]
            E6[/files]
            E7[/health]
        end
    end

    subgraph "External Services"
        D[OpenAI API<br/>Conversations + Responses]
    end

    subgraph "Storage Layer"
        E[(PostgreSQL)]
        F[(S3 Storage)]
    end

    subgraph "Support Modules"
        G[health.py]
        H[rate_limiter.py]
        I[logger.py]
        J[validators.py]
        K[config_validator.py]
        L[api_client.py]
        M[exceptions.py]
    end

    A -->|HTTP| C
    B -->|HTTP| C
    C --> M1 --> M2 --> M3
    C -->|OpenAI SDK| D
    C -->|psycopg2| E
    C -->|boto3| F
    C --> G
    C --> H
    C --> I
    C --> J
```

---

## Request Pipeline

### Text Chat Flow

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant RateLimiter
    participant Validator
    participant Database
    participant OpenAI
    
    Client->>Server: POST /chat {message, user_id}
    Server->>RateLimiter: Check rate limit
    RateLimiter-->>Server: OK / 429 Error
    Server->>Validator: Sanitize input
    Validator-->>Server: Clean message
    Server->>Database: Get/Create session
    Server->>OpenAI: conversations.create() if new
    Server->>OpenAI: responses.create(input, conversation)
    OpenAI-->>Server: Response + usage
    Server->>Database: Store usage stats
    Server-->>Client: {text, usage, cost}
```

### Image Chat Flow (Vision)

| Step | Action | Details |
|------|--------|---------|
| 1 | Receive request | `POST /chat/image` with form data |
| 2 | Rate limit check | Per-user token bucket |
| 3 | Validate images | Size check (max 20MB each) |
| 4 | Upload to S3 | Each image → `chat_images/{user}/{timestamp}_{filename}` |
| 5 | Encode base64 | Prepare for OpenAI Vision |
| 6 | Build input | `[{type: "input_text"}, {type: "input_image", image_url: "data:..."}]` |
| 7 | Call GPT-4o | Vision model processes text + images |
| 8 | Store usage | Tokens + cost in PostgreSQL |
| 9 | Return response | Text + S3 URLs |

### Document Chat Flow

| Step | Action | Details |
|------|--------|---------|
| 1 | Receive file | `POST /chat/file` with PDF/DOCX/TXT |
| 2 | Upload to S3 | Store original document |
| 3 | Extract text | PyPDF2 for PDF, python-docx for DOCX |
| 4 | Build context | Append extracted text to user message |
| 5 | Call OpenAI | For PDFs: use `input_file` type |
| 6 | Return response | Summary/analysis + file URL |

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Authentication
Currently no authentication required. **Recommended:** Add API key middleware for production.

---

### Chat Endpoints

#### `POST /chat`
Send a text message and receive a response.

**Request Body:**
```json
{
  "message": "Hello, how are you?",
  "conversation_id": "conv_abc123",  // optional, creates new if omitted
  "user_id": "user_xyz",             // optional, auto-generated if omitted
  "stream": false
}
```

**Response:**
```json
{
  "text": "I'm doing well, thank you for asking!",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz",
  "message_number": 1,
  "usage": {
    "input": 12,
    "output": 15,
    "total": 27
  },
  "cost": 0.000165
}
```

---

#### `POST /chat/stream`
Stream response tokens via Server-Sent Events.

**Request:** Same as `/chat`

**Response:** SSE stream
```
data: Hello
data: , how
data:  are
data:  you?
data: [DONE]
```

**Headers:**
- `X-Conversation-ID`: The conversation ID

---

#### `POST /chat/image`
Send one or more images with a text prompt (GPT-4 Vision).

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Text prompt |
| `images` | file[] | Yes | One or more image files |
| `conversation_id` | string | No | Existing conversation |
| `user_id` | string | No | User identifier |

**Response:**
```json
{
  "text": "I can see two cats in this image...",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz",
  "message_number": 1,
  "usage": {"input": 1200, "output": 150, "total": 1350},
  "cost": 0.0045,
  "image_urls": [
    "https://s3.endpoint/bucket/LLM/chat_images/user/image1.jpg",
    "https://s3.endpoint/bucket/LLM/chat_images/user/image2.jpg"
  ],
  "image_count": 2
}
```

---

#### `POST /chat/file`
Chat with a document (PDF, DOCX, or TXT).

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Question about the document |
| `file` | file | Yes | PDF, DOCX, or TXT file |
| `conversation_id` | string | No | Existing conversation |
| `user_id` | string | No | User identifier |

**Response:**
```json
{
  "text": "This document discusses...",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz",
  "message_number": 1,
  "usage": {"input": 5000, "output": 500, "total": 5500},
  "cost": 0.0175,
  "file_url": "https://s3.endpoint/bucket/LLM/documents/user/report.pdf",
  "file_type": ".pdf"
}
```

---

### Conversation Endpoints

#### `POST /conversations`
Create a new conversation.

**Request:**
```json
{
  "user_id": "user_xyz",
  "metadata": {"topic": "support"}
}
```

**Response:**
```json
{
  "id": "conv_abc123",
  "user_id": "user_xyz",
  "created_at": "2026-01-29T12:00:00Z",
  "message_count": 0
}
```

---

#### `GET /conversations?user_id={id}&limit=50&offset=0`
List conversations for a user.

---

#### `GET /conversations/{conversation_id}`
Get a specific conversation.

---

#### `DELETE /conversations/{conversation_id}?soft=true`
Delete a conversation. Soft delete by default.

---

#### `GET /conversations/{conversation_id}/messages?limit=100&order=asc`
Get messages from OpenAI's conversation history.

---

### Session Endpoints

Sessions track usage within a time period (typically daily).

#### `GET /sessions/{session_id}`
Get session details including total tokens and cost.

#### `GET /users/{user_id}/sessions`
Get today's session for a user.

#### `POST /sessions/{session_id}/end`
Mark a session as ended.

---

### User Endpoints

#### `GET /users/{user_id}/stats?days=30`
Get usage statistics for a user.

**Response:**
```json
{
  "total_conversations": 25,
  "total_messages": 150,
  "total_tokens_input": 45000,
  "total_tokens_output": 12000,
  "total_cost": 1.25
}
```

---

### File Endpoints

#### `POST /files/upload`
Upload a file to S3 storage.

**Request:** `multipart/form-data`
| Field | Type | Required |
|-------|------|----------|
| `file` | file | Yes |
| `user_id` | string | Yes |
| `folder` | string | No (default: "uploads") |

**Response:**
```json
{
  "success": true,
  "key": "LLM/uploads/user_xyz/20260129_120000_report.pdf",
  "url": "https://s3.endpoint/bucket/LLM/uploads/...",
  "size": 102400
}
```

---

#### `GET /files?user_id={id}&folder=uploads&limit=100`
List files for a user.

---

#### `DELETE /files/{file_key}`
Delete a file from S3.

---

### Health Endpoints

#### `GET /health`
Comprehensive health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T12:30:00Z",
  "uptime": {"uptime_seconds": 3600, "uptime_human": "1:00:00"},
  "checks": {
    "openai_api": {"status": "configured"},
    "database": {"status": "healthy"},
    "storage": {"status": "healthy"},
    "disk": {"free_gb": 50.2}
  }
}
```

---

#### `GET /health/live`
Kubernetes liveness probe. Returns `{"status": "alive"}`.

---

#### `GET /health/ready`
Kubernetes readiness probe. Checks OpenAI + S3 are configured.

---

### Config Endpoint

#### `GET /config`
Get non-sensitive configuration.

**Response:**
```json
{
  "config": {
    "model": "gpt-4o",
    "max_input_length": 10000,
    "max_file_size_mb": 20,
    "rate_limit_per_minute": 10,
    "stream_responses": true
  }
}
```

---

## Data Models

### PostgreSQL Schema

```sql
-- Conversations (metadata only, messages stored in OpenAI)
CREATE TABLE conversations (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    message_count INTEGER DEFAULT 0,
    total_tokens_input INTEGER DEFAULT 0,
    total_tokens_output INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    title VARCHAR(500),
    summary TEXT,
    tags TEXT[],
    metadata JSONB,
    is_deleted BOOLEAN DEFAULT FALSE,
    last_image_url TEXT
);

-- Sessions (usage tracking)
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    conversation_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
```

### S3 Structure

```
{bucket}/LLM/
├── chat_images/
│   └── {user_id}/
│       └── {timestamp}_{filename}.jpg
├── documents/
│   └── {user_id}/
│       └── {timestamp}_{filename}.pdf
└── uploads/
    └── {user_id}/
        └── {timestamp}_{filename}
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=sk-...

# Database (optional, falls back to JSON)
DATABASE_ENABLED=true
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# S3 Storage (required for production)
S3_ENABLED=true
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=your-bucket
S3_ENDPOINT=https://s3.amazonaws.com
S3_URL_EXPIRATION=86400

# Optional
LOG_LEVEL=INFO
```

### `config.yaml`

```yaml
model: gpt-4o
max_history_items: 100
rate_limit_per_minute: 10
max_input_length: 10000
max_file_size_mb: 20
cost_limit_per_session: 5.0
warn_at_cost: 1.0
pricing:
  input_per_1k: 0.0025
  output_per_1k: 0.01
stream_responses: true
logging:
  level: INFO
  log_to_file: true
  log_dir: ./logs
```

---

## Deployment

### Development
```bash
python server.py
# Runs on http://localhost:8000
```

### Production (Docker)
```bash
docker-compose up -d
```

### Production (Manual)
```bash
pip install gunicorn uvicorn[standard]
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Kubernetes Health Probes
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 1388 | FastAPI server with all endpoints |
| `terminal_chatbot.py` | 1233 | Interactive CLI client |
| `database.py` | 429 | PostgreSQL connection pooling & queries |
| `storage.py` | 367 | S3 upload/download operations |
| `health.py` | 262 | Health check functions |
| `rate_limiter.py` | ~150 | Token bucket rate limiter |
| `validators.py` | ~200 | Input sanitization |
| `config_validator.py` | ~250 | Configuration schema validation |
| `api_client.py` | ~150 | OpenAI client wrapper with retry |
| `exceptions.py` | ~100 | Custom exception classes |
| `logger.py` | ~100 | Structured logging setup |
| `config.yaml` | ~30 | Application configuration |
| `prompts.yaml` | ~50 | System prompts |

---

## Interactive Documentation

Access Swagger UI at: **http://localhost:8000/docs**

Access ReDoc at: **http://localhost:8000/redoc**
