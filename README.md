# 🤖 TextLLM - OpenAI-Compatible Chat API

> **Version:** 2.0.0  
> **Base URL:** `http://localhost:8000`  
> **Interactive Docs:** `http://localhost:8000/docs`

A production-ready conversational AI backend with **OpenAI-compatible API**, supporting multimodal inputs (text, images, documents), streaming, and enterprise features.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI's `/v1/chat/completions` |
| **Unified Endpoint** | Single `/chat` endpoint for text, images, files, and streaming |
| **Multimodal Support** | Text, images (GPT-4 Vision), PDF, DOCX, TXT processing |
| **Streaming** | Server-Sent Events (SSE) compatible with OpenAI format |
| **Enterprise Storage** | PostgreSQL + S3 for persistence |
| **Rate Limiting** | Token bucket per user |
| **Cost Tracking** | Real-time token usage and cost calculation |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-your-key-here

# Run server
python server.py
```

Server runs at: `http://localhost:8000`

---

## 📡 API Endpoints (18 Total)

### 🌟 Main Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | **Unified chat** - text, images, files, streaming |

**Example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Conversation Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/conversations` | Create conversation |
| `GET` | `/conversations` | List conversations |
| `GET` | `/conversations/{id}` | Get conversation |
| `DELETE` | `/conversations/{id}` | Delete conversation |
| `GET` | `/conversations/{id}/messages` | Get messages |

### User & Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/users/{id}/stats` | User statistics |
| `GET` | `/sessions/{id}` | Get session |
| `GET` | `/users/{id}/sessions` | List sessions |
| `POST` | `/sessions/{id}/end` | End session |

### Files

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/files/upload` | Upload file |
| `GET` | `/files` | List files |
| `DELETE` | `/files/{key}` | Delete file |

### Health & Config

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/config` | Configuration |
| `GET` | `/` | API info |
| `GET` | `/docs` | Swagger UI |

---

## 🔥 Detailed API Usage

### Unified Chat Endpoint (`POST /chat`)

#### Text Chat
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 1.0
}
```

#### With Image (Multimodal)
```json
{
  "model": "gpt-4o",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's this?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```

#### File Upload (Multipart)
```bash
curl -X POST http://localhost:8000/chat \
  -F "messages=[{\"role\":\"user\",\"content\":\"Summarize\"}]" \
  -F "files=@document.pdf"
```

#### Response Format (OpenAI-Compatible)
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  },
  "conversation_id": "conv_abc123",
  "cost": 0.0001
}
```

---

## 🗄️ Database Schema (PostgreSQL)

### Table 1: `conversations`

```sql
CREATE TABLE conversations (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    total_tokens_input INTEGER DEFAULT 0,
    total_tokens_output INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 6) DEFAULT 0,
    image_urls TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    title VARCHAR(500),
    summary TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_deleted BOOLEAN DEFAULT FALSE
);
```

### Table 2: `sessions`

```sql
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 6) DEFAULT 0,
    conversation_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Indexes
```sql
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_last_used ON conversations(last_used DESC);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
```

---

## 🧪 Testing

```bash
# Quick test
python tests_standalone/manual_test.py

# All tests
python tests_standalone/manual_test.py --all

# Pytest
pip install pytest fastapi httpx
python tests_standalone/run_tests.py -v
```

---

## 📁 Project Structure

```
LLM_Wrapper/
├── server.py              # FastAPI server
├── terminal_chatbot.py    # CLI client
├── database.py            # PostgreSQL operations
├── storage.py             # S3 operations
├── validators.py          # Input validation
├── exceptions.py          # Custom exceptions
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── tests_standalone/      # Test suite
│   ├── manual_test.py
│   ├── run_tests.py
│   └── README.md
└── README.md              # This file
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Database (optional)
DATABASE_ENABLED=true
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# S3 Storage (optional)
S3_ENABLED=true
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_BUCKET=your-bucket
```

### config.yaml

```yaml
model: gpt-4o
rate_limit_per_minute: 10
max_input_length: 10000
max_file_size_mb: 20
pricing:
  input_per_1k: 0.0025
  output_per_1k: 0.01
stream_responses: true
```

---

## 🐳 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

---

## 📄 License

MIT License

---

**Built with ❤️ using FastAPI + OpenAI**
