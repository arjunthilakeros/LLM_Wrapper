# 🤖 LLM Wrapper (TextLLM)

A production-ready conversational AI backend built on OpenAI's Conversations API with multimodal support, persistent storage, and enterprise features.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Stateful Conversations** | Messages persist via OpenAI's conversation threading |
| **Multimodal Support** | Text, images (GPT-4 Vision), documents (PDF/DOCX/TXT) |
| **Multiple Clients** | REST API + Interactive Terminal CLI |
| **Enterprise Storage** | PostgreSQL + S3 for files |
| **Rate Limiting** | Token bucket algorithm per user |
| **Cost Tracking** | Real-time token usage and cost calculation |
| **Health Monitoring** | Liveness/readiness probes |
| **Streaming** | Server-Sent Events for real-time responses |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Clients                              │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │ Terminal CLI    │    │ REST API (Web/Mobile/etc)   │  │
│  └────────┬────────┘    └──────────────┬──────────────┘  │
└───────────┼─────────────────────────────┼────────────────┘
            │                             │
            ▼                             ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI Server (Port 8000)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ /chat    │ │ /chat/   │ │ /chat/   │ │ /files       │ │
│  │          │ │ image    │ │ stream   │ │              │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  OpenAI API  │ │  PostgreSQL  │ │  S3 Storage  │
│  (GPT-4o)    │ │  (Metadata)  │ │  (Files)     │
└──────────────┘ └──────────────┘ └──────────────┘
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/arjunthilakeros/LLM_Wrapper.git
cd LLM_Wrapper

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# Run the server
python server.py

# Or run the terminal chatbot
python terminal_chatbot.py
```

## 📡 API Endpoints

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send text message, get response |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `POST` | `/chat/image` | Vision: multiple images + text |
| `POST` | `/chat/file` | Document chat (PDF/DOCX/TXT) |

### Example: Text Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "user123"}'
```

### Example: Image Chat

```bash
curl -X POST http://localhost:8000/chat/image \
  -F "message=What's in this image?" \
  -F "images=@photo.jpg"
```

### Other Endpoints

| Category | Endpoints |
|----------|-----------|
| **Conversations** | `POST/GET/DELETE /conversations` |
| **Sessions** | `GET /sessions/{id}`, `POST /sessions/{id}/end` |
| **Users** | `GET /users/{id}/stats` |
| **Files** | `POST/GET/DELETE /files` |
| **Health** | `GET /health`, `/health/live`, `/health/ready` |

📖 **Full API Docs:** `http://localhost:8000/docs`

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `DATABASE_ENABLED` | ❌ | Enable PostgreSQL (default: false) |
| `DATABASE_URL` | ❌ | PostgreSQL connection string |
| `S3_ENABLED` | ❌ | Enable S3 storage (default: false) |
| `AWS_ACCESS_KEY_ID` | ❌ | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | ❌ | AWS secret key |
| `S3_BUCKET` | ❌ | S3 bucket name |

### config.yaml

```yaml
model: gpt-4o
rate_limit_per_minute: 10
max_input_length: 10000
max_file_size_mb: 20
pricing:
  input_per_1k: 0.0025
  output_per_1k: 0.01
```

## 📁 Project Structure

```
LLM_Wrapper/
├── server.py           # FastAPI server (all endpoints)
├── terminal_chatbot.py # Interactive CLI client
├── database.py         # PostgreSQL operations
├── storage.py          # S3 file operations
├── health.py           # Health check endpoints
├── rate_limiter.py     # Token bucket rate limiter
├── validators.py       # Input validation
├── config.yaml         # Application config
├── prompts.yaml        # System prompts
└── requirements.txt    # Python dependencies
```

## 🔧 Performance

| Metric | Target |
|--------|--------|
| Text chat latency | < 2s (p95) |
| Image chat latency | < 5s (p95) |
| File upload (10MB) | < 3s |
| Health check | < 100ms |

## 💰 Cost Estimation

| Usage Pattern | Tokens/Month | Cost/Month |
|---------------|--------------|------------|
| Light (100 msgs) | 50,000 | ~$0.50 |
| Medium (1,000 msgs) | 500,000 | ~$5.00 |
| Heavy (10,000 msgs) | 5,000,000 | ~$50.00 |

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## 📚 Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed system architecture
- [API Docs](http://localhost:8000/docs) - Interactive Swagger UI

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ using FastAPI + OpenAI**
